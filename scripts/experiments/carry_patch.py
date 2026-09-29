#!/usr/bin/env python3
"""Experiment E2 — are the outputs of L22.H10/H11 *sufficient* to inject or remove the units carry?

E1 showed necessity (ablating the heads makes carry problems drop the carry). E2 transplants the heads' output
between matched problems: for a no-carry problem N = (10·t1 + u1) + (10·t2 + u2) we build a carry problem C with
the SAME tens digits and units that do carry. Prompts have identical token lengths, so positions align.

Conditions (readout: tens digit at the answer position, patch only at the answer position):
  inject   z_{L22.H10,H11}(C) → run N     does N now output t1 + t2 + 1 (a spurious carry)?
  remove   z_{L22.H10,H11}(N) → run C     does C now output t1 + t2 (carry removed)?
  control  same transplant for two other layer-22 heads (default H8, H9)

Usage: .venv/bin/python scripts/experiments/carry_patch.py [--n 40] [--seed 0]
Writes docs/research/E2_carry_patch.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def pairs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t1, t2 = rng.randint(1, 8), rng.randint(1, 8)
        if t1 + t2 + 1 > 9:
            continue
        nu = [(u1, u2) for u1 in range(10) for u2 in range(10) if u1 + u2 < 10]
        cu = [(u1, u2) for u1 in range(10) for u2 in range(10) if u1 + u2 >= 10]
        (a1, a2), (c1, c2) = rng.choice(nu), rng.choice(cu)
        out.append({"tens": (t1, t2), "N": (10 * t1 + a1, 10 * t2 + a2), "C": (10 * t1 + c1, 10 * t2 + c2)})
    return out


def sem(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--heads", default="10,11")
    ap.add_argument("--control-heads", default="8,9")
    ap.add_argument("--out", default=str(ROOT / "docs/research/E2_carry_patch.json"))
    args = ap.parse_args()

    from neuralscope.config import Settings
    from neuralscope.instrumentation import HeadSetEdit, load_adapter

    s = Settings()
    s.model_id = args.model
    a = load_adapter(s)
    L = args.layer
    heads = [int(h) for h in args.heads.split(",")]
    ctrl = [int(h) for h in args.control_heads.split(",")]
    digit = {d: a.encode(str(d))[0] for d in range(10)}
    t0 = time.time()
    rows = []
    for i, pr in enumerate(pairs(args.n, args.seed)):
        t1, t2 = pr["tens"]
        runs = {}
        for key in ("N", "C"):
            x, y = pr[key]
            ids = a.tokenize(f"What is {x} + {y}? Give only the answer.").input_ids
            runs[key] = (ids, a.capture(ids, "standard"))
        assert len(runs["N"][0]) == len(runs["C"][0])
        pos = len(runs["N"][0]) - 1

        def evaluate(target: str, source: str, hs: list[int]) -> dict:
            ids, tr = runs[target]
            src = runs[source][1]
            edits = [[]] + [[HeadSetEdit(L, h, src.head_z[L, pos, h], positions=[pos]) for h in hs]]
            lg = a.final_logits_batch(ids, edits, start_depth=L, resid_in=tr.resid[L])[:, pos]
            dl = torch.stack([lg[:, digit[d]] for d in range(10)], -1)  # [2, 10] digit logits
            lp = torch.log_softmax(lg, -1)
            nc, c = t1 + t2, t1 + t2 + 1
            return {"argmax_before": int(dl[0].argmax()), "argmax_after": int(dl[1].argmax()),
                    "p_carry_before": float(lp[0, digit[c]].exp()), "p_carry_after": float(lp[1, digit[c]].exp()),
                    "p_nocarry_before": float(lp[0, digit[nc]].exp()), "p_nocarry_after": float(lp[1, digit[nc]].exp()),
                    "ld_before": float(dl[0, c] - dl[0, nc]), "ld_after": float(dl[1, c] - dl[1, nc])}

        row = {**pr, "inject": evaluate("N", "C", heads), "remove": evaluate("C", "N", heads),
               "inject_ctrl": evaluate("N", "C", ctrl), "remove_ctrl": evaluate("C", "N", ctrl)}
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n} ({time.time() - t0:.0f}s)", flush=True)

    def summ(key: str, want: str) -> dict:
        xs = [r[key] for r in rows]
        tens = [(r["tens"][0] + r["tens"][1] + (1 if want == "carry" else 0)) for r in rows]
        d = [x["ld_after"] - x["ld_before"] for x in xs]
        return {"n": len(xs),
                "baseline_correct": sum(x["argmax_before"] == (t - 1 if want == "carry" else t + 1) for x, t in zip(xs, tens)) / len(xs),
                "flipped_to_" + want: sum(x["argmax_after"] == t for x, t in zip(xs, tens)) / len(xs),
                "mean_delta_ld_carry_minus_nocarry": round(sum(d) / len(d), 3), "sem": round(sem(d), 3)}

    summary = {
        f"inject L{L}.H{'+'.join(map(str, heads))} (C→N)": summ("inject", "carry"),
        f"remove L{L}.H{'+'.join(map(str, heads))} (N→C)": summ("remove", "nocarry"),
        f"control inject L{L}.H{'+'.join(map(str, ctrl))}": summ("inject_ctrl", "carry"),
        f"control remove L{L}.H{'+'.join(map(str, ctrl))}": summ("remove_ctrl", "nocarry"),
    }
    report = {"experiment": "E2_carry_patch", "model_id": a.info.model_id, "layer": L, "heads": heads,
              "control_heads": ctrl, "n_pairs": args.n, "seed": args.seed, "seconds": round(time.time() - t0, 1),
              "status": "causal",
              "scope": f"{args.n} matched pairs of 2-digit additions (same tens digits, carry vs no carry); "
                       "head outputs transplanted at the answer position only",
              "summary": summary, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nE2 — {a.info.model_id} · {args.n} pairs · {report['seconds']}s")
    print("| condition | baseline correct | flipped | Δ(logit carry − logit no-carry) ± SEM |\n|---|---|---|---|")
    for k, v in summary.items():
        flip = [x for x in v if x.startswith("flipped")][0]
        print(f"| {k} | {v['baseline_correct']:.2f} | {v[flip]:.2f} ({flip[8:]}) | "
              f"{v['mean_delta_ld_carry_minus_nocarry']:+.2f} ± {v['sem']:.2f} |")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
