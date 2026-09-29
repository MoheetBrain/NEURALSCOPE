#!/usr/bin/env python3
"""Experiment E1 — are L22.H10 / L22.H11 necessary for the units→tens carry in 2-digit addition?

Origin (single-prompt observations, see RESEARCH_LOG.md):
  * exhaustive single-head ablation on "What is 37 + 58?" (Qwen2.5-1.5B-Instruct): zero-ablating L22.H10
    drops P("9") 0.9998 → 0.0003 and the output becomes "85" — the sum *without* the carry;
  * residual patching 37+58 vs 37+48 shows the tens-operand information moving from the operand token to the
    answer position between L21 and L24.

INFERRED hypothesis under test: H10/H11 in layer 22 carry information needed to add the units carry into the
tens digit. If true, ablating them should (a) hurt carry problems much more than no-carry problems and
(b) move the prediction specifically to the no-carry digit (a_t + b_t), and (c) the effect should be specific
relative to the other heads of the same layer and to random heads elsewhere.

Design: all problems are "What is A + B? Give only the answer." with A, B two-digit, A + B < 100, so every
prompt has the same token length and the first answer token is the tens digit. Readout at the answer position.
For each problem we measure, under each condition, the argmax tens digit, P(correct), P(alternative) and the
logit difference correct − alternative, where alternative = the digit you get by mis-handling the carry
(carry problems: a_t + b_t; no-carry problems: a_t + b_t + 1).

Usage:  .venv/bin/python scripts/experiments/carry_heads.py [--n 40] [--seed 0] [--model Qwen/Qwen2.5-1.5B-Instruct]
Writes docs/research/E1_carry_heads.json and prints a markdown summary.
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


def build_problems(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    carry, nocarry = [], []
    for a in range(10, 100):
        for b in range(10, 100):
            if a + b >= 100:
                continue
            at, au, bt, bu = a // 10, a % 10, b // 10, b % 10
            if au + bu >= 10:
                carry.append({"a": a, "b": b, "carry": True, "alt_tens": at + bt})
            else:
                if at + bt + 1 <= 9:
                    nocarry.append({"a": a, "b": b, "carry": False, "alt_tens": at + bt + 1})
    return rng.sample(carry, n) + rng.sample(nocarry, n)


def sem(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="problems per group (carry / no-carry)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--heads", default="10,11")
    ap.add_argument("--random-controls", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "docs/research/E1_carry_heads.json"))
    args = ap.parse_args()

    from neuralscope.config import DEMO_MODEL_ID, Settings
    from neuralscope.instrumentation import HeadScaleEdit, HeadSetEdit, load_adapter

    s = Settings()
    s.model_id = args.model
    a = load_adapter(s)
    info = a.info
    L, H = info.n_layers, info.n_heads
    layer = min(args.layer, L - 1)
    focus = [int(h) for h in args.heads.split(",")]
    rng = random.Random(args.seed + 1)
    others_same_layer = [h for h in range(H) if h not in focus]
    random_heads = []
    while len(random_heads) < args.random_controls:
        l, h = rng.randrange(L), rng.randrange(H)
        if l != layer and (l, h) not in random_heads:
            random_heads.append((l, h))

    # conditions evaluated by resuming from depth `layer` (exact; see tests)
    same_layer_conditions = {"baseline": []}
    for h in focus:
        same_layer_conditions[f"L{layer}.H{h}"] = [HeadScaleEdit(layer, h, 0.0)]
    same_layer_conditions["L{}.H{}".format(layer, "+".join(map(str, focus)))] = [HeadScaleEdit(layer, h, 0.0) for h in focus]
    for h in others_same_layer:
        same_layer_conditions[f"ctrl L{layer}.H{h}"] = [HeadScaleEdit(layer, h, 0.0)]
    rand_conditions = {f"rand L{l}.H{h}": [HeadScaleEdit(l, h, 0.0)] for l, h in random_heads}

    digit_ids = {d: a.encode(str(d))[0] for d in range(10)}
    problems = build_problems(args.n, args.seed)
    t0 = time.time()

    # pass 1: dataset-mean output z of each focus head at every position (all prompts share one token length),
    # for MEAN-ablation — a control for zero-ablation pushing activations off-distribution
    zs = {h: [] for h in focus}
    lengths = set()
    for p in problems:
        ids = a.tokenize(f"What is {p['a']} + {p['b']}? Give only the answer.").input_ids
        lengths.add(len(ids))
        tr = a.capture(ids, "standard")
        for h in focus:
            zs[h].append(tr.head_z[layer, :, h])
    if len(lengths) != 1:
        raise SystemExit(f"prompts differ in token length {lengths}; mean-ablation needs aligned positions")
    mean_z = {h: torch.stack(zs[h]).mean(0) for h in focus}  # [T, d_head]
    both_key = "L{}.H{}".format(layer, "+".join(map(str, focus)))
    same_layer_conditions[f"mean-abl {both_key}"] = [HeadSetEdit(layer, h, mean_z[h]) for h in focus]
    for h in focus:
        same_layer_conditions[f"mean-abl L{layer}.H{h}"] = [HeadSetEdit(layer, h, mean_z[h])]
    print(f"  pass 1 (mean head outputs) done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for i, p in enumerate(problems):
        q = f"What is {p['a']} + {p['b']}? Give only the answer."
        tp = a.tokenize(q)
        ids = tp.input_ids
        P = len(ids)
        pos = P - 1
        true_tens = (p["a"] + p["b"]) // 10
        tid, aid = digit_ids[true_tens], digit_ids[p["alt_tens"]]
        tr = a.capture(ids, "standard")
        names = list(same_layer_conditions)
        lg = a.final_logits_batch(ids, [same_layer_conditions[n] for n in names], start_depth=layer,
                                  resid_in=tr.resid[layer])[:, pos]
        rnames = list(rand_conditions)
        rlg = a.final_logits_batch(ids, [rand_conditions[n] for n in rnames])[:, pos] if rnames else None
        res = {}
        for j, n in enumerate(names + rnames):
            x = lg[j] if j < len(names) else rlg[j - len(names)]
            lp = torch.log_softmax(x, -1)
            digit_logits = torch.stack([x[digit_ids[d]] for d in range(10)])
            res[n] = {"argmax_digit": int(digit_logits.argmax()), "top1_is_digit": int(x.argmax()) in digit_ids.values(),
                      "p_true": float(lp[tid].exp()), "p_alt": float(lp[aid].exp()),
                      "logit_diff": float(x[tid] - x[aid])}
        # attention of the focus heads from the answer position (DERIVED descriptive statistic)
        span = tp.user_span
        user = ids[span[0]:span[1]]
        toks = a.token_strings(user)
        # locate operand digit tokens inside the user span: "What is A + B?" → digits appear in order
        digit_pos = [span[0] + k for k, t in enumerate(toks) if t.isdigit()]
        a_t, a_u, b_t, b_u = digit_pos[:4]
        attn = {}
        for h in focus:
            row = tr.attn[layer, h, pos]
            attn[f"H{h}"] = {"a_tens": float(row[a_t]), "a_units": float(row[a_u]), "b_tens": float(row[b_t]),
                             "b_units": float(row[b_u]), "first_token": float(row[0]), "self": float(row[pos])}
        # Logit-Lens trajectories (DERIVED) of the true and carry-mishandled digit at the answer position,
        # baseline and with the focus heads jointly ablated
        both = "L{}.H{}".format(layer, "+".join(map(str, focus)))
        tr_abl = a.capture(ids, "minimal", same_layer_conditions[both])
        lens = {}
        for name, t_ in (("baseline", tr), ("ablated", tr_abl)):
            lp = torch.log_softmax(a.logit_lens(t_.resid[:, pos]), -1)  # [D, V]
            lens[name] = {"p_true": [round(float(v), 5) for v in lp[:, tid].exp()],
                          "p_alt": [round(float(v), 5) for v in lp[:, aid].exp()]}
        # free-running outputs for baseline and the focus heads jointly ablated
        gen_base = a.decode(a.generate(ids, 4).generated_ids)
        both_key = "L{}.H{}".format(layer, "+".join(map(str, focus)))
        gen_h = {n: a.decode(a.generate(ids, 4, same_layer_conditions[n]).generated_ids)
                 for n in [f"L{layer}.H{h}" for h in focus] + [both_key]}
        rows.append({**p, "question": q, "sum": p["a"] + p["b"], "true_tens": true_tens, "conditions": res,
                     "attention": attn, "lens": lens, "output_baseline": gen_base, "outputs_ablated": gen_h})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(problems)}  ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ aggregate
    def agg(group: bool, cond: str) -> dict:
        rs = [r for r in rows if r["carry"] == group]
        base = [r["conditions"]["baseline"] for r in rs]
        cs = [r["conditions"][cond] for r in rs]
        acc = sum(c["argmax_digit"] == r["true_tens"] for c, r in zip(cs, rs)) / len(rs)
        to_alt = sum(c["argmax_digit"] == r["alt_tens"] for c, r in zip(cs, rs)) / len(rs)
        dld = [c["logit_diff"] - b["logit_diff"] for c, b in zip(cs, base)]
        dp = [c["p_true"] - b["p_true"] for c, b in zip(cs, base)]
        return {"n": len(rs), "tens_accuracy": round(acc, 4), "frac_predicts_alt_digit": round(to_alt, 4),
                "mean_p_true": round(sum(c["p_true"] for c in cs) / len(cs), 5),
                "mean_delta_logit_diff": round(sum(dld) / len(dld), 4), "sem_delta_logit_diff": round(sem(dld), 4),
                "mean_delta_p_true": round(sum(dp) / len(dp), 5)}

    conds = list(same_layer_conditions) + list(rand_conditions)
    summary = {c: {"carry": agg(True, c), "no_carry": agg(False, c)} for c in conds}
    attn_summary = {}
    for h in focus:
        for g in (True, False):
            rs = [r for r in rows if r["carry"] == g]
            attn_summary[f"H{h}/{'carry' if g else 'no_carry'}"] = {
                k: round(sum(r["attention"][f"H{h}"][k] for r in rs) / len(rs), 4)
                for k in ("a_tens", "a_units", "b_tens", "b_units", "first_token", "self")}
    both_key = "L{}.H{}".format(layer, "+".join(map(str, focus)))
    out_pattern = {}
    for g in (True, False):
        rs = [r for r in rows if r["carry"] == g]
        def first_int(s):
            digits = ""
            for ch in s:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            return int(digits) if digits else None
        base_ok = sum(first_int(r["output_baseline"]) == r["sum"] for r in rs)
        ab = [first_int(r["outputs_ablated"][both_key]) for r in rs]
        exact_nocarry = sum(v == (r["sum"] - 10 if g else r["sum"] + 10) for v, r in zip(ab, rs))
        out_pattern["carry" if g else "no_carry"] = {
            "baseline_full_answer_correct": base_ok, "ablated_full_answer_correct": sum(v == r["sum"] for v, r in zip(ab, rs)),
            "ablated_equals_carry_mishandled_answer": exact_nocarry, "n": len(rs)}

    # two-stage signature: does the carry-mishandled digit dominate the lens (P > 0.5) at some depth *before*
    # the correct digit first exceeds 0.5?  (baseline, per group) + mean lens curves
    def first_above(xs, thr=0.5):
        return next((d for d, v in enumerate(xs) if v > thr), None)

    lens_summary = {}
    D = info.n_layers + 1
    for g in (True, False):
        rs = [r for r in rows if r["carry"] == g]
        two_stage = 0
        for r in rs:
            lt, la = r["lens"]["baseline"]["p_true"], r["lens"]["baseline"]["p_alt"]
            dt, da = first_above(lt), first_above(la)
            if da is not None and (dt is None or da < dt):
                two_stage += 1
        mean_curve = lambda key, cond: [round(sum(r["lens"][cond][key][d] for r in rs) / len(rs), 4) for d in range(D)]
        lens_summary["carry" if g else "no_carry"] = {
            "n": len(rs), "alt_digit_dominates_before_true": two_stage,
            "mean_p_true_baseline": mean_curve("p_true", "baseline"), "mean_p_alt_baseline": mean_curve("p_alt", "baseline"),
            "mean_p_true_ablated": mean_curve("p_true", "ablated"), "mean_p_alt_ablated": mean_curve("p_alt", "ablated"),
        }

    report = {
        "experiment": "E1_carry_heads", "model_id": info.model_id, "is_demo": info.is_demo,
        "layer": layer, "focus_heads": focus, "random_control_heads": random_heads, "n_per_group": args.n,
        "seed": args.seed, "seconds": round(time.time() - t0, 1),
        "status": "causal", "scope": f"{2 * args.n} controlled 2-digit additions (seed {args.seed}), zero-ablation, readout at the tens digit",
        "summary": summary, "attention_from_answer_position": attn_summary, "free_running_outputs": out_pattern,
        "lens_two_stage": lens_summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    print(f"\nE1 — {info.model_id}{' (DEMO fixture)' if info.is_demo else ''} · {2 * args.n} problems · {report['seconds']}s\n")
    print("| condition | group | tens acc | → alt digit | mean P(true) | Δ logit-diff (± SEM) |")
    print("|---|---|---|---|---|---|")
    show = ["baseline"] + [f"L{layer}.H{h}" for h in focus] + [both_key] + \
        [f"mean-abl L{layer}.H{h}" for h in focus] + [f"mean-abl {both_key}"]
    ctrl = [c for c in conds if c.startswith("ctrl")]  # zero-ablation of the other heads in the same layer
    rnd = [c for c in conds if c.startswith("rand")]
    for c in show:
        for g in ("carry", "no_carry"):
            x = summary[c][g]
            print(f"| {c} | {g} | {x['tens_accuracy']:.2f} | {x['frac_predicts_alt_digit']:.2f} | {x['mean_p_true']:.3f} | "
                  f"{x['mean_delta_logit_diff']:+.2f} ± {x['sem_delta_logit_diff']:.2f} |")
    for name, group in (("same-layer controls (mean over heads)", ctrl), ("random other-layer heads (mean)", rnd)):
        for g in ("carry", "no_carry"):
            if not group:
                continue
            acc = sum(summary[c][g]["tens_accuracy"] for c in group) / len(group)
            d = sum(summary[c][g]["mean_delta_logit_diff"] for c in group) / len(group)
            worst = min(group, key=lambda c: summary[c][g]["mean_delta_logit_diff"])
            print(f"| {name} | {g} | {acc:.2f} | – | – | {d:+.2f} (worst {worst}: {summary[worst][g]['mean_delta_logit_diff']:+.2f}) |")
    print("\nattention from answer position (mean weight):", json.dumps(attn_summary, indent=1))
    print("free-running outputs:", json.dumps(out_pattern, indent=1))
    for g, v in lens_summary.items():
        print(f"lens two-stage ({g}): carry-mishandled digit dominates before the true digit in "
              f"{v['alt_digit_dominates_before_true']}/{v['n']} problems")
        print("   depth       " + " ".join(f"{d:>5d}" for d in range(D)))
        for k in ("mean_p_true_baseline", "mean_p_alt_baseline", "mean_p_true_ablated", "mean_p_alt_ablated"):
            print(f"   {k[7:]:<14}" + " ".join(f"{x:5.2f}" for x in v[k]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
