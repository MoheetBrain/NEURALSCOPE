#!/usr/bin/env python3
"""V1 acceptance flow against a running API (real model), step by step, with timings.

    ./scripts/api.sh &   then   .venv/bin/python scripts/acceptance.py [--api http://127.0.0.1:8000] [--skip-sweep]

Mirrors what the /app workspace does: run → tokens → matrix → cell → Logit Lens → attention → head →
(head sweep) → ablate + rerun → diff → save + reopen experiment → Math Lab benchmark → correct-vs-incorrect compare.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PROMPT = "What is 37 + 58? Give only the answer."


class Api:
    def __init__(self, base: str):
        self.base = base

    def __call__(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(self.base + path, method=method, headers={"content-type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"FAIL {method} {path}: HTTP {e.code} {e.read().decode()[:300]}")

    def job(self, j: dict) -> dict:
        while True:
            j = self("GET", f"/api/jobs/{j['id']}")
            if j["status"] == "done":
                return j["result"]
            if j["status"] == "error":
                raise SystemExit(f"FAIL job {j['kind']}: {j['error']}")
            time.sleep(2)


def step(name: str):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            out = fn(*a, **k)
            print(f"  ✓ {name:<44} {time.time() - t:6.1f}s  {out if isinstance(out, str) else ''}")
            return out
        return run
    return deco


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--bench-n", type=int, default=12)
    args = ap.parse_args()
    api = Api(args.api)
    ctx: dict = {}

    h = api("GET", "/api/health")
    check(h["status"] == "ready", f"API not ready: {h}")
    m = api("GET", "/api/model")
    print(f"NeuralScope acceptance · {m['model_id']} · {m['n_layers']}L×{m['n_heads']}H · {m['device']}"
          f"{' · DEMO FIXTURE' if m['is_demo'] else ''}")

    @step("run prompt (generate + capture)")
    def s_run():
        r = api("POST", "/api/runs", {"prompt": PROMPT, "max_new_tokens": 8})
        check(r["trace_consistent"], "teacher-forced capture must reproduce greedy tokens")
        ctx["run"] = r
        return f"→ {r['generated_text']!r} · {len(r['tokens'])} tokens"

    @step("tokens + layers")
    def s_tokens():
        rid = ctx["run"]["id"]
        toks = api("GET", f"/api/runs/{rid}/tokens")
        lay = api("GET", f"/api/runs/{rid}/layers")
        check(len(lay["depth_labels"]) == m["n_layers"] + 1, "depth labels")
        return f"{sum(t['role'] == 'user' for t in toks)} user tokens · {len(lay['depth_labels'])} depths"

    @step("TOKEN × LAYER matrix (target rank)")
    def s_matrix():
        mat = api("GET", f"/api/runs/{ctx['run']['id']}/matrix?metric=target_rank")
        pos = ctx["run"]["target"]["position"]
        check(mat["ranks"][-1][pos] == 1, "target must be rank 1 at the final depth")
        return f"{mat['n_depths']}×{mat['n_tokens']} cells"

    @step("cell inspector")
    def s_cell():
        pos = ctx["run"]["target"]["position"]
        c = api("GET", f"/api/runs/{ctx['run']['id']}/cell/{m['n_layers'] - 3}/{pos}")
        check(len(c["vector"]["values"]) == m["d_model"], "full residual vector")
        return f"{c['depth_label']} ‖h‖={c['vector']['norm']:.1f} lens top-1 {c['lens']['top'][0]['text']!r}"

    @step("Logit Lens through depth")
    def s_lens():
        L = api("GET", f"/api/runs/{ctx['run']['id']}/logit-lens")
        return f"target {L['target']['text']!r} first top-1 at depth {L['first_depth_target_top1']}"

    @step("attention (layer) + head detail")
    def s_attn():
        rid = ctx["run"]["id"]
        layer = min(22, m["n_layers"] - 1)
        a = api("GET", f"/api/runs/{rid}/attention/{layer}")
        hd = api("GET", f"/api/runs/{rid}/heads/{layer}/{min(10, m['n_heads'] - 1)}")
        check(len(a["patterns"]) == m["n_heads"], "all heads")
        return f"L{layer}: {len(a['patterns'])} heads · head DLA→target {hd['dla_to_target']['value']:+.3f}"

    @step("direct logit attribution")
    def s_dla():
        d = api("GET", f"/api/runs/{ctx['run']['id']}/dla")
        check(d["decomposition_error"] < 1e-2, "DLA must sum to the logit")
        return f"Σ components {d['sum_of_components']:.4f} vs logit {d['actual_logit']:.4f}"

    @step("head-ablation sweep (all heads)")
    def s_sweep():
        res = api.job(api("POST", "/api/sweeps/heads", {"run_id": ctx["run"]["id"]}))
        w = res["most_harmful"][0]
        ctx["head"] = (w["layer"], w["head"])
        return f"most harmful L{w['layer']}.H{w['head']} Δ={w['delta']:+.2f}"

    @step("ABLATE + RERUN")
    def s_ablate():
        layer, head = ctx.get("head", (min(22, m["n_layers"] - 1), min(10, m["n_heads"] - 1)))
        iv = api("POST", "/api/interventions", {"base_run_id": ctx["run"]["id"],
                                                "interventions": [{"type": "head_ablation", "layer": layer, "head": head}]})
        c = iv["comparison"]
        check(c["status"] == "causal", "causal status")
        ctx["child"] = iv["run"]
        return (f"L{layer}.H{head}: P {c['before']['prob']:.4f}→{c['after']['prob']:.4f} KL {c['kl_at_target']:.2f} "
                f"out {c['output_before']!r}→{c['output_after']!r}")

    @step("run diff (compare baseline vs intervention)")
    def s_diff():
        d = api("POST", "/api/compare", {"run_a": ctx["run"]["id"], "run_b": ctx["child"]["id"]})
        dm = api("GET", f"/api/runs/{ctx['child']['id']}/matrix?metric=delta_resid")
        check(d["mode"] == "intervention" and dm["status"] == "causal", "diff")
        return f"effect starts at depth {d['comparison']['earliest_affected_depth']}"

    @step("save + reopen experiment")
    def s_exp():
        e = api("POST", "/api/experiments", {"name": "acceptance: 37+58 ablation", "run_ids": [ctx["run"]["id"], ctx["child"]["id"]],
                                             "notes": "hypothesis (INFERRED): head routes the carry", "payload": {"metric": "delta_resid"}})
        again = api("GET", f"/api/experiments/{e['id']}")
        check(len(again["runs"]) == 2, "experiment reopens with both runs")
        return e["id"]

    @step("Math Lab benchmark")
    def s_bench():
        b = api.job(api("POST", "/api/math/benchmarks", {"n": args.bench_n, "seed": 0}))
        s = b["summary"]
        ctx["bench"] = b
        return f"{s['correct']}/{s['n']} correct · {s['incorrect']} incorrect · {s['truncated']} truncated"

    @step("compare correct vs incorrect trace")
    def s_cmp():
        rs = ctx["bench"]["results"]
        ok = next((r for r in rs if r["status"] == "correct"), None)
        bad = next((r for r in rs if r["status"] == "incorrect"), None)
        if not (ok and bad):
            return "skipped (need one correct and one incorrect run)"
        c = api("POST", "/api/compare", {"run_a": ok["run_id"], "run_b": bad["run_id"]})
        check(c["mode"] == "cross_run" and len(c["cross_cosine"]) == m["n_layers"] + 1, "cross-run compare")
        return f"{ok['question'][:24]}… vs {bad['question'][:24]}… (model {bad['model_output'][:10]!r})"

    for s in [s_run, s_tokens, s_matrix, s_cell, s_lens, s_attn, s_dla] + ([] if args.skip_sweep else [s_sweep]) + \
             [s_ablate, s_diff, s_exp, s_bench, s_cmp]:
        s()
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    sys.exit(main())
