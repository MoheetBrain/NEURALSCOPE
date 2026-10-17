#!/usr/bin/env python3
"""In-process SelfLab pilot runner (SA3): plan → trials (persisted as JSONL, resumable) → analysis → summary JSON.

Runs a registered SelfLab template against a real local model WITHOUT the API/engine, so a pilot can run before (or
independently of) the job engine. It follows the same contract as the engine:

* the spec is committed first (spec and plan sha256 written to ``spec.json``);
* every trial is appended to ``trials.jsonl`` and fsync'ed as soon as it finishes (errors included);
* ``--resume`` re-derives the plan and REFUSES if its hash differs from the committed one; completed trials are never
  re-run or re-randomised;
* the analysis is the template's pure ``analyze`` over the stored trials.

ALWAYS run real models under the model lock::

    .venv/bin/python scripts/model_lock.py -- .venv/bin/python scripts/selflab/run_pilot.py \\
        --template source_influence_battery --model Qwen/Qwen2.5-1.5B-Instruct --n-items 20 --split debug_pilot

Outputs: ``data/selflab/pilots/<run_id>/{spec.json,trials.jsonl,analysis.json,lab/}`` (local data) and, with
``--summary-out``, a small committed summary (provenance, hashes, numbers, evidence, ≤3 example trials).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _jsonable(o):
    return json.loads(json.dumps(o, default=str))


def _load_trials(path: Path):
    from neuralscope.selflab.schemas import Trial

    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                t = Trial.model_validate_json(line)
                out[t.index] = t  # last write wins
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="source_influence_battery")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF id, or 'demo' for the fixture model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--split", default="debug_pilot", choices=["debug_pilot", "discovery", "heldout_confirmation"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=None, help="sets params.n_items when the template has it")
    ap.add_argument("--param", action="append", default=[], help="key=JSON value, e.g. patching_subset=3")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "data/selflab/pilots"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-trials", type=int, default=None, help="stop after this many NEW trials (debugging)")
    ap.add_argument("--summary-out", default=None, help="committed summary JSON path (docs/selflab/results/...)")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-run the (pure) analysis of an existing run's stored trials; loads no model")
    ap.add_argument("--rules-version", default=None,
                    help="with --analyze-only: analyse under another analysis version's rules (spec.version and the "
                         "template's default hypotheses for that version). The output is labelled POST-HOC.")
    ap.add_argument("--label", default=None, help="free-text label stored in the summary (e.g. 'post-hoc …')")
    args = ap.parse_args()
    if args.analyze_only:
        return _analyze_only(args)

    import torch

    from neuralscope.config import DEMO_MODEL_ID, Settings
    from neuralscope.experiments.service import NeuralScopeService
    from neuralscope.experiments.store import ExperimentStore
    from neuralscope.instrumentation import load_adapter
    from neuralscope.selflab.provenance import now_iso, run_provenance, sha256_json
    from neuralscope.selflab.registry import RunContext, get_template

    tpl = get_template(args.template)
    model_id = DEMO_MODEL_ID if args.model == "demo" else args.model
    short = model_id.split("/")[-1].replace(".", "")
    run_id = args.run_id or f"{args.template}_{short}_{args.split}_s{args.seed}_{date.today().isoformat()}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path, trials_path = run_dir / "spec.json", run_dir / "trials.jsonl"

    s = Settings()
    s.model_id = model_id
    s.device = args.device
    s.dtype = args.dtype
    s.num_threads = args.threads
    s.data_dir = run_dir / "lab"
    s.trace_level = "minimal"
    if args.threads:
        torch.set_num_threads(args.threads)
    t_load = time.time()
    adapter = load_adapter(s)
    print(f"[pilot] loaded {adapter.info.model_id} on {adapter.info.device} ({adapter.info.dtype}) "
          f"in {time.time() - t_load:.1f}s", flush=True)
    service = NeuralScopeService(adapter, ExperimentStore(s.data_dir), s)

    # ---------------------------------------------------------------- commit (or verify on resume)
    if spec_path.exists():
        if not args.resume:
            print(f"[pilot] {spec_path} exists; pass --resume to continue this run or choose another --run-id",
                  file=sys.stderr)
            return 2
        committed = json.loads(spec_path.read_text())
        from neuralscope.selflab.schemas import ExperimentSpec

        spec = ExperimentSpec.model_validate(committed["spec"])
    else:
        spec = tpl.default_spec(None)
        spec = spec.model_copy(update={"split": args.split, "seed": args.seed})
        params = dict(spec.params)
        if args.n_items is not None and "n_items" in params:
            params["n_items"] = args.n_items
        for kv in args.param:
            k, v = kv.split("=", 1)
            params[k] = json.loads(v)
        tpl.params_model.model_validate(params)
        spec = spec.model_copy(update={"params": params})
        committed = None
    plan = tpl.build_plan(spec, None, adapter)
    spec_sha = sha256_json(spec.model_dump())
    plan_sha = sha256_json([p.model_dump() for p in plan])
    if committed is not None:
        if committed["plan_sha256"] != plan_sha or committed["spec_sha256"] != spec_sha:
            print(f"[pilot] REFUSING to resume: plan/spec hash differs from the committed one "
                  f"({committed['plan_sha256'][:12]} vs {plan_sha[:12]})", file=sys.stderr)
            return 3
    else:
        prov = run_provenance(adapter, decoding={"strategy": "greedy"}, seed=spec.seed)
        committed = {"run_id": run_id, "template": args.template, "spec": spec.model_dump(), "spec_sha256": spec_sha,
                     "plan_sha256": plan_sha, "n_planned": len(plan), "committed_at": now_iso(),
                     "provenance": prov.model_dump(), "command": " ".join(sys.argv)}
        spec_path.write_text(json.dumps(committed, indent=1, default=str))
    print(f"[pilot] run {run_id}: {len(plan)} planned trials, spec {spec_sha[:12]}, plan {plan_sha[:12]}", flush=True)

    # ---------------------------------------------------------------- trials
    done = _load_trials(trials_path)
    ctx = RunContext(adapter=adapter, service=service, experiment_id=run_id, spec=spec, case=None)
    t_start = time.time()
    n_new = 0
    with open(trials_path, "a") as f:
        for item in plan:
            if item.index in done:
                continue
            if args.max_trials is not None and n_new >= args.max_trials:
                break
            tr = tpl.run_trial(item, ctx)
            f.write(tr.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
            done[item.index] = tr
            n_new += 1
            el = time.time() - t_start
            print(f"[pilot] {item.index + 1}/{len(plan)} {item.condition:<20} {item.family} {tr.status:<10} "
                  f"{tr.output_text[:24]!r:<28} {tr.latency_s:6.2f}s  (elapsed {el / 60:.1f} min)"
                  + (f"  ERROR {tr.error[:120]}" if tr.error else ""), flush=True)

    trials = [done[i] for i in sorted(done)]
    complete = len(trials) == len(plan)
    res = tpl.analyze(spec, None, trials, run_id)
    analysis = {"run_id": run_id, "complete": complete, "n_trials": len(trials), "n_planned": len(plan),
                "summary": res.summary, "evidence": [e.model_dump() for e in res.evidence],
                "analysis_version": res.analysis_version}
    (run_dir / "analysis.json").write_text(json.dumps(_jsonable(analysis), indent=1))
    secs = sum(t.latency_s for t in trials)
    print(f"[pilot] analysis written ({'complete' if complete else 'PARTIAL'}; {len(trials)}/{len(plan)} trials; "
          f"sum latency {secs / 60:.1f} min)", flush=True)
    for e in res.evidence:
        print(f"  {e.id.split(':ev:')[-1]:<24} {e.kind:<26} {e.status:<15} {e.title}")

    if args.summary_out:
        _write_summary(args, run_id, run_dir, committed, trials, res, complete, len(plan))
    return 0 if complete else 1


def _write_summary(args, run_id, run_dir, committed, trials, res, complete, n_planned, reanalysis=False,
                   posthoc=None, analysed_spec=None) -> None:
    secs = sum(t.latency_s for t in trials)
    ex = []
    for t in trials[: args.n_examples]:
        ex.append({"index": t.index, "condition": t.condition, "family": t.family, "status": t.status,
                   "output_text": t.output_text[:80], "ground_truth": t.ground_truth, "parsed": t.parsed,
                   "source_report": [{k: r.get(k) for k in ("label_order", "argmax_category", "mass",
                                                              "category_probs", "free_text", "free_text_parse")}
                                     for r in (t.measurements.get("source_report") or [])]})
    evid = [{k: e.model_dump().get(k) for k in ("id", "kind", "status", "title", "statement", "scope",
                                                 "limitations", "alternatives", "run_ids")}
            for e in res.evidence]  # numbers: see "summary" (measurements are in analysis.json)
    out = {"experiment": committed.get("template", args.template), "run_id": run_id, "date": date.today().isoformat(), "complete": complete,
           "n_trials": len(trials), "n_planned": n_planned, "spec_sha256": committed["spec_sha256"],
           "plan_sha256": committed["plan_sha256"], "committed_at": committed["committed_at"],
           "provenance": committed["provenance"], "command": committed.get("command"),
           "spec": committed["spec"], "wall_seconds_sum_latency": round(secs, 1),
           "summary": res.summary, "evidence": evid, "example_trials": ex,
           "data_location": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
           "reanalysis_with_current_code": reanalysis, "analysis_version": res.analysis_version,
           "label": getattr(args, "label", None), "posthoc_rules": posthoc,
           "analysed_spec": analysed_spec.model_dump() if (posthoc and analysed_spec is not None) else None,
           "note": "Local data (trials.jsonl, Lab runs) is not committed. Free text is limited to the examples."}
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(_jsonable(out), indent=1))
    print(f"[pilot] summary → {args.summary_out}", flush=True)


def _analyze_only(args) -> int:
    from neuralscope.selflab.registry import get_template
    from neuralscope.selflab.schemas import ExperimentSpec

    run_dir = Path(args.out_dir) / args.run_id
    committed = json.loads((run_dir / "spec.json").read_text())
    tpl = get_template(committed["template"])
    spec = ExperimentSpec.model_validate(committed["spec"])
    posthoc = None
    if args.rules_version and str(args.rules_version) != str(spec.version):
        default = tpl.default_spec(None)
        if str(default.version) != str(args.rules_version):
            raise SystemExit(f"template default spec is version {default.version}, not {args.rules_version}")
        posthoc = {"committed_spec_version": str(spec.version), "analysed_under_version": str(args.rules_version),
                   "hypotheses_replaced_with": "template default hypotheses of that version",
                   "params_unchanged": True}
        spec = spec.model_copy(update={"version": str(args.rules_version), "hypotheses": default.hypotheses,
                                       "primary_metric": default.primary_metric})
    trials = [t for _, t in sorted(_load_trials(run_dir / "trials.jsonl").items())]
    res = tpl.analyze(spec, None, trials, args.run_id)
    out = {"run_id": args.run_id, "reanalysis": True, "posthoc_rules": posthoc, "label": args.label,
           "n_trials": len(trials), "n_planned": committed["n_planned"],
           "summary": res.summary, "evidence": [e.model_dump() for e in res.evidence],
           "analysis_version": res.analysis_version}
    dest = run_dir / (f"reanalysis_rules_v{args.rules_version}.json" if posthoc else "reanalysis.json")
    dest.write_text(json.dumps(_jsonable(out), indent=1))
    if args.summary_out:
        _write_summary(args, args.run_id, run_dir, committed, trials, res,
                       len(trials) == committed["n_planned"], committed["n_planned"], reanalysis=True,
                       posthoc=posthoc, analysed_spec=spec)
    for e in res.evidence:
        print(f"  {e.id.split(':ev:')[-1]:<24} {e.kind:<26} {e.status:<15} {e.title}")
    print(f"[pilot] re-analysis → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
