#!/usr/bin/env python3
"""Pre-register (lock) a SelfLab battery run BEFORE any of its trials exist.

Builds the spec exactly as ``run_pilot.py`` would, derives the deterministic trial plan with the real tokenizer
and model geometry, and writes:

* ``<out-dir>/<run-id>/spec.json`` — the commitment ``run_pilot.py --resume`` will verify (plan/spec hash);
* ``docs/selflab/prereg/<run-id>.lock.json`` — the committed pre-registration: spec, spec/plan sha256, sha256 of
  every analysis-relevant source file, the analysis rules text, and the exact command that will run it.

It runs NO trial and no generation. ALWAYS run it under the model lock::

    .venv/bin/python scripts/model_lock.py -- .venv/bin/python scripts/selflab/lock_spec.py \
        --template source_influence_battery --split heldout_confirmation --seed 200 --n-items 20 --run-id <id>

Commit the lock file to git BEFORE running ``run_pilot.py --resume --run-id <id>``: the commit history then shows the
protocol fixed before any held-out outcome existed. (Hashes support integrity checks; they do not make local files
tamper-proof.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = [
    "packages/neuralscope/neuralscope/selflab/templates/source_influence.py",
    "packages/neuralscope/neuralscope/selflab/datasets/source_corpus.py",
    "packages/neuralscope/neuralscope/selflab/parsing.py",
    "packages/neuralscope/neuralscope/selflab/stats.py",
    "packages/neuralscope/neuralscope/selflab/measure.py",
    "packages/neuralscope/neuralscope/selflab/compose.py",
    "packages/neuralscope/neuralscope/selflab/schemas.py",
    "packages/neuralscope/neuralscope/instrumentation/hf_decoder.py",
    "scripts/selflab/run_pilot.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="source_influence_battery")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--split", default="heldout_confirmation")
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "data/selflab/pilots"))
    ap.add_argument("--prereg-dir", default=str(ROOT / "docs/selflab/prereg"))
    args = ap.parse_args()

    from neuralscope.config import Settings
    from neuralscope.instrumentation import load_adapter
    from neuralscope.selflab.provenance import now_iso, run_provenance, sha256_json
    from neuralscope.selflab.registry import get_template

    tpl = get_template(args.template)
    run_dir = Path(args.out_dir) / args.run_id
    if (run_dir / "spec.json").exists() or (run_dir / "trials.jsonl").exists():
        print(f"[lock] {run_dir} already has a spec or trials; refusing to re-lock", file=sys.stderr)
        return 2
    s = Settings()
    s.model_id, s.device, s.dtype = args.model, args.device, args.dtype
    adapter = load_adapter(s)

    spec = tpl.default_spec(None).model_copy(update={"split": args.split, "seed": args.seed})
    params = dict(spec.params)
    if "n_items" in params:
        params["n_items"] = args.n_items
    for kv in args.param:
        k, v = kv.split("=", 1)
        params[k] = json.loads(v)
    tpl.params_model.model_validate(params)
    spec = spec.model_copy(update={"params": params})
    plan = tpl.build_plan(spec, None, adapter)
    spec_sha = sha256_json(spec.model_dump())
    plan_sha = sha256_json([p.model_dump() for p in plan])
    prov = run_provenance(adapter, decoding={"strategy": "greedy"}, seed=spec.seed)
    run_cmd = (f".venv/bin/python scripts/model_lock.py -- .venv/bin/python scripts/selflab/run_pilot.py "
               f"--template {args.template} --model {args.model} --device {args.device} --dtype {args.dtype} "
               f"--run-id {args.run_id} --resume --summary-out docs/selflab/results/{args.run_id}.json")
    committed = {"run_id": args.run_id, "template": args.template, "spec": spec.model_dump(),
                 "spec_sha256": spec_sha, "plan_sha256": plan_sha, "n_planned": len(plan),
                 "committed_at": now_iso(), "provenance": prov.model_dump(), "command": " ".join(sys.argv)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(committed, indent=1, default=str))

    mod = sys.modules.get(type(tpl).__module__)
    rules = getattr(mod, "V2_RULES_TEXT", None)
    h3_text = getattr(mod, "V2_H3_TEXT", None)
    lock = {
        "kind": "selflab_preregistration",
        "run_id": args.run_id,
        "locked_at": committed["committed_at"],
        "statement": "Spec, plan and analysis code fixed before any trial of this run was executed.",
        "spec": committed["spec"], "spec_sha256": spec_sha, "plan_sha256": plan_sha, "n_planned": len(plan),
        "families_in_plan": sorted({p.family for p in plan}),
        "conditions_in_plan": sorted({p.condition for p in plan}),
        "n_units": len({p.group_id for p in plan}),
        "model": prov.model.model_dump(),
        "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES},
        "analysis_rules_text": rules,
        "analysis_rules_text_sha256": hashlib.sha256(rules.encode()).hexdigest() if rules else None,
        "h3_text": h3_text,
        "h3_text_sha256": hashlib.sha256(h3_text.encode()).hexdigest() if h3_text else None,
        "run_command": run_cmd,
    }
    out = Path(args.prereg_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run_id}.lock.json").write_text(json.dumps(lock, indent=1, default=str))
    print(f"[lock] {args.run_id}: {len(plan)} planned trials, {lock['n_units']} units, families "
          f"{lock['families_in_plan']}, spec {spec_sha[:12]}, plan {plan_sha[:12]}")
    print(f"[lock] wrote {run_dir / 'spec.json'} and {out / (args.run_id + '.lock.json')}")
    print(f"[lock] run after committing the lock:\n  {run_cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
