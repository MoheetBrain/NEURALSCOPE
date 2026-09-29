#!/usr/bin/env python3
"""Capture the landing-page hero trace from a REAL model run via the running API.

Nothing here is hand-shaped: the run, the Logit Lens matrix, the target-rank
trajectory, the causally selected head (most harmful in an exhaustive
single-head ablation sweep) and the before/after intervention metrics all come
from the API.

    ./scripts/api.sh &          # real model
    .venv/bin/python scripts/export_hero_snapshot.py [--run-id RUN] [--api http://127.0.0.1:8000]

Writes apps/web/public/snapshots/hero-trace.json (source: "REAL"); rebuild the web app afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "apps/web/public/snapshots/hero-trace.json"
PROMPT = "What is 37 + 58? Give only the answer."


def call(api: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(api + path, method=method, headers={"content-type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def wait_job(api: str, job: dict) -> dict:
    while True:
        j = call(api, "GET", f"/api/jobs/{job['id']}")
        if j["status"] == "done":
            return j["result"]
        if j["status"] == "error":
            raise RuntimeError(j["error"])
        print(f"  … {j['message']}", flush=True)
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--run-id")
    args = ap.parse_args()
    api = args.api

    info = call(api, "GET", "/api/model")
    if info["is_demo"]:
        raise SystemExit("refusing to export a hero snapshot from the DEMO fixture model")
    run = call(api, "GET", f"/api/runs/{args.run_id}") if args.run_id else \
        call(api, "POST", "/api/runs", {"prompt": PROMPT, "max_new_tokens": 8})
    rid = run["id"]
    pos = run["target"]["position"]
    start = run["user_span"][0]
    cols = list(range(start, pos + 1))  # user text + chat-template tail up to the answer position
    print(f"run {rid}: output {run['generated_text']!r}, target {run['target']['token']!r} @ t{pos}")

    mat = call(api, "GET", f"/api/runs/{rid}/matrix?metric=lens_top1_prob")
    lens = call(api, "GET", f"/api/runs/{rid}/logit-lens?position={pos}")
    print("head-ablation sweep (all heads, one at a time)…")
    sweep = wait_job(api, call(api, "POST", "/api/sweeps/heads", {"run_id": rid}))
    worst = sweep["most_harmful"][0]
    print(f"most harmful head: L{worst['layer']}.H{worst['head']} Δlogprob={worst['delta']}")
    iv = call(api, "POST", "/api/interventions", {
        "base_run_id": rid, "interventions": [{"type": "head_ablation", "layer": worst["layer"], "head": worst["head"]}]})
    c = iv["comparison"]

    def clean(s: str) -> str:
        return s.replace("<|im_end|>", "").strip()

    snap = {
        "source": "REAL",
        "model_id": info["model_id"],
        "run_id": rid,
        "captured_at": run["created_at"],
        "prompt": PROMPT,
        "tokens": [run["tokens"][i]["text"] for i in cols],
        "depth_labels": mat["depth_labels"],
        "matrix": {"metric": "lens_top1_prob", "label": "Logit Lens top-1 probability (DERIVED)",
                   "values": [[round(float(row[i] or 0.0), 4) for i in cols] for row in mat["values"]]},
        "target": {"token": run["target"]["token"], "column": len(cols) - 1,
                   "rank_by_depth": [d["target"]["rank"] for d in lens["depths"]],
                   "prob_by_depth": [round(d["target"]["prob"], 6) for d in lens["depths"]]},
        "intervention": {
            "type": "head_ablation", "layer": worst["layer"], "head": worst["head"],
            "description": (f"Zero-ablate attention head L{worst['layer']}.H{worst['head']} (all positions) — "
                            f"the most harmful of {len(sweep['delta_metric']) * len(sweep['delta_metric'][0])} "
                            "heads in an exhaustive single-head ablation sweep"),
            "target_prob_before": c["before"]["prob"], "target_prob_after": c["after"]["prob"],
            "kl": c["kl_at_target"], "output_before": clean(c["output_before"]), "output_after": clean(c["output_after"]),
        },
        "intervention_run_id": iv["run"]["id"],
        "scope": c["scope"],
    }
    OUT.write_text(json.dumps(snap, indent=1))
    print(f"wrote {OUT} — rebuild the web app to bundle it")


if __name__ == "__main__":
    main()
