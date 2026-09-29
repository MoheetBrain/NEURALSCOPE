"""Math Lab ↔ model integration through the service (fixture model)."""

from __future__ import annotations

from neuralscope.mathlab import DEMO_PROBLEM, default_benchmark, make_problem
from neuralscope.schemas.interventions import HeadAblation


def test_run_math_problem_records_oracle_label(service):
    rec = service.run_math_problem(DEMO_PROBLEM, max_new_tokens=4)
    m = rec["math"]
    assert rec["kind"] == "math" and rec["trace_level"] == "minimal"
    assert m["problem"]["oracle_answer"] == "95"
    # a random-weight model will not stop within 4 tokens → the output is labelled truncated, never "incorrect"
    assert rec["correctness"] in {"correct", "incorrect", "truncated", "unparseable", "ambiguous"}
    if m["truncated"]:
        assert rec["correctness"] == "truncated"
    lp = m["oracle_answer_logprob"]
    assert lp["tokens"] == ["9", "5"] and lp["total"] <= 0
    assert rec["target"]["source"] == "oracle"
    assert rec["target"]["token"] == "9"
    # lens summaries are lazy for math runs
    assert "lens_top_ids" not in rec["tensor_meta"]["keys"]
    assert service.matrix(rec["id"], "lens_top1_prob")["values"]


def test_answer_logprob_matches_capture(service):
    tp = service.adapter.tokenize(DEMO_PROBLEM.question)
    lp = service.answer_logprob(tp.input_ids, "95")
    ids = tp.input_ids + service.adapter.encode("95")
    import torch

    logits = service.adapter.capture(ids, "minimal").logits
    P = len(tp.input_ids)
    ref = torch.log_softmax(logits, -1)
    total = float(ref[P - 1, ids[P]] + ref[P, ids[P + 1]])
    assert abs(total - lp["total"]) < 1e-3


def test_benchmark_and_intervention_eval(service):
    probs = default_benchmark(seed=3, n=4)
    bench = service.run_benchmark(probs, name="t", seed=3)
    assert bench["status"] == "done" and len(bench["results"]) == 4
    s = bench["summary"]
    assert s["n"] == 4 and s["correct"] + s["incorrect"] + s["unparseable"] + s["truncated"] == 4
    runs = service.store.list_runs(benchmark_id=bench["id"])
    assert len(runs) == 4 and all(r["math"] for r in runs)
    ev = service.benchmark_intervention_eval(bench["id"], [HeadAblation(layer=1, head=0)])
    assert ev["n"] == 4 and ev["status"] == "causal" and len(ev["rows"]) == 4
    again = service.store.get_benchmark(bench["id"])
    assert again["summary"]["intervention_evals"][0]["id"] == ev["id"]


def test_math_intervention_reports_correctness(service):
    rec = service.run_math_problem(make_problem("add", {"a": 12, "b": 34}), max_new_tokens=3)
    res = service.run_intervention(rec["id"], [HeadAblation(layer=0, head=0)])
    c = res["comparison"]
    assert "correctness_before" in c and "correctness_after" in c
