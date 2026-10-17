"""SA6 falsification pass over STORED Experiment A trials (no model calls, no re-runs).

Run as a script to print the analysis used in docs/selflab/VALIDATION_REPORT.md:

    .venv/bin/python -m tests.test_selflab_validation_falsification \
        data/selflab/pilots/expA_pilot_1p5b_debug_s0_2026-10-17_v2/trials.jsonl \
        docs/selflab/results/EXP_A_pilot_2026-10-17.json

As tests (skipped when the local pilot data is absent — it is not committed): the stored analysis must be
reproducible from the stored trials (``analyze`` is pure), and the falsification numbers must be computable.

Questions asked of an apparent "self-report beats chance" finding:
  1. Is it matched by the prompt-only heuristic or by the transcript observer?
  2. Does it survive label reversal (per-order accuracy, letter-position bias)?
  3. Is it driven by one family or a few items (leave-one-out)?
  4. Are exclusions (overdetermined / undetermined / abstentions / malformed) hiding failures?
  5. Is the letter-option probability mass low?
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/selflab/results/EXP_A_pilot_2026-10-17.json"


def _pilot_trials() -> Path:
    """The stored trials the committed summary points at (``data_location``); local data, not committed."""
    try:
        return ROOT / json.loads(SUMMARY.read_text())["data_location"] / "trials.jsonl"
    except (OSError, KeyError, ValueError):
        return ROOT / "data/selflab/pilots/expA_pilot_1p5b_debug_s0_2026-10-17/trials.jsonl"


PILOT = _pilot_trials()


def load(trials_path: Path, summary_path: Path):
    from neuralscope.selflab.schemas import ExperimentSpec, Trial

    trials = [Trial.model_validate_json(line) for line in trials_path.read_text().splitlines() if line.strip()]
    stored = json.loads(summary_path.read_text())
    return trials, ExperimentSpec.model_validate(stored["spec"]), stored


def falsify(trials, spec, B: int = 2000) -> dict:
    from neuralscope.selflab import stats
    from neuralscope.selflab.schemas import SOURCE_CATEGORIES
    from neuralscope.selflab.templates import source_influence as si

    labels = si.causal_labels(trials, "2" if si.is_v2(spec) else "1")
    live = [t for t in trials if t.status != "error"]
    scored = [t for t in live if labels[t.id] in si.SCORED_LABELS and si.report_probs(t) is not None]
    y = [labels[t.id] for t in scored]
    g = [t.group_id for t in scored]
    rep = [si._argmax(si.report_probs(t)) for t in scored]
    maj = Counter(y).most_common(1)[0][0]
    self_c = [float(a == b) for a, b in zip(rep, y)]
    maj_c = [float(lbl == maj) for lbl in y]
    pr_c = [float(si.prompt_heuristic_label(t) == lbl) for t, lbl in zip(scored, y)]
    tr_c = [float(si.transcript_observer_label(t) == lbl) for t, lbl in zip(scored, y)]

    def pe(a, b, groups=None):
        r = stats.paired_effect(a, b, groups or g, B=B, seed=0)
        return {k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_value", "n_units", "n_obs")}

    out: dict = {"n_trials": len(trials), "n_scored": len(scored), "n_units": len(set(g)), "majority_label": maj,
                 "accuracy": {"self_letter": stats.mean_of(self_c), "majority": stats.mean_of(maj_c),
                              "prompt_heuristic": stats.mean_of(pr_c), "transcript": stats.mean_of(tr_c)},
                 "paired": {"self_minus_majority": pe(self_c, maj_c), "self_minus_prompt": pe(self_c, pr_c),
                            "self_minus_transcript": pe(self_c, tr_c)}}

    # 2. label reversal / position bias
    per_order, letters, pos_acc = [], [Counter(), Counter()], defaultdict(list)
    for k in (0, 1):
        acc = [float(t.measurements["source_report"][k]["argmax_category"] == lbl) for t, lbl in zip(scored, y)]
        per_order.append(stats.mean_of(acc))
    for t in live:
        for k, r in enumerate(t.measurements.get("source_report") or []):
            letters[k][r["argmax_letter"]] += 1
    for t, lbl in zip(scored, y):
        for r in t.measurements["source_report"]:
            pos = r["label_order"].index(lbl)
            pos_acc["ABCD"[pos]].append(float(r["argmax_category"] == lbl))
    consistent = [(t, lbl) for t, lbl in zip(scored, y)
                  if t.measurements["source_report"][0]["argmax_category"]
                  == t.measurements["source_report"][1]["argmax_category"]]
    out["label_reversal"] = {
        "accuracy_order_permutation": per_order[0], "accuracy_order_reversed": per_order[1],
        "argmax_letter_counts_permutation": dict(letters[0]), "argmax_letter_counts_reversed": dict(letters[1]),
        "accuracy_by_position_of_true_category": {k: (stats.mean_of(v), len(v)) for k, v in sorted(pos_acc.items())},
        "n_scored_consistent": len(consistent),
        "accuracy_on_consistent": stats.mean_of(
            [float(t.measurements["source_report"][0]["argmax_category"] == lbl) for t, lbl in consistent]),
    }

    # 3. families and items
    fam = {}
    for f in sorted({t.family for t in scored}):
        idx = [i for i, t in enumerate(scored) if t.family == f]
        loo = [i for i, t in enumerate(scored) if t.family != f]
        fam[f] = {"n": len(idx), "self": stats.mean_of([self_c[i] for i in idx]),
                  "prompt": stats.mean_of([pr_c[i] for i in idx]), "majority_on_family": stats.mean_of([maj_c[i] for i in idx]),
                  "loo_self_minus_majority": pe([self_c[i] for i in loo], [maj_c[i] for i in loo], [g[i] for i in loo]),
                  "loo_self_minus_prompt": pe([self_c[i] for i in loo], [pr_c[i] for i in loo], [g[i] for i in loo])}
    out["per_family"] = fam
    items = sorted(set(g))
    loo_items = []
    for it in items:
        keep = [i for i, gg in enumerate(g) if gg != it]
        r = pe([self_c[i] for i in keep], [maj_c[i] for i in keep], [g[i] for i in keep])
        loo_items.append(r["estimate"])
    item_acc = {it: stats.mean_of([self_c[i] for i, gg in enumerate(g) if gg == it]) for it in items}
    out["items"] = {"loo_self_minus_majority_min": min(loo_items), "loo_self_minus_majority_max": max(loo_items),
                    "item_accuracy_quantiles": sorted(item_acc.values()),
                    "n_items_self_ge_prompt": sum(1 for it in items if stats.mean_of(
                        [self_c[i] for i, gg in enumerate(g) if gg == it]) >= stats.mean_of(
                        [pr_c[i] for i, gg in enumerate(g) if gg == it]))}

    # 4. exclusions and per-condition errors
    excl = Counter((t.condition, labels[t.id]) for t in live if labels[t.id] not in si.SCORED_LABELS)
    by_cond = defaultdict(lambda: Counter())
    for t, lbl, r in zip(scored, y, rep):
        by_cond[t.condition][f"{lbl}->{r}"] += 1
    out["exclusions"] = {"by_condition_label": {f"{c}|{lbl}": n for (c, lbl), n in excl.items()},
                         "status_counts": dict(Counter(t.status for t in trials)),
                         "excluded_reports": dict(Counter(si._argmax(si.report_probs(t)) for t in live
                                                          if labels[t.id] not in si.SCORED_LABELS and si.report_probs(t)))}
    out["per_condition_label_to_report"] = {c: dict(v) for c, v in by_cond.items()}
    out["per_condition_accuracy"] = {c: {"self": stats.mean_of([s for s, t in zip(self_c, scored) if t.condition == c]),
                                         "prompt": stats.mean_of([s for s, t in zip(pr_c, scored) if t.condition == c]),
                                         "n": sum(1 for t in scored if t.condition == c)}
                                     for c in sorted({t.condition for t in scored})}
    # sensitivity: drop the conflict-abstention trials (label 'insufficient_evidence' while document AND hint present)
    keep = [i for i, t in enumerate(scored) if not (t.condition == "doc_v_hint_conflict" and y[i] == "insufficient_evidence")]
    out["sensitivity_drop_conflict_abstentions"] = {
        "n": len(keep), "self_minus_majority": pe([self_c[i] for i in keep], [maj_c[i] for i in keep], [g[i] for i in keep]),
        "self_minus_prompt": pe([self_c[i] for i in keep], [pr_c[i] for i in keep], [g[i] for i in keep])}

    # 5. letter mass, free text
    masses = [r["mass"] for t in live for r in (t.measurements.get("source_report") or [])]
    ft = [(r["free_text_parse"], lbl) for t, lbl in zip(scored, y) for r in t.measurements["source_report"]]
    agree = [float(r["free_text_parse"].get("category") == r["argmax_category"]) for t in live
             for r in (t.measurements.get("source_report") or []) if r["free_text_parse"].get("status") == "ok"]
    out["letter_mass"] = {"min": min(masses), "mean": sum(masses) / len(masses),
                          "n_below_0.5": sum(m < 0.5 for m in masses),
                          "top_is_option_rate": stats.mean_of([float(r["top_is_option"]) for t in live
                                                               for r in t.measurements["source_report"]])}
    out["free_text"] = {"parse_ok_rate": stats.mean_of([float(p.get("status") == "ok") for p, _ in ft]),
                        "accuracy": stats.mean_of([float(p.get("category") == lbl) for p, lbl in ft if p.get("status") == "ok"]),
                        "agreement_with_letter": stats.mean_of(agree)}
    out["structural"] = {"scored_trials_where_transcript_differs_from_label": sum(1 for c in tr_c if c < 1.0),
                         "undetermined": sum(1 for t in live if labels[t.id] == "undetermined")}
    out["categories"] = list(SOURCE_CATEGORIES)
    return out


# ------------------------------------------------------------------------------------------ tests
needs_pilot = pytest.mark.skipif(not (PILOT.exists() and SUMMARY.exists()),
                                 reason="local pilot data not present (not committed)")


@needs_pilot
def test_stored_analysis_is_reproducible_from_stored_trials():
    from neuralscope.selflab.templates import source_influence as si

    trials, spec, stored = load(PILOT, SUMMARY)
    assert len(trials) == stored["n_trials"]
    res = si.analyze_battery(spec, trials, stored["run_id"])
    assert res.summary["statuses"] == stored["summary"]["statuses"]
    assert res.summary["primary"]["accuracy_letter"] == stored["summary"]["primary"]["accuracy_letter"]
    assert res.summary["observers"]["transcript_accuracy"] == stored["summary"]["observers"]["transcript_accuracy"]


@needs_pilot
def test_falsification_numbers_are_computable():
    trials, spec, _ = load(PILOT, SUMMARY)
    out = falsify(trials, spec, B=300)
    assert out["n_scored"] > 0 and 0.0 <= out["accuracy"]["self_letter"] <= 1.0
    assert out["structural"]["scored_trials_where_transcript_differs_from_label"] == 0  # see D-A1


if __name__ == "__main__":
    tp = Path(sys.argv[1]) if len(sys.argv) > 1 else PILOT
    sp = Path(sys.argv[2]) if len(sys.argv) > 2 else SUMMARY
    tr, sp_, _ = load(tp, sp)
    print(json.dumps(falsify(tr, sp_), indent=1, default=str))
