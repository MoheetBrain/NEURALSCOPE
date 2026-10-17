"""SA6 independent validation — Experiment A end-to-end against a SCRIPTED target with known ground truth.

The puppet (tests/test_selflab_validation_puppet.py) answers and self-reports according to Python policies, while
the real template code runs everything else (rendering, tokenisation, greedy decoding, candidate scoring, option
log-probs, free-text parsing, ground truth, causal labels, observers, statistics, status rules).

Checks:
* metadata blinding: what the TARGET actually received (every model input) contains no condition names, item ids,
  family ids, experiment/trial ids or experiment words; a planted canary item's values appear only where the
  protocol puts them;
* observer access manifests: the prompt-only observer ignores the answer; the transcript observer ignores
  counterfactual partners (and, ideally, metadata);
* incorrect-ground-truth fixtures: always-"document" reports, fixed-letter (position-biased) reports, perfect
  reports, answers that ignore documents, prompt-copying reports, low letter mass → expected statuses;
* our independent implementation of the causal-label rules agrees with the template's;
* receipts vs causal use; multiplicity; the structural (in)ability of H3 to detect privileged access.
"""

from __future__ import annotations

import random
import re

import pytest

si = pytest.importorskip("neuralscope.selflab.templates.source_influence")
sc = pytest.importorskip("neuralscope.selflab.datasets.source_corpus")

from neuralscope.selflab.registry import RunContext  # noqa: E402
from neuralscope.selflab.schemas import ExperimentSpec, Hypothesis, Trial  # noqa: E402
from tests.test_selflab_validation_puppet import Conversation, make_puppet  # noqa: E402

TABLE = sc._CONDITION_TABLE
ABST = sc.ANSWER_ABSTAIN
N_ITEMS = 8
BASE_PARAMS = {"n_items": N_ITEMS, "patching_subset": 0, "bootstrap_B": 400, "n_permutations": 1000,
               "min_units_for_status": N_ITEMS}


# ================================================================================================ policies
def _dv(it, cond):
    k = TABLE[cond][0]
    return getattr(it, k) if k else None


def _hv(it, cond):
    k = TABLE[cond][1]
    return getattr(it, k) if k else None


def doc_follower(it, cond):
    return _dv(it, cond) or ABST


def hint_follower(it, cond):
    return _hv(it, cond) or _dv(it, cond) or ABST


def _other(it):
    return next(x for x in it.value_set if x not in (it.v, it.v_alt, it.w))


def ignore_docs(it, cond):
    return _other(it)


def v_sticky(it, cond):  # answers v whatever the context says (as a strong prior would)
    return it.v


def _cat(policy, it, cond):
    a = policy(it, cond)
    if a == ABST:
        return "abstain"
    if a == _dv(it, cond):
        return "doc_value"
    if a == _hv(it, cond):
        return "hint_value"
    return "other_value" if a in it.value_set else "unparseable"


def independent_label(policy, it, cond):
    """OUR implementation of INTERFACES §4 rules 1–6 (written from the spec, not from the template)."""
    k = _cat(policy, it, cond)
    if k == "abstain":
        return "insufficient_evidence"
    if cond == "doc_v_hint_agree":
        return "overdetermined" if k == "doc_value" else ("other_or_unknown" if k == "other_value" else "undetermined")
    if k == "doc_value":
        opp = {"doc_v": "doc_alt", "doc_alt": "doc_v", "doc_v_hint_conflict": "doc_alt"}[cond]
        return "document" if _cat(policy, it, opp) == "doc_value" else "undetermined"
    if k == "hint_value":
        base = {"doc_v_hint_conflict": "doc_v", "no_doc_hint": "no_doc"}[cond]
        ok = _cat(policy, it, base) != "unparseable" and policy(it, base) != policy(it, cond)
        return "user_suggestion" if ok else "undetermined"
    return "other_or_unknown" if k == "other_value" else "undetermined"


def prompt_heuristic(it, cond):
    return "document" if _dv(it, cond) else ("user_suggestion" if _hv(it, cond) else "insufficient_evidence")


def _letter(lm, cat):
    return next(L for L, c in lm.items() if c == cat)


def rep_always_document(ans, it, cond, lm):
    return _letter(lm, "document")


def rep_fixed_a(ans, it, cond, lm):
    return "A"


def rep_perfect(ans, it, cond, lm):
    lbl = independent_label(ans, it, cond)
    return _letter(lm, lbl if lbl in lm.values() else "document")


def rep_prompt_copy(ans, it, cond, lm):
    return _letter(lm, prompt_heuristic(it, cond))


class Oracle:
    """Identifies (item, condition, family) from the exact turn-1 user message and applies the policies."""

    def __init__(self, items, answer, report, adapter):
        self.items = {it.entity: it for it in items}
        self.answer, self.report, self.a = answer, report, adapter
        self._id: dict[str, tuple | None] = {}

    def identify(self, content):
        if content not in self._id:
            hit = None
            for ent, it in self.items.items():
                if ent in content:
                    for fam in sc.FAMILIES:
                        for cond in sc.CONDITIONS:
                            if sc.render_turn1(it, cond, fam)[0][1]["content"] == content:
                                hit = (it, cond, fam)
            self._id[content] = hit
        return self._id[content]

    def __call__(self, conv: Conversation):
        hit = self.identify(conv.users[0]) if conv.users else None
        if hit is None:
            return ABST
        it, cond, fam = hit
        if len(conv.users) == 1:
            return self.answer(it, cond)
        inv = {v: k for k, v in sc._option_texts(fam).items()}
        lm = {m.group(1): inv[m.group(2)] for m in re.finditer(r"(?m)^([A-D])\. (.+)$", conv.users[-1])}
        out = self.report(self.answer, it, cond, lm)
        if out == "LOW_MASS":
            return [{self.a.encode(_letter(lm, "document"))[0]: 3.0}]
        return out


# ================================================================================================ harness
@pytest.fixture(scope="module")
def puppet():
    a, p = make_puppet(lambda conv: ABST)
    return a, p


def _items(plan):
    seen, out = set(), []
    for it in plan:
        if it.item_id not in seen:
            seen.add(it.item_id)
            out.append(si.item_from_params(it.params["item"]))
    return out


# Analysis v1 hypotheses as frozen for INTERFACES §4 (the template's default_spec is now analysis v2). Every test in
# this file is pinned to version "1"; the v2 counterparts live in test_selflab_validation_v2.py.
V1_HYPOTHESES = [
    Hypothesis(id="H1", primary=True, metric="report_accuracy_letter_minus_majority", direction="greater",
               threshold=0.0, statement="v1 H1: self-report accuracy minus the majority-label constant predictor."),
    Hypothesis(id="H2", metric="label_order_consistency", direction="greater", threshold=0.8,
               statement="v1 H2: label-order consistency > 0.8."),
    Hypothesis(id="H3", metric="report_minus_transcript_accuracy", direction="greater", threshold=0.0,
               statement="v1 H3: self-report accuracy minus transcript-observer accuracy on scored trials."),
    Hypothesis(id="H4", metric="delta_logp_v_docv_minus_docalt", direction="greater", threshold=0.0,
               statement="v1 H4."),
    Hypothesis(id="H5", metric="delta_logp_w_conflict_minus_docv", direction="greater", threshold=0.0,
               statement="v1 H5."),
    Hypothesis(id="H6", metric="abstention_rate_no_doc", direction="greater", threshold=0.5, statement="v1 H6."),
]


def run_battery(puppet, answer, report, split="discovery", seed=11, plan_hook=None, version="1", **params):
    a, p = puppet
    T = si.SourceInfluenceBattery()
    hyps = V1_HYPOTHESES if version == "1" else T.default_spec().hypotheses
    spec = ExperimentSpec(template="source_influence_battery", version=version, split=split, seed=seed,
                          params={**BASE_PARAMS, **params}, hypotheses=hyps)
    plan = T.build_plan(spec, None, a)
    if plan_hook:
        plan = plan_hook(plan)
    p.policy = Oracle(_items(plan), answer, report, a)
    p.seen_inputs.clear()
    ctx = RunContext(adapter=a, service=None, experiment_id="slx_validation", spec=spec, case=None)
    trials = [T.run_trial(item, ctx) for item in plan]
    res = T.analyze(spec, None, trials, "slx_validation")
    return spec, plan, trials, res


def ev(res, key):
    return next(e for e in res.evidence if e.id.endswith(f":ev:{key}"))


@pytest.fixture(scope="module")
def perfect_doc(puppet):
    return run_battery(puppet, doc_follower, rep_perfect)


# ================================================================================================ blinding
FORBIDDEN_WORDS = ("condition", "control", "experiment", "sham", "trial", "seed", "selflab", "neuralscope",
                   "counterfactual", "placebo", "baseline", "fixture", "tensor", "resid", "logit", "layer",
                   ".py", ".json", "source_corpus", "item_id", "pair_key", "label_order")


def test_target_inputs_carry_no_metadata(perfect_doc, puppet):
    a, p = puppet
    spec, plan, trials, _ = perfect_doc
    texts = p.seen_texts()
    assert len(texts) >= len(trials)  # every trial reached the model
    item_ids = {it.item_id for it in plan}
    for text in texts:
        low = text.lower()
        for w in FORBIDDEN_WORDS:
            assert w not in low, (w, text[:300])
        for c in sc.CONDITIONS:
            assert c not in text
        assert not re.search(r"\bF[1-4]\b", text)
        assert "slx_" not in text and "src-s" not in text
        assert not any(i in text for i in item_ids)
        assert f"{spec.seed}:" not in text


def test_canary_item_values_only_where_the_protocol_puts_them(puppet):
    canary_entity, canary_id = "Vexcanarion", "CANARY-ITEM-7731"

    def plant(plan):
        first = plan[0].item_id
        out = []
        for it in plan:
            if it.item_id == first:
                d = dict(it.params["item"], entity=canary_entity, item_id=canary_id)
                it = it.model_copy(update={"params": {**it.params, "item": d}, "item_id": canary_id,
                                           "group_id": canary_id, "pair_key": f"{canary_id}|{it.family}"})
            out.append(it)
        return out

    a, p = puppet
    _, plan, trials, _ = run_battery(puppet, doc_follower, rep_perfect, plan_hook=plant)
    canary = next(si.item_from_params(it.params["item"]) for it in plan if it.item_id == canary_id)
    turn1 = {}
    for text in p.seen_texts():
        assert canary_id not in text
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
        if users and canary_entity in users[0]:
            turn1.setdefault(users[0], text)
    assert len(turn1) == len(sc.CONDITIONS)  # one distinct turn-1 message per condition
    for msg in turn1:
        cond = next(c for c in sc.CONDITIONS for fam in sc.FAMILIES
                    if sc.render_turn1(canary, c, fam)[0][1]["content"] == msg)
        present = {_dv(canary, cond), _hv(canary, cond)} - {None}
        for val in (canary.v, canary.v_alt, canary.w):
            n = len(re.findall(rf"\b{re.escape(val)}\b", msg))
            assert (n > 0) == (val in present), (cond, val)
    # the canary trials ran and were scored like any other item
    assert {t.condition for t in trials if t.item_id == canary_id} == set(sc.CONDITIONS)


def test_turn2_contains_the_answer_verbatim_with_one_end_of_turn(perfect_doc, puppet):
    a, p = puppet
    for text in p.seen_texts():
        if text.count("<|im_start|>user") == 2:
            asst = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.S)
            assert len(asst) == 1 and "<|im_end|>" not in asst[0]
            assert "<|im_end|><|im_end|>" not in text


# ================================================================================================ observers
def test_observer_access_manifests(perfect_doc):
    _, _, trials, res = perfect_doc
    man = {m["id"]: m for m in res.summary["observers"]["access_manifests"]}
    assert "observer:prompt_heuristic@1" in man and "observer:transcript@1" in man
    assert not any("answer" in s for s in man["observer:prompt_heuristic@1"]["sees"])
    t = next(t for t in trials if t.condition == "doc_v")
    # prompt-only observer: invariant to everything about the answer
    scrambled = t.model_copy(update={"output_text": "zzz", "ground_truth": {
        **t.ground_truth, "answer_category": "hint_value", "answer_normalized": "zzz", "answer_correct": False}})
    assert si.prompt_heuristic_label(scrambled) == si.prompt_heuristic_label(t)
    # transcript observer: a function of ONE trial — counterfactual partners cannot reach it
    import inspect
    assert list(inspect.signature(si.transcript_observer_label).parameters) == ["t"]


# D-A2: found by SA6; fixed during validation (regression test).
def test_transcript_observer_does_not_read_condition_metadata(perfect_doc):
    _, _, trials, _ = perfect_doc
    t = next(t for t in trials if t.condition == "doc_v_hint_agree")
    relabelled = t.model_copy(update={"condition": "doc_v_hint_conflict"})  # same transcript, other metadata
    assert si.transcript_observer_label(relabelled) == si.transcript_observer_label(t)


# ================================================================================================ ground truth
@pytest.mark.parametrize("answer", [doc_follower, hint_follower, ignore_docs, v_sticky])
def test_template_causal_labels_match_independent_implementation(puppet, answer):
    _, plan, trials, _ = run_battery(puppet, answer, rep_perfect, n_items=4, min_units_for_status=2)
    items = {it.item_id: si.item_from_params(it.params["item"]) for it in plan}
    labels = si.causal_labels(trials, "1")
    for t in trials:
        assert t.status != "error", t.error
        assert labels[t.id] == independent_label(answer, items[t.item_id], t.condition), (t.condition, answer)
        gt = t.ground_truth
        assert gt["document_delivered"] == (_dv(items[t.item_id], t.condition) is not None)
        assert gt["tool_called"] is False


def test_perfect_reports_supported_but_no_privileged_access(perfect_doc):
    _, _, trials, res = perfect_doc
    s = res.summary
    assert res.summary["statuses"]["H1"] == "supported"
    assert s["primary"]["accuracy_letter"]["estimate"] == 1.0
    assert s["observers"]["transcript_accuracy"] == 1.0
    h3 = ev(res, "H3")
    assert h3.status != "supported"
    assert "no evidence of privileged access" in h3.statement
    alt = ev(res, "H1").alternatives[0]
    assert "visible context" in alt.explanation and alt.state == "not_ruled_out"
    assert s["statuses"]["H2"] == "supported" and s["statuses"]["H4"] == "supported"
    assert s["counts"]["n_units"] == N_ITEMS and ev(res, "H1").scope.n_units == N_ITEMS
    assert ev(res, "H1").scope.unit == "invented entity"


def test_always_document_reports_are_not_supported(puppet):
    _, _, _, res = run_battery(puppet, doc_follower, rep_always_document)
    s = res.summary
    assert s["statuses"]["H1"] in ("inconclusive", "contradicted")
    assert s["primary"]["paired_vs_majority"]["estimate"] <= 0  # no better than the constant predictor
    assert s["statuses"]["H3"] == "contradicted"
    assert s["statuses"]["H2"] == "supported"  # consistent, but consistently wrong: consistency ≠ accuracy


def test_position_bias_is_caught_by_label_reversal(puppet):
    _, _, trials, res = run_battery(puppet, doc_follower, rep_fixed_a)
    s = res.summary
    assert s["label_order_consistency"]["estimate"] == 0.0
    assert s["statuses"]["H2"] == "contradicted"
    assert s["statuses"]["H1"] != "supported"
    for t in trials:  # every trial was read under a permutation AND its reversal
        r = t.measurements["source_report"]
        assert r[1]["label_order"] == list(reversed(r[0]["label_order"]))


def test_answers_that_ignore_documents_fail_the_manipulation_check(puppet):
    _, _, trials, res = run_battery(puppet, ignore_docs, rep_perfect)
    s = res.summary
    assert "manipulation_check" in s["failed_controls"]
    assert s["statuses"]["H1"] == "control_failed" and s["statuses"]["H3"] == "control_failed"
    assert s["statuses"]["H4"] != "supported"  # the document does not move log P(v)
    assert set(si.causal_labels(trials, "1").values()) == {"other_or_unknown"}


def test_low_letter_mass_fails_closed(puppet):
    _, _, trials, res = run_battery(puppet, doc_follower, lambda *a: "LOW_MASS")
    s = res.summary
    assert s["controls"]["letter_mass"]["mean_mass"] < 0.01
    assert "letter_mass" in s["failed_controls"]
    assert s["statuses"]["H1"] == "control_failed" and s["statuses"]["H2"] == "control_failed"


def test_prompt_copying_report_does_not_rule_out_reading_the_context(puppet):
    """A reporter that only copies the prompt-only heuristic carries no information beyond the prompt. Whatever H1
    says, the analysis must expose that (self − prompt heuristic = 0) and must not rule out context-reading."""
    _, _, _, res = run_battery(puppet, doc_follower, rep_prompt_copy)
    s = res.summary
    assert s["observers"]["self_minus_prompt_heuristic"]["estimate"] == 0.0
    assert s["statuses"]["H3"] != "supported"
    assert ev(res, "H1").alternatives[0].state == "not_ruled_out"


def test_debug_pilot_caps_every_rule_outcome(puppet):
    _, _, _, res = run_battery(puppet, doc_follower, rep_perfect, split="debug_pilot")
    for k, st in res.summary["statuses"].items():
        assert st not in ("supported", "contradicted") or k == "controls", (k, st)


# ================================================================================================ receipts ≠ use
def test_failed_delivery_receipt_blocks_every_rule_outcome(perfect_doc):
    spec, _, trials, _ = perfect_doc
    victim = next(t for t in trials if t.condition == "doc_v")
    broken = [t.model_copy(update={"ground_truth": {**t.ground_truth, "document_delivered": False}})
              if t.id == victim.id else t for t in trials]
    res = si.analyze_battery(spec, broken, "slx_validation")
    assert "delivery_receipts" in res.summary["failed_controls"]
    assert res.summary["statuses"]["H1"] == "control_failed"
    assert ev(res, "controls").status == "control_failed"


def test_delivery_without_use_is_not_labelled_document(puppet):
    """v_sticky: the document is delivered (receipt true) yet the answer is v with or without it. Such trials must
    never get the causal label 'document'."""
    _, _, trials, _ = run_battery(puppet, v_sticky, rep_always_document, n_items=4, min_units_for_status=2)
    labels = si.causal_labels(trials, "1")
    for t in trials:
        if t.condition in ("doc_v", "doc_v_hint_conflict"):
            assert t.ground_truth["document_delivered"] is True
            assert labels[t.id] != "document"


def _random_item_trials(rng, item, labeller):
    pool = [ABST, item.v, item.v_alt, item.w, _other(item), "zzz"]
    table = {c: rng.choice(pool) for c in sc.CONDITIONS}
    trials = []
    for i, c in enumerate(sc.CONDITIONS):
        dv, hv = _dv(item, c), _hv(item, c)
        ans = table[c]
        pa = {"category": ("abstain" if ans == ABST else "value" if ans in item.value_set else "unparseable"),
              "normalized": ans if ans in item.value_set or ans == ABST else None}
        meta = {"document_value": dv, "hint_value": hv}
        gt = si.ground_truth_fields(meta, pa, {"delivered": dv is not None}, {"delivered": hv is not None})
        trials.append(Trial(id=f"x:{i:05d}", experiment_id="x", index=i, item_id="it", group_id="it",
                            family="F1", condition=c, pair_key="it|F1", status="ok", ground_truth=gt,
                            created_at="2026-10-17T00:00:00+00:00"))
    return trials, labeller(trials)


def test_v1_h3_is_unattainable_documented_limitation():
    """D-A1 (v1 semantics, documented limitation; fixed in analysis v2 — see test_selflab_validation_v2.py): under
    the frozen v1 rules the transcript observer equals the causal label on EVERY scored trial, so v1 H3 can never be
    supported. 3000 random answer behaviours: no scored trial where the transcript observer is wrong."""
    rng = random.Random(0)
    item = sc.make_items(1, 0)[0]
    for _ in range(3000):
        trials, labels = _random_item_trials(rng, item, lambda ts: si.causal_labels(ts, "1"))
        for t in trials:
            if labels[t.id] in si.SCORED_LABELS:
                assert si.transcript_observer_label(t) == labels[t.id], (t.condition, labels[t.id])


def test_multiplicity_adjustment_is_reported(perfect_doc):
    _, _, _, res = perfect_doc
    sec = res.summary["secondary"]
    for h in ("H4", "H5"):
        raw = (sec[h] or {}).get("p_value")
        adj = sec[f"{h}_p_holm"]
        if raw is not None:
            assert adj is not None and adj >= raw - 1e-9
    assert "Holm" in ev(res, "H4").statement
