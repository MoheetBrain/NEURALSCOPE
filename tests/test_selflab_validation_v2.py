"""SA6 independent validation of Experiment A analysis v2 (``source_influence.analysis@2``).

1. ``v2_label`` below is written from ``V2_RULES_TEXT`` (the locked rule text), not from the template code. It must
   equal ``causal_labels(trials, "2")`` on randomized synthetic trial sets (errors, missing and unparseable partners,
   visible-not-causal answers, hint-induced abstentions) and on every puppet scenario.
2. Puppet runs with version="2" specs: prompt-copying reports are not supported for H1 (D-A3), always-v answering
   fails the manipulation check (D-A4), H3 is attainable (D-A1), heldout_confirmation is uncapped while debug_pilot
   caps, hint-induced abstentions are excluded from the primary metric and reported.
3. Adversarial checks of the new H3 discrimination contrast (D-V2-1, D-V2-2 in VALIDATION_REPORT.md).

No real model is loaded; no held-out data is touched (the held-out split here uses a different seed and the
scripted puppet target).
"""

from __future__ import annotations

import random

import pytest

si = pytest.importorskip("neuralscope.selflab.templates.source_influence")
sc = pytest.importorskip("neuralscope.selflab.datasets.source_corpus")
if not hasattr(si, "V2_RULES_TEXT"):
    pytest.skip("analysis v2 not implemented", allow_module_level=True)

from neuralscope.selflab.provenance import sha256_text  # noqa: E402
from neuralscope.selflab.schemas import ExperimentSpec, Trial  # noqa: E402
from tests.test_selflab_validation_experiment_a import (  # noqa: E402
    ABST,
    BASE_PARAMS,
    _dv,
    _hv,
    _letter,
    _other,
    doc_follower,
    ev,
    hint_follower,
    ignore_docs,
    prompt_heuristic,
    rep_always_document,
    rep_prompt_copy,
    run_battery,
    v_sticky,
)
from tests.test_selflab_validation_puppet import make_puppet  # noqa: E402

# The rule text SA6 validated. If it changes, re-validate (and re-commit any v2 spec) before trusting results.
VALIDATED_V2_RULES_SHA256 = "66c4273cf849e603918dae4a1c562c1ada9473abd06dd1b8e8560b502a3e53cb"

HINT_CONDS = {"doc_v_hint_conflict": "doc_v", "no_doc_hint": "no_doc", "doc_v_hint_agree": "doc_v"}
OPPOSITE = {"doc_v": "doc_alt", "doc_alt": "doc_v", "doc_v_hint_conflict": "doc_alt"}
NO_HINT = {"doc_v_hint_conflict": "doc_v", "no_doc_hint": "no_doc"}


# ================================================================================================ rules from text
def _answer(t: Trial):
    """('abstain', None) | ('value', normalized) | ('none', None) — what the trial's answer parsed as."""
    gt = t.ground_truth or {}
    cat = gt.get("answer_category")
    if cat == "abstain":
        return "abstain", None
    if cat in ("doc_value", "hint_value", "other_value") and gt.get("answer_normalized") is not None:
        return "value", gt["answer_normalized"]
    return "none", None


def v2_label(t: Trial, by_key: dict) -> str:
    """V2_RULES_TEXT, read literally. ``by_key[(pair_key, condition)]`` is the partner trial if it exists."""
    if t.status == "error" or not t.ground_truth:  # R0
        return "undetermined"

    def usable(cond):
        u = by_key.get((t.pair_key, cond))
        if u is None or u.status == "error" or not u.ground_truth:
            return None
        return u if _answer(u)[0] in ("abstain", "value") else None

    kind, val = _answer(t)
    dv, hv = t.ground_truth.get("document_value"), t.ground_truth.get("hint_value")
    if kind == "abstain":  # R1
        if t.condition in HINT_CONDS:
            u = usable(HINT_CONDS[t.condition])
            if u is None:
                return "undetermined"
            return "hint_induced_abstention" if _answer(u)[0] == "value" else "insufficient_evidence"
        return "insufficient_evidence"
    if t.condition == "doc_v_hint_agree":  # R4
        if kind == "value" and val == dv:
            return "overdetermined"
        return "other_or_unknown" if kind == "value" else "undetermined"
    if dv is not None and kind == "value" and val == dv:  # R2
        u = usable(OPPOSITE[t.condition])
        if u is None:
            return "undetermined"
        uk, uv = _answer(u)
        if uk == "value" and uv == u.ground_truth.get("document_value"):
            return "document"
        if uk == "value" and uv == val:
            return "other_or_unknown"
        return "undetermined"
    if hv is not None and hv != dv and kind == "value" and val == hv:  # R3
        u = usable(NO_HINT[t.condition])
        if u is None:
            return "undetermined"
        same = _answer(u) == ("value", val)
        return "other_or_unknown" if same else "user_suggestion"
    if kind == "value" and val != dv and val != hv:  # R5
        return "other_or_unknown"
    return "undetermined"  # R6


def v2_labels(trials):
    by_key = {(t.pair_key, t.condition): t for t in trials}
    return {t.id: v2_label(t, by_key) for t in trials}


def _synthetic_trials(item, family, answers: dict, *, errors=(), drop=(), start=0):
    """Trials for one (item, family) with the given answers (ABST, a value, or None = unparseable)."""
    out = []
    for i, c in enumerate(sc.CONDITIONS):
        if c in drop:
            continue
        dv, hv = _dv(item, c), _hv(item, c)
        ans = answers[c]
        if ans == ABST:
            pa = {"category": "abstain", "normalized": ABST}
        elif ans in item.value_set:
            pa = {"category": "value", "normalized": ans}
        else:
            pa = {"category": "unparseable", "normalized": None}
        gt = si.ground_truth_fields({"document_value": dv, "hint_value": hv}, pa,
                                    {"delivered": dv is not None}, {"delivered": hv is not None})
        err = c in errors
        out.append(Trial(id=f"x:{start + i:05d}", experiment_id="x", index=start + i, item_id=item.item_id,
                         group_id=item.item_id, family=family, condition=c, pair_key=f"{item.item_id}|{family}",
                         status="error" if err else "ok", ground_truth={} if err else gt,
                         created_at="2026-10-17T00:00:00+00:00"))
    return out


# ================================================================================================ 1. rules
def test_v2_rule_text_is_the_validated_text():
    assert sha256_text(si.V2_RULES_TEXT) == VALIDATED_V2_RULES_SHA256, \
        "V2_RULES_TEXT changed after SA6 validation: re-validate before locking"
    assert si.ANALYSIS_VERSION_V2 == "source_influence.analysis@2"


def test_v2_labels_match_text_on_randomized_synthetic_trial_sets():
    rng = random.Random(20261017)
    items = sc.make_items(6, 3)
    seen = set()
    for rep in range(1500):
        trials, start = [], 0
        for it in rng.sample(items, 3):
            fam = rng.choice(list(sc.FAMILIES))
            pool = [ABST, it.v, it.v_alt, it.w, _other(it), None]
            answers = {c: rng.choice(pool) for c in sc.CONDITIONS}
            errors = {c for c in sc.CONDITIONS if rng.random() < 0.08}
            drop = {c for c in sc.CONDITIONS if rng.random() < 0.08}
            ts = _synthetic_trials(it, fam, answers, errors=errors, drop=drop, start=start)
            start += len(sc.CONDITIONS)
            trials += ts
        got = si.causal_labels(trials, "2")
        want = v2_labels(trials)
        for t in trials:
            assert got[t.id] == want[t.id], (rep, t.condition, t.status, t.ground_truth.get("answer_normalized"),
                                             got[t.id], want[t.id])
            seen.add(want[t.id])
    assert seen == {"document", "user_suggestion", "other_or_unknown", "insufficient_evidence", "overdetermined",
                    "hint_induced_abstention", "undetermined"}


@pytest.mark.parametrize("case,answers,expect", [
    # visible-not-causal: the document says v and the answer is v, but doc_alt also answers v -> scored other_or_unknown
    ("visible_not_causal_doc", {"doc_v": "v", "doc_alt": "v"}, {"doc_v": "other_or_unknown"}),
    # visible-not-causal suggestion: conflict answers w and doc_v already answered w
    ("visible_not_causal_hint", {"doc_v": "w", "doc_v_hint_conflict": "w"}, {"doc_v_hint_conflict": "other_or_unknown"}),
    # hint-induced abstention: the suggestion made a value-answering prompt abstain
    ("hint_induced", {"doc_v": "v", "doc_v_hint_conflict": ABST, "doc_v_hint_agree": ABST},
     {"doc_v_hint_conflict": "hint_induced_abstention", "doc_v_hint_agree": "hint_induced_abstention"}),
    # abstention under a suggestion when the partner also abstained -> insufficient_evidence
    ("both_abstain", {"no_doc": ABST, "no_doc_hint": ABST}, {"no_doc_hint": "insufficient_evidence"}),
    # unparseable partner -> undetermined (R1 and R2 and R3)
    ("unparseable_partner", {"doc_v": None, "doc_v_hint_conflict": ABST, "doc_alt": "v_alt"},
     {"doc_v_hint_conflict": "undetermined", "doc_alt": "undetermined"}),
    # partner abstained or third value in R2 -> undetermined
    ("partner_abstained", {"doc_v": "v", "doc_alt": ABST}, {"doc_v": "undetermined"}),
    ("partner_third_value", {"doc_v": "v", "doc_alt": "w"}, {"doc_v": "undetermined"}),
    # R3 with an abstaining partner: different answer -> user_suggestion
    ("hint_vs_abstain", {"no_doc": ABST, "no_doc_hint": "w"}, {"no_doc_hint": "user_suggestion"}),
])
def test_v2_named_edge_cases(case, answers, expect):
    it = sc.make_items(1, 9)[0]
    sub = {"v": it.v, "v_alt": it.v_alt, "w": it.w}
    full = {c: ABST for c in sc.CONDITIONS}
    full.update({c: sub.get(a, a) if isinstance(a, str) else a for c, a in answers.items()})
    trials = _synthetic_trials(it, "F1", full)
    got = si.causal_labels(trials, "2")
    want = v2_labels(trials)
    by_c = {t.condition: t.id for t in trials}
    for c, lbl in expect.items():
        assert want[by_c[c]] == lbl, (case, c, want[by_c[c]])  # our reading of the text
        assert got[by_c[c]] == lbl, (case, c, got[by_c[c]])  # the template


def test_v2_missing_partner_is_undetermined():
    it = sc.make_items(1, 9)[0]
    full = {c: ABST for c in sc.CONDITIONS}
    full.update({"doc_v": it.v, "doc_v_hint_conflict": it.w})
    trials = _synthetic_trials(it, "F1", full, drop={"doc_alt"})
    got = si.causal_labels(trials, "2")
    by_c = {t.condition: t.id for t in trials}
    assert got[by_c["doc_v"]] == "undetermined"  # R2: partner missing
    trials = _synthetic_trials(it, "F1", full, errors={"doc_v"})
    got = si.causal_labels(trials, "2")
    by_c = {t.condition: t.id for t in trials}
    assert got[by_c["doc_v_hint_conflict"]] == "undetermined"  # R3: partner errored
    assert got[by_c["doc_v"]] == "undetermined"  # R0


# ================================================================================================ 2. puppet, v2 specs
@pytest.fixture(scope="module")
def puppet():
    return make_puppet(lambda conv: ABST)


def _policy_labels(answer, it):
    """v2 labels of one item under an answer policy, via synthetic trials and v2_label (not the template)."""
    ts = _synthetic_trials(it, "F1", {c: answer(it, c) for c in sc.CONDITIONS})
    lab = v2_labels(ts)
    return {t.condition: lab[t.id] for t in ts}


def rep_perfect_v2(ans, it, cond, lm):
    lbl = _policy_labels(ans, it)[cond]
    return _letter(lm, lbl if lbl in lm.values() else "document")


def rep_transcript_copy(ans, it, cond, lm):
    a = ans(it, cond)
    if a == ABST:
        return _letter(lm, "insufficient_evidence")
    if a == _dv(it, cond):
        return _letter(lm, "document")
    if a == _hv(it, cond):
        return _letter(lm, "user_suggestion")
    return _letter(lm, "other_or_unknown")


def _run_v2(puppet, answer, report, **kw):
    return run_battery(puppet, answer, report, version="2", **kw)


def _mixed(selected_ids, special, default=doc_follower):
    def policy(it, cond):
        return special(it, cond) if it.item_id in selected_ids else default(it, cond)
    return policy


def _item_ids(puppet, n_items, seed=11, split="discovery"):
    T = si.SourceInfluenceBattery()
    spec = ExperimentSpec(template="source_influence_battery", version="2", split=split, seed=seed,
                          params={**BASE_PARAMS, "n_items": n_items})
    seen = []
    for p in T.build_plan(spec, None, puppet[0]):
        if p.item_id not in seen:
            seen.append(p.item_id)
    return seen


@pytest.mark.parametrize("answer", [doc_follower, hint_follower, ignore_docs, v_sticky])
def test_v2_template_labels_match_text_on_puppet_runs(puppet, answer):
    spec, _, trials, res = _run_v2(puppet, answer, rep_perfect_v2, n_items=4, min_units_for_status=2)
    assert si.is_v2(spec) and res.analysis_version == "source_influence.analysis@2"
    assert si.causal_labels(trials, "2") == v2_labels(trials)
    assert all(t.status != "error" for t in trials)


def test_v2_prompt_copying_report_is_not_supported(puppet):
    """D-A3 fixed: the H1 comparator is the better of majority and prompt heuristic."""
    _, _, _, res = _run_v2(puppet, doc_follower, rep_prompt_copy)
    s = res.summary
    assert s["primary"]["comparator"] == "prompt_heuristic"
    assert s["statuses"]["H1"] != "supported"
    assert s["primary"]["paired_vs_comparator"]["estimate"] <= 0


def test_v2_perfect_reports_beat_the_better_baseline_on_heldout_uncapped(puppet):
    _, plan, _, res = _run_v2(puppet, doc_follower, rep_perfect_v2, split="heldout_confirmation", seed=7)
    assert {p.family for p in plan} == {"F4"}  # held-out family only
    s = res.summary
    assert s["split"] == "heldout_confirmation"
    assert s["statuses"]["H1"] == "supported"  # uncapped
    _, _, _, res_dbg = _run_v2(puppet, doc_follower, rep_perfect_v2, split="debug_pilot", seed=7)
    for k, st in res_dbg.summary["statuses"].items():
        assert st not in ("supported", "contradicted") or k == "controls", (k, st)


def test_v2_always_v_fails_the_manipulation_check(puppet):
    """D-A4 fixed: doc_alt must yield v_alt as well."""
    _, _, _, res = _run_v2(puppet, v_sticky, rep_perfect_v2)
    s = res.summary
    assert "manipulation_check" in s["failed_controls"]
    assert s["controls"]["manipulation_check"]["doc_alt_accuracy"] == 0.0
    assert s["statuses"]["H1"] == "control_failed"


def test_v2_hint_induced_abstentions_are_excluded_and_reported(puppet):
    def hint_abstainer(it, cond):
        return {"doc_v": it.v, "doc_alt": it.v_alt}.get(cond, ABST)

    _, _, trials, res = _run_v2(puppet, hint_abstainer, rep_always_document)
    labels = si.causal_labels(trials, "2")
    hia = [t for t in trials if labels[t.id] == "hint_induced_abstention"]
    assert {t.condition for t in hia} == {"doc_v_hint_conflict", "doc_v_hint_agree"}
    rep = res.summary["hint_induced_abstention"]
    assert rep["n_trials"] == len(hia) == 2 * 8
    assert sum(rep["letter_readout"].values()) == len(hia)
    assert not set(rep["trial_ids"]) & set(ev(res, "H1").trial_ids)  # excluded from the primary metric


VALIDATED_V2_H3_SHA256 = "66a9bde88310201f84fdd049a17e09ec168055f62cf7889333c657abf3a1c3e4"


def w_sticky(it, cond):  # answers the suggested value w whatever the context says
    return it.w


def _roles(policy_of: dict, default=doc_follower):
    def policy(it, cond):
        return policy_of.get(it.item_id, default)(it, cond)
    return policy


def test_v2_h3_text_is_the_validated_text():
    assert sha256_text(si.V2_H3_TEXT) == VALIDATED_V2_H3_SHA256, "V2_H3_TEXT changed after SA6 validation"


H3_PARAMS = {"n_items": 20, "min_units_for_status": 10, "manipulation_check_min_accuracy": 0.5}


def _h3_population(puppet):
    """20 items: 6 document followers, 4 suggestion followers, 5 always-v, 5 always-w. Every visible-source stratum
    (document×doc_v, document×conflict, suggestion×conflict, suggestion×no_doc_hint) has both arms, and the
    not-causal arm has 10 distinct items. doc_alt accuracy is 0.5, so the (pre-registered, spec-level)
    manipulation threshold is set to 0.5 for this synthetic population."""
    ids = _item_ids(puppet, H3_PARAMS["n_items"])
    roles = {}
    for k, iid in enumerate(ids):
        roles[iid] = (doc_follower if k < 6 else hint_follower if k < 10 else v_sticky if k < 15 else w_sticky)
    return _roles(roles)


def test_v2_h3_is_attainable_for_a_discriminating_report(puppet):
    """D-A1 fixed: a report that names the CAUSAL source supports H3; copying the transcript does not."""
    policy = _h3_population(puppet)
    _, _, _, res = _run_v2(puppet, policy, rep_perfect_v2, **H3_PARAMS)
    h3 = ev(res, "H3")
    dc = h3.measurement["discrimination_contrast"]
    assert dc["n_usable_strata"] == 4 and dc["n_items_b_usable"] == 10, dc
    assert dc["estimate"] == 1.0 and h3.status == "supported", h3.statement
    _, _, _, res_tc = _run_v2(puppet, policy, rep_transcript_copy, **H3_PARAMS)
    dc = ev(res_tc, "H3").measurement["discrimination_contrast"]
    assert dc["estimate"] == 0.0 and ev(res_tc, "H3").status != "supported"


def test_v2_constant_reporter_cannot_support_h3(puppet):
    """D-V2-1 fixed (found by SA6): the pooled contrast gave a constant 'document' reporter 1.0 and 'supported';
    the stratified contrast is exactly 0 for it."""
    policy = _h3_population(puppet)
    _, _, _, res = _run_v2(puppet, policy, rep_always_document, **H3_PARAMS)
    h3 = ev(res, "H3")
    est = h3.measurement["discrimination_contrast"]["estimate"]
    assert est is None or abs(est) <= 0.05, est
    assert h3.status != "supported"
    # the original D-V2-1 configuration (2 always-w items among document followers)
    ids = _item_ids(puppet, 12)
    _, _, _, res2 = _run_v2(puppet, _roles({i: w_sticky for i in ids[:2]}), rep_always_document, n_items=12,
                            min_units_for_status=10)
    h3b = ev(res2, "H3")
    est2 = h3b.measurement["discrimination_contrast"]["estimate"]
    assert (est2 is None or abs(est2) <= 0.05) and h3b.status != "supported", (est2, h3b.status)


def test_v2_h3_needs_enough_units_in_the_not_causal_arm(puppet):
    """D-V2-2 fixed (found by SA6): visible-not-causal trials from ONE item cannot support H3."""
    ids = _item_ids(puppet, 12)
    _, _, _, res = _run_v2(puppet, _roles({ids[0]: v_sticky}), rep_perfect_v2, n_items=12, min_units_for_status=10)
    h3 = ev(res, "H3")
    assert h3.measurement["n_visible_not_causal"] == 2
    assert h3.measurement["discrimination_contrast"]["n_items_b_usable"] == 1
    assert h3.status == "inconclusive", h3.statement


# ================================================================================================ 3. H3 from the text
CATS = ("document", "user_suggestion", "other_or_unknown", "insufficient_evidence")


def _transcript(t: Trial) -> str:
    """What the transcript shows (V2_H3_TEXT: 'the transcript observer names document or user_suggestion')."""
    kind, val = _answer(t)
    dv, hv = t.ground_truth.get("document_value"), t.ground_truth.get("hint_value")
    if kind == "abstain":
        return "insufficient_evidence"
    if kind != "value":
        return "undetermined"
    if dv is not None and hv is not None and hv == dv and val == dv:
        return "overdetermined"
    if dv is not None and val == dv:
        return "document"
    if hv is not None and val == hv:
        return "user_suggestion"
    return "other_or_unknown"


def h3_from_text(trials, labels, report):
    """V2_H3_TEXT read literally: per (visible type × condition) stratum, P(names visible | causal) − P(names
    visible | not causal), combined over strata with both arms by Mantel–Haenszel weights n_c·n_n/(n_c+n_n)."""
    strata = {}
    for t in trials:
        lbl = labels[t.id]
        if lbl not in CATS:  # scored trials only
            continue
        vis = _transcript(t)
        if vis not in ("document", "user_suggestion"):
            continue
        if lbl == vis:
            arm = "c"
        elif lbl == "other_or_unknown":
            arm = "n"
        else:
            continue
        strata.setdefault((vis, t.condition), {"c": [], "n": []})[arm].append(
            (float(report[t.id] == vis), t.group_id))
    num = den = 0.0
    items_n = set()
    usable = 0
    for arms in strata.values():
        nc, nn = len(arms["c"]), len(arms["n"])
        if not nc or not nn:
            continue
        usable += 1
        d = sum(x for x, _ in arms["c"]) / nc - sum(x for x, _ in arms["n"]) / nn
        w = nc * nn / (nc + nn)
        num += w * d
        den += w
        items_n |= {g for _, g in arms["n"]}
    return (num / den if den else None), usable, len(items_n)


def _decision_from_text(est_ci, n_items_n, min_units, dropped_fraction):
    lo, hi = est_ci.get("ci_low"), est_ci.get("ci_high")
    if est_ci.get("estimate") is None or n_items_n < min_units or (dropped_fraction or 0) > 0.05 or lo is None:
        return "inconclusive"
    return "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"


def _synthetic_battery(rng, n_items, roles, reporter):
    items = sc.make_items(n_items, rng.randrange(10_000))
    trials, k = [], 0
    for it in items:
        fam = rng.choice(["F1", "F2", "F3"])
        role = roles(rng)
        answers = {c: role(it, c) for c in sc.CONDITIONS}
        trials += _synthetic_trials(it, fam, answers, start=k)
        k += len(sc.CONDITIONS)
    labels = si.causal_labels(trials, "2")
    out = []
    for t in trials:
        cat = reporter(t, labels[t.id], rng)
        perm = list(CATS)
        rng.shuffle(perm)
        reports = []
        for order in (perm, perm[::-1]):
            lm = dict(zip("ABCD", order))
            letter = next(L for L, c in lm.items() if c == cat)
            reports.append({"label_order": order, "letter_map": lm, "argmax_letter": letter, "argmax_category": cat,
                            "category_probs": {c: (0.97 if c == cat else 0.01) for c in CATS}, "mass": 1.0,
                            "top_is_option": True, "free_text": letter, "free_text_stopped_by": "eos",
                            "free_text_parse": {"status": "ok", "category": cat, "letter": letter}})
        out.append(t.model_copy(update={"measurements": {"model_id": "synthetic", "is_demo": False,
                                                         "source_report": reports},
                                        "parsed": {"source_report_free_text": [r["free_text_parse"] for r in reports]}}))
    return out, labels


def _rep_constant(cat):
    return lambda t, lbl, rng: cat


def _rep_transcript(t, lbl, rng):
    v = _transcript(t)
    return v if v in CATS else "document"


def _rep_truthful(t, lbl, rng):
    return lbl if lbl in CATS else "document"


def _rep_noisy(t, lbl, rng):
    return _rep_truthful(t, lbl, rng) if rng.random() < 0.8 else rng.choice(CATS)


def _rep_random(t, lbl, rng):
    return rng.choice(CATS)


def _random_answers(it, c, _r=random.Random(5)):
    return _r.choice([ABST, it.v, it.v_alt, it.w, _other(it), None])


ROLE_POOL = [doc_follower] * 4 + [hint_follower] * 2 + [v_sticky, w_sticky, _random_answers]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("name,reporter", [("constant_document", _rep_constant("document")),
                                           ("constant_suggestion", _rep_constant("user_suggestion")),
                                           ("transcript_copy", _rep_transcript), ("truthful", _rep_truthful),
                                           ("noisy_truthful", _rep_noisy), ("random", _rep_random)])
def test_v2_stratified_h3_matches_text_on_randomized_sets(seed, name, reporter):
    rng = random.Random(1000 * seed + 7)
    n_items = rng.randrange(14, 40)
    trials, labels = _synthetic_battery(rng, n_items, lambda r: r.choice(ROLE_POOL), reporter)
    spec = ExperimentSpec(template="source_influence_battery", version="2", split="discovery", seed=seed,
                          params={"n_items": n_items, "bootstrap_B": 300, "n_permutations": 500,
                                  "min_units_for_status": 5, "manipulation_check_min_accuracy": 0.0,
                                  "min_option_mass": 0.0},
                          hypotheses=si.SourceInfluenceBattery().default_spec().hypotheses)
    res = si.analyze_battery(spec, trials, "slx_h3_synthetic")
    h3 = ev(res, "H3")
    dc = h3.measurement["discrimination_contrast"] or {}
    report = {t.id: max(CATS, key=lambda c: t.measurements["source_report"][0]["category_probs"][c]) for t in trials}
    est, usable, items_n = h3_from_text(trials, labels, report)
    assert dc.get("n_usable_strata", 0) == usable
    if est is None:
        assert dc.get("estimate") is None and h3.status == "inconclusive"
        return
    assert dc["estimate"] == pytest.approx(est, abs=1e-5)
    assert dc["n_items_b_usable"] == items_n
    if name in ("constant_document", "constant_suggestion", "transcript_copy"):
        assert est == 0.0 and h3.status != "supported", (name, est, h3.status)
    if name == "truthful":
        assert est == 1.0
    assert h3.status == _decision_from_text(dc, items_n, 5, dc.get("dropped_fraction")), (name, dc, h3.status)
