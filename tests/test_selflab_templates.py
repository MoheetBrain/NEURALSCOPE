"""SelfLab experiment templates (SA3): plan determinism, leakage, demo execution, analysis on synthetic trials."""

from __future__ import annotations

import re

import pytest

from neuralscope.selflab import compose
from neuralscope.selflab.datasets import source_corpus as sc
from neuralscope.selflab.provenance import sha256_json
from neuralscope.selflab.registry import RunContext, get_template, list_templates
from neuralscope.selflab.schemas import SOURCE_CATEGORIES, Case, ClaimSpan, Message, Trial
from neuralscope.selflab.templates import source_influence as si

BAT = "source_influence_battery"


def _plan_hash(plan) -> str:
    return sha256_json([p.model_dump() for p in plan])


def _spec(**params):
    t = get_template(BAT)
    spec = t.default_spec(None)
    spec.params.update(params)
    return t, spec


# ------------------------------------------------------------------------------------------ registry / spec
def test_templates_registered_with_contract():
    ids = {t.id for t in list_templates()}
    assert {"source_influence_battery", "source_influence_case"} <= ids
    for t in list_templates():
        assert t.feature_status in ("implemented", "research_prototype", "fixture", "api_only_view", "unsupported",
                                    "roadmap")
        assert hasattr(t, "params_model")


def test_default_spec_is_preregistered():
    t = get_template(BAT)
    spec = t.default_spec(None)
    t.params_model.model_validate(spec.params)
    prim = [h for h in spec.hypotheses if h.primary]
    assert len(prim) == 1 and prim[0].id == "H1" and spec.primary_metric == prim[0].metric
    assert spec.version == "2" and spec.primary_metric == "report_accuracy_letter_minus_best_baseline"
    assert {h.id for h in spec.hypotheses} == {"H1", "H2", "H3", "H4", "H5", "H6", "H8"}
    assert spec.controls and spec.budget.max_trials >= 120
    assert spec.params["answer_max_new_tokens"] <= 16 and spec.params["report_max_new_tokens"] <= 24
    with pytest.raises(Exception):
        t.params_model.model_validate({"conditions": ["doc_v", "bogus"]})
    with pytest.raises(Exception):
        t.params_model.model_validate({"answer_max_new_tokens": 17})


# ------------------------------------------------------------------------------------------ plan
def test_plan_deterministic_and_seed_sensitive(demo_adapter):
    t, spec = _spec(n_items=4)
    h1 = _plan_hash(t.build_plan(spec, None, demo_adapter))
    h2 = _plan_hash(t.build_plan(spec, None, demo_adapter))
    assert h1 == h2
    spec2 = spec.model_copy(update={"seed": 1})
    assert _plan_hash(t.build_plan(spec2, None, demo_adapter)) != h1
    plan = t.build_plan(spec, None, None)  # no tokenizer → no alignment redraws, still deterministic
    assert _plan_hash(plan) == _plan_hash(t.build_plan(spec, None, None))


def test_plan_structure_and_split_policy(demo_adapter):
    t, spec = _spec(n_items=5, patching_subset=2)
    plan = t.build_plan(spec, None, demo_adapter)
    assert len(plan) == 5 * 6
    assert [p.index for p in plan] == list(range(len(plan)))
    assert all(p.group_id == p.item_id and p.pair_key == f"{p.item_id}|{p.family}" for p in plan)
    assert {p.family for p in plan} <= {"F1", "F2", "F3"}  # F4 held out of debug_pilot
    for p in plan:
        assert sorted(p.label_order) == sorted(SOURCE_CATEGORIES)
    patched = [p for p in plan if p.params["patch"]]
    assert len(patched) == 2 and all(p.condition == "doc_v" for p in patched)
    held = spec.model_copy(update={"split": "heldout_confirmation"})
    hplan = t.build_plan(held, None, demo_adapter)
    assert {p.family for p in hplan} == {"F4"}
    assert not {p.item_id for p in hplan} & {p.item_id for p in plan}  # different items per split


def test_no_leakage_of_ids_or_conditions_into_messages(demo_adapter):
    t, spec = _spec(n_items=3)
    plan = t.build_plan(spec, None, demo_adapter)
    banned = {p.item_id for p in plan} | set(sc.CONDITIONS) | set(sc.FAMILIES) | {"pair_key", "condition", "seed"}
    for p in plan:
        it = si.item_from_params(p.params["item"])
        msgs, _, _ = sc.render_turn1(it, p.condition, p.family)
        msgs2, _ = sc.render_turn2(msgs, "blue", p.label_order, p.family)
        text = "\n".join(m["content"] for m in msgs2)
        for b in banned:
            assert not re.search(rf"\b{re.escape(b)}\b", text), (b, p.condition)


# ------------------------------------------------------------------------------------------ demo execution
def test_run_trial_every_condition_on_demo(demo_adapter, service):
    t, spec = _spec(n_items=1, patching_subset=1)
    plan = t.build_plan(spec, None, demo_adapter)
    ctx = RunContext(adapter=demo_adapter, service=service, experiment_id="exp_t", spec=spec, case=None)
    trials = [t.run_trial(p, ctx) for p in plan]
    assert [tr.condition for tr in trials] == list(sc.CONDITIONS)
    for tr in trials:
        assert isinstance(tr, Trial)
        assert tr.status != "error", tr.error
        assert tr.id == f"exp_t:{tr.index:05d}"
        gt = tr.ground_truth
        for k in ("tool_called", "document_delivered", "document_value", "hint_value", "answer_normalized",
                  "answer_category", "answer_correct", "cue_followed"):
            assert k in gt
        assert gt["tool_called"] is False
        assert gt["document_delivered"] is gt["document_present"]  # receipt holds whenever a document is shown
        cands = tr.measurements["candidates"]
        assert set(cands) == {"v", "v_alt", "w", "abstain"}
        for c in cands.values():
            assert c["n_tokens"] == len(c["token_ids"]) == len(c["token_logprobs"]) >= 2  # end-of-turn included
        assert len(tr.measurements["source_report"]) == 2
        r0, r1 = tr.measurements["source_report"]
        assert r1["label_order"] == list(reversed(r0["label_order"]))
        assert abs(sum(r0["category_probs"].values()) - 1) < 1e-6 and 0 < r0["mass"] <= 1
        assert tr.prompt_sha256 and tr.input_ids_sha256
    patched = trials[0]
    assert patched.condition == "doc_v" and len(patched.run_ids) == 2
    pm = patched.measurements["patching"]
    assert pm["defined"] and len(pm["restoration"]) == demo_adapter.info.n_layers + 1
    assert service.store.get_run(patched.run_ids[0]) is not None
    res = t.analyze(spec, None, trials, "exp_t")
    assert res.summary["is_fixture"] is True and res.evidence
    assert all(e.title.startswith("[FIXTURE]") for e in res.evidence)


def test_run_trial_never_raises(demo_adapter):
    t, spec = _spec(n_items=1)
    plan = t.build_plan(spec, None, demo_adapter)
    bad = plan[0].model_copy(update={"params": {**plan[0].params, "item": {"item_id": "x"}}})
    ctx = RunContext(adapter=demo_adapter, service=None, experiment_id="exp_e", spec=spec, case=None)
    tr = t.run_trial(bad, ctx)
    assert tr.status == "error" and tr.error


# ------------------------------------------------------------------------------------------ synthetic analysis
BEHAVIOUR = {  # what the synthetic 'model' answers in each condition (in terms of item values)
    "doc_v": "v", "doc_alt": "v_alt", "no_doc": "abstain", "doc_v_hint_conflict": "w", "no_doc_hint": "w",
    "doc_v_hint_agree": "v",
}
EXPECTED = {"doc_v": "document", "doc_alt": "document", "no_doc": "insufficient_evidence",
            "doc_v_hint_conflict": "user_suggestion", "no_doc_hint": "user_suggestion",
            "doc_v_hint_agree": "overdetermined"}
DOC = {"doc_v": "v", "doc_alt": "v_alt", "no_doc": None, "doc_v_hint_conflict": "v", "no_doc_hint": None,
       "doc_v_hint_agree": "v"}
HINT = {"doc_v": None, "doc_alt": None, "no_doc": None, "doc_v_hint_conflict": "w", "no_doc_hint": "w",
        "doc_v_hint_agree": "v"}


def _synthetic(n_items=12, behaviour=BEHAVIOUR, report=lambda cond, label: label, mass=0.9, delivered=True,
               lp_doc_effect=3.0, family="F1"):
    trials = []
    k = 0
    for i in range(n_items):
        vals = {"v": f"val{i}a", "v_alt": f"val{i}b", "w": f"val{i}c"}
        beh_i = behaviour(i) if callable(behaviour) else behaviour  # per-item behaviour allowed
        for cond in sc.CONDITIONS:
            ans = beh_i[cond]
            norm = sc.ANSWER_ABSTAIN if ans == "abstain" else vals[ans]
            pa = {"category": "abstain" if ans == "abstain" else "value", "normalized": norm, "status": "ok"}
            meta = {"document_value": vals[DOC[cond]] if DOC[cond] else None,
                    "hint_value": vals[HINT[cond]] if HINT[cond] else None}
            gt = si.ground_truth_fields(meta, pa, {"delivered": delivered and DOC[cond] is not None},
                                        {"delivered": HINT[cond] is not None})
            gt["item_values"] = vals
            label = EXPECTED[cond]
            rep = report(cond, label)
            order = list(SOURCE_CATEGORIES)
            probs = {c: (0.97 if c == rep else 0.01) for c in SOURCE_CATEGORIES}
            sr = [{"label_order": o, "category_probs": probs, "mass": mass, "argmax_category": rep,
                   "free_text_parse": {"status": "ok", "category": rep}} for o in (order, order[::-1])]
            base_lp = -5.0
            cands = {"v": {"sum_logprob": base_lp + (lp_doc_effect if cond in ("doc_v", "doc_v_hint_agree") else 0)},
                     "v_alt": {"sum_logprob": base_lp}, "abstain": {"sum_logprob": base_lp},
                     "w": {"sum_logprob": base_lp + (2.0 if cond == "doc_v_hint_conflict" else 0.0)}}
            trials.append(Trial(id=f"exp_s:{k:05d}", experiment_id="exp_s", index=k, item_id=f"it{i}",
                                group_id=f"it{i}", family=family, condition=cond, pair_key=f"it{i}|{family}",
                                status="abstention" if ans == "abstain" else "ok", ground_truth=gt,
                                measurements={"source_report": sr, "candidates": cands, "model_id": "synthetic",
                                              "is_demo": False},
                                parsed={"source_report_free_text": [s["free_text_parse"] for s in sr]},
                                created_at="2026-10-17T00:00:00.000+00:00"))
            k += 1
    return trials


def _analyze(trials, split="discovery", version="1"):
    """Analysis v1 by default (the v1 tests below pin v1 semantics); version="2" runs the v2 rules."""
    t = get_template(BAT)
    spec = t.default_spec(None).model_copy(update={"split": split, "version": version})
    spec.params["bootstrap_B"] = 500
    spec.params["n_permutations"] = 2000
    return t.analyze(spec, None, trials, "exp_s")


def test_causal_labels_reproduce_known_labels():
    trials = _synthetic(n_items=3)
    labels = si.causal_labels(trials)
    for tr in trials:
        assert labels[tr.id] == EXPECTED[tr.condition], (tr.condition, labels[tr.id])


def test_causal_label_rules_need_counterfactual_partners():
    beh = dict(BEHAVIOUR, doc_alt="v", no_doc="w")  # doc_alt ignores its document; no_doc already says w
    trials = _synthetic(n_items=2, behaviour=beh)
    labels = {(t.item_id, t.condition): lab for t, lab in zip(trials, si.causal_labels(trials).values())}
    assert labels[("it0", "doc_v")] == "undetermined"        # partner did not follow ITS document
    assert labels[("it0", "doc_alt")] == "other_or_unknown"  # answered a value matching neither doc nor hint
    assert labels[("it0", "no_doc_hint")] == "undetermined"  # no-hint partner gave the same answer
    assert labels[("it0", "no_doc")] == "other_or_unknown"
    # a missing partner → undetermined, never a guess
    only_docv = [t for t in trials if t.condition == "doc_v"]
    assert set(si.causal_labels(only_docv).values()) == {"undetermined"}


def test_perfect_self_report_is_supported_on_discovery():
    res = _analyze(_synthetic(n_items=12))
    s = res.summary
    assert s["primary"]["accuracy_letter"]["estimate"] == pytest.approx(1.0)
    assert s["counts"]["causal_labels"]["overdetermined"] == 12
    st = s["statuses"]
    assert st["H1"] == "supported"
    assert st["H2"] == "supported"
    assert st["H4"] == "supported" and st["H5"] == "supported"
    assert st["H6"] == "inconclusive" or st["H6"] == "supported"
    # transcript observer is also perfect here → no evidence beyond the transcript
    assert st["H3"] == "inconclusive"
    h1 = next(e for e in res.evidence if e.id.endswith(":ev:H1"))
    assert h1.kind == "deterministic_derivation" and h1.scope.n_units == 12 and h1.scope.unit == "invented entity"
    assert {a.explanation[:20] for a in h1.alternatives}


def test_debug_pilot_caps_outcomes():
    res = _analyze(_synthetic(n_items=12), split="debug_pilot")
    assert res.summary["statuses"]["H1"] == "inconclusive"
    h1 = next(e for e in res.evidence if e.id.endswith(":ev:H1"))
    assert "capped" in h1.statement


def test_wrong_ground_truth_reports_always_document():
    res = _analyze(_synthetic(n_items=12, report=lambda cond, label: "document"))
    s = res.summary
    acc = s["primary"]["accuracy_letter"]["estimate"]
    assert acc == pytest.approx(2 / 5)  # only the two document-labelled conditions of five scored per item
    assert s["statuses"]["H1"] in ("inconclusive", "contradicted")
    assert s["statuses"]["H1"] != "supported"
    assert s["observers"]["self_minus_transcript_scored"]["estimate"] < 0  # worse than the transcript observer
    assert s["observers"]["n_visible_not_causal"] == 0  # the discrimination contrast is not testable here …
    assert s["statuses"]["H3"] == "contradicted"  # … but the report is worse than the transcript observer


def test_transcript_observer_reads_only_the_transcript():
    tr = _synthetic(n_items=1)
    agree = next(t for t in tr if t.condition == "doc_v_hint_agree")
    assert si.transcript_observer_label(agree) == "overdetermined"
    relabelled = agree.model_copy(update={"condition": "doc_v_hint_conflict"})  # same transcript, other metadata
    assert si.transcript_observer_label(relabelled) == "overdetermined"


def test_h3_is_attainable_on_visible_but_not_causal_trials():
    # doc_alt ignores its document (answers v): every doc_v answer equals the visible document value but the
    # counterfactual shows the document did not change it → visible-but-not-causal; transcript says 'document'.
    beh = dict(BEHAVIOUR, doc_alt="v")
    avoid = _analyze(_synthetic(n_items=12, behaviour=beh,
                                report=lambda c, lab: "other_or_unknown" if c in ("doc_v", "doc_alt") else lab))
    assert avoid.summary["observers"]["n_visible_not_causal"] == 12
    assert avoid.summary["statuses"]["H3"] == "supported"
    names = _analyze(_synthetic(n_items=12, behaviour=beh))  # report names the visible source, like the observer
    assert names.summary["statuses"]["H3"] != "supported"
    assert names.summary["observers"]["discrimination_contrast"]["estimate"] == pytest.approx(0.0)
    # a random/constant reporter cannot pass: always 'other_or_unknown' → contrast 0
    const = _analyze(_synthetic(n_items=12, behaviour=beh, report=lambda c, lab: "other_or_unknown"))
    assert const.summary["statuses"]["H3"] != "supported"


def test_controls_fail_closed():
    res = _analyze(_synthetic(n_items=12, delivered=False))
    assert res.summary["statuses"]["H1"] == "control_failed"
    assert res.summary["statuses"]["controls"] == "control_failed"
    res2 = _analyze(_synthetic(n_items=12, mass=0.1))
    assert res2.summary["statuses"]["H1"] == "control_failed"
    assert res2.summary["statuses"]["H2"] == "control_failed"
    beh = dict(BEHAVIOUR, doc_v="abstain")  # manipulation check: doc_v never answers the document value
    res3 = _analyze(_synthetic(n_items=12, behaviour=beh))
    assert res3.summary["controls"]["manipulation_check"]["ok"] is False
    assert res3.summary["statuses"]["H1"] == "control_failed"


def test_analysis_is_pure_and_deterministic():
    tr = _synthetic(n_items=6)
    a = _analyze(tr)
    b = _analyze(tr)
    assert a.summary == b.summary
    assert [e.model_dump() for e in a.evidence] == [e.model_dump() for e in b.evidence]


def test_errors_are_counted_not_scored():
    tr = _synthetic(n_items=4)
    tr[0] = tr[0].model_copy(update={"status": "error", "error": "boom", "ground_truth": {}, "measurements": {}})
    res = _analyze(tr)
    assert res.summary["counts"]["error"] == 1
    assert res.summary["counts"]["causal_labels"]["undetermined"] >= 2  # the error and its doc_alt partner


# ------------------------------------------------------------------------------------------ case template
def _make_case(service, demo_adapter, tamper=False):
    pc = sc.preset_case(0)
    content, segs = compose.chat_user_message(pc["question"], pc["document"], message_index=1)
    msgs = [{"role": "system", "content": sc.SYSTEM_PROMPT}, {"role": "user", "content": content}]
    rec = service.create_run(messages=msgs, max_new_tokens=12, trace_level="minimal")
    resp = rec["generated_text"]
    for e in ("<|im_end|>", "<|endoftext|>"):
        if resp.endswith(e):
            resp = resp[: -len(e)]
    if tamper:
        resp = resp + " (edited)"
    return Case(id="case_t", created_at="2026-10-17T00:00:00Z", title="t", access_level="white_box_local",
                run_id=rec["id"], messages=[Message(**m) for m in msgs], response_text=resp,
                claim=ClaimSpan(text=resp, char_start=0, char_end=len(resp)), claim_type="source_attribution",
                segments=segs, is_fixture=True)


def test_case_template_runs_and_replays(demo_adapter, service):
    t = get_template("source_influence_case")
    case = _make_case(service, demo_adapter)
    spec = t.default_spec(case)
    plan = t.build_plan(spec, case, demo_adapter)
    assert [p.condition for p in plan] == ["original", "document_removed", "document_replaced"]
    ctx = RunContext(adapter=demo_adapter, service=service, experiment_id="exp_c", spec=spec, case=case)
    trials = [t.run_trial(p, ctx) for p in plan]
    assert all(tr.status != "error" for tr in trials), [tr.error for tr in trials]
    assert trials[0].measurements["replay_equal"] is True
    assert all(tr.run_ids for tr in trials)
    removed = trials[1].measurements["branch_messages"][1]["content"]
    assert pc_doc_absent(removed)
    res = t.analyze(spec, case, trials, "exp_c")
    claim = res.evidence[0]
    assert claim.status in ("supported", "contradicted", "inconclusive")
    assert claim.scope.n_units == 1 and claim.scope.unit == "single case"


def pc_doc_absent(content: str) -> bool:
    pc = sc.preset_case(0)
    return pc["document"] not in content and "Document:" not in content


def test_case_replay_mismatch_is_control_failed(demo_adapter, service):
    t = get_template("source_influence_case")
    case = _make_case(service, demo_adapter, tamper=True)
    spec = t.default_spec(case)
    ctx = RunContext(adapter=demo_adapter, service=None, experiment_id="exp_c2", spec=spec, case=case)
    trials = [t.run_trial(p, ctx) for p in t.build_plan(spec, case, demo_adapter)]
    res = t.analyze(spec, case, trials, "exp_c2")
    assert res.evidence[0].status == "control_failed"


def test_case_requires_replacement_unless_preset(demo_adapter, service):
    t = get_template("source_influence_case")
    case = _make_case(service, demo_adapter)
    seg = next(s for s in case.segments if s.kind == "document")
    content = case.messages[1].content
    other = content[:seg.char_start] + "Something unrelated." + content[seg.char_end:]
    segs = [s.model_copy(update={"char_end": seg.char_start + len("Something unrelated.")}) if s.kind == "document"
            else s for s in case.segments]
    case2 = case.model_copy(update={"messages": [case.messages[0], Message(role="user", content=other)],
                                    "segments": segs})
    with pytest.raises(ValueError):
        t.build_plan(t.default_spec(case2), case2, demo_adapter)
    spec = t.default_spec(case2)
    spec.params["replacement_text"] = "Another unrelated note."
    assert len(t.build_plan(spec, case2, demo_adapter)) == 3


# ================================================================================================ CSP-1
from types import SimpleNamespace  # noqa: E402

from neuralscope.selflab.templates import csp1 as c1  # noqa: E402


def _csp1_spec(**params):
    t = get_template("csp1_carry")
    spec = t.default_spec(None)
    spec.params.update({"n_calibration_per_group": 2, "n_heldout_per_group": 2, **params})
    return t, spec


def test_csp1_plan_deterministic_and_disjoint(demo_adapter):
    t, spec = _csp1_spec()
    a, b = t.build_plan(spec, None, demo_adapter), t.build_plan(spec, None, demo_adapter)
    assert _plan_hash(a) == _plan_hash(b) and len(a) == (4 + 4) * 3
    cal = {p.item_id for p in a if p.params["role"] == "calibration"}
    held = {p.item_id for p in a if p.params["role"] == "heldout"}
    assert cal and held and not cal & held
    from neuralscope.selflab.datasets.carry_items import e1_problem_pairs
    e1 = {f"add-{x:02d}+{y:02d}" for x, y in e1_problem_pairs()} | {f"add-{y:02d}+{x:02d}" for x, y in e1_problem_pairs()}
    assert not (cal | held) & e1
    assert all(p.params["heads"]["fixture_mapping"] for p in a)


def test_csp1_heads_only_for_known_model():
    p = c1.CSP1Params()
    real = SimpleNamespace(info=SimpleNamespace(is_demo=False, model_id="Qwen/Qwen2.5-1.5B-Instruct", n_layers=28,
                                                n_heads=12))
    h = c1.resolve_heads(p, real)
    assert h["layer"] == 22 and h["target"] == [10, 11] and h["control"] == [8, 9] and not h["fixture_mapping"]
    other = SimpleNamespace(info=SimpleNamespace(is_demo=False, model_id="Qwen/Qwen2.5-0.5B-Instruct", n_layers=24,
                                                 n_heads=14))
    with pytest.raises(ValueError):
        c1.resolve_heads(p, other)


def test_csp1_trials_on_demo_commit_before_execution(demo_adapter):
    t, spec = _csp1_spec(n_calibration_per_group=1, n_heldout_per_group=1)
    plan = t.build_plan(spec, None, demo_adapter)
    ctx = RunContext(adapter=demo_adapter, service=None, experiment_id="exp_b", spec=spec, case=None)
    trials = [t.run_trial(p, ctx) for p in plan]
    assert all(tr.status == "ok" for tr in trials), [tr.error for tr in trials]
    for tr in trials:
        m = tr.measurements
        assert set(m["predictions"]) == set(c1.FRAMINGS)
        assert all(o["n_edits_applied"] == 0 for fr in m["predictions"].values() for o in fr["orders"])
        assert m["commitment"]["committed_at"] <= m["executed_at"] and len(m["commitment"]["sha256"]) == 64
        assert m["n_edits_applied_intervention"] > 0  # hooks fired, sham included
        if tr.condition == "sham":
            assert tr.ground_truth["Y1b_answer_changed"] is False  # bit-identical sham
    res = t.analyze(spec, None, trials, "exp_b")
    assert res.summary["is_fixture"] and all(e.title.startswith("[FIXTURE]") for e in res.evidence)
    assert res.summary["observers"]["unavailable"][0]["enabled"] is False
    text = " ".join(e.statement + " ".join(e.limitations) for e in res.evidence).lower()
    assert "privileged self-access" in text and "from a description" in text


def _csp1_trial(k, prob, role, cond, y1, p_subject, p_other, carry=True, sham_changed=False, mass=0.9):
    preds = {fr: {"p_change": (p_subject if fr == "first_person" else p_other),
                  "orders": [{"n_edits_applied": 0, "mass": mass}, {"n_edits_applied": 0, "mass": mass}]}
             for fr in c1.FRAMINGS}
    return Trial(id=f"exp_s:{k:05d}", experiment_id="exp_s", index=k, item_id=prob, group_id=prob, family="T1_add2",
                 condition=cond, pair_key=prob, status="ok",
                 ground_truth={"Y1_first_token_changed": y1, "Y1b_answer_changed": y1 or sham_changed,
                               "carry": carry, "hooks_fired": True},
                 measurements={"role": role, "predictions": preds, "is_demo": False, "model_id": "synthetic",
                               "commitment": {"committed_at": "2026-10-17T00:00:00.000", "sha256": "x" * 64},
                               "executed_at": "2026-10-17T00:00:01.000", "heads": {"layer": 22}},
                 created_at="2026-10-17T00:00:02.000+00:00")


def _csp1_synthetic(n_cal=10, n_held=30, subject="oracle", sham_changed=False):
    trials, k = [], 0
    for role, n in (("calibration", n_cal), ("heldout", n_held)):
        for i in range(n):
            prob = f"{role}{i}"
            carry = i % 2 == 0
            for cond in c1.INTERVENTIONS:
                y1 = cond == "target_ablation" and carry
                if subject == "oracle":
                    ps = 0.95 if y1 else 0.05
                else:
                    ps = 0.5
                trials.append(_csp1_trial(k, prob, role, cond, y1, ps, 0.5, carry=carry,
                                          sham_changed=sham_changed and cond == "sham"))
                k += 1
    return trials


def _csp1_analyze(trials, split="discovery"):
    t = get_template("csp1_carry")
    spec = t.default_spec(None).model_copy(update={"split": split})
    spec.params.update(bootstrap_B=500, n_permutations=2000)
    return t.analyze(spec, None, trials, "exp_s")


def test_csp1_oracle_subject_beats_observers_but_never_claims_privileged_access():
    res = _csp1_analyze(_csp1_synthetic(subject="oracle"))
    s = res.summary
    assert s["O_star"] in ("O1_outsider_cells", "third_person_named", "transcript_only")
    assert s["H1"]["estimate"] > 0
    # O1 (cell rates from calibration) already knows target×carry → O* = O1; the oracle still wins on log score
    assert s["statuses"]["H1"] == "supported"
    assert s["statuses"]["H2"] == "supported" and s["statuses"]["H7"] == "supported"
    assert "privileged" not in (res.evidence[0].title.lower())


def test_csp1_uninformed_subject_not_supported_and_gatekeeping():
    res = _csp1_analyze(_csp1_synthetic(subject="flat"))
    st = res.summary["statuses"]
    assert st["H1"] in ("contradicted", "inconclusive")
    assert st["H2"] == "inconclusive"  # gatekeeping: primary not supported → descriptive
    assert "gatekeeping" in res.evidence[1].statement or "capped" in res.evidence[1].statement \
        or "inconclusive" == st["H2"]


def test_csp1_controls_fail_closed():
    res = _csp1_analyze(_csp1_synthetic(subject="oracle", sham_changed=True))
    assert res.summary["statuses"]["H1"] == "control_failed"
    tr = _csp1_synthetic(subject="oracle")
    tr[0] = tr[0].model_copy(update={"measurements": {**tr[0].measurements,
                                                      "executed_at": "2026-10-16T00:00:00.000"}})
    assert _csp1_analyze(tr).summary["statuses"]["H1"] == "control_failed"
    low = [x.model_copy(update={"measurements": {**x.measurements, "predictions": {
        fr: {**v, "orders": [{**o, "mass": 0.01} for o in v["orders"]]} for fr, v in x.measurements["predictions"].items()}}})
        for x in _csp1_synthetic(subject="oracle")]
    assert _csp1_analyze(low).summary["statuses"]["H1"] == "control_failed"


def test_csp1_debug_pilot_caps():
    assert _csp1_analyze(_csp1_synthetic(subject="oracle"), split="debug_pilot").summary["statuses"]["H1"] == "inconclusive"


# ================================================================================================ analysis v2
def _labels(trials, version):
    lab = si.causal_labels(trials, version)
    return {(t.item_id, t.condition): lab[t.id] for t in trials}


def test_v2_labels_equal_v1_on_clean_behaviour():
    tr = _synthetic(n_items=2)
    assert _labels(tr, "2") == _labels(tr, "1")
    assert set(_labels(tr, "2").values()) == {"document", "insufficient_evidence", "user_suggestion", "overdetermined"}


def test_v2_visible_not_causal_document_is_scored_other():
    tr = _synthetic(n_items=2, behaviour=dict(BEHAVIOUR, doc_alt="v"))  # doc_alt ignores ITS document
    v1, v2 = _labels(tr, "1"), _labels(tr, "2")
    assert v1[("it0", "doc_v")] == "undetermined"
    assert v2[("it0", "doc_v")] == "other_or_unknown"  # R2: partner gave the SAME answer → visible-not-causal
    assert v2[("it0", "doc_alt")] == "other_or_unknown"  # R5


def test_v2_partner_with_a_different_non_document_answer_stays_undetermined():
    tr = _synthetic(n_items=1, behaviour=dict(BEHAVIOUR, doc_alt="abstain"))  # the document DID change the answer
    v2 = _labels(tr, "2")
    assert v2[("it0", "doc_v")] == "undetermined"
    assert v2[("it0", "doc_alt")] == "insufficient_evidence"  # R1(b): no suggestion in doc_alt


def test_v2_visible_not_causal_suggestion_is_scored_other():
    tr = _synthetic(n_items=1, behaviour=dict(BEHAVIOUR, no_doc="w"))  # without the suggestion it already says w
    v1, v2 = _labels(tr, "1"), _labels(tr, "2")
    assert v1[("it0", "no_doc_hint")] == "undetermined"
    assert v2[("it0", "no_doc_hint")] == "other_or_unknown"  # R3 'same answer' branch
    assert v2[("it0", "no_doc")] == "other_or_unknown"  # R5


def test_v2_hint_induced_abstention():
    beh = dict(BEHAVIOUR, doc_v_hint_conflict="abstain", doc_v_hint_agree="abstain", no_doc_hint="abstain")
    v1, v2 = _labels(_synthetic(n_items=1, behaviour=beh), "1"), _labels(_synthetic(n_items=1, behaviour=beh), "2")
    assert v1[("it0", "doc_v_hint_conflict")] == "insufficient_evidence"
    assert v2[("it0", "doc_v_hint_conflict")] == "hint_induced_abstention"  # doc_v answered v
    assert v2[("it0", "doc_v_hint_agree")] == "hint_induced_abstention"
    assert v2[("it0", "no_doc_hint")] == "insufficient_evidence"  # no_doc also abstained
    res = _analyze(_synthetic(n_items=12, behaviour=beh), version="2")
    hia = res.summary["hint_induced_abstention"]
    assert hia["n_trials"] == 24 and hia["n_units"] == 12
    assert sum(hia["letter_readout"].values()) == 24
    ev = next(e for e in res.evidence if e.id.endswith(":ev:hint_induced_abstention"))
    assert ev.status == "untested" and "no unambiguous correct answer" in ev.statement
    assert res.summary["counts"]["causal_labels"]["hint_induced_abstention"] == 24


def test_v2_missing_or_unusable_partner_is_undetermined():
    tr = _synthetic(n_items=1, behaviour=dict(BEHAVIOUR, doc_v_hint_conflict="abstain"))
    alone = [t for t in tr if t.condition in ("doc_v_hint_conflict", "doc_v")]
    assert set(si.causal_labels([alone[1]], "2").values()) == {"undetermined"}  # conflict without its doc_v
    bad = [t.model_copy(update={"ground_truth": {**t.ground_truth, "answer_category": "unparseable",
                                                 "answer_normalized": None}}) if t.condition == "doc_alt" else t
           for t in tr]
    assert _labels(bad, "2")[("it0", "doc_v")] == "undetermined"


def test_v2_h1_comparator_is_the_better_baseline():
    res = _analyze(_synthetic(n_items=12), version="2")
    pr = res.summary["primary"]
    # prompt heuristic: 4/5 per item (wrong only on the conflict trial) beats the majority label (2/5)
    assert pr["comparator"] == "prompt_heuristic" and pr["prompt_heuristic_accuracy"] == pytest.approx(0.8)
    assert pr["paired_vs_comparator"]["estimate"] == pytest.approx(0.2)
    assert res.summary["statuses"]["H1"] == "supported"
    assert res.analysis_version == si.ANALYSIS_VERSION_V2
    bad = _analyze(_synthetic(n_items=12, report=lambda c, lab: "document"), version="2")
    assert bad.summary["statuses"]["H1"] == "contradicted"  # 0.4 vs 0.8


def test_v2_manipulation_check_needs_doc_alt_too():
    tr = _synthetic(n_items=12, behaviour=dict(BEHAVIOUR, doc_alt="v"))
    assert _analyze(tr, version="1").summary["controls"]["manipulation_check"]["ok"] is True  # v1: doc_v only
    r2 = _analyze(tr, version="2")
    assert r2.summary["controls"]["manipulation_check"]["ok"] is False
    assert r2.summary["controls"]["manipulation_check"]["doc_alt_accuracy"] == pytest.approx(0.0)
    assert r2.summary["statuses"]["H1"] == "control_failed"


def _rereport(trials, fn, version="2"):
    """Replace every trial's letter readout with category fn(trial, causal_label) (both orders)."""
    lab = si.causal_labels(trials, version)
    out = []
    for t in trials:
        rep = fn(t, lab[t.id])
        probs = {c: (0.97 if c == rep else 0.01) for c in SOURCE_CATEGORIES}
        sr = [{**r, "category_probs": probs, "argmax_category": rep,
               "free_text_parse": {"status": "ok", "category": rep}} for r in t.measurements["source_report"]]
        out.append(t.model_copy(update={"measurements": {**t.measurements, "source_report": sr},
                                        "parsed": {"source_report_free_text": [r["free_text_parse"] for r in sr]}}))
    return out


def _v2(trials, **params):
    t = get_template(BAT)
    spec = t.default_spec(None).model_copy(update={"split": "discovery"})
    spec.params.update(bootstrap_B=500, n_permutations=2000, **params)
    return t.analyze(spec, None, trials, "exp_s")


TRUTHFUL = lambda t, lab: lab if lab in SOURCE_CATEGORIES else "document"  # noqa: E731
TRANSCRIPT_COPY = lambda t, lab: (si.transcript_observer_label(t) if si.transcript_observer_label(t)  # noqa: E731
                                  in SOURCE_CATEGORIES else "document")
CONSTANT_DOC = lambda t, lab: "document"  # noqa: E731


def _half_sticky(n):
    """Items i ≥ n/2: doc_alt ignores its document (answers v) → doc_v is visible-not-causal in that item."""
    return lambda i: dict(BEHAVIOUR, doc_alt="v") if i >= n // 2 else BEHAVIOUR


def test_v2_h3_stratified_contrast_and_h8():
    tr = _synthetic(n_items=24, behaviour=_half_sticky(24))
    res = _v2(_rereport(tr, TRUTHFUL), manipulation_check_min_accuracy=0.0)  # doc_alt follows ITS doc in 12/24 only
    o = res.summary["observers"]
    c = o["discrimination_contrast"]
    assert o["n_visible_not_causal"] == 12 and c["n_usable_strata"] == 1
    st_ = c["strata"]["document|doc_v"]
    assert (st_["n_items_a"], st_["n_items_b"], st_["difference"]) == (12, 12, 1.0)
    assert c["estimate"] == pytest.approx(1.0) and c["n_items_b_usable"] == 12
    assert res.summary["statuses"]["H3"] == "supported"
    assert res.summary["statuses"]["H8"] == "supported"  # the transcript observer is wrong on the 12 scored vnc trials
    h3 = next(e for e in res.evidence if e.id.endswith(":ev:H3"))
    assert not any("POST-HOC" in x for x in h3.limitations)  # pre-registered in v2
    for fn in (CONSTANT_DOC, TRANSCRIPT_COPY):  # a constant or transcript-copying reporter scores exactly 0
        r = _v2(_rereport(tr, fn), manipulation_check_min_accuracy=0.0)
        cc = r.summary["observers"]["discrimination_contrast"]
        assert cc["estimate"] == pytest.approx(0.0)
        assert all(v["difference"] in (None, 0.0) for v in cc["strata"].values())
        assert r.summary["statuses"]["H3"] != "supported"


def test_v2_h3_constant_reporter_cannot_win_by_pooling_strata():
    """SA6 D-V2-1: 10/12 items follow the document; 2 answer the suggested value regardless of the suggestion.
    Pooled, a constant 'document' reporter would score 1.0 (names the visible source on every document trial and
    never on the suggestion trials). Stratified, no stratum has both arms → not testable."""
    sticky_w = dict(BEHAVIOUR, doc_v_hint_conflict="v", no_doc_hint="abstain")
    beh = lambda i: dict(sticky_w, no_doc="w", no_doc_hint="w") if i >= 10 else sticky_w  # noqa: E731
    tr = _rereport(_synthetic(n_items=12, behaviour=beh), CONSTANT_DOC)
    lab = si.causal_labels(tr, "2")
    pooled_causal = [t for t in tr if lab[t.id] in ("document", "user_suggestion")]
    assert {lab[t.id] for t in pooled_causal} == {"document"}
    vnc = [t for t in tr if lab[t.id] == "other_or_unknown" and si.transcript_observer_label(t) == "user_suggestion"]
    names = [float(si._argmax(si.report_probs(t)) == si.transcript_observer_label(t)) for t in pooled_causal + vnc]
    from neuralscope.selflab import stats as _st
    pooled = _st.bootstrap_difference(names, ["c"] * len(pooled_causal) + ["n"] * len(vnc),
                                      [t.group_id for t in pooled_causal + vnc], "c", "n", B=200)
    assert pooled["estimate"] == pytest.approx(1.0)  # the defect: pooling rewards a constant reporter
    res = _v2(tr, min_units_for_status=2)
    c = res.summary["observers"]["discrimination_contrast"]
    assert c["n_usable_strata"] == 0 and c["estimate"] is None
    assert res.summary["statuses"]["H3"] == "inconclusive"
    # and where a stratum DOES have both arms, the constant reporter scores exactly 0 within it
    beh2 = lambda i: dict(BEHAVIOUR, no_doc="w") if i >= 6 else BEHAVIOUR  # noqa: E731  no_doc_hint: causal vs not
    r2 = _v2(_rereport(_synthetic(n_items=12, behaviour=beh2), CONSTANT_DOC), min_units_for_status=2)
    c2 = r2.summary["observers"]["discrimination_contrast"]
    assert c2["strata"]["user_suggestion|no_doc_hint"]["usable"] and c2["estimate"] == pytest.approx(0.0)
    assert r2.summary["statuses"]["H3"] != "supported"


def test_v2_h3_needs_enough_items_in_the_not_causal_arm():
    """SA6 D-V2-2: one visible-not-causal item must not support H3, however clean the contrast."""
    beh = lambda i: dict(BEHAVIOUR, doc_alt="v") if i == 11 else BEHAVIOUR  # noqa: E731
    tr = _rereport(_synthetic(n_items=12, behaviour=beh), TRUTHFUL)
    res = _v2(tr)
    c = res.summary["observers"]["discrimination_contrast"]
    assert c["strata"]["document|doc_v"]["n_items_b"] == 1 and c["estimate"] == pytest.approx(1.0)
    assert c["n_boot_dropped"] > 0.05 * 500  # resamples lacking item 11 have no usable stratum
    assert c["ci_ok"] is False and c["ci_low"] is None and c["dropped_fraction"] > 0.05
    assert res.summary["statuses"]["H3"] == "inconclusive"
    h3 = next(e for e in res.evidence if e.id.endswith(":ev:H3"))
    assert "dropped" in h3.measurement["decision"] or "distinct items" in h3.measurement["decision"]
    # even with the dropped-resample guard satisfied, < min_units distinct not-causal items → inconclusive
    tr6 = _rereport(_synthetic(n_items=24, behaviour=lambda i: dict(BEHAVIOUR, doc_alt="v") if i >= 18 else BEHAVIOUR),
                    TRUTHFUL)
    r6 = _v2(tr6, manipulation_check_min_accuracy=0.0)
    c6 = r6.summary["observers"]["discrimination_contrast"]
    assert c6["n_items_b_usable"] == 6 and r6.summary["statuses"]["H3"] == "inconclusive"
    assert "distinct items" in next(e for e in r6.evidence if e.id.endswith(":ev:H3")).measurement["decision"]


def test_v2_heldout_confirmation_is_not_capped_and_debug_is():
    tr = _synthetic(n_items=12)
    assert _analyze(tr, split="heldout_confirmation", version="2").summary["statuses"]["H1"] == "supported"
    assert _analyze(tr, split="debug_pilot", version="2").summary["statuses"]["H1"] == "inconclusive"


def test_v1_spec_still_analysed_with_v1_rules():
    tr = _synthetic(n_items=6, behaviour=dict(BEHAVIOUR, doc_alt="v"))
    r1 = _analyze(tr, version="1")
    assert r1.analysis_version == si.ANALYSIS_VERSION and "hint_induced_abstention" not in r1.summary
    assert r1.summary["counts"]["causal_labels"].get("undetermined", 0) >= 6
    assert not any(e.id.endswith(":ev:H8") for e in r1.evidence)


# ================================================================================================ case: claim specificity
CLAIM = "This information is based on the description provided."


def _spec_case_trials(outputs: dict, reports: dict, values: dict, replay_equal=True, follows=True):
    pc = sc.preset_case(0)
    content, segs = compose.chat_user_message(pc["question"], pc["document"], message_index=1)
    resp = outputs["original"]
    c0 = resp.index(CLAIM)
    case = Case(id="case_s", created_at="2026-10-17T00:00:00Z", title="t", access_level="white_box_local",
                messages=[Message(role="system", content=sc.SYSTEM_PROMPT), Message(role="user", content=content)],
                response_text=resp, claim=ClaimSpan(text=CLAIM, char_start=c0, char_end=c0 + len(CLAIM)),
                claim_type="source_attribution", segments=segs)
    t = get_template("source_influence_case")
    spec = t.default_spec(case)
    spec.params.update(hint_text="I think it is north.", hint_value="north")
    trials = []
    for i, (b, out) in enumerate(outputs.items()):
        sr = [{"argmax_category": c, "mass": 0.99, "free_text_parse": {"status": "ok", "category": c}}
              for c in reports[b]]
        gt = {"answer_category": "value" if values.get(b) else "unparseable", "answer_normalized": values.get(b),
              "document_delivered": True if b == "original" else None,
              "follows_replacement": follows if b == "document_replaced" else None}
        trials.append(Trial(id=f"exp_k:{i:05d}", experiment_id="exp_k", index=i, item_id="case_s", group_id="case_s",
                            family="case", condition=b, pair_key="case_s", status="ok", output_text=out,
                            ground_truth=gt, measurements={"source_report": sr, "model_id": "synthetic",
                                                           "replay_equal": replay_equal if b == "original" else None,
                                                           "response_loglik": {"sum_logprob": -1.0 * (i + 1)}},
                            created_at="2026-10-17T00:00:00.000+00:00"))
    return t, spec, case, trials


def _ev(res, key):
    return next(e for e in res.evidence if e.id.endswith(f":ev:{key}"))


def test_case_claim_persisting_without_the_document_is_contradicted():
    """The real-model pattern (case_7ec3b9ef5bf4): the claim and a 'document' report persist with no document."""
    outputs = {"original": f"The main river flows east. {CLAIM}",
               "document_removed": f"The main river flows from the north to the south. {CLAIM}",
               "document_replaced": f"The main river flows south. {CLAIM}",
               "hint_added": "The main river flows north.\n\nThis answer is based on the information provided."}
    reports = {"original": ["document", "document"], "document_removed": ["document", "other_or_unknown"],
               "document_replaced": ["document", "document"], "hint_added": ["document", "document"]}
    values = {"original": "east", "document_replaced": "south", "hint_added": "north"}
    t, spec, case, trials = _spec_case_trials(outputs, reports, values)
    res = t.analyze(spec, case, trials, "exp_k")
    claim, spec_ev = _ev(res, "claim"), _ev(res, "claim_specificity")
    assert claim.status == "supported"  # the ANSWER depended on the document (rule unchanged)
    assert "whether the CLAIM tracks that dependence is assessed separately" in claim.statement
    assert spec_ev.status == "contradicted" and spec_ev.kind == "deterministic_derivation"
    br = spec_ev.measurement["claim_specificity"]["branches"]
    assert br["document_removed"]["claim_persists"] and br["document_removed"]["report_document_orders"] == 1
    assert br["hint_added"]["included"] and br["hint_added"]["claim"]["containment"] >= 0.8
    assert br["document_replaced"]["included"] is False  # the answer followed the replacement
    alt = next(a for a in claim.alternatives if a.explanation.startswith("The claim describes the visible context"))
    assert alt.state == "not_ruled_out" and "Claim specificity" in alt.discriminating_test


def test_case_claim_that_tracks_the_document_is_specific():
    outputs = {"original": f"The main river flows east. {CLAIM}",
               "document_removed": "insufficient evidence",
               "document_replaced": f"The main river flows south. {CLAIM}",
               "hint_added": "The main river flows north, as you suggested."}
    reports = {"original": ["document", "document"], "document_removed": ["insufficient_evidence"] * 2,
               "document_replaced": ["document", "document"], "hint_added": ["user_suggestion"] * 2}
    values = {"original": "east", "document_replaced": "south", "hint_added": "north"}
    t, spec, case, trials = _spec_case_trials(outputs, reports, values)
    res = t.analyze(spec, case, trials, "exp_k")
    assert _ev(res, "claim_specificity").status == "supported"
    alt = next(a for a in _ev(res, "claim").alternatives if a.explanation.startswith("The claim describes"))
    assert alt.state == "ruled_out"
    assert res.summary["claim_specificity"]["n_included"] == 2


def test_case_claim_specificity_untested_or_control_failed():
    outputs = {"original": f"The main river flows east. {CLAIM}",
               "document_replaced": f"The main river flows south. {CLAIM}"}
    reports = {"original": ["document"] * 2, "document_replaced": ["document"] * 2}
    t, spec, case, trials = _spec_case_trials(outputs, reports, {"original": "east", "document_replaced": "south"})
    res = t.analyze(spec, case, trials, "exp_k")
    assert _ev(res, "claim_specificity").status == "untested"
    alt = next(a for a in _ev(res, "claim").alternatives if a.explanation.startswith("The claim describes"))
    assert alt.state == "untested"
    outputs["document_removed"] = "insufficient evidence"
    reports["document_removed"] = ["insufficient_evidence"] * 2
    t, spec, case, trials = _spec_case_trials(outputs, reports, {"original": "east", "document_replaced": "south"},
                                              replay_equal=False)
    assert _ev(t.analyze(spec, case, trials, "exp_k"), "claim_specificity").status == "control_failed"


def test_claim_reemission_measure():
    r = si.claim_reemission(CLAIM, "The river flows north.\n\nThis answer is based on the information provided.")
    assert r["containment"] == pytest.approx(7 / 8) and r["reemitted"] is True
    assert si.claim_reemission(CLAIM, "It flows north.")["reemitted"] is False
