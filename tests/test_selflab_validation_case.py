"""SA6 independent validation — single-case source-attribution claims vs counterfactual use (puppet target).

Receipts prove delivery, never use. The claim "My answer was based on the provided document" must be:
* supported only when removing/replacing the document changes the answer AND the replacement value is followed;
* contradicted when the document is delivered (receipt true) but the answer does not change without it;
* control_failed when the replay does not reproduce the case or the delivery receipt fails — never "verified".
"""

from __future__ import annotations

import re

import pytest

si = pytest.importorskip("neuralscope.selflab.templates.source_influence")
sc = pytest.importorskip("neuralscope.selflab.datasets.source_corpus")

from neuralscope.selflab.compose import chat_user_message  # noqa: E402
from neuralscope.selflab.provenance import now_iso  # noqa: E402
from neuralscope.selflab.registry import RunContext  # noqa: E402
from neuralscope.selflab.schemas import Case, ClaimSpan, ExperimentSpec, Message  # noqa: E402
from tests.test_selflab_validation_puppet import Conversation, make_puppet  # noqa: E402

CLAIM = "My answer was based on the provided document."
PRESET = sc.preset_case(0)
ATTR = sc.ATTRIBUTES[PRESET["attribute"]]


def _doc_value(text: str) -> str | None:
    m = re.search(r"Document:\n\n(.*?)(?:\n\n|$)", text, re.S)
    if not m:
        return None
    hits = [v for v in ATTR.values if re.search(rf"\b{re.escape(v)}\b", m.group(1))]
    return hits[0] if len(hits) == 1 else None


def _report_document(conv: Conversation) -> str:
    inv = {v: k for k, v in sc._option_texts("F1").items()}
    lm = {m.group(1): inv[m.group(2)] for m in re.finditer(r"(?m)^([A-D])\. (.+)$", conv.users[-1])}
    return next(L for L, c in lm.items() if c == "document")


def follower(conv: Conversation) -> str:
    if len(conv.users) > 1:
        return _report_document(conv)
    v = _doc_value(conv.users[0])
    return f"{v}. {CLAIM}" if v else "insufficient evidence."


def ignorer(conv: Conversation) -> str:
    if len(conv.users) > 1:
        return _report_document(conv)
    return f"{PRESET['expected_values']['v']}. {CLAIM}"  # same answer with or without the document


@pytest.fixture(scope="module")
def puppet():
    return make_puppet(follower)


def _case(adapter, response_text: str | None = None) -> Case:
    content, segs = chat_user_message(PRESET["question"], PRESET["document"], None, message_index=0)
    msgs = [{"role": "user", "content": content}]
    if response_text is None:
        g = adapter.generate(adapter.tokenize_messages(msgs).input_ids, 64)
        response_text = adapter.decode([t for t in g.generated_ids if t not in adapter.eos_ids])
    i = response_text.find(CLAIM)
    a, b = (i, i + len(CLAIM)) if i >= 0 else (0, len(response_text))
    return Case(id="case_validation", created_at=now_iso(), title="validation", access_level="white_box_local",
                messages=[Message(**m) for m in msgs], response_text=response_text,
                claim=ClaimSpan(text=response_text[a:b], char_start=a, char_end=b), claim_type="source_attribution",
                segments=segs, is_fixture=True)


def run_case(adapter, case: Case, **params):
    T = si.SourceInfluenceCase()
    spec = ExperimentSpec(template="source_influence_case", case_id=case.id, params={"patch": False, **params},
                          hypotheses=T.default_spec(case).hypotheses)
    plan = T.build_plan(spec, case, adapter)
    ctx = RunContext(adapter=adapter, service=None, experiment_id="slx_case_validation", spec=spec, case=case)
    trials = [T.run_trial(it, ctx) for it in plan]
    return trials, T.analyze(spec, case, trials, "slx_case_validation")


def test_document_follower_claim_is_supported(puppet):
    a, p = puppet
    p.policy = follower
    case = _case(a)
    assert case.response_text.startswith(PRESET["expected_values"]["v"])
    trials, res = run_case(a, case)
    by = {t.condition: t for t in trials}
    assert by["original"].measurements["replay_equal"] is True
    assert by["original"].ground_truth["document_delivered"] is True
    assert by["document_removed"].output_text != case.response_text
    assert by["document_replaced"].ground_truth["follows_replacement"] is True
    assert res.summary["claim_status"] == "supported"
    e = res.evidence[0]
    assert e.scope.n_units == 1 and e.scope.unit == "single case"


def test_delivered_but_unused_document_is_contradicted_not_supported(puppet):
    """Receipt says the document was delivered; the answer is identical without it and with a replacement."""
    a, p = puppet
    p.policy = ignorer
    try:
        case = _case(a)
        trials, res = run_case(a, case)
    finally:
        p.policy = follower
    by = {t.condition: t for t in trials}
    assert by["original"].ground_truth["document_delivered"] is True  # access …
    assert by["document_removed"].output_text == by["original"].output_text  # … but no use
    assert by["document_replaced"].ground_truth["follows_replacement"] is False
    assert res.summary["claim_status"] == "contradicted"
    assert "claim" in res.evidence[0].contradicts and not res.evidence[0].supports


def test_replay_mismatch_is_control_failed(puppet):
    a, p = puppet
    p.policy = follower
    case = _case(a, response_text=f"{PRESET['expected_values']['v_alt']}. {CLAIM}")  # not what the model says
    _, res = run_case(a, case)
    assert res.summary["claim_status"] == "control_failed"
    assert any("replay" in r for r in res.summary["reasons"])


def test_failed_delivery_receipt_is_control_failed_never_supported(puppet, monkeypatch):
    a, p = puppet
    p.policy = follower
    case = _case(a)
    real = a.tokenize_messages

    def hide_document(messages, add_generation_prompt=True):
        tc = real(messages, add_generation_prompt)
        if any("Document:" in m["content"] for m in messages):
            tc.message_char_spans = [None] * len(tc.message_char_spans)
            tc.message_token_spans = [None] * len(tc.message_token_spans)
        return tc

    monkeypatch.setattr(a, "tokenize_messages", hide_document)
    trials, res = run_case(a, case)
    by = {t.condition: t for t in trials}
    assert by["original"].ground_truth["document_delivered"] is False
    assert res.summary["claim_status"] == "control_failed"
    assert res.summary["claim_status"] not in ("supported", "verified")


def test_code_like_params_are_inert_data(puppet, tmp_path):
    """A replacement 'document' that looks like code is only text in the replaced branch; nothing runs."""
    a, p = puppet
    p.policy = follower
    marker = tmp_path / "pwned"
    payload = f"__import__('os').system('touch {marker}')"
    case = _case(a)
    trials, _ = run_case(a, case, replacement_text=payload, replacement_value="blue",
                         original_value=PRESET["expected_values"]["v"], value_set=list(ATTR.values))
    rep = next(t for t in trials if t.condition == "document_replaced")
    assert payload in rep.measurements["branch_messages"][0]["content"]
    assert not marker.exists()
