"""SA6 independent validation — /api/selflab end to end with REAL templates and a scripted (puppet) target.

* full flow: chat → case (claim span) → suggest → commit → run → evidence → open in Lab → export preview;
* fixture marking is present everywhere a fixture result can be seen;
* bounded inputs, unknown templates and code-like params are rejected (400) or remain inert data;
* non-loopback mutating requests are rejected even with a spoofed X-Forwarded-For; cancellation works.
"""

from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("neuralscope_api.selflab_routes")
sc = pytest.importorskip("neuralscope.selflab.datasets.source_corpus")

from neuralscope_api.main import create_app  # noqa: E402
from tests.test_selflab_validation_puppet import Conversation, make_puppet  # noqa: E402

LOCAL = ("127.0.0.1", 50001)
REMOTE = ("198.51.100.23", 40001)  # TEST-NET-2
PRESET = sc.preset_case(0)
ATTR = sc.ATTRIBUTES[PRESET["attribute"]]
CLAIM = "My answer was based on the provided document."


def _policy(conv: Conversation):
    last = conv.users[-1] if conv.users else ""
    if "single letter" in last:
        inv = {v: k for k, v in sc._option_texts("F1").items()}
        lm = {m.group(1): inv.get(m.group(2)) for m in re.finditer(r"(?m)^([A-D])\. (.+)$", last)}
        return next((L for L, c in lm.items() if c == "document"), "A")
    m = re.search(r"Document:\n\n(.*?)(?:\n\n|$)", conv.users[0], re.S) if conv.users else None
    vals = [v for v in ATTR.values if m and re.search(rf"\b{re.escape(v)}\b", m.group(1))]
    return f"{vals[0]}. {CLAIM}" if len(vals) == 1 else "insufficient evidence."


@pytest.fixture(scope="module")
def puppet():
    return make_puppet(_policy)


@pytest.fixture()
def app(puppet, settings, monkeypatch):
    monkeypatch.delenv("NEURALSCOPE_API_TOKEN", raising=False)
    monkeypatch.delenv("NEURALSCOPE_SELFLAB_API_ONLY", raising=False)
    return create_app(settings, adapter=puppet[0])


@pytest.fixture()
def client(app):
    with TestClient(app, client=LOCAL) as c:
        yield c


def _wait(client, eid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/selflab/experiments/{eid}").json()
        if r["status"] in ("done", "cancelled", "error"):
            return r
        time.sleep(0.05)
    raise TimeoutError(eid)


def _chat_and_case(client):
    chat = client.post("/api/selflab/chat", json={"question": PRESET["question"], "document": PRESET["document"],
                                                  "max_new_tokens": 32})
    assert chat.status_code == 200, chat.text
    c = chat.json()
    assert c["is_fixture"] is True
    i = c["response_text"].find(CLAIM)
    assert i > 0, c["response_text"]
    case = client.post("/api/selflab/cases", json={"run_id": c["run_id"], "char_start": i,
                                                   "char_end": i + len(CLAIM)})
    assert case.status_code == 200, case.text
    return c, case.json()


# ------------------------------------------------------------------------------------------ full flow
def test_full_flow_case_template(client):
    chat, case = _chat_and_case(client)
    rec = [r for r in chat["receipts"] if r["data"]["segment_kind"] == "document"]
    assert rec and rec[0]["data"]["verified"] is True
    assert "not" in rec[0]["does_not_prove"].lower() or "never" in rec[0]["does_not_prove"].lower()
    assert case["claim"]["text"] == CLAIM and case["claim_type"] == "source_attribution" and case["is_fixture"]
    sug = client.post("/api/selflab/investigate/suggest", json={"case_id": case["id"]})
    assert sug.status_code == 200
    tpls = client.get(f"/api/selflab/templates?case_id={case['id']}").json()
    t = next(t for t in tpls if t["id"] == "source_influence_case")
    spec = t["default_spec"]
    spec["params"] = {**spec["params"], "patch": True}
    com = client.post("/api/selflab/experiments", json={"spec": spec})
    assert com.status_code == 200, com.text
    eid = com.json()["id"]
    assert com.json()["status"] == "committed" and len(com.json()["committed"]["plan_sha256"]) == 64
    assert client.post(f"/api/selflab/experiments/{eid}/run").json()["status"] == "queued"
    done = _wait(client, eid)
    assert done["status"] == "done", done
    assert done["is_fixture"] is True
    evid = client.get(f"/api/selflab/evidence?experiment_id={eid}").json()
    claim_ev = next(e for e in evid if e["id"].endswith(":ev:claim"))
    assert claim_ev["status"] == "supported" and claim_ev["title"].startswith("[FIXTURE]")
    assert any("FIXTURE" in lim for lim in claim_ev["limitations"])
    assert claim_ev["run_ids"], "each branch must link a Lab run"
    for rid in claim_ev["run_ids"]:
        assert client.get(f"/api/runs/{rid}").status_code == 200  # the tensors exist in the Lab
    trials = client.get(f"/api/selflab/experiments/{eid}/trials").json()
    assert {t["condition"] for t in trials} == {"original", "document_removed", "document_replaced"}
    exp = client.get(f"/api/selflab/experiments/{eid}/export?preview=1")
    assert exp.status_code == 200 and exp.json()["preview"] is True


# D-X1: found by SA6; fixed during validation (regression test).
def test_open_in_lab_for_a_battery_trial(client):
    spec = {"template": "source_influence_battery", "split": "debug_pilot", "seed": 0,
            "params": {"n_items": 1, "patching_subset": 0, "conditions": ["doc_v", "doc_alt"]}}
    eid = client.post("/api/selflab/experiments", json={"spec": spec}).json()["id"]
    client.post(f"/api/selflab/experiments/{eid}/run")
    assert _wait(client, eid)["status"] == "done"
    tid = client.get(f"/api/selflab/experiments/{eid}/trials").json()[0]["id"]
    r = client.post(f"/api/selflab/trials/{tid}/open-in-lab")
    assert r.status_code == 200, r.text
    assert r.json()["reproduced"] is True


# D-X1: found by SA6; fixed during validation (regression test).
def test_open_in_lab_for_a_case_trial(client):
    _, case = _chat_and_case(client)
    tpls = client.get(f"/api/selflab/templates?case_id={case['id']}").json()
    spec = next(t for t in tpls if t["id"] == "source_influence_case")["default_spec"]
    spec["params"] = {**spec["params"], "patch": False}
    eid = client.post("/api/selflab/experiments", json={"spec": spec}).json()["id"]
    client.post(f"/api/selflab/experiments/{eid}/run")
    assert _wait(client, eid)["status"] == "done"
    trials = client.get(f"/api/selflab/experiments/{eid}/trials").json()
    orig = next(t for t in trials if t["condition"] == "original")
    r = client.post(f"/api/selflab/trials/{orig['id']}/open-in-lab")
    assert r.status_code == 200, r.text[:300]
    assert r.json()["reproduced"] is True


def _finished_case_experiment(client):
    _, case = _chat_and_case(client)
    tpls = client.get(f"/api/selflab/templates?case_id={case['id']}").json()
    spec = next(t for t in tpls if t["id"] == "source_influence_case")["default_spec"]
    spec["params"] = {**spec["params"], "patch": False}
    eid = client.post("/api/selflab/experiments", json={"spec": spec}).json()["id"]
    client.post(f"/api/selflab/experiments/{eid}/run")
    _wait(client, eid)
    return eid


# D-X2: found by SA6; fixed during validation (regression test).
def test_redacted_export_contains_no_target_output(client):
    import json as _json
    eid = _finished_case_experiment(client)
    red = _json.dumps(client.get(f"/api/selflab/experiments/{eid}/export?exclude_free_text=1").json())
    assert CLAIM not in red


def test_export_preview_flags_free_text_and_has_no_tensors(client):
    _, case = _chat_and_case(client)
    tpls = client.get(f"/api/selflab/templates?case_id={case['id']}").json()
    spec = next(t for t in tpls if t["id"] == "source_influence_case")["default_spec"]
    spec["params"] = {**spec["params"], "patch": False}
    eid = client.post("/api/selflab/experiments", json={"spec": spec}).json()["id"]
    client.post(f"/api/selflab/experiments/{eid}/run")
    _wait(client, eid)
    full = client.get(f"/api/selflab/experiments/{eid}/export").json()
    prev = client.get(f"/api/selflab/experiments/{eid}/export?preview=1").json()
    red = client.get(f"/api/selflab/experiments/{eid}/export?exclude_free_text=1").json()
    import json as _json
    full_s, red_s = _json.dumps(full), _json.dumps(red)
    assert PRESET["document"][:40] in full_s
    assert PRESET["document"][:40] not in red_s  # the user's document is redacted
    assert '"response_text": {"redacted": true' in red_s  # target output redacted (title: see D-X2)
    assert "free" in _json.dumps(prev).lower()
    assert '"resid"' not in full_s and '"attn"' not in full_s  # tensor references only


# ------------------------------------------------------------------------------------------ bounds & inert data
@pytest.mark.parametrize("body", [
    {"question": "x" * 100_000},
    {"question": "q", "document": "d" * 1_000_000},
    {"question": "q", "max_new_tokens": 100_000},
    {"question": "q", "messages": [{"role": "user", "content": "m"}] * 500},
    {"question": "q", "messages": [{"role": "tool", "content": "m"}]},
    {"question": "q", "exec": "__import__('os').system('id')"},
])
def test_unbounded_or_unknown_chat_inputs_are_rejected(client, body):
    assert client.post("/api/selflab/chat", json=body).status_code == 400


@pytest.mark.parametrize("spec,needle", [
    ({"template": "not_a_template"}, None),
    ({"template": "__import__('os').system('id')"}, None),
    ({"template": "source_influence_battery", "params": {"n_items": "__import__('os').getcwd()"}}, None),
    ({"template": "source_influence_battery", "params": {"code": "__import__('os').system('id')"}}, None),
    ({"template": "source_influence_battery", "params": {"families": ["F1; import os"]}}, None),
    ({"template": "source_influence_battery", "params": {"n_items": 10 ** 9}}, None),
    ({"template": "source_influence_battery", "budget": {"max_trials": 10 ** 9}}, None),
    ({"template": "source_influence_battery", "budget": {"max_prompt_tokens": 2048}}, None),
    ({"template": "source_influence_battery", "eval": "1+1"}, None),
])
def test_specs_with_code_or_unbounded_values_are_rejected(client, spec, needle):
    r = client.post("/api/selflab/experiments", json={"spec": spec})
    assert r.status_code == 400, (spec, r.status_code, r.text[:200])
    assert client.get("/api/selflab/experiments").json() == []  # nothing was committed


def test_unregistered_but_schema_valid_template_is_refused(client):
    r = client.post("/api/selflab/experiments", json={"spec": {"template": "blinded_state"}})
    if r.status_code == 200:
        pytest.skip("blinded_state is registered now")
    assert r.status_code == 400 and "not registered" in r.text


def test_code_in_free_text_fields_is_stored_as_data(client, tmp_path):
    marker = tmp_path / "pwned"
    payload = f"__import__('os').system('touch {marker}')"
    r = client.post("/api/selflab/chat", json={"question": payload, "document": payload, "max_new_tokens": 4})
    assert r.status_code == 200
    assert payload in r.json()["messages"][-1]["content"] and not marker.exists()


# ------------------------------------------------------------------------------------------ loopback guard
def test_non_loopback_mutations_are_rejected_even_with_spoofed_forwarding(app):
    with TestClient(app, client=REMOTE) as c:
        for path, body in (("/api/selflab/chat", {"question": "q"}),
                           ("/api/selflab/experiments", {"spec": {"template": "source_influence_battery"}}),
                           ("/api/selflab/investigate/suggest", {"case_id": "x"})):
            for headers in ({}, {"X-Forwarded-For": "127.0.0.1"}, {"X-Real-IP": "127.0.0.1"},
                            {"Host": "127.0.0.1:8000"}, {"Forwarded": "for=127.0.0.1"}):
                assert c.post(path, json=body, headers=headers).status_code == 403, (path, headers)
        assert c.get("/api/selflab/status").status_code == 200  # read-only stays readable


def test_loopback_client_with_rebinding_host_is_rejected(client):
    r = client.post("/api/selflab/chat", json={"question": "q"}, headers={"Host": "attacker.example:8000"})
    assert r.status_code == 403
    r = client.post("/api/selflab/chat", json={"question": "q"}, headers={"Origin": "https://attacker.example"})
    assert r.status_code == 403


def test_token_allows_remote_and_wrong_token_does_not(app, monkeypatch):
    monkeypatch.setenv("NEURALSCOPE_API_TOKEN", "s3cret-token-for-tests")
    with TestClient(app, client=REMOTE) as c:
        bad = c.post("/api/selflab/chat", json={"question": "q", "max_new_tokens": 2},
                     headers={"X-NeuralScope-Token": "wrong"})
        assert bad.status_code == 403
        ok = c.post("/api/selflab/chat", json={"question": "q", "max_new_tokens": 2},
                    headers={"X-NeuralScope-Token": "s3cret-token-for-tests"})
        assert ok.status_code == 200


# ------------------------------------------------------------------------------------------ cancellation
def test_cancel_then_resume_via_api(client):
    spec = {"template": "source_influence_battery", "split": "debug_pilot", "seed": 3,
            "params": {"n_items": 2, "patching_subset": 0}}
    eid = client.post("/api/selflab/experiments", json={"spec": spec}).json()["id"]
    assert client.post(f"/api/selflab/experiments/{eid}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/selflab/experiments/{eid}/run").status_code == 409  # cancelled ≠ committed
    r = client.post(f"/api/selflab/experiments/{eid}/resume")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    done = _wait(client, eid)
    assert done["status"] == "done" and done["n_done"] == 12
    assert client.post(f"/api/selflab/experiments/{eid}/cancel").status_code == 409  # finished


# D-X3: found by SA6; fixed during validation (regression test).
def test_case_commit_without_replacement_is_a_400_not_a_500(app):
    with TestClient(app, client=LOCAL, raise_server_exceptions=False) as c:
        chat = c.post("/api/selflab/chat", json={"question": "What colour is the Zorblat?",
                                                 "document": "The Zorblat is teal.", "max_new_tokens": 8}).json()
        case = c.post("/api/selflab/cases", json={"run_id": chat["run_id"], "char_start": 0,
                                                  "char_end": max(1, len(chat["response_text"]))}).json()
        spec = next(t for t in c.get(f"/api/selflab/templates?case_id={case['id']}").json()
                    if t["id"] == "source_influence_case")["default_spec"]
        r = c.post("/api/selflab/experiments", json={"spec": spec})
        assert r.status_code == 400 and "replacement_text" in r.text, (r.status_code, r.text[:200])
