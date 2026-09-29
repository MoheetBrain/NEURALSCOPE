"""HTTP API: schemas, validation, full flow (fixture model)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from neuralscope_api.main import create_app
from tests.conftest import PROMPT


@pytest.fixture()
def client(demo_adapter, settings):
    app = create_app(settings, adapter=demo_adapter)
    with TestClient(app) as c:
        yield c


def _wait(client, job, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job)


def test_health_and_model(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ready" and h["is_demo"] is True
    m = client.get("/api/model").json()
    assert m["n_depths"] == m["n_layers"] + 1 and m["capabilities"]["head_ablation"] is True


def test_tokenize(client):
    r = client.post("/api/tokenize", json={"prompt": PROMPT}).json()
    assert r["n_tokens"] == len(r["tokens"]) and r["status"] == "observed"
    assert "".join(t["text"] for t in r["tokens"] if t["role"] == "user") == PROMPT


def test_validation_errors(client):
    assert client.post("/api/runs", json={"prompt": ""}).status_code == 422
    assert client.post("/api/runs", json={"prompt": "hi", "max_new_tokens": 0}).status_code == 422
    assert client.get("/api/runs/run_doesnotexist").status_code == 404
    bad = {"base_run_id": "x", "interventions": [{"type": "head_ablation", "layer": -1, "head": 0}]}
    assert client.post("/api/interventions", json=bad).status_code == 422
    unknown = {"base_run_id": "x", "interventions": [{"type": "telepathy"}]}
    assert client.post("/api/interventions", json=unknown).status_code == 422


def test_full_flow(client):
    run = client.post("/api/runs", json={"prompt": PROMPT, "max_new_tokens": 4}).json()
    rid = run["id"]
    assert run["tokens"] and run["is_demo"] is True
    assert client.get(f"/api/runs/{rid}/tokens").json() == run["tokens"]
    layers = client.get(f"/api/runs/{rid}/layers").json()
    assert layers["depth_labels"][0] == "emb"
    m = client.get(f"/api/runs/{rid}/matrix", params={"metric": "target_rank"}).json()
    assert m["status"] == "derived"
    assert client.get(f"/api/runs/{rid}/matrix", params={"metric": "nope"}).status_code == 400
    pos = run["target"]["position"]
    assert client.get(f"/api/runs/{rid}/cell/2/{pos}").json()["depth"] == 2
    assert client.get(f"/api/runs/{rid}/cell/99/{pos}").status_code == 400
    assert len(client.get(f"/api/runs/{rid}/logit-lens").json()["depths"]) == layers["n_layers"] + 1
    att = client.get(f"/api/runs/{rid}/attention/1").json()
    assert att["status"]["patterns"] == "observed"
    assert client.get(f"/api/runs/{rid}/heads/1/0").json()["layer"] == 1
    assert client.get(f"/api/runs/{rid}/dla").json()["decomposition_error"] < 1e-3
    tgt = client.post(f"/api/runs/{rid}/target", json={"target_text": "95", "target_position": pos}).json()
    assert tgt["token"] == "9" and "tokenises to 2 tokens" in tgt["note"]

    iv = client.post("/api/interventions", json={
        "base_run_id": rid, "interventions": [{"type": "head_ablation", "layer": 2, "head": 1}]}).json()
    assert iv["comparison"]["status"] == "causal" and iv["run"]["parent_run_id"] == rid
    cmp = client.post("/api/compare", json={"run_a": rid, "run_b": iv["run"]["id"]}).json()
    assert cmp["mode"] == "intervention"

    job = client.post("/api/sweeps/heads", json={"run_id": rid}).json()
    assert job["status"] in ("queued", "running", "done")
    j = _wait(client, job)
    assert j["status"] == "done" and j["result"]["status"] == "causal"

    exp = client.post("/api/experiments", json={"name": "demo exp", "run_ids": [rid, iv["run"]["id"]],
                                                "notes": "n"}).json()
    got = client.get(f"/api/experiments/{exp['id']}").json()
    assert len(got["runs"]) == 2
    patched = client.patch(f"/api/experiments/{exp['id']}", json={"notes": "updated"}).json()
    assert patched["notes"] == "updated" and patched["created_at"] == exp["created_at"]
    assert any(e["id"] == exp["id"] for e in client.get("/api/experiments").json())
    assert client.post("/api/experiments", json={"name": "x", "run_ids": ["run_missing"]}).status_code == 400

    runs = client.get("/api/runs").json()
    assert {r["id"] for r in runs} >= {rid, iv["run"]["id"]}
    upd = client.patch(f"/api/runs/{rid}", json={"notes": "carry?"}).json()
    assert upd["notes"] == "carry?"
    assert client.delete(f"/api/runs/{iv['run']['id']}").status_code == 200


def test_math_endpoints(client):
    v = client.post("/api/math/verify", json={"expression": "73*14", "answer": "1,022"}).json()
    assert v["correct"] is True and v["oracle_answer"] == "1022"
    assert client.post("/api/math/verify", json={"expression": "__import__('os')", "answer": "1"}).status_code == 400
    probs = client.post("/api/math/problems", json={"n": 5, "seed": 1}).json()
    assert len(probs) == 5 and all("oracle_answer" in p for p in probs)
    r = client.post("/api/math/run", json={"kind": "add", "operands": {"a": 37, "b": 58}}).json()
    assert r["math"]["problem"]["oracle_answer"] == "95" and r["kind"] == "math"
    job = client.post("/api/math/benchmarks", json={"problems": [{"kind": "add", "operands": {"a": 12, "b": 34}},
                                                                 {"kind": "mul", "operands": {"a": 7, "b": 8}}]}).json()
    j = _wait(client, job)
    assert j["status"] == "done" and j["result"]["summary"]["n"] == 2
    bid = j["result"]["id"]
    assert client.get(f"/api/math/benchmarks/{bid}").json()["id"] == bid
    ev = _wait(client, client.post(f"/api/math/benchmarks/{bid}/intervention-eval",
                                   json={"interventions": [{"type": "head_ablation", "layer": 1, "head": 0}]}).json())
    assert ev["status"] == "done" and ev["result"]["n"] == 2
    bad = client.post("/api/math/benchmarks", json={"problems": [{"kind": "add"}]})
    assert bad.status_code == 400
