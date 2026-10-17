"""SA6 independent validation — persistence and resume with the REAL Experiment A template (puppet target).

SA4's engine tests use a fake template. Here the real ``source_influence_battery`` runs (seeded label orders,
tokenizer-checked items, candidate scoring), so "no re-run, no re-randomisation" is checked on the data that
matters: completed trial rows must be byte-identical after a crash/cancel and resume, every plan index must run
exactly once, every trial must have been read under the label order its committed plan item fixed, and the plan
hash must be reproducible in a fresh interpreter with a different hash seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

si = pytest.importorskip("neuralscope.selflab.templates.source_influence")
engine_mod = pytest.importorskip("neuralscope.selflab.engine")
store_mod = pytest.importorskip("neuralscope.selflab.store")

from neuralscope.experiments.jobs import JobManager  # noqa: E402
from neuralscope.selflab import registry  # noqa: E402
from neuralscope.selflab.schemas import ExperimentSpec  # noqa: E402
from tests.test_selflab_validation_puppet import Conversation, make_puppet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Crash(BaseException):
    """Simulates the process dying mid-trial (not an Exception, so nothing converts it into an error trial)."""


def _policy(conv: Conversation):
    return "A" if len(conv.users) > 1 else "insufficient evidence"


@pytest.fixture(scope="module")
def puppet():
    return make_puppet(_policy)


SPEC = {"template": "source_influence_battery", "split": "discovery", "seed": 5,
        "params": {"n_items": 3, "patching_subset": 0, "bootstrap_B": 200, "n_permutations": 500,
                   "min_units_for_status": 2}}


def _rows(st, eid):
    with st._lock:
        rows = st._conn.execute("SELECT idx, data_sha256, data FROM selflab_trials WHERE experiment_id=? ORDER BY idx",
                                (eid,)).fetchall()
    return {int(r["idx"]): (r["data_sha256"], r["data"]) for r in rows}


@pytest.fixture()
def battery_spy(monkeypatch):
    registry._load_builtin()
    T = registry.get_template("source_influence_battery")
    real = type(T).run_trial
    spy = {"calls": [], "crash_at": None, "on_call": None}

    def run_trial(item, ctx):
        spy["calls"].append(item.index)
        if spy["on_call"]:
            spy["on_call"](item, ctx)
        if spy["crash_at"] is not None and item.index == spy["crash_at"]:
            raise _Crash()
        return real(T, item, ctx)

    monkeypatch.setattr(T, "run_trial", run_trial)
    return spy


def _check_label_orders(st, eid):
    plan = {p.index: p for p in st.get_plan(eid)}
    for t in st.list_trials(eid):
        rep = t.measurements["source_report"]
        assert rep[0]["label_order"] == plan[t.index].label_order
        assert rep[1]["label_order"] == list(reversed(plan[t.index].label_order))


def test_crash_then_resume_keeps_completed_trials_byte_identical(puppet, battery_spy, tmp_path):
    a, _ = puppet
    db = tmp_path / "sl.db"
    st = store_mod.SelfLabStore(db)
    e1 = engine_mod.SelfLabEngine(st, JobManager(), adapter=a)
    rec = e1.commit(ExperimentSpec.model_validate(SPEC))
    n = rec.committed.n_planned
    assert n == 18
    battery_spy["crash_at"] = 7
    with pytest.raises(_Crash):
        e1.run_now(rec.id)
    assert st.require_experiment(rec.id).status == "running"  # the "process" died mid-trial
    before = _rows(st, rec.id)
    assert sorted(before) == list(range(7))

    st2 = store_mod.SelfLabStore(db)  # restart: new connection, new engine
    e2 = engine_mod.SelfLabEngine(st2, JobManager(), adapter=a)
    assert e2.recover_interrupted() == [rec.id]
    battery_spy["crash_at"] = None
    battery_spy["calls"].clear()
    final = e2.run_now(rec.id)
    assert final.status == "done" and final.n_done == n
    assert battery_spy["calls"] == list(range(7, n))  # nothing re-run
    after = _rows(st2, rec.id)
    assert {i: after[i] for i in before} == before  # byte-identical rows (data + stored hash)
    assert sorted(after) == list(range(n))
    assert st2.require_experiment(rec.id).committed.plan_sha256 == rec.committed.plan_sha256
    _check_label_orders(st2, rec.id)
    ev = [e["event"] for e in st2.list_events(rec.id)]
    assert ev.count("committed") == 1 and "interrupted" in ev and "resumed" in ev


def test_cancel_then_resume_runs_each_index_once(puppet, battery_spy, tmp_path):
    a, _ = puppet
    st = store_mod.SelfLabStore(tmp_path / "sl.db")
    eng = engine_mod.SelfLabEngine(st, JobManager(), adapter=a)
    rec = eng.commit(ExperimentSpec.model_validate(SPEC))

    def cancel_at_4(item, ctx):
        if item.index == 4:
            eng.cancel(rec.id)

    battery_spy["on_call"] = cancel_at_4
    r = eng.run_now(rec.id)
    assert r.status == "cancelled" and r.n_done == 5  # the trial in flight finishes; the next one never starts
    first = _rows(st, rec.id)
    battery_spy["on_call"] = None
    r = eng.run_now(rec.id)
    assert r.status == "done" and r.n_done == 18
    assert battery_spy["calls"] == list(range(18))
    after = _rows(st, rec.id)
    assert {i: after[i] for i in first} == first
    _check_label_orders(st, rec.id)
    assert r.analysis["complete"] is True and r.analysis["n_trials_analyzed"] == 18


def test_errors_are_retained_through_resume(puppet, battery_spy, tmp_path):
    a, _ = puppet
    st = store_mod.SelfLabStore(tmp_path / "sl.db")
    eng = engine_mod.SelfLabEngine(st, JobManager(), adapter=a)
    rec = eng.commit(ExperimentSpec.model_validate(SPEC))

    def boom(item, ctx):
        if item.index == 2:
            raise RuntimeError("synthetic failure")

    battery_spy["on_call"] = boom
    r = eng.run_now(rec.id)
    assert r.status == "done" and r.n_errors == 1
    t = st.get_trial(store_mod.trial_id(rec.id, 2))
    assert t.status == "error" and "synthetic failure" in t.error
    assert r.analysis["summary"]["counts"]["error"] == 1


def test_plan_hash_is_reproducible_in_a_fresh_interpreter(puppet):
    """build_plan must not depend on per-process state (hash randomisation, set order, time)."""
    a, _ = puppet
    T = registry.get_template("source_influence_battery")
    spec = ExperimentSpec.model_validate(SPEC)
    spec = spec.model_copy(update={"params": T.params_model.model_validate(spec.params).model_dump(mode="json")})
    here = engine_mod.plan_hash(T.build_plan(spec, None, a))
    code = (
        "import json,sys\n"
        "from neuralscope.instrumentation.demo import build_demo_adapter\n"
        "from neuralscope.selflab.registry import get_template\n"
        "from neuralscope.selflab.engine import plan_hash\n"
        "from neuralscope.selflab.schemas import ExperimentSpec\n"
        "from tests.conftest import DEMO_TOKENIZER\n"
        "spec = ExperimentSpec.model_validate(json.loads(sys.argv[1]))\n"
        "a = build_demo_adapter(DEMO_TOKENIZER)\n"
        "print(plan_hash(get_template(spec.template).build_plan(spec, None, a)))\n")
    for hs in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        out = subprocess.run([sys.executable, "-c", code, json.dumps(spec.model_dump(mode="json"))], cwd=ROOT,
                             env=env, capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == here


# Regression test for D-E1 (fixed 2026-10-17 during coordinator integration; previously a strict xfail).
def test_resume_refuses_different_weights_with_identical_metadata(puppet, tmp_path):
    from neuralscope.instrumentation.demo import build_demo_adapter

    from tests.conftest import DEMO_TOKENIZER

    a, _ = puppet
    st = store_mod.SelfLabStore(tmp_path / "sl.db")
    rec = engine_mod.SelfLabEngine(st, JobManager(), adapter=a).commit(ExperimentSpec.model_validate(SPEC))
    st.update_experiment(rec.id, status="cancelled")
    other = build_demo_adapter(DEMO_TOKENIZER, seed=1)  # different random weights, identical metadata
    with pytest.raises(store_mod.ConflictError):
        engine_mod.SelfLabEngine(st, JobManager(), adapter=other).verify_resumable(rec.id)
