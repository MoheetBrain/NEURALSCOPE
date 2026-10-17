"""SA6 independent validation — persistence: append-only trials, immutable commitments, retained errors.

Store-level checks (selflab/store.py, SA4) that go around the class and talk to SQLite directly, because the
store's docstring claims the integrity rules are enforced "by the DATABASE itself". Engine-level resume checks
(cancel/crash → resume, no re-run, no re-randomisation) live in test_selflab_validation_engine.py.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

store_mod = pytest.importorskip("neuralscope.selflab.store")

from neuralscope.selflab.provenance import now_iso, sha256_json  # noqa: E402
from neuralscope.selflab.schemas import (  # noqa: E402
    CommittedSpec,
    ExperimentRecord,
    ExperimentSpec,
    Trial,
    TrialPlanItem,
)


@pytest.fixture()
def st(tmp_path):
    s = store_mod.SelfLabStore(tmp_path / "selflab.db")
    yield s
    s.close()


def _trial(i: int, status="ok", **kw) -> Trial:
    return Trial(id=store_mod.trial_id("exp_v", i), experiment_id="exp_v", index=i, item_id=f"it{i}",
                 group_id=f"it{i}", family="F1", condition="doc_v", status=status, created_at=now_iso(), **kw)


def _experiment(st) -> ExperimentRecord:
    spec = ExperimentSpec(template="source_influence_battery", seed=3)
    plan = [TrialPlanItem(index=i, item_id=f"it{i}", group_id=f"it{i}", family="F1", condition="doc_v")
            for i in range(3)]
    committed = CommittedSpec(spec=spec, spec_sha256=sha256_json(spec.model_dump(mode="json")),
                              plan_sha256=sha256_json([p.model_dump(mode="json") for p in plan]), n_planned=3,
                              committed_at=now_iso())
    rec = ExperimentRecord(id="exp_v", committed=committed, status="committed", created_at=now_iso())
    return st.insert_experiment(rec, plan)


def test_trials_are_append_only_through_the_class(st):
    t = _trial(0, output_text="red")
    st.insert_trial(t)
    with pytest.raises(store_mod.DuplicateTrialError):
        st.insert_trial(_trial(0, output_text="blue"))
    assert st.get_trial(t.id).output_text == "red"
    with pytest.raises(ValueError):  # id must be derived from (experiment, index): no aliasing
        st.insert_trial(_trial(1).model_copy(update={"id": "exp_v:00007"}))


@pytest.mark.parametrize("sql", ["UPDATE selflab_trials SET status='error'", "DELETE FROM selflab_trials",
                                 "UPDATE selflab_trials SET data='{}'"])
def test_trials_are_append_only_at_the_sqlite_level(st, sql):
    st.insert_trial(_trial(0, output_text="red"))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        st._conn.execute(sql)
    st._conn.rollback()
    assert st.get_trial(store_mod.trial_id("exp_v", 0)).output_text == "red"


# D-S1 (found by SA6, fixed by SA4 during validation): REPLACE used to bypass the append-only triggers.
def test_insert_or_replace_cannot_overwrite_a_trial(st):
    t = _trial(0, output_text="red")
    st.insert_trial(t)
    forged = t.model_copy(update={"output_text": "blue"}).model_dump(mode="json")
    try:
        st._conn.execute("INSERT OR REPLACE INTO selflab_trials (id, experiment_id, idx, status, created_at, "
                         "data_sha256, data) VALUES (?,?,?,?,?,?,?)",
                         (t.id, "exp_v", 0, "ok", t.created_at, sha256_json(forged), json.dumps(forged)))
        st._conn.commit()
    except sqlite3.DatabaseError:
        st._conn.rollback()
    assert st.get_trial(t.id).output_text == "red"


def test_error_trials_are_retained_and_counted(st):
    st.insert_trial(_trial(0))
    st.insert_trial(_trial(1, status="error", error="RuntimeError: boom"))
    st.insert_trial(_trial(2, status="malformed", output_text="??"))
    got = st.list_trials("exp_v")
    assert [t.status for t in got] == ["ok", "error", "malformed"]
    assert got[1].error == "RuntimeError: boom"
    assert st.trial_counts("exp_v") == {"ok": 1, "error": 1, "malformed": 1}


def test_stored_trial_hash_matches_its_canonical_json(st):
    t = _trial(0, output_text="red", parsed={"letter": "A"}, measurements={"v": {"sum_logprob": -1.25}})
    st.insert_trial(t)
    row = st.trial_rows("exp_v")[0]
    assert row["data_sha256"] == sha256_json(st.get_trial(t.id).model_dump(mode="json"))


def test_committed_spec_and_plan_are_immutable(st):
    rec = _experiment(st)
    for col, val in (("committed", "{}"), ("plan", "[]"), ("plan_sha256", "0" * 64), ("spec_sha256", "f" * 64)):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            st._conn.execute(f"UPDATE selflab_experiments SET {col}=? WHERE id=?", (val, rec.id))
        st._conn.rollback()
    with pytest.raises(ValueError):
        st.update_experiment(rec.id, committed=rec.committed)
    # status may change; the commitment read back is the stored one
    st.update_experiment(rec.id, status="queued")
    back = st.require_experiment(rec.id)
    assert back.status == "queued" and back.committed.plan_sha256 == rec.committed.plan_sha256
    assert [p.index for p in st.get_plan(rec.id)] == [0, 1, 2]
