"""SelfLab store: append-only trials, immutable commitments, same SQLite file as the ExperimentStore."""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from neuralscope.experiments.store import ExperimentStore
from neuralscope.selflab.provenance import now_iso, sha256_json
from neuralscope.selflab.schemas import (
    Case,
    ClaimSpan,
    CommittedSpec,
    EvidenceItem,
    ExperimentRecord,
    ExperimentSpec,
    Message,
    Trial,
    TrialPlanItem,
)
from neuralscope.selflab.store import (
    ConflictError,
    DuplicateTrialError,
    NotFoundError,
    SelfLabStore,
    trial_id,
)


@pytest.fixture()
def sstore(settings) -> SelfLabStore:
    return SelfLabStore(settings.db_path)


def _case(cid: str = "case_1") -> Case:
    return Case(id=cid, created_at=now_iso(), title="t", access_level="white_box_local", run_id="run_x",
                messages=[Message(role="user", content="Document:\n\nThe Zorblat is teal.\n\nWhat colour?")],
                response_text="Teal, according to the document.",
                claim=ClaimSpan(text="according to the document", char_start=6, char_end=32),
                claim_type="source_attribution", is_fixture=True)


def _experiment(eid: str = "slx_1", n: int = 3, case_id: str | None = None) -> tuple[ExperimentRecord, list]:
    spec = ExperimentSpec(template="blinded_state", seed=3, params={"n": n})
    plan = [TrialPlanItem(index=i, item_id=f"i{i}", group_id=f"g{i}", family="f", condition="c") for i in range(n)]
    c = CommittedSpec(spec=spec, spec_sha256=sha256_json(spec.model_dump(mode="json")),
                      plan_sha256=sha256_json([p.model_dump(mode="json") for p in plan]), n_planned=n,
                      committed_at=now_iso())
    return ExperimentRecord(id=eid, case_id=case_id, committed=c, status="committed", created_at=now_iso()), plan


def _trial(eid: str, i: int, status: str = "ok", text: str = "out") -> Trial:
    return Trial(id=trial_id(eid, i), experiment_id=eid, index=i, item_id=f"i{i}", group_id=f"g{i}", family="f",
                 condition="c", status=status, output_text=text, created_at=now_iso())


def test_trial_id_format():
    assert trial_id("slx_abc", 7) == "slx_abc:00007"


def test_shares_the_experiment_store_database(settings, sstore):
    es = ExperimentStore(settings.data_dir)
    assert sstore.db_path == es.db_path == settings.db_path
    with sqlite3.connect(settings.db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "experiments", "benchmarks"} <= names
    assert {"selflab_cases", "selflab_experiments", "selflab_trials", "selflab_evidence", "selflab_events",
            "selflab_chats", "selflab_trial_runs"} <= names


def test_case_roundtrip_and_immutable_fields(sstore):
    c = sstore.insert_case(_case())
    assert sstore.get_case(c.id) == c
    with pytest.raises(ConflictError):
        sstore.insert_case(_case())
    sstore.add_case_experiment(c.id, "slx_1")
    sstore.add_case_experiment(c.id, "slx_1")  # idempotent
    assert sstore.require_case(c.id).experiment_ids == ["slx_1"]
    with pytest.raises(ValueError):
        sstore.update_case(c.id, response_text="rewritten")
    with pytest.raises(NotFoundError):
        sstore.require_case("case_missing")
    assert [x.id for x in sstore.list_cases()] == [c.id]


def test_trials_are_append_only(sstore):
    rec, plan = _experiment()
    sstore.insert_experiment(rec, plan)
    t0 = sstore.insert_trial(_trial(rec.id, 0, text="first"))
    with pytest.raises(DuplicateTrialError):
        sstore.insert_trial(_trial(rec.id, 0, text="overwrite attempt"))
    assert sstore.get_trial(t0.id).output_text == "first"  # never overwritten
    # the database itself refuses UPDATE / DELETE, even through raw SQL
    with sqlite3.connect(sstore.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE selflab_trials SET status='error' WHERE id=?", (t0.id,))
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM selflab_trials WHERE id=?", (t0.id,))
    assert sstore.get_trial(t0.id) == t0
    with pytest.raises(ValueError):  # ids must follow f"{experiment_id}:{index:05d}"
        sstore.insert_trial(_trial(rec.id, 1).model_copy(update={"id": "custom"}))


def test_trial_listing_counts_and_integrity_hash(sstore):
    rec, plan = _experiment(n=4)
    sstore.insert_experiment(rec, plan)
    for i, st in [(2, "ok"), (0, "error"), (1, "ok")]:
        sstore.insert_trial(_trial(rec.id, i, status=st))
    assert [t.index for t in sstore.list_trials(rec.id)] == [0, 1, 2]
    assert [t.index for t in sstore.list_trials(rec.id, offset=1, limit=1)] == [1]
    assert sstore.trial_indices(rec.id) == {0, 1, 2}
    assert sstore.trial_counts(rec.id) == {"ok": 2, "error": 1}
    for row, t in zip(sstore.trial_rows(rec.id), sstore.list_trials(rec.id)):
        assert row["data_sha256"] == sha256_json(t.model_dump(mode="json"))


def test_committed_spec_is_immutable(sstore):
    rec, plan = _experiment()
    sstore.insert_experiment(rec, plan)
    with pytest.raises(ValueError):
        sstore.update_experiment(rec.id, committed=rec.committed)
    with sqlite3.connect(sstore.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE selflab_experiments SET plan_sha256='x' WHERE id=?", (rec.id,))
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE selflab_experiments SET plan='[]' WHERE id=?", (rec.id,))
    r = sstore.update_experiment(rec.id, status="queued", progress=0.5)
    assert r.status == "queued" and r.committed == rec.committed
    assert sstore.get_plan(rec.id) == plan
    with pytest.raises(ConflictError):
        sstore.transition(rec.id, {"committed"}, status="running")
    with pytest.raises(ConflictError):
        sstore.insert_experiment(rec, plan)


def test_evidence_events_and_links_append_only(sstore):
    rec, plan = _experiment(case_id="case_1")
    sstore.insert_case(_case())
    sstore.insert_experiment(rec, plan)
    ev = EvidenceItem(id="ev1", case_id="case_1", experiment_id=rec.id, kind="direct_observation",
                      status="inconclusive", title="t", statement="s", created_at=now_iso())
    sstore.insert_evidence(ev)
    with pytest.raises(ConflictError):
        sstore.insert_evidence(ev)
    assert sstore.list_evidence(case_id="case_1") == [ev]
    assert sstore.list_evidence(experiment_id=rec.id) == [ev]
    assert sstore.list_evidence(case_id="other") == []
    sstore.log_event(rec.id, "committed", {"a": 1})
    sstore.log_event(rec.id, "queued")
    assert [e["event"] for e in sstore.list_events(rec.id)] == ["committed", "queued"]
    sstore.link_trial_run(trial_id(rec.id, 0), "run_1", {"reproduced": True})
    assert sstore.trial_runs(trial_id(rec.id, 0))[0]["run_id"] == "run_1"
    assert sstore.experiment_trial_runs(rec.id)[0]["reproduced"] is True
    with sqlite3.connect(sstore.db_path) as conn:
        for sql in ("DELETE FROM selflab_evidence", "UPDATE selflab_events SET event='x'", "DELETE FROM selflab_events"):
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                conn.execute(sql)


def test_concurrent_inserts_exactly_one_wins(sstore):
    rec, plan = _experiment(n=40)
    sstore.insert_experiment(rec, plan)
    results: list[str] = []
    lock = threading.Lock()

    def worker(i: int, tag: str) -> None:
        try:
            sstore.insert_trial(_trial(rec.id, i, text=tag))
            r = "ok"
        except DuplicateTrialError:
            r = "dup"
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i % 20, f"t{k}")) for k, i in enumerate(range(40))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 20 and results.count("dup") == 20
    assert sstore.trial_indices(rec.id) == set(range(20))


def test_persists_across_instances(settings, sstore):
    rec, plan = _experiment()
    sstore.insert_experiment(rec, plan)
    sstore.insert_trial(_trial(rec.id, 0))
    sstore.insert_chat("run_1", {"response_text": "hi"})
    sstore.close()
    s2 = SelfLabStore(settings.db_path)  # simulated restart
    assert s2.require_experiment(rec.id).committed == rec.committed
    assert s2.trial_indices(rec.id) == {0}
    assert s2.get_chat("run_1") == {"response_text": "hi"}
    with pytest.raises(DuplicateTrialError):
        s2.insert_trial(_trial(rec.id, 0, text="again"))


def test_insert_or_replace_and_deletes_are_refused_by_the_database(sstore):
    """REPLACE's implicit delete does not fire delete triggers, so BEFORE INSERT triggers guard every key."""
    sstore.insert_case(_case())
    rec, plan = _experiment(case_id="case_1")
    sstore.insert_experiment(rec, plan)
    t = sstore.insert_trial(_trial(rec.id, 0, text="red"))
    ev = sstore.insert_evidence(EvidenceItem(id="ev1", experiment_id=rec.id, kind="direct_observation",
                                             status="inconclusive", title="t", statement="s", created_at=now_iso()))
    forged = t.model_copy(update={"output_text": "blue"}).model_dump(mode="json")
    with sqlite3.connect(sstore.db_path) as conn:  # a raw connection: recursive_triggers is OFF here
        attempts = [
            ("INSERT OR REPLACE INTO selflab_trials (id, experiment_id, idx, status, created_at, data_sha256, data) "
             "VALUES (?,?,?,?,?,?,?)", (t.id, rec.id, 0, "ok", t.created_at, sha256_json(forged), json.dumps(forged))),
            ("INSERT OR REPLACE INTO selflab_trials (id, experiment_id, idx, status, created_at, data_sha256, data) "
             "VALUES (?,?,?,?,?,?,?)", ("other-id", rec.id, 0, "ok", t.created_at, "x", json.dumps(forged))),
            ("INSERT OR REPLACE INTO selflab_evidence (id, experiment_id, case_id, kind, status, created_at, data) "
             "VALUES (?,?,?,?,?,?,?)", (ev.id, rec.id, None, "model_report", "supported", "now", "{}")),
            ("INSERT OR REPLACE INTO selflab_experiments (id, created_at, updated_at, case_id, template, status, "
             "spec_sha256, plan_sha256, committed, plan, data) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
             (rec.id, "now", "now", None, "x", "done", "a", "b", "{}", "[]", "{}")),
            ("INSERT OR REPLACE INTO selflab_cases (id, created_at, run_id, access_level, claim_type, is_fixture, data) "
             "VALUES (?,?,?,?,?,?,?)", ("case_1", "now", None, "api_only", "other", 0, "{}")),
            ("DELETE FROM selflab_experiments", ()),
            ("DELETE FROM selflab_cases", ()),
            ("UPDATE selflab_cases SET data = json_set(data, '$.response_text', 'rewritten')", ()),
        ]
        for sql, args in attempts:
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(sql, args)
        conn.rollback()
    assert sstore.get_trial(t.id).output_text == "red" and sstore.trial_indices(rec.id) == {0}
    assert sstore.get_evidence("ev1") == ev
    assert sstore.require_experiment(rec.id).committed == rec.committed
    assert sstore.require_case("case_1").response_text == _case().response_text
    sstore.update_case("case_1", title="renamed", notes="ok")  # the mutable fields still change
    assert sstore.require_case("case_1").title == "renamed"
