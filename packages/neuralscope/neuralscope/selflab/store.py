"""SelfLab persistence: cases, committed experiments, append-only trials, evidence, Lab-run links.

Tables live in the SAME SQLite file as the ``ExperimentStore`` (``settings.db_path``), all prefixed
``selflab_``. Every persisted object is a pydantic model from ``selflab/schemas.py`` serialised as JSON
(``model_dump_json``) plus a few indexed scalar columns for querying.

Integrity rules enforced by the DATABASE itself (SQLite triggers), not only by this class:

* ``selflab_trials`` is append-only: one row per ``(experiment_id, idx)``; a duplicate insert raises
  ``DuplicateTrialError``; UPDATE, DELETE and ``INSERT OR REPLACE`` are rejected by triggers (a BEFORE
  INSERT trigger rejects an existing key, because REPLACE's implicit delete does not fire delete
  triggers). A completed trial can never be overwritten, re-run or re-randomised.
* ``selflab_experiments``: the commitment (committed spec, plan, hashes) is immutable once written; an
  experiment cannot be replaced or deleted. Status/progress/cost/analysis fields are updated in place.
* ``selflab_cases``: context, response, claim, receipts and provenance are immutable; only
  ``experiment_ids``, ``notes`` and ``title`` change. Cases cannot be replaced or deleted.
* ``selflab_evidence``, ``selflab_events``, ``selflab_trial_runs`` and ``selflab_chats`` are append-only.

Each trial row also stores ``data_sha256`` (sha256 of its canonical JSON) so an export can be checked
against the database. Hashes support integrity checks of local records; they do not make a local
file tamper-proof (anyone with write access to the file can rewrite it).

Thread-safe: one connection (``check_same_thread=False``) guarded by an ``RLock``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from neuralscope.selflab.provenance import now_iso, sha256_json
from neuralscope.selflab.schemas import (
    Case,
    CommittedSpec,
    EvidenceItem,
    ExperimentRecord,
    Trial,
    TrialPlanItem,
)

STORE_VERSION = "selflab-store-1"


# ------------------------------------------------------------------------------------------ errors
class SelfLabError(Exception):
    """Base error. ``status_code`` is the HTTP status the API maps it to."""

    status_code = 400


class NotFoundError(SelfLabError):
    status_code = 404


class ConflictError(SelfLabError):
    """Invalid state transition (e.g. running an experiment that is already done)."""

    status_code = 409


class ForbiddenError(SelfLabError):
    status_code = 403


class UnavailableError(SelfLabError):
    status_code = 503


class DuplicateTrialError(ConflictError):
    """A trial with this (experiment_id, index) is already stored. Trials are append-only."""


# ------------------------------------------------------------------------------------------ schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS selflab_cases (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    run_id TEXT,
    access_level TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    is_fixture INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selflab_cases_created ON selflab_cases(created_at);

CREATE TABLE IF NOT EXISTS selflab_chats (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS selflab_experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    case_id TEXT,
    template TEXT NOT NULL,
    status TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    committed TEXT NOT NULL,
    plan TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selflab_experiments_case ON selflab_experiments(case_id);
CREATE INDEX IF NOT EXISTS selflab_experiments_created ON selflab_experiments(created_at);

CREATE TABLE IF NOT EXISTS selflab_trials (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data_sha256 TEXT NOT NULL,
    data TEXT NOT NULL,
    UNIQUE (experiment_id, idx)
);
CREATE INDEX IF NOT EXISTS selflab_trials_exp ON selflab_trials(experiment_id, idx);

CREATE TABLE IF NOT EXISTS selflab_evidence (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    case_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selflab_evidence_exp ON selflab_evidence(experiment_id);
CREATE INDEX IF NOT EXISTS selflab_evidence_case ON selflab_evidence(case_id);

CREATE TABLE IF NOT EXISTS selflab_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selflab_events_exp ON selflab_events(experiment_id, seq);

CREATE TABLE IF NOT EXISTS selflab_trial_runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selflab_trial_runs_trial ON selflab_trial_runs(trial_id);

CREATE TRIGGER IF NOT EXISTS selflab_trials_no_update BEFORE UPDATE ON selflab_trials
BEGIN SELECT RAISE(ABORT, 'selflab_trials is append-only: a stored trial cannot be modified'); END;
CREATE TRIGGER IF NOT EXISTS selflab_trials_no_delete BEFORE DELETE ON selflab_trials
BEGIN SELECT RAISE(ABORT, 'selflab_trials is append-only: a stored trial cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS selflab_experiments_commit_immutable
BEFORE UPDATE OF committed, plan, spec_sha256, plan_sha256 ON selflab_experiments
WHEN OLD.committed IS NOT NEW.committed OR OLD.plan IS NOT NEW.plan
  OR OLD.spec_sha256 IS NOT NEW.spec_sha256 OR OLD.plan_sha256 IS NOT NEW.plan_sha256
BEGIN SELECT RAISE(ABORT, 'a committed experiment spec/plan is immutable'); END;
CREATE TRIGGER IF NOT EXISTS selflab_evidence_no_update BEFORE UPDATE ON selflab_evidence
BEGIN SELECT RAISE(ABORT, 'selflab_evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_evidence_no_delete BEFORE DELETE ON selflab_evidence
BEGIN SELECT RAISE(ABORT, 'selflab_evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_events_no_update BEFORE UPDATE ON selflab_events
BEGIN SELECT RAISE(ABORT, 'selflab_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_events_no_delete BEFORE DELETE ON selflab_events
BEGIN SELECT RAISE(ABORT, 'selflab_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_trial_runs_no_update BEFORE UPDATE ON selflab_trial_runs
BEGIN SELECT RAISE(ABORT, 'selflab_trial_runs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_trial_runs_no_delete BEFORE DELETE ON selflab_trial_runs
BEGIN SELECT RAISE(ABORT, 'selflab_trial_runs is append-only'); END;

-- "INSERT OR REPLACE" deletes the conflicting row WITHOUT firing delete triggers (recursive_triggers is off by
-- default), so every append-only / immutable table also rejects an insert whose key already exists.
CREATE TRIGGER IF NOT EXISTS selflab_trials_no_replace BEFORE INSERT ON selflab_trials
WHEN EXISTS (SELECT 1 FROM selflab_trials WHERE id = NEW.id OR (experiment_id = NEW.experiment_id AND idx = NEW.idx))
BEGIN SELECT RAISE(ABORT, 'selflab_trials is append-only: this trial is already stored'); END;
CREATE TRIGGER IF NOT EXISTS selflab_evidence_no_replace BEFORE INSERT ON selflab_evidence
WHEN EXISTS (SELECT 1 FROM selflab_evidence WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'selflab_evidence is append-only: this evidence id is already stored'); END;
CREATE TRIGGER IF NOT EXISTS selflab_events_no_replace BEFORE INSERT ON selflab_events
WHEN EXISTS (SELECT 1 FROM selflab_events WHERE seq = NEW.seq)
BEGIN SELECT RAISE(ABORT, 'selflab_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_trial_runs_no_replace BEFORE INSERT ON selflab_trial_runs
WHEN EXISTS (SELECT 1 FROM selflab_trial_runs WHERE seq = NEW.seq)
BEGIN SELECT RAISE(ABORT, 'selflab_trial_runs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_experiments_no_replace BEFORE INSERT ON selflab_experiments
WHEN EXISTS (SELECT 1 FROM selflab_experiments WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'experiment already committed: a commitment cannot be replaced'); END;
CREATE TRIGGER IF NOT EXISTS selflab_experiments_no_delete BEFORE DELETE ON selflab_experiments
BEGIN SELECT RAISE(ABORT, 'a committed experiment cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS selflab_cases_no_replace BEFORE INSERT ON selflab_cases
WHEN EXISTS (SELECT 1 FROM selflab_cases WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'case already exists: a case cannot be replaced'); END;
CREATE TRIGGER IF NOT EXISTS selflab_cases_no_delete BEFORE DELETE ON selflab_cases
BEGIN SELECT RAISE(ABORT, 'a case cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS selflab_cases_immutable BEFORE UPDATE ON selflab_cases
WHEN OLD.id IS NOT NEW.id OR OLD.created_at IS NOT NEW.created_at OR OLD.run_id IS NOT NEW.run_id
  OR OLD.access_level IS NOT NEW.access_level OR OLD.claim_type IS NOT NEW.claim_type
  OR OLD.is_fixture IS NOT NEW.is_fixture
  OR json_extract(OLD.data, '$.messages') IS NOT json_extract(NEW.data, '$.messages')
  OR json_extract(OLD.data, '$.response_text') IS NOT json_extract(NEW.data, '$.response_text')
  OR json_extract(OLD.data, '$.claim') IS NOT json_extract(NEW.data, '$.claim')
  OR json_extract(OLD.data, '$.claim_type') IS NOT json_extract(NEW.data, '$.claim_type')
  OR json_extract(OLD.data, '$.segments') IS NOT json_extract(NEW.data, '$.segments')
  OR json_extract(OLD.data, '$.receipts') IS NOT json_extract(NEW.data, '$.receipts')
  OR json_extract(OLD.data, '$.provenance') IS NOT json_extract(NEW.data, '$.provenance')
  OR json_extract(OLD.data, '$.access_level') IS NOT json_extract(NEW.data, '$.access_level')
  OR json_extract(OLD.data, '$.run_id') IS NOT json_extract(NEW.data, '$.run_id')
BEGIN SELECT RAISE(ABORT, 'a case''s context, response, claim, receipts and provenance are immutable'); END;
CREATE TRIGGER IF NOT EXISTS selflab_chats_no_replace BEFORE INSERT ON selflab_chats
WHEN EXISTS (SELECT 1 FROM selflab_chats WHERE run_id = NEW.run_id)
BEGIN SELECT RAISE(ABORT, 'selflab_chats is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_chats_no_update BEFORE UPDATE ON selflab_chats
BEGIN SELECT RAISE(ABORT, 'selflab_chats is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selflab_chats_no_delete BEFORE DELETE ON selflab_chats
BEGIN SELECT RAISE(ABORT, 'selflab_chats is append-only'); END;
"""

# Mutable fields of an ExperimentRecord (everything else is fixed at commit).
_MUTABLE_EXPERIMENT_FIELDS = {
    "status", "progress", "message", "n_done", "n_errors", "started_at", "finished_at", "cost", "analysis",
    "evidence_ids", "error",
}
# Mutable fields of a Case.
_MUTABLE_CASE_FIELDS = {"experiment_ids", "notes", "title"}


def trial_id(experiment_id: str, index: int) -> str:
    return f"{experiment_id}:{int(index):05d}"


class SelfLabStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # belt and braces: REPLACE on THIS connection also fires delete triggers
            self._conn.execute("PRAGMA recursive_triggers = ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ================================================================== cases
    def insert_case(self, case: Case) -> Case:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO selflab_cases (id, created_at, run_id, access_level, claim_type, is_fixture, data) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (case.id, case.created_at, case.run_id, case.access_level, case.claim_type,
                     int(case.is_fixture), case.model_dump_json()))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise ConflictError(f"case {case.id} already exists") from e
        return case

    def get_case(self, case_id: str) -> Case | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM selflab_cases WHERE id=?", (case_id,)).fetchone()
        return Case.model_validate_json(row["data"]) if row else None

    def require_case(self, case_id: str) -> Case:
        c = self.get_case(case_id)
        if c is None:
            raise NotFoundError(f"case {case_id} not found")
        return c

    def list_cases(self, limit: int = 100, offset: int = 0) -> list[Case]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM selflab_cases ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset))).fetchall()
        return [Case.model_validate_json(r["data"]) for r in rows]

    def update_case(self, case_id: str, **fields: Any) -> Case:
        bad = set(fields) - _MUTABLE_CASE_FIELDS
        if bad:
            raise ValueError(f"case fields {sorted(bad)} are immutable")
        with self._lock:
            case = self.require_case(case_id)
            case = case.model_copy(update=fields)
            Case.model_validate(case.model_dump())  # re-validate
            self._conn.execute("UPDATE selflab_cases SET data=? WHERE id=?", (case.model_dump_json(), case_id))
            self._conn.commit()
        return case

    def add_case_experiment(self, case_id: str, experiment_id: str) -> Case:
        with self._lock:
            case = self.require_case(case_id)
            if experiment_id in case.experiment_ids:
                return case
            return self.update_case(case_id, experiment_ids=[*case.experiment_ids, experiment_id])

    # ================================================================== chats (SelfLab chat-run metadata)
    def insert_chat(self, run_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            try:
                self._conn.execute("INSERT INTO selflab_chats (run_id, created_at, data) VALUES (?,?,?)",
                                   (run_id, now_iso(), json.dumps(data)))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise ConflictError(f"chat metadata for run {run_id} is already stored") from e

    def get_chat(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM selflab_chats WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    # ================================================================== experiments
    def insert_experiment(self, record: ExperimentRecord, plan: list[TrialPlanItem]) -> ExperimentRecord:
        c = record.committed
        plan_json = json.dumps([p.model_dump(mode="json") for p in plan])
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO selflab_experiments (id, created_at, updated_at, case_id, template, status, "
                    "spec_sha256, plan_sha256, committed, plan, data) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (record.id, record.created_at, now_iso(), record.case_id, c.spec.template, record.status,
                     c.spec_sha256, c.plan_sha256, c.model_dump_json(), plan_json, record.model_dump_json()))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise ConflictError(f"experiment {record.id} already exists") from e
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT data, committed FROM selflab_experiments WHERE id=?",
                                     (experiment_id,)).fetchone()
        if not row:
            return None
        rec = ExperimentRecord.model_validate_json(row["data"])
        # the committed column is the immutable source of truth for the commitment
        committed = CommittedSpec.model_validate_json(row["committed"])
        return rec.model_copy(update={"committed": committed})

    def require_experiment(self, experiment_id: str) -> ExperimentRecord:
        r = self.get_experiment(experiment_id)
        if r is None:
            raise NotFoundError(f"experiment {experiment_id} not found")
        return r

    def get_plan(self, experiment_id: str) -> list[TrialPlanItem]:
        with self._lock:
            row = self._conn.execute("SELECT plan FROM selflab_experiments WHERE id=?", (experiment_id,)).fetchone()
        if not row:
            raise NotFoundError(f"experiment {experiment_id} not found")
        return [TrialPlanItem.model_validate(p) for p in json.loads(row["plan"])]

    def list_experiments(self, case_id: str | None = None, limit: int = 100, offset: int = 0) -> list[ExperimentRecord]:
        q, args = "SELECT id FROM selflab_experiments", []
        if case_id:
            q += " WHERE case_id=?"
            args.append(case_id)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [r for row in rows if (r := self.get_experiment(row["id"])) is not None]

    def update_experiment(self, experiment_id: str, **fields: Any) -> ExperimentRecord:
        bad = set(fields) - _MUTABLE_EXPERIMENT_FIELDS
        if bad:
            raise ValueError(f"experiment fields {sorted(bad)} are fixed at commit")
        with self._lock:
            rec = self.require_experiment(experiment_id)
            rec = ExperimentRecord.model_validate({**rec.model_dump(), **fields})
            self._conn.execute("UPDATE selflab_experiments SET data=?, status=?, updated_at=? WHERE id=?",
                               (rec.model_dump_json(), rec.status, now_iso(), experiment_id))
            self._conn.commit()
        return rec

    def transition(self, experiment_id: str, allowed_from: set[str], **fields: Any) -> ExperimentRecord:
        """Atomically check the current status and update. Raises ConflictError if not in ``allowed_from``."""
        with self._lock:
            rec = self.require_experiment(experiment_id)
            if rec.status not in allowed_from:
                raise ConflictError(f"experiment {experiment_id} is '{rec.status}'; "
                                    f"this action needs one of {sorted(allowed_from)}")
            return self.update_experiment(experiment_id, **fields)

    # ================================================================== trials (append-only)
    def insert_trial(self, trial: Trial) -> Trial:
        expected = trial_id(trial.experiment_id, trial.index)
        if trial.id != expected:
            raise ValueError(f"trial id {trial.id!r} must be {expected!r}")
        data = trial.model_dump(mode="json")
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO selflab_trials (id, experiment_id, idx, status, created_at, data_sha256, data) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (trial.id, trial.experiment_id, trial.index, trial.status, trial.created_at,
                     sha256_json(data), json.dumps(data, ensure_ascii=False)))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise DuplicateTrialError(
                    f"trial {trial.experiment_id}#{trial.index} is already stored; trials are append-only "
                    "and are never overwritten") from e
        return trial

    def get_trial(self, tid: str) -> Trial | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM selflab_trials WHERE id=?", (tid,)).fetchone()
        return Trial.model_validate_json(row["data"]) if row else None

    def require_trial(self, tid: str) -> Trial:
        t = self.get_trial(tid)
        if t is None:
            raise NotFoundError(f"trial {tid} not found")
        return t

    def list_trials(self, experiment_id: str, offset: int = 0, limit: int | None = None) -> list[Trial]:
        q = "SELECT data FROM selflab_trials WHERE experiment_id=? ORDER BY idx LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(q, (experiment_id, -1 if limit is None else int(limit), int(offset))).fetchall()
        return [Trial.model_validate_json(r["data"]) for r in rows]

    def trial_rows(self, experiment_id: str) -> list[dict[str, Any]]:
        """Raw rows (id, idx, status, data_sha256) — used by exports to report stored integrity hashes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, idx, status, data_sha256 FROM selflab_trials WHERE experiment_id=? ORDER BY idx",
                (experiment_id,)).fetchall()
        return [dict(r) for r in rows]

    def trial_indices(self, experiment_id: str) -> set[int]:
        with self._lock:
            rows = self._conn.execute("SELECT idx FROM selflab_trials WHERE experiment_id=?",
                                      (experiment_id,)).fetchall()
        return {int(r["idx"]) for r in rows}

    def trial_counts(self, experiment_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM selflab_trials WHERE experiment_id=? GROUP BY status",
                (experiment_id,)).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    # ================================================================== evidence (append-only)
    def insert_evidence(self, item: EvidenceItem) -> EvidenceItem:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO selflab_evidence (id, experiment_id, case_id, kind, status, created_at, data) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (item.id, item.experiment_id, item.case_id, item.kind, item.status, item.created_at,
                     item.model_dump_json()))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise ConflictError(f"evidence {item.id} already exists") from e
        return item

    def evidence_exists(self, evidence_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM selflab_evidence WHERE id=?", (evidence_id,)).fetchone() is not None

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM selflab_evidence WHERE id=?", (evidence_id,)).fetchone()
        return EvidenceItem.model_validate_json(row["data"]) if row else None

    def list_evidence(self, case_id: str | None = None, experiment_id: str | None = None,
                      limit: int = 500) -> list[EvidenceItem]:
        q, conds, args = "SELECT data FROM selflab_evidence", [], []
        if case_id:
            conds.append("case_id=?")
            args.append(case_id)
        if experiment_id:
            conds.append("experiment_id=?")
            args.append(experiment_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at, rowid LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [EvidenceItem.model_validate_json(r["data"]) for r in rows]

    # ================================================================== events (append-only audit log)
    def log_event(self, experiment_id: str, event: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO selflab_events (experiment_id, at, event, data) VALUES (?,?,?,?)",
                               (experiment_id, now_iso(), event, json.dumps(data or {}, default=str)))
            self._conn.commit()

    def list_events(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, at, event, data FROM selflab_events WHERE experiment_id=? ORDER BY seq",
                (experiment_id,)).fetchall()
        return [{"seq": r["seq"], "at": r["at"], "event": r["event"], "data": json.loads(r["data"])} for r in rows]

    # ================================================================== trial → Lab run links (append-only)
    def link_trial_run(self, tid: str, run_id: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO selflab_trial_runs (trial_id, run_id, created_at, data) VALUES (?,?,?,?)",
                               (tid, run_id, now_iso(), json.dumps(data or {}, default=str)))
            self._conn.commit()

    def trial_runs(self, tid: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_at, data FROM selflab_trial_runs WHERE trial_id=? ORDER BY seq",
                (tid,)).fetchall()
        return [{"run_id": r["run_id"], "created_at": r["created_at"], **json.loads(r["data"])} for r in rows]

    def experiment_trial_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        prefix = f"{experiment_id}:"
        with self._lock:
            rows = self._conn.execute(
                "SELECT trial_id, run_id, created_at, data FROM selflab_trial_runs "
                "WHERE substr(trial_id, 1, ?) = ? ORDER BY seq", (len(prefix), prefix)).fetchall()
        return [{"trial_id": r["trial_id"], "run_id": r["run_id"], "created_at": r["created_at"],
                 **json.loads(r["data"])} for r in rows]
