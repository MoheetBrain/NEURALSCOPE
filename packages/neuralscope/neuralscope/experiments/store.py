"""Experiment persistence.

Split storage (see docs/ARCHITECTURE.md §Storage):

  * metadata  → SQLite (``data/neuralscope.db``): runs, experiments, benchmarks.
                Small, queryable, JSON columns for nested fields.
  * tensors   → one safetensors file per run (``data/activations/<run_id>.safetensors``):
                residual stream, attention, head outputs, lens summaries, output
                log-probs at generated positions. Never stored in SQL, never sent
                wholesale to the browser.

Runs are deterministic (greedy decoding, eager attention, fp32), so any missing
tensor can be regenerated from (model_id, input_ids, interventions) — the
"lazy capture" path used when a view needs more than the stored trace level.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

RUN_JSON_FIELDS = {
    "generation", "input_ids", "user_span", "generated_ids", "free_generated_ids", "target", "tensor_meta",
    "interventions", "metrics", "math", "tags", "messages",
}

# Columns added after the first schema. Each is applied with ``ALTER TABLE runs ADD COLUMN`` when missing, so older
# databases keep working (existing rows read the new column as NULL).
RUN_MIGRATIONS: list[tuple[str, str]] = [
    ("messages", "TEXT"),  # JSON list of {"role", "content"} chat messages (SelfLab multi-message runs)
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    prompt TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    chat INTEGER NOT NULL DEFAULT 1,
    system_prompt TEXT,
    generation TEXT,
    input_ids TEXT NOT NULL,
    prompt_len INTEGER NOT NULL,
    user_span TEXT,
    generated_ids TEXT,
    generated_text TEXT,
    free_generated_ids TEXT,
    free_generated_text TEXT,
    target TEXT,
    trace_level TEXT NOT NULL,
    tensor_path TEXT,
    tensor_meta TEXT,
    parent_run_id TEXT,
    interventions TEXT,
    metrics TEXT,
    math TEXT,
    correctness TEXT,
    notes TEXT,
    tags TEXT,
    benchmark_id TEXT,
    trace_consistent INTEGER,
    messages TEXT
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS runs_parent ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS runs_benchmark ON runs(benchmark_id);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    name TEXT NOT NULL,
    notes TEXT,
    run_ids TEXT NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_id TEXT NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    name TEXT,
    spec TEXT,
    results TEXT,
    summary TEXT,
    status TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ExperimentStore:
    def __init__(self, data_dir: Path, tensor_cache_size: int = 12):
        self.data_dir = Path(data_dir)
        self.tensor_dir = self.data_dir / "activations"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "experiments").mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "neuralscope.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._migrate()
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._cache_size = tensor_cache_size

    def _migrate(self) -> None:
        """Backwards-compatible additive migrations (idempotent; safe if another process migrated first)."""
        with self._lock:
            have = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col, typ in RUN_MIGRATIONS:
                if col in have:
                    continue
                try:
                    self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError as e:  # concurrent migration by another connection
                    if "duplicate column" not in str(e).lower():
                        raise
            self._conn.commit()

    # ------------------------------------------------------------------ runs
    def save_run(self, record: dict[str, Any], tensors: dict[str, torch.Tensor] | None) -> dict[str, Any]:
        rec = dict(record)
        rec.setdefault("id", new_id("run"))
        rec.setdefault("created_at", now_iso())
        if tensors:
            path = self.tensor_dir / f"{rec['id']}.safetensors"
            clean = {k: v.contiguous().cpu() for k, v in tensors.items() if v is not None}
            save_file(clean, str(path))
            rec["tensor_path"] = str(path.relative_to(self.data_dir))
            rec["tensor_meta"] = {
                "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in clean.items()},
                "bytes": path.stat().st_size,
            }
            self._cache_put(rec["id"], clean)
        cols = [c for c in rec.keys()]
        vals = [self._enc(c, rec[c]) for c in cols]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
            self._conn.commit()
        return self.get_run(rec["id"])  # type: ignore[return-value]

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get_run(run_id)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE runs SET {sets} WHERE id=?",
                               [self._enc(k, v) for k, v in fields.items()] + [run_id])
            self._conn.commit()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._dec_run(row) if row else None

    def list_runs(self, limit: int = 100, kind: str | None = None, benchmark_id: str | None = None,
                  parent_run_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs", []
        conds = []
        if kind:
            conds.append("kind=?"); args.append(kind)
        if benchmark_id:
            conds.append("benchmark_id=?"); args.append(benchmark_id)
        if parent_run_id:
            conds.append("parent_run_id=?"); args.append(parent_run_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._dec_run(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        rec = self.get_run(run_id)
        if not rec:
            return False
        if rec.get("tensor_path"):
            p = self.data_dir / rec["tensor_path"]
            if p.exists():
                p.unlink()
        with self._lock:
            self._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
            self._conn.commit()
        self._cache.pop(run_id, None)
        return True

    def load_tensors(self, run_id: str) -> dict[str, torch.Tensor]:
        if run_id in self._cache:
            self._cache.move_to_end(run_id)
            return self._cache[run_id]
        rec = self.get_run(run_id)
        if not rec or not rec.get("tensor_path"):
            raise KeyError(f"no tensors stored for run {run_id}")
        t = load_file(str(self.data_dir / rec["tensor_path"]))
        self._cache_put(run_id, t)
        return t

    def add_tensors(self, run_id: str, extra: dict[str, torch.Tensor]) -> None:
        """Merge lazily-captured tensors into a run's tensor file."""
        t = dict(self.load_tensors(run_id))
        t.update({k: v.contiguous().cpu() for k, v in extra.items()})
        rec = self.get_run(run_id)
        assert rec is not None
        path = self.data_dir / rec["tensor_path"]
        save_file(t, str(path))
        self._cache_put(run_id, t)
        meta = {
            "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in t.items()},
            "bytes": path.stat().st_size,
        }
        self.update_run(run_id, tensor_meta=meta)

    def _cache_put(self, run_id: str, t: dict[str, torch.Tensor]) -> None:
        self._cache[run_id] = t
        self._cache.move_to_end(run_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------ experiments
    def save_experiment(self, name: str, run_ids: list[str], notes: str = "", payload: dict | None = None,
                        experiment_id: str | None = None) -> dict[str, Any]:
        eid = experiment_id or new_id("exp")
        existing = self.get_experiment(eid) if experiment_id else None
        created = existing["created_at"] if existing else now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiments (id, created_at, updated_at, name, notes, run_ids, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (eid, created, now_iso(), name, notes, json.dumps(run_ids), json.dumps(payload or {})))
            self._conn.commit()
        exp = self.get_experiment(eid)
        assert exp is not None
        # human-readable JSON snapshot next to the DB (easy to diff / share / archive)
        (self.data_dir / "experiments" / f"{eid}.json").write_text(json.dumps(exp, indent=2))
        return exp

    def get_experiment(self, eid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["run_ids"] = json.loads(d["run_ids"])
        d["payload"] = json.loads(d["payload"] or "{}")
        return d

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM experiments ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_experiment(r["id"]) for r in rows]  # type: ignore[misc]

    def delete_experiment(self, eid: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM experiments WHERE id=?", (eid,))
            self._conn.commit()
        p = self.data_dir / "experiments" / f"{eid}.json"
        if p.exists():
            p.unlink()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ benchmarks
    def save_benchmark(self, rec: dict[str, Any]) -> dict[str, Any]:
        rec = dict(rec)
        rec.setdefault("id", new_id("bench"))
        rec.setdefault("created_at", now_iso())
        cols = list(rec.keys())
        vals = [json.dumps(rec[c]) if c in {"spec", "results", "summary"} else rec[c] for c in cols]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO benchmarks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
            self._conn.commit()
        return self.get_benchmark(rec["id"])  # type: ignore[return-value]

    def get_benchmark(self, bid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM benchmarks WHERE id=?", (bid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("spec", "results", "summary"):
            d[k] = json.loads(d[k]) if d[k] else None
        d["is_demo"] = bool(d["is_demo"])
        return d

    def list_benchmarks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM benchmarks ORDER BY created_at DESC, rowid DESC LIMIT ?",
                                      (limit,)).fetchall()
        return [self.get_benchmark(r["id"]) for r in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------ encoding
    @staticmethod
    def _enc(col: str, v: Any) -> Any:
        if col in RUN_JSON_FIELDS:
            return None if v is None else json.dumps(v)
        if isinstance(v, bool):
            return int(v)
        return v

    @staticmethod
    def _dec_run(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for k in RUN_JSON_FIELDS:
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        d["is_demo"] = bool(d["is_demo"])
        d["chat"] = bool(d["chat"])
        if d.get("trace_consistent") is not None:
            d["trace_consistent"] = bool(d["trace_consistent"])
        return d
