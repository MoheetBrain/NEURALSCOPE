"""Minimal in-process background jobs (sweeps, benchmarks).

One worker thread: the model is a single shared resource, so long jobs run
serially; interactive requests interleave with them between forward passes
(the adapter holds a lock per forward pass, not per job).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class JobManager:
    def __init__(self, max_jobs: int = 200):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuralscope-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max = max_jobs

    def submit(self, kind: str, fn: Callable[[Callable[[float, str], None]], Any], meta: dict | None = None) -> dict:
        jid = f"job_{uuid.uuid4().hex[:10]}"
        job = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "message": "queued",
               "result": None, "error": None, "created_at": time.time(), "started_at": None,
               "finished_at": None, "meta": meta or {}}
        with self._lock:
            self._jobs[jid] = job
            if len(self._jobs) > self._max:
                for k in list(self._jobs)[: len(self._jobs) - self._max]:
                    if self._jobs[k]["status"] in ("done", "error"):
                        self._jobs.pop(k, None)

        def progress(frac: float, msg: str) -> None:
            job["progress"] = round(float(frac), 4)
            job["message"] = msg

        def run() -> None:
            job["status"] = "running"
            job["started_at"] = time.time()
            try:
                job["result"] = fn(progress)
                job["status"] = "done"
                job["progress"] = 1.0
                job["message"] = "done"
            except Exception as e:  # surfaced to the client
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc(limit=6)
            finally:
                job["finished_at"] = time.time()

        self._pool.submit(run)
        return self.public(job)

    def get(self, jid: str) -> dict | None:
        job = self._jobs.get(jid)
        return self.public(job) if job else None

    @staticmethod
    def public(job: dict) -> dict:
        return {k: v for k, v in job.items() if k != "traceback"}

    def list(self) -> list[dict]:
        return [self.public(j) for j in sorted(self._jobs.values(), key=lambda j: -j["created_at"])]
