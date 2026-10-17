#!/usr/bin/env python3
"""Serialise device-heavy commands on this machine: only ONE process holds a real model at a time.

Usage:  .venv/bin/python scripts/model_lock.py -- <command> [args...]

Takes an exclusive fcntl lock on data/.model.lock, runs the command, releases the lock.
Fast fixture-model tests do not need it; anything that loads Qwen2.5 weights does.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "data" / ".model.lock"


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1:
            print(f"[model_lock] waited {waited:.0f}s for the model lock", file=sys.stderr, flush=True)
        f.seek(0)
        f.truncate()
        f.write(f"pid={os.getpid()} cmd={' '.join(argv)[:200]}\n")
        f.flush()
        try:
            return subprocess.call(argv)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
