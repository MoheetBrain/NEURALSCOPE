"""Provenance records: exactly which model, tokenizer, template, code and libraries produced a result.

Hashes support integrity checking of local records; they do not make local logs tamper-proof.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralscope.config import REPO_ROOT
from neuralscope.selflab.schemas import ModelProvenance, RunProvenance

ANALYSIS_VERSION = "selflab-analysis-1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Canonical JSON hash (sorted keys, no whitespace) — used for spec and plan commitments."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))


def sha256_ids(ids: list[int]) -> str:
    return sha256_text(",".join(str(int(i)) for i in ids))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except Exception:
        return None


def code_state() -> tuple[str | None, bool | None]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return commit, (bool(status) if status is not None else None)


def _snapshot_revision(name_or_path: Any) -> str | None:
    p = str(name_or_path or "")
    if "/snapshots/" in p:
        return p.split("/snapshots/", 1)[1].split("/", 1)[0] or None
    return None


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "safetensors", "numpy", "pydantic", "fastapi", "tokenizers"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:
            pass
    return out


_FINGERPRINT_CACHE: dict[int, str] = {}


def weights_fingerprint(model: Any) -> str | None:
    """sha256 over every parameter's name, shape, dtype and a fixed strided sample of its values (≤ 4096 per tensor).

    Detects swapped or fine-tuned weights behind an unchanged model id/revision. A sample, not a full hash: two
    checkpoints that differ only outside the sampled entries would collide (cheap, not cryptographic coverage).
    """
    if model is None:
        return None
    key = id(model)
    if key in _FINGERPRINT_CACHE:
        return _FINGERPRINT_CACHE[key]
    import torch

    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            flat = p.detach().reshape(-1)
            step = max(1, flat.numel() // 4096)
            sample = flat[::step][:4096].to("cpu", torch.float32).contiguous()
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|".encode())
            h.update(sample.numpy().tobytes())
    _FINGERPRINT_CACHE[key] = h.hexdigest()
    return _FINGERPRINT_CACHE[key]


def model_provenance(adapter: Any) -> ModelProvenance:
    info = adapter.info
    tok = getattr(adapter, "tokenizer", None)
    model = getattr(adapter, "model", None)
    template = getattr(tok, "chat_template", None) if tok is not None else None
    if isinstance(template, dict):
        template = json.dumps(template, sort_keys=True)
    rev = None
    if model is not None:
        cfg = getattr(model, "config", None)
        rev = _snapshot_revision(getattr(cfg, "_name_or_path", None)) or getattr(cfg, "_commit_hash", None)
    tok_rev = _snapshot_revision(getattr(tok, "name_or_path", None)) if tok is not None else None
    if rev is None and not info.is_demo:
        rev = _resolve_cached_revision(info.model_id)
    if tok_rev is None and tok is not None:
        tok_rev = _resolve_cached_revision(getattr(tok, "name_or_path", "") or "")
    return ModelProvenance(
        model_id=info.model_id,
        revision=rev,
        tokenizer_id=str(getattr(tok, "name_or_path", "") or info.model_id),
        tokenizer_revision=tok_rev,
        chat_template_sha256=sha256_text(template) if isinstance(template, str) else None,
        dtype=info.dtype,
        device=info.device,
        library_versions=library_versions(),
        is_demo=bool(info.is_demo),
        weights_fingerprint=weights_fingerprint(model),
    )


def _resolve_cached_revision(repo_id: str) -> str | None:
    """Commit hash of the locally cached HF snapshot (no network)."""
    if not repo_id or "/" not in repo_id:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(repo_id, "config.json")
        return _snapshot_revision(p) if isinstance(p, str) else None
    except Exception:
        return None


def run_provenance(adapter: Any, decoding: dict[str, Any] | None = None, seed: int | None = None) -> RunProvenance:
    commit, dirty = code_state()
    return RunProvenance(
        model=model_provenance(adapter),
        code_commit=commit,
        code_dirty=dirty,
        analysis_version=ANALYSIS_VERSION,
        decoding=decoding or {},
        seed=seed,
        platform=f"{platform.system()} {platform.machine()}",
        created_at=now_iso(),
    )
