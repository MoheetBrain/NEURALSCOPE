"""Instrumentation layer: model adapters, hook sessions, trace capture, compiled edits."""

from __future__ import annotations

import threading

from neuralscope.config import Settings, get_settings
from neuralscope.instrumentation.adapter import (
    Edit,
    GenerationResult,
    HeadScaleEdit,
    HeadSetEdit,
    MLPScaleEdit,
    ModelAdapter,
    ModelInfo,
    ResidAddEdit,
    ResidSetEdit,
    TokenizedChat,
    TokenizedPrompt,
    Trace,
    TraceLevel,
)

_adapter: ModelAdapter | None = None
_adapter_lock = threading.Lock()


def load_adapter(settings: Settings | None = None) -> ModelAdapter:
    """Build and load the adapter for ``settings.model_id`` (demo fixture or HF decoder)."""
    settings = settings or get_settings()
    if settings.is_demo:
        from neuralscope.instrumentation.demo import build_demo_adapter

        return build_demo_adapter(settings.demo_tokenizer_id, device="cpu")
    from neuralscope.instrumentation.hf_decoder import HFDecoderAdapter

    a = HFDecoderAdapter(settings.model_id, device=settings.device, dtype=settings.dtype,
                         num_threads=settings.num_threads)
    a.load()
    return a


def get_adapter(settings: Settings | None = None) -> ModelAdapter:
    """Process-wide singleton adapter (one model in memory)."""
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = load_adapter(settings)
        return _adapter


def set_adapter(adapter: ModelAdapter | None) -> None:
    global _adapter
    with _adapter_lock:
        _adapter = adapter


__all__ = [
    "Edit", "GenerationResult", "HeadScaleEdit", "HeadSetEdit", "MLPScaleEdit", "ModelAdapter", "ModelInfo", "ResidAddEdit",
    "ResidSetEdit", "TokenizedChat", "TokenizedPrompt", "Trace", "TraceLevel", "get_adapter", "load_adapter", "set_adapter",
]
