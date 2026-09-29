"""Runtime configuration for NeuralScope.

All settings come from environment variables (see `.env.example`) so the same
code runs in the API server, the test-suite and one-off research scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TraceLevel = Literal["minimal", "standard", "full"]

REPO_ROOT = Path(__file__).resolve().parents[3]

# The V1 model. See docs/ARCHITECTURE.md §"Model selection" and RESEARCH_LOG.md.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Special id: a tiny *randomly initialised* Qwen2-architecture model used for
# DEMO mode and fast tests. Its outputs carry no scientific meaning and every
# run produced by it is flagged `is_demo=True` end-to-end.
DEMO_MODEL_ID = "demo/random-qwen2-tiny"


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model_id: str = field(default_factory=lambda: os.environ.get("NEURALSCOPE_MODEL", DEFAULT_MODEL_ID))
    # Tokenizer used by the demo fixture model (it needs a real vocabulary so the
    # UI shows real tokenisation). Falls back to a byte-level toy vocab if absent.
    demo_tokenizer_id: str = field(
        default_factory=lambda: os.environ.get("NEURALSCOPE_DEMO_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("NEURALSCOPE_DATA_DIR", str(REPO_ROOT / "data"))).resolve()
    )
    device: str = field(default_factory=lambda: os.environ.get("NEURALSCOPE_DEVICE", "auto"))
    # float32 by default: interpretability measurements (logit lens ranks, small
    # intervention deltas) should not be dominated by bf16 rounding noise.
    dtype: str = field(default_factory=lambda: os.environ.get("NEURALSCOPE_DTYPE", "float32"))
    trace_level: TraceLevel = field(
        default_factory=lambda: os.environ.get("TRACE_LEVEL", "standard")  # type: ignore[return-value]
    )
    max_prompt_tokens: int = field(default_factory=lambda: int(os.environ.get("NEURALSCOPE_MAX_PROMPT_TOKENS", "160")))
    max_new_tokens_cap: int = field(default_factory=lambda: int(os.environ.get("NEURALSCOPE_MAX_NEW_TOKENS", "48")))
    num_threads: int = field(default_factory=lambda: int(os.environ.get("NEURALSCOPE_NUM_THREADS", "0")))
    lazy_load: bool = field(default_factory=lambda: _env_bool("NEURALSCOPE_LAZY_LOAD", False))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "neuralscope.db"

    @property
    def activations_dir(self) -> Path:
        return self.data_dir / "activations"

    @property
    def is_demo(self) -> bool:
        return self.model_id == DEMO_MODEL_ID


def get_settings() -> Settings:
    return Settings()
