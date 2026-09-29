"""Model-agnostic instrumentation interface.

NeuralScope never talks to a specific interpretability library directly. Every
model family is wrapped by a ``ModelAdapter`` which exposes:

    load() · tokenize() · generate() · capture() · logit_lens() · unembed_direction()

Interventions are passed to ``generate``/``capture`` as *compiled edits*
(``Edit`` objects below): low-level, already-resolved tensor operations. The
service layer compiles user-facing specs (``neuralscope.schemas.interventions``)
into edits, e.g. by loading a patch source vector from another stored run.

Future instrumentation points (Q/K/V, MLP pre-activations, SwiGLU gates,
gradients, SAE features, transcoders, Jacobians, KV-cache) plug in as new
capture keys on ``Trace`` and new hook sites in concrete adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch

TraceLevel = Literal["minimal", "standard", "full"]


# --------------------------------------------------------------------------------------
# Compiled edits (the only intervention vocabulary adapters need to understand)
# --------------------------------------------------------------------------------------


@dataclass
class HeadScaleEdit:
    layer: int
    head: int
    factor: float = 0.0
    positions: list[int] | None = None  # absolute positions; None = all


@dataclass
class MLPScaleEdit:
    layer: int
    factor: float = 0.0
    positions: list[int] | None = None


@dataclass
class ResidSetEdit:
    """h[depth, position] <- (1 - alpha) * h + alpha * vector."""

    depth: int
    position: int
    vector: torch.Tensor
    alpha: float = 1.0


@dataclass
class ResidAddEdit:
    """h[depth, positions] += vector."""

    depth: int
    vector: torch.Tensor
    positions: list[int] | None = None


Edit = HeadScaleEdit | MLPScaleEdit | ResidSetEdit | ResidAddEdit


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass
class ModelInfo:
    model_id: str
    architecture: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    vocab_size: int
    dtype: str
    device: str
    n_params: int
    tied_embeddings: bool
    norm_type: str
    is_demo: bool
    capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def n_depths(self) -> int:
        return self.n_layers + 1


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    stopped_by: Literal["eos", "max_new_tokens"]


@dataclass
class Trace:
    """Tensors captured from ONE teacher-forced forward pass over ``input_ids``.

    Shapes (T = sequence length, D = n_layers + 1 residual depths, N = n_layers,
    H = n_heads, dh = d_head, d = d_model, V = vocab):

      resid     [D, T, d]    OBSERVED  residual stream; depth 0 = embeddings,
                                        depth l+1 = output of block l (pre final-norm)
      logits    [T, V]       OBSERVED  final logits (position t predicts token t+1)
      attn      [N, H, T, T] OBSERVED  attention probabilities (standard+)
      head_z    [N, T, H, dh] OBSERVED per-head attention outputs z = A·V, i.e.
                                        the input of o_proj split by head (standard+)
      mlp_out   [N, T, d]    OBSERVED  MLP block outputs (full); otherwise derivable as
                                        resid[l+1] - resid[l] - attn_out[l]
    """

    input_ids: torch.Tensor
    resid: torch.Tensor
    logits: torch.Tensor
    level: TraceLevel
    attn: torch.Tensor | None = None
    head_z: torch.Tensor | None = None
    mlp_out: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass
class TokenizedPrompt:
    text: str  # the exact string fed to the tokenizer (after chat templating)
    input_ids: list[int]
    user_span: tuple[int, int] | None  # [start, end) token span of the user's own text, if locatable


class ModelAdapter(ABC):
    """Abstract instrumentation adapter."""

    info: ModelInfo

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def tokenize(self, prompt: str, chat: bool = True, system_prompt: str | None = None) -> TokenizedPrompt: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenise raw text without special tokens."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def token_strings(self, ids: list[int]) -> list[str]: ...

    @abstractmethod
    def is_special(self, token_id: int) -> bool: ...

    @abstractmethod
    def generate(self, input_ids: list[int], max_new_tokens: int, edits: list[Edit] | None = None) -> GenerationResult: ...

    @abstractmethod
    def capture(self, input_ids: list[int], level: TraceLevel = "standard", edits: list[Edit] | None = None) -> Trace: ...

    @abstractmethod
    def final_logits_batch(self, input_ids: list[int], edit_sets: list[list[Edit]], start_depth: int = 0,
                           resid_in: torch.Tensor | None = None) -> torch.Tensor:
        """Run B copies of the same sequence, copy b under edit_sets[b]. Returns logits [B, T, V].

        start_depth > 0 resumes from a cached residual state ``resid_in`` [T, d] (sweep acceleration)."""

    @abstractmethod
    def logit_lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Project residual states [..., d] to vocabulary logits [..., V] (final norm + unembedding)."""

    @abstractmethod
    def unembed_direction(self, token_id: int) -> torch.Tensor:
        """Direction u such that logit(token) = <norm_hat(h), u>, i.e. gamma ⊙ W_U[token]."""

    @abstractmethod
    def readout_direction(self, vec: torch.Tensor) -> torch.Tensor:
        """Linear readout of a residual-space direction [..., d] → [..., V]: (γ ⊙ vec) · W_Uᵀ (no rms factor)."""

    @abstractmethod
    def final_norm_scale(self, resid: torch.Tensor) -> torch.Tensor:
        """The (input-dependent) 1/rms factor the final norm applies to ``resid`` [..., d] → [...]."""

    @abstractmethod
    def head_output(self, layer: int, head: int, z: torch.Tensor) -> torch.Tensor:
        """Map per-head z [..., dh] to the head's residual-stream write [..., d] (W_O slice)."""

    @abstractmethod
    def attn_out_from_z(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        """Map all-head z [..., H, dh] to the attention block output [..., d]."""
