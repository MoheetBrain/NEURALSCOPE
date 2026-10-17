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
class HeadSetEdit:
    """Replace head z (the head's input slice of o_proj) with a given value — mean / resample ablation.

    ``vector`` is [d_head] (same value at every position) or [T, d_head] indexed by absolute position.
    """

    layer: int
    head: int
    vector: torch.Tensor
    positions: list[int] | None = None


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


Edit = HeadScaleEdit | HeadSetEdit | MLPScaleEdit | ResidSetEdit | ResidAddEdit


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
    # Hook hit count: number of (edit, batch row, forward call) applications that actually fired, i.e. calls in
    # which the edit's absolute positions intersected the positions processed by that call. None = not tracked.
    n_edits_applied: int | None = None
    # log P(generated_ids[i] | prompt, generated_ids[:i]) read off the KV-cached decode logits (natural log, float32
    # log-softmax over the full vocabulary). None = not tracked.
    generated_logprobs: list[float] | None = None
    # top-1 minus top-2 logit at each greedy step (decision margin). A tiny margin (e.g. < 1e-3) marks a near-tie
    # whose argmax can flip across devices/kernels; see docs/selflab/INSTRUMENTATION_NOTES.md. None = not tracked.
    generated_margins: list[float] | None = None


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


@dataclass
class TokenizedChat:
    """A chat (list of ``{"role", "content"}`` messages) rendered with the model's chat template and tokenised.

    ``message_token_spans[i]`` is the [start, end) token span of message i's CONTENT (template tokens excluded),
    or None when the content is empty or could not be located verbatim in the templated text. Spans contain every
    token that overlaps the content's characters; ``span_exact[i]`` says whether the first/last token boundaries
    coincide exactly with the content's first/last character (False means a token straddles the boundary).
    """

    text: str  # exact templated string that was tokenised
    input_ids: list[int]
    message_token_spans: list[tuple[int, int] | None]
    message_char_spans: list[tuple[int, int] | None] = field(default_factory=list)  # [start,end) chars in ``text``
    offsets: list[tuple[int, int]] = field(default_factory=list)  # per-token [start,end) chars in ``text``
    span_exact: list[bool | None] = field(default_factory=list)
    add_generation_prompt: bool = True

    def char_span_to_tokens(self, message_index: int, char_start: int, char_end: int) -> tuple[int, int] | None:
        """Map a [char_start, char_end) span INSIDE message ``message_index``'s content to a [start, end) token span.

        Returns the smallest token span covering every token that overlaps the character span, or None when the
        character span is empty or the message content was not located. Raises ValueError for spans outside the
        message content.
        """
        if not 0 <= message_index < len(self.message_token_spans):
            raise ValueError(f"message_index {message_index} out of range (0..{len(self.message_token_spans) - 1})")
        cs = self.message_char_spans[message_index] if message_index < len(self.message_char_spans) else None
        if cs is None:
            return None
        n = cs[1] - cs[0]
        if not 0 <= char_start <= char_end <= n:
            raise ValueError(f"char span [{char_start}, {char_end}) outside message content of length {n}")
        if char_start == char_end:
            return None
        a, b = cs[0] + char_start, cs[0] + char_end
        idx = [i for i, (s, e) in enumerate(self.offsets) if e > a and s < b and e > s]
        return (idx[0], idx[-1] + 1) if idx else None


class ModelAdapter(ABC):
    """Abstract instrumentation adapter."""

    info: ModelInfo
    # Hook statistics of the most recent instrumented call (set by concrete adapters; read under the adapter lock):
    # {"n_edits_applied", "n_forward_calls", "n_rows", "n_edits"}.
    last_session_stats: dict | None = None

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def tokenize(self, prompt: str, chat: bool = True, system_prompt: str | None = None) -> TokenizedPrompt: ...

    def tokenize_messages(self, messages: list[dict], add_generation_prompt: bool = True) -> TokenizedChat:
        """Render ``messages`` with the chat template, tokenise, and locate each message's content tokens.

        For a single user message (no system message) the ids equal ``tokenize(content).input_ids``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement tokenize_messages")

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
                           resid_in: torch.Tensor | None = None,
                           logit_positions: list[int] | None = None) -> torch.Tensor:
        """Run B copies of the same sequence, copy b under edit_sets[b]. Returns logits [B, T, V].

        start_depth > 0 resumes from a cached residual state ``resid_in`` [T, d] (sweep acceleration).
        ``logit_positions`` (optional) unembeds only those positions → [B, len(logit_positions), V]."""

    def sequence_logits(self, sequences: list[list[int]], positions: list[list[int]],
                        edits: list[Edit] | None = None) -> list[torch.Tensor]:
        """Teacher-forced logits of several DIFFERENT sequences, each under the same ``edits`` (absolute positions).

        Returns one float32 CPU tensor [len(positions[b]), V] per sequence: the logits at the requested positions.
        Generic fallback: one ``final_logits_batch`` call per sequence. Concrete adapters may batch.
        """
        if len(sequences) != len(positions):
            raise ValueError("sequences and positions must have the same length")
        return [self.final_logits_batch(list(seq), [list(edits or [])], logit_positions=list(pos))[0]
                for seq, pos in zip(sequences, positions)]

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
