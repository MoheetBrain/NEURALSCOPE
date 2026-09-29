"""Typed specifications of causal interventions.

An intervention is a *declarative* description of an edit to the forward pass.
Specs are serialisable (stored with every run) so that any intervention run can
be re-executed deterministically and audited later.

Position semantics: positions are absolute token indices in the sequence
(prompt + generated). ``positions=None`` means "every position", including
tokens produced later during free-running generation.

Depth semantics (residual stream sites): depth 0 is the token embedding
(input to block 0); depth d (1..n_layers) is the residual stream *after*
transformer block d-1. Head/MLP ``layer`` indices are block indices 0..n_layers-1.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class HeadAblation(BaseModel):
    """Scale one attention head's write into the residual stream.

    Implemented on the input of the attention output projection (o_proj), whose
    input is the concatenation of per-head outputs z_h = A_h V_h. Because o_proj
    is linear and (for Qwen2/Llama) bias-free, scaling the head's slice by
    ``factor`` scales exactly that head's contribution W_O^h z_h to the residual
    stream and nothing else. factor=0 is zero-ablation.
    """

    type: Literal["head_ablation"] = "head_ablation"
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    factor: float = 0.0
    positions: list[int] | None = None

    def describe(self) -> str:
        what = "zero-ablate" if self.factor == 0 else f"scale ×{self.factor:g}"
        return f"{what} head L{self.layer}.H{self.head}" + _pos_suffix(self.positions)


class MLPAblation(BaseModel):
    """Scale the output of one MLP block (factor=0 removes its write)."""

    type: Literal["mlp_ablation"] = "mlp_ablation"
    layer: int = Field(ge=0)
    factor: float = 0.0
    positions: list[int] | None = None

    def describe(self) -> str:
        what = "zero-ablate" if self.factor == 0 else f"scale ×{self.factor:g}"
        return f"{what} MLP L{self.layer}" + _pos_suffix(self.positions)


class ResidualPatch(BaseModel):
    """Activation patching: h[depth, position] <- (1-alpha) h + alpha h_src.

    The source vector is the residual stream of another stored run
    (``source_run_id``) at (``depth``, ``source_position``). alpha=1 is a full
    patch; 0<alpha<1 interpolates (a mild, bounded form of steering along the
    source−target difference). The downstream computation then proceeds
    normally from the patched state.
    """

    type: Literal["resid_patch"] = "resid_patch"
    depth: int = Field(ge=0)
    position: int = Field(ge=0)
    source_run_id: str
    source_position: int | None = None  # defaults to ``position``
    alpha: float = 1.0

    def describe(self) -> str:
        sp = self.position if self.source_position is None else self.source_position
        a = "" if self.alpha == 1 else f" (α={self.alpha:g})"
        return f"patch resid[d{self.depth}, t{self.position}] ← run {self.source_run_id[:8]}[t{sp}]{a}"


class ResidualSteer(BaseModel):
    """Add ``scale`` × (h_src[depth, source_position] − h_base_src[depth, base_position]) at depth.

    A difference-of-activations steering vector computed from two stored runs.
    Steering is only scientifically meaningful as a *test*; the resulting
    behaviour change says the direction is causally sufficient to move the
    output, not that it "is" a concept.
    """

    type: Literal["resid_steer"] = "resid_steer"
    depth: int = Field(ge=0)
    positions: list[int] | None = None
    plus_run_id: str
    plus_position: int
    minus_run_id: str
    minus_position: int
    scale: float = 1.0

    def describe(self) -> str:
        return (
            f"steer resid[d{self.depth}] += {self.scale:g}·(run {self.plus_run_id[:8]}[t{self.plus_position}] − "
            f"run {self.minus_run_id[:8]}[t{self.minus_position}])" + _pos_suffix(self.positions)
        )


Intervention = Annotated[
    Union[HeadAblation, MLPAblation, ResidualPatch, ResidualSteer],
    Field(discriminator="type"),
]


def _pos_suffix(positions: list[int] | None) -> str:
    if positions is None:
        return " (all positions)"
    if len(positions) <= 4:
        return " @ t" + ",".join(str(p) for p in positions)
    return f" @ {len(positions)} positions"


def describe_all(interventions: list) -> str:
    return "; ".join(i.describe() for i in interventions) if interventions else "none"
