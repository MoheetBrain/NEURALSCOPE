"""Direct logit attribution (DLA) — a DERIVED, exact linear decomposition.

The final residual state at position t is a sum of component writes:

    h_final = embed + Σ_l Σ_h W_O^{l,h} z^{l,h} + Σ_l MLP_l

The final RMSNorm is x ↦ γ ⊙ x · s(x) with scalar s = 1/rms(h_final). Freezing s
at its actual value makes the readout linear, so the (contrast) logit splits
exactly into per-component terms:

    logit_τ − logit_κ = s · Σ_c ⟨c, γ ⊙ (W_U[τ] − W_U[κ])⟩

The sum reproduces the real logit to float precision (tested). DLA measures
*direct* effects only: a component that matters by changing later components
(indirect effect) can have ~0 DLA. Only interventions test total effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuralscope.instrumentation.adapter import ModelAdapter, Trace


@dataclass
class DLAResult:
    position: int
    target_id: int
    contrast_id: int | None
    embed: float
    heads: torch.Tensor  # [N, H]
    mlps: torch.Tensor  # [N]
    total: float  # Σ components
    actual: float  # actual final (contrast) logit from the model
    norm_scale: float


@torch.no_grad()
def direct_logit_attribution(adapter: ModelAdapter, trace: Trace, position: int, target_id: int,
                             contrast_id: int | None = None) -> DLAResult:
    if trace.head_z is None:
        raise ValueError("DLA requires per-head outputs (trace level 'standard' or 'full')")
    info = adapter.info
    N, H = info.n_layers, info.n_heads
    u = adapter.unembed_direction(target_id)
    if contrast_id is not None:
        u = u - adapter.unembed_direction(contrast_id)
    s = float(adapter.final_norm_scale(trace.resid[-1, position]))

    embed = float(trace.resid[0, position] @ u) * s
    heads = torch.zeros(N, H)
    mlps = torch.zeros(N)
    for l in range(N):
        z = trace.head_z[l, position]  # [H, dh]
        for h in range(H):
            heads[l, h] = float(adapter.head_output(l, h, z[h]) @ u) * s
        if trace.mlp_out is not None:
            m = trace.mlp_out[l, position]
        else:  # derived: mlp_out = h[l+1] − h[l] − attn_out
            m = trace.resid[l + 1, position] - trace.resid[l, position] - adapter.attn_out_from_z(l, z)
        mlps[l] = float(m @ u) * s
    total = embed + float(heads.sum()) + float(mlps.sum())
    actual = float(trace.logits[position, target_id])
    if contrast_id is not None:
        actual -= float(trace.logits[position, contrast_id])
    return DLAResult(position, target_id, contrast_id, embed, heads, mlps, total, actual, s)


@torch.no_grad()
def head_vocab_projection(adapter: ModelAdapter, trace: Trace, layer: int, head: int, position: int,
                          k: int = 8) -> dict:
    """Tokens most promoted / suppressed by one head's write at one position (DERIVED).

    Projects W_O^{l,h} z through γ ⊙ W_U (no rms scaling — relative ordering only).
    """
    if trace.head_z is None:
        raise ValueError("requires head outputs")
    out = adapter.head_output(layer, head, trace.head_z[layer, position, head])  # [d]
    s = adapter.final_norm_scale(trace.resid[-1, position])
    # the head's direct contribution to every final logit (final-norm scale frozen at its actual value)
    lg = adapter.readout_direction(out) * float(s)
    top = torch.topk(lg, k)
    bot = torch.topk(-lg, k)
    return {
        "promoted": list(zip(top.indices.tolist(), top.values.tolist())),
        "suppressed": list(zip(bot.indices.tolist(), (-bot.values).tolist())),
        "output_norm": float(out.norm()),
    }
