"""Comparison metrics between runs (DERIVED from OBSERVED logits / states)."""

from __future__ import annotations

import torch


def kl_divergence(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) in nats over the last axis, from log-probabilities."""
    return (logp.exp() * (logp - logq)).sum(-1)


def entropy(logp: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).sum(-1)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def successive_cosine_change(resid: torch.Tensor) -> torch.Tensor:
    """1 − cos(h_d, h_{d−1}) for each depth d ≥ 1 → [D−1, T]. How much each block rotates the stream."""
    return 1.0 - cosine(resid[1:], resid[:-1])


def relative_delta(base: torch.Tensor, other: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """‖other − base‖ / ‖base‖ over the last axis."""
    return (other - base).norm(dim=-1) / (base.norm(dim=-1) + eps)


def rank_of(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """1-based rank of token_id along the last axis."""
    return 1 + (logits > logits[..., token_id:token_id + 1]).sum(-1)
