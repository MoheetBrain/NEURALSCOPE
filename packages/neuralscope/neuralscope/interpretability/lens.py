"""Logit Lens (nostalgebraist, 2020) — DERIVED quantities.

For a residual state h at depth d we compute

    lens_logits(h) = W_U · FinalNorm(h)

using the model's *own* final RMSNorm (with its learned gain and the per-vector
1/rms computed from h itself) and unembedding. At the last depth this is
exactly the model's output logits (tested). At earlier depths it is a
projection of an intermediate state through a readout the model was never
trained to use there; it is systematically biased in early layers (see Belrose
et al., 2023, "tuned lens"). It is therefore labelled DERIVED and must never be
described as "what the model was thinking".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuralscope.instrumentation.adapter import ModelAdapter


@dataclass
class LensSummary:
    top_ids: torch.Tensor  # [D, T, k] int64
    top_probs: torch.Tensor  # [D, T, k] float32
    top_logits: torch.Tensor  # [D, T, k] float32
    entropy: torch.Tensor  # [D, T] float32 (nats)


@torch.no_grad()
def lens_summary(adapter: ModelAdapter, resid: torch.Tensor, k: int = 10) -> LensSummary:
    """Top-k decoded tokens + entropy for every (depth, position) cell. Chunked per depth."""
    D, T, _ = resid.shape
    ids, probs, logits_k, ent = [], [], [], []
    for d in range(D):
        lg = adapter.logit_lens(resid[d])  # [T, V]
        lp = torch.log_softmax(lg, dim=-1)
        top = torch.topk(lg, k, dim=-1)
        ids.append(top.indices)
        logits_k.append(top.values)
        probs.append(lp.gather(-1, top.indices).exp())
        ent.append(-(lp.exp() * lp).sum(-1))
    return LensSummary(torch.stack(ids), torch.stack(probs), torch.stack(logits_k), torch.stack(ent))


@dataclass
class LensTarget:
    logit: torch.Tensor  # [D, T]
    prob: torch.Tensor  # [D, T]
    logprob: torch.Tensor  # [D, T]
    rank: torch.Tensor  # [D, T] int64, 1 = top candidate


@torch.no_grad()
def lens_target(adapter: ModelAdapter, resid: torch.Tensor, token_id: int,
                positions: list[int] | None = None) -> LensTarget:
    """Track one vocabulary item through depth: logit, probability and rank per cell.

    rank = 1 + #{v : logit_v > logit_target} (ties resolved in the target's favour).
    """
    sub = resid if positions is None else resid[:, positions]
    D, T, _ = sub.shape
    logit, prob, lp_out, rank = [], [], [], []
    for d in range(D):
        lg = adapter.logit_lens(sub[d])  # [T, V]
        lp = torch.log_softmax(lg, dim=-1)
        tl = lg[:, token_id]
        logit.append(tl)
        lp_out.append(lp[:, token_id])
        prob.append(lp[:, token_id].exp())
        rank.append(1 + (lg > tl[:, None]).sum(-1))
    return LensTarget(torch.stack(logit), torch.stack(prob), torch.stack(lp_out), torch.stack(rank))


@torch.no_grad()
def lens_at(adapter: ModelAdapter, resid_vec: torch.Tensor, k: int = 10) -> dict:
    """Full lens readout for a single residual vector [d]."""
    lg = adapter.logit_lens(resid_vec[None])[0]
    lp = torch.log_softmax(lg, dim=-1)
    top = torch.topk(lg, k)
    return {
        "ids": top.indices.tolist(),
        "logits": top.values.tolist(),
        "probs": lp[top.indices].exp().tolist(),
        "entropy": float(-(lp.exp() * lp).sum()),
        "logits_full": lg,
        "logprobs_full": lp,
    }
