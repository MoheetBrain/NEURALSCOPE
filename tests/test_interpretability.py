"""Logit Lens, DLA and metric calculations."""

from __future__ import annotations

import torch

from neuralscope.interpretability.attribution import direct_logit_attribution
from neuralscope.interpretability.lens import lens_at, lens_summary, lens_target
from neuralscope.interpretability.metrics import (
    cosine,
    entropy,
    kl_divergence,
    rank_of,
    relative_delta,
    successive_cosine_change,
)
from tests.conftest import PROMPT


def _trace(a, level="standard"):
    return a.capture(a.tokenize(PROMPT).input_ids, level)


def test_lens_uses_final_norm_and_unembedding(demo_adapter):
    tr = _trace(demo_adapter, "minimal")
    h = tr.resid[2, 7]
    m = demo_adapter.model
    manual = m.lm_head(m.model.norm(h)).detach()
    assert torch.allclose(demo_adapter.logit_lens(h[None])[0], manual, atol=1e-5)
    # the rms normalisation is per-vector: scaling h leaves the lens unchanged (up to eps)
    assert torch.allclose(demo_adapter.logit_lens((3.0 * h)[None])[0], manual, atol=1e-3)


def test_lens_target_rank_matches_bruteforce(demo_adapter):
    tr = _trace(demo_adapter, "minimal")
    tid = int(tr.logits[-1].argmax())
    lt = lens_target(demo_adapter, tr.resid, tid)
    for d in range(tr.resid.shape[0]):
        for t in (0, 5, tr.resid.shape[1] - 1):
            lg = demo_adapter.logit_lens(tr.resid[d, t][None])[0]
            order = torch.argsort(lg, descending=True)
            brute = int((order == tid).nonzero()[0]) + 1
            assert int(lt.rank[d, t]) == brute
            assert torch.isclose(lt.prob[d, t], torch.softmax(lg, -1)[tid], atol=1e-6)
    assert int(lt.rank[-1, -1]) == 1


def test_lens_summary_consistent_with_single_cell(demo_adapter):
    tr = _trace(demo_adapter, "minimal")
    s = lens_summary(demo_adapter, tr.resid, k=5)
    one = lens_at(demo_adapter, tr.resid[3, 4], k=5)
    assert s.top_ids[3, 4].tolist() == one["ids"]
    assert torch.allclose(s.top_probs[3, 4], torch.tensor(one["probs"]), atol=1e-6)
    assert abs(float(s.entropy[3, 4]) - one["entropy"]) < 1e-4
    assert (s.top_probs[..., :-1] >= s.top_probs[..., 1:]).all()


def test_dla_decomposition_sums_to_logit(demo_adapter):
    tr = _trace(demo_adapter, "standard")
    pos = tr.seq_len - 1
    tid = int(tr.logits[pos].argmax())
    r = direct_logit_attribution(demo_adapter, tr, pos, tid)
    assert abs(r.total - r.actual) < 1e-3
    cid = int(torch.topk(tr.logits[pos], 2).indices[1])
    r2 = direct_logit_attribution(demo_adapter, tr, pos, tid, cid)
    assert abs(r2.total - r2.actual) < 1e-3 and r2.actual >= 0


def test_metrics():
    p = torch.log_softmax(torch.randn(10, 50), -1)
    q = torch.log_softmax(torch.randn(10, 50), -1)
    assert torch.allclose(kl_divergence(p, p), torch.zeros(10), atol=1e-6)
    assert (kl_divergence(p, q) > 0).all()
    u = torch.log_softmax(torch.zeros(50), -1)
    assert abs(float(entropy(u)) - float(torch.log(torch.tensor(50.0)))) < 1e-5
    a = torch.randn(7)
    assert abs(float(cosine(a, 2 * a)) - 1) < 1e-6
    R = torch.randn(4, 3, 8)
    assert successive_cosine_change(R).shape == (3, 3)
    assert torch.allclose(relative_delta(R, R), torch.zeros(4, 3))
    lg = torch.tensor([0.1, 3.0, 2.0, 5.0])
    assert int(rank_of(lg, 3)) == 1 and int(rank_of(lg, 2)) == 3
