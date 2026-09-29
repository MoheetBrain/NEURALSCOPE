"""Instrumentation invariants (fixture model: fast, exact)."""

from __future__ import annotations

import torch

from neuralscope.instrumentation import HeadScaleEdit, MLPScaleEdit, ResidAddEdit, ResidSetEdit
from tests.conftest import PROMPT


def _ids(a):
    return a.tokenize(PROMPT).input_ids


def test_tokenizer_and_user_span(demo_adapter):
    tp = demo_adapter.tokenize(PROMPT)
    assert tp.input_ids and tp.user_span is not None
    s, e = tp.user_span
    assert "".join(demo_adapter.token_strings(tp.input_ids[s:e])) == PROMPT
    # digits are single tokens in the Qwen vocabulary
    toks = demo_adapter.token_strings(tp.input_ids[s:e])
    assert "3" in toks and "7" in toks and "5" in toks and "8" in toks
    raw = demo_adapter.tokenize("37 + 58 =", chat=False)
    assert raw.user_span == (0, len(raw.input_ids))


def test_model_info(demo_adapter):
    i = demo_adapter.info
    assert i.is_demo and i.n_layers == 4 and i.n_heads == 4 and i.n_kv_heads == 2
    assert i.d_head * i.n_heads == i.d_model
    assert i.capabilities["head_ablation"] and not i.capabilities["sae_features"]


def test_deterministic_generation(demo_adapter):
    ids = _ids(demo_adapter)
    g1 = demo_adapter.generate(ids, 6)
    g2 = demo_adapter.generate(ids, 6)
    assert g1.generated_ids == g2.generated_ids
    assert len(g1.generated_ids) == 6 or g1.stopped_by == "eos"


def test_capture_shapes(demo_adapter):
    ids = _ids(demo_adapter)
    i = demo_adapter.info
    T = len(ids)
    tr = demo_adapter.capture(ids, "full")
    assert tr.resid.shape == (i.n_layers + 1, T, i.d_model)
    assert tr.attn.shape == (i.n_layers, i.n_heads, T, T)
    assert tr.head_z.shape == (i.n_layers, T, i.n_heads, i.d_head)
    assert tr.mlp_out.shape == (i.n_layers, T, i.d_model)
    assert tr.logits.shape == (T, i.vocab_size)
    minimal = demo_adapter.capture(ids, "minimal")
    assert minimal.attn is None and minimal.head_z is None and minimal.mlp_out is None
    assert torch.equal(minimal.resid, tr.resid)


def test_attention_is_causal_and_normalised(demo_adapter):
    tr = demo_adapter.capture(_ids(demo_adapter), "standard")
    A = tr.attn
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-5)
    T = A.shape[-1]
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    assert A[..., upper].abs().max() == 0


def test_depth_zero_is_embedding(demo_adapter):
    ids = _ids(demo_adapter)
    tr = demo_adapter.capture(ids, "minimal")
    emb = demo_adapter.model.model.embed_tokens(torch.tensor(ids)).detach()
    assert torch.allclose(tr.resid[0], emb)


def test_token_layer_indexing_is_causal(demo_adapter):
    """Changing token t changes resid[d, t'] only for t' >= t (for d >= 1), never earlier positions."""
    ids = _ids(demo_adapter)
    t = len(ids) - 6
    ids2 = list(ids)
    ids2[t] = demo_adapter.encode("9")[0] if ids[t] != demo_adapter.encode("9")[0] else demo_adapter.encode("1")[0]
    a, b = demo_adapter.capture(ids, "minimal"), demo_adapter.capture(ids2, "minimal")
    diff = (a.resid - b.resid).abs().amax(-1)  # [D, T]
    assert diff[:, :t].max() == 0
    assert (diff[:, t] > 0).all()
    assert diff[0, t + 1:].max() == 0  # embeddings of other tokens unchanged
    assert (diff[1:, t + 1:].amax(0) > 0).all()  # later positions see it through attention


def test_final_logit_lens_equals_logits(demo_adapter):
    tr = demo_adapter.capture(_ids(demo_adapter), "minimal")
    assert torch.allclose(demo_adapter.logit_lens(tr.resid[-1]), tr.logits, atol=1e-4)


def test_residual_decomposition_exact(demo_adapter):
    tr = demo_adapter.capture(_ids(demo_adapter), "full")
    for l in range(demo_adapter.info.n_layers):
        ao = demo_adapter.attn_out_from_z(l, tr.head_z[l])
        assert torch.allclose(tr.resid[l] + ao + tr.mlp_out[l], tr.resid[l + 1], atol=1e-5)
    # per-head outputs sum to the attention output
    z = tr.head_z[1]
    per = sum(demo_adapter.head_output(1, h, z[:, h]) for h in range(demo_adapter.info.n_heads))
    assert torch.allclose(per, demo_adapter.attn_out_from_z(1, z), atol=1e-5)


def test_head_ablation_modifies_forward_pass_downstream_only(demo_adapter):
    ids = _ids(demo_adapter)
    base = demo_adapter.capture(ids, "standard")
    abl = demo_adapter.capture(ids, "standard", [HeadScaleEdit(layer=2, head=1, factor=0.0)])
    d = (abl.resid - base.resid).abs().amax(dim=(1, 2))
    assert d[:3].max() == 0  # depths 0..2 (before block 2's output) untouched
    assert d[3] > 0 and d[4] > 0
    assert abl.head_z[2, :, 1].abs().max() == 0
    assert torch.equal(abl.head_z[2, :, 0], base.head_z[2, :, 0])
    assert torch.equal(abl.attn[2], base.attn[2])  # the head's own pattern is unchanged
    assert not torch.allclose(abl.logits, base.logits)
    # Inside block 2 the only change is the removed head write: the post-attention residual differs by
    # exactly −W_O^{2,1} z^{2,1} (the MLP then sees a different input, so compare before the MLP).
    removed = demo_adapter.head_output(2, 1, base.head_z[2, :, 1])
    mid_base = base.resid[2] + demo_adapter.attn_out_from_z(2, base.head_z[2])
    mid_abl = abl.resid[2] + demo_adapter.attn_out_from_z(2, abl.head_z[2])
    assert torch.allclose(mid_abl - mid_base, -removed, atol=1e-5)


def test_head_scale_factor_one_is_identity(demo_adapter):
    ids = _ids(demo_adapter)
    base = demo_adapter.capture(ids, "minimal")
    same = demo_adapter.capture(ids, "minimal", [HeadScaleEdit(layer=1, head=2, factor=1.0)])
    assert torch.equal(base.logits, same.logits)


def test_position_specific_edits(demo_adapter):
    ids = _ids(demo_adapter)
    p = len(ids) - 3
    base = demo_adapter.capture(ids, "minimal")
    e = demo_adapter.capture(ids, "minimal", [HeadScaleEdit(layer=1, head=0, factor=0.0, positions=[p])])
    diff = (e.resid - base.resid).abs().amax(-1)
    assert diff[:, :p].max() == 0 and diff[2, p] > 0
    m = demo_adapter.capture(ids, "minimal", [MLPScaleEdit(layer=0, factor=0.0, positions=[p])])
    assert (m.resid - base.resid).abs().amax(-1)[:, :p].max() == 0


def test_resid_patch_and_steer(demo_adapter):
    ids = _ids(demo_adapter)
    base = demo_adapter.capture(ids, "minimal")
    same = demo_adapter.capture(ids, "minimal", [ResidSetEdit(depth=2, position=5, vector=base.resid[2, 5])])
    assert torch.equal(same.logits, base.logits)
    v = torch.randn(demo_adapter.info.d_model)
    pt = demo_adapter.capture(ids, "minimal", [ResidSetEdit(depth=2, position=5, vector=v)])
    assert torch.allclose(pt.resid[2, 5], v)
    assert torch.equal(pt.resid[2, :5], base.resid[2, :5]) and torch.equal(pt.resid[2, 6:], base.resid[2, 6:])
    half = demo_adapter.capture(ids, "minimal", [ResidSetEdit(depth=2, position=5, vector=v, alpha=0.5)])
    assert torch.allclose(half.resid[2, 5], 0.5 * base.resid[2, 5] + 0.5 * v, atol=1e-6)
    st = demo_adapter.capture(ids, "minimal", [ResidAddEdit(depth=1, vector=v, positions=[3, 4])])
    assert torch.allclose(st.resid[1, 3], base.resid[1, 3] + v, atol=1e-6)
    assert torch.equal(st.resid[1, 5], base.resid[1, 5])


def test_batched_edits_match_individual_runs(demo_adapter):
    ids = _ids(demo_adapter)
    sets = [[], [HeadScaleEdit(2, 1, 0.0)], [MLPScaleEdit(1, 0.0)]]
    lb = demo_adapter.final_logits_batch(ids, sets)
    for b, es in enumerate(sets):
        assert torch.allclose(lb[b], demo_adapter.capture(ids, "minimal", es).logits, atol=1e-5)


def test_resume_from_depth_is_exact(demo_adapter):
    ids = _ids(demo_adapter)
    tr = demo_adapter.capture(ids, "minimal")
    for sd in range(1, demo_adapter.info.n_layers):
        sets = [[HeadScaleEdit(sd, 0, 0.0)], []]
        full = demo_adapter.final_logits_batch(ids, sets)
        part = demo_adapter.final_logits_batch(ids, sets, start_depth=sd, resid_in=tr.resid[sd])
        assert torch.allclose(full, part, atol=1e-5)


def test_generation_under_intervention_matches_capture(demo_adapter):
    """Free-running generation under an edit is consistent with a teacher-forced capture under the same edit."""
    ids = _ids(demo_adapter)
    edits = [HeadScaleEdit(3, 2, 0.0)]
    g = demo_adapter.generate(ids, 5, edits)
    tr = demo_adapter.capture(ids + g.generated_ids, "minimal", edits)
    P = len(ids)
    assert [int(tr.logits[P - 1 + i].argmax()) for i in range(len(g.generated_ids))] == g.generated_ids
