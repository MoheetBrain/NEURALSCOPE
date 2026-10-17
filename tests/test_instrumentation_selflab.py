"""SelfLab instrumentation: chat tokenisation, measurement primitives, Experiment C causal-state checks.

Fast tests use the DEMO fixture (random-weight Qwen2 architecture, real Qwen tokenizer); ``slow`` tests load the
real Qwen2.5-0.5B-Instruct (run them under ``scripts/model_lock.py``).
"""

from __future__ import annotations

import sqlite3
import random

import pytest
import torch

from neuralscope.experiments.service import MAX_PROMPT_TOKENS_HARD_CAP, ServiceError
from neuralscope.experiments.store import ExperimentStore
from neuralscope.instrumentation import HeadScaleEdit, HeadSetEdit, MLPScaleEdit, ResidAddEdit, ResidSetEdit
from neuralscope.selflab.measure import (
    assistant_continuation_ids,
    continuation_logprobs,
    differing_positions,
    greedy,
    last_hook_stats,
    option_logprobs,
    patch_restoration,
)
from tests.conftest import PROMPT

SYSTEM = "You are a careful assistant. Answer from the document when one is given."
IM_END = 151645


def _user(c: str) -> dict:
    return {"role": "user", "content": c}


def _chat(question: str = "What colour is the Velbrook lantern?", document: str | None = None,
          hint: str | None = None) -> list[dict]:
    from neuralscope.selflab.compose import chat_user_message

    content, _ = chat_user_message(question, document=document, hint=hint, message_index=1)
    return [{"role": "system", "content": SYSTEM}, _user(content)]


def _manual_cont_logprobs(adapter, prefix, cont, edits=None) -> list[float]:
    """Reference: one plain teacher-forced capture of prefix+cont, log-softmax, gather the continuation tokens."""
    lg = adapter.capture(prefix + cont, "minimal", edits).logits
    lp = torch.log_softmax(lg.double(), -1)
    P = len(prefix)
    return [float(lp[P - 1 + i, t]) for i, t in enumerate(cont)]


# ============================================================================== tokenize_messages
def test_tokenize_messages_identity_single_user(demo_adapter):
    tc = demo_adapter.tokenize_messages([_user(PROMPT)])
    tp = demo_adapter.tokenize(PROMPT)
    assert tc.input_ids == tp.input_ids and tc.text == tp.text
    assert tc.message_token_spans == [tp.user_span] and tc.span_exact == [True]
    # with a system prompt, too (both paths use the same template)
    tc2 = demo_adapter.tokenize_messages([{"role": "system", "content": SYSTEM}, _user(PROMPT)])
    tp2 = demo_adapter.tokenize(PROMPT, system_prompt=SYSTEM)
    assert tc2.input_ids == tp2.input_ids and tc2.message_token_spans[1] == tp2.user_span


def test_tokenize_messages_spans_decode_to_content(demo_adapter):
    msgs = _chat(document="The Velbrook lantern is teal.", hint="I think it might be amber.")
    msgs += [{"role": "assistant", "content": "teal"}, _user("Where did that answer come from?")]
    tc = demo_adapter.tokenize_messages(msgs)
    assert tc.text.endswith("<|im_start|>assistant\n")
    for m, span, cs, ex in zip(msgs, tc.message_token_spans, tc.message_char_spans, tc.span_exact):
        assert span is not None and ex
        assert tc.text[cs[0]:cs[1]] == m["content"]
        assert demo_adapter.decode(tc.input_ids[span[0]:span[1]]) == m["content"]
    spans = tc.message_token_spans
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))  # ordered, disjoint


def test_tokenize_messages_trailing_assistant_multi_turn(demo_adapter):
    msgs = _chat() + [{"role": "assistant", "content": "insufficient evidence"}]
    closed = demo_adapter.tokenize_messages(msgs, add_generation_prompt=False)
    assert closed.text.endswith("insufficient evidence<|im_end|>\n") and not closed.add_generation_prompt
    s, e = closed.message_token_spans[2]
    assert demo_adapter.decode(closed.input_ids[s:e]) == "insufficient evidence"
    assert closed.input_ids[e] == IM_END
    nxt = demo_adapter.tokenize_messages(msgs + [_user("Why?")])
    assert nxt.input_ids[:len(closed.input_ids)] == closed.input_ids  # the next turn extends the closed prefix
    assert nxt.message_token_spans[:3] == closed.message_token_spans


def test_tokenize_messages_located_by_construction_not_first_match(demo_adapter):
    # "You" occurs in Qwen's default system prompt and "d" occurs inside "<|im_end|>" — a naive search would
    # place these contents in template text.
    msgs = [_user("You"), {"role": "assistant", "content": "d"}, _user("assistant")]
    tc = demo_adapter.tokenize_messages(msgs)
    assert "You are Qwen" in tc.text
    for m, (s, e) in zip(msgs, tc.message_token_spans):
        assert demo_adapter.decode(tc.input_ids[s:e]) == m["content"]
        assert tc.text[tc.offsets[s - 1][0]:tc.offsets[s - 1][1]].endswith("\n")  # preceded by the role header
    assert tc.message_token_spans[0][0] > 15  # after the injected default system prompt


def test_tokenize_messages_boundary_straddle_is_reported(demo_adapter):
    tc = demo_adapter.tokenize_messages([_user("\n\nstarts with newlines")])
    assert tc.message_token_spans[0] is not None and tc.span_exact == [False]


def test_tokenize_messages_rejects_bad_input(demo_adapter):
    with pytest.raises(ValueError):
        demo_adapter.tokenize_messages([])
    with pytest.raises(ValueError):
        demo_adapter.tokenize_messages([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError):
        demo_adapter.tokenize_messages([{"role": "user", "content": 3}])
    tc = demo_adapter.tokenize_messages([{"role": "system", "content": ""}, _user("hi")])
    assert tc.message_token_spans[0] is None and tc.message_token_spans[1] is not None


def test_char_span_to_tokens_round_trip(demo_adapter):
    from neuralscope.selflab.compose import chat_user_message

    content, segs = chat_user_message("What colour is the Velbrook lantern? Reply with only the colour.",
                                      document="Field notes, 1912: the Velbrook lantern was painted teal.",
                                      hint="A friend said it was amber.", message_index=1)
    msgs = [{"role": "system", "content": SYSTEM}, _user(content)]
    tc = demo_adapter.tokenize_messages(msgs)
    for seg in segs:  # every composed segment maps to tokens that decode to (a superset of) its text
        s, e = tc.char_span_to_tokens(1, seg.char_start, seg.char_end)
        assert content[seg.char_start:seg.char_end] in demo_adapter.decode(tc.input_ids[s:e])
    rng = random.Random(0)
    for _ in range(200):
        a = rng.randrange(0, len(content))
        b = rng.randrange(a + 1, len(content) + 1)
        s, e = tc.char_span_to_tokens(1, a, b)
        assert content[a:b] in demo_adapter.decode(tc.input_ids[s:e])
        # minimal: dropping the first or last token loses part of the span
        if e - s > 1:
            assert content[a:b] not in demo_adapter.decode(tc.input_ids[s + 1:e])
            assert content[a:b] not in demo_adapter.decode(tc.input_ids[s:e - 1])
    whole = tc.char_span_to_tokens(1, 0, len(content))
    assert whole == tc.message_token_spans[1]
    assert tc.char_span_to_tokens(1, 5, 5) is None
    with pytest.raises(ValueError):
        tc.char_span_to_tokens(1, 0, len(content) + 1)
    with pytest.raises(ValueError):
        tc.char_span_to_tokens(7, 0, 1)


# ============================================================================== measurement primitives
def test_continuation_logprobs_equals_manual_full_forward(demo_adapter):
    a = demo_adapter
    prefix = a.tokenize_messages(_chat()).input_ids
    conts = [a.encode("teal"), a.encode("insufficient evidence") + [IM_END], a.encode(" a much longer candidate answer")]
    assert len({len(c) for c in conts}) == 3 and all(len(c) >= 1 for c in conts)
    scores = continuation_logprobs(a, prefix, conts)
    for c, sc in zip(conts, scores):
        ref = _manual_cont_logprobs(a, prefix, c)
        assert sc.token_ids == c and sc.n_tokens == len(c) and len(sc.token_logprobs) == len(c)
        assert sc.token_strs == a.token_strings(c)
        assert torch.allclose(torch.tensor(sc.token_logprobs), torch.tensor(ref), atol=1e-4)
        assert sc.sum_logprob == pytest.approx(sum(ref), abs=5e-4)
        assert sc.mean_logprob == pytest.approx(sc.sum_logprob / len(c))
    # multi-token: the sum is not the first-token log-prob
    long = scores[2]
    assert long.n_tokens > 1 and long.sum_logprob < long.token_logprobs[0] - 1e-3


def test_continuation_logprobs_under_edits_uses_absolute_positions(demo_adapter):
    a = demo_adapter
    prefix = a.tokenize_messages(_chat()).input_ids
    P = len(prefix)
    conts = [a.encode(" teal lantern"), a.encode(" amber")]
    edits = [HeadScaleEdit(1, 0, 0.0, positions=[P - 3]), ResidAddEdit(2, torch.randn(a.info.d_model) * 2, [P])]
    scores = continuation_logprobs(a, prefix, conts, edits)
    # hit count: each edit fires once per batch row (2 rows), in the single batched forward call
    assert last_hook_stats(a) == {"n_edits_applied": 4, "n_forward_calls": 1, "n_rows": 2, "n_edits": 4}
    for c, sc in zip(conts, scores):
        assert torch.allclose(torch.tensor(sc.token_logprobs), torch.tensor(_manual_cont_logprobs(a, prefix, c, edits)),
                              atol=1e-4)
    base = continuation_logprobs(a, prefix, conts)
    assert abs(base[0].sum_logprob - scores[0].sum_logprob) > 1e-4  # the edits matter


def test_continuation_logprobs_rejects_bad_input(demo_adapter):
    with pytest.raises(ValueError):
        continuation_logprobs(demo_adapter, [], [[1]])
    with pytest.raises(ValueError):
        continuation_logprobs(demo_adapter, [1, 2], [[]])
    with pytest.raises(ValueError):
        continuation_logprobs(demo_adapter, [1, 2], [[10 ** 9]])
    assert continuation_logprobs(demo_adapter, [1, 2], []) == []


def test_assistant_continuation_ids_matches_template(demo_adapter):
    a = demo_adapter
    for text in ["teal", "insufficient evidence", "The answer is 95.", " leading space", "multi\nline"]:
        ids = assistant_continuation_ids(a, text)
        assert ids[-1] == IM_END and a.decode(ids[:-1]) == text
        assert assistant_continuation_ids(a, text, end_of_turn=False) == ids[:-1]
        # verify against a full re-templated conversation: prefix + continuation is a prefix of the rendered turn
        msgs = _chat()
        prefix = a.tokenize_messages(msgs).input_ids
        full = a.tokenize_messages(msgs + [{"role": "assistant", "content": text}], add_generation_prompt=False)
        assert full.input_ids[:len(prefix) + len(ids)] == prefix + ids
    with pytest.raises(ValueError):
        assistant_continuation_ids(a, "\nnewline-first content merges with the header newline")
    with pytest.raises(ValueError):
        assistant_continuation_ids(a, "")


def test_option_logprobs(demo_adapter):
    a = demo_adapter
    prefix = a.tokenize_messages(_chat()).input_ids
    opts = [a.encode(x)[0] for x in ("A", "B", "C", "D")]
    assert len(set(opts)) == 4
    r = option_logprobs(a, prefix, opts)
    lp = torch.log_softmax(a.capture(prefix, "minimal").logits[-1].double(), -1)
    assert torch.allclose(torch.tensor(r["logprobs"]), lp[opts].float(), atol=1e-5)
    assert r["mass"] == pytest.approx(float(lp[opts].exp().sum()), rel=1e-4)
    assert sum(r["probs_renorm"]) == pytest.approx(1.0)
    assert r["argmax_option"] == int(torch.argmax(lp[opts]))
    assert r["top_token_id"] == int(torch.argmax(lp)) and r["n_edits_applied"] == 0
    e = option_logprobs(a, prefix, opts, [ResidAddEdit(1, torch.randn(a.info.d_model), [len(prefix) - 1])])
    assert e["n_edits_applied"] == 1 and e["logprobs"] != r["logprobs"]
    with pytest.raises(ValueError):
        option_logprobs(a, prefix, [opts[0], opts[0]])
    with pytest.raises(ValueError):
        option_logprobs(a, prefix, [])


def test_greedy_wrapper(demo_adapter):
    ids = demo_adapter.tokenize(PROMPT).input_ids
    g = greedy(demo_adapter, ids, 5)
    assert g.generated_ids == demo_adapter.generate(ids, 5).generated_ids
    assert g.n_edits_applied == 0 and len(g.generated_logprobs) == len(g.generated_ids)
    # decision margins: top-1 minus top-2 logit of the teacher-forced logits at each step
    lg = demo_adapter.capture(ids + g.generated_ids, "minimal").logits
    P = len(ids)
    ref = [float(torch.topk(lg[P - 1 + k], 2).values.diff().neg()) for k in range(len(g.generated_ids))]
    assert len(g.generated_margins) == len(g.generated_ids) and all(m >= 0 for m in g.generated_margins)
    assert torch.allclose(torch.tensor(g.generated_margins), torch.tensor(ref), atol=1e-4)


def test_differing_positions():
    assert differing_positions([1, 2, 3, 4], [1, 9, 3, 8]) == [1, 3]
    assert differing_positions([], []) == []
    with pytest.raises(ValueError):
        differing_positions([1, 2], [1, 2, 3])


def _patch_pair(a):
    clean = a.tokenize(PROMPT).input_ids
    i = clean.index(a.encode("8")[0])  # "37 + 58" → "37 + 59"
    corrupt = list(clean)
    corrupt[i] = a.encode("9")[0]
    r = len(clean) - 1
    lc = a.capture(clean, "minimal").logits[r]
    lx = a.capture(corrupt, "minimal").logits[r]
    tc = int(lc.argmax())
    order = torch.argsort(lx, descending=True).tolist()
    tx = next(t for t in order if t != tc)
    return clean, corrupt, i, r, tc, tx


def test_patch_restoration_full_restore_and_null(demo_adapter):
    a = demo_adapter
    clean, corrupt, i, r, tc, tx = _patch_pair(a)
    L = a.info.n_layers
    positions = [i - 2, i, r]
    res = patch_restoration(a, clean, corrupt, positions, list(range(L + 1)), r, tc, tx)
    assert res["defined"] and res["diff_positions"] == [i]
    assert abs(res["denominator"]) > 1e-3
    R = res["restoration"]
    # depth 0 at the only differing position makes the corrupt input identical to the clean one → 1
    assert R[0][1] == pytest.approx(1.0, abs=1e-3)
    # final depth at the readout position: the unembedded state IS the clean state → 1
    assert R[L][2] == pytest.approx(1.0, abs=1e-3)
    # a position before the difference holds identical clean/corrupt states at every depth → 0
    assert all(R[d][0] == pytest.approx(0.0, abs=1e-3) for d in range(L + 1))
    # joint patch of all positions at depth 0 → 1
    assert res["restoration_joint"][0] == pytest.approx(1.0, abs=1e-3)
    # resuming from the cached corrupt residual equals re-running the full forward pass
    full = patch_restoration(a, clean, corrupt, positions, list(range(L + 1)), r, tc, tx, resume=False)
    assert torch.allclose(torch.tensor(full["ld_patched"]), torch.tensor(res["ld_patched"]), atol=1e-4)
    assert res["ld_corrupt"] == pytest.approx(float(a.capture(corrupt, "minimal").logits[r, tc]
                                                   - a.capture(corrupt, "minimal").logits[r, tx]), abs=1e-4)


def test_patch_restoration_self_patch(demo_adapter):
    """corrupt→corrupt (identical inputs): the patch is a no-op, LD_patched = LD_corrupt; restoration undefined."""
    a = demo_adapter
    clean, corrupt, i, r, tc, tx = _patch_pair(a)
    res = patch_restoration(a, corrupt, corrupt, [i, r], [0, 2, a.info.n_layers], r, tc, tx)
    assert not res["defined"] and res["diff_positions"] == []
    assert all(v is None for row in res["restoration"] for v in row)
    for row in res["ld_patched"]:
        for v in row:
            assert v == pytest.approx(res["ld_corrupt"], abs=1e-4)


def test_patch_restoration_refusals(demo_adapter):
    a = demo_adapter
    clean, corrupt, i, r, tc, tx = _patch_pair(a)
    with pytest.raises(ValueError, match="lengths"):
        patch_restoration(a, clean, corrupt[:-1], [i], [0], r - 1, tc, tx)
    shifted = clean[1:] + [clean[-1]]  # equal length but every token moved: misaligned
    with pytest.raises(ValueError, match="aligned"):
        patch_restoration(a, clean, shifted, [i], [0], r, tc, tx)
    with pytest.raises(ValueError, match="readout"):
        patch_restoration(a, clean, corrupt, [r], [0], r - 1, tc, tx)
    with pytest.raises(ValueError):
        patch_restoration(a, clean, corrupt, [i], [a.info.n_layers + 1], r, tc, tx)
    with pytest.raises(ValueError):
        patch_restoration(a, clean, corrupt, [i], [0], r, tc, tc)


# ============================================================================== Experiment C: causal-state checks
def test_sham_edits_are_bit_identical(demo_adapter):
    a = demo_adapter
    ids = a.tokenize_messages(_chat()).input_ids
    P = len(ids)
    sham = [HeadScaleEdit(1, 2, 1.0), ResidAddEdit(2, torch.zeros(a.info.d_model)),
            ResidAddEdit(3, torch.zeros(a.info.d_model), positions=list(range(P - 6, P))),
            MLPScaleEdit(0, 1.0, positions=[P - 2])]
    base, sh = a.capture(ids, "standard"), a.capture(ids, "standard", sham)
    assert torch.equal(base.logits, sh.logits) and torch.equal(base.resid, sh.resid)
    assert torch.equal(base.head_z, sh.head_z) and torch.equal(base.attn, sh.attn)
    g0, g1 = a.generate(ids, 8), a.generate(ids, 8, sham)
    assert g0.generated_ids == g1.generated_ids and g0.generated_logprobs == g1.generated_logprobs
    assert g1.n_edits_applied > 0  # the sham edits really fired
    conts = [a.encode(" teal"), a.encode(" insufficient evidence")]
    s0, s1 = continuation_logprobs(a, ids, conts), continuation_logprobs(a, ids, conts, sham)
    assert [s.token_logprobs for s in s0] == [s.token_logprobs for s in s1]


def test_prompt_position_edit_changes_only_downstream_states(demo_adapter):
    a = demo_adapter
    ids = a.tokenize_messages(_chat(document="The Velbrook lantern is teal.")).input_ids
    p = len(ids) - 12
    base = a.capture(ids, "full")
    for edit, first_depth in [(ResidAddEdit(2, torch.randn(a.info.d_model), positions=[p]), 2),
                              (HeadScaleEdit(1, 3, 0.0, positions=[p]), 2),
                              (MLPScaleEdit(2, 0.0, positions=[p]), 3),
                              (ResidSetEdit(0, p, torch.randn(a.info.d_model)), 0)]:
        e = a.capture(ids, "full", [edit])
        diff = (e.resid - base.resid).abs().amax(-1)  # [D, T]
        assert diff[:, :p].max() == 0, edit  # upstream positions: bit-identical at every depth
        if first_depth > 0:
            assert diff[:first_depth].max() == 0, edit  # depths before the edit: bit-identical everywhere
        assert diff[first_depth, p] > 0, edit  # the edited state itself changed
        assert diff[first_depth, p + 1:].max() == 0, edit  # same depth, later positions: not yet reached
        assert (diff[-1, p:] > 0).all(), edit  # ...but downstream depths see it through attention
        assert torch.equal(e.logits[:p], base.logits[:p])
        assert torch.equal(e.attn[:, :, :p], base.attn[:, :, :p])


def test_kv_cached_greedy_under_prompt_edits_equals_teacher_forced_rescoring(demo_adapter):
    """The manipulated prompt state reaches the generated report only through the KV cache: decoding under
    prompt-position edits must equal teacher-forced re-scoring of the same tokens under the same edits."""
    a = demo_adapter
    ids = a.tokenize_messages(_chat(document="The Velbrook lantern is teal.")).input_ids
    P = len(ids)
    edits = [ResidAddEdit(2, torch.randn(a.info.d_model) * 3, positions=list(range(P - 15, P - 10))),
             HeadScaleEdit(3, 1, 0.0, positions=[P - 4, P - 1])]
    g = greedy(a, ids, 10, edits)
    assert g.n_edits_applied == 2  # both edits fire once, during the prefill; decode steps add no hits
    st = last_hook_stats(a)
    assert st["n_forward_calls"] == len(g.generated_ids)  # prefill + (n-1) decode steps
    sc = continuation_logprobs(a, ids, [g.generated_ids], edits)[0]
    assert torch.allclose(torch.tensor(sc.token_logprobs, dtype=torch.float64),
                          torch.tensor(g.generated_logprobs, dtype=torch.float64),
                          atol=1e-4)
    tf = a.capture(ids + g.generated_ids, "minimal", edits).logits
    assert [int(tf[P - 1 + k].argmax()) for k in range(len(g.generated_ids))] == g.generated_ids
    # and the edits are not a no-op on this prompt
    assert greedy(a, ids, 10).generated_logprobs != g.generated_logprobs


def test_hook_hit_counts(demo_adapter):
    a = demo_adapter
    ids = a.tokenize(PROMPT).input_ids
    P = len(ids)
    n = 6
    # all positions: one hit per forward call (prefill + n-1 decode steps), if generation is not cut short
    g = a.generate(ids, n, [HeadScaleEdit(1, 0, 0.5)])
    calls = len(g.generated_ids)
    assert g.n_edits_applied == calls and last_hook_stats(a)["n_forward_calls"] == calls
    # two all-position edits → 2 per call
    assert a.generate(ids, n, [HeadScaleEdit(1, 0, 0.5), MLPScaleEdit(2, 0.5)]).n_edits_applied == 2 * calls
    # a prompt position fires once (prefill); a decode position fires once, in the step that processes it
    assert a.generate(ids, n, [ResidAddEdit(1, torch.ones(a.info.d_model), positions=[P - 2])]).n_edits_applied == 1
    gd = a.generate(ids, n, [HeadScaleEdit(2, 1, 0.0, positions=[P + 1])])
    assert gd.n_edits_applied == (1 if len(gd.generated_ids) >= 3 else 0)
    # a position never processed never fires
    assert a.generate(ids, n, [MLPScaleEdit(0, 0.0, positions=[P + 100])]).n_edits_applied == 0
    # HeadSetEdit with per-position vectors counts like any other edit
    hs = HeadSetEdit(2, 1, torch.zeros(P + n, a.info.d_head))
    assert a.generate(ids, n, [hs]).n_edits_applied == len(a.generate(ids, n, [hs]).generated_ids)
    # batched teacher-forced call: one hit per (edit, row) in a single forward call
    a.final_logits_batch(ids, [[HeadScaleEdit(1, 0, 0.0)], [], [ResidSetEdit(1, 3, torch.zeros(a.info.d_model))]],
                         logit_positions=[P - 1])
    assert last_hook_stats(a) == {"n_edits_applied": 2, "n_forward_calls": 1, "n_rows": 3, "n_edits": 2}
    # resumed batch counts as one forward call
    tr = a.capture(ids, "minimal")
    a.final_logits_batch(ids, [[HeadScaleEdit(2, 0, 0.0)]], start_depth=2, resid_in=tr.resid[2])
    assert last_hook_stats(a) == {"n_edits_applied": 1, "n_forward_calls": 1, "n_rows": 1, "n_edits": 1}


def test_logit_positions_and_sequence_logits_match_full(demo_adapter):
    a = demo_adapter
    ids = a.tokenize(PROMPT).input_ids
    T = len(ids)
    sets = [[], [HeadScaleEdit(2, 1, 0.0)]]
    full = a.final_logits_batch(ids, sets)
    pos = [0, T // 2, T - 1]
    part = a.final_logits_batch(ids, sets, logit_positions=pos)
    assert part.shape == (2, 3, a.info.vocab_size) and torch.allclose(part, full[:, pos], atol=1e-5)
    tr = a.capture(ids, "minimal")
    res = a.final_logits_batch(ids, sets[1:], start_depth=2, resid_in=tr.resid[2], logit_positions=pos)
    assert torch.allclose(res[0], full[1, pos], atol=1e-5)
    with pytest.raises(ValueError):
        a.final_logits_batch(ids, sets, logit_positions=[T])
    seqs = [ids, ids[:-5], ids + a.encode(" extra tokens")]
    out = a.sequence_logits(seqs, [[len(s) - 1, 0] for s in seqs])
    for s, o in zip(seqs, out):
        ref = a.capture(s, "minimal").logits
        assert torch.allclose(o, ref[[len(s) - 1, 0]], atol=1e-5)


# ============================================================================== service: messages runs
def test_create_run_with_messages(service, demo_adapter):
    msgs = _chat(document="The Velbrook lantern is teal.", hint="I think it is amber.")
    rec = service.create_run(messages=msgs, max_new_tokens=4, trace_level="minimal")
    tc = demo_adapter.tokenize_messages(msgs)
    assert rec["messages"] == msgs and rec["chat"] and rec["prompt"] == msgs[1]["content"]
    assert rec["system_prompt"] == SYSTEM and rec["prompt_text"] == tc.text
    assert rec["input_ids"][:rec["prompt_len"]] == tc.input_ids
    assert tuple(rec["user_span"]) == tc.message_token_spans[1]
    s, e = rec["user_span"]
    assert demo_adapter.decode(rec["input_ids"][s:e]) == msgs[1]["content"]
    d = service.run_detail(rec["id"])  # the /app?run= payload
    assert d["messages"] == msgs and len(d["tokens"]) == len(rec["input_ids"])
    assert {t["role"] for t in d["tokens"][s:e]} == {"user"} and d["tokens"][0]["role"] == "template"
    # intervention children keep the chat
    child = service.run_intervention(rec["id"], [{"type": "head_ablation", "layer": 1, "head": 0}])
    assert service.store.get_run(child["run"]["id"])["messages"] == msgs
    # legacy runs are unchanged and read messages as None
    legacy = service.create_run(PROMPT, max_new_tokens=2, trace_level="minimal")
    assert legacy["messages"] is None and legacy["prompt"] == PROMPT
    assert legacy["input_ids"][:legacy["prompt_len"]] == demo_adapter.tokenize(PROMPT).input_ids


def test_create_run_messages_multi_turn_and_validation(service):
    msgs = _chat() + [{"role": "assistant", "content": "teal"}, _user("Which source did you use?")]
    rec = service.create_run(messages=msgs, max_new_tokens=2, trace_level="minimal")
    assert rec["prompt"] == "Which source did you use?"
    with pytest.raises(ServiceError):
        service.create_run(messages=msgs[:3])  # must end with a user message
    with pytest.raises(ServiceError):
        service.create_run(messages=[{"role": "robot", "content": "x"}])
    with pytest.raises(ServiceError):
        service.create_run(messages=_chat(), system_prompt="dup")
    with pytest.raises(ServiceError):
        service.create_run(messages=[_user("  ")])


def test_create_run_max_prompt_tokens_override(service, settings):
    long_doc = " ".join(f"Item {i} of the ledger is entry number {i * 7}." for i in range(30))
    msgs = _chat(document=long_doc)
    assert settings.max_prompt_tokens == 160
    with pytest.raises(ServiceError, match="limit is 160"):
        service.create_run(messages=msgs, max_new_tokens=1, trace_level="minimal")
    rec = service.create_run(messages=msgs, max_new_tokens=1, trace_level="minimal", max_prompt_tokens=600)
    assert 160 < rec["prompt_len"] <= 600
    with pytest.raises(ServiceError):
        service.create_run(messages=msgs, max_prompt_tokens=MAX_PROMPT_TOKENS_HARD_CAP + 1)
    with pytest.raises(ServiceError):
        service.create_run(messages=msgs, max_prompt_tokens=0)
    assert settings.max_prompt_tokens == 160  # the global default is untouched


def test_store_migrates_old_database(tmp_path):
    from neuralscope.experiments import store as store_mod

    d = tmp_path / "old"
    d.mkdir()
    old_schema = store_mod.SCHEMA.replace("trace_consistent INTEGER,\n    messages TEXT\n", "trace_consistent INTEGER\n")
    assert "messages" not in old_schema  # the pre-SelfLab schema
    con = sqlite3.connect(str(d / "neuralscope.db"))
    con.executescript(old_schema)
    con.execute("INSERT INTO runs (id, created_at, kind, model_id, prompt, prompt_text, input_ids, prompt_len, "
                "trace_level) VALUES ('run_old', 't', 'baseline', 'm', 'p', 'p', '[1,2]', 2, 'minimal')")
    con.commit()
    con.close()
    st = ExperimentStore(d)
    old = st.get_run("run_old")
    assert old["messages"] is None and old["input_ids"] == [1, 2]
    st.update_run("run_old", messages=[{"role": "user", "content": "p"}])
    assert st.get_run("run_old")["messages"] == [{"role": "user", "content": "p"}]
    ExperimentStore(d)  # idempotent re-open


# ============================================================================== slow: real Qwen2.5-0.5B-Instruct
@pytest.mark.slow
def test_real_tokenize_messages_identity(real_adapter):
    a = real_adapter
    assert a.tokenize_messages([_user(PROMPT)]).input_ids == a.tokenize(PROMPT).input_ids
    msgs = _chat(document="The Velbrook lantern is teal.", hint="I think it is amber.")
    tc = a.tokenize_messages(msgs)
    for m, (s, e) in zip(msgs, tc.message_token_spans):
        assert a.decode(tc.input_ids[s:e]) == m["content"]


@pytest.mark.slow
def test_real_continuation_logprobs_and_options(real_adapter):
    a = real_adapter
    msgs = _chat(question="What colour is the Velbrook lantern? Reply with only the colour, or exactly "
                          "'insufficient evidence'.", document="Field notes: the Velbrook lantern is teal.")
    prefix = a.tokenize_messages(msgs).input_ids
    cands = ["teal", "amber", "insufficient evidence"]
    conts = [assistant_continuation_ids(a, c) for c in cands]
    scores = continuation_logprobs(a, prefix, conts)
    for c, sc in zip(conts, scores):
        assert sc.n_tokens == len(c) >= 2  # content + <|im_end|>
        ref = _manual_cont_logprobs(a, prefix, c)
        assert torch.allclose(torch.tensor(sc.token_logprobs), torch.tensor(ref), atol=2e-4)
    assert scores[0].sum_logprob > scores[1].sum_logprob  # the document value beats the alternative
    g = greedy(a, prefix, 8)
    # what the model generated, re-derived through the template, is the same token sequence
    assert g.stopped_by == "eos"
    assert assistant_continuation_ids(a, a.decode(g.generated_ids[:-1])) == g.generated_ids
    opts = [a.encode(x)[0] for x in ("A", "B", "C", "D")]
    q = msgs + [{"role": "assistant", "content": a.decode(g.generated_ids[:-1])},
                _user("Where did your answer come from? A) the document B) my suggestion C) prior knowledge "
                      "D) not enough information. Reply with one letter.")]
    r = option_logprobs(a, a.tokenize_messages(q).input_ids, opts)
    lp = torch.log_softmax(a.capture(a.tokenize_messages(q).input_ids, "minimal").logits[-1].double(), -1)
    assert torch.allclose(torch.tensor(r["logprobs"]), lp[opts].float(), atol=1e-4)
    assert 0 < r["mass"] <= 1 and sum(r["probs_renorm"]) == pytest.approx(1.0)


@pytest.mark.slow
def test_real_patch_restoration_and_kv_consistency(real_adapter):
    a = real_adapter
    clean, corrupt, i, r, tc, tx = _patch_pair(a)
    L = a.info.n_layers
    res = patch_restoration(a, clean, corrupt, [i - 1, i, r], [0, L // 2, L], r, tc, tx)
    assert res["defined"] and differing_positions(clean, corrupt) == [i]
    assert res["restoration"][0][1] == pytest.approx(1.0, abs=1e-3)  # depth 0 at the differing token
    assert res["restoration"][2][2] == pytest.approx(1.0, abs=1e-3)  # final depth at the readout
    assert all(abs(res["restoration"][d][0]) < 1e-3 for d in range(3))  # before the difference: 0
    same = patch_restoration(a, corrupt, corrupt, [i, r], [0, L], r, tc, tx)
    assert not same["defined"]
    assert all(v == pytest.approx(same["ld_corrupt"], abs=1e-3) for row in same["ld_patched"] for v in row)
    # Experiment C on the real model: cached decoding under prompt edits == teacher-forced re-scoring
    ids = a.tokenize_messages(_chat(document="The Velbrook lantern is teal.")).input_ids
    P = len(ids)
    edits = [HeadScaleEdit(l, h, 0.0, positions=list(range(P - 20, P))) for l in (10, 14) for h in range(4)]
    g = greedy(a, ids, 12, edits)
    assert g.n_edits_applied == len(edits)
    sc = continuation_logprobs(a, ids, [g.generated_ids], edits)[0]
    assert torch.allclose(torch.tensor(sc.token_logprobs, dtype=torch.float64),
                          torch.tensor(g.generated_logprobs, dtype=torch.float64),
                          atol=2e-3)
