"""SelfLab measurement primitives over a ``ModelAdapter`` (INTERFACES §3).

Every function here is deterministic and, unless it says otherwise, teacher-forced: the model reads a fixed token
sequence and we read numbers off its logits. All model calls go through the adapter's hook session, so compiled
edits (``HeadScaleEdit``, ``ResidSetEdit``, …) apply at ABSOLUTE token positions exactly as they do during
generation and capture. Each function holds ``adapter.lock`` for its whole duration so that the hook statistics in
``adapter.last_session_stats`` (e.g. ``n_edits_applied``) belong to the call that just returned.

Nothing here interprets text. Candidate strings are turned into token ids with the chat template
(``assistant_continuation_ids``) and scored as full multi-token continuations — never by their first token.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from neuralscope.instrumentation.adapter import Edit, GenerationResult, ModelAdapter, ResidSetEdit

# Default refusal threshold for "misaligned" counterfactual pairs: equal length but this fraction of positions or
# more differ (typical of a shifted / re-tokenised prompt, not a controlled single-span replacement).
MAX_DIFF_FRACTION = 0.25


@dataclass
class ContinuationScore:
    token_ids: list[int]
    token_strs: list[str]
    token_logprobs: list[float]  # log P(c_i | prefix, c_<i), natural log
    sum_logprob: float  # log P(c_1..c_k | prefix)
    n_tokens: int
    mean_logprob: float  # sum_logprob / n_tokens (a length-normalised view; the sum is the likelihood)

    def to_dict(self) -> dict[str, Any]:
        return {"token_ids": self.token_ids, "token_strs": self.token_strs, "token_logprobs": self.token_logprobs,
                "sum_logprob": self.sum_logprob, "n_tokens": self.n_tokens, "mean_logprob": self.mean_logprob}


def _lock(adapter: ModelAdapter):
    lk = getattr(adapter, "lock", None)
    return lk if lk is not None else contextlib.nullcontext()


def _vocab_check(adapter: ModelAdapter, ids: list[int], what: str) -> None:
    V = adapter.info.vocab_size
    bad = [int(i) for i in ids if not 0 <= int(i) < V]
    if bad:
        raise ValueError(f"{what}: token ids {bad[:5]} outside vocabulary of size {V}")


def last_hook_stats(adapter: ModelAdapter) -> dict | None:
    """Hook statistics of the adapter's most recent instrumented call (copy)."""
    st = getattr(adapter, "last_session_stats", None)
    return dict(st) if st else None


# ------------------------------------------------------------------------------------------------ scoring
def continuation_logprobs(adapter: ModelAdapter, prefix_ids: list[int], continuations: list[list[int]],
                          edits: list[Edit] | None = None) -> list[ContinuationScore]:
    """FULL multi-token scoring: log P(c_1..c_k | prefix) = Σ_i log P(c_i | prefix, c_<i), for each continuation.

    All continuations are scored in one right-padded teacher-forced batch (exact under causal attention) under
    ``edits``. Log-softmax is taken over the full vocabulary in float64.
    """
    prefix = [int(t) for t in prefix_ids]
    if not prefix:
        raise ValueError("prefix_ids is empty (the first continuation token needs a context)")
    conts = [[int(t) for t in c] for c in continuations]
    if any(not c for c in conts):
        raise ValueError("every continuation must contain at least one token")
    if not conts:
        return []
    _vocab_check(adapter, prefix, "prefix")
    for c in conts:
        _vocab_check(adapter, c, "continuation")
    P = len(prefix)
    seqs = [prefix + c for c in conts]
    positions = [list(range(P - 1, P - 1 + len(c))) for c in conts]
    with _lock(adapter):
        logits = adapter.sequence_logits(seqs, positions, list(edits or []))
    out: list[ContinuationScore] = []
    for c, lg in zip(conts, logits):
        lp = torch.log_softmax(lg.double(), dim=-1)
        per = [float(lp[i, t]) for i, t in enumerate(c)]
        s = float(math.fsum(per))
        out.append(ContinuationScore(token_ids=c, token_strs=adapter.token_strings(c), token_logprobs=per,
                                     sum_logprob=s, n_tokens=len(c), mean_logprob=s / len(c)))
    return out


def assistant_continuation_ids(adapter: ModelAdapter, text: str, end_of_turn: bool = True) -> list[int]:
    """The exact tokens the chat template produces for ``text`` as an assistant turn's content (+ end-of-turn).

    Derived from the template, not assumed: we render a probe conversation ending in the generation prompt and the
    same conversation with ``text`` as the assistant's reply, check that the first is a token-level prefix of the
    second, and take the content tokens (located by character offsets) followed — when ``end_of_turn`` — by the
    template's end-of-turn token (e.g. ``<|im_end|>`` for Qwen), which must be a stop token of the model.

    Raises ValueError if the template is not prefix-stable, re-writes the content, or tokenises across the
    content's boundaries (a token straddling the generation prompt and the content), because then "the
    continuation" is not well defined as a token sequence appended to the prefix.
    """
    if not text:
        raise ValueError("empty assistant text")
    probe = [{"role": "user", "content": "Hello."}]
    base = adapter.tokenize_messages(probe, add_generation_prompt=True)
    full = adapter.tokenize_messages(probe + [{"role": "assistant", "content": text}], add_generation_prompt=False)
    nb = len(base.input_ids)
    if not full.text.startswith(base.text) or full.input_ids[:nb] != base.input_ids:
        raise ValueError("chat template is not prefix-stable: the generation prompt is not a prefix of the "
                         "rendered assistant turn")
    cs = full.message_char_spans[1]
    span = full.message_token_spans[1]
    if cs is None or span is None or cs[0] != len(base.text) or full.text[cs[0]:cs[1]] != text:
        raise ValueError("the chat template does not render the assistant content verbatim right after the "
                         "generation prompt")
    if not full.span_exact[1] or span[0] != nb:
        raise ValueError("assistant content is not token-aligned with the generation prompt (a token straddles "
                         "the boundary); score a re-templated full conversation instead")
    ids = full.input_ids[span[0]:span[1]]
    if not end_of_turn:
        return list(ids)
    if span[1] >= len(full.input_ids):
        raise ValueError("chat template emits no end-of-turn token after assistant content")
    eot = int(full.input_ids[span[1]])
    stop = set(getattr(adapter, "eos_ids", set()) or set())
    if stop and eot not in stop:
        raise ValueError(f"token after the assistant content ({adapter.token_strings([eot])[0]!r}, id {eot}) is "
                         f"not a stop token {sorted(stop)}")
    return list(ids) + [eot]


def option_logprobs(adapter: ModelAdapter, prefix_ids: list[int], option_token_ids: list[int],
                    edits: list[Edit] | None = None) -> dict[str, Any]:
    """Next-token log-probs of single-token options at the end of ``prefix_ids``.

    Returns ``logprobs`` (full-vocabulary log-softmax), ``probs``, ``probs_renorm`` (renormalised over the options),
    ``mass`` (total probability of the options; a low mass means the model's next-token weight is mostly
    elsewhere and the renormalised readout is weak evidence), plus the overall top token for reference.
    """
    prefix = [int(t) for t in prefix_ids]
    opts = [int(t) for t in option_token_ids]
    if not prefix:
        raise ValueError("prefix_ids is empty")
    if not opts:
        raise ValueError("no options")
    if len(set(opts)) != len(opts):
        raise ValueError(f"duplicate option token ids {opts}")
    _vocab_check(adapter, prefix, "prefix")
    _vocab_check(adapter, opts, "options")
    with _lock(adapter):
        lg = adapter.sequence_logits([prefix], [[len(prefix) - 1]], list(edits or []))[0][0]
        stats = last_hook_stats(adapter)
    lp = torch.log_softmax(lg.double(), dim=-1)
    lps = [float(lp[t]) for t in opts]
    probs = [math.exp(x) for x in lps]
    mass = float(math.fsum(probs))
    renorm = [p / mass for p in probs] if mass > 0 else [float("nan")] * len(opts)
    top = int(torch.argmax(lg))
    return {
        "option_token_ids": opts,
        "option_token_strs": adapter.token_strings(opts),
        "logprobs": lps,
        "probs": probs,
        "probs_renorm": renorm,
        "mass": mass,
        "argmax_option": int(max(range(len(opts)), key=lambda i: lps[i])),
        "top_token_id": top,
        "top_token_str": adapter.token_strings([top])[0],
        "top_token_logprob": float(lp[top]),
        "top_is_option": top in opts,
        "n_edits_applied": (stats or {}).get("n_edits_applied"),
    }


def greedy(adapter: ModelAdapter, input_ids: list[int], max_new_tokens: int,
           edits: list[Edit] | None = None) -> GenerationResult:
    """Greedy (argmax, first-index tie-break) KV-cached decoding under ``edits`` — thin wrapper over ``generate``."""
    with _lock(adapter):
        return adapter.generate(list(input_ids), int(max_new_tokens), list(edits or []))


# ------------------------------------------------------------------------------------------------ patching
def differing_positions(ids_a: list[int], ids_b: list[int]) -> list[int]:
    """Positions where two equal-length token sequences differ. Unequal lengths → ValueError."""
    if len(ids_a) != len(ids_b):
        raise ValueError(f"sequences have different lengths ({len(ids_a)} vs {len(ids_b)}); they are not "
                         "position-aligned")
    return [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if int(a) != int(b)]


def _nan_to_none(x: float) -> float | None:
    return None if x is None or math.isnan(x) or math.isinf(x) else float(x)


def patch_restoration(adapter: ModelAdapter, clean_ids: list[int], corrupt_ids: list[int], positions: list[int],
                      depths: list[int], readout_pos: int, tok_clean: int, tok_corrupt: int, *,
                      joint: bool = True, batch_size: int = 16, resume: bool = True,
                      max_diff_fraction: float = MAX_DIFF_FRACTION) -> dict[str, Any]:
    """Position-ALIGNED residual patching clean → corrupt.

    For each depth d in ``depths`` and position p in ``positions``: run the CORRUPT sequence with
    h[d, p] ← h_clean[d, p] (a ``ResidSetEdit`` holding the clean run's residual at the same depth and position) and
    read LD = logit(tok_clean) − logit(tok_corrupt) at ``readout_pos``.

        restoration[i][j] = (LD_patched(depths[i], positions[j]) − LD_corrupt) / (LD_clean − LD_corrupt)

    1 = the single patched state restores the clean metric, 0 = no effect. Values are None when undefined
    (|LD_clean − LD_corrupt| ≤ 1e-6, e.g. identical inputs). With ``joint`` an extra row per depth patches ALL
    ``positions`` at once (``restoration_joint[i]``) — needed when the counterfactual span is several tokens.

    Refuses (ValueError) unequal lengths, positions after ``readout_pos`` (they cannot causally reach it), depths
    outside 0..n_layers, identical target tokens, and pairs that differ at ≥ ``max_diff_fraction`` of positions
    (misaligned rather than a controlled replacement). Each depth is one batched call; with ``resume`` the
    corrupt run is resumed from its cached residual at that depth (blocks before d are not re-run; exactness of
    resume is covered by tests). This is localized intervention evidence about one path, never complete provenance.
    """
    clean = [int(t) for t in clean_ids]
    corrupt = [int(t) for t in corrupt_ids]
    diff = differing_positions(clean, corrupt)
    T = len(clean)
    if T == 0:
        raise ValueError("empty sequences")
    if T > 1 and len(diff) >= max(2, math.ceil(max_diff_fraction * T)):
        raise ValueError(f"clean/corrupt differ at {len(diff)} of {T} positions — not a position-aligned "
                         f"counterfactual pair (limit {max_diff_fraction:.0%}); refusing to patch")
    r = int(readout_pos)
    if not 0 <= r < T:
        raise ValueError(f"readout_pos {r} outside sequence of length {T}")
    pos = [int(p) for p in positions]
    if not pos:
        raise ValueError("no positions to patch")
    if any(not 0 <= p <= r for p in pos):
        raise ValueError(f"positions {pos} must lie in [0, readout_pos={r}] (later positions cannot affect the readout)")
    if len(set(pos)) != len(pos):
        raise ValueError("duplicate positions")
    n_layers = adapter.info.n_layers
    deps = [int(d) for d in depths]
    if not deps or any(not 0 <= d <= n_layers for d in deps):
        raise ValueError(f"depths {deps} must be non-empty and within 0..{n_layers}")
    tc, tx = int(tok_clean), int(tok_corrupt)
    if tc == tx:
        raise ValueError("tok_clean == tok_corrupt: the logit difference is identically 0")
    _vocab_check(adapter, [tc, tx], "target tokens")
    _vocab_check(adapter, clean + corrupt, "inputs")

    def ld(row: torch.Tensor) -> float:
        return float(row[tc].double() - row[tx].double())

    xs, cs = corrupt[:r + 1], clean[:r + 1]  # causal: nothing after the readout matters
    t0 = time.time()
    n_rows = 0
    with _lock(adapter):
        Rc = adapter.capture(cs, "minimal").resid  # [D, r+1, d]
        Rx = adapter.capture(xs, "minimal").resid
        ld_clean = ld(adapter.final_logits_batch(cs, [[]], logit_positions=[r])[0, 0])
        ld_corrupt = ld(adapter.final_logits_batch(xs, [[]], logit_positions=[r])[0, 0])
        denom = ld_clean - ld_corrupt
        defined = abs(denom) > 1e-6
        raw = [[float("nan")] * len(pos) for _ in deps]
        raw_joint = [float("nan")] * len(deps)
        for i, d in enumerate(deps):
            sets = [[ResidSetEdit(d, p, Rc[d, p].clone(), 1.0)] for p in pos]
            if joint:
                sets.append([ResidSetEdit(d, p, Rc[d, p].clone(), 1.0) for p in pos])
            vals: list[float] = []
            for k in range(0, len(sets), max(1, int(batch_size))):
                chunk = sets[k:k + max(1, int(batch_size))]
                if resume and d > 0:
                    lg = adapter.final_logits_batch(xs, chunk, start_depth=d, resid_in=Rx[d], logit_positions=[r])
                else:
                    lg = adapter.final_logits_batch(xs, chunk, logit_positions=[r])
                vals += [ld(lg[b, 0]) for b in range(lg.shape[0])]
                n_rows += len(chunk)
            raw[i] = vals[:len(pos)]
            if joint:
                raw_joint[i] = vals[len(pos)]

    def rest(v: float) -> float | None:
        return _nan_to_none((v - ld_corrupt) / denom) if defined else None

    return {
        "metric": "logit(tok_clean) - logit(tok_corrupt) at readout_pos",
        "readout_pos": r, "positions": pos, "depths": deps,
        "tok_clean": tc, "tok_corrupt": tx,
        "tok_clean_str": adapter.token_strings([tc])[0], "tok_corrupt_str": adapter.token_strings([tx])[0],
        "diff_positions": diff,
        "ld_clean": ld_clean, "ld_corrupt": ld_corrupt, "denominator": denom, "defined": defined,
        "restoration": [[rest(v) for v in row] for row in raw],  # [len(depths)][len(positions)]
        "ld_patched": [[_nan_to_none(v) for v in row] for row in raw],
        "restoration_joint": [rest(v) for v in raw_joint] if joint else None,  # [len(depths)]
        "ld_patched_joint": [_nan_to_none(v) for v in raw_joint] if joint else None,
        "resumed_from_depth": bool(resume),
        "n_patched_rows": n_rows,
        "seconds": round(time.time() - t0, 3),
        "evidence_kind": "deterministic_derivation",
        "scope_note": "localized intervention evidence for one counterfactual pair; not complete provenance",
    }


__all__ = [
    "ContinuationScore", "assistant_continuation_ids", "continuation_logprobs", "differing_positions", "greedy",
    "last_hook_stats", "option_logprobs", "patch_restoration",
]
