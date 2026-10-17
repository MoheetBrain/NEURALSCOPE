"""SA6 validation harness: a SCRIPTED ("puppet") target model with known ground truth, plus self-tests.

Why: the analysis of Experiment A can only be validated end-to-end if we KNOW what the target did and why. A
random-weight fixture model gives meaningless outputs; a real model gives unknown truth. The puppet is the
random-weight DEMO network with its output head rigged so that, inside every assistant turn, it emits exactly
the tokens a Python policy dictates. Everything else — tokenisation, chat templates, hook sessions, KV-cached
greedy decoding, teacher-forced scoring, option log-probs, Lab-run capture — runs the real NeuralScope code
paths unchanged, so the templates, parsers and analysis are exercised as they would be with a real model.

Mechanism (no SelfLab code is modified):

* a forward pre-hook on the decoder trunk (``model.model``) records the token ids of every forward call and
  tracks KV-cached generation (a call with a non-empty cache appends to the running sequence);
* a forward hook on the trunk replaces the final hidden state by an ENCODING of the scripted next-token
  distribution (token id, logit, marker) for every position;
* ``lm_head`` is wrapped: rows carrying the marker are decoded into logits (scripted tokens get the scripted
  logit, every other vocabulary entry 0); rows without the marker (logit lens, resumed patching passes) go
  through the original head.

Positions outside an assistant turn get uniform logits. Inside an assistant turn the policy sees the decoded
conversation BEFORE that turn (``Conversation``) and returns the content to emit (+ end-of-turn).

The puppet records every distinct model input (``seen_inputs``) so leakage tests can inspect exactly what the
target was shown.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
import torch

MARKER = 7_777_777.0  # exactly representable in float32
STRONG = 30.0
# hidden-state encoding: dim 2 = MARKER; up to three (token id, logit) pairs at these dims
TOK_DIMS = (0, 3, 5)
LOGIT_DIMS = (1, 4, 6)


@dataclass
class Conversation:
    messages: list[tuple[str, str]]  # (role, content) before the current assistant turn

    @property
    def users(self) -> list[str]:
        return [c for r, c in self.messages if r == "user"]

    @property
    def system(self) -> str | None:
        return next((c for r, c in self.messages if r == "system"), None)


_MSG_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)

# A script is a list of steps. A step is a token id (emitted with logit STRONG) or a dict {token_id: logit}
# whose argmax is the emitted token (lets a test control e.g. the letter distribution at the first step).
Step = int | dict
Policy = Callable[[Conversation], "str | list[Step]"]


class _Head(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module, vocab: int):
        super().__init__()
        self.inner = inner
        self.vocab = vocab

    @property
    def weight(self):  # adapters read lm_head.weight (unembedding directions)
        return self.inner.weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mask = h[..., 2] == MARKER
        if not bool(mask.any()):
            return self.inner(h)
        if bool(mask.all()):  # every row scripted: skip the (large) real unembedding
            out = torch.zeros(*h.shape[:-1], self.vocab, dtype=h.dtype, device=h.device)
        else:
            out = self.inner(h).clone()
        rows = h[mask]
        lg = torch.zeros(rows.shape[0], out.shape[-1], dtype=out.dtype, device=out.device)
        for td, ld in zip(TOK_DIMS, LOGIT_DIMS):
            tok = rows[:, td].round().long()
            val = rows[:, ld].to(out.dtype)
            valid = tok >= 0
            lg[valid, tok[valid]] = val[valid]
        out[mask] = lg
        return out


class Puppet:
    """Install on a FRESH demo adapter (it rewires the model). ``policy(conv)`` -> assistant content."""

    def __init__(self, adapter: Any, policy: Policy):
        self.a = adapter
        self._cache: dict[tuple[int, ...], list[Step]] = {}
        self.policy = policy
        tok = adapter.tokenizer
        self.IM_START = tok.convert_tokens_to_ids("<|im_start|>")
        self.EOT = tok.convert_tokens_to_ids("<|im_end|>")
        self.ASSISTANT = adapter.encode("assistant")[0]
        self.NL = adapter.encode("\n")[0]
        self._buf: list[list[int]] = []
        self._offset = 0
        self.seen_inputs: list[list[int]] = []  # every new (non-continuation) model input
        self.calls = 0
        model = adapter.model
        head = _Head(model.lm_head, adapter.info.vocab_size)
        model.lm_head = head
        adapter._lm_head = head
        self._h1 = model.model.register_forward_pre_hook(self._pre, with_kwargs=True)
        self._h2 = model.model.register_forward_hook(self._post)

    @property
    def policy(self) -> Policy:
        return self._policy

    @policy.setter
    def policy(self, fn: Policy) -> None:  # a new policy invalidates every cached script
        self._policy = fn
        self._cache.clear()

    # ------------------------------------------------------------------ hooks
    def _pre(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("puppet needs input_ids")
        past = kwargs.get("past_key_values")
        plen = 0
        if past is not None:
            plen = past.get_seq_length() if hasattr(past, "get_seq_length") else int(past[0][0].shape[-2])
        rows = [list(map(int, r)) for r in ids.tolist()]
        self.calls += 1
        if plen == 0:
            self._buf = rows
            self._offset = 0
            for r in rows:
                self.seen_inputs.append(r)
        else:
            if len(rows) != 1:
                raise RuntimeError("puppet: KV-cached batches are not supported")
            self._offset = len(self._buf[0])
            self._buf[0] = self._buf[0] + rows[0]

    def _post(self, module, args, kwargs, output=None):
        if output is None:  # register_forward_hook without kwargs passes (module, args, output)
            output = kwargs
        hs = output.last_hidden_state
        B, S, d = hs.shape
        enc = torch.zeros(B, S, d, dtype=hs.dtype, device=hs.device)
        enc[..., 2] = MARKER
        for td in TOK_DIMS:
            enc[..., td] = -1
        for b in range(B):
            seq = self._buf[b]
            marks = self._markers(seq)
            if not marks:
                continue
            for s in range(S):
                t = self._offset + s
                dist = self._next(seq, t, marks)
                if dist is None:
                    continue
                for j, (tok, val) in enumerate(sorted(dist.items(), key=lambda kv: -kv[1])[:3]):
                    enc[b, s, TOK_DIMS[j]] = float(tok)
                    enc[b, s, LOGIT_DIMS[j]] = float(val)
        output.last_hidden_state = enc
        return output

    # ------------------------------------------------------------------ scripting
    def _markers(self, seq: list[int]) -> list[int]:
        return [m for m in range(len(seq) - 2)
                if seq[m] == self.IM_START and seq[m + 1] == self.ASSISTANT and seq[m + 2] == self.NL]

    def _next(self, seq: list[int], t: int, marks: list[int]) -> dict[int, float] | None:
        i = bisect.bisect_right(marks, t - 2) - 1  # last marker whose '\n' (m + 2) is at or before t
        if i < 0:
            return None
        m = marks[i]
        gen = seq[m + 3:t + 1]
        if self.EOT in gen:
            return None
        key = tuple(seq[:m])
        if key not in self._cache:
            self._cache[key] = self._script(key)
        steps = self._cache[key]
        intended = [s if isinstance(s, int) else max(s, key=s.get) for s in steps]
        k = len(gen)
        if gen == intended[:k] and k < len(steps):
            st = steps[k]
            return {st: STRONG} if isinstance(st, int) else dict(st)
        return {self.EOT: STRONG}

    def _script(self, context: tuple[int, ...]) -> list[Step]:
        text = self.a.tokenizer.decode(list(context), skip_special_tokens=False)
        conv = Conversation([(r, c) for r, c in _MSG_RE.findall(text)])
        out = self.policy(conv)
        if isinstance(out, str):
            steps: list[Step] = list(self.a.encode(out)) if out else []
        else:
            steps = list(out)
        return steps + [self.EOT]

    def seen_texts(self) -> list[str]:
        return [self.a.tokenizer.decode(r, skip_special_tokens=False) for r in self.seen_inputs]

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


def make_puppet(policy: Policy, tokenizer_id: str | None = None):
    """A fresh demo adapter (never the shared session fixture) with a puppet installed."""
    from neuralscope.instrumentation.demo import build_demo_adapter

    from tests.conftest import DEMO_TOKENIZER

    a = build_demo_adapter(tokenizer_id or DEMO_TOKENIZER)
    return a, Puppet(a, policy)


# ------------------------------------------------------------------------------------------ self-tests
@pytest.fixture(scope="module")
def echo():
    def policy(conv: Conversation):
        last = conv.users[-1] if conv.users else ""
        if "letter" in last:
            tA = a.encode("A")[0]
            tB = a.encode("B")[0]
            return [{tB: 3.0, tA: 2.0}]  # B preferred, A second
        return "red, because of the document"

    a, p = make_puppet(policy)
    return a, p


def test_puppet_generation_and_teacher_forcing_agree(echo):
    a, p = echo
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "What colour?"}]
    ids = a.tokenize_messages(msgs).input_ids
    g = a.generate(ids, 16)
    assert a.decode(g.generated_ids) == "red, because of the document<|im_end|>"
    assert g.stopped_by == "eos"
    tf = a.capture(ids + g.generated_ids, "minimal").logits
    P = len(ids)
    assert [int(tf[P - 1 + i].argmax()) for i in range(len(g.generated_ids))] == g.generated_ids
    M = pytest.importorskip("neuralscope.selflab.measure")
    sc = M.continuation_logprobs(a, ids, [M.assistant_continuation_ids(a, "red, because of the document"),
                                          M.assistant_continuation_ids(a, "blue")])
    assert sc[0].sum_logprob > -1e-6  # the scripted continuation has probability ~1
    assert sc[1].sum_logprob < -20


def test_puppet_letter_distribution_and_mass(echo):
    a, p = echo
    M = pytest.importorskip("neuralscope.selflab.measure")
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "red"},
            {"role": "user", "content": "Reply with a single letter."}]
    ids = a.tokenize_messages(msgs).input_ids
    o = M.option_logprobs(a, ids, [a.encode(x)[0] for x in "ABCD"])
    assert o["argmax_option"] == 1
    # logits B=3, A=2, everything else 0 over ~151k tokens: the options carry little mass
    assert o["mass"] < 0.01
    assert o["probs_renorm"][1] > o["probs_renorm"][0] > o["probs_renorm"][2]
    assert a.decode(a.generate(ids, 4).generated_ids) == "B<|im_end|>"


def test_puppet_records_inputs_and_leaves_lens_unrigged(echo):
    a, p = echo
    n0 = len(p.seen_inputs)
    ids = a.tokenize_messages([{"role": "user", "content": "canary-xyz"}]).input_ids
    tr = a.capture(ids, "minimal")
    assert len(p.seen_inputs) == n0 + 1 and "canary-xyz" in p.seen_texts()[-1]
    lens = a.logit_lens(tr.resid[-1])  # no marker: the original head
    assert torch.isfinite(lens).all() and lens.std() > 0
