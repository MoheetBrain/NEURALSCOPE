"""Native-PyTorch-hook adapter for Hugging Face Llama-family decoders.

Supported ``model_type``: qwen2, llama, mistral — pre-norm decoder blocks

    h_mid  = h + Attn(RMSNorm(h))          Attn = o_proj(concat_h z_h)
    h_next = h_mid + MLP(RMSNorm(h_mid))

with a final RMSNorm and an (optionally tied) linear unembedding.

Hook sites (all standard ``nn.Module`` hooks; nothing is monkey-patched):

    embed_tokens            forward hook      → resid[0]            (+ resid edits at depth 0)
    layers[l]               forward hook      → resid[l+1]          (+ resid edits at depth l+1)
    layers[l].self_attn     forward hook      → attn[l]             (eager attention returns probs)
    layers[l].self_attn.o_proj  pre-hook      → head_z[l]           (+ head scale edits)
    layers[l].mlp           forward hook      → mlp_out[l]          (+ MLP scale edits)

Why o_proj's *input*: it is exactly concat_h(z_h). o_proj is linear and bias-free
in these architectures, so the residual write of head h is W_O[:, h] z_h and
scaling z_h scales that write and nothing else. This is correct under grouped-
query attention too (K/V are shared across query heads but each query head has
its own z_h and its own W_O slice).

Eager attention is used everywhere (generation, capture, sweeps) so that every
run in an experiment goes through the same kernels and attention probabilities
are materialised.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

import torch

from neuralscope.instrumentation.adapter import (
    Edit,
    GenerationResult,
    HeadScaleEdit,
    HeadSetEdit,
    MLPScaleEdit,
    ModelAdapter,
    ModelInfo,
    ResidAddEdit,
    ResidSetEdit,
    TokenizedChat,
    TokenizedPrompt,
    Trace,
    TraceLevel,
)

SUPPORTED_MODEL_TYPES = {"qwen2", "llama", "mistral"}
CHAT_ROLES = {"system", "user", "assistant"}
# Private-use code points: cannot collide with real message text in practice; used to locate content by construction.
_SENTINEL = "\ue000NS_CONTENT_SENTINEL\ue001"


def _validate_messages(messages: list[dict]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"message {i} must be a dict with 'role' and 'content'")
        if m["role"] not in CHAT_ROLES:
            raise ValueError(f"message {i}: unsupported role {m['role']!r} (allowed: {sorted(CHAT_ROLES)})")
        if not isinstance(m["content"], str):
            raise ValueError(f"message {i}: content must be a string")
        out.append({"role": m["role"], "content": m["content"]})  # extra keys (tool_calls, …) are dropped
    return out


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16}[dtype]


class _Session:
    """State shared by all hooks during one forward-pass sequence (a capture, a generation, a batch)."""

    def __init__(self, edit_sets: list[list[Edit]], capture: TraceLevel | None, n_layers: int):
        self.offset = 0  # absolute position of the first token in the current forward call
        self.edit_sets = edit_sets
        self.capture = capture
        self.head: dict[int, list[tuple[int, HeadScaleEdit | HeadSetEdit]]] = {}
        self.mlp: dict[int, list[tuple[int, MLPScaleEdit]]] = {}
        self.resid: dict[int, list[tuple[int, ResidSetEdit | ResidAddEdit]]] = {}
        for b, edits in enumerate(edit_sets):
            for e in edits:
                if isinstance(e, (HeadScaleEdit, HeadSetEdit)):
                    self.head.setdefault(e.layer, []).append((b, e))
                elif isinstance(e, MLPScaleEdit):
                    self.mlp.setdefault(e.layer, []).append((b, e))
                elif isinstance(e, (ResidSetEdit, ResidAddEdit)):
                    self.resid.setdefault(e.depth, []).append((b, e))
                else:  # pragma: no cover
                    raise TypeError(f"unknown edit {e!r}")
        self.resid_cap: dict[int, torch.Tensor] = {}
        self.attn_cap: dict[int, torch.Tensor] = {}
        self.z_cap: dict[int, torch.Tensor] = {}
        self.mlp_cap: dict[int, torch.Tensor] = {}
        self.n_edits_applied = 0  # (edit, batch row, forward call) applications that fired
        self.n_forward_calls = 0

    def local(self, positions: list[int] | None, seq_len: int) -> list[int] | slice:
        if positions is None:
            return slice(None)
        return [p - self.offset for p in positions if self.offset <= p < self.offset + seq_len]


class HFDecoderAdapter(ModelAdapter):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "float32", num_threads: int = 0,
                 model=None, tokenizer=None, is_demo: bool = False):
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.num_threads = num_threads
        self.model = model
        self.tokenizer = tokenizer
        self.is_demo = is_demo
        self.lock = threading.RLock()
        self.load_seconds: float | None = None
        self.last_session_stats: dict | None = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        t0 = time.time()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.model is None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype=self.dtype, attn_implementation="eager")
            except TypeError:  # transformers < 4.56 used torch_dtype
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=self.dtype, attn_implementation="eager")
        self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        if cfg.model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"model_type={cfg.model_type!r} is not supported by HFDecoderAdapter "
                f"(supported: {sorted(SUPPORTED_MODEL_TYPES)}). Gemma needs (1+w) RMSNorm, embedding scaling "
                "and logit soft-capping handling — see docs/ARCHITECTURE.md.")
        attn_impl = getattr(cfg, "_attn_implementation", None)
        if attn_impl not in (None, "eager"):
            raise RuntimeError(f"expected eager attention, got {attn_impl!r}")

        self._layers = self.model.model.layers
        self._norm = self.model.model.norm
        self._embed = self.model.model.embed_tokens
        self._lm_head = self.model.lm_head
        self._eps = float(getattr(self._norm, "variance_epsilon", getattr(self._norm, "eps", 1e-6)))

        n_heads = cfg.num_attention_heads
        d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        eos = self.model.generation_config.eos_token_id if self.model.generation_config else None
        eos_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids = eos_ids
        self._special = set(self.tokenizer.all_special_ids)
        for tid, tok in getattr(self.tokenizer, "added_tokens_decoder", {}).items():
            if getattr(tok, "special", False):
                self._special.add(int(tid))

        self.info = ModelInfo(
            model_id=self.model_id,
            architecture=f"{cfg.model_type} ({type(self.model).__name__})",
            n_layers=cfg.num_hidden_layers,
            n_heads=n_heads,
            n_kv_heads=getattr(cfg, "num_key_value_heads", n_heads),
            d_model=cfg.hidden_size,
            d_head=d_head,
            d_mlp=cfg.intermediate_size,
            vocab_size=self._lm_head.weight.shape[0],
            dtype=str(self.dtype).replace("torch.", ""),
            device=str(self.device),
            n_params=sum(p.numel() for p in self.model.parameters()),
            tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
            norm_type="RMSNorm",
            is_demo=self.is_demo,
            capabilities={
                "residual_stream": True,
                "attention_patterns": True,
                "head_outputs": True,
                "mlp_outputs": True,
                "logit_lens": True,
                "direct_logit_attribution": True,
                "head_ablation": True,
                "mlp_ablation": True,
                "activation_patching": True,
                "steering": True,
                "qkv": False,
                "mlp_preactivations": False,
                "gradients": False,
                "sae_features": False,
                "transcoders": False,
                "kv_cache_inspection": False,
            },
        )
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------ tokenisation
    def tokenize(self, prompt: str, chat: bool = True, system_prompt: str | None = None) -> TokenizedPrompt:
        tok = self.tokenizer
        if chat and getattr(tok, "chat_template", None):
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        else:
            text = prompt
            enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        span = None
        start = text.rfind(prompt) if prompt else -1
        if start >= 0 and "offset_mapping" in enc:
            end = start + len(prompt)
            idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and b > a]
            if idx:
                span = (idx[0], idx[-1] + 1)
        return TokenizedPrompt(text=text, input_ids=ids, user_span=span)

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _reject_control_strings(self, msgs: list[dict[str, str]]) -> None:
        """Message content is data: a special-token string inside it (e.g. ``<|im_end|>``) would be tokenised as a
        real control token and could forge extra chat turns, so it is refused rather than silently re-tokenised."""
        specials = {str(t.content) for t in getattr(self.tokenizer, "added_tokens_decoder", {}).values()
                    if getattr(t, "special", False) and str(t.content)}
        specials |= {str(s) for s in (self.tokenizer.all_special_tokens or []) if s}
        for i, m in enumerate(msgs):
            hit = next((s for s in specials if s in m["content"]), None)
            if hit is not None:
                raise ValueError(f"message {i} contains the special-token string {hit!r}; message content is data and "
                                 "may not contain chat-control tokens")

    def tokenize_messages(self, messages: list[dict], add_generation_prompt: bool = True) -> TokenizedChat:
        """Render ``messages`` with the chat template, tokenise the exact string, and locate each content span.

        Location is by construction where possible: message i is re-rendered with its content replaced by a
        sentinel; the sentinel's offset is where the content starts (everything before it is identical) and the
        real text is verified to hold the content verbatim there. If the template transforms content (so the check
        fails) we fall back to an ordered verbatim search from the end of the previous message's match. Content
        that cannot be found verbatim gets span None (reported, never guessed).
        """
        tok = self.tokenizer
        if not getattr(tok, "chat_template", None):
            raise ValueError(f"tokenizer of {self.model_id} has no chat template; use tokenize(prompt, chat=False)")
        msgs = _validate_messages(messages)
        self._reject_control_strings(msgs)
        text = self._render(msgs, add_generation_prompt)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        char_spans: list[tuple[int, int] | None] = []
        cursor = 0
        for i, m in enumerate(msgs):
            c = m["content"]
            if not c:
                char_spans.append(None)
                continue
            start = -1
            try:
                probe = [dict(x) for x in msgs]
                probe[i]["content"] = _SENTINEL
                k = self._render(probe, add_generation_prompt).find(_SENTINEL)
                if k >= cursor and text[k:k + len(c)] == c:
                    start = k
            except Exception:  # template rejected the probe → fall back to search
                start = -1
            if start < 0:
                start = text.find(c, cursor)
            if start < 0:
                char_spans.append(None)
                continue
            char_spans.append((start, start + len(c)))
            cursor = start + len(c)
        tok_spans: list[tuple[int, int] | None] = []
        exact: list[bool | None] = []
        for cs in char_spans:
            if cs is None:
                tok_spans.append(None)
                exact.append(None)
                continue
            a, b = cs
            idx = [j for j, (x, y) in enumerate(offsets) if y > a and x < b and y > x]
            if not idx:
                tok_spans.append(None)
                exact.append(None)
                continue
            tok_spans.append((idx[0], idx[-1] + 1))
            exact.append(offsets[idx[0]][0] == a and offsets[idx[-1]][1] == b)
        return TokenizedChat(text=text, input_ids=ids, message_token_spans=tok_spans, message_char_spans=char_spans,
                             offsets=offsets, span_exact=exact, add_generation_prompt=add_generation_prompt)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.tokenizer.decode([i], skip_special_tokens=False) for i in ids]

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self._special

    # ------------------------------------------------------------------ hooks
    @contextmanager
    def _session(self, edit_sets: list[list[Edit]], capture: TraceLevel | None) -> Iterator[_Session]:
        cfg = self.info
        s = _Session(edit_sets, capture, cfg.n_layers)
        dh = cfg.d_head
        handles = []
        want_attn = capture in ("standard", "full")
        want_z = capture in ("standard", "full")
        want_mlp = capture == "full"

        def resid_hook(depth: int):
            def apply(h: torch.Tensor) -> torch.Tensor:
                h = self._apply_resid_edits(s, depth, h)
                if capture is not None:
                    s.resid_cap[depth] = h[0].detach().to("cpu", torch.float32)
                return h

            def hook(module, args, output):
                if isinstance(output, tuple):
                    return (apply(output[0]),) + tuple(output[1:])
                return apply(output)

            return hook

        def attn_hook(layer: int):
            def hook(module, args, output):
                w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if w is None:
                    raise RuntimeError("attention probabilities unavailable; the model must use eager attention")
                s.attn_cap[layer] = w[0].detach().to("cpu", torch.float32)
            return hook

        def oproj_prehook(layer: int):
            def hook(module, args):
                x = args[0]
                edits = s.head.get(layer)
                if edits:
                    x = x.clone()
                    S = x.shape[1]
                    for b, e in edits:
                        loc = s.local(e.positions, S)
                        if not (loc == slice(None) or loc):
                            continue
                        sl = slice(e.head * dh, (e.head + 1) * dh)
                        if isinstance(e, HeadSetEdit):
                            v = e.vector.to(x.device, x.dtype)
                            if v.dim() == 2:  # per absolute position
                                idx = (torch.arange(S, device=x.device) + s.offset) if loc == slice(None) else \
                                    torch.tensor(loc, device=x.device) + s.offset
                                v = v[idx]
                            x[b, loc, sl] = v
                        else:
                            x[b, loc, sl] *= e.factor
                        s.n_edits_applied += 1
                if want_z:
                    B, S, _ = x.shape
                    s.z_cap[layer] = x[0].detach().reshape(S, cfg.n_heads, dh).to("cpu", torch.float32)
                if edits:
                    return (x,) + tuple(args[1:])
                return None
            return hook

        def mlp_hook(layer: int):
            def hook(module, args, output):
                edits = s.mlp.get(layer)
                out = output
                if edits:
                    out = out.clone()
                    S = out.shape[1]
                    for b, e in edits:
                        loc = s.local(e.positions, S)
                        if loc == slice(None) or loc:
                            out[b, loc] *= e.factor
                            s.n_edits_applied += 1
                if want_mlp:
                    s.mlp_cap[layer] = out[0].detach().to("cpu", torch.float32)
                return out if edits else None
            return hook

        def count_forward(module, args, kwargs=None):
            s.n_forward_calls += 1

        try:
            handles.append(self.model.model.register_forward_pre_hook(count_forward))
            if capture is not None or 0 in s.resid:
                handles.append(self._embed.register_forward_hook(resid_hook(0)))
            for l, layer in enumerate(self._layers):
                if capture is not None or (l + 1) in s.resid:
                    handles.append(layer.register_forward_hook(resid_hook(l + 1)))
                if want_attn:
                    handles.append(layer.self_attn.register_forward_hook(attn_hook(l)))
                if want_z or l in s.head:
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(oproj_prehook(l)))
                if want_mlp or l in s.mlp:
                    handles.append(layer.mlp.register_forward_hook(mlp_hook(l)))
            yield s
        finally:
            for h in handles:
                h.remove()
            self.last_session_stats = {"n_edits_applied": s.n_edits_applied, "n_forward_calls": s.n_forward_calls,
                                       "n_rows": len(edit_sets), "n_edits": sum(len(es) for es in edit_sets)}

    @staticmethod
    def _apply_resid_edits(s: _Session, depth: int, h: torch.Tensor) -> torch.Tensor:
        edits = s.resid.get(depth)
        if not edits:
            return h
        h = h.clone()
        S = h.shape[1]
        for b, e in edits:
            if isinstance(e, ResidSetEdit):
                loc = s.local([e.position], S)
                if loc:
                    v = e.vector.to(h.device, h.dtype)
                    h[b, loc] = (1.0 - e.alpha) * h[b, loc] + e.alpha * v
                    s.n_edits_applied += 1
            else:
                loc = s.local(e.positions, S)
                if loc == slice(None) or loc:
                    h[b, loc] = h[b, loc] + e.vector.to(h.device, h.dtype)
                    s.n_edits_applied += 1
        return h

    # ------------------------------------------------------------------ execution
    def _ids(self, ids: list[int], batch: int = 1) -> torch.Tensor:
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        return t.expand(batch, -1).contiguous() if batch > 1 else t

    @torch.no_grad()
    def generate(self, input_ids: list[int], max_new_tokens: int, edits: list[Edit] | None = None) -> GenerationResult:
        """Greedy (argmax, first-index tie-break) KV-cached decoding under ``edits``.

        Edits with explicit positions fire only when that absolute position is
        processed (prefill or the matching decode step).
        """
        with self.lock, self._session([list(edits or [])], capture=None) as s:
            s.offset = 0
            out = self.model(input_ids=self._ids(input_ids), use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            gen: list[int] = []
            gen_lp: list[float] = []
            gen_margin: list[float] = []
            stopped = "max_new_tokens"
            for _ in range(max_new_tokens):
                nxt = int(torch.argmax(logits))
                gen.append(nxt)
                gen_lp.append(float(torch.log_softmax(logits.float(), dim=-1)[nxt]))
                top2 = torch.topk(logits.float(), 2).values
                gen_margin.append(float(top2[0] - top2[1]))
                if nxt in self.eos_ids:
                    stopped = "eos"
                    break
                if len(gen) >= max_new_tokens:
                    break
                s.offset = len(input_ids) + len(gen) - 1
                out = self.model(input_ids=self._ids([nxt]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
        return GenerationResult(prompt_ids=list(input_ids), generated_ids=gen, stopped_by=stopped,  # type: ignore[arg-type]
                                n_edits_applied=s.n_edits_applied, generated_logprobs=gen_lp,
                                generated_margins=gen_margin)

    @torch.no_grad()
    def capture(self, input_ids: list[int], level: TraceLevel = "standard", edits: list[Edit] | None = None) -> Trace:
        with self.lock, self._session([list(edits or [])], capture=level) as s:
            s.offset = 0
            out = self.model(input_ids=self._ids(input_ids), use_cache=False)
        n = self.info.n_layers
        resid = torch.stack([s.resid_cap[d] for d in range(n + 1)])
        trace = Trace(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            resid=resid,
            logits=out.logits[0].detach().to("cpu", torch.float32),
            level=level,
        )
        if level in ("standard", "full"):
            trace.attn = torch.stack([s.attn_cap[l] for l in range(n)])
            trace.head_z = torch.stack([s.z_cap[l] for l in range(n)])
        if level == "full":
            trace.mlp_out = torch.stack([s.mlp_cap[l] for l in range(n)])
        return trace

    def _check_logit_positions(self, positions: list[int], T: int) -> list[int]:
        pos = [int(p) for p in positions]
        if not pos or any(not 0 <= p < T for p in pos):
            raise ValueError(f"logit positions {pos} must be non-empty and inside a sequence of length {T}")
        return pos

    @torch.no_grad()
    def final_logits_batch(self, input_ids: list[int], edit_sets: list[list[Edit]], start_depth: int = 0,
                           resid_in: torch.Tensor | None = None,
                           logit_positions: list[int] | None = None) -> torch.Tensor:
        """Logits [B, T, V] for B copies of ``input_ids``, copy b under ``edit_sets[b]``.

        With ``start_depth > 0`` the forward pass resumes from ``resid_in`` [T, d] — the residual stream
        entering block ``start_depth`` (depth index ``start_depth``) — skipping blocks that cannot be
        affected by edits at or after that depth. Residual edits at ``start_depth`` are applied to
        ``resid_in``. Mathematically identical to the full pass (verified in tests).

        ``logit_positions``: unembed only these positions and return [B, len(logit_positions), V] (memory: a
        full [B, T, V] float32 tensor is ~0.6 MB per token for a 152k vocabulary).
        """
        B = len(edit_sets)
        if start_depth == 0:
            if logit_positions is None:
                with self.lock, self._session(edit_sets, capture=None) as s:
                    s.offset = 0
                    out = self.model(input_ids=self._ids(input_ids, batch=B), use_cache=False)
                return out.logits.detach().to("cpu", torch.float32)
            pos = self._check_logit_positions(logit_positions, len(input_ids))
            with self.lock, self._session(edit_sets, capture=None) as s:
                s.offset = 0
                hs = self.model.model(input_ids=self._ids(input_ids, batch=B), use_cache=False).last_hidden_state
                logits = self._lm_head(hs[:, pos])  # last_hidden_state is already final-normed
            return logits.detach().to("cpu", torch.float32)
        if resid_in is None:
            raise ValueError("start_depth > 0 requires resid_in")
        for es in edit_sets:
            for e in es:
                first = e.depth if isinstance(e, (ResidSetEdit, ResidAddEdit)) else e.layer
                if first < start_depth:
                    raise ValueError(f"edit {e} acts before start_depth={start_depth} (those blocks are not re-run)")
        T = resid_in.shape[0]
        pos = None if logit_positions is None else self._check_logit_positions(logit_positions, T)
        with self.lock, self._session(edit_sets, capture=None) as s:
            s.offset = 0
            s.n_forward_calls += 1  # resumed pass: the base model's forward (and its counting hook) is bypassed
            h = resid_in.to(self.device, self.dtype)[None].expand(B, -1, -1).contiguous()
            h = self._apply_resid_edits(s, start_depth, h)
            pos_ids = torch.arange(T, device=self.device)[None].expand(B, -1)
            mask = torch.full((T, T), torch.finfo(self.dtype).min, device=self.device, dtype=self.dtype).triu(1)
            mask = mask[None, None].expand(B, 1, T, T)
            pos_emb = self.model.model.rotary_emb(h, pos_ids)
            for l in range(start_depth, self.info.n_layers):
                out = self._layers[l](h, attention_mask=mask, position_ids=pos_ids, past_key_values=None,
                                      use_cache=False, position_embeddings=pos_emb)
                h = out[0] if isinstance(out, tuple) else out
            logits = self._lm_head(self._norm(h if pos is None else h[:, pos]))
        return logits.detach().to("cpu", torch.float32)

    @torch.no_grad()
    def sequence_logits(self, sequences: list[list[int]], positions: list[list[int]],
                        edits: list[Edit] | None = None) -> list[torch.Tensor]:
        """Teacher-forced logits of B different sequences in ONE right-padded batch, all under ``edits``.

        Right padding is exact for causal attention: a real token at position t only attends to positions <= t, so
        pad tokens (appended after each sequence) never influence the requested logits, and absolute positions —
        hence position-restricted edits — are identical to an unbatched run. Edits with ``positions=None`` also
        fire on pad positions, which is harmless for the same reason; per-position ``HeadSetEdit`` vectors must
        cover the longest sequence. Returns float32 CPU tensors [len(positions[b]), V].
        """
        if len(sequences) != len(positions):
            raise ValueError("sequences and positions must have the same length")
        if not sequences:
            return []
        lens = [len(x) for x in sequences]
        if min(lens) == 0:
            raise ValueError("empty sequence")
        pos = [self._check_logit_positions(p, n) for p, n in zip(positions, lens)]
        T = max(lens)
        pad = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else sequences[0][-1])
        batch = torch.tensor([list(x) + [pad] * (T - len(x)) for x in sequences], dtype=torch.long, device=self.device)
        with self.lock, self._session([list(edits or []) for _ in sequences], capture=None) as s:
            s.offset = 0
            hs = self.model.model(input_ids=batch, use_cache=False).last_hidden_state
            rows = torch.cat([hs[b, p] for b, p in enumerate(pos)])
            logits = self._lm_head(rows).detach().to("cpu", torch.float32)
        return list(torch.split(logits, [len(p) for p in pos]))

    # ------------------------------------------------------------------ readout geometry
    @torch.no_grad()
    def logit_lens(self, resid: torch.Tensor) -> torch.Tensor:
        x = resid.to(self.device, self.dtype)
        return self._lm_head(self._norm(x)).to("cpu", torch.float32)

    @torch.no_grad()
    def unembed_direction(self, token_id: int) -> torch.Tensor:
        return (self._norm.weight * self._lm_head.weight[token_id]).detach().to("cpu", torch.float32)

    @torch.no_grad()
    def readout_direction(self, vec: torch.Tensor) -> torch.Tensor:
        w = self._lm_head.weight
        return ((vec.to(w.device, w.dtype) * self._norm.weight) @ w.T).to("cpu", torch.float32)

    @torch.no_grad()
    def final_norm_scale(self, resid: torch.Tensor) -> torch.Tensor:
        x = resid.to(torch.float32)
        return torch.rsqrt(x.pow(2).mean(-1) + self._eps)

    @torch.no_grad()
    def head_output(self, layer: int, head: int, z: torch.Tensor) -> torch.Tensor:
        dh = self.info.d_head
        W = self._layers[layer].self_attn.o_proj.weight[:, head * dh:(head + 1) * dh]  # [d, dh]
        return (z.to(W.device, W.dtype) @ W.T).to("cpu", torch.float32)

    @torch.no_grad()
    def attn_out_from_z(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        W = self._layers[layer].self_attn.o_proj.weight  # [d, H*dh]
        flat = z.reshape(*z.shape[:-2], -1)
        return (flat.to(W.device, W.dtype) @ W.T).to("cpu", torch.float32)
