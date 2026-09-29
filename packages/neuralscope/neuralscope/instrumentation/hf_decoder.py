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
    MLPScaleEdit,
    ModelAdapter,
    ModelInfo,
    ResidAddEdit,
    ResidSetEdit,
    TokenizedPrompt,
    Trace,
    TraceLevel,
)

SUPPORTED_MODEL_TYPES = {"qwen2", "llama", "mistral"}


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
        self.head: dict[int, list[tuple[int, HeadScaleEdit]]] = {}
        self.mlp: dict[int, list[tuple[int, MLPScaleEdit]]] = {}
        self.resid: dict[int, list[tuple[int, ResidSetEdit | ResidAddEdit]]] = {}
        for b, edits in enumerate(edit_sets):
            for e in edits:
                if isinstance(e, HeadScaleEdit):
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
        self.n_edits_applied = 0

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
                        if loc == slice(None) or loc:
                            x[b, loc, e.head * dh:(e.head + 1) * dh] *= e.factor
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

        try:
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
            stopped = "max_new_tokens"
            for _ in range(max_new_tokens):
                nxt = int(torch.argmax(logits))
                gen.append(nxt)
                if nxt in self.eos_ids:
                    stopped = "eos"
                    break
                if len(gen) >= max_new_tokens:
                    break
                s.offset = len(input_ids) + len(gen) - 1
                out = self.model(input_ids=self._ids([nxt]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
        return GenerationResult(prompt_ids=list(input_ids), generated_ids=gen, stopped_by=stopped)  # type: ignore[arg-type]

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

    @torch.no_grad()
    def final_logits_batch(self, input_ids: list[int], edit_sets: list[list[Edit]], start_depth: int = 0,
                           resid_in: torch.Tensor | None = None) -> torch.Tensor:
        """Logits [B, T, V] for B copies of ``input_ids``, copy b under ``edit_sets[b]``.

        With ``start_depth > 0`` the forward pass resumes from ``resid_in`` [T, d] — the residual stream
        entering block ``start_depth`` (depth index ``start_depth``) — skipping blocks that cannot be
        affected by edits at or after that depth. Residual edits at ``start_depth`` are applied to
        ``resid_in``. Mathematically identical to the full pass (verified in tests).
        """
        B = len(edit_sets)
        if start_depth == 0:
            with self.lock, self._session(edit_sets, capture=None) as s:
                s.offset = 0
                out = self.model(input_ids=self._ids(input_ids, batch=B), use_cache=False)
            return out.logits.detach().to("cpu", torch.float32)
        if resid_in is None:
            raise ValueError("start_depth > 0 requires resid_in")
        for es in edit_sets:
            for e in es:
                first = e.depth if isinstance(e, (ResidSetEdit, ResidAddEdit)) else e.layer
                if first < start_depth:
                    raise ValueError(f"edit {e} acts before start_depth={start_depth} (those blocks are not re-run)")
        T = resid_in.shape[0]
        with self.lock, self._session(edit_sets, capture=None) as s:
            s.offset = 0
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
            logits = self._lm_head(self._norm(h))
        return logits.detach().to("cpu", torch.float32)

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
