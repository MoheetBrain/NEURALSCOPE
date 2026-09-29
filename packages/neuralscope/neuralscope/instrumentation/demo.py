"""DEMO fixture model.

A tiny, *randomly initialised* Qwen2-architecture network (seeded) that runs
through exactly the same instrumentation code paths as a real model. It exists
so the UI and the test-suite can exercise capture → lens → attention →
intervention → diff without downloading or loading 1.5B parameters.

Its outputs are meaningless by construction. Every model_info / run produced by
it carries ``is_demo=True`` and the UI renders a permanent
"DEMO — RANDOM-WEIGHT FIXTURE MODEL" banner, so fixture traces cannot be
confused with real model traces.
"""

from __future__ import annotations

import torch

from neuralscope.config import DEMO_MODEL_ID
from neuralscope.instrumentation.hf_decoder import HFDecoderAdapter


def build_demo_adapter(tokenizer_id: str, device: str = "cpu", seed: int = 0,
                       n_layers: int = 4, d_model: int = 64, n_heads: int = 4, n_kv_heads: int = 2) -> HFDecoderAdapter:
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        eos_token_id=tok.eos_token_id,
    )
    cfg._attn_implementation = "eager"
    g = torch.random.fork_rng()
    with g:
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(cfg)
    model.config._attn_implementation = "eager"
    adapter = HFDecoderAdapter(DEMO_MODEL_ID, device=device, dtype="float32", model=model, tokenizer=tok, is_demo=True)
    adapter.load()
    return adapter
