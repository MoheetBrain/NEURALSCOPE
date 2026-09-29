# NeuralScope Architecture (V1)

```
neuralscope/
├── apps/
│   ├── web/                      Next.js 16 · TypeScript · Tailwind v4
│   │   ├── app/(marketing)/      /  /research  /docs
│   │   ├── app/app/              /app — the research workspace (+ Math Lab)
│   │   ├── components/{ui,brand,marketing,docs,workspace}/
│   │   └── lib/                  typed API client, colormaps, epistemic definitions
│   └── api/neuralscope_api/      FastAPI app (main.py: create_app; server.py: ASGI entry)
├── packages/neuralscope/neuralscope/     the Python library (one distribution, five layers)
│   ├── instrumentation/          ModelAdapter ABC · HFDecoderAdapter (hooks) · demo fixture model
│   ├── interpretability/         Logit Lens · direct logit attribution · metrics
│   ├── experiments/              service (orchestration) · store (SQLite + safetensors) · jobs
│   ├── mathlab/                  problem generator · SymPy oracle · answer parser · alignment
│   └── schemas/                  Pydantic: intervention specs, API request/response models
├── scripts/                      setup.sh · dev.sh · api.sh · web.sh · export_hero_snapshot.py · experiments/
├── tests/                        pytest (fixture-model invariants, service, API, Math Lab, real-model integration)
├── data/                         neuralscope.db + activations/*.safetensors + experiments/*.json (gitignored)
└── docs/
```

The suggested `packages/{instrumentation,interpretability,experiments,mathlab,schemas}` became **subpackages of a
single Python distribution** (`neuralscope`): they share types and are always versioned together, so five
separate installable packages would add packaging overhead without isolation benefits. The TypeScript side mirrors
the API schemas in `apps/web/lib/api/`.

## Layers

```
 Browser (/app)  ──fetch /api/*──▶  Next.js rewrite  ──▶  FastAPI (apps/api)
                                                            │
                                     NeuralScopeService (experiments/service.py)
                                     ├── compiles intervention specs → Edits
                                     ├── summarises tensors → browser-sized views (+ epistemic status)
                                     ├── JobManager (sweeps, benchmarks; one worker thread)
                                     │
                     ┌───────────────┴───────────────┐
              ModelAdapter (instrumentation)     ExperimentStore
              HFDecoderAdapter: hooks on the     SQLite metadata  +  safetensors tensors
              HF model; generate / capture /     (one file per run)
              final_logits_batch / logit_lens
```

### Instrumentation — `ModelAdapter`

```python
class ModelAdapter:
    load(); tokenize(prompt, chat) ; encode(text) ; decode(ids) ; token_strings(ids) ; is_special(id)
    generate(input_ids, max_new_tokens, edits)            # greedy, KV-cached, edits applied at every step
    capture(input_ids, level, edits) -> Trace             # one teacher-forced forward pass, hooks record tensors
    final_logits_batch(input_ids, edit_sets, start_depth=0, resid_in=None)   # B edit sets in one batch; resume
    logit_lens(resid) ; unembed_direction(tok) ; readout_direction(vec) ; final_norm_scale(resid)
    head_output(layer, head, z) ; attn_out_from_z(layer, z)
```

`HFDecoderAdapter` supports `model_type ∈ {qwen2, llama, mistral}` (pre-norm RMSNorm decoders). Hook sites:

| site | hook | captures | edits applied |
|---|---|---|---|
| `model.embed_tokens` | forward | `resid[0]` | residual set/add at depth 0 |
| `model.layers[l]` | forward | `resid[l+1]` | residual set/add at depth l+1 |
| `layers[l].self_attn` | forward | `attn[l]` (eager probs) | — |
| `layers[l].self_attn.o_proj` | forward-pre | `head_z[l]` = concat_h z_h | head scale (slice of input) |
| `layers[l].mlp` | forward | `mlp_out[l]` | MLP scale |

Hooks live only for the duration of one *session* (context manager) and are always removed. A session carries
the absolute position offset of the current forward call, so position-specific edits fire correctly during
KV-cached generation (prefill + one token per step). Edits are **compiled** by the service from user-facing
Pydantic specs: `HeadAblation → HeadScaleEdit`, `MLPAblation → MLPScaleEdit`, `ResidualPatch → ResidSetEdit`
(source vector loaded from another run's stored residuals), `ResidualSteer → ResidAddEdit` (difference of two
stored residual vectors).

**Adding a model family**: implement the same hook sites (or a new adapter class). Gemma needs `(1 + w)` RMSNorm,
√d embedding scaling and final-logit soft-capping in `logit_lens`/DLA; that is why it is not in the supported set.

**Adding instrumentation** (planned: Q/K/V, MLP pre-activations and SwiGLU gates, gradients, SAE features,
transcoders, Jacobians, KV-cache): add a capture key to `Trace`, a hook site in the adapter, a trace-level gate,
a capability flag in `ModelInfo.capabilities` (the UI shows COMING SOON for `false`), and a view in the service.

### Trace levels (`TRACE_LEVEL`)

| level | stored tensors | 47-token run (1.5B) |
|---|---|---|
| `minimal` | tokens, residual stream at all 29 depths, final top-20 log-probs, next-token log-prob/rank, full output log-probs at generated positions | ≈ 11 MB |
| `standard` (default) | + attention probabilities (fp16), per-head outputs `z` | ≈ 20 MB |
| `full` | + MLP outputs (otherwise derivable as `resid[l+1] − resid[l] − attn_out[l]`) | ≈ 28 MB |

Logit-Lens summaries (top-10 + entropy per cell) are added eagerly for interactive runs and lazily for Math Lab
runs. Because runs are deterministic, any higher level can be **re-captured lazily** (`ensure_level`): the
service replays `(input_ids, interventions)` and records the drift (0.0 in tests).

### Storage

* **SQLite** `data/neuralscope.db`: `runs` (id, timestamps, model, prompt, templated text, generation settings,
  input ids, prompt length, user span, generated ids/text, free-running output for intervention runs, target,
  trace level, tensor path + shapes, parent run, interventions, metrics (comparisons, sweeps), Math Lab payload,
  correctness, notes, tags, benchmark id, trace-consistency flag), `experiments` (named bundles of run ids +
  notes + UI payload; also mirrored to `data/experiments/<id>.json`), `benchmarks`.
* **safetensors** `data/activations/<run_id>.safetensors`: all tensors for one run. Never in SQL, never sent
  wholesale to the browser. An in-memory LRU keeps recently used traces hot.

### Intervention runs

An intervention run is a child of a baseline run and stores two things:
1. **free-running output** under the intervention (does the answer change?);
2. a **teacher-forced trace on the baseline token sequence** under the intervention, so every position/depth
   aligns with the parent for diffs (Δ residual heatmap, per-position KL, target logit/prob/rank before → after).

### Sweeps

* **Head ablation sweep**: zero-ablate each head (one at a time, all positions), read out at the target position.
  Batched per layer (B = n_heads) and resumed from the cached residual entering that layer — exact, ≈ 80 s for
  336 heads on CPU.
* **Activation-patching sweep**: for equal-length counterfactual pairs, patch `h_clean[d, t]` into the corrupt run
  for every depth `d` and every position from the first differing token to the readout; restoration =
  (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt). Batched per depth and resumed from depth `d`.

### Concurrency & performance

One model per process; the adapter holds a re-entrant lock per forward pass. Long jobs run on a single worker
thread; interactive requests interleave between forward passes. The API never returns raw tensors except the
one selected residual vector (d_model floats) and attention patterns for one layer (H×T×T, 3 decimals).
Frontend heatmaps are drawn on `<canvas>`.

## Model selection (V1)

**Qwen/Qwen2.5-1.5B-Instruct** — see RESEARCH_LOG.md (2026-09-29). Candidates considered: Gemma-2-2B (gated;
architecture quirks: soft-capping, (1+w) norms, sliding-window layers), Llama-3.2-1B/3B (gated; multi-digit
number tokens obscure digit-level analysis), Qwen2.5-0.5B/1.5B/3B (ungated; digit-level tokenisation; clean
Llama-family block). 1.5B is the largest that keeps interactive latency on CPU acceptable (≈ 5–8 s per traced
run) while still solving most 2–5-digit additions, so Math Lab has both correct and incorrect traces.

## Scaling beyond V1

* GPU: `NEURALSCOPE_DEVICE=cuda`; everything is device-agnostic (tensors are moved to CPU only for storage).
* Larger models: bf16 weights + fp32 readouts; selective capture (layers of interest); attention for chosen layers.
* Multi-user: the store and job manager are process-local by design for V1 (no auth); move metadata to Postgres
  and tensors to object storage when needed.
