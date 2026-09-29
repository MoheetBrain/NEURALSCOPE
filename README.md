# NeuralScope

**Debugger for learned intelligence.**

NeuralScope is a universal causal debugger for artificial intelligence. It captures the internal computation of
neural networks at token, layer, head, feature and circuit resolution. It translates those states into
experimentally testable representations. It then lets you intervene on them, rerun the model, and measure
exactly what changed.

Deep learning turned software from *human writes algorithm → computer executes it* into *human specifies an
objective → training discovers an algorithm → humans do not fully understand the learned computation*.
NeuralScope's mission: **recover the algorithms that learning discovered.**

```
OBSERVE → INTERPRET → HYPOTHESISE → INTERVENE → MEASURE → CAUSALLY VALIDATE → (IMPROVE MODEL)
```

V1 is a working neural debugger plus a Math Lab, running a real open-weight model locally. Nothing in the
interface is a mock-up: tokenisation, generation, hidden-state capture, Logit Lens, attention, interventions,
reruns and comparisons all come from real forward passes. A DEMO mode exists for UI work. It uses a tiny
random-weight fixture model and is permanently bannered as such.

---

## 1. What works in V1

| capability | status |
|---|---|
| Open-weight model inference (Qwen2.5-1.5B-Instruct, local CPU/GPU, greedy, deterministic) | ✅ real |
| Tokenisation view (chat-template / user / generated tokens; per-token next-token log-prob and top-5) | ✅ real |
| Activation capture: residual stream at every depth, attention probabilities, per-head outputs, MLP outputs, final logits | ✅ real |
| TOKEN × LAYER explorer (29 depths × T tokens) with 8 cell metrics, click-to-inspect | ✅ real |
| Cell inspector: the actual 1,536-dim residual vector, stats, histogram, top dimensions, cosine geometry | ✅ real |
| Logit Lens (model's own final RMSNorm + unembedding): top-k per depth, target rank/probability through depth | ✅ real |
| Attention explorer: every head of a layer, T×T patterns, query-row routing, head summaries, vocab projection | ✅ real |
| Direct logit attribution (exact decomposition over embed / 336 heads / 28 MLPs) | ✅ real |
| Interventions: head ablation/scaling, MLP ablation, residual activation patching, difference steering → rerun | ✅ real |
| Baseline-vs-intervention diff: Δ logit / Δ prob / Δ rank, KL, top-k shift, changed tokens, where the effect starts | ✅ real |
| Exhaustive head-ablation sweep and activation-patching sweep (background jobs) | ✅ real |
| Experiment persistence (SQLite + safetensors), run history, saved experiments, notes | ✅ real |
| Math Lab: seeded benchmark, SymPy oracle, correct/incorrect/truncated labels, correct-vs-incorrect trajectory comparison, benchmark-wide intervention evaluation | ✅ real |
| Website: landing page (hero rendered from a REAL captured trace), research, docs | ✅ |
| Q/K/V, MLP pre-activations, gradients, SAE features, transcoders, KV-cache inspection | COMING SOON |

## 2. Architecture

```
apps/web  (Next.js 16, TypeScript, Tailwind)   /  /research  /docs  /app
   │  /api/* rewrite
apps/api  (FastAPI)  ──  NeuralScopeService  ──┬── ModelAdapter (native PyTorch hooks on HF transformers)
                                                ├── ExperimentStore (SQLite metadata + safetensors tensors)
                                                └── JobManager (sweeps, benchmarks)
packages/neuralscope/neuralscope/{instrumentation, interpretability, experiments, mathlab, schemas}
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Methods and epistemics:
[docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md). Experiment workflows: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
Findings: [RESEARCH_LOG.md](RESEARCH_LOG.md).

## 3. Setup

Requirements: Python ≥ 3.10 (3.11 tested), Node ≥ 20.9 (22 tested), ~8 GB free disk (model + deps).

```bash
./scripts/setup.sh        # venv + CPU torch + package, npm install, downloads Qwen2.5-1.5B/0.5B-Instruct
# CUDA: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./scripts/setup.sh
```

Configuration lives in `.env` (copied from [.env.example](.env.example)): model id, device, dtype, `TRACE_LEVEL`
(`minimal | standard | full`), prompt/generation limits, data directory, ports.

## 4. Hardware requirements

| setup | RAM | notes |
|---|---|---|
| CPU only (reference) | 16 GB | 1.5B in float32 ≈ 6.2 GB weights; server RSS ≈ 8–9 GB with trace cache. Traced run ≈ 5–8 s; intervention ≈ 5 s; 336-head sweep ≈ 80 s; 25-problem benchmark ≈ 100 s (4 cores) |
| NVIDIA GPU | ≥ 8 GB VRAM | `NEURALSCOPE_DEVICE=cuda`; everything is device-agnostic |
| Apple Silicon | 16 GB | `NEURALSCOPE_DEVICE=mps` (not benchmarked) |
| Low memory | 8 GB | `NEURALSCOPE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` |

## 5. Run it

```bash
make dev                  # = ./scripts/dev.sh : API on :8000 (model loads in background) + web on :3000
open http://localhost:3000            # website → "OPEN NEURAL DEBUGGER" → /app
make demo                 # DEMO mode: random-weight fixture model, no download, UI development only
make api / make web       # run either side alone
```

Acceptance flow: open `/app` → type `What is 37 + 58? Give only the answer.` → **RUN** → inspect tokens and
output → click TOKEN × LAYER cells → LOGIT LENS → ATTENTION (pick a head) → INTERVENTIONS (e.g. ablate L22.H10,
or run the head sweep) → **ABLATE + RERUN** → DIFF → **SAVE EXPERIMENT** → MATH LAB → run a benchmark → compare a
correct and an incorrect trace.

Tests:

```bash
make test-fast            # fixture-model invariants, service, API, Math Lab oracle
make test-slow            # end-to-end on real Qwen2.5-0.5B-Instruct weights (~80 s)
make test-web             # frontend unit tests (vitest)
make lint                 # eslint + TypeScript
```

API reference: `http://localhost:8000/docs` (OpenAPI) and the website's `/docs` page.

## 6. Chosen model

**Qwen/Qwen2.5-1.5B-Instruct** is Apache-2.0 and ungated. It has 28 layers, 12 query heads and 2 KV heads (GQA),
d_model 1536, tied embeddings and RMSNorm. We chose it for compatibility and clean instrumentation rather than
benchmark scores:
- It uses a standard Llama-family block.
- It fits in float32 on a CPU.
- Its tokenizer gives every digit its own token, which makes arithmetic analysable digit by digit.
- It solves most 2–5-digit additions but not all, so Math Lab gets both correct and incorrect traces.

The full decision record is in [RESEARCH_LOG.md](RESEARCH_LOG.md). Qwen2.5-0.5B-Instruct is the fast real-model
test target. Supported families: qwen2, llama, mistral.

## 7. How instrumentation works

Standard PyTorch module hooks are attached for the duration of one forward-pass session:
- the embedding and each decoder block give the residual stream (pre-final-norm);
- each attention module gives the attention probabilities (eager attention);
- a pre-hook on each attention output projection (`o_proj`) gives the per-head outputs. Its input is the
  concatenation of per-head `z_h`, and it has no bias, so scaling one slice removes exactly that head's
  residual write;
- each MLP gives its output.

User-facing intervention specs are compiled into edits that these same hooks apply. The edits run during
KV-cached greedy generation (position-aware) and during teacher-forced capture. Sweeps batch many edit sets into
one forward pass, and resume from the cached residual at the edited depth. The following are verified by tests:
- the per-layer residual decomposition, and direct logit attribution, sum exactly;
- the Logit Lens at the final depth equals the model's logits;
- an ablation leaves upstream depths bit-identical;
- batched and resumed passes equal full passes.

## 8. Epistemic statuses

| | |
|---|---|
| **OBSERVED** | direct tensor/model measurement: tokens, outputs, residual vectors, attention probabilities, oracle labels |
| **DERIVED** | mathematical transformation of observed quantities: Logit Lens, norms, cosines, entropy, DLA, cross-run differences |
| **INFERRED** | semantic interpretation/hypothesis: only user-authored notes and hypotheses; never auto-generated, never ground truth |
| **CAUSALLY TESTED** | measured effect of an actual intervention, always with its scope (e.g. "single prompt, n = 1"; "25 problems") |

The labels appear on every panel. NeuralScope never says what a model "is thinking". It reports what was
observed, how a view was derived, and what an intervention changed, and by how much.

## 9. First findings (Qwen2.5-1.5B-Instruct, reproducible)

- On `37 + 58 = 95`, an exhaustive single-head ablation of all 336 heads isolates L22.H10 and L22.H11. Each
  drops log P("9") by about 8 nats. Ablating L22.H10 changes the output to **`85`**: the sum without the units
  carry.
- Activation patching of `37+58` vs `37+48` shows the distinguishing information moving from the operand token
  to the answer position between L21 and L24. That is the same layers as the critical heads.
- Qwen2.5-0.5B answers `37 + 58` with `85` unaided.
- Hypothesis (INFERRED): L22.H10/H11 route the carry. Experiment E1 tests this across controlled problem sets.
  See [RESEARCH_LOG.md](RESEARCH_LOG.md).

## 10. Known scientific limitations

- The Logit Lens is not a tuned lens, so early-layer readouts are biased.
- Attention weights are not explanations.
- DLA measures direct effects only.
- Zero-ablation can push activations off-distribution; mean or resample ablation is planned.
- Single-head sweeps miss redundancy and interactions.
- Single-prompt causal results do not establish general mechanisms.
- Correct-vs-incorrect comparisons across different problems confound problem identity with correctness;
  controlled counterfactual pairs are the remedy.
- Patching sweeps need position-aligned pairs.
- One model family is supported, and there are no feature-level (SAE/transcoder) views yet.

## 11. Roadmap

- **V1: Neural debugger** (this release). Activations, token × layer explorer, Logit Lens, attention, causal
  ablation and patching, Math Lab.
- **V2: Feature debugger.** SAEs, transcoders, probes, automated activation patching, attribution patching,
  representation geometry, mean/resample ablation.
- **V3: Circuit compiler.** Attribution graphs, automated circuit discovery, causal validation, model diffing.
- **V4: Agent + training control.** Autonomous interpretability scientist, agent flight recorder, pre-action
  Neural Interlock, training-time feedback. Future Math Lab work: formal proof state ↕ neural state (not in V1).

## Repository layout

```
apps/web                 Next.js site + /app workspace
apps/api                 FastAPI application
packages/neuralscope     Python library: instrumentation, interpretability, experiments, mathlab, schemas
scripts/                 setup/dev/api/web launchers, hero-snapshot exporter, experiments/
tests/                   pytest suite
docs/                    ARCHITECTURE, INTERPRETABILITY, EXPERIMENTS, screenshots, research outputs
data/                    local runs, activations, experiments (gitignored)
RESEARCH_LOG.md          findings and decisions with evidence
```
