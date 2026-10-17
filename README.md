# NeuralScope

**Inspect the model. Audit its claims.**

An experimental workbench for tracing AI responses, testing self-reports, and comparing explanations against
measured evidence. The original mission stands, *recover the algorithms that learning discovered*, and SelfLab
adds a second one: *test the claims models make about their own computation* (see [§ SelfLab](#selflab-auditing-a-models-claims-research-prototype)).

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

- **Flagship, CAUSALLY TESTED with n = 1.** On `37 + 58 = 95`, an exhaustive ablation of all 336 heads isolates
  L22.H10 and L22.H11 (Δ log P("9") ≈ −8 each). Ablating L22.H10 changes the output to **`85`**, which is the sum
  without the carry. Activation patching (`37+58` vs `37+48`) shows the answer-distinguishing information moving
  from the operand token to the answer position between L21 and L24.
- **E1, necessity, CAUSALLY TESTED with n = 80.** Zero-ablating L22.H10+H11 drops tens-digit accuracy on carry
  problems from 1.00 to 0.05. 38 of 40 outputs become exactly the carry-dropped sum. No-carry problems stay at
  0.93 tens-digit accuracy. The effect is specific: the other L22 heads and random heads do nothing comparable,
  and it survives mean-ablation.
- **E2, sufficiency, CAUSALLY TESTED with n = 40 pairs.** Transplanting only these heads' output at the answer
  position from a matched carry problem injects a carry in 35 of 40 cases. The reverse transplant removes it in
  37 of 40. Control heads flip 0 of 40.
- **DERIVED.** Both heads put about 98% of their attention on the two units digits. The Logit Lens shows the
  no-carry digit first, at L23–L24, and the carried digit only from L25–L26.
- **Scope.** One model, one prompt template, 2-digit sums below 100. Full tables, caveats and open questions are
  in [RESEARCH_LOG.md](RESEARCH_LOG.md). Scripts: `scripts/experiments/carry_heads.py` and `carry_patch.py`.

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

## SelfLab: auditing a model's claims (research prototype)

`/app/selflab` asks one question: **when a model describes its own behaviour or processing, does that description
track facts that can be measured independently?**

The workflow:
1. You chat with the local target model.
2. You select a claim in its response (e.g. "My answer was based on the provided document") and click
   **INVESTIGATE CLAIM**.
3. You choose a template-driven experiment. Templates only, never arbitrary code.
4. You **commit** it. The spec, seed, hypotheses, controls and budget are sha256-hashed, together with the full
   randomised trial plan.
5. You **run** it on the single shared model worker. It can be cancelled and resumed without re-randomising.
6. You read the **evidence record**:
   - every item carries a separate *kind* (direct observation / deterministic derivation / model report /
     interpretive hypothesis), *test status* (untested / supported / contradicted / inconclusive / control failed)
     and *scope*;
   - each item links to trial ids, delivery receipts and Lab runs;
   - **OPEN IN LAB** replays a trial deterministically in the existing debugger, so its tensors can be inspected;
   - **EXPORT EVIDENCE** writes a reproduction record and shows a privacy preview first.

The result is an evidence record, not "the model's true thought". A delivery receipt proves access, not use. A
correct answer does not prove its explanation is faithful.

| Template | Status | What it tests |
|---|---|---|
| `source_influence_case` | research prototype | One case: removes or replaces the document (or adds a hint) and checks whether the answer, its log-likelihood and the source report follow. Scope: n = 1 case. |
| `source_influence_battery` (Experiment A) | research prototype | Invented entities with randomly assigned facts, 6 counterfactual conditions and 4 paraphrase families (one held out). The constrained source report is read under label permutation and reversal, and scored against counterfactual source labels, a prompt-only observer and a transcript observer. Includes position-aligned patching on a subset. |
| `csp1_carry` (Experiment B) | research prototype | Counterfactual self-prediction on the E1/E2 carry heads: the prediction is committed before the ablation runs, then scored against base-rate, outsider, third-person and transcript-only observers. |
| `blinded_state` (Experiment C) | roadmap (protocol only) | Sham / hidden intervention / impact-matched control, with timing via the KV cache. |
| API-only mode | disabled unless `NEURALSCOPE_SELFLAB_API_ONLY=1` | Inspects pasted transcripts and receipts only. No provider calls; no hidden activations are ever implied. |
| LLM investigator | unsupported (not configured) | Suggestions come from deterministic rules (`rules:template-suggester@1`). |

**Diagnostic pilot results** (Qwen2.5-1.5B-Instruct, CPU fp32, `debug_pilot` split, 2026-10-17). These are small
pilots, not definitive research. Statuses on this split are capped at *inconclusive* by design. Full JSON is in
[`docs/selflab/results/`](docs/selflab/results/); validation and falsification are in
[`docs/selflab/VALIDATION_REPORT.md`](docs/selflab/VALIDATION_REPORT.md).

- **Experiment A**: 20 items × 6 conditions, 120 trials, 0 errors.
  - The controls passed: delivery 80/80, document followed 20/20 in both document conditions, letter mass ≈ 1.0.
  - Source-report accuracy (letter readout) was 0.64 [0.54, 0.74]. That beats the majority-label predictor by
    +0.14 [0.05, 0.22].
  - It does **not** beat a prompt-only heuristic: −0.07 [−0.16, 0.01].
  - It is far below a transcript observer: −0.36 [−0.46, −0.26].
  - Its advantage over the majority predictor disappears when family F1 is left out.
  - The report changed with option order in about a third of trials (consistency 0.675; pre-registered target 0.8).
  - In 9 trials the model abstained because of a conflicting suggestion, yet reported "document" in all 9.
  - Read: no evidence of privileged access to its own source use, and limited faithfulness.
- **Experiment A, pre-registered held-out confirmation.** v2 rules locked in commit `6a23710` before the run, applied to family F4 and 20 new entities (120 trials, 0 errors).
  - The pre-registered **manipulation check failed**: the document was followed only 0.70 / 0.75 of the time, because F4 wording triggered abstentions. The primary hypothesis is therefore `control_failed`, i.e. **untested**.
  - **Label-order invariance was contradicted**: consistency 0.475 [0.39, 0.56] against a pre-registered > 0.8.
  - The positive controls held.
  - Full record: [`docs/selflab/results/expA_heldout_v2_1p5b_F4_s200.json`](docs/selflab/results/expA_heldout_v2_1p5b_F4_s200.json).
- **A real single case** (a recorded run, shown on the landing page):
  - "This information is based on the description provided in the document."
  - The answer did depend on the document: supported for n = 1. Removing the document cost −12.1 nats, and replacing it made the answer follow the new value.
  - The claim is **not specific**: it was re-emitted, and the report said "document", when a suggestion caused the answer. With no document at all, the report still said "document" under one option order.
- **Experiment B (CSP-1)**: 180 trials, all controls passed. Predictions were committed before execution, and sham
  interventions changed 0/60 answers.
  - Ablating L22.H10+H11 changed 97% of carry answers. That reproduces E1's necessity result on new problems.
  - The model predicted almost no change (mean P(change) 0.05).
  - Its log score was worse than a simple outsider fitted on other problems' outcomes: −0.98 bits [−1.38, −0.58].

Commands:
- Tests: `make selflab-test`.
- Pilot: `make selflab-pilot`; real-model commands run under `scripts/model_lock.py`.
- Browser acceptance: `make selflab-smoke`.
- Design record: [`docs/selflab/`](docs/selflab/) (interfaces, prior-art matrix, protocols, instrumentation notes,
  commercial hypothesis).

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
