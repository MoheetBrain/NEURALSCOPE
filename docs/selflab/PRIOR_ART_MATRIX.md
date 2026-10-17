# SelfLab prior-art matrix

*2026-10-17, SA1. Short form. The long notes are in [`research_notes/`](research_notes/) and
[`reports/`](reports/), and the condensed table is in [`SELFLAB_AUDIT_AND_PLAN.md` §A](SELFLAB_AUDIT_AND_PLAN.md).*

**Provenance tags.** **V** means the primary page was fetched this session (WebFetch, 2026-10-17). Licence and
last-push data came from the GitHub API (`gh api repos/…`) the same day. **N** means the claim comes from the existing
notes (2026-09-30), where every claim is already labelled paper / code-verified / ran-it. Nothing below is guessed. A
gap is written as "not stated" or "not verified".

## Matrix

| Source | What it does | Overlap with our workflow | Must NOT be claimed as ours | Usable here? (Qwen2.5-1.5B/0.5B, licence) |
|---|---|---|---|---|
| **Goodfire Silico** (goodfire.com/silico) **V** | Commercial interpretability agent. It covers auto-interp, probes/SAEs, model diffing, causal analysis, data attribution and activation verbalizers. An agent drafts a plan, runs experiments in parallel and returns results. | High at the workflow level: an agent proposes and runs experiments about a model's behaviour. Its page does not mention checking a model's *self-reports*. | Agent-driven experiment planning and execution. "Interpretability agent". Running hypothesis tests on model internals. | Access is by request, demo or pricing. Terms of use were linked but not read. The page lists no models. **Not usable or verifiable here.** |
| **Goodfire capabilities docs** (docs.goodfire.com/concepts/capabilities/) **V** | A long capability list: activation harvesting, SAE training, probing, monitoring, steering, causal analysis, parameter decomposition, J-lens, data attribution, training pipelines, LLM-judge evals, paper replication and report generation. | Our evidence-record and report layer overlaps "results publication / reports". Our interventions overlap "causal analysis". | Any generic "causal analysis", steering, probing or report-generation capability. | The page states no supported-model list and no licence. Its examples include Qwen3-8B. **Not verified for Qwen2.5.** |
| **causalab** (github.com/goodfire-ai/causalab) **V** | Causal-abstraction hypothesis testing. An experiment is a single JSON document (model, data, interventions, measurements). Interchange interventions, DBM masks and DAS rotations. CPU/MPS/CUDA/NDIF. | High at the *method* level: typed experiment specs, reproducible JSON records, competing hypotheses compared on the same input pairs. Our `ExperimentSpec` + commit hash is the same idea. | Declarative or JSON experiment specs. Comparing hypotheses under interventions. Interchange interventions and DAS (Geiger et al. lineage). | The page names HF model keys, and its example is a tiny Llama. GitHub reports **no licence** (`spdx_id` empty; last push 2026-09-30). **Do not vendor code.** The ideas are citable. |
| **NNsight** (nnsight.net) **V + N** | Read and write any module's activations, gradients and batched interventions; remote execution on NDIF. | Our hook layer (`hf_decoder.py`) duplicates it. **N:** bit-exact on Qwen2.5-0.5B in five checks, at the same speed. | Activation access, intervention or resume-from-depth as a capability. | MIT (GitHub API, push 2026-10-16). Works on HF causal LMs (**N**: ran it on Qwen2.5-0.5B). **Usable.** A migration is planned, not done. |
| **interp-engine** (github.com/decoderesearch/interp-engine) **V + N** | Neuronpedia's inference engine. It captures 34 standardised hook points (eager backend: all 34; vLLM: 28), steers, and validates against TransformerLens and nnsight. | Our capture and logit-lens code. | Standardised hook points and capture. | Apache-2.0 (push 2026-10-16). The README lists Qwen3/Gemma/Llama/DeepSeek, not Qwen2.5-1.5B. **N:** the HF eager backend is architecture-generic. **Usable with checks.** |
| **Neuronpedia** (neuronpedia.org) **V + N** | Open platform with feature dashboards, steering, circuit tracer, NLAs, J-lens, Assistant Axis, search and an API. | Our viewer and steering UI. | Feature browsing, steering UI, circuit viewer, NLA or J-lens readouts. | Apache-2.0 (GitHub API). **V:** it lists Qwen2.5-7B-IT with a Qwen2.5-7B SAE. **N:** it has a community NLA for Qwen2.5-1.5B-IT at L18, but no SAEs and no J-lens fitted for Qwen2.5-1.5B. **Partially usable.** |
| **Signs of introspection** (anthropic.com/research/introspection) **V** | Concept injection on Claude 3–4.1. Opus 4.1 detects an injected concept "about 20% of the time" under the best protocol. Detection works only in a strength band. The authors describe it as unreliable and narrow. | Experiment C: hidden state manipulation, then a report. | Concept-injection detection paradigm. "Model reports an injected state". | Claude only. No weights. The paradigm is portable, and open-weight replications exist (**N**: Lederman & Mahowald, Hahami, Zou 2609.35108). |
| **A global workspace in language models** (anthropic.com/research/global-workspace, 2026-07-06) **V + N** | J-lens finds a "J-space". Swapping concepts there changes Claude's *reported* answer (soccer→rugby). Injected concepts get reported. Code is in `anthropics/jacobian-lens`. | The closest causal link between internal state and self-report. Relevant to Experiment C. | Showing that self-reports causally depend on a manipulated internal representation. J-lens. | Claude in the paper. **N:** pre-fitted lenses cover about 40 open models, including Qwen2.5-7B-IT and Qwen3, but **not Qwen2.5-1.5B**. We could fit one locally (untested). |
| **Activation Oracles** (alignment.anthropic.com/2025/activation-oracles/) **V + N** | A copy of the model is fine-tuned to answer questions about injected activations. It matches or beats prior methods on 3 of 4 auditing tasks. The authors warn the answers may reflect the oracle's own inferences. | An *external decoder* that could be one rung of our observer ladder. It is not self-access. | "Expose activations to a model and ask about them". | **V:** Qwen-3 8B, Gemma-2 9B, Llama-3.3 70B, Claude Haiku 3.5. **N:** checkpoints exist for Qwen3-1.7B/4B/8B, **none for Qwen2.5-0.5B/1.5B**. **Not usable without training.** |
| **Introspection Adapters** (alignment.anthropic.com/2026/introspection-adapters/) **V** | A LoRA, trained with SFT and DPO, makes fine-tuned models describe implanted behaviours. 59% on AuditBench. High false-positive rate on models without implants. | Trained self-description of *learned* behaviour, scored against known ground truth. | Training a model to report its own behaviours. Scoring self-reports against implanted ground truth. | Llama-3.3-70B and Qwen3 0.6B/4B/14B. GitHub reports no licence for the code repo. **Not applicable** to an untrained Qwen2.5-1.5B audit. |
| **AuditBench** (alignment.anthropic.com/2026/auditbench/) **V** | 56 models with 14 hidden implanted behaviours, trained not to confess. An investigator agent is tested with 13 tool configurations. Finds a "tool-to-agent gap". | Benchmarks for agentic auditing. Their warning that tools that look good in isolation often do not help agents applies to our UI. | An agent auditing a model with white-box tools. | Llama-3.3-70B and Qwen3-32B. No licence stated on the page or by the GitHub API. **Not applicable at our scale.** |
| **OWMI** (arXiv 2608.20569, Ferrara) **V + N** | Masked interventions on 8 open models (residual, head, SAE), with sham, impact-matched random and text-only observer controls. Self-report does not separate intervention from sham (AUROC ≈ 0.5007), while probes recover it (75–95.8%). | Very high for Experiment C: same controls (sham, impact-matched, observer). | Sham / impact-matched controls. A text-only observer. Equivalence testing of a null. | Code on HF (`emilioferrara/owmi`). **N:** edits hit only template positions 0–2 during prefill, so our C protocol fixes timing. Usable as a statistics reference. |
| **Monitorability evals** (alignment.openai.com/monitorability-evals/) **V** | Measures whether a monitor can predict a property of agent behaviour from its chain of thought. Intervention evals compare a control arm and an intervention arm, with total effect estimated from the rate difference. Metric is g-mean2, with cross-fit filtering. | **The closest method to Experiment A:** a counterfactual intervention defines ground truth for whether a cue influenced the output, and the stated reasoning is scored against it. | Using paired intervention arms to define cue influence and scoring explanations against it. Filtering noise-driven positives. | Apache-2.0 (`openai/monitorability-evals`, push 2026-08-17). Datasets are partly released. Applies to reasoning models' CoT, not to our constrained source report. **Reusable as a method reference.** |

## What follows for SelfLab

- **Not ours:**
  - activation access and intervention (NNsight, interp-engine);
  - feature, steering and lens UIs (Neuronpedia);
  - agentic experiment execution (Silico, AuditBench agents);
  - typed JSON experiment specs and hypothesis comparison under interventions (causalab);
  - injection-and-report paradigms (Anthropic introspection, J-space; OWMI);
  - trained self-reporters (Activation Oracles, Introspection Adapters);
  - intervention-arm ground truth for whether a cue influenced output (OpenAI monitorability; also the
    hint/cue-faithfulness literature in the notes).
- **Candidate contribution (narrow, not established):** a reproducible, local, white-box workflow. It takes one
  specific claim a model makes about its own behaviour or processing (for example "my answer was based on the
  document") and does four things:
  1. turns the claim into a committed, typed experiment with counterfactual branches and position-aligned controls;
  2. records independent evidence, with delivery receipts kept separate from causal use;
  3. compares the self-report with prompt-only and transcript observers, so a report an outsider could produce is not
     credited as privileged;
  4. returns a scoped, calibrated conclusion (`supported` / `contradicted` / `inconclusive` / `control_failed`) that
     links to the tensors in the Lab.

  We have **not** shown this combination is useful or novel. Silico and causalab overlap with most of it at the
  workflow level, and monitorability evals overlap with its ground-truth logic. We could not inspect Silico.
  Re-search before any external claim.
- **Practical:**
  - Qwen2.5-1.5B has no Activation Oracle, no SAEs and no fitted J-lens (**N**). It has a community NLA at L18 (**N**).
  - Every decoder rung for this model would need training or fitting first.
  - Use MIT/Apache tools (NNsight, interp-engine, Neuronpedia, monitorability-evals) only as references or
    dependencies. Do not copy code from repositories without a licence (causalab, introspection-adapters,
    auditing-agents).

**Not verified this session:**
- Silico's terms of use;
- causalab's underlying library;
- whether the Neuronpedia J-lens demo covers Qwen2.5-1.5B (only the notes say it does not).
