# NeuralScope → SelfLab: audit, gap analysis and plan

*Phase 0–10 deliverable, 2026-09-30. Design only: no implementation code has been written or deleted.*

**Mission.** Measure what internal states an artificial system possesses, what those states causally do,
what privileged access the system itself has to them, and give the system tools to investigate itself.
Nothing here assumes consciousness, experience or sentience. "Self-experience" is translated throughout into
privileged self-access, internal-state discrimination, causal self-knowledge, calibrated self-description and
metacognitive access.

**Inputs to this document.**
- Prior-art report: [`reports/Machine self knowledge prior art.md`](reports/Machine%20self%20knowledge%20prior%20art.md)
- Research notes (≈41k words; every claim labelled paper / code-verified / ran-it / inference):
  - [frameworks](research_notes/Machine%20self%20knowledge%20prior%20art/frameworks.md): NNsight, NDIF, TransformerLens, interp-engine, nnterp, pyvene, SAELens, circuit-tracer. Hands-on and bit-exact checks on Qwen2.5-0.5B.
  - [neuronpedia_and_decoders](research_notes/Machine%20self%20knowledge%20prior%20art/neuronpedia_and_decoders.md): Neuronpedia, Jacobian Lens, NLAs, Assistant Axis, Activation Oracles and their calibration, LatentQA, SelfIE, Patchscopes, self-explanation.
  - [introspection_literature](research_notes/Machine%20self%20knowledge%20prior%20art/introspection_literature.md): OWMI, Introspection Adapters, Reality Check, Lindsey and replications, Looking Inward, self-modelling, neurofeedback. It includes a 24-item confound catalogue and a 14-item requirements checklist.
  - [novelty_and_methods](research_notes/Machine%20self%20knowledge%20prior%20art/novelty_and_methods.md): novelty verdicts, interpretability agents, statistics (V-information, CRT/HRT, smECE, meta-d′), construct validity, "Plurience".
- The repository audit: every file on `claude/zen-knuth-hhtjxh` at `4b94d45`, with line counts and dependency tracing.

---

## 0. Verdict

**What the field already has (so NeuralScope must not rebuild it):**
- **Activation access and intervention.** NNsight 0.7.0 reproduces every primitive in NeuralScope's hook layer bit-for-bit, at the same speed, including resume-from-depth via `.skip()`. interp-engine (Neuronpedia's backend), TransformerLens 4.0 and nnterp cover the rest. NDIF serves 8B–405B models for free.
- **Reading activations as text.** Activation Oracles, LatentQA, NLAs, SelfIE/Patchscopes and the Jacobian Lens all do this. In every case the reader is trained or external. Text-only baselines match them surprisingly often. Their stated confidence is uninformative: AUROC 0.40–0.53 on the secret-word task, arXiv 2605.26045.
- **Detecting that one's activations were perturbed.** Lindsey 2601.01828 and its replications did this with concept injection. OWMI 2608.20569 did it with content-free perturbations: chance on 8 open models, pooled d′ = 0.0039. The Reality Check (2605.26242) shows the positives are largely explained by anomaly detection or yes-bias.
- **Training a model to describe or predict itself.** Introspection Adapters 2604.16812 do this for implanted behaviours. Li et al. 2511.08579 trained explainers to predict activation-patching outcomes at four layer blocks: self 64.0 vs other 54.1 exact match, untrained 5.02. Most of the self-advantage survives removing the activation (59.9), which points to self-simulation.
- **Behavioural self-modelling.** Binder 2410.13787, Zeng 2608.30980, Strangers 2609.09899 and CHIVE 2608.16747 measure it. Self-advantages are small or null, and the cross-model mean beats self-report in Strangers (0.53 vs 0.04).

**What is missing:**
- No one has scored a model's own zero-shot predictions of what interventions on its internals will do: predictions committed before the intervention runs, then executed at head, representation or position granularity. No one has compared such predictions against a ladder of equally informed observers. No one has measured their calibration.
- No one has let a model run bounded experiments on itself while an identically equipped outsider runs the same experiments as a control.
- The pieces exist. The measurement layer that turns self-report into scored, falsifiable self-knowledge does not. That layer is SelfLab.

**The first experiment is CSP-1, counterfactual self-prediction with an observer ladder (§D).** The model predicts, before execution, whether and how its answer changes when a specified internal component or representation is suppressed. We then run the interventions. Its predictions are scored against the outcomes and against five observers. The expected result for 1.5–1.7B models is null or near-null. The design treats a null as a finding.

**What survives of NeuralScope (details in §B):**

| | Lines |
|---|---|
| Delete (generic viewer, API, trace store, Math Lab UI) | ≈ 11.2k |
| Delete if the website is also dropped | ≈ 15.8k |
| Replaced by a thin NNsight adapter after a parity test | ≈ 0.75k |
| Kept (edit vocabulary, DLA, test fixture, Math Lab oracle, epistemic discipline) | — |
| Preserved as a research asset (carry-head experiments E1/E2) | — |

The only NeuralScope component with no external equivalent is the declarative, validated intervention language. It becomes the core of SelfLab.

**Decisions I need from you** (none block writing the plan):
1. Approve the deletion map, option A or B (§B.3).
2. Free NDIF key for replication at 8B–70B (login.ndif.us) — yes or no?
3. Any paid frontier-model observer (API costs) — yes or no? The plan does not need one.
4. Merge this branch to `main` before or after M1?

---

## A. Prior-art matrix

Condensed from the prior-art report; exact designs, numbers and citations are in the notes.

| Tool / paper | Existing capability | What it makes redundant (in NeuralScope or in SelfLab plans) | Missing capability relevant to our mission |
|---|---|---|---|
| **NNsight 0.7.0 / 0.8.0rc1** (MIT) | Read and write of any module's input or output, batching, sessions, generation-time edits, gradients inside traces, `.skip()` resume, vLLM, NDIF remote. Five checks bit-exact on Qwen2.5-0.5B, same speed as our hooks (ran it). | NeuralScope's whole hook layer, `hf_decoder.py`: capture, edits, batched sessions, resume-from-depth. | No declarative or validated intervention spec: traces need real source files. Silent attention-mask bug for tensor inputs in multi-invoke (0.7.0). `generate()` inherits `repetition_penalty`. No prediction/outcome harness. |
| **NDIF** | Free research API. 21 hot models on 2026-09-30, including Llama-3.1 8B/70B/405B and Qwen2.5 7B–72B. | Any plan to self-host models from 8B to 405B. | Small Qwen models cold. Key required. Sandbox not hardened for model-authored code. |
| **interp-engine 1.12.0** (Neuronpedia, Apache-2.0) | 34 standardised hook points, an exact HF backend, vLLM backends, steering, logit lens, Jacobian-lens transport, gradients. | Our Logit Lens code (identical output) and capture layer. | Write API is steering-only. No resume-from-depth. No self-report harness. |
| **TransformerLens 4.0** (TransformerBridge) | Bit-exact bridge over HF, 1,383 hooks, patching utilities, attribution patching, DLA, JacobianLens fit/readout/steer/swap, 145 architectures. | Capture hooks, DLA, J-lens fitting. | `start_at_layer` saves no compute. Patching about 6× slower than batched sweeps. Breaks SAELens and circuit-tracer. |
| **nnterp 1.3** | Standardised names over nnsight across 16 families; logit/Patchscope lens. | Model-family plumbing, lens code. | No per-head z, no sweeps, no storage. |
| **pyvene 0.1.8** | Declarative interventions; trainable DAS/subspace interventions. | In principle, declarative edits. | Head-level crashes on transformers 5.x; stale releases. |
| **SAELens 6.51 / circuit-tracer** | SAEs for 38 models, including Qwen3 and Qwen-Scope; transcoders and attribution graphs. | Any in-house SAE or transcoder work. | No SAEs or transcoders for Qwen2.5-0.5B/1.5B. circuit-tracer pins transformers ≤ 4.57.3. |
| **Neuronpedia** (Apache-2.0, self-hostable) | Feature dashboards, activation/search/steer/graph/lens/NLA APIs (rate-limited per IP, e.g. 120/h for NLA), SAEs for Qwen3-1.7B+, circuit tracing for Qwen3-1.7B/4B. | Any generic feature browser, steering UI or circuit viewer, including our workspace UI. | No self-report or calibration layer. Auto-interp labels are hypotheses (INFERRED), not facts. |
| **Jacobian Lens** (Anthropic 2026-07-06; `anthropics/jacobian-lens`) | A corpus-averaged Jacobian transports mid-layer states to the vocabulary. Concept swaps change the model's own report of what it is thinking about (target in top-5 on 88% of trials, Claude only). | Our Logit Lens for early layers. | Not fitted for Qwen2.5-1.5B (we can fit locally). Self-report evidence only on Claude; no equal-information observer. |
| **Natural Language Autoencoders** (2026-05-07) | Trained activation→text→activation loop with 0.6–0.8 FVE. A community NLA exists for Qwen2.5-1.5B-Instruct layer 18. | An activation-to-text decoder we might have built. | Audits find about 2% of specific claims grounded. Reconstruction does not validate claims; no calibration. |
| **Assistant Axis** (2601.10387) | The leading persona direction. Capping it cuts persona jailbreaks by about 60%. Self-reflection conversations cause drift. | Persona-direction discovery. | It is a **confound for SelfLab**: introspective prompts move the model off its default persona. |
| **Activation Oracles / LatentQA** (+ "Building Better AOs" 2606.02609) | Trained LoRA readers answer questions about frozen activations and generalise out of distribution (e.g. secrets). Checkpoints exist for Qwen3-1.7B/4B/8B; none for Qwen2.5-0.5B/1.5B. | "Expose activations to the model." | The reader is a separate trained copy: evidence of decodability, not self-access. Cross-model readers are as good. Text-only readers perform similarly. They never significantly beat probes. |
| **AO calibration** (2605.26045) | Stated confidence useless (AUROC 0.40–0.53). Candidate scoring gives AUROC 0.92–0.96; agreement over 20 samples gives ECE 0.04–0.12. | Naive verbal-confidence readouts. | Secret-word task only; nothing on causal claims. |
| **Verbalizers** (Li, Wallace et al. 2509.13316) | A text-only zero-shot baseline matches LatentQA on feature extraction; readers leak their own knowledge. | Claims that decoders see "hidden" information. | Target-only-knowledge controls (PersonaQA) must be standard. |
| **Training LMs to Explain Their Own Computations** (Li, Guo, Huang, Steinhardt, Andreas, 2511.08579) | Trained explainers predict patching outcomes (four layer blocks, one token, CounterFact) and input-ablation effects. Self 64.0 vs other 54.1 EM; untrained 5.02; 59.9 without the activation. The self-advantage tracks activation similarity. | "Train a model to predict its patching outcomes." | Zero-shot. Heads/MLPs/features/positions. Graded effects. Compensation. Calibration. Observers matched on information and compute. |
| **CHIVE** (Karvonen, Ong, Kantamneni, Marks, 2608.16747) | Counterfactual prediction of prompt edits. AOs/NLAs/SAEs give no uplift over transcript reading. Self-trained ≈ cross-trained. | Behavioural counterfactual self-prediction (null for privilege). | Internal interventions excluded by design. |
| **Zeng, Assis, Wang** (2608.30980) | Ten behavioural self-modelling tasks, incl. COMPONENT-ATTRIBUTION (which *prompt* part matters) and LOGIT-ESTIMATION. RL improves skill. No consistent own-model advantage. Skill = score − dummy baseline. | Behavioural self-modelling benchmark; the dummy-baseline convention (we adopt it). | No internal interventions; explicitly no privileged-access claim. |
| **Looking Inward** (Binder et al. 2410.13787) | Trained self-prediction vs cross-prediction (a model trained on the subject's behaviour). Small self-advantage after training. | The cross-prediction observer idea (we adopt it). | Behaviour only. |
| **Strangers to Themselves** (2609.09899) | Cross-model mean predicts a model's behaviour better than its own report (+0.53 vs +0.04). Pre-registered. | "Models predict their own behaviour" claims. | Behaviour only. |
| **Lindsey 2601.01828 (+ Macar 2603.21396 mechanism)** | Concept-injection detection about 20% on Opus 4.1 with 0/100 false positives. Random vectors 9/100; negated vectors comparable. | Concept-injection detection. | Content-specific access unshown; no equal-information observer. |
| **Open-weight replications** (Lederman & Mahowald, Hahami, Zou 2609.35108, Latent Introspection 2602.20031) | Detection is content-independent, or yes-bias (r = 0.999). Identification ≤ 13.9%. Sentence-position localisation 44–66% vs random 0–6.7%. | Further injection-detection studies. | Layer or component localisation. |
| **OWMI** (Ferrara 2608.20569; code `emilioferrara/owmi`) | Content-free perturbations (residual, head, SAE, block) on 8 models 0.49–14.8B; paired sham, d′. Chance detection. Probes decode (0.958/0.750); trained LoRA d′ = 5.15; graded-confidence AUROC 0.647. Layer naming refused 967/1,184. Reusable statistics module. | "Perturb and ask whether it noticed." | Code issues: edits hit only template positions 0–2 during prefill; the lifted arm cannot carry signal. No observer ladder, no counterfactual prediction. |
| **Introspection Adapters** (2604.16812) | SFT+DPO LoRA makes fine-tuned models describe implanted behaviours (AuditBench 59%). | "Train an adapter to describe learned behaviour." | Ground truth is a training property, not a current state. No interventions, observer or calibration. May be classifiers (2608.04347). |
| **Reality Check** (Singh, Linzen, Ravfogel, 2605.26242) | "Gaslight" prompt manipulation breaks injection detection. Layer-0 probes match neurofeedback self-labels. | Two-way detection designs; neurofeedback-as-introspection. | Leaves "second-order computation" tests open. |
| **Neurofeedback / metacognition** (Ji-An 2505.13763; Kadavath 2207.05221; Cacioli 2603.25112) | In-context reporting and control along directions; P(IK); Type-2 SDT on QA. | Confidence-on-QA studies. | meta-d′ applied to claims about internal causes. |
| **Interpretability agents** (MAIA 2404.14394, Anthropic auditing agents, AuditBench 2602.22755, HARP 2607.16448, HyVE 2606.24026, SAEScientist-Bench 2609.09113) | Tool-using agents investigate *other* models. Known failure modes: the tool-to-agent gap, confirmation bias, weak validation loops. | Building a generic interpretability agent. | No agent investigates *itself*. No self vs other comparison with identical tools. |
| **Construct validity and discovery** (Cronbach & Meehl, Borsboom, MTMM, Bean 2511.04703; Schut 2310.16410; neologisms 2510.08506; persona vectors 2507.21509) | Validity theory; machine-native concept discovery with teachability and novelty filters. | Ad-hoc "new construct" claims. | No registry of machine-native constructs with rivals and falsification conditions. "Plurience" is an unused term. |

---

## B. Current-repository deletion map

The inventory covers all tracked files at `4b94d45`. Product code is ≈ 20k lines: ≈ 4.7k Python library and API, ≈ 12.8k web, ≈ 1.6k tests and ≈ 0.8k scripts. There are also ≈ 31k lines of stored research JSON.

**Dependency tracing.** The reproducible research assets (E1/E2) import only `config` plus `instrumentation`. No service, store, API, UI or Math Lab code is needed to reproduce them.

### B.1 Phase-1 table

| # | Existing component (path, lines) | What it does | Existing external equivalent | Better / worse | Action | Reason |
|---|---|---|---|---|---|---|
| 1 | Hook layer: `instrumentation/hf_decoder.py` (472), `adapter.py` ABC and containers (216), `__init__` (60) | Captures residuals, eager attention, per-head z and MLP outputs. Applies edits. Batched edit sets, resume-from-depth, KV-cached greedy generation. | nnsight 0.7.0 (bit-exact, same speed; `.skip()` equals resume); interp-engine; TL 4.0 bridge; nnterp | Equal on exactness and speed (336-head sweep 7.6 s vs 7.8 s). **Worse** on coverage (3 families vs dozens), no gradients, no remote/NDIF, no vLLM. | **Thin adapter** | Rule 11. Keep the file frozen as a parity reference until the NNsight provider reproduces E2 exactly (M2), then delete. |
| 2 | Edit vocabulary: `adapter.py` Edit dataclasses; `schemas/interventions.py` (120) | Declarative, serialisable, validated interventions: head scale/set, MLP scale, residual set (patch), residual add (steer), with `describe()`. | pyvene `IntervenableConfig` (declarative but broken for heads on transformers 5.x); nnsight (imperative; needs source files); interp-engine (steering only) | **Better.** It is the only validated, serialisable spec a model could safely propose and a pre-registration could freeze. | **Keep** and extend | Core of SelfLab's bounded intervention language (§E). |
| 3 | DEMO fixture: `instrumentation/demo.py` (47) | Tiny random-weight Qwen2 with a real tokenizer, for fast tests. | none (test utility) | — | **Keep** (test fixture) | Fast CI for providers, scoring and claims. Works under nnsight too. |
| 4 | Logit Lens: `interpretability/lens.py` (91) | Final-norm + unembed readout; top-k per cell; target rank by depth. | interp-engine `decode_residuals` (identical output); nnterp; TL; Jacobian Lens (better at early depths) | Equal or worse | **Replace** | Generic. When a dossier needs a lens, call the provider (or J-lens once fitted). |
| 5 | DLA: `interpretability/attribution.py` (90) | Exact direct logit attribution with frozen norm scale (error 4e-6); head vocabulary projection. | TL 4.0 DLA (not numerically compared); interp-engine `head_contributions` (ingredients only) | Equal; ours is small and tested | **Thin adapter** (re-home to `selflab/instruments`) | A cheap, exact observational instrument for CSP-1 dossiers. |
| 6 | Metrics: `interpretability/metrics.py` (33) | KL, entropy, cosine, rank. | numpy/scipy | — | **Keep** (fold into scoring) | Trivial. |
| 7 | Service views: `experiments/service.py` run/trace/view parts (≈ 600 of 1,146) | Backend for the token×layer matrix, cell, lens, attention, head, DLA and diff views. | Neuronpedia UI/APIs; TL/nnsight notebooks | Worse (generic viewer) | **Delete** | Rule 2; "another activation viewer" is explicitly out of mission. |
| 8 | Service intervention runs and comparisons (≈ 180) | Baseline-vs-intervention Δ logit/prob/rank, KL, flips, changed tokens. | Primitives exist; no harness | Useful logic, UI-shaped payloads | **Replace** with `selflab/groundtruth` | Keep the metric logic; drop view payloads. |
| 9 | Sweeps: `head_ablation_sweep`, `patch_sweep` (≈ 130) | Exhaustive batched head-ablation and residual-patching sweeps with resume. | nnsight primitives (same speed with `.skip()`), no orchestration; TL patching utilities (≈ 6× slower) | Better orchestration | **Thin adapter** (re-home over provider) | Ground truth for self-localisation (Y4). |
| 10 | Service Math Lab orchestration, lazy re-capture, locks (≈ 230) | Benchmark runner, benchmark intervention eval, trace-level upgrades. | — | — | **Delete** | Math Lab no longer drives the roadmap. The eval pattern moves into ground truth. |
| 11 | Trace store: `experiments/store.py` (314) | SQLite metadata + safetensors activation traces; experiments; benchmarks. | none | Useful provenance, wrong object | **Replace** with an append-only SelfLab ledger | SelfLab needs committed predictions, outcomes and claims. Activations are recomputable deterministically. (Deliberately overrides the frameworks note and the prior-art report, which advise keeping it: their reason is provenance, which the ledger keeps, while SelfLab has no use for stored activation tensors.) |
| 12 | Jobs: `experiments/jobs.py` (68) | Background job runner for the API. | — | — | **Delete** | Batch CLI instead. |
| 13 | HTTP API: `apps/api` (355), `schemas/api.py` (162) | FastAPI for the viewer and Math Lab. | Neuronpedia API for generic needs | — | **Delete** | CSP-1 is a batch pipeline; an API returns only if a claims ledger viewer needs one. |
| 14 | Math Lab core: `mathlab/` (1,472), `tests/test_mathlab.py` (722, 244 tests) | Seeded generator, SymPy oracle, conservative parser (truncation/ambiguity aware). | none needed (domain code) | Good; well tested | **Keep**, demoted to task oracle | Parses and grades task T1 answers under intervention; supports SELF-ACCURACY-style calibration items. |
| 15 | `config.py` (70) | Environment settings. | — | — | **Keep** (trim) | — |
| 16 | Tests: instrumentation invariants (199) | Exact invariants: decomposition, downstream-only ablation, batch = single, resume = full. | — | — | **Keep**, becomes the provider parity suite | Runs against both providers until #1 is deleted. |
| 17 | Tests: service (229), api (120), mathlab_service (61), lens part of interpretability (≈ 50) | Viewer, API and Math Lab service tests. | — | — | **Delete** | They follow the deleted code. Keep the DLA exactness test. |
| 18 | Tests: real-model integration (104) | End-to-end on Qwen2.5-0.5B. | — | — | **Rewrite** | Provider parity and E2 reproduction on real weights. |
| 19 | **Carry-head experiments:** `scripts/experiments/carry_heads.py` (286), `carry_patch.py` (136), `docs/research/E1_carry_heads.json`, `E2_carry_patch.json` (≈ 31k lines), RESEARCH_LOG entries | E1: L22.H10/H11 necessary for the carry (n = 80, controls, mean-ablation). E2: sufficient (35/40 inject, 37/40 remove, controls 0/40). | none | — | **Research asset** | Move to `research/legacy/carry_heads/` with a reproduction README. The E2 rerun becomes the acceptance test of the new provider (M2). E1 is also a known-positive localisation anchor for CSP-1. |
| 20 | Ops scripts: `acceptance.py` (184), `smoke_browser.cjs` (35), `export_hero_snapshot.py` (106), `dev.sh`/`api.sh`/`web.sh` (45) | API/UI acceptance, hero export, launchers. | — | — | **Delete** (`setup.sh` and Makefile are rewritten) | Tied to deleted API/UI. |
| 21 | Web workspace: `components/workspace/**` (6,097), `app/app/**` (115), `lib/api/**` (1,035), `lib/{matrix,metrics,format,colormap}.ts` (541), `__tests__/workspace` (671) = **8,459** | Token×layer matrix, inspector tabs (trace, lens, attention, interventions, diff), sweeps, Math Lab UI. | Neuronpedia (features, steering, circuit graphs, lens); notebooks | Duplicative and out of mission | **Delete** | "Another activation viewer / generic mech-interp frontend." Preserved in the tag. |
| 22 | Marketing site: `app/(marketing)`, `components/{marketing,brand,docs,ui}`, `lib/{epistemic,hero-trace,cn}.ts`, globals, config, tests (≈ 4.6k) | Landing, research and docs pages for the debugger mission; design system; epistemic badges. | — | Copy now obsolete | **Deprecate** (option A) or **delete** (option B) | Its copy describes the old mission. Its design system and epistemic ladder are reusable for communicating verified self-knowledge claims later (M8). |
| 23 | Epistemic discipline: status fields in service, `lib/epistemic.ts`, `EpistemicBadge`, `docs/INTERPRETABILITY.md` § statuses | OBSERVED / DERIVED / INFERRED / CAUSALLY TESTED labels on every quantity. | none | Unique discipline | **Keep** and extend to 8 statuses (§4) | Required by the new mission. |
| 24 | Docs: README, ARCHITECTURE, INTERPRETABILITY, EXPERIMENTS; workspace screenshots | Debugger-era documentation. | — | — | **Rewrite** (screenshots deleted); RESEARCH_LOG **kept** append-only | — |

### B.2 Numbers

| | Lines | Components |
|---|---|---|
| Deleted, option A | ≈ 11.2k | #7, 10, 12, 13, 17, 20, 21, parts of #4 |
| Deleted, option B | ≈ 15.8k | Option A + #22 |
| Replaced by a thin adapter (#1) | ≈ 0.75k → ≈ 0.3k | nnsight provider |
| Moved as a research asset | ≈ 0.42k code + ≈ 31k data lines | #19 |
| Kept | ≈ 2.9k | #2, 3, 5, 6, 14, 15, 16, 23 (incl. Math Lab tests) |
| Kept dormant under option A | ≈ 4.6k | #22 |

### B.3 Deletion protocol

1. Before deleting anything, tag the current head `v1-neural-debugger` and push the tag, so everything remains recoverable.
2. Delete #21, #13, #12, #7, #10 and #17 in one PR (M1).
3. Delete #1 only after the NNsight provider reproduces E2 exactly (M2).
4. Never delete #19.

**Option A** (recommended) keeps the marketing site frozen until there is a verified SelfLab result to communicate (M8). **Option B** deletes it too. That removes a Next.js dependency tree, but loses a reusable design system.

---

## C. Unique thesis

NeuralScope measures self-knowledge instead of eliciting self-description. It turns what a language model says about its own computation into pre-committed, probabilistic predictions about the outcomes of randomised interventions on its own internals. It executes those interventions with established tooling (NNsight). It then scores the model against a ladder of equally informed observers: a base-rate learner, other models, a same-family model, the same weights under third-person framing, and an observer holding the same interpretability instruments. The scoring uses proper scoring rules, calibration, metacognitive sensitivity and conditional V-information.

The output is a ledger of falsifiable self-knowledge claims and a per-model profile of what the system verifiably knows about itself. That includes the possibility, currently the most likely outcome, that it knows nothing an informed outsider could not infer.

Existing work reads activations (NNsight, Neuronpedia, Activation Oracles). It detects perturbations after the fact (OWMI, concept injection). It trains reporters (Introspection Adapters, Li et al.). Or it measures behavioural self-modelling (Binder, Zeng, CHIVE). None of it scores a model's own zero-shot causal predictions about its internals against equal-information observers. None lets a model run bounded experiments on itself while an identically equipped outsider runs the same experiments as the control.

---

## Phase 2. The actual gap, and corrections to the working hypothesis

### 2.1 An identification result that changes the design

For a deterministic transformer, the internal state Z at a token is a function of the visible context X and the fixed weights. Therefore:

1. **Shannon I(Z;R|X) = 0 unless the experimenter randomises something X does not reveal.** That something is a hidden intervention or sampling noise. Privileged access in the strict, information-theoretic sense is only identifiable relative to randomised hidden manipulations. That is why the concept-injection and OWMI paradigms exist. With randomisation, the test of R ⊥ Z | X is exact: a conditional randomisation test, because we know P(Z|X).
2. **Everything else is computational self-knowledge.** A model predicting what an announced intervention will do is predicting a deterministic function f_W(item, intervention). What it can have is a predictive advantage relative to a bounded observer class. This is conditional predictive V-information (Xu et al. 2002.10689; Hewitt et al. 2109.09234), not Shannon information. The observer class V must be declared and pre-registered.
3. **Most published "self-advantage" is plausibly shared-weights self-simulation or representational similarity, not state access.** Li et al.'s advantage survives removing the activation and tracks cross-model activation similarity; Strangers and CHIVE find none. A same-family observer and a third-person-framed self are the controls that separate these.

So SelfLab has **two estimands**, and they are never conflated:

| | Estimand | Needs | First measured in |
|---|---|---|---|
| **Structural (computational) self-knowledge** | Skill of the model's predictions about causal facts of its own computation, relative to the best equal-information observer; V-relative | Announced interventions; observer ladder | CSP-1 |
| **Occurrent privileged access** | I(Z; R \| X) for experimenter-randomised hidden internal conditions Z, with an exact CRT | Hidden randomised interventions; same-weights twin; OWMI-grade hygiene at *content* positions | CSP-2 |

### 2.2 The observer ladder

A self-report is evidence of self-knowledge only to the extent it beats every rung below. Each rung receives exactly the external information the subject receives: the prompt, its answer, the architecture primer, the intervention description and the candidate options.

| Rung | Observer | Controls for |
|---|---|---|
| O₀ | Best constant predictor | Base rates (Zeng's "skill") |
| O₁ | Learned outsider: a regression on surface features of (item, intervention), trained on *other* items' executed outcomes for this model | What external experimentation cheaply yields; surface-information confound (#10) |
| O₂ | Different-family LLM, zero-shot | Generic knowledge of how transformers behave (#11, #22) |
| O₃ | Same-family, different-size LLM | Representational similarity (the Li et al. confound) |
| O₄ | The subject itself, third-person and generic framing | Self-reference: does "you" add anything over "a 28-layer transformer"? (#19 context-bound reporting; Assistant-Axis drift) |
| O₅ | Instrument-equipped observer: O₂ plus the same interpretability dossier the subject gets | Whether instruments, not self, carry the information |

### 2.3 The user's hypothesis diagram, corrected

The diagram is directionally right. Four corrections:
1. **Self-report is not an input to a "verified self-model".** It is a *prediction* that is scored. The verified self-model is the set of scored claims that survived, with their observer margins.
2. **The observer ladder is mandatory, not optional.** Without it nothing distinguishes self-knowledge from interpretability that happens to be performed by the subject.
3. **There are two estimands (§2.1).** "Privileged access" must name which one.
4. **The loop needs two things to be science rather than theatre:**
   - a bounded intervention language: the model proposes experiments only in a validated grammar;
   - a commitment protocol: hypothesis, then predicted effect and confidence, then hash-locked, then executed, then scored.

   "Self-discovery" must also be compared against an other-model agent with identical tools. Otherwise it is just interpretability by an agent.

```
   EXTERNAL STACK (NNsight · interp-engine · Neuronpedia · SAELens · NDIF)
                            │  activations, interventions, decoders, lenses
                            ▼
   ┌──────────────────────── SELFLAB ────────────────────────────────┐
   │ bounded intervention language ──► randomised execution ──► outcomes (OBSERVED)
   │        ▲                                                     │
   │ subject's committed predictions (SELF-REPORTED, hash-locked) │
   │ observer ladder O₀…O₅ predictions (same information)         │
   │                            ▼                                  │
   │ scoring: proper scores · calibration · meta-d′ · V-information · CRT
   │                            ▼                                  │
   │ claim ledger: SelfKnowledgeClaim with 8 epistemic statuses    │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
         self-knowledge profile (C, L, F, G, I, A) — may be all zeros
                                ▼
           CSP-3: the model proposes new experiments on itself,
           scored against an identically equipped outsider  ↺
```

### 2.4 Novelty, stated conservatively

As of 2026-09-30, no located work combines all of the following:

| # | Ingredient | Closest existing work, and what it lacks |
|---|---|---|
| 1 | Zero-shot predictions of *internal* intervention outcomes, committed before execution | Li et al.: trained, four layer blocks. Zeng and CHIVE: prompt edits only. Konrad 2609.32449: steering without an observer. Tomaszewski 2608.14894: toy networks. |
| 2 | Head, representation or position granularity, including compensation (self-repair) and self-localisation | Compensation ground truth exists (COAX 2607.01940, backup heads); prediction of it does not. Localisation succeeded only at the token or sentence level. |
| 3 | A multi-rung equal-information observer ladder | — |
| 4 | Calibration and meta-d′ on causal self-claims | Calibration exists for QA and behaviour only. |

The self-discovery loop with an identically tooled outsider (CSP-3) is also unlocated. Every tool-using interpretability agent studies another model.

The field is moving fast: five relevant papers in August–September 2026. Searches must be repeated before any publication claim.

---

## Phase 3. SelfLab evidence sources

| # | Evidence source | Question | Methods (external provider where possible) | Status it can confer | What it can *not* establish | In CSP-1? |
|---|---|---|---|---|---|---|
| 1 | Behavioural self-report | What does the model claim about its internals? | Forced-choice elicitation with log-probability readout; framings; paraphrases; option permutations | SELF-REPORTED only | Anything about truth; fluency is not evidence | Yes (as predictions) |
| 2 | Internal decodability | Is the claimed state independently recoverable? | Linear probes with control tasks, selectivity, a layer-0 baseline and a shuffled-label null; SAEs (SAELens/Neuronpedia) for Qwen3; NLA (Qwen2.5-1.5B L18, via Neuronpedia); Activation Oracles where released | EXTERNALLY DECODED (only if admission tests pass; else DERIVED) | That the model uses the state, or knows it | Instruments in dossiers only |
| 3 | Causal efficacy | Does intervening on the state change behaviour as predicted? | Zero, mean or resample ablation; patching; impact-matched random directions; via NNsight | CAUSALLY TESTED (scoped) | Semantic content of the state (INFERRED) | Yes (ground truth) |
| 4 | Privileged self-access | Does the model know something about its current internal condition an equal-information observer cannot? | Randomised hidden interventions; same-weights twin; I(Z;R\|X) via cross-fitted log-loss gain; exact CRT/HRT | SELF-PREDICTED + CAUSALLY VERIFIED when combined with #6; otherwise a DERIVED statistic | Structural self-knowledge (a different estimand) | No: CSP-2 |
| 5 | Self-model calibration | When it says 80%, is it right 80% of the time? | smECE (Błasiok & Nakkiran), CORP reliability diagrams, Brier/log decomposition, risk–coverage for abstentions | DERIVED property of a claim class | Accuracy by itself | Yes |
| 6 | Counterfactual self-prediction | Does the model predict what an intervention on itself will do? | Commit, then execute, then score against the observer ladder | SELF-PREDICTED + CAUSALLY VERIFIED (claim-class level, CI beats the best observer) | Mechanism; occurrent access | **Yes (primary)** |
| 7 | Self-localisation | Can it say where decision-relevant information lives? | Candidate-set ranking scored by NDCG@k and log score against executed sweeps; E1/E2 as a known-positive anchor | As #6 | That the named site "represents" a concept | Yes (secondary) |
| 8 | Self-discovery | Can it build a more accurate self-model by running experiments on itself? | Bounded read-only tools plus a commit-then-execute protocol; an identically tooled other-model agent as control; budgets; execution-grounded scoring | Claims it produces enter the ledger with ordinary statuses; the *capability* is measured by profile dimension I | Privilege unless it beats the equally tooled outsider | No: CSP-3 |

---

## Phase 4. Epistemic claim system

### 4.1 Statuses

Each *evidence item* carries exactly one status. A *claim* carries a profile of statuses (supporting and contradicting), never a collapsed label. Admission rules are deterministic code, and a status is only granted when its rule passes.

| Status | Definition | Admission rule (all required) |
|---|---|---|
| **OBSERVED** | Direct measurement: a token, a logit, an activation value, an intervention outcome | Provenance: model and weights hash, provider and version, code commit, inputs hash |
| **DERIVED** | Deterministic function of OBSERVED items: scores, statistics, lens projections, decoder outputs | Function id and version; inputs are OBSERVED/DERIVED ids |
| **SELF-REPORTED** | What the subject asserts about itself, parsed into a proposition | Elicitation protocol id, framing, context hash, parse status. For predictions: commitment hash and time *before* the corresponding outcome exists |
| **EXTERNALLY DECODED** | Proposition recovered by an independent method (probe, SAE, NLA, AO, J-lens) | Held-out performance above control-task and shuffled-label nulls; beats the layer-0 / text-only baseline; decoder is not the subject in situ. Otherwise stays DERIVED |
| **CAUSALLY TESTED** | Executed intervention with a pre-specified predicted effect | Effect CI excludes 0 in the predicted direction; specificity control (impact-matched random or other component) is smaller; scope recorded |
| **SELF-PREDICTED + CAUSALLY VERIFIED** | The subject's own committed predictions about interventions on itself were scored against executed outcomes and beat the best pre-registered observer | Claim-class level (never a single prediction). Commitment precedes outcome. Skill vs O\* has a CI > 0 (primary: > SESOI). Calibration reported |
| **REPLICATED** | *Qualifier* on the three statuses above: the effect holds across new units | ≥ 2 independent units of a pre-registered replication dimension (fresh items, paraphrases, seeds, models), each with its own CI |
| **INFERRED** | Semantic interpretation ("routes the carry", "a Plurience signature") | Always INFERRED. May cite supporting evidence ids; never upgraded by them |

### 4.2 `SelfKnowledgeClaim`: specification

This is the schema to be implemented in M5 as a Pydantic model with validators. It is listed here as a specification only.

| Field | Content |
|---|---|
| `id`, `version`, `created_at`, `author` | `author ∈ {human:<id>, model:<id>, pipeline:<id>}` |
| `subject` | `{model_id, weights_sha256, revision, dtype, provider, provider_version}` |
| `proposition` | `{kind ∈ [state_present, state_value, causal_dependence, counterfactual_outcome, location, compensation, calibration, construct_measure], target: {site \| representation \| construct_id}, predicate, value, units}` |
| `statement` | Natural-language rendering generated **from `proposition` by template**, never free text |
| `scope` | Task family, item hashes, conditions, intervention specs, framing, protocol version |
| `self_report` | SELF-REPORTED items: elicitation id, framing, raw text or log-probabilities, parsed probability, `commit_sha256`, `committed_at` |
| `decoded_evidence` | EXTERNALLY DECODED items, each with its admission-test results |
| `causal_evidence` | CAUSALLY TESTED items: intervention spec, effect estimate, CI, n, specificity control |
| `observer_control` | Per rung: `{observer_id, class, info_set_hash, score}`; `best_observer` (chosen on the pilot split); `self_minus_best {estimate, ci, test, p}`; `v_information {bits, ci, hrt_p, V}` |
| `calibration` | n, log score, Brier (CORP decomposition), smECE, reliability curve, meta-d′ / M-ratio, abstention coverage |
| `replication` | Units, per-unit estimates, heterogeneity |
| `alternatives` | `[{explanation, discriminating_test, evidence_ids, state ∈ ruled_out \| not_ruled_out \| untested}]` |
| `falsification` | `{criterion, threshold, prereg_id, outcome ∈ pending \| survived \| falsified}` |
| `epistemic_profile` | *Computed*: status → {supporting ids, contradicting ids} |
| `verdict` | *Computed*: SUPPORTED \| REFUTED \| EQUIVOCAL \| UNTESTED, plus the strongest admitted status and REPLICATED qualifier |
| `construct_id`, `lineage` | Optional link to the Construct Registry; parent or refined claims |

Invariants enforced in code:
- A claim whose only evidence is SELF-REPORTED has `verdict = UNTESTED`.
- `statement` cannot be edited independently of `proposition`.
- A prediction whose `committed_at` is later than its outcome's creation time is rejected.
- `REPLICATED` requires an underlying admitted status.

---

## Phase 5. Verified self-knowledge profile (no scalar)

Each dimension is computed per `(model, task family, granularity, protocol version)` and reported with a CI, n and coverage. Until a dimension is measured it is reported as "not measured", never as a number.

| Dim | Name | Estimand (vs baseline) | Estimator | Measured by |
|---|---|---|---|---|
| **C** | Causal self-prediction | Log-score skill of committed intervention-outcome predictions vs the best observer O\* (bits/prediction); also vs O₀ | Paired, item-clustered differences; cross-fitted conditional V-information I_V(R_S→Y \| X, R_O\*) | CSP-1 |
| **L** | Self-localisation | NDCG@k (and log score) of predicted causal-site rankings vs O\*, against executed sweeps | Paired differences; near-equivalent ground-truth sets | CSP-1 (secondary) |
| **F** | Calibration and metacognitive sensitivity | smECE (lower is better); resolution; meta-d′ / M-ratio of confidence on self-predictions | smECE, CORP, Maniscalco–Lau | CSP-1 |
| **G** | Generalisation | Skill on a held-out task family and a held-out intervention type, relative to in-distribution | Same as C, split by pre-registered hold-outs | CSP-1 |
| **I** | Instrument leverage | Self-specific gain from interpretability instruments: [S+I − S] − [O₂+I − O₂] | Difference-in-differences, cluster bootstrap | CSP-1 (2×2 arm) |
| **A** | Occurrent privileged access | I(Z;R \| X), in bits, for randomised hidden internal conditions, vs same-weights twin | Cross-fitted log-loss gain; exact CRT | CSP-2 |

**Always-reported diagnostics** (these are not dimensions):
- self-reference specificity (first-person minus third-person skill);
- confident-confabulation rate on decoy (non-existent) components;
- refusal and abstention rates;
- position and yes-bias indices.

The dimensions differ from the proposed (E, C, L, F, G) in three ways:
- **E is split into A (occurrent) and C (structural)**, because §2.1 shows they are different quantities.
- **I is added.** It is the operational form of "can instruments expand self-knowledge?".
- **No weighting into a composite.** Any scalar would hide exactly the dissociations (e.g. high C, zero A) that are scientifically interesting.

---

## Phase 6. Construct Registry

Constructs are registered as *hypotheses*, one YAML file each. The fields follow Borsboom (the attribute must exist and cause variation in the measure), the multitrait-multimethod matrix (convergent and discriminant validity), Schut et al. (teachability and novelty filters) and Bean et al. (construct-validity recommendations).

**Required fields:**
- `id`, `name`, `status` (hypothesis → registered → under_test → supported | refuted | merged_into:<id>)
- `definition` (functional or causal terms only)
- `background`, `rivals`
- `operationalisations` (≥ 2 independent methods, from behavioural, decoding and interventional)
- `predicted_signatures` (behavioural, representational, causal)
- `rival_explanations` (each with a discriminating test)
- `causal_tests`, `teachability_test`, `novelty_test` (vs human-concept bases)
- `falsification` (pre-registered)
- `observer_class`, `linked_claims`, `changelog`

**Entry 1 — registered as a hypothesis, not as truth:**

```yaml
id: C-0001
name: Plurience
status: hypothesis            # "Plurience" has no prior use in AI or cognitive science (searched 2026-09-30)
definition: >
  The integrated capacity to preserve, relate and act upon multiple mutually incompatible
  world-models, each with a calibrated weight, without prematurely collapsing to one.
operationalisations:
  - decoding: simultaneous linear decodability of incompatible hypothesis variables (belief-state
    geometry, Shai et al. 2405.15943) at the decision position, with control tasks and selectivity
  - behaviour: action choice under ambiguity that hedges or seeks information in proportion to the
    decoded weights (AmbiEnt-style items, 2304.14399; belief dominance, 2602.02467)
  - intervention: ablation or steering of the minority-hypothesis subspace
predicted_signatures:
  representational: both hypotheses decodable at weights tracking evidence, not output entropy alone
  causal: removing the minority subspace causes premature collapse (lower hedging, worse
    performance after a reversal cue) greater than an impact-matched random ablation
rival_explanations:
  - {explanation: output-distribution (posterior) entropy only,
     test: does the mixture representation add V-information about behaviour beyond output entropy?}
  - {explanation: working memory / context retrieval,
     test: effect persists when both hypotheses are re-stated in context (memory load equalised)}
  - {explanation: beam-search-like enumeration or prompt-induced verbal listing,
     test: signature present under forced single-token answers with no enumeration in context}
  - {explanation: ensemble diversity across samples,
     test: signature present within a single greedy forward pass}
  - {explanation: ordinary cognitive flexibility / task switching,
     test: dissociation — flexibility benchmarks do not predict the signature}
  - {explanation: stylistic hedging persona (Assistant Axis, persona vectors),
     test: signature survives persona-direction capping; hedging text without the representation fails}
falsification:
  - mixture representation adds no V-information about behaviour beyond output entropy (TOST)
  - interventions on the subspace leave behaviour unchanged vs impact-matched random ablation
  - fails the discriminant test against stylistic hedging
  - fails teachability (a student model given prototypes does not improve on reversal tasks)
observer_class: O2/O5 as in Phase 2 (for any self-report about Plurience states)
```

The registry is data plus a validator. Adding a construct means adding a YAML file that passes the validator and linking claims to it.

---

## D. First experiment: CSP-1, counterfactual self-prediction with an observer ladder

**Question.** Can a language model predict the causal consequences of specified interventions on its own internals, before they are executed, better than equally informed observers?

This goes beyond OWMI, which detects *after* a perturbation. It goes beyond Activation Oracles, which use a trained external reader, and beyond Introspection Adapters, which report trained-in behaviour. It extends Li et al. from trained, block-level patch prediction to zero-shot, component and representation level prediction with compensation, calibration and a full observer ladder.

### D.1 Hypotheses (pre-registered; H1 is the single primary)

**H1, primary (C).** On held-out (item, intervention) pairs, the subject S has higher expected log score on Y1 (answer flip) than the best equal-information observer O\*, chosen on the pilot split from {O₁, O₂, O₃}.

**Secondary hypotheses** (BH-FDR within the family):

| # | Profile dim | Hypothesis |
|---|---|---|
| H2 | C | S beats O₀ (skill > 0) on Y1, Y2 and Y3 |
| H3 | L | S's candidate-site rankings beat O\* on NDCG@3 |
| H4 | F | S's smECE is lower than O\*'s. S's meta-d′ > 0 |
| H5 | I | Self-specific instrument gain > 0 |
| H6 | G | H1's effect holds on the held-out task family and held-out intervention type |
| H7 | — | First-person skill exceeds third-person/generic self-framing (self-reference specificity) |

**Exploratory:** compensation prediction (Y5); decoy confabulation; internal vs input-edit skill gap.

### D.2 Design

**Subjects:**
- Qwen2.5-1.5B-Instruct: pilot and known-positive anchor. The E1/E2 carry mechanism is fully mapped, and an NLA exists at L18. There are no SAEs, transcoders, Jacobian lens or Activation Oracle for it.
- Qwen3-1.7B: main. It is the only CPU-scale model for which every external instrument already exists (report, coverage table):
  - Qwen-Scope residual SAEs, trained on the *base* model. Transfer to the post-trained model is checked in M4.
  - Neuronpedia L14 SAEs.
  - Transcoders and attribution graphs.
  - A pre-fitted Jacobian lens.
  - An Activation Oracle checkpoint.

  It is directly comparable with Li et al., CHIVE, Zou et al. and the Introspection Adapter runs, which all used Qwen3.
- Replication on Qwen3-4B/8B (GPU, or NDIF once hot) and Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct via NDIF, only if you approve a key (§0).

**Task families (items):**

| Family | Items | Oracle / notes |
|---|---|---|
| T1 arithmetic | Two-digit sums, carry and no-carry | Math Lab oracle; known mechanism for Qwen2.5-1.5B |
| T2 factual recall | Single-token answers, e.g. capitals | — |
| T3 in-context induction/copying | — | **Held-out family** for G |

**Independent variables:**
1. **Intervention**, drawn per item by pre-registered stratified randomisation from a catalogue:
   - single-head zero-ablation;
   - single-head mean-ablation;
   - attention-layer ablation;
   - MLP-block ablation;
   - information ablation (resample the residual at a named token position from depth d onward);
   - impact-matched random residual perturbation;
   - input edit (behavioural comparison arm);
   - SAE-feature ablation (Qwen3 only; the feature's label is presented as a label, i.e. INFERRED);
   - decoy (non-existent component; confabulation check, excluded from primary).

   Strata: type × depth third. Flip-rate balancing uses the pilot split only.
2. **Predictor**: S (first-person), O₀–O₅.
3. **Evidence condition**: none vs instrument dossier. The dossier contains, for the evaluated item:
   - DLA top components;
   - Logit Lens or J-lens trajectory of the answer;
   - attention summaries of the candidate heads;
   - on Qwen3-1.7B only, top SAE features at the answer position and the Activation Oracle's answer to a fixed question about the answer position. Both are *external decoders*, so their outputs are DERIVED instrument readings, never evidence of self-access.

   Plus k = 8 executed (item, intervention, outcome) examples from *training* items. It never contains the evaluated pair's outcome.
4. **Framing** of S: first-person / third-person named model / generic 28-layer transformer.

**Dependent variables** (ground truth OBSERVED/DERIVED from executed interventions):

| Variable | Definition | Role |
|---|---|---|
| Y1 | Answer flip (argmax change of the answer token), binary | **Primary** |
| Y2 | New answer, categorical over options | Secondary |
| Y3 | \|Δ log p(original answer)\| in 3 pre-registered bins | Secondary |
| Y4 | Ranking of a K = 5 candidate-site set by causal importance, from sweeps (top site plus 4 stratified random sites) | Secondary |
| Y5 | Compensating component among candidates: largest importance increase when the target is ablated, from pairwise ablations (T1 on Qwen2.5-1.5B, where H10/H11 give a known pair) | Exploratory |

**Elicitation.** Every predictor gets identical text: the task, the subject's baseline answer, a fixed architecture primer (layer and head counts, what the intervention does), the intervention description and lettered options. Probabilities are read from next-token log-probabilities over option letters, renormalised, with two option orders averaged. There is no sampling and no parser. Verbal confidence with an allowed "cannot say" is collected on a 5% subset and treated as abstention (risk–coverage).

### D.3 Controls

Confound numbers (#n) refer to the 24-item catalogue in the introspection notes §13. The prior-art report extends it to 32 confounds and a 19-item checklist, adding decoder, agent and infrastructure confounds such as persona drift and shared tool outputs. M7's pre-registration will map CSP-1 to the full 32/19.

| Control | Addresses |
|---|---|
| Prediction committed (hash and time) **before** the intervention is executed; the prediction forward pass is asserted edit-free in code | Output reading (#6); causal bypass (#7) — satisfied by design |
| Observer ladder O₀–O₅ with identical information; O\* fixed on the pilot split | Surface information (#10), model similarity (#11), difficulty (#22), yes-bias (#1) |
| Impact-matched random perturbations and null-effect components in the catalogue | Specificity nulls (#3–#5), instrument insensitivity (#24) |
| Input-edit arm alongside internal interventions | Input-vs-internal discrimination (#3); separates behavioural self-modelling from internal causal knowledge |
| Decoy (non-existent) components | Confabulated identification (#8) |
| First/third/generic framings, question paraphrases, option-order permutation; Assistant-Axis projection logged during elicitation | Context-bound reporting (#19), persona drift |
| Log-probability readout, balanced Y1, d′/AUROC reported | Parse failure (#16), constant responders (#17), yes-bias (#1) |
| O₁ trained on executed outcomes of *other* items; E1/E2 carry mechanism as a known-positive localisation anchor | Sensitivity anchors (checklist 9) |
| Pre-registration of catalogue, sampling, endpoints, SESOI and analysis, frozen and hashed before the test split is touched | Selection bias (#15), low power (#23) |

### D.4 Statistical analysis

- **Primary test.** Paired per-prediction log-score difference S − O\* (bits) on Y1.
  - Two-way cluster bootstrap 95% CI (items × components).
  - One-sided item-level sign-flip permutation test, α = 0.05.
  - Gatekeeping: secondaries are interpreted only if the primary passes; otherwise they are descriptive.
- **Null claims.** TOST with a smallest effect size of interest (SESOI) of ±0.03 bits/prediction, fixed after the pilot and before the test split. Brier and AUROC (DeLong, paired) are also reported.
- **V-information.** Cross-fitted (5-fold by item) L2-logistic probes predict Y1 from [φ(X), logit R_O\*] with and without logit R_S. The held-out log-loss gain in bits is I_V(R_S → Y1 | X, R_O\*). Significance is a holdout randomisation test permuting R_S within intervention-type × family strata. It is labelled *V-relative computational* self-knowledge, not privileged access.
- **Calibration.** smECE (`relplot` or an equivalent implementation validated on synthetic data), CORP reliability diagrams, meta-d′/M-ratio.
- **Localisation.** NDCG@3 and log score over K = 5 candidates, paired vs O\*.
- **Power.** Variance components come from the pilot (≥ 200 predictions) and a simulation targets 80% power for Δ = SESOI. By Miller's formula this is expected to be 1,000–2,000 predictions per subject, about 2.5 CPU-hours per subject with log-probability readout and batched ground truth.

### D.5 Falsification criteria (pre-registered)

| Code | Condition | Conclusion |
|---|---|---|
| **F1** | The 90% CI of Δ(S − O\*) lies within ±SESOI (TOST) | *No causal self-knowledge beyond equally informed observers* at this granularity, for this model and task families |
| **F1′** | Δ < 0 with CI upper bound < 0 | *Self-report is worse than an informed outsider* (as Strangers found for behaviour) |
| F2 | Skill vs O₀ ≤ 0 | *No causal self-knowledge beyond base rates* |
| F3 | TOST on the instrument DiD | *Instruments give the self no advantage over an outsider* |
| F4 | TOST on the localisation difference | *No self-localisation* |

**Insensitivity guard:** if O₁ also fails to beat O₀, the targets are unpredictable from X at this granularity. The run is then reported as *uninformative*, not as negative.

### D.6 Interpretation

| Outcome | Meaning |
|---|---|
| S ≈ O\* ≈ O₀ | No usable causal self-knowledge; self-reports about internals are uninformative |
| S > O₀ but S ≈ O₂/O₃ | Generic knowledge of transformers, not self-knowledge |
| S > O₂ but S ≈ O₃ | Similarity or self-simulation, not self-specific |
| S (first-person) ≈ S (third-person) > O₂ | Weights-shared structural knowledge, no self-reference effect |
| S > all rungs, including O₅ | Candidate structural self-knowledge. Escalate to replication (other models, NDIF) and to CSP-2 |
| Large I (self gains more from instruments) | Instruments expand self-knowledge beyond what they give outsiders. This is the strongest possible support for the north star |

**Threats:**
- Small models may simply be at floor; the untrained floor in Li et al. was 5.02 EM. Mitigations are forced-choice readout, a balanced Y1 and an NDIF replication.
- Prompt sensitivity; mitigated by framings and paraphrases.
- Non-identifiability of localisation ground truth; mitigated by near-equivalent sets and graded NDCG.

### D.7 What comes after CSP-1

**CSP-2, occurrent access.** Hidden, randomised interventions at *content* positions, with graded strength, paired shams and impact-matched nulls. The model predicts the *consequence* for its upcoming answer, not whether "something happened". It is compared against a same-weights twin without the hidden edit, with an exact CRT for I(Z;R|X). The design fixes OWMI's scope bugs (position 0–2 only; the lifted arm).

**CSP-3, self-discovery.** The subject gets bounded tools (below) and a budget, and must produce committed claims about its own computation on a task family. An other-model agent gets identical tools as the control. Claims are scored by an independent executor against executed interventions, never by LLM-judged plausibility. It uses MAIA/AuditBench-style confabulation controls.

The tools are: inspect activations, compare runs, lens/J-lens, probe with mandatory control task, SAE features, `run_intervention` from the catalogue grammar only, and `log_hypothesis` (must precede confirmatory calls).

---

## E. Minimum architecture for CSP-1

Only what CSP-1 needs. Everything else is either delegated or not built.

```
packages/neuralscope/neuralscope/
  provider/
    base.py            Provider protocol: chat_tokenize, run(edit_sets, resume_depth) → logits,
                       option_logprobs(prompt, options), generate_greedy (generation_config neutralised)
    nnsight_provider.py  primary implementation (explicit attention masks to dodge the 0.7.0
                       multi-invoke bug; traces executed from files; .skip() resume)
    reference_hooks.py   = today's hf_decoder, frozen as parity oracle; deleted at end of M2
  interventions/
    spec.py            today's edit vocabulary + new types (resample-at-position-from-depth,
                       layer/MLP blocks, impact-matched random, input edit, SAE feature, decoy)
    catalog.py         stratified randomisation, natural-language descriptions, validator/bounds,
                       JSON-schema export (the only language a model may later propose in)
  tasks/               T1 (uses mathlab oracle), T2, T3 generators with deterministic answers
  selflab/
    groundtruth.py     execute interventions → Y1–Y5; sweeps; pairwise compensation; impact matching
    elicitation.py     templates, framings, paraphrases, option permutations, log-prob readout,
                       commitment hashing into the ledger
    observers.py       O0 constant, O1 learned (scikit-learn), O2/O3 LLM observers, O5 dossier
    instruments.py     DLA, lens/J-lens, attention summaries → dossier (observational only)
    scoring.py         log/Brier, skill, smECE, CORP, meta-d′, NDCG, DeLong, cluster bootstrap,
                       sign-flip permutation, TOST, BH, cross-fitted V-information + HRT
    claims.py          8 statuses, EvidenceItem, SelfKnowledgeClaim, admission rules, verdicts
    ledger.py          append-only SQLite: predictions (commit hash/time), outcomes, claims, provenance
    prereg.py          freeze + hash config; refuse test-split access before the freeze commit
    experiments/csp1.py  CLI orchestration → results JSON + markdown report + ledger claims
  mathlab/             kept (task oracle)
research/legacy/carry_heads/   E1/E2 scripts, data, README (research asset)
```

**Explicitly not built:** HTTP API, UI, activation trace store, SAE/AO/NLA training, generic lens or attention viewers, an agent loop (CSP-3 comes later), and the Construct Registry beyond schema and YAML (M8).

**External dependencies:**

| Status | Packages |
|---|---|
| New | `nnsight` (pin 0.7.0 with explicit-mask guards, or 0.8.0 once stable; `repetition_penalty` forced to 1.0), `scikit-learn`, `scipy` |
| Optional | `relplot` (smECE), `sae-lens` (Qwen3 SAE weights only; we do not import TransformerLens), Neuronpedia API (NLA instrument; 120 requests/hour) |
| Unchanged | `torch`, `transformers` 5.x, `pydantic`, `sympy`, `numpy` |

**Size.** Estimated new code ≈ 3k lines plus ≈ 1.5k lines of tests. After M1–M2 the Python codebase is ≈ 6k lines including Math Lab.

---

## F. Implementation plan (PR-sized milestones)

Each milestone is one PR on a branch off `claude/zen-knuth-hhtjxh` and is merged only when its acceptance criterion passes.

| M | Scope | Files changed | External deps | Tests | Scientific acceptance criterion |
|---|---|---|---|---|---|
| **M0** Archive and freeze | Tag `v1-neural-debugger`. Move E1/E2 to `research/legacy/carry_heads/` with a README (commands, expected tables, data hashes). Add a mission-pivot note to the README and link this plan. | `research/legacy/**`, `README.md`, `RESEARCH_LOG.md` | none | Existing suite green | Re-running E2 on the current code reproduces the published table exactly (35/40, 37/40, Δ within 1e-4). E1 JSON hash recorded. |
| **M1** Delete (option A or B) | Remove #7, 10, 12, 13, 17, 20, 21 (and #22 if B). Trim Makefile/setup. | `apps/web/**` (workspace), `apps/api/**`, `experiments/{service,store,jobs}.py`, `schemas/api.py`, tests, scripts | removes FastAPI/uvicorn (and Next.js if B) | Remaining suite green; no dangling imports (`python -c "import neuralscope..."`, ruff) | E2 reproduction still exact. LOC delta reported. |
| **M2** Provider adapter | `NNsightProvider` + `reference_hooks.py`. Parity suite (existing invariant tests parametrised over providers). Guard tests for the 0.7.0 mask bug and `generation_config` drift. Then delete `reference_hooks.py`. | `provider/**`, `tests/test_provider_*.py`, `research/legacy/carry_heads/*` (import path) | `nnsight` | Bit-exact parity on the DEMO model and Qwen2.5-0.5B: residuals, head ablation, patch, resample, batched sets, `.skip()`, greedy generation | **E2 reproduced through NNsight exactly.** The 336-head sweep runs within 1.2× of today's time. |
| **M3** Intervention language | Extended specs, catalogue sampler, validator/bounds, descriptions, JSON schema, decoys | `interventions/**`, tests | none | Validator rejects out-of-range and malformed specs. Each spec compiles to provider edits whose effect equals a hand-written reference. Descriptions round-trip. | Catalogue covers the pre-registered CSP-1 set. Impact-matching achieves JS divergence within ±10% of the target on pilot items. |
| **M4** Ground truth and tasks | T1–T3 generators; Y1–Y5; sweeps; pairwise compensation | `tasks/**`, `selflab/groundtruth.py`, tests | none | Null-effect and self-patch interventions give exactly zero Δ. Deterministic replays are identical. | **Known-positive anchor:** ablating L22.H10+H11 on T1 carry items reproduces E1 (tens accuracy 0.05 ± CI). Pilot Y1 flip rate in [0.3, 0.7] after stratification. |
| **M5** Elicitation, observers, ledger, claims | Templates/framings/permutations, log-prob readout, O₀–O₅, dossier, SQLite ledger with commitments, `SelfKnowledgeClaim` and admission rules | `selflab/{elicitation,observers,instruments,ledger,claims}.py`, tests | `scikit-learn` | Ground-truth access before commit raises. Option-permutation invariance of readout (tolerance). Every admission rule has positive and negative unit tests (e.g. self-report alone → UNTESTED). | On synthetic predictors with injected skill, the ledger and claim pipeline assigns SELF-PREDICTED + CAUSALLY VERIFIED only when the true skill exceeds the observer's. |
| **M6** Scoring and statistics | Proper scores, smECE, CORP, meta-d′, NDCG, DeLong, cluster bootstrap, sign-flip permutation, TOST, BH, V-information + HRT | `selflab/scoring.py`, `tests/test_scoring_*.py` | `scipy`, optional `relplot` | Unit tests against textbook values | **Estimator validation by simulation:** CI coverage within 93–97% of nominal 95%; type-I error ≤ α + 0.01 under the null; unbiased skill recovery across 1,000 simulated experiments with LLM-like variance components. |
| **M7** Pilot, pre-registration, CSP-1 | Pilot split, then power analysis, then freeze `docs/selflab/prereg/CSP-1.md` (hash committed), then the confirmatory run on Qwen2.5-1.5B and Qwen3-1.7B, then results, ledger claims and RESEARCH_LOG | `selflab/experiments/csp1.py`, `docs/selflab/prereg/**`, `docs/selflab/results/**` | NDIF key only if approved | End-to-end smoke on the DEMO model in CI | Git history shows the pre-registration committed before any test-split outcome exists. The report includes every pre-registered endpoint, nulls included, with statuses assigned by the admission rules, not by hand. |
| **M8** Constructs and communication | Construct Registry schema + validator + `C-0001 Plurience` YAML. Epistemic badge extended to 8 statuses. Website copy rewritten around verified claims (option A). | `constructs/**`, `apps/web/**` (if kept), docs | none | Validator tests. Badge tests for 8 statuses. | The registry rejects entries without rivals, discriminating tests and falsification conditions. |

**After M8, gated on CSP-1 results:**
- **CSP-2 (occurrent access):** hidden randomised interventions at content positions, twin observer, exact CRT.
- **CSP-3 (self-discovery):** bounded tool API, commit-before-confirm, identically tooled outsider agent.
- **Replication at scale** on NDIF (8B–70B) if approved.

---

## Phase 9. What NeuralScope already proved, preserved

| Result | Status | Evidence |
|---|---|---|
| **E1**: in Qwen2.5-1.5B-Instruct, heads L22.H10/H11 are *necessary* for the units→tens carry. Zero-ablation drops carry-problem tens accuracy 1.00 → 0.05 vs 0.93 on no-carry problems; same-layer and random controls ≈ 0; survives mean-ablation. | CAUSALLY TESTED · n = 80 | `docs/research/E1_carry_heads.json` |
| **E2**: the two heads' output at one position is *sufficient* to inject (35/40) or remove (37/40) a carry; control heads 0/40. | CAUSALLY TESTED · n = 40 pairs | `docs/research/E2_carry_patch.json` |
| "The Logit Lens reads the no-carry digit first" | DERIVED (34/40) | — |
| "Carry-routing heads" | INFERRED | Label only; supported by the two results above |

**How they are preserved.** They move to `research/legacy/carry_heads/` unchanged in content.

**What they become:**
1. The acceptance test of the NNsight migration (M2).
2. The known-positive localisation anchor for CSP-1 (M4).
3. A concrete self-localisation question with a known answer. *Does Qwen2.5-1.5B, asked where its carry is computed, point anywhere near L22.H10/H11 and the L21–L24 hand-off?*

Math Lab no longer drives the roadmap. Its oracle survives only as a task oracle.

---

## Risks, and what a negative programme looks like

- **Most likely CSP-1 outcome:** F1/F2 hold for 1.5–1.7B models. SelfLab then publishes a carefully controlled null for zero-shot internal causal self-knowledge at small scale, and the question moves to scale (NDIF) and to instrument leverage (I). The system is built to make exactly that conclusion credible.
- **Measurement risk:** the forced-choice readout may favour models with better instruction following, which inflates *all* LLM rungs equally. Paired comparisons within item control for this.
- **Scope risk:** the temptation to add viewers or infrastructure. The rule is that anything not in §E needs a written justification that no external tool supplies it.
- **Novelty risk:** a fast-moving field. Re-run the prior-art searches at M7 and before any external claim.
