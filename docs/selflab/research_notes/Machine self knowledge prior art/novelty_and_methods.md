# Machine self-knowledge prior art: novelty assessment and methods dossier for SelfLab (NeuralScope)

**Conventions.** Literature and code were checked up to 2026-09-30. Every bullet has one of three labels:
- **(paper)**: stated in the cited source.
- **(code-verified)**: confirmed by reading the released code.
- **(inference)**: my own reasoning from the cited material.

On 2026-09-30 I re-resolved every arXiv ID against the arXiv API and every DOI against Crossref. The framing throughout is privileged self-access, causal self-knowledge, calibrated self-description and metacognitive access. Where a source title uses other vocabulary, it is quoted only as a title.

**Notation (inference).**
- **X**: what an external observer can see (the prompt, the visible output and the intervention specification).
- **Z**: the ground-truth internal fact. This can be the causally relevant component, the effect of an intervention, the compensating component, or where decision-critical information sits.
- **R**: the model's self-report or self-prediction.
- **O**: an observer's prediction.

## 1. NOVELTY: For each SelfLab capability (a)–(f), what is the closest prior work, what did it do, and is the capability already done, partially done, or apparently open?

### Takeaway
No single SelfLab capability is entirely new, but the core combination is open.

Already done:
- **Trained self-prediction of patching and ablation outcomes.** Li et al. 2511.08579 did this at layer-block granularity, and the model explained itself better than other models did.
- **Post-hoc reporting of interventions with sham and observer controls.** OWMI 2608.20569 did this and found a clean null on 8 open models.
- **Behavioural self-prediction against observers.** Zeng 2608.30980, Blandfort 2609.09899 and Song 2503.07513 did this and found no self-specific advantage. Binder 2410.13787 is a small exception, and only after training.

Apparently open:
- zero-shot, component-level, graded prediction made before the intervention, compared against a ladder of equally informed observers;
- predicting which component compensates;
- layer- or component-level self-localisation scored against sweeps;
- calibration of reports about internal causal structure;
- a formal estimate of I(Z;R|X) with valid inference;
- an LLM that uses interpretability tools on itself;
- the term "Plurience".

### Cited Findings

#### (a)(i)–(ii) Trained prediction of which internals matter and what an intervention will do
- **Li, Guo, Huang, Steinhardt, Andreas et al., "Training Language Models to Explain Their Own Computations"** (v1 Nov 2025, v3 Feb 2026). They fine-tune explainer models on three tasks:
  1. describing SAE features;
  2. predicting the effect of activation patching;
  3. predicting the effect of input ablation.

  Ground truth comes from existing interpretability tools. The abstract says self-explanation "generally works better than using a *different* model to explain its computations (even if the explainer model is significantly more capable than the target)". Code is at github.com/TransluceAI/introspective-interp. (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Li et al. task design.**
  - Activation-patching task: counterfactual prompt pairs from CounterFact. A patch is applied at one of four layer blocks at one token. The explainer predicts whether the output changes and what the new content is. The patched vector is passed to the explainer as a continuous token.
  - Input-ablation task: MMLU items carrying a "Hint: X". The explainer predicts whether removing the hint changes the answer.

  (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Li et al. results** (Exact Match / Has-Changed F1 / Content Match). Untrained means the prompted base model. (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)

  | Target and task | Self-explainer | Other-model explainer | Untrained |
  |---|---|---|---|
  | Qwen3-8B, patching | 64.0 / 80.2 / 71.0 | Llama-3.1-8B: 54.1 / 73.9 / 65.4 | 5.02 / 24.4 / 18.8 |
  | Qwen3-8B, input ablation | 83.4 / 87.0 / 90.6 | Llama: 58.1 / 70.5 / 71.6 | 8.9 / 44.4 / 35.3 |
  | Llama-3.1-8B, patching | 48.6 / 77.5 / 54.6 | Qwen: 41.7 / 75.1 / 47.6 | 3.2 / 31.4 / 7.0 |
- **Li et al. appendix ablation** (Qwen target, patching). Removing the patched activation from the explainer's input still leaves 59.9 / 76.5 / 68.6, against 64.0 / 80.2 / 71.0 with it. Removing the layer or token identity has a negligible effect. A trained decoder recovers layer and token from the activation plus context at 98.7% exact match. (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Li et al. caveats.**
  - The "self" explainer is a fine-tuned copy of the target, and the paper notes this is not strict self-explanation.
  - Non-self explainers receive the target's activations through a learned projection.
  - Explainer–target activation similarity tracks explanation quality. Table 4 gives 96.6% dot-product similarity for Llama-3-8B against 0.10% for Qwen3-8B.
  - Self-explanation is reported as data-efficient, working with 0.8% of the training data.

  (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Li et al. calibration.** In the sections I read, I found no calibration analysis of the explainers' predictions. (inference; see Gaps) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Zeng, Assis & Wang, "Evaluating and Improving LLM Self-Modeling"** (31 Aug 2026).
  - The benchmark covers verifiable behavioural self-modelling questions, for example "whether a prompt edit would change the model's final answer".
  - Task types: FLIP-DECISION, FLIP-RATE, SELF-ACCURACY, PERTURBATION-CHOICE, COMPONENT-ATTRIBUTION, FEATURE-RATE, OUTPUT-PREDICTION, EDIT-PROPOSAL, CONFIDENCE-RECALL and LOGIT-ESTIMATION, with normalised skill scores.
  - GPT-5.5, Claude Opus 4.7 and Gemini 3.1 Pro fail a simple GSM8K answer-flip example.
  - RL on synthetic data improves aggregate skill for Llama-3.1-8B, Qwen3-8B and GPT-OSS-20B, with some held-out transfer.
  - Cross-model transfer shows no consistent own-model advantage. Qwen explains Llama better than Llama explains itself. An own-model advantage appears only for the Qwen3-8B/GPT-OSS-20B pair on FLIP-RATE.
  - The abstract says the improvement "may not arise from privileged access to the model's internal decision process".

  (paper) — [Zeng et al. 2608.30980](https://arxiv.org/abs/2608.30980)
- **Guo, Ruis, Andreas, Li, "Introspective Coupling"** (30 Jun 2026).
  - Some models are trained on fixed counterfactual (cue-removal) explanations taken from earlier checkpoints of themselves, "or even from behaviorally similar models in different families". These models frequently give explanations that are more faithful to their own current behaviour than to their training targets.
  - Tasks: Hint-MMLU, AITA sycophancy and refusal.
  - Under patching, behaviour logits and explanation logits co-move: r = 0.89 with behaviour regularisation, against 0.527 without.
  - With Llama-label mixing, the Self > Orig ordering holds at every mixing weight, while Self exact match falls from 88% to 69%.
  - The coupling tracks newly acquired behaviours.

  (paper) — [Guo et al. 2606.32038](https://arxiv.org/abs/2606.32038)
- **Plunkett et al.** GPT-4o and GPT-4o-mini were fine-tuned to choose according to randomly generated attribute weights across many decision contexts (100 in my notes). The models report those weights at r = .54 and .50. After introspection training this rises to .74 and .75. The training transfers to reporting native preferences: .46 → .71 and .40 → .70. (paper) — [Plunkett et al. 2505.17120](https://arxiv.org/abs/2505.17120)

#### (a) Behavioural self-prediction against cross-model and observer baselines
- **Binder et al., "Looking Inward"** (ICLR 2025).
  - GPT-4o, GPT-4, GPT-3.5 and Llama-3-70B were fine-tuned on about 30k self-prediction samples about properties of their hypothetical responses. Untrained models were barely above the modal baseline.
  - Cross-prediction baseline: M2 is trained on M1's behaviour. Llama-3-70B predicts itself at 48.5%, against 31.8% for GPT-4o predicting Llama. GPT-4o predicts itself at 49.4%, against 36.6% for Llama predicting GPT-4o. GPT-3.5's self-advantage is only +0.8 points.
  - Training raises GPT-4o's self-prediction from 32.6% to 49.4%.
  - Self-predictions are better calibrated. Mean absolute deviation for Llama's "second character" property: self 8.7, cross 21.3, untrained 23.0.
  - After further fine-tuning changes its behaviour, the model predicts its new behaviour better than its old one (35.4% vs 21.7%).
  - Failures: long outputs, and a sycophancy-related task (self 74.5% vs cross 76.5%).
  - The paper discusses self-simulation as a possible mechanism.

  (paper) — [Binder et al. 2410.13787](https://arxiv.org/abs/2410.13787)
- **Blandfort & Pawar, "Strangers to Themselves"** (9 Sep 2026). Nine behavioural evaluations, 12 models.

  Correlations with actual behaviour:

  | Predictor | r |
  |---|---|
  | Direct self-report | +0.04 |
  | Item-informed self-prediction | +0.24 |
  | Same question about "capable AI agents in general" | +0.28 |
  | Other models' self-answers | +0.35 |
  | Self-specific advantage | +0.02 |

  - An analyst given the same behavioural history outperforms the model on its own history: +0.28 vs +0.19, paired difference −0.10 [−0.19, −0.02].
  - In a reader × subject matrix the self-gain is −0.00.
  - First-person framing adds a flattering bias (−0.15).
  - The paper identifies an "answerer signature" (procedural leakage).
  - Fine-tuning on a model's own record teaches narrow self-predictions but changes the behaviour being predicted.
  - The study was pre-registered with a frozen test split.

  (paper) — [Blandfort & Pawar 2609.09899](https://arxiv.org/abs/2609.09899)
- **Song, Hu & Mahowald** (COLM 2025). 21 open models, tested on grammaticality (BLiMP, Linguistic Inquiry) and word prediction. They regress one model's metalinguistic judgements on another model's direct probabilities, controlling for model similarity, and find no same-model advantage. Agreement between meta and direct judgements is about κ = 0.25. (paper) — [Song et al. 2503.07513](https://arxiv.org/abs/2503.07513)
- **Song, Lederman, Hu & Mahowald.**
  - They define introspection as a process that yields information about internal states more reliably than any third-party process of equal or lower computational cost.
  - LLM temperature self-reports track prompt framing ("crazy" vs "factual") rather than the true temperature.
  - Self-reflection, within-model prediction and across-model prediction all score near 0.5.

  (paper) — [Song et al. 2508.14802](https://arxiv.org/abs/2508.14802)
- **Comsa & Shanahan.** An LLM correctly inferring its own sampling temperature can count as a minimal example of introspection. An LLM describing the process behind its own creative writing does not. (paper) — [Comsa & Shanahan 2506.05068](https://arxiv.org/abs/2506.05068)
- **Kadavath et al.** Models' P(True) self-evaluations and P(IK) ("I know") predictions are reasonably calibrated in suitable formats. (paper) — [Kadavath et al. 2207.05221](https://arxiv.org/abs/2207.05221)
- **Betley et al.** Models fine-tuned on implicit behaviours, such as insecure code or risky choices, can describe those behaviours without being trained to. (paper) — [Betley et al. 2501.11120](https://arxiv.org/abs/2501.11120)
- **Faithfulness of self-explanations.** These give a baseline expectation that self-explanations are often unfaithful. (paper)
  - Counterfactual simulatability: explanations often do not let an observer predict counterfactual behaviour. — [Chen et al. 2307.08678](https://arxiv.org/abs/2307.08678)
  - Faithfulness varies by model and task. — [Madsen et al. 2401.07927](https://arxiv.org/abs/2401.07927)
  - Faithfulness and plausibility are in tension. — [Agarwal et al. 2402.04614](https://arxiv.org/abs/2402.04614)
  - Chain-of-thought omits biasing features. — [Turpin et al. 2305.04388](https://arxiv.org/abs/2305.04388)
  - Reasoning models often do not verbalise hints they use. — [Chen et al. 2505.05410](https://arxiv.org/abs/2505.05410)

#### (a)/(b) Post-hoc reports of interventions: detection, identification, localisation
- **OWMI (Ferrara, 20 Aug 2026).**
  - Interventions: residual-stream sites, attention heads, Gemma Scope SAE features and block outputs, using zero, scale, noise and replace operators.
  - Controls: sham runs, impact-matched random perturbations, and a text-only observer.
  - Scale: 8 open models from 7 families (0.5B–15B), 12 benchmarks, more than 78,000 measurements.
  - Main result: pooled AUROC ≈ 0.5007. An equivalence test bounds the effect below 0.15 percentage points of AUROC.
  - Known positive: a LoRA fine-tuned to report this class of intervention reaches near-perfect held-out recovery (d′ = 5.15, AUROC ≈ 1.0; 400 held-in and 100 held-out directions).
  - Linear probes recover intervention presence at 75%–95.8% held-out accuracy, with no held-out error at the last layer. The label-shuffled null used 200 refits.
  - Qwen2.5-7B's yes/no report never varies, but its confidence separates intervention from sham at AUROC 0.647.

  (paper) — [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **OWMI localisation and limitations.**
  - Of 1,184 intervention trials, 967 answered that no layer could be named and only 88 gave a numeric guess. The paper treats this as refusal, not measurement.
  - Breadth battery (19,520 measurements, 3 models): detection d′ = 0.022 [0.000, 0.046].
  - Stated limitations: reconstruction was not scored, Track C produced no pairs, models were ≤15B, and the max-over-observers bound was not computed.

  (paper) — [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **OWMI code: probe wording.** The localisation probe reads "If you think your computation was altered, guess which layer was changed" and returns JSON keys `layer_guess`, `confidence` and `reason`. The self-probe header gives the benchmark, the target object kind and, for every family except characterisation and reconstruction, the intervention mode. The observer query gives only the benchmark name and the visible output, plus "Using only this visible output, do not infer hidden activations." (code-verified) — [owmi/benchmarks/probe_bank.py](https://huggingface.co/emilioferrara/owmi/blob/main/owmi/benchmarks/probe_bank.py)
- **OWMI code: context per track.** In Track A the self-probe is appended to the task prompt, and the model's own task output is not in the probe context. Tracks B and C use a follow-up turn that contains the prompt and the model's output. The text-only observer runs on the same backend model unless a stronger observer model is configured. (code-verified) — [owmi/benchmarks/runner.py](https://huggingface.co/emilioferrara/owmi/blob/main/owmi/benchmarks/runner.py)
- **OWMI code: observer bound.** The release implements a bag-of-words logistic "classifier observer". It also implements `compute_observer_bound`, a max over the text-only, stronger-model and classifier observers on a common balanced-accuracy-over-chance scale. The module docstring calls the classifier "the third observer type the paper commits to but never implements". (code-verified) — [owmi/observer.py](https://huggingface.co/emilioferrara/owmi/blob/main/owmi/observer.py)
- **Lindsey (Anthropic), concept injection.**
  - Claude Opus 4 and 4.1 detect and identify injected concepts in about 20% of trials, at the best layer (about two-thirds depth) and strength 2–4.
  - Production models gave 0 false positives in 100 control trials.
  - Further experiments cover distinguishing injected representations from text, disavowing prefilled outputs, and intentional control.
  - Base models show no net performance, and helpful-only variants have higher false-positive rates.
  - Criteria used: accuracy, grounding, internality and metacognitive representation.

  (paper) — [Lindsey 2601.01828](https://arxiv.org/abs/2601.01828); [transformer-circuits post](https://transformer-circuits.pub/2025/introspection/index.html)
- **Macar et al., "Mechanisms of Introspective Awareness".** Models: Gemma3-27B, Qwen3-235B and OLMo. Code is at safety-research/introspection-mechanisms.
  - 0% false positives in their setup.
  - The capability emerges after DPO, not SFT.
  - Mechanism: two stages, with evidence-carrier features feeding "gate" features.
  - Ablating the layer-45 MLP drops detection from 39.0% to 24.2%, and no single attention head is necessary.
  - Ablating the refusal direction raises detection from 10.8% to 63.8%, with false positives rising from 0% to 7.3%.
  - A trained bias vector adds 75% detection on held-out concepts.

  (paper) — [Macar et al. 2603.21396](https://arxiv.org/abs/2603.21396)
- **Zou, Sun, Kong, Wang** (28 Sep 2026).
  - Design: fixed input text, with a concept vector injected at one of ten token positions or no intervention. The model names the position or reports none, scored on first-token logits with no judge.
  - Mean localisation: Qwen3-4B 54.49%, LLaMA-3.1-8B 44.31%, Gemma-3-12B 65.95% (chance is 10%).
  - Controls: norm-matched random directions give 0.84% / 6.71% / 0%; lexical-replacement controls give 0.87–11.13%.
  - Mechanism: middle-layer gate heads plus later router heads.

  (paper) — [Zou et al. 2609.35108](https://arxiv.org/abs/2609.35108)
- **Hahami, Sinha & Jain, "Introspection Fine-Tuning (IFT)".**
  - Yes/no injection detection is confounded: it correlates r = 0.999 with a factual-"no" control in Llama-3.1-8B.
  - They add sentence-localisation and strength-comparison tasks.
  - IFT lifts Llama-1B localisation from 9.6% to 60.6%, and zero-shot strength comparison from 30.2% to 52.2%.
  - Training at random layers is key. Gaussian vectors do not generalise.

  (paper) — [Hahami et al. 2607.14111](https://arxiv.org/abs/2607.14111); earlier work by the same group: [2512.12411](https://arxiv.org/abs/2512.12411) (title-level only)
- **Rivera & Africa, "Steering Awareness".**
  - Seven fine-tuned models. The best reaches 95.5% detection and 71.2% identification with 0 false positives on held-out concepts.
  - Detection generalises to unseen ways of building steering vectors only when they have high cosine similarity to the training vectors (a "geometric detector").
  - Detection training makes models more steerable.

  (paper) — [Rivera & Africa 2511.21399](https://arxiv.org/abs/2511.21399)
- **Lederman & Mahowald** (v2 title "Emergent Introspection in AI is Content-Agnostic"). Qwen3-235B and Llama-3.1-405B, 821 concepts.
  - Detection is content-agnostic.
  - "Apple" accounts for 74.8% of Qwen's wrong identifications.
  - Identification needs continued steering.
  - An earlier direct-access claim was withdrawn because of a prompt-length confound.

  (paper) — [Lederman & Mahowald 2603.05414](https://arxiv.org/abs/2603.05414)
- **Pearson-Vogel et al.** Qwen 32B denies injections in sampled outputs, but the logit lens shows detection signals in the residual stream that weaken in the final layers. Giving the model accurate information about introspection mechanisms raises sensitivity from 0.3% to 39.9%, with only 0.6% more false positives. (paper) — [Pearson-Vogel et al. 2602.20031](https://arxiv.org/abs/2602.20031)
- **Singh, Linzen & Ravfogel** (COLM 2026).
  - Introspection requires two conditions: privileged access and second-order computation.
  - Ji-An-style neurofeedback labels are predictable from layer-0 input features, and probes trained on permuted labels collapse to near chance.
  - Under "gaslight" prompts, Llama-3.1-70B and Qwen-3-32B label prompt manipulations as injections, so they fail a three-way discrimination.
  - They recommend converging mechanistic evidence.

  (paper) — [Singh et al. 2605.26242](https://arxiv.org/abs/2605.26242)
- **Ji-An et al.** (NeurIPS 2025). In-context neurofeedback: labels come from projecting activations onto logistic-regression or PCA axes. Reporting and control depend on the number of examples, on semantic interpretability (LR axes beat PC axes) and on variance explained. The resulting "metacognitive space" is low-dimensional. (paper) — [Ji-An et al. 2505.13763](https://arxiv.org/abs/2505.13763)
- **Aoki et al., "In-Context Neurofeedback"** (1 Sep 2026).
  - A fixed-sentence design guarantees that the target is privileged.
  - Models: Llama-3.1-8B/70B and Qwen3-8B/32B. Tasks: SST, ETHICS and True-False. 5 layers, 50 turns.
  - 45 of 120 tests are significant after BH correction, mostly SST and Qwen3-8B. Every effect is small (d < 0.5, h < 0.5).

  (paper) — [Aoki et al. 2609.00904](https://arxiv.org/abs/2609.00904)
- **Li, Ceballos Arroyo, Rogers, Saphra, Wallace** (ICML 2026). A zero-shot baseline that sees only the input text matches activation-verbalisation methods on earlier benchmarks. On PersonaQA-Fantasy, a fictional-persona dataset whose facts only the target model has, the verbalisers fail (reported score 0). This suggests their outputs reflect the verbaliser's own knowledge. (paper) — [Li et al. 2509.13316](https://arxiv.org/abs/2509.13316)

#### (a)(iii) Compensation and self-repair: ground truth, but nothing on self-prediction
- **Hydra effect.** When one attention layer is ablated, other layers compensate. (paper) — [McGrath et al. 2307.15771](https://arxiv.org/abs/2307.15771)
- **Rushing & Nanda.** Self-repair happens when individual attention heads are ablated. It is imperfect and noisy. The mechanisms include final-LayerNorm rescaling and sparse anti-erasure neurons. (paper) — [Rushing & Nanda 2402.15390](https://arxiv.org/abs/2402.15390)
- **IOI circuit.** "Backup name mover" heads take over when the primary name movers are ablated. (paper) — [Wang et al. 2211.00593](https://arxiv.org/abs/2211.00593)
- **COAX (Gong et al., "Conditional Co-Ablation").** Candidates are ranked by how much their ablation effect grows once the primary set is removed.
  - It recovers the documented IOI backup heads at 0.941 ROC-AUC, against 0.815 for the strongest intact-state attribution baseline and 0.758 for a matched conditional-energy control.
  - Alternative sets matched on behaviour score 0.40 ± 0.13.
  - Completing the circuit cuts incompleteness from 0.75 to 0.21.
  - Conditional growth matches repair measured by intervention in 11 of 12 held-out instances.
  - COAX beats matched random completions on all 8 non-GPT-2 models, across 6 architecture families.

  (paper) — [Gong et al. 2607.01940](https://arxiv.org/abs/2607.01940)

#### (a)(iv)/(c) Calibration of self-reports
- **Existing calibration evidence.** (paper)
  - Binder: self-predictions have lower mean absolute deviation (above). — [Binder et al. 2410.13787](https://arxiv.org/abs/2410.13787)
  - OWMI: the confidence family is scored with equal-mass ECE and selective prediction, with confidence AUROC 0.647 (above). — [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
  - Verbalised confidence can be taught. — [Lin et al. 2205.14334](https://arxiv.org/abs/2205.14334)
  - RLHF models' verbalised confidences are often better calibrated than their token probabilities. — [Tian et al. 2305.14975](https://arxiv.org/abs/2305.14975)
- **Review.** Steyvers & Peters review metacognition and uncertainty communication in humans and LLMs, separating metacognitive sensitivity from calibration. (paper) — [Steyvers & Peters 2504.14045](https://arxiv.org/abs/2504.14045)

#### (e) Self-reading, self-experimentation and self-modelling systems
- **Introspection adapters (Shenoy et al.).** One LoRA is trained, with SFT and then DPO, across fine-tunes of Llama-3.3-70B that carry implanted behaviours.
  - It generalises to AuditBench at state of the art: 59%, against 53% for the next best and 44% for the best white-box tool.
  - It detects encrypted fine-tuning API attacks.
  - It scales with model size and data diversity (Qwen series 37.7% → 77.3%).
  - In ablation, the DPO stage lifts AuditBench from 34.1% to 53.6%.
  - It reports high false-positive rates and hallucinated behaviours.
  - It builds on weight-diff interpretation. — [Goel et al. 2510.05092](https://arxiv.org/abs/2510.05092)
  - An attack on it has already been published. — [Merrill & Medley 2605.27836](https://arxiv.org/abs/2605.27836)

  (paper) — [Shenoy et al. 2604.16812](https://arxiv.org/abs/2604.16812); [Anthropic post](https://alignment.anthropic.com/2026/introspection-adapters/)
- **Activation Oracles.** These are LatentQA-trained models that answer natural-language questions about activations; in Karvonen et al. the oracle is trained from the same base model it reads.
  - They recover fine-tuned information that does not appear in the input text.
  - The best oracles match or beat white-box baselines on all four tasks, and beat the best overall baseline on 3 of 4.
  - My notes record the authors saying the outputs are not calibrated.

  Related methods: [LatentQA 2412.08686](https://arxiv.org/abs/2412.08686); [Patchscopes 2401.06102](https://arxiv.org/abs/2401.06102); [SelfIE 2403.10949](https://arxiv.org/abs/2403.10949). (paper) — [Karvonen et al. 2512.15674](https://arxiv.org/abs/2512.15674)
- **Predictive Concept Decoders (Transluce).** An encoder compresses activations into sparse top-k concepts, and a decoder answers questions about them. Case studies include detecting jailbreaks and hint use. (paper) — [Huang et al. 2512.15712](https://arxiv.org/abs/2512.15712)
- **Self-Interventional Learning (Tomaszewski, 14 Aug 2026).** Networks perturb their own components and learn a predictive self-model of the consequences.
  - Settings: a 24-component synthetic system, Fashion-MNIST, and CIFAR-10 with a ResNet.
  - It recovers redundancy and replaceability, but not synergy.
  - Raising the budget from 4 to 56 pairs moves MAE from 0.0335 to 0.0148 and Spearman from 0.629 to 0.883, and calibration improves.
  - Acting on the self-model cuts regret by 31.7%, but does not beat empirical memory.
  - Not LLM-scale. Pre-registered.

  (paper) — [Tomaszewski 2608.14894](https://arxiv.org/abs/2608.14894)
- **Premakumar et al.** An auxiliary task of predicting one's own hidden activations makes networks simpler (narrower weight distributions, lower RLCT) on MNIST, CIFAR-10 and IMDB. (paper) — [Premakumar et al. 2407.10188](https://arxiv.org/abs/2407.10188)
- **Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models"** (transformer-circuits, 10 Jul 2026; title as published). A Jacobian-based lens ("J-lens") identifies a subspace of verbalizable representations ("J-space"). The paper studies J-space ablations and gives an auditing agent the lens as a tool. (paper) — [Gurnee et al. 2026](https://transformer-circuits.pub/2026/workspace/index.html)

### Inferences
- **(a)(i) Predicting which components or representations are causally relevant: PARTIALLY DONE.** Li et al. do this with trained explainers, for residual-stream patches at 4 layer blocks, predicting binary change plus new content. Zeng et al. include COMPONENT-ATTRIBUTION and LOGIT-ESTIMATION task types. What remains (inference):
  1. untrained, zero-shot prediction (Li's prompted baseline is near the floor);
  2. component granularity: attention heads, MLPs, SAE or transcoder features;
  3. graded targets such as Δlogit or KL rather than binary change;
  4. calibrated confidence;
  5. observers matched on information and cost.
- **(a)(ii) Predicting the effect of suppressing a component: PARTIALLY DONE.** Li et al. cover residual patching and input-hint ablation, both trained. OWMI applies zero-ablation to heads and SAE features, but only asks for post-hoc reports, never a prediction. I found no study where a model predicts, before the intervention, the graded effect of ablating a named head, MLP or feature on its own output, scored against executed ablations. (inference)
- **(a)(iii) Predicting which component compensates: APPARENTLY OPEN.** Good ground truth exists: COAX, IOI backup heads, Hydra, Rushing & Nanda. Web (Exa) and arXiv searches combining "self-repair", "backup heads" or "compensation" with "self-prediction", "self-report", "introspection" or "model predicts ablation effect" found no study where a model predicts its own compensation. (inference)
- **(a)(iv) Confidence: PARTIALLY DONE.** Calibration exists for behavioural self-prediction (Binder, mean absolute deviation), for post-hoc detection confidence (OWMI, ECE and AUROC 0.647), and for answer knowledge (Kadavath). What remains is calibration and metacognitive sensitivity (meta-d′) of confidence about predicted internal causal effects, compared against an observer's confidence. (inference)
- **(a) Central question, "does self-prediction beat an equally informed observer?": PARTIALLY DONE, and the evidence leans null.** (inference)
  - Self-advantages appear only after training (Binder, Li, Guo).
  - Where observers are well matched, the advantage vanishes: Zeng (cross-model), Blandfort (analyst and generic-agent controls), Song (similarity-controlled regression), OWMI (sham plus observer).
  - Li's self-advantage has three confounds:
    - most accuracy survives removing the activation (EM 64.0 → 59.9);
    - activation-space similarity predicts explainer quality;
    - the other-model explainer reads the target only through a learned projection.

    Together these point to self-simulation and representational familiarity, not access to the live state.
- **Conceptual point for (a) (inference).** A prediction made before the intervention cannot draw on the altered live state, because the intervention has not happened yet. Any self-advantage there must come from shared weights, meaning self-simulation, which Binder discusses. SelfLab should say so explicitly. To test live-state access it should pair predictions with post-intervention reports: predict, intervene, report.

  The observer ladder for (a) should include:
  1. the same weights under a third-person framing;
  2. a different model with identical X;
  3. a stronger model;
  4. a classifier trained on X and component metadata (layer index, type, output norm);
  5. a black-box observer allowed to query the unmodified model;
  6. a white-box observer with activations but different weights, as in Li et al.

  This follows Song et al.'s cost-matched definition. OWMI's observer is not matched on information. The self-probe gets the task prompt (Track A), the object kind and the intervention mode. The observer gets only the visible output (code-verified). So SelfLab should give the observer exactly the model's external context X.
- **(b) Self-localisation.**
  - Token-position localisation of injected vectors: DONE (Zou; IFT).
  - Layer or component localisation of decision-critical information, scored against patching and ablation sweeps: APPARENTLY OPEN.
  - OWMI's layer probe produced mostly refusals (967 of 1,184).
  - Li's decoder shows the layer is decodable from the vector, not reported by the model.
  - Searches: an Exa search for LLM self-localisation of layers or components returned only Macar et al., the IOI and path-patching literature, an unrelated retrieval-heads paper, and Hahami 2512.12411. None scores a model's own layer or component claims against sweeps.
  - Caution: when nothing is intervened on, Z is a deterministic function of X. The Shannon quantity I(Z;R|X) is then 0, so (b) must be framed as computationally bounded (V-information) privileged access. See Section 3.

  (inference)
- **(c) Calibration of self-reports: PARTIALLY DONE.** It exists for answers, behaviours and detection confidence. It is open for claims about causal structure, localisation and compensation, and for type-2 sensitivity compared against observers. (inference)
- **(d) I(Z;R|X): the ESTIMATORS EXIST (Section 3), the APPLICATION IS APPARENTLY OPEN.** Closest operationalisations: OWMI's observer margin in AUROC/d′ plus a probe ceiling, Blandfort's paired correlation differences and reader × subject matrix, Song's similarity-controlled regression, and Binder's cross-prediction. None uses a conditional-information estimator with valid inference such as a conditional randomisation test. I found none in the literature reviewed here; this is negative evidence only. (inference)
- **(e) Tool-assisted self-discovery: APPARENTLY OPEN for LLMs.** Details are in Section 2. Every tool-using interpretability agent found investigates another model. The self-reading systems (Activation Oracles, PCD, introspection adapters, Li et al.'s explainers) are fine-tuned readers, not agents running experiments on themselves. SIL does self-experimentation, but only in small non-LLM networks. (inference)
- **(f) Construct Registry: PARTIALLY DONE in pieces; "Plurience" UNCLAIMED.** Details are in Section 4. (inference)

### Gaps
- **Zeng et al.'s COMPONENT-ATTRIBUTION and LOGIT-ESTIMATION tasks.** I did not read their exact definitions (granularity, zero-shot vs trained). If they already ask models to predict the effect of ablating named internal components, gap (a)(i)–(ii) narrows to "graded, calibrated, observer-matched". Check this before claiming novelty.
- **Li et al.** I found no calibration analysis in the sections I read. I did not verify whether the appendices contain head, MLP or feature-level experiments. The code repository was not cloned.
- **Every absence claim is negative evidence from searches up to 2026-09-30.** At least five directly relevant papers appeared in Aug–Sep 2026 (OWMI, Zeng, Blandfort, Aoki, Zou). Repeat the searches immediately before any publication. Unpublished industry work cannot be searched.
- **Unextracted details.** I did not extract quantitative results from Gurnee et al. 2026 or Lindsey's secondary experiments. Hahami et al. 2512.12411 was verified at title level only.

## 2. AGENTS: Which AI systems run interpretability experiments with tools (especially on themselves), with what tool APIs, and how do they control confabulation and verify claims?

### Takeaway
Tool-using interpretability agents exist and have been benchmarked, including MAIA, Anthropic's auditing agents, AuditBench, InterpAgent, HARP, HyVE and SAEScientist-Bench. Every one investigates a different model. I found no LLM agent that runs interpretability tools on its own weights or activations.

The literature has three main lessons for SelfLab's self-discovery arm:
- **A tool-to-agent gap.** Tools that work in isolation are under-used or misread by agents.
- **Confirmation bias and weak validation loops.**
- **The need for verification grounded in execution and held-out ground truth.**

### Cited Findings

#### Interpretability agents and their tool APIs
- **MAIA** (ICML 2024). A GPT-4V agent with a System/Tools API.
  - Tools: `dataset_exemplars`, `text2image`, `edit_images`, `describe_images` and `summarize_images`, where description runs on a separate GPT-4V instance to limit confirmation bias, plus `log_experiment`.
  - Evaluation is predictive, on real and synthetic neurons. MAIA's descriptions beat MILAN and were comparable to human experts (reported values 0.73 and 0.53).
  - Failure modes: confirmation bias and drawing conclusions from small samples.

  (paper) — [Rott Shaham et al. 2404.14394](https://arxiv.org/abs/2404.14394)
- **FIND** gives interpretability agents functions with known ground truth, used to evaluate automated-interpretability agents. (paper) — [Schwettmann et al. 2309.03886](https://arxiv.org/abs/2309.03886)
- **Anthropic, "Building and evaluating alignment auditing agents"** (Jul 2025).
  - Tools: chat, raw sampling, grep and semantic search over data, SAE feature tools, and steering.
  - The investigator agent solves the Marks et al. auditing game 13% of the time under realistic conditions. Interpretability tools help (+10–20 points), and a "super-agent" that aggregates parallel runs does better.
  - An evaluation-building agent succeeds in 88% of runs.
  - Failure modes: fixating on early hypotheses, memory problems, overconfidence, and not paging through results.

  (paper) — [Anthropic alignment blog](https://alignment.anthropic.com/2025/automated-auditing/); game: [Marks et al. 2503.10965](https://arxiv.org/abs/2503.10965)
- **AuditBench.** 56 Llama-3.3-70B models implanted with 14 behaviours.
  - 13 tool configurations: steering, SAE features (with LLM coherence filtering), logit lens, activation oracle, prefill, user sampling, text completion, Petri, and scaffolded variants.
  - Scaffolded black-box tools do best.
  - The "tool-to-agent gap" has three causes: agents under-use tools, are distracted by noisy outputs, and fail to turn evidence into hypotheses.
  - Scoring: an LLM classifier grades 10 predictions per run, 16 runs per condition.

  (paper) — [Sheshadri et al. 2602.22755](https://arxiv.org/abs/2602.22755)
- **InterpAgent.** Uses FeatureFinder and FeatureExplainer on Gemma-2, with optional steering and auditable traces.
  - Hypotheses are checked with a seven-metric battery: detection F1, fuzzing F1, surprisal AUROC, embedding similarity, an LLM judge, a t-test p-value and Cohen's d.
  - Refined hypotheses beat the initial ones 63–83% of the time.

  (paper) — [Marin-Llobet & Ferrando 2605.01555](https://arxiv.org/abs/2605.01555)
- **SAEScientist-Bench.** Gemma-2-9B-IT with 131K Gemma Scope features, 20 tasks and 10 agents.
  - Best composite: Kimi K3 at 65.8, against 85.6 for a human expert.
  - Agent vs expert on sub-scores: selectivity 92.91 vs 98.92; steering 31.47 vs 57.75.
  - Agents misread measurements.

  (paper) — [Tan et al. 2609.09113](https://arxiv.org/abs/2609.09113)
- **HARP** (Hypothesis-driven Agentic Retrieval and Probing). An LLM agent gets a vector database of activations paired with their text contexts, plus tools to project out directions and to compute activation differences and averages. It validates hypotheses by building linear probes. With no training at all, it beats activation oracles and SAE-based agents on concept discovery, detection, steering and secret elicitation. (paper) — [Balasubramanian & Feizi 2607.16448](https://arxiv.org/abs/2607.16448)
- **HyVE (Hypothesize, Validate, Explain)** on AgenticInterpBench: 84 semi-synthetic circuits with 163 component-level annotations. Across four LM backbones, the failures come mainly in the validation loop: incomplete validation plans, code errors, and hypotheses left unresolved. (paper) — [Khan et al. 2606.24026](https://arxiv.org/abs/2606.24026)
- **Other automation.** (paper)
  - MechRL learns a transferable circuit-discovery policy on GPT-2 small. — [Khadka 2605.26343](https://arxiv.org/abs/2605.26343)
  - ADAG automates descriptions of attribution graphs with an explainer–simulator setup. — [Arora et al. 2604.07615](https://arxiv.org/abs/2604.07615)
  - Transluce's neuron-description pipeline describes all 458,752 MLP neurons of Llama-3.1-8B-Instruct with a fine-tuned explainer and simulator. — [Transluce report](https://transluce.org/neuron-descriptions)
  - An LLM interpretability agent given activation-difference tools identifies fine-tuning domains better than prompting-only agents. — [Minder et al. 2510.13900](https://arxiv.org/abs/2510.13900)
  - Transluce's investigator models elicit target behaviours, a behavioural rather than internal method. — [Li et al. 2502.01236](https://arxiv.org/abs/2502.01236)
- **Kim et al., agentic interpretability** (position paper). It proposes multi-turn interpretability in which the model helps build the human's mental model. It flags a "human-entangled-in-the-loop" evaluation problem. My notes also record the paper discussing conversing with a model while intervening on it ("open-model surgery") and noting that models do not reliably know why they behave as they do. (paper) — [Kim et al. 2506.12152](https://arxiv.org/abs/2506.12152)

#### Verification and confabulation-control devices reported in prior work
- **Separate describer and predictive scoring.** MAIA uses a separate describer model and predictive evaluation; ADAG and Transluce use explainer–simulator scoring. (paper) — [Rott Shaham et al. 2404.14394](https://arxiv.org/abs/2404.14394); [Arora et al. 2604.07615](https://arxiv.org/abs/2604.07615)
- **Known positives and nulls in OWMI.** A LoRA known to succeed, sham runs, impact-matched random perturbations, and a label-shuffled probe null. (paper) — [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **Controls in Zou et al.** Norm-matched random directions and lexical-replacement controls. (paper) — [Zou et al. 2609.35108](https://arxiv.org/abs/2609.35108)
- **Controls in Singh et al.** Gaslight prompts, permuted-label probes, and a baseline using layer-0 features. (paper) — [Singh et al. 2605.26242](https://arxiv.org/abs/2605.26242)
- **Target-only knowledge.** PersonaQA-Fantasy holds facts only the target knows, which separates verbaliser knowledge from target knowledge. (paper) — [Li et al. 2509.13316](https://arxiv.org/abs/2509.13316)
- **Implanted ground truth.** AuditBench and the introspection-adapter work implant behaviours to create ground truth; the adapters still report high false-positive rates and hallucinated behaviours. (paper) — [Sheshadri et al. 2602.22755](https://arxiv.org/abs/2602.22755); [Shenoy et al. 2604.16812](https://arxiv.org/abs/2604.16812)
- **Execution-grounded evaluation.** MechEvalAgent checks code, data and results, not just the paper's narrative. It agrees with human judges more than 80% of the time and surfaces 51 issues that human reviewers missed. (paper) — [Bai et al. 2602.18458](https://arxiv.org/abs/2602.18458)

### Inferences
- **(e) verdict: APPARENTLY OPEN.** Searches for LLM agents that run interpretability tools on their own activations or weights ("self-audit", "self-interpretation agent", "model inspects its own activations with tools") found only agents that audit other models (above) and fine-tuned activation readers. The closest designs are HARP (retrieval over activations plus probes) and Anthropic's investigator tools. Pointing those tools at the agent's own base model is new as far as I found. (inference)
- **Key confound for (e).** Once a model reads its own activations through tools, the tool outputs are text in its context, which an observer could be shown too. So privileged self-access in (e) must be measured as self-plus-tools against an observer given the same tools and tool outputs. Otherwise "self-discovery" is just interpretability performed by a model that happens to share weights with its subject. (inference)
- **Suggested bounded, read-only tool API for SelfLab** (inference, modelled on MAIA, Anthropic's auditing agents, AuditBench and HARP):
  - `get_activations(prompt, sites)`
  - `logit_lens(prompt, layer)`
  - `sae_features(prompt, layer, top_k)` and `feature_exemplars(id)`
  - `run_ablation(prompt, component, operator)` and `run_patch(src, dst, site)`, both returning only output deltas
  - `probe(dataset, site)` with a mandatory control task
  - `steer(direction, strength)`, gated
  - `log_hypothesis(pre-registered claim, predicted effect, confidence)` before any confirmatory tool call

  Budgets on calls and tokens reduce fishing. Ground-truth sweeps must stay hidden until predictions are locked.
- **Confabulation-control checklist** (inference, from the devices above):
  1. Pre-register each self-hypothesis with a numeric prediction.
  2. Run a known-positive condition (for example a LoRA trained to report).
  3. Include sham runs and norm- or impact-matched random interventions.
  4. Hold out intervention types and concepts.
  5. Include an observer with the same tools.
  6. Score by an independent executor against executed interventions, not by LLM-judged plausibility.
  7. Include target-only-knowledge items to catch confabulation from the model's own world knowledge.
  8. Include gaslight and suggestion prompts.
  9. Log full traces for audit.

### Gaps
- I did not extract per-tool numbers from AuditBench or the detailed ablations in Anthropic's auditing-agent report.
- I did not review the AI Scientist ([Lu et al. 2408.06292](https://arxiv.org/abs/2408.06292)) beyond its existence.
- I found no public tool API for a model introspecting itself, and no benchmark that scores self-directed interpretability against executed interventions.
- It is unknown whether HARP-style retrieval tools, pointed at a model's own activations, would raise self-prediction above an observer with the same tools. No data exists.

## 3. STATISTICS: Which estimators, bounds, scoring rules, calibration and agreement metrics, multiple-comparison, pre-registration and power methods should SelfLab use?

### Takeaway
Treat privileged self-access as conditional predictive V-information, I_V(R→Z|X) = H_V(Z|X) − H_V(Z|X,R). Estimate it as the held-out cross-entropy gain from adding the self-report R to observer-class predictors of Z from X.

Recommended inference:
- **Conditional randomisation test (CRT) or holdout randomisation test (HRT).** These are exact here because SelfLab itself randomises the interventions Z given X.
- **Variable-importance inference (VIM/GCM)** for targets that are not randomised.
- **Strictly proper scores and consistent calibration measures** (Brier or log score; smECE or debiased ECE).
- **meta-d′** for metacognitive sensitivity.
- **Item-clustered, paired inference.**
- **TOST** for null claims.
- **BH** for families of secondary tests.
- **Pre-registered, simulation-based power analysis.**

### Cited Findings

#### Information estimators and their limits
- **Predictive V-information.** Mutual information restricted to a computationally bounded predictor family V. It can increase through computation, unlike Shannon MI. (paper) — [Xu et al. 2002.10689](https://arxiv.org/abs/2002.10689)
- **Conditional probing.** I_V(φ→Y|B) = H_V(Y|B) − H_V(Y|B,φ), estimated with two probes as the difference in held-out cross-entropy. It is non-negative, it is zero under independence, and it has monotonicity properties. (paper) — [Hewitt et al. 2109.09234](https://arxiv.org/abs/2109.09234)
- **Pointwise V-information (PVI).** A per-item version, useful for finding which items carry privileged information. (paper) — [Ethayarajh et al. 2110.08420](https://arxiv.org/abs/2110.08420)
- **Limits on MI lower bounds.** Distribution-free, high-confidence lower bounds on mutual information cannot exceed about 2 ln N + 5 for N samples. The difference-of-entropies (DoE) estimator gives no guarantee either way. Cross-entropy upper bounds on entropy concentrate at rate 1/√N. (paper) — [McAllester & Stratos 1811.04251](https://arxiv.org/abs/1811.04251)
- **Variational MI bounds.** Their bias–variance trade-offs are well documented (InfoNCE is capped by log batch size). MINE/NWJ-type estimators have variance that grows exponentially with the true MI. (paper) — [Poole et al. 1905.06922](https://arxiv.org/abs/1905.06922); [Song & Ermon 1910.06222](https://arxiv.org/abs/1910.06222)
- **Nonparametric and classifier-based conditional MI.** (paper)
  - KSG. — [Kraskov et al. cond-mat/0305641](https://arxiv.org/abs/cond-mat/0305641)
  - CMI-kNN, with a local-permutation CI test. — [Runge 1709.01447](https://arxiv.org/abs/1709.01447)
  - CCMI, classifier-based. — [Mukherjee et al. 1906.01824](https://arxiv.org/abs/1906.01824)
- **Entropy bias for discrete Z.** Plug-in estimators are biased downward, which biases MI upward. NSB is a Bayesian estimator for the undersampled regime. (paper) — [Paninski 2003](https://doi.org/10.1162/089976603321780272); [Nemenman et al. physics/0108025](https://arxiv.org/abs/physics/0108025)
- **Probe capacity.** With unbounded probes, a representation cannot carry more information about a property than the input it came from. This motivates bounded probe families. (paper) — [Pimentel et al. 2004.03061](https://arxiv.org/abs/2004.03061)

#### Conditional-independence tests and variable importance
- **Model-X knockoffs and the CRT.** If the conditional law of the tested variable given the controls is known, the CRT gives exact finite-sample validity. Knockoffs control FDR. (paper) — [Candès et al. 1610.02351](https://arxiv.org/abs/1610.02351)
- **Conditional permutation test (CPT).** Valid when that conditional law is known. Excess Type I error is bounded by the total-variation error of an estimated law. (paper) — [Berrett et al. 1807.05405](https://arxiv.org/abs/1807.05405)
- **Holdout randomisation test (HRT).** A cheaper CRT that uses one model fitted on training data. (paper) — [Tansey et al. 1811.00645](https://arxiv.org/abs/1811.00645)
- **No free lunch.** Without restricting the null, no CI test is both valid and powerful against every alternative. The generalised covariance measure (GCM) is valid when the product of the two regressions' prediction errors shrinks faster than 1/n. (paper) — [Shah & Peters 1804.07203](https://arxiv.org/abs/1804.07203)
- **Algorithm-agnostic variable importance.** Defined as ψ = V(f₀) − V(f₀,ₛ) under deviance, AUC or R². It uses cross-fitting, with sample-splitting to handle the zero-importance boundary. (paper) — [Williamson et al. 2004.03683](https://arxiv.org/abs/2004.03683)
- **LOCO.** Leave-one-covariate-out predictive inference. (paper) — [Lei et al. 1604.04173](https://arxiv.org/abs/1604.04173)

#### Proper scoring rules and calibration
- **Strictly proper scoring rules** (log, Brier, CRPS) reward honest probabilities. (paper) — [Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437); [Brier 1950](https://doi.org/10.1175/1520-0493%281950%29078%3C0001:VOFEIT%3E2.0.CO;2)
- **ECE and temperature scaling** in modern networks. (paper) — [Guo et al. 1706.04599](https://arxiv.org/abs/1706.04599)
- **Binning underestimates calibration error.** A plug-in estimator needs O(B/ε²) samples against O(√B/ε²) for a debiased one. (paper) — [Kumar et al. 1909.10155](https://arxiv.org/abs/1909.10155)
- **Binned ECE is biased, most of all for well-calibrated models.** Equal-mass binning and ECE_sweep reduce the bias. (paper) — [Roelofs et al. 2012.08668](https://arxiv.org/abs/2012.08668)
- **smECE** is a consistent calibration measure, satisfying ½·dCE ≤ smECE ≤ 2√dCE, with no free hyperparameters. It ships as the `relplot` package. (paper) — [Błasiok & Nakkiran 2309.12236](https://arxiv.org/abs/2309.12236); [Błasiok et al. 2211.16886](https://arxiv.org/abs/2211.16886)
- **CORP reliability diagrams** use isotonic regression (PAV) and decompose the score into miscalibration, discrimination and uncertainty. (paper) — [Dimitriadis et al. 2008.03033](https://arxiv.org/abs/2008.03033)
- **Hypothesis tests of calibration.** (paper) — [Vaicenavicius et al. 1902.06977](https://arxiv.org/abs/1902.06977); [Widmann et al. 1910.11385](https://arxiv.org/abs/1910.11385)
- **Selective prediction** (risk–coverage) for abstentions, and **conformal prediction and conformal risk control** for distribution-free prediction sets. (paper) — [Geifman & El-Yaniv 1705.08500](https://arxiv.org/abs/1705.08500); [Angelopoulos & Bates 2107.07511](https://arxiv.org/abs/2107.07511); [Angelopoulos et al. 2208.02814](https://arxiv.org/abs/2208.02814)

#### Metacognitive sensitivity
- **meta-d′.** A signal-detection estimate of metacognitive sensitivity in d′ units. The M-ratio, meta-d′/d′, controls for first-order performance. (paper) — [Maniscalco & Lau 2012](https://doi.org/10.1016/j.concog.2011.09.021)
- **Measurement guidance.** Separate sensitivity, bias and efficiency. Type-2 AUROC is confounded by first-order performance. (paper) — [Fleming & Lau 2014](https://doi.org/10.3389/fnhum.2014.00443)
- **A Bayesian model of self-evaluation** in which confidence can draw on information beyond the first-order decision variable. This is relevant to Singh et al.'s "second-order computation" condition. (paper) — [Fleming & Daw 2017](https://doi.org/10.1037/rev0000045)
- **Review applying these ideas to LLMs.** (paper) — [Steyvers & Peters 2504.14045](https://arxiv.org/abs/2504.14045)

#### Agreement metrics and robust ground truth for interventions
- **Correlated AUCs.** DeLong's test compares AUCs on the same items, for example self vs observer. **NDCG** scores graded rankings, for example predicted vs measured component importance. (paper) — [DeLong et al. 1988](https://doi.org/10.2307/2531595); [Järvelin & Kekäläinen 2002](https://doi.org/10.1145/582415.582418)
- **Patching choices change conclusions.** The corruption method (Gaussian noise vs symmetric token replacement) and the metric (logit difference vs probability) can both change results. (paper) — [Zhang & Nanda 2309.16042](https://arxiv.org/abs/2309.16042); [Heimersheim & Nanda 2404.15255](https://arxiv.org/abs/2404.15255)
- **MIB** standardises circuit localisation and causal-variable localisation. (paper) — [Mueller et al. 2504.13151](https://arxiv.org/abs/2504.13151)
- **Non-identifiability.** Several circuits or interpretations can fit equally well. (paper) — [Méloux et al. 2502.20914](https://arxiv.org/abs/2502.20914)
- **Localisation does not predict editing success.** (paper) — [Hase et al. 2301.04213](https://arxiv.org/abs/2301.04213)
- **Subspace patching can create illusions.** (paper) — [Makelov et al. 2311.17030](https://arxiv.org/abs/2311.17030)
- **Causal abstraction.** Interchange interventions give a formal basis. With unconstrained non-linear alignment maps it becomes vacuous. (paper) — [Geiger et al. 2301.04709](https://arxiv.org/abs/2301.04709); [Sutter et al. 2507.08802](https://arxiv.org/abs/2507.08802)

#### Probing controls
- **Control tasks and selectivity.** (paper) — [Hewitt & Liang 1909.03368](https://arxiv.org/abs/1909.03368)
- **MDL probing.** (paper) — [Voita & Titov 2003.12298](https://arxiv.org/abs/2003.12298)
- **Review of probing's shortcomings.** (paper) — [Belinkov 2102.12452](https://arxiv.org/abs/2102.12452)

#### Multiple comparisons, equivalence, clustering, power, pre-registration
- **Benjamini–Hochberg FDR control.** Aoki et al. applied it (45 of 120 tests significant). (paper) — [Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); [Aoki et al. 2609.00904](https://arxiv.org/abs/2609.00904)
- **Equivalence testing (TOST) with a smallest effect size of interest.** OWMI's equivalence bound (<0.15 AUROC percentage points) is an example. (paper) — [Lakens 2017](https://doi.org/10.1177/1948550617697177); [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **Error bars for LLM evaluations.**
  - Clustered standard errors can be about 3× larger than naive ones in the paper's examples.
  - Resample K times per item.
  - Use next-token probabilities where possible.
  - Use paired differences.
  - Power formula: n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ². The worked example needs about 969 questions for δ = 0.03.

  (paper) — [Miller 2411.00640](https://arxiv.org/abs/2411.00640)
- **Underpowered comparisons are widespread in NLP.** (paper) — [Card et al. 2010.06595](https://arxiv.org/abs/2010.06595)
- **Do not rely on CLT intervals with fewer than a few hundred items.** (paper) — [Bowyer et al. 2503.01747](https://arxiv.org/abs/2503.01747)
- **Seed and benchmark variance.** (paper) — [Madaan et al. 2406.10229](https://arxiv.org/abs/2406.10229)
- **Bootstrap.** (paper) — [Efron 1979](https://doi.org/10.1214/aos/1176344552)
- **Pre-registration.** (paper) — [Nosek et al. 2018](https://doi.org/10.1073/pnas.1708274114)
- **Pre-registration in this area.** Blandfort (frozen test split) and SIL pre-registered. OWMI planned power by beta-binomial simulation (32 items give a 95% CI width of 1.235 d′). (paper) — [Blandfort & Pawar 2609.09899](https://arxiv.org/abs/2609.09899); [Tomaszewski 2608.14894](https://arxiv.org/abs/2608.14894); [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **OWMI's statistics module** implements method-of-moments variance components and ICCs, bootstrap intervals, edge-corrected d′ (`_edge_corrected_dprime`), simulated CI widths (`_simulated_ci_width`), and a beta-binomial `power_recommendation`. It is reusable as a reference implementation. (code-verified) — [owmi/stats.py](https://huggingface.co/emilioferrara/owmi/blob/main/owmi/stats.py)

### Inferences
Recommended estimator and inference stack for SelfLab. All items are (inference), built on the sources above.

1. **Randomise Z by design.** For each item X, draw the intervention from a known distribution P(Z|X). The draw covers component, operator, strength and sham, and should include impact-matched random directions. Two consequences follow:
   - Shannon I(Z;R|X) becomes meaningful. Without randomisation Z is a deterministic function of X, so the quantity is 0.
   - The CRT/HRT null for R ⊥ Z | X is exact. This sidesteps Shah–Peters no-free-lunch, which applies only when P(Z|X) is unknown.
2. **Primary estimand.** I_V(R→Z|X) = H_V(Z|X) − H_V(Z|X,R), with V the declared observer class.
   - Fit two predictors on a training split: Z from X, and Z from (X, R).
   - Report the difference in held-out cross-entropy in bits.
   - Use cross-fitting, and an item-cluster bootstrap for the CI.
   - Report per-item PVI to find where privileged information concentrates.
   - The choice of V operationalises Song et al.'s cost-matched third party. Report the result for each observer class, plus a max over the ladder, as OWMI intended.
3. **Hypothesis test.** Use a CRT (or HRT) with the held-out log-loss gain as the test statistic. For null findings, add TOST with a pre-registered smallest effect size of interest, in bits or AUROC points.
4. **Targets that are not randomised** (natural localisation in (b)). State that "privilege" here means only computational privilege relative to V. Use VIM with cross-fitting (Williamson) or GCM with strong regressions, and label claims accordingly.
5. **Direct comparison of self and observer.**
   - Paired differences in a proper score (log score, or Brier with CORP decomposition) on the same items.
   - DeLong for paired AUROC, and meta-d′ / M-ratio for binary predictions.
   - Item-clustered SEs; K ≥ 3 samples per item, or use token probabilities.
6. **Calibration.**
   - Primary: smECE with reliability diagrams.
   - Secondary: debiased or sweep ECE.
   - Treat refusals such as OWMI's localisation "cannot say" as abstentions and report risk–coverage curves, rather than dropping them.
7. **Graded agreement.**
   - (a)(ii): MAE or Spearman on Δlogit or KL.
   - (a)(i) and (b): NDCG@k over component rankings, and layer-distance or hit@±1 for layers.
   - (a)(iii): ROC-AUC against COAX-identified backups.
   - Build ground truth from symmetric-token-replacement patching plus logit difference, run both noising and denoising, and score against a set of near-equivalent ground truths to respect non-identifiability.
8. **Multiplicity and power.**
   - Pre-register one primary endpoint per capability.
   - Use gatekeeping for primaries and BH within secondary families.
   - Size studies by simulation from pilot variance components (OWMI's `power_recommendation` pattern, or Miller's formula).
   - Avoid CLT intervals below a few hundred items.

### Gaps
- I found no LLM study that validates conditional V-information, or a CRT, for self-reports. No standard exists for choosing V (the observer class). That choice is substantive and should be pre-registered.
- The power of the CRT/HRT against the small effects seen so far is unknown. For scale: OWMI's pooled AUROC is 0.5007 and Aoki's effects are all d < 0.5. SelfLab will need its own simulation.
- meta-d′ assumes equal-variance Gaussian signal detection. I found no check of that assumption for LLM confidence distributions.
- I did not verify a canonical source for rank-correlation or κ conventions in this setting. κ appears only descriptively in Song et al.

## 4. CONSTRUCTS: What exists on machine-native construct discovery, construct validity and falsification, prior use of "Plurience", self-model research, and 2025–2026 position papers?

### Takeaway
Validity frameworks are mature. The classics are Cronbach & Meehl, Borsboom, the multitrait-multimethod matrix (MTMM) and Messick; LLM-specific work includes Wallach, Chouldechova and Bean. Several pipelines show how to discover and causally test machine-native concepts: Schut's teachability and novelty filters, neologism "plug-in" evaluation, and the persona-vector pipeline.

I found no registry of machine-native cognitive constructs with alternative explanations and falsification conditions. "Plurience" has no prior use in AI or cognitive science; the only hit is a French property-management federation. The nearest measurable phenomena are belief-state geometry, ambiguity handling, and competing-belief dynamics.

### Cited Findings

#### Construct-validity frameworks
- **Classical construct validity.** Cronbach & Meehl: construct validity via a nomological network. Borsboom et al.: a test is valid if the attribute exists and variation in it causes variation in outcomes. Campbell & Fiske: convergent and discriminant validation with the multitrait-multimethod matrix. Messick: unified validity. (paper) — [Cronbach & Meehl 1955](https://doi.org/10.1037/h0040957); [Borsboom et al. 2004](https://doi.org/10.1037/0033-295X.111.4.1061); [Campbell & Fiske 1959](https://doi.org/10.1037/h0046016); [Messick 1995](https://doi.org/10.1037/0003-066X.50.9.741)
- **Bean et al.** A review of 445 LLM benchmarks by 29 expert reviewers.
  - 47.8% of phenomenon definitions are contested.
  - Only 16% of benchmarks use statistical tests.
  - Eight recommendations: define the phenomenon; measure only it; use a representative dataset; acknowledge reuse; prepare for contamination; use statistical methods; do error analysis; justify construct validity.

  (paper) — [Bean et al. 2511.04703](https://arxiv.org/abs/2511.04703)
- **Measurement theory for generative AI.** (paper)
  - Evaluating GenAI is a social-science measurement problem, running from a background concept to a systematised concept, then operationalisation, then instances. — [Wallach et al. 2502.00561](https://arxiv.org/abs/2502.00561)
  - A shared standard extending Adcock & Collier to contexts and metrics. — [Chouldechova et al. 2412.01934](https://arxiv.org/abs/2412.01934)
  - Measurement modelling with reliability and validity types. — [Jacobs & Wallach 1912.05511](https://arxiv.org/abs/1912.05511)
- **Psychometrics applied to LLMs, with cautions.** (paper)
  - A reliability and validity framework for LLM personality. — [Serapio-García et al. 2307.00184](https://arxiv.org/abs/2307.00184)
  - Human personality tests are not valid for LLMs. — [Sühr et al. 2311.05297](https://arxiv.org/abs/2311.05297)
  - LLM survey responses are driven by ordering and labelling biases. — [Dominguez-Olmedo et al. 2306.07951](https://arxiv.org/abs/2306.07951)
  - Factor structure of LLM capabilities. — [Burnell et al. 2306.10062](https://arxiv.org/abs/2306.10062)

#### Machine-native concept discovery, naming and causal validation
- **Schut et al. (AlphaZero).**
  - Sparse convex concept vectors are learned from MCTS rollouts, contrasting optimal with sub-par play.
  - A teachability filter trains a student AlphaZero checkpoint on concept prototypes and removes 97.6% of concepts.
  - A novelty filter compares spectral reconstruction on AlphaZero games against human games and removes a further 27.1%.
  - Four grandmasters improved after studying the surviving concepts.

  (paper) — [Schut et al. 2310.16410](https://arxiv.org/abs/2310.16410)
- **Hewitt, Geirhos & Kim** (position paper). Understanding AI needs new words, "neologisms", both for machine concepts and for human concepts that need to be conveyed to machines. (paper) — [Hewitt et al. 2502.07586](https://arxiv.org/abs/2502.07586)
- **Neologism learning.** On Gemma-3-4B-IT, a new word embedding is trained for a concept, and the model then verbalises its meaning.
  - Plug-in evaluation averages 92% for the neologism, 83% for the model's long self-verbalisation, and 39% for the best synonym.
  - It surfaces "machine-only synonyms" such as "lack" (sentence count 42.9 → 15.8) that transfer to Gemini.

  (paper) — [Hewitt et al. 2510.08506](https://arxiv.org/abs/2510.08506)
- **Persona vectors.** An automated pipeline goes from a trait description to contrastive system prompts, questions and a judge rubric. The vector is the difference in mean response activations.
  - Monitoring correlation: r = 0.75–0.83.
  - Trait shifts during fine-tuning correlate r = 0.76–0.97 with the vector, against 0.34–0.86 across traits.
  - Other uses: preventative steering and flagging training data.
  - Limitations: supervised, and requires a trait that can be elicited by prompting.

  (paper) — [Chen et al. 2507.21509](https://arxiv.org/abs/2507.21509)
- **The Assistant Axis.** The leading principal component of persona space tracks the default assistant persona, predicts persona drift, and can be stabilised by activation capping. (paper) — [Lu et al. 2601.10387](https://arxiv.org/abs/2601.10387)
- **Other discovery tools.** (paper)
  - Representation engineering. — [Zou et al. 2310.01405](https://arxiv.org/abs/2310.01405)
  - Sparse autoencoders. — [Cunningham et al. 2309.08600](https://arxiv.org/abs/2309.08600)
  - SAE features are not canonical units. — [Leask et al. 2502.04878](https://arxiv.org/abs/2502.04878)
  - Sparse feature circuits. — [Marks et al. 2403.19647](https://arxiv.org/abs/2403.19647)
  - A disentanglement benchmark. — [RAVEL 2402.17700](https://arxiv.org/abs/2402.17700)

#### Unsupervised discovery and its identification failures
- **CCS and its critique.** CCS finds truth-like directions without labels; Farquhar et al. show that consistency-based methods can latch onto arbitrary salient features. This is a warning for any "machine-native" construct found without supervision. (paper) — [Burns et al. 2212.03827](https://arxiv.org/abs/2212.03827); [Farquhar et al. 2312.10029](https://arxiv.org/abs/2312.10029)
- **Internal Coherence Maximisation.** Labels elicited without supervision through mutual predictability and logical consistency. (paper) — [Wen et al. 2506.10139](https://arxiv.org/abs/2506.10139)

#### Self-model research
- **Self-modelling in small networks.** Self-modelling as an auxiliary task simplifies networks. Self-interventional learning learns a self-model of the functional consequences of perturbing oneself. (paper) — [Premakumar et al. 2407.10188](https://arxiv.org/abs/2407.10188); [Tomaszewski 2608.14894](https://arxiv.org/abs/2608.14894)
- **LLM behavioural and learned self-models.** (paper) — [Binder et al. 2410.13787](https://arxiv.org/abs/2410.13787); [Betley et al. 2501.11120](https://arxiv.org/abs/2501.11120); [Plunkett et al. 2505.17120](https://arxiv.org/abs/2505.17120); [Zeng et al. 2608.30980](https://arxiv.org/abs/2608.30980); [Shenoy et al. 2604.16812](https://arxiv.org/abs/2604.16812)
- **Representations of the default persona.** (paper) — [Lu et al. 2601.10387](https://arxiv.org/abs/2601.10387); [Chen et al. 2507.21509](https://arxiv.org/abs/2507.21509)

#### "Plurience": prior use and nearest measurable phenomena
- **Prior use of the term.** Web searches for "Plurience", alone and combined with AI, language-model and world-model terms, returned only PLURIENCE, a French property-management federation. There was no AI or cognitive-science usage. A re-check of the registry page on 2026-09-30 returned HTTP 429 (rate-limited), but the page had been found earlier by search. (paper, for the registry entry; inference, for the absence) — [societe.com entry](https://www.societe.com/societe/plurience-500962410.html)
- **Belief-state geometry.** Transformers linearly represent belief states over the hidden states of the data-generating process in the residual stream, even when that geometry is fractal. Sometimes this representation is spread across layers. (paper) — [Shai et al. 2405.15943](https://arxiv.org/abs/2405.15943)
- **Ambiguity (AmbiEnt).** 1,645 examples. GPT-4's generated disambiguations were judged correct 32% of the time, against 90% for the dataset's own. (paper) — [Liu et al. 2304.14399](https://arxiv.org/abs/2304.14399)
- **Competing beliefs.** Steinmetz Yalon et al. introduce a metric for how dominant each competing belief is in latent space during generation. They report three findings: external manipulations modulate belief formation, belief formation causally drives action selection, and models can report their belief states. (paper) — [Steinmetz Yalon et al. 2602.02467](https://arxiv.org/abs/2602.02467)

#### 2025–2026 position and conceptual papers relevant to SelfLab
- [Kim et al. 2506.12152](https://arxiv.org/abs/2506.12152): agentic interpretability.
- [Hewitt, Geirhos & Kim 2502.07586](https://arxiv.org/abs/2502.07586): neologisms.
- [Wallach et al. 2502.00561](https://arxiv.org/abs/2502.00561): evaluation as measurement.
- [Bowyer et al. 2503.01747](https://arxiv.org/abs/2503.01747): statistics for small evaluations.
- [Song et al. 2508.14802](https://arxiv.org/abs/2508.14802): privileged self-access.
- [Comsa & Shanahan 2506.05068](https://arxiv.org/abs/2506.05068): a minimal definition of introspection.
- [Singh et al. 2605.26242](https://arxiv.org/abs/2605.26242): introspection reality check.
- [Bean et al. 2511.04703](https://arxiv.org/abs/2511.04703): construct validity recommendations.

(paper)

### Inferences
- **(f) verdict: PARTIALLY DONE in pieces, with no integrated precedent.** The validity theory, the discovery pipelines and the causal-test pipelines each exist separately. A registry of machine-native constructs, each carrying rival explanations, measurements, causal tests and falsification conditions, is new as far as I found. The term "Plurience" is unclaimed. (inference)
- **Suggested fields for each registry entry** (inference, synthesising Borsboom, MTMM, Schut, persona vectors and Bean):
  1. a definition in functional or causal terms;
  2. at least two independent measurement methods (behavioural, probe, intervention), to allow MTMM convergent and discriminant checks;
  3. rival explanations, each with a discriminating test;
  4. a causal test in which intervening on the putative representation changes behaviour as predicted;
  5. a teachability or plug-in test (Schut, neologism plug-in);
  6. a novelty test against human-concept bases;
  7. pre-registered falsification conditions;
  8. the observer class V used for any privileged-access claim.
- **Draft entry: Plurience (all inference).**
  - **Definition.** Keeping two or more mutually incompatible world-models, each with a calibrated weight, and acting on them without collapsing to one too early.
  - **Measurements:**
    - posterior over hypotheses decodable in the style of Shai (belief states);
    - AmbiEnt-type ambiguity tasks;
    - the belief-dominance metric of Steinmetz Yalon et al.;
    - action choices that hedge or seek information under ambiguity.
  - **Rival explanations:**
    - hedging as a stylistic or persona habit (Assistant Axis, persona vectors);
    - output entropy with no internal representation of the alternatives;
    - enumerating options because the prompt invites it;
    - a salient but unrelated feature (Farquhar).
  - **Causal tests:**
    - ablate or steer the subspace that encodes the minority hypothesis, and predict premature collapse, measured as lower hedging and worse decisions under reversal;
    - patch belief states between ambiguous and disambiguated contexts.
  - **Falsification conditions:**
    - adding the mixture representation gives no V-information about behaviour beyond output entropy;
    - interventions on the subspace leave behaviour unchanged;
    - the construct fails the discriminant test against stylistic hedging;
    - it fails the teachability or plug-in tests.

### Gaps
- I found no prior "construct registry" for machine cognition, so there is nothing to benchmark the format against.
- My search for "Plurience" relied on web search. Academic databases outside arXiv and Crossref-indexed venues were not searched exhaustively.
- The Steinmetz Yalon et al. paper motivates itself with a framing SelfLab excludes. Only its measurement method (belief dominance, belief-state reporting) is cited here.
- I did not verify the exact wording of the persona-vector and Assistant-Axis limitations beyond my extracted notes.

## 5. SUMMARY: For each tool or paper, what exists, what it makes redundant, and what is still missing for the self-knowledge mission, and which gaps are most defensible?

### Takeaway
SelfLab should not claim a first demonstration of LLM self-knowledge, intervention detection, trained self-explanation or behavioural self-prediction; all of these exist. Its defensible contribution is threefold:
- **A measurement science.** Pre-intervention causal self-prediction at component granularity, with graded effects, calibration and compensation, scored against executed interventions.
- **A matched observer ladder,** with a formally estimated and tested I_V(R→Z|X).
- **A tool-assisted self-audit arm** compared against an observer that has the same tools.

### Cited Findings
- **Trained self-explanation beats other-model explainers,** but untrained models are near the floor. (paper) — [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579)
- **Post-hoc detection of interventions is at chance across 8 open models,** with a known-positive LoRA and probes showing the information is present internally. (paper) — [Ferrara 2608.20569](https://arxiv.org/abs/2608.20569)
- **No consistent own-model advantage in behavioural self-modelling or self-prediction.** (paper) — [Zeng et al. 2608.30980](https://arxiv.org/abs/2608.30980); [Blandfort & Pawar 2609.09899](https://arxiv.org/abs/2609.09899); [Song et al. 2503.07513](https://arxiv.org/abs/2503.07513)
- **Compensation ground truth exists, but not self-prediction of it.** (paper) — [Gong et al. 2607.01940](https://arxiv.org/abs/2607.01940)
- **Tool-using interpretability agents exist, all aimed at other models, and all show a tool-to-agent gap.** (paper) — [Sheshadri et al. 2602.22755](https://arxiv.org/abs/2602.22755); [Tan et al. 2609.09113](https://arxiv.org/abs/2609.09113)

### Inferences
Summary table. Columns 3 and 4 are my inferences; column 2 restates the cited findings above.

| Tool/paper | Existing capability | What it makes redundant | Missing capability relevant to the self-knowledge mission |
|---|---|---|---|
| [Li et al. 2511.08579](https://arxiv.org/abs/2511.08579) (code: TransluceAI/introspective-interp) | Trained explainers predict activation-patching (4 layer blocks) and input-ablation outcomes. Self beats other-model explainers (Qwen3-8B patching EM 64.0 vs 54.1). | A "trained self-predicts-patching" demonstration; reuse its CounterFact and Hint-MMLU tasks as a trained baseline | Zero-shot; heads, MLPs and features; graded effects; calibration; same-weights and cost-matched observers; compensation; separating self-simulation from reading the activation (EM 59.9 without it) |
| [Binder et al. 2410.13787](https://arxiv.org/abs/2410.13787) | Cross-prediction design; a small trained self-advantage on hypothetical-behaviour properties; better self-calibration (mean absolute deviation) | The cross-prediction observer design for behaviour | Internal causal targets; untrained models; formal conditional-information estimate |
| [OWMI, Ferrara 2608.20569](https://arxiv.org/abs/2608.20569) + [code](https://huggingface.co/emilioferrara/owmi) | Post-hoc detection, localisation and confidence probes with sham, impact-matched random and text-only observer controls; known-positive LoRA; probe ceiling; TOST; power code | Re-running "was your computation altered?" on ≤15B open models; much of the control and statistics tooling | Pre-intervention prediction; an information-matched observer (observer lacks the prompt; the self-probe gets the object kind and mode); max-over-observer bound actually computed; layer localisation beyond refusals; models >15B |
| [Zeng et al. 2608.30980](https://arxiv.org/abs/2608.30980) | Behavioural self-modelling benchmark (10 task types), RL training, cross-model transfer (no consistent own-model advantage) | A new behavioural "would an edit flip your answer?" benchmark | Internal-component targets scored against executed interventions (check its COMPONENT-ATTRIBUTION task first); privileged-access estimation |
| [Blandfort & Pawar 2609.09899](https://arxiv.org/abs/2609.09899) | Pre-registered self vs generic-agent vs analyst vs other-model prediction of behaviour rates; self-gain ≈ 0; flattering bias | Any "do models know their behavioural rates?" study | Internal-state targets; interventional randomisation |
| [Song et al. 2503.07513](https://arxiv.org/abs/2503.07513), [2508.14802](https://arxiv.org/abs/2508.14802) | Similarity-controlled test of same-model advantage (null); definition of privileged access as cost-matched | A new definitional argument | An operational estimator of the cost-matched definition (V-information with a declared observer class) |
| [Singh et al. 2605.26242](https://arxiv.org/abs/2605.26242); [Ji-An et al. 2505.13763](https://arxiv.org/abs/2505.13763); [Aoki et al. 2609.00904](https://arxiv.org/abs/2609.00904) | Neurofeedback reporting and control; confound analyses (layer-0 predictability, gaslight prompts); small effects in fixed-sentence designs | Another neurofeedback replication | Causal-effect and compensation targets; calibrated reports |
| [Lindsey 2601.01828](https://arxiv.org/abs/2601.01828); [Macar et al. 2603.21396](https://arxiv.org/abs/2603.21396); [Lederman & Mahowald 2603.05414](https://arxiv.org/abs/2603.05414); [Pearson-Vogel et al. 2602.20031](https://arxiv.org/abs/2602.20031) | Detection and identification of injected concepts (~20%, 0 false positives in controls); gate-feature mechanism; content-agnostic detection | Another concept-injection detection study | Prediction before the intervention; component-level causal claims; comparison with an observer |
| [Zou et al. 2609.35108](https://arxiv.org/abs/2609.35108); [IFT 2607.14111](https://arxiv.org/abs/2607.14111); [Rivera & Africa 2511.21399](https://arxiv.org/abs/2511.21399) | Token-position localisation of injections (up to 65.95% vs 10% chance); fine-tuned localisation; steering detection limited by geometry | Token-level "where was I perturbed?" studies | Layer and component localisation of decision-critical information; generalisation across intervention families |
| [Guo et al. 2606.32038](https://arxiv.org/abs/2606.32038); [Plunkett et al. 2505.17120](https://arxiv.org/abs/2505.17120) | Training makes explanations track the model's own current behaviour; reporting of learned attribute weights | A "can training improve self-description?" study | Internal causal structure; an untrained baseline with matched observers |
| [Introspection adapters 2604.16812](https://arxiv.org/abs/2604.16812); [Activation Oracles 2512.15674](https://arxiv.org/abs/2512.15674); [PCD 2512.15712](https://arxiv.org/abs/2512.15712) | Fine-tuned readers of the same base model's weights and activations; strong auditing results; known false positives and hallucinations; an attack exists | Building a trained activation-reader or behaviour-verbaliser | Readers that are the live model itself; calibration; privileged-access tests (see [Li et al. 2509.13316](https://arxiv.org/abs/2509.13316) on verbaliser knowledge leakage) |
| [Gong et al. COAX 2607.01940](https://arxiv.org/abs/2607.01940) (+ [2307.15771](https://arxiv.org/abs/2307.15771), [2402.15390](https://arxiv.org/abs/2402.15390), [2211.00593](https://arxiv.org/abs/2211.00593)) | Ground truth for which components compensate after ablation (0.941 ROC-AUC on IOI backups) | Inventing a compensation ground-truth method | Any model's prediction of its own compensation |
| [Tomaszewski SIL 2608.14894](https://arxiv.org/abs/2608.14894) | A self-model learned from self-perturbation in small networks; recovers redundancy but not synergy | A toy proof of concept that self-experimentation yields predictive self-knowledge | LLM scale; verbal self-reports; comparison with an observer |
| [MAIA 2404.14394](https://arxiv.org/abs/2404.14394); [Anthropic auditing agents](https://alignment.anthropic.com/2025/automated-auditing/); [AuditBench 2602.22755](https://arxiv.org/abs/2602.22755); [HARP 2607.16448](https://arxiv.org/abs/2607.16448) | Tool APIs for agentic interpretability and auditing of other models; documented failure modes (confirmation bias, tool-to-agent gap) | Designing a tool API and agent scaffold from scratch | An agent auditing itself; comparison with an equally tooled observer; pre-registered self-hypotheses |
| [InterpAgent 2605.01555](https://arxiv.org/abs/2605.01555); [SAEScientist-Bench 2609.09113](https://arxiv.org/abs/2609.09113); [HyVE 2606.24026](https://arxiv.org/abs/2606.24026); [MechEvalAgent 2602.18458](https://arxiv.org/abs/2602.18458) | Metric batteries, gaps between agents and experts, execution-grounded verification | Designing verification metrics for agent claims | Applying them to self-directed claims |
| [Hewitt 2109.09234](https://arxiv.org/abs/2109.09234) / [Xu 2002.10689](https://arxiv.org/abs/2002.10689); [Candès 1610.02351](https://arxiv.org/abs/1610.02351) / [Tansey 1811.00645](https://arxiv.org/abs/1811.00645); [Williamson 2004.03683](https://arxiv.org/abs/2004.03683) | Conditional V-information via two probes; exact conditional randomisation tests; variable-importance inference | New estimator theory | Application to self-reports, with a declared observer class |
| [smECE 2309.12236](https://arxiv.org/abs/2309.12236); [Roelofs 2012.08668](https://arxiv.org/abs/2012.08668); [meta-d′](https://doi.org/10.1016/j.concog.2011.09.021); [Miller 2411.00640](https://arxiv.org/abs/2411.00640); [TOST](https://doi.org/10.1177/1948550617697177) | Consistent calibration measures, metacognitive sensitivity, clustered errors and power, equivalence tests | Home-grown calibration and statistics methods | Applying them to confidence about internal causal claims |
| [Schut 2310.16410](https://arxiv.org/abs/2310.16410); [neologisms 2510.08506](https://arxiv.org/abs/2510.08506); [persona vectors 2507.21509](https://arxiv.org/abs/2507.21509); [Assistant Axis 2601.10387](https://arxiv.org/abs/2601.10387) | Discovering machine-native concepts with teachability and novelty filters, plug-in tests, and causal steering and monitoring | A new concept-discovery method | A registry with rival explanations and falsification conditions per construct |
| [Bean 2511.04703](https://arxiv.org/abs/2511.04703); [Wallach 2502.00561](https://arxiv.org/abs/2502.00561); [Borsboom 2004](https://doi.org/10.1037/0033-295X.111.4.1061) | Construct-validity theory and LLM benchmark checklists | A validity framework written from scratch | Applying it to machine-native self-knowledge constructs such as "Plurience" (term unclaimed) |
| [Gurnee et al. 2026](https://transformer-circuits.pub/2026/workspace/index.html) | A Jacobian lens locating verbalizable representations; used by an auditing agent | A method for finding which representations can reach verbal report | Scoring the model's own reports about those representations against sweeps (details not extracted) |

### Gaps
Most defensible open gaps, ranked (inference; each reflects the negative searches recorded in Sections 1–4, current as of 2026-09-30):
1. **Zero-shot, component-level counterfactual self-prediction before the intervention, with graded effects, scored against executed interventions, against a matched and cost-matched observer ladder.** Privileged information would be estimated as I_V(R→Z|X) and tested with a CRT made exact by randomised interventions. Why it is open: Li et al. are trained, block-level and binary-plus-content; OWMI is post-hoc with an observer that does not see the same information; Zeng and Blandfort are behavioural. Check Zeng's COMPONENT-ATTRIBUTION definition first.
2. **Self-prediction of compensation** ("which component takes over if X is ablated?"), scored against COAX backups. No prior work was found by searches combining self-repair, backup heads or compensation with self-prediction, self-report or introspection.
3. **Layer and component self-localisation of decision-critical information, scored against patching and ablation sweeps,** with refusals treated as abstentions and non-identifiability respected. OWMI's layer probe mostly produced refusals, and prior localisation is only at the level of token positions.
4. **Calibration and metacognitive sensitivity (smECE, Brier with CORP, meta-d′) of confidence about internal causal claims, compared against observer confidence.** Existing calibration work covers behaviour, detection or answers.
5. **Tool-assisted self-audit, where an LLM runs read-only interpretability tools on its own activations.** It needs pre-registered self-hypotheses, verification by executed interventions, and comparison with an observer holding the same tools. Every existing agent audits other models.
6. **Zero-shot generalisation of self-knowledge across intervention families** (patching, ablation, steering, feature clamping). Evidence that it is hard: steering awareness is geometric (Rivera & Africa), IFT depends on random-layer training, and OWMI's known-positive covers one class.
7. **A Construct Registry with falsification conditions.** "Plurience" is an unclaimed term; the nearest measurement handles are belief-state geometry, AmbiEnt and belief-dominance metrics. Ranked last because the validity and discovery frameworks exist; the novelty lies mostly in integration and causal-test design.
