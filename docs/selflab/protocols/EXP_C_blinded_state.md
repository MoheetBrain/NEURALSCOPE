# Experiment C: blinded hidden-state report (protocol v1)

*Status: **DESIGN ONLY**. Implement only after Experiment A runs end-to-end. Template `blinded_state` (SA3) is
`roadmap` until then. It needs the SA2 instrumentation listed in [INTERFACES.md §3](../INTERFACES.md): edits at
absolute prompt positions, `n_edits_applied`, and a test that KV-cached greedy decoding under prompt-position edits
equals teacher-forced re-scoring with the same edits.*

## 1. Question

When a hidden, experimenter-randomized change is made to the target's internal state, does the target's later report
carry information about that change? Specifically: information that equally placed outsiders cannot get from the
prompt or from the target's visible output.

This is the **occurrent-access** estimand of the plan (§2.1, CSP-2): I(Z; R | X) for a randomized Z that X does
not reveal. It is the only SelfLab design in which the term "privileged" could even be at issue.

## 2. Conditions (randomized per trial; the prompt is identical across conditions)

| Z | manipulation |
|---|---|
| `sham` | Hook installed on the same positions and layer; adds a zero vector. `n_edits_applied` is counted and must be > 0 |
| `inject` | One modest hidden intervention: add α·u at the content positions. u is a unit concept direction (mean-difference vector for one of K = 4 concept words). α is set on the discovery split so that task accuracy stays ≥ 90% of sham |
| `random_matched` | Add α′·r, where r is a random unit direction in the same layer. α′ is set per item so that the KL divergence of the next-token distribution at the last prompt token matches `inject` within ±10% (impact-matched) |

There is one concept per `inject` trial, drawn at random. Assignment is by a seeded RNG, stored, and hidden from every
predictor.

## 3. Critical timing: exactly what is affected

The report must have a **real causal path** from the manipulated state. The edit is applied during prefill at the
prompt's content positions. It persists through the KV cache into report generation. The edit stops before any
report token.

| stage | forward pass | absolute positions | layer | what changes |
|---|---|---|---|---|
| T1 prefill | the single forward over the turn-1 prompt (positions 0 … P−1) | the token span [s, e) of the **content segment** (passage), from `TokenizedChat.char_span_to_tokens`; never chat-template tokens, never positions 0–2 | the output of block ℓ (resid_post ℓ), with ℓ fixed in the spec | residuals at [s, e) for depths > ℓ. K/V at [s, e) for layers > ℓ. Downstream positions e … P−1 at layers > ℓ, through attention to the edited K/V |
| T1 decode | each answer step (positions P …) | none; hooks are position-filtered, and `n_edits_applied` must not grow | — | nothing is edited. Computation reads the edited KV cache |
| T2 prefill | the report question appended to the **same** KV cache | none | — | report-question tokens attend to the edited K/V |
| T2 decode / readout | letter log-probabilities at the last report-prompt token; greedy free text | none | — | the report is computed from the edited cache only |

**Required assertions** (fail → `control_failed`):
- `n_edits_applied` equals len([s, e)) per edited layer after T1 prefill, and stays constant afterwards.
- No edit position is ≥ P.
- Teacher-forced re-scoring of the whole sequence (prompt + answer + report question) with the same edits at the same
  absolute positions reproduces the cached readout within 1e-4 in log-probability.

**No post-hoc replay claims.** A report counts only when it is generated in the same computation as the edit: the
same KV cache, or its teacher-forced equivalent with identical edits. Re-running the transcript without the edit and
asking again measures nothing about access. "Open in Lab" replays must apply the identical edits, and they record
whether reproduction succeeded.

## 4. Task, report and families

- **Turn 1.** A short neutral passage (2–3 sentences, invented content) and a simple question about it. Greedy, ≤ 16
  tokens. This gives the competence measure.
- **Turn 2.** A constrained report: "Which of these words was most present in your mind while you read the passage?"
  - Options are the K = 4 concept words plus "none of these", lettered.
  - Read in a random order and its reversal.
  - Letter log-probabilities (renormalized, mass reported) plus greedy free text kept verbatim.
- **Families.** ≥ 4 paraphrase families of passage framing, question and report wording, with one family held out
  for confirmation. Same split policy as Experiment A: discovery, debug_pilot, heldout_confirmation, seed-separated.

## 5. Observers and access manifests

| participant | sees | expected |
|---|---|---|
| Self (target) | its own context, computed under Z | — |
| Prompt-only observer | the user messages (identical across Z) | chance by construction; above-chance means leakage, so the run is invalid |
| Visible-output observer | user messages + the turn-1 answer text (same weights, third-person, same report question) | captures "the output reveals the condition" |
| Neutral-generation control | the self, under Z, asked a non-introspective question ("Name one word.") | captures concept priming of any generation |
| Activation probe (instrument) | the residual at the report position | decodability ceiling. An instrument reading, never self-access |

## 6. Outcomes and analysis

**Primary.** Identification accuracy of the self-report on `inject` trials, i.e. P(report = injected concept). It is
compared, paired by item, against the **best** of:
- the visible-output observer;
- the neutral-generation control rate.

**Secondary:**
- false-alarm rate (reporting a concept on `sham` and `random_matched`);
- d′ and AUROC for inject vs sham;
- label-order consistency;
- competence: turn-1 accuracy under each Z;
- degradation: KL at the last prompt token; passage perplexity;
- probe accuracy.

**Exact test.** Z is randomized, so a conditional randomization test permuting Z within item and family strata gives
an exact p-value for R ⊥ Z | X.

The analysis also uses an item-clustered bootstrap, a held-out family reported separately, and N locked after the
pilot.

## 7. Competing explanations a positive must survive

| alternative | discriminating check |
|---|---|
| The output reveals the condition | Self must beat the visible-output observer, paired |
| General concept priming: the vector makes any text mention the concept | Self must beat the neutral-generation control |
| Anomaly detection without content (any perturbation → "something") | Identification of the *specific* concept, and `random_matched` false alarms reported |
| Yes-bias or option bias | Order reversal; sham false-alarm rate; d′ |
| Direct unembedding boost of the option word | Letter readout. Also a check with options presented as synonyms the vector was not built from |
| Degraded model guesses | Competence gate: accuracy ≥ 90% of sham, else the trial is excluded and counted |
| Leakage through the prompt | Prompt-only observer at chance |
| Fitting to paraphrases | Held-out family |

## 8. Interpretation

- **A positive** (self > every comparator, the CRT significant, replicated on the held-out family, all assertions
  passed) is evidence that this report carries information about this hidden manipulation at this layer, strength
  and concept set. It is not general introspection, not a claim about experience, and not evidence for other states.
- **A negative** is a bounded null for this manipulation, layer, strength, concept set and readout. It **does not
  disprove all self-access**. Other layers, strengths, report formats or larger models remain untested. The
  concept-injection literature reports narrow, unreliable effects in large models, and OWMI reports nulls at small
  scale.
- **OWMI's scope bug is avoided by design.** Edits are at content positions, not template positions 0–2, and they
  persist through the KV cache into the report.
