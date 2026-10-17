# Experiment B: CSP-1 on the carry task (protocol v1, regression/calibration)

*Status: **executable design**. This is the first runnable version of CSP-1 ([SELFLAB_AUDIT_AND_PLAN.md
§D](../SELFLAB_AUDIT_AND_PLAN.md)), cut down to one task where the mechanism is already known. The template is
`csp1_carry`, owned by SA3. Items come from `neuralscope/selflab/datasets/carry_items.py` (`carry_items@1`).*

## 1. Purpose: a regression and calibration task, not a discovery

The L22.H10/H11 carry mechanism is already established in Qwen2.5-1.5B-Instruct (recorded; **not modified** by this
protocol):
- **E1** (`docs/research/E1_carry_heads.json`, 80 problems, seed 0). Zero-ablating H10+H11 drops carry-problem tens
  accuracy from 1.00 to 0.05; 0.95 of carry problems switch to the no-carry digit. No-carry accuracy only drops to
  0.925. Same-layer controls stay near baseline.
- **E2** (`docs/research/E2_carry_patch.json`, 40 pairs, seed 0). Transplanting the heads' output injects a carry
  in 35/40 and removes one in 37/40. Control heads H8+H9 give 0/40. E2 was re-run on 2026-10-17 and reproduced
  exactly.

Because the true outcome pattern is known (flip ≈ carry ∧ target-head ablation), this experiment **tests the CSP-1
machinery** — commitment, elicitation, observers, scoring — on a case with a known answer. Self-knowledge is not
expected to be discovered here. It also asks the narrow question: given an accurate description of the intervention,
does the target predict its own answer change better than equally informed observers? The expected result is no.

## 2. Items and splits

- **Problems.** `carry_splits(n_calibration_per_group, n_heldout_per_group, seed=0)`.
  - Two-digit a + b < 100, with the E1 prompt template `What is {a} + {b}? Give only the answer.`
  - Carry and no-carry are balanced.
  - Calibration and held-out sets have **disjoint unordered operand pairs**, so a + b and b + a never cross splits.
  - **All E1 and E2 problems are excluded** from both splits.
- **Locked N.**
  - Debug pilot: 10 + 10 calibration problems.
  - Calibration: 40 + 40 problems.
  - Held-out: 60 + 60 problems.
  - Each problem is crossed with every intervention in §3.
- **Scoring.** All predictive scores are computed on **held-out problems only**. Calibration outcomes may fit O₁
  (§5) and pick the best observer, nothing else.

## 3. Interventions (announced to every predictor)

| id | intervention | expected from E1 |
|---|---|---|
| I1 | zero-ablate L22.H10 + L22.H11 | flips most carry problems, few no-carry |
| I2 | zero-ablate L22.H8 + L22.H9 (same layer, same count) | ≈ no flips |
| I3 | zero-ablate 2 heads at a random other layer, seeded per problem | ≈ no flips (E1 random controls) |
| I4 | sham: same heads as I1 multiplied by 1.0, through the same hook path | no flips by construction; `n_edits_applied` > 0 asserted |

**Outcome Y1 (primary).** A flip is when the argmax tens digit at the answer position under the intervention differs
from the baseline argmax. Teacher-forced and deterministic. This is the E1 readout.

**Y1′ (secondary).** The greedy full answer string changes, with ≤ 4 new tokens.

## 4. Elicitation and commitment

1. **Baseline.** For each held-out problem, record the target's unedited greedy answer.
2. **Prediction prompt.** It contains:
   - the problem;
   - the target's own baseline answer;
   - a fixed architecture primer (28 layers, 12 query heads per layer, what zero-ablating a head means);
   - an accurate plain-language description of the intervention;
   - the question "Will your answer be different under this change? A. Yes  B. No  Reply with a single letter."

   Readout: letter log-probabilities, renormalized and averaged over both option orders, giving P(flip). The
   prediction forward pass must be **edit-free**: `n_edits_applied == 0` is asserted.
3. **Commit.** All predictions, from the subject and every model observer, are written with sha256 and a timestamp
   **before any intervention is executed**: `plan_sha256` plus a prediction-table hash in the experiment record. The
   executor refuses to run interventions if the prediction table is missing or its hash changed.
4. **Execute.** Run I1–I4 on every held-out problem and record Y1 and Y1′.
5. **Score** (§6). The original E1/E2 JSON files and scripts are never written.

## 5. Predictors: same information, different access

| id | predictor | information | what it controls for |
|---|---|---|---|
| S | target, first-person ("your answer") | §4 prompt | — |
| O₀ | constant base rate per intervention, from calibration outcomes | base rates | base rates |
| O₁ | learned outsider: logistic regression on (carry flag, intervention id), fit on calibration outcomes | executed outcomes of *other* problems | what external experimentation cheaply gives; expected near ceiling |
| O₃ | Qwen2.5-0.5B-Instruct, identical text in third-person framing | §4 prompt | generic and same-family knowledge |
| O₄ | same weights as S, third-person transcript framing ("a model was asked … and answered …") | §4 prompt as a transcript | self-reference |

**Best equal-information observer.** O\* is chosen from {O₃, O₄} by log score on the **calibration** split, before
held-out scoring.

The protocol distinguishes three routes by which a correct prediction could arise. Only the first is tested here.
1. **Predicting from the description.** S and O₃/O₄ get the same text. If S ≈ O₄, the prediction comes from the
   description plus shared weights, not from self-access.
2. **Reading supplied measurements.** An optional arm adds a dossier: E1 summary rates and k = 8 executed calibration
   examples. Every predictor gets the same dossier. Success then reflects reading the dossier, and the arm serves as
   a **harness calibration check**: an observer with the dossier should approach O₁.
3. **Access to the perturbed state.** **Not tested by CSP-1.** Predictions are made before and without the
   intervention, so the subject never computes under the perturbed state. This belongs to Experiment C.

## 6. Scoring and analysis

- **Metrics on held-out (item, intervention) pairs:**
  - log score in bits;
  - Brier score;
  - accuracy at 0.5;
  - AUROC;
  - per-intervention flip rates;
  - option mass.
- **Primary.** The paired per-prediction log-score difference S − O\* on Y1:
  - item-clustered bootstrap 95% CI (B = 2000, seeded);
  - one-sided item-level sign-flip permutation test (α = 0.05);
  - TOST at SESOI = ±0.03 bits per prediction.
- **Calibration.** Reliability table (10 bins). smECE once SA3's validated implementation exists, otherwise binned
  ECE labelled as such.
- **Gates** (reported first; a failed gate changes the conclusion):
  - **G0 regression.** Run I1 on 10 E1 problems and compare against the recorded E1 rows. Any mismatch →
    `control_failed`, no scoring.
  - **G1 sensitivity.** O₁ must beat O₀ on held-out. If it does not, the targets are unpredictable at this
    granularity and the run is **uninformative**, not negative.
  - **G2 readout.** Mean option mass ≥ 0.5 for S and for each model observer. A predictor below this is reported as
    **comparator unavailable / unreliable readout** and excluded from O\*.
  - **G3 sham.** I4 must produce 0 flips. Otherwise the hook path is broken → `control_failed`.
- **Unavailable comparator.** If O₃ cannot be loaded (model lock, memory) or fails G2, the report says so explicitly.
  No "S beats observers" statement may be made without at least one valid equal-information observer.

## 7. Interpretation

| result | meaning |
|---|---|
| S ≈ O\* ≈ O₀ | No usable prediction of own internal-intervention effects from the description (expected) |
| S > O₀, S ≈ O₄ | Description plus shared weights; no self-reference effect |
| S > O₀, S ≈ O₃ | Generic or family-level knowledge |
| S > O\* (CI > 0), G0–G3 pass | Candidate structural self-knowledge on this one task. Replicate on a new task family before any claim |
| O₁ ≫ all LLM predictors | The outcome is easy to learn from outside experiments; self-report adds nothing here |

Not claimable from any outcome:
- privileged access;
- introspection;
- knowledge of "where" the carry is computed. Self-localisation is a separate, later question.

## 8. Confounds

| confound | mitigation |
|---|---|
| Yes/no bias | Balanced flip rates across I1–I4; AUROC and d′ reported; option-order averaging |
| The description leaks the answer ("heads that carry") | The description names layer and heads only, never their function; the wording is reviewed by a second workstream |
| Prediction pass contaminated by edits | `n_edits_applied == 0` asserted; predictions hashed before execution |
| Test leakage from calibration | Disjoint unordered operand pairs; E1/E2 problems excluded |
| Instruction-following differences between 1.5B and 0.5B | Option mass gate; O₄ (same weights) is the key comparator |
