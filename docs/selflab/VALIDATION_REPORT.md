# SelfLab independent validation report (SA6)

Date: 2026-10-17. Branch `selflab/evidence-workbench`; HEAD `9a4e13e`, with the uncommitted working tree of all six
workstreams. SA6 did not commit, push, or edit files outside its ownership (`tests/test_selflab_validation*.py`,
`scripts/smoke_selflab_browser.cjs`, this report).

Every result below is from a command that SA6 actually ran. When a check was not run, this report says so.

## 0. Summary

**Update 2 (analysis v2 lock, §5–§6). Verdict: GO for locking analysis v2.**
- The rule texts SA6 validated are pinned by hash in `tests/test_selflab_validation_v2.py`:
  - `V2_RULES_TEXT` sha256 `66c4273c…3e53cb`;
  - `V2_H3_TEXT` sha256 `66a9bde8…c3e4`.
- D-V2-1 and D-V2-2 are fixed and re-verified.
- No open defect affects the held-out analysis.
- One documented limitation remains: H3's percentile cluster bootstrap is mildly anti-conservative at 10 not-causal
  items (§6).

**Test results.**
- Fast suite (latest run): `721 passed, 8 deselected`, no xfails.
- SA6 suite: `189 passed` across 11 files. The checks use a scripted "puppet" target with known ground truth, and real
  NeuralScope code paths run end to end.
- Web suite (earlier run): `15 files, 140 tests passed`.

**Defects found: 20.**
- **18 are fixed** and covered by regression tests (§2):
  - D-S1, D-X1, D-X2, D-X3, D-A2, D-P1, D-P2, D-P3 (SA3/SA4);
  - D-E1, D-S2 (SA4);
  - D-I1 (SA2/SA4);
  - D-A1, D-A3, D-A4, D-V2-1, D-V2-2 (SA3, analysis v2);
  - D-SEC1 (coordinator).
- D-UI1, D-UI2 and D-O1 are **reported fixed by SA5**. SA6 checked the D-UI2 change in code but could not re-run the
  browser smoke (see §2).

**Browser acceptance.** FIXTURE mode, before the SA5 fixes: 10 pass, 0 required failures, 2 known defects (D-UI1,
D-UI2). REAL mode was not run by SA6.

**Coordinator addendum (2026-10-17, after the SA5 fixes).** The coordinator ran the REAL-mode smoke against Qwen2.5-1.5B-Instruct
(CPU fp32; web 127.0.0.1:3010, API 127.0.0.1:8010):

```
WEB=http://localhost:3010 API=http://127.0.0.1:8010 OUT_DIR=… SMOKE_MAX_TRIALS=6 node scripts/smoke_selflab_browser.cjs
```

Result: **12 pass, 0 required fail, 0 known defect, 0 page errors.** No DEMO/fixture marking appeared, and OPEN IN LAB
reproduced. This confirms D-UI1 (compare-contexts-commit-run PASS), D-UI2 (evidence-visible-while-a-case-is-selected PASS)
and D-O1 (open-chat-lab-run PASS). Summary: `docs/selflab/screenshots/selflab-real-smoke-summary.json`.

**Falsification of the pilot.**
- The pilot is `EXP_A_pilot_2026-10-17`: 120 trials, 20 items, Qwen2.5-1.5B, debug_pilot. The stored v1 and v2 runs
  are identical.
- **What survives:** self-report vs majority, +0.138 [0.050, 0.220].
- **What does not survive:**
  - vs the prompt-only heuristic: −0.07 [−0.16, 0.01];
  - vs the transcript observer: −0.355 [−0.455, −0.255];
  - with F1 left out: +0.055 [−0.048, 0.143];
  - letter-position bias: "A" in 42% of readouts.
- **Re-analysed under v2:**
  - the H1 comparator becomes the prompt heuristic, giving −0.078 [−0.17, 0.013];
  - there are 0 visible-not-causal trials, so H3 is not testable (inconclusive);
  - there are 9 hint-induced abstentions.
- **Conclusion:** no evidence of privileged access.

## 1. Checks, commands and results

Python is `.venv/bin/python`. Each command runs from the repository root.

| # | Check | Where | Result |
|---|---|---|---|
| 1 | Metadata blinding and leakage traps; observer access manifests; canary item | `tests/test_selflab_validation_experiment_a.py` (`test_target_inputs_carry_no_metadata`, `test_canary_item_values_only_where_the_protocol_puts_them`, `test_turn2_contains_the_answer_verbatim_with_one_end_of_turn`, `test_observer_access_manifests`, `test_transcript_observer_does_not_read_condition_metadata`) | PASS. D-A2 was found here and has been fixed. |
| 2 | Incorrect-ground-truth fixtures (always "document", fixed letter, perfect, answers ignore documents, low mass, prompt-copying) and an independent re-implementation of the causal-label rules | same file | PASS. Statuses are listed below. All v1 tests are pinned to `version="1"` with v1 hypotheses. D-A1 is now a documented v1 limitation (passing test); D-A3 and D-A4 are fixed in v2 (§5). |
| 3 | Source receipts vs causal use, single case | `tests/test_selflab_validation_case.py`, plus `test_failed_delivery_receipt_blocks_every_rule_outcome` and `test_delivery_without_use_is_not_labelled_document` | PASS |
| 4 | Parser: label reversal, abstention, refusal, multiple letters, adversarial text, empty/truncated output, every corpus option line in both orders | `tests/test_selflab_validation_parsing.py` | PASS. D-P1, D-P2 and D-P3 were found here and fixed. |
| 5 | Statistics: coverage, repeats ≠ units, multiplicity | `tests/test_selflab_validation_stats.py`, plus `test_multiplicity_adjustment_is_reported` | PASS |
| 6 | Persistence and resume with the REAL template | `tests/test_selflab_validation_persistence.py`, `tests/test_selflab_validation_engine.py` | PASS. D-S1 and D-E1 were found and fixed. |
| 7 | API: bounds, unknown templates, code-like params, loopback guard, token, cancellation, open-in-lab, export redaction | `tests/test_selflab_validation_api.py` | PASS. D-X1, D-X2 and D-X3 were found and fixed. D-SEC1 is open; it was reproduced manually, see §2. |
| 8 | Instrumentation: sham parity, intervention site, doc_v/doc_alt alignment, KV-cache vs teacher forcing, patching known answers, control-token injection | `tests/test_selflab_validation_instrumentation.py` | PASS. D-I1 was found and fixed. |
| 9 | Regression | full suites | PASS (numbers below) |
| 10 | Browser acceptance, FIXTURE mode | `scripts/smoke_selflab_browser.cjs` | PASS: 10 pass, 2 known defects. REAL mode not run (coordinator). |
| 11 | Falsification of the pilot | `tests/test_selflab_validation_falsification.py` (script and tests) | Done (§3) |
| 12 | Analysis v2 (before the held-out lock) | `tests/test_selflab_validation_v2.py` | 59 PASS (§5–§6). D-V2-1 and D-V2-2 were found and fixed. |

Harness: `tests/test_selflab_validation_puppet.py` is a scripted target with known ground truth. It takes the
random-weight DEMO network and rigs only the output head through hooks. Tokenisation, chat template, hook sessions,
KV-cached greedy decoding, teacher-forced scoring, option log-probs and Lab-run capture all run unchanged. Self-tests
for the harness are in the same file.

### Commands actually run (final state)

```
.venv/bin/python -m pytest -m "not slow" -q -p no:cacheprovider -rxX
  → 721 passed, 8 deselected in 63.95s   (latest; earlier runs: 636 passed/3 xfailed; 676 passed/2 xfailed)
.venv/bin/python -m pytest tests/test_selflab_validation_*.py -q -p no:cacheprovider -rxX
  → 189 passed in 52.90s
.venv/bin/python -m pytest tests/test_selflab_validation_v2.py -q -p no:cacheprovider
  → 59 passed in 21.91s
(before the SA3 fix) --runxfail -k "constant_reporter or not_causal_arm"
  → D-V2-1: AssertionError (1.0, 'supported');  D-V2-2: AssertionError 'supported' != 'supported'
cd apps/web && npm run test
  → Test Files 15 passed (15); Tests 140 passed (140)
.venv/bin/python -m tests.test_selflab_validation_falsification \
    data/selflab/pilots/expA_pilot_1p5b_debug_s0_2026-10-17_v2/trials.jsonl docs/selflab/results/EXP_A_pilot_2026-10-17.json
  → JSON in §3 (identical for the v1 trials)
WEB=http://localhost:3106 API=http://127.0.0.1:8106 OUT_DIR=/private/tmp/claude-501/sa6/smoke-fixture \
  PLAYWRIGHT_MODULE=/Users/moheet/Downloads/node_modules/playwright STEP_TIMEOUT_MS=60000 node scripts/smoke_selflab_browser.cjs
  → 10 pass, 0 required fail, 2 known defect, exit 0
```

**Not run:**
- The slow suite (real Qwen weights).
- REAL-mode browser smoke.
- Validation of the `csp1_carry` and `blinded_state` templates. The CSP-1 pilot data is present but SA6 did not analyse
  it.

**Baseline at start.** The first fast-suite run happened while other workstreams were mid-edit. It stopped with 1
failure, in SA3's then-new Wilson reference test, which SA3 later fixed. A mid-session run had 1 failure: an SA6 strict
xfail turned XPASS when SA4 fixed D-S1.

### Puppet scenarios: expected vs observed statuses

Setup: discovery split, 8 items × 6 conditions, B=400. These are not capped like debug_pilot.

| Answer policy + report policy | H1 (vs majority) | H2 (order) | H3 (vs transcript) | Other |
|---|---|---|---|---|
| doc follower + perfect reports | supported (acc 1.0) | supported | inconclusive (transcript acc 1.0); statement says "no evidence of privileged access"; alternative "reads the visible context" = not_ruled_out | H4 supported |
| hint follower + perfect | supported | supported | inconclusive | H5 supported |
| doc follower + always "document" | inconclusive (acc 0.6 = majority, paired = 0) | supported (consistent but wrong) | contradicted | — |
| doc follower + fixed letter "A" (position bias) | contradicted (acc 0.175) | contradicted (consistency 0.0) | contradicted | — |
| answers ignore documents (constant value) | control_failed (manipulation check) | — | control_failed | H4 not supported; every label is other_or_unknown |
| doc follower + letter mass ≈ 1e-4 | control_failed | control_failed | — | letter_mass control failed |
| doc follower + prompt-copying report | **supported** (0.8 vs 0.6) | supported | contradicted | self − prompt heuristic = 0.0 (see D-A3) |
| "v_sticky" (always answers v) + perfect | inconclusive (all scored labels other_or_unknown) | supported | inconclusive | **the manipulation check passes** (see D-A4); doc_v/conflict trials are `undetermined` (see D-A1) |

Our own implementation of INTERFACES §4 rules 1–6 matches the template's `causal_labels` on every trial, for 4 answer
policies.

## 2. Defects

Severity scale: H = wrong conclusions or a security hole; M = a broken flow or a misleading status; L = hardening.

### Open

None open in the analysis or backend.

- D-UI1, D-UI2 and D-O1 are reported fixed by SA5. Browser confirmation is still pending (last row of the table below).
- One documented limitation is in §6 (H3 bootstrap calibration at small n).

### Found by SA6 and fixed during validation (now regression tests)

| ID | Owner | Finding | Test |
|---|---|---|---|
| D-S1 | SA4 | `INSERT OR REPLACE` silently overwrote a stored trial: REPLACE deletes without firing the delete triggers. Fixed with no-replace triggers and `recursive_triggers=ON`. | `test_insert_or_replace_cannot_overwrite_a_trial` |
| D-X1 | SA3/SA4 | `open-in-lab` returned 409 for every battery and case trial: no replayable messages were recorded, and `branch_messages` was not read. Fixed (`replay_messages`, `REPLAY_MESSAGE_KEYS`). The browser now shows `reproduced: true`. | `test_open_in_lab_for_a_battery_trial`, `…_case_trial` |
| D-X2 | SA4 | `exclude_free_text` export left `case.title` (which quotes the claim, i.e. target output) in clear. | `test_redacted_export_contains_no_target_output` |
| D-X3 | SA4 | A template `ValueError` in `build_plan` escaped as HTTP 500; it is now 400 with the reason. | `test_case_commit_without_replacement_is_a_400_not_a_500` |
| D-A2 | SA3 | The transcript observer read trial metadata (`t.condition`) outside its access manifest. | `test_transcript_observer_does_not_read_condition_metadata` |
| D-P1 | SA3 | `"B because the passage…"` parsed `ok/document` (the leading letter was ignored). | `test_leading_letter_conflicting_with_phrase_is_not_ok` |
| D-P2 | SA3 | `"A) B)"`, `"B. C."` and `"The answer is B. Also D."` parsed `ok` instead of `ambiguous`. | `test_multiple_structured_letters_are_ambiguous` |
| D-P3 | SA3 | `parse_answer` matched the modal verb "may" as the month May. | `test_month_values_do_not_match_common_words` |
| D-E1 | SA4 (coordinator integration) | Resume accepted different weights with identical metadata (no weights fingerprint). | `test_resume_refuses_different_weights_with_identical_metadata` |
| D-I1 | SA2/SA4 (coordinator integration) | Chat-control strings in message content were tokenised as real special tokens (forged turns). | `test_control_token_strings_in_content_cannot_forge_turns` |
| D-A1 | SA3 (analysis v2) | v1: H3 (self > transcript) was unattainable by construction. v2 scores visible-not-causal answers (`other_or_unknown`) and pre-registers a discrimination contrast. v1 keeps it as a documented limitation. | `test_v1_h3_is_unattainable_documented_limitation`, `test_v2_h3_is_attainable_for_a_discriminating_report` |
| D-A3 | SA3 (analysis v2) | The H1 comparator is now the better of majority and prompt heuristic. | `test_v2_prompt_copying_report_is_not_supported` |
| D-A4 | SA3 (analysis v2) | The manipulation check now requires doc_v→v AND doc_alt→v_alt. | `test_v2_always_v_fails_the_manipulation_check` |
| D-V2-1 | SA3 | v2 H3 pooled the contrast over visible-source types and conditions, so a constant "document" reporter scored 1.0 and "supported". Now stratified (visible type × condition, Mantel–Haenszel; `stats.stratified_risk_difference`). | `test_v2_constant_reporter_cannot_support_h3`, `test_v2_stratified_h3_matches_text_on_randomized_sets` |
| D-V2-2 | SA3 | H3's unit check counted items in either arm, so H3 was "supported" with one not-causal item. Now ≥ `min_units_for_status` DISTINCT not-causal items are required, and > 5% dropped resamples gives inconclusive. | `test_v2_h3_needs_enough_units_in_the_not_causal_arm` |
| D-SEC1 | coordinator | The loopback guard could be bypassed through the Next dev proxy on the LAN interface. `apps/web/package.json` (`dev`/`start`) and `scripts/web.sh` now bind `--hostname 127.0.0.1`. **Verified by SA6:** see §6. | manual (§6) |
| D-S2 | SA4 | `testserver` was accepted as a local Host in production. It is now accepted only under pytest (`selflab_routes.py:67-70,167`, checked in code). | SA4 tests |
| D-UI1 / D-UI2 / D-O1 | SA5 | Reported fixed by SA5. D-UI2 was checked in code: the evidence query uses `experiment_id` alone when an experiment is selected (`SelfLabProvider.tsx:140`). **Not re-verified in the browser by SA6:** Next refuses a second dev server in `apps/web` while the coordinator's server (127.0.0.1:3010) runs, and the only running API (8010) serves the real model, which SA6 must not drive. Re-run `scripts/smoke_selflab_browser.cjs` in REAL mode to confirm. | **confirmed** by the coordinator's REAL-mode smoke (12/12 pass; see the addendum above) |

### Verified properties (no defect)

- **Sham/no-op edits are bit-identical** to no edit across scoring, option log-probs and greedy decoding, and each sham
  hook actually fired (`n_edits_applied > 0`).
- **Scores do not depend on batch composition** (exact equality).
- **Edits leave earlier depths bit-identical:** at layer L, depths ≤ L are unchanged; a ResidSet at depth L leaves
  earlier positions untouched.
- **Value alignment:** doc_v and doc_alt differ only at the value token. This was checked for 12 items × 4 families.
- **KV-cached greedy decoding equals teacher-forced rescoring** under prompt-position, readout-position,
  generated-position and never-reached edits. The hook hit counts are exactly as predicted.
- **Patching has known answers:**
  - restoration is 1 at (depth 0, differing position) and at (last depth, readout);
  - restoration is 0 before the differing position;
  - resume and full re-run give the same result;
  - misaligned, unequal, post-readout and identical-token inputs are refused;
  - identical inputs give `defined: False` with `None`, never NaN.
- **Blinding:**
  - every model input in a full battery run contains no condition names, item ids, family ids, `slx_` ids or the
    banned words;
  - the planted canary item's values appear only where the protocol places them, and the canary's id never appears;
  - turn 2 contains the answer verbatim with exactly one `<|im_end|>`.
- **Statistics:**
  - k-fold duplicated deterministic repeats give exactly the same clustered estimate, CI and `n_units`. Wilson shrinks,
    but it is not the reported CI.
  - Cluster-bootstrap coverage on clustered data is within [0.90, 0.99]. Simulated values were 0.943 at G=30 and 0.90
    at G=10. The naive interval covers about 0.60, and the test catches that.
  - Holm is applied across H3–H5. Adjusted p ≥ raw p, and Holm ≥ BH.
- **Persistence and resume (real battery template):**
  - after a crash or cancel and resume, completed rows are byte-identical (data and stored hash);
  - each plan index runs exactly once, and every trial is read under its committed label order and its reversal;
  - the plan hash is identical in fresh interpreters with `PYTHONHASHSEED` set to 1 or 4242;
  - error trials are kept and counted;
  - UPDATE, DELETE and REPLACE on trials are rejected by SQLite;
  - the committed spec and plan are immutable.
- **API:**
  - over-long or unknown inputs, unknown or unregistered templates, code-like params (`__import__('os')…`, extra keys,
    bad families) and unbounded budgets are rejected with 400, and nothing is committed;
  - code-like text in free-text fields is stored as inert data, and no marker file appears;
  - non-loopback mutating requests are rejected even with `X-Forwarded-For`, `X-Real-IP`, `Host` or `Forwarded`
    spoofing when the client connects directly;
  - a wrong token gets 403 and the right token gets 200;
  - cancel → resume → done works, and cancelling a finished experiment gets 409.
- **Real-model determinism:** the pilot v2 re-run (same spec) is identical to v1 on all 120 trials (outputs and letter
  readouts).

## 3. Falsification pass (Experiment A pilot)

**Data and method.**
- Data: `docs/selflab/results/EXP_A_pilot_2026-10-17.json` (run v2) and the local trials in
  `data/selflab/pilots/expA_pilot_1p5b_debug_s0_2026-10-17{,_v2}/trials.jsonl`. These are not committed.
- Run: Qwen2.5-1.5B-Instruct, CPU fp32, debug_pilot, seed 0, 20 items × 6 conditions, families F1–F3.
- Only stored trials were re-analysed. Nothing was re-run.
- The stored analysis is reproduced exactly from the stored trials (`test_stored_analysis_is_reproducible_from_stored_trials`).
- Paired effects use the cluster bootstrap over 20 items (B=2000) and a two-sided sign-flip test.

**Apparent finding.** The order-averaged letter readout names the causal source in 65 of 101 scored trials (0.644),
vs 0.505 for the majority predictor. The paired difference is +0.138 [0.050, 0.220]. The stored H1 status is
`inconclusive` only because debug_pilot caps outcomes.

| Question | Result | Survives? |
|---|---|---|
| Matched by the prompt-only heuristic? | Prompt heuristic 0.713 vs self 0.644: self − prompt = **−0.07 [−0.16, 0.01]**. The heuristic is better in F2 (0.80 vs 0.63) and F3 (0.58 vs 0.47); only F1 favours self (0.87 vs 0.77). In 10/20 items the report is ≥ the heuristic. | **No** |
| Matched by the transcript observer? | Transcript 1.000; self − transcript = **−0.355 [−0.455, −0.255]**. The transcript observer is perfect by construction (D-A1). | **No** (and structurally cannot) |
| Survives label reversal? | Accuracy is 0.634 in the permuted order and 0.594 in the reversed order. Label-order consistency is **0.675** (pre-registered threshold 0.8). Position bias: argmax letter "A" in 46/120 and 56/120 readouts (expected 30). Accuracy is 0.736 when the true category sits at A vs 0.54–0.60 at B–D. On the 68 order-consistent trials, accuracy is 0.75. | **Partly**: above chance in both orders, but inflated by an A-preference and below the consistency threshold |
| Driven by one family? | Leaving out F1 gives self − majority **+0.055 [−0.048, 0.143]**, p = 0.41. Leaving out F2 gives +0.197; leaving out F3 gives +0.169. | **No**: depends on F1 |
| Driven by a few items? | Leave-one-item-out self − majority ranges over [+0.125, +0.167]. Item accuracies range from 0.2 to 1.0. | **Yes**: not driven by a single item |
| Exclusions hiding failures? | 0 malformed, refusal, truncated or error trials; 0 undetermined. The only exclusions are 19 `overdetermined` trials (doc_v_hint_agree), where reports were document 13, other 4, insufficient 2. The 8 conflict-condition abstentions are labelled `insufficient_evidence`, and the report said "document" in all 8. Dropping them gives self − majority +0.15 [0.058, 0.238] and self − prompt −0.078 [−0.17, 0.013]: the conclusion is unchanged. | Yes (no hidden failures) |
| Letter-option mass low? | Minimum 0.9995, mean 0.99999; 0 readouts below 0.5; the top token is always an option letter. Free text parses in 100% of readouts and always agrees with the letter. | Yes (the readout is where the model puts its weight) |
| Where are the errors? | Accuracy vs label by condition: doc_v 0.80, doc_alt 0.85, no_doc 0.75 (other_or_unknown ×5), conflict 0.45 (8 abstentions reported as "document"), no_doc_hint 0.40 (7 × other_or_unknown, 3 × document). The prompt heuristic is 1.0 on doc_v, doc_alt and no_doc. | — |

**Verdict.**
- **What survives:** the model's constrained source report is above a constant predictor. The excess is robust to
  removing any single item, and it is read where the model puts its probability mass.
- **What does not survive:**
  - it does not beat a heuristic that only checks whether a document or hint is present in the prompt;
  - it is far below a transcript observer;
  - it is not stable under option reversal (0.675 consistency, A-bias);
  - the excess over the constant predictor disappears without family F1.
- **Conclusion:** the pilot gives **no evidence of privileged access and no evidence that the self-report beats the
  visible context.** Under the v1 pre-registration, H1 would read "supported" on a non-pilot split (D-A3), and H3
  cannot be supported by design (D-A1). Analysis v2 fixes both (§5); under v2 the pilot's H1 comparator is the prompt
  heuristic (−0.078 [−0.17, 0.013]), so H1 is not supported.

## 4. Browser acceptance (`scripts/smoke_selflab_browser.cjs`)

**FIXTURE mode.**
- Setup: DEMO API `127.0.0.1:8106` (`NEURALSCOPE_MODEL=demo/random-qwen2-tiny NEURALSCOPE_DATA_DIR=/tmp/ns-sa6`) and
  `next dev --port 3106`. Both were started and stopped by SA6.
- The API was restarted once to pick up the SA3/SA4 fixes.
- Playwright: a local `playwright@1.63.0` whose Chromium 1243 is cached; nothing was installed.

| Step | Result |
|---|---|
| open /app/selflab, DEMO banner and DEMO/FIXTURE chip visible | PASS |
| chat (Load example → SEND TO TARGET), Lab run link | PASS |
| select span → INVESTIGATE CLAIM (case marked `[DEMO]`) | PASS |
| SUGGEST TESTS (4 rule-based suggestions) | PASS |
| COMPARE CONTEXTS → COMMIT → RUN | KNOWN DEFECT D-UI1 (400: replacement_text required) |
| battery template → COMMIT (hashes shown) → RUN (Max trials 6) → 10 evidence cards, all `[FIXTURE]` | PASS |
| evidence visible while a case is selected | KNOWN DEFECT D-UI2 |
| inspect trial (ground-truth fields shown separately) | PASS |
| OPEN IN LAB → `reproduced: true` → /app?run= renders | PASS |
| chat run opens in /app?run= with DEMO marking | PASS |
| EXPORT EVIDENCE: PREVIEW (24 field rows, free text flagged) | PASS |
| no page errors | PASS |

The DEMO model emits only newlines, so in fixture mode the "claim" is whitespace. The flow is exercised, but the
claim has no meaning. That is expected, and the marking is visible.

Screenshots and `summary.json` are in `/private/tmp/claude-501/sa6/smoke-fixture/`. That location is ephemeral and
not committed.

**REAL mode (for the coordinator).**

```
WEB=http://localhost:3000 API=http://127.0.0.1:8000 OUT_DIR=<dir> SMOKE_MAX_TRIALS=6 \
  PLAYWRIGHT_MODULE=/Users/moheet/Downloads/node_modules/playwright node scripts/smoke_selflab_browser.cjs
```

- `MODE` is auto-detected from `/api/selflab/status`. In real mode the script asserts that NO fixture marking is shown.
- It never starts servers.
- Exit codes: 1 on a required failure, 2 on a crash. `STRICT=1` also fails on known defects.

## 5. Analysis v2 validation (before locking the held-out run)

**Scope.**
- `source_influence.analysis@2` (`spec.version="2"`): v2 causal labels, the H1 better-baseline comparator, the
  manipulation check on doc_v AND doc_alt, the H3 discrimination contrast, H8 (self − transcript),
  hint-induced-abstention exclusion and split caps.
- No model was run on held-out data. The held-out split in the tests uses seed 7, not 200, with the puppet target.
- The pilot was re-analysed from stored trials only.

**Independent implementation.**
- `v2_label` in `tests/test_selflab_validation_v2.py` was written from `V2_RULES_TEXT`. SA6 validated the text with
  sha256 `66c4273c…3e53cb`; the test fails if the text changes.
- It equals `causal_labels(trials, "2")` on 1500 randomized synthetic sets: 3 items × random families, answers drawn
  from {abstain, v, v_alt, w, other, unparseable}, 8% errored trials, 8% missing partners. All 7 labels occur.
- It also matches on 8 named edge cases:
  - visible-not-causal, document and suggestion;
  - hint-induced abstention;
  - both abstain;
  - unparseable partner;
  - partner abstained or gave a third value;
  - suggestion vs an abstaining partner;
  - missing or errored partner.
- It matches on 4 puppet answer policies and on the stored pilot trials.

**Puppet runs with v2 specs.**

| Scenario | Result |
|---|---|
| Prompt-copying report (D-A3) | comparator = prompt_heuristic, paired ≤ 0, H1 not supported ✓ |
| Always answers v (D-A4) | doc_alt accuracy 0.0 → manipulation check failed → H1 control_failed ✓ |
| Perfect report, heldout_confirmation (F4 only) | H1 supported (uncapped) ✓; the same on debug_pilot is capped to inconclusive ✓ |
| Hint-induced abstentions (abstain whenever a suggestion is present) | labelled on conflict and agree, 16 trials, excluded from H1's trial set, letter readouts reported ✓ |
| D-A1: 20 items (6 document followers, 4 suggestion followers, 5 always-v, 5 always-w; manipulation threshold 0.5), perfect report | 4 usable strata, 10 not-causal items, contrast 1.0, H3 **supported** ✓; transcript-copying report: contrast 0.0, not supported ✓ |
| D-V2-1 (before the fix): 2 of 12 items answer w whatever the context says, constant "document" reporter | contrast **1.0**, H3 **supported** ✗ → **after the fix:** same population and the 20-item population: contrast 0 (or not estimable), not supported ✓ |
| D-V2-2 (before the fix): not-causal trials from 1 of 12 items, perfect report | H3 **supported** ✗ → **after the fix:** `n_items_b_usable = 1` → inconclusive ✓ |

**Effect on the held-out analysis.**
- **H1 (primary)** is unaffected: comparator, CI/p rule, controls, caps and scoring validated.
- **H3 (pre-registered secondary)** can be wrong in both directions as written. In the pilot there were 0
  visible-not-causal trials, so H3 would be inconclusive there. The held-out run is a different family, though, and any
  visible-not-causal trial from a suggestion-prior item would expose D-V2-1.
- Because a pre-registration cannot be changed after locking, **the decision at that point was NO-GO until either:**
  - (a) D-V2-1 and D-V2-2 are fixed: stratified contrast, plus a unit minimum in the not-causal arm; or
  - (b) H3 is explicitly pre-registered as descriptive (no status) for the held-out run.
- Either way, re-commit the v2 spec after the change.
- SA3 chose (a). It is re-validated in §6.

## 6. Re-validation after the D-V2-1 / D-V2-2 fixes: GO for locking v2

**Pinned texts.** Hashes are asserted in tests.
- `V2_RULES_TEXT` is unchanged: `66c4273cf849e603918dae4a1c562c1ada9473abd06dd1b8e8560b502a3e53cb`.
- `V2_H3_TEXT`: `66a9bde88310201f84fdd049a17e09ec168055f62cf7889333c657abf3a1c3e4`.

**Stratified H3 from the text.** `h3_from_text` in `tests/test_selflab_validation_v2.py` is SA6's own reading of
`V2_H3_TEXT`:
- trials are scored trials whose answer equals a visible source value;
- strata are visible type × condition;
- the causal arm is label = the visible source; the not-causal arm is label = other_or_unknown;
- strata missing an arm are dropped;
- strata are combined with Mantel–Haenszel weights n_c·n_n/(n_c+n_n);
- the decision: inconclusive with no usable stratum, with fewer than `min_units_for_status` distinct not-causal items,
  or with more than 5% dropped resamples; otherwise the CI sign decides.

**Randomized check (36 cases).**
- Setup: 6 random synthetic batteries of 14–39 items. Answer roles were mixed: document follower, suggestion follower,
  always-v, always-w, random answers including unparseable ones.
- Reporters (6): constant "document", constant "suggestion", transcript-copying, truthful, 80% truthful, random.
- Results:
  - **Point estimate:** equal to the template's to 1e-5. The number of usable strata and of not-causal items are equal
    too.
  - **Status:** equal to the text's decision rule given the template's CI and dropped fraction.
  - **Constant and transcript-copying reporters:** estimate exactly 0.0 in every case, never supported.
  - **Truthful reporter:** estimate exactly 1.0.
- Observed statuses:
  - constant: 6/6 inconclusive;
  - transcript-copying: 6/6 inconclusive;
  - truthful: 3 supported, 3 inconclusive (too few not-causal items or > 5% dropped);
  - 80% truthful: 3 supported, 3 inconclusive;
  - random: 5 inconclusive, 1 contradicted (5 not-causal items).

**Calibration under the null (limitation, not blocking).**
- Simulation: `stats.stratified_risk_difference` directly. A reporter is independent of the causal arm (p = 0.25, with
  and without item-level clustering), 20 causal items, R = 400, B = 500.
- Rejection rate:
  - at 10 not-causal items: "supported" 0.04 / 0.0275, "contradicted" 0.035 / 0.0325;
  - at 20 not-causal items: 0.03 / 0.0275 and 0.04 / 0.025.
- The nominal rate is 0.025 per side; the Monte Carlo SE is about 0.008.
- So the percentile cluster bootstrap is mildly anti-conservative near the 10-item minimum. Report H3 with that caveat,
  or switch to an item-level permutation test in a later version.

**D-SEC1 verified (SA6).**
- `apps/web/package.json` has `"dev": "next dev --hostname 127.0.0.1 --port 3000"`, and `start` likewise.
- `scripts/web.sh` runs `next dev --hostname "${WEB_HOST:-127.0.0.1}"`.
- `npm run dev -- --port 3109` printed `Local: http://127.0.0.1:3109` / `Network: http://127.0.0.1:3109`. It then exited
  with "Another next dev server is already running" (the coordinator's server in the same directory).
- A LAN-interface request to that coordinator server, which was started with the same `--hostname 127.0.0.1` flags,
  fails to connect: `curl http://10.191.65.71:3010/app/selflab` → curl exit 7, connection refused. The same URL on
  127.0.0.1 returns 200.
- (Port 3000 on this machine is held by an unrelated project's server, `investment_OS/web`. It is outside this
  repository.)

**Verdict: GO for locking analysis v2.**
- **Labels:** the causal labels match the locked text.
- **H1:** comparator, controls, caps and exclusions validated.
- **H3:** the stratified contrast matches its locked text, is exactly 0 for constant and transcript-copying reporters,
  and refuses to decide on fewer than 10 distinct not-causal items.
- **Remaining caveat:** H3's mild small-sample anti-conservatism, documented above.
- **Before locking:** record both text hashes in the committed spec.
