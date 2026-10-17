# Project Ledger

Append-only chronology. Never rewrite an old entry, except to correct a demonstrable factual error, and document any
such correction in the entry itself.

---

## 2026-10-17 22:40 (Europe/London) · `selflab/evidence-workbench` · SelfLab evidence workbench v1 + locked held-out protocol

**Branch.** `selflab/evidence-workbench`, cut from `claude/zen-knuth-hhtjxh` at `9a4e13e`. Nothing pushed.

**Project stage.** V1 debugger plus the first working SelfLab vertical slice (research prototype).

**Objective.** Build an evidence-producing workflow above the existing microscope:
1. a real local model makes a claim;
2. the user selects it;
3. a committed counterfactual experiment runs;
4. the result is measured evidence that opens in the existing Lab.

Also: implement Experiment A (source influence vs claimed source) and CSP-1, run diagnostic pilots, and lock a
held-out confirmation, all without deleting the debugger, Math Lab, website or E1/E2 data.

**Files and systems changed.**
- **New Python package:** `packages/neuralscope/neuralscope/selflab/`
  - schemas, registry, provenance, compose, measure, stats, parsing, store, engine, service, export, investigator;
  - datasets: `source_corpus`, `carry_items`;
  - templates: `source_influence`, `csp1`.
- **API router:** `apps/api/neuralscope_api/selflab_routes.py`, mounted in `main.py`.
- **Instrumentation, additive:**
  - `tokenize_messages`;
  - hook hit counts and generation margins;
  - `sequence_logits`;
  - control-token rejection in message content.
- **Lab service and store:** `create_run(messages=…)`, plus a `runs.messages` column.
- **Web:**
  - the `/app/selflab` workspace;
  - SelfLab API client and types;
  - SelfLab nav entries;
  - landing headline and subheadline, feature-status list, SelfLab section;
  - Lab run-context panel;
  - root metadata and footer.
- **Network binding:** the web dev/start servers and `scripts/web.sh` bind to 127.0.0.1.
- **Scripts:** `scripts/model_lock.py`, `scripts/selflab/{run_pilot,lock_spec}.py`,
  `scripts/smoke_selflab_browser.cjs`.
- **Tests:** about 25 new test files.
- **Docs and repo files:**
  - `docs/selflab/`: INTERFACES, PRIOR_ART_MATRIX, protocols A/B/C, INSTRUMENTATION_NOTES, VALIDATION_REPORT,
    COMMERCIAL_HYPOTHESIS, results, prereg, screenshots;
  - README SelfLab section;
  - RESEARCH_LOG entry;
  - Makefile targets;
  - `.gitignore` additions for local data.

**What was completed.**
- **Baseline audit and reproduction.**
  - Fast suite 300 passed; slow 5 passed; web 83 passed; lint and typecheck clean.
  - E2 re-run on CPU fp32 matches the recorded summary exactly.
- **Parallel build.** Six workstreams ran as real parallel sub-agents on non-overlapping paths, under a frozen
  interface (`selflab-1`), with the coordinator integrating. Every real-model job ran under one fcntl model lock.
- **Workflow verified on the real model in the browser.** Chat → claim → case → commit (spec and plan hashes) → run →
  evidence → OPEN IN LAB (replay reproduced) → export preview.
- **Experiment A pilot** (debug split, 120 trials): see RESEARCH_LOG.
- **CSP-1 pilot** (180 trials): see RESEARCH_LOG.
- **Validation.** 20 defects were found by independent validation and 20 fixed or confirmed. They include:
  - LAN exposure of the localhost API through the Next dev server;
  - chat-control tokens forging turns;
  - resume accepting different weights;
  - H3 rules that could never be supported, or could be supported by a constant reporter;
  - an over-claiming case-template alternative.
- **Analysis v2** decided and validated, then the held-out run locked:
  `docs/selflab/prereg/expA_heldout_v2_1p5b_F4_s200.lock.json` (spec `cba95e9c…`, plan `9d5b0ef0…`).

**Decisions made and why.**
- **SelfLab deletion map not executed.** It was not authorised, and the debugger is the OPEN IN LAB target.
- **No NNsight migration.** No blocker requires one.
- **CPU fp32 is the reference device.** MPS flipped 1/32 near-tie answers.
- **Templates only.** Experiments run only through typed templates, never arbitrary code. Investigator suggestions
  come from deterministic rules, because no LLM investigator is approved.
- **Debug-pilot statuses are capped at inconclusive.**
- **The v2 rules were fixed before any held-out data existed.**
- **The real-model app ran on ports 3010/8010.** Ports 3000/8000 belong to the user's other projects and were left
  untouched.

**Tests and verification actually executed (2026-10-17).**
- `.venv/bin/python -m pytest -m "not slow" -q` → 721 passed.
- `.venv/bin/python scripts/model_lock.py -- .venv/bin/python -m pytest -m slow -q` → 8 passed.
- `cd apps/web && npm run test` → 148 passed. `npm run lint` → exit 0. `npm run typecheck` → exit 0.
- REAL-mode browser smoke → 12 pass, 0 fail, 0 page errors.
- Independent v2 validation (`tests/test_selflab_validation_v2.py`) → 59 passed. It returned GO.
- E2 reproduction → identical.
- The LAN request to the dev server was refused.

**Results.**
- **Experiment A pilot.** The self-report beats a majority guess (+0.14 [0.05, 0.22]) but not a prompt-only
  heuristic (−0.07 [−0.16, 0.01]). It sits far below a transcript observer (−0.36). Label-order consistency was 0.675,
  and the model misattributed 9/9 suggestion-induced abstentions to the document.
- **CSP-1 pilot.** The ablation flips 97% of carry answers. The model predicts almost no change and scores −0.98 bits
  vs an outsider [−1.38, −0.58].
- **Real case.** The answer depended on the document (supported, n = 1). The claim/report was not specific to
  document-caused answers (contradicted).

**Failures and unresolved uncertainty.**
- The pilots are small.
- H3 is likely untestable on this model, because it always followed the document (0 visible-not-causal trials).
- H3 is mildly anti-conservative near its minimum.
- The joint L22.H8+H9 "control" ablation is not a null (documented in RESEARCH_LOG).
- CSP-1 has no different-model observer.
- Experiment C is not implemented.
- The case template has not been tested statistically.
- Cross-hardware determinism has not been tested.
- The landing-page recorded example is still pending; it is filled from a clean-commit case run next.

**Effect on the overall project.** NeuralScope now has a working, test-covered evidence workflow above the
debugger. Its first measurements say that small-model source self-reports are unreliable and no better than
outsiders. These are pilot-scale results, and the held-out confirmation is pre-registered.

**Next intended step.** Execute the locked held-out run with the command in its lock file, and record its results
exactly.

---

## 2026-10-17 22:55 (Europe/London) · `selflab/evidence-workbench` · held-out confirmation executed + real recorded example

**Branch.** `selflab/evidence-workbench`, parent `6a23710`. Nothing pushed.

**Project stage.** SelfLab vertical slice verified on the real model. Experiment A's pre-registered held-out run has
been executed.

**Objective.** Execute the locked held-out confirmation exactly as committed, record its results as they came out
(including the failed control), and publish one genuine real-model case record on the landing page.

**Files and systems changed.**
- `docs/selflab/results/expA_heldout_v2_1p5b_F4_s200.json` (new)
- `docs/selflab/results/exports/slx_574e679faed6_case_export.json` (new)
- `apps/web/lib/selflab-example.ts`: filled verbatim from that export
- `apps/web/components/marketing/SelfLabSection.tsx`: the results list no longer squeezes columns; the provenance
  line wraps, which fixes phone-width overflow
- `apps/web/__tests__/selflab/landing.test.tsx`
- landing screenshots
- RESEARCH_LOG, README, PROJECT_STATUS, `docs/selflab/FINAL_REPORT_2026-10-17.md`

**What was completed.**
- **Held-out run.** The command from the lock file ran under the model lock. It re-derived the identical plan
  (`9d5b0ef0…`) and completed 120/120 trials with 0 errors.
- **Real case.** Re-run at clean commit `6a23710` with identical numbers (the third identical run), exported, and
  published on the landing page.

**Decisions made and why.**
- **The failed control stands.** Results are reported exactly under the locked rules, with no re-run or tuning: a
  failed pre-registered control means the primary hypothesis is untested.
- **F4 is spent.** Any new held-out run needs a new family and a new lock.

**Tests and verification actually executed.**
- Held-out run: complete, `EXIT 0`.
- `cd apps/web && npm run test` → 149 passed. `npm run lint` → exit 0. `npm run typecheck` → exit 0.
- Playwright landing check: 0 page errors, no horizontal overflow at 390 px (`scrollWidth` 390).
- `.venv/bin/python -m pytest -m "not slow" -q` → 721 passed, re-run before this commit.
- The slow suite was not re-run after this step (no Python code changed). Its last run was 8 passed, at `6a23710`.

**Results.**

| Hypothesis | Status | Detail |
|---|---|---|
| Manipulation check (pre-registered) | **failed** | doc_v 0.70, doc_alt 0.75, threshold 0.8 |
| H1, H3, H8 | `control_failed` | untested |
| H2 (label-order invariance) | `contradicted` | 0.475 [0.39, 0.56] |
| H4, H5, H6 | `supported` | +11.3 nats; +8.9 nats; 20/20 |

The real case shows answer dependence on the document as supported (n = 1) and claim specificity as contradicted.

**Failures and unresolved uncertainty.**
- The primary faithfulness question remains untested on held-out data.
- The descriptive accuracy (0.92) is not interpretable, because the manipulation check failed.

**Effect on the overall project.** The pipeline demonstrably refuses to over-interpret: the positive control caught
a family where the manipulation does not take. The report-instability finding (H2) replicated on held-out data.

**Next intended step.** Design Experiment A v3 on the discovery split (family calibration, `doc_alt_hint_conflict`,
per-order metrics). Lock it, then run a new held-out family.
