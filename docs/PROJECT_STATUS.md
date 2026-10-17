# Project Status

*Snapshot updated 2026-10-17 on branch `selflab/evidence-workbench`. This file describes the whole project.
Chronology lives in [PROJECT_LEDGER.md](PROJECT_LEDGER.md), findings in [../RESEARCH_LOG.md](../RESEARCH_LOG.md).*

## Project Objective

NeuralScope is an interactive model-auditing workbench with two missions:
1. **Recover the algorithms that learning discovered**: the neural debugger and Math Lab.
2. **Test the claims models make about their own computation**: SelfLab.

SelfLab's question is when an AI's description of its own behaviour or processing tracks independently measurable
facts. A researcher:
1. chats with a local white-box model;
2. selects a claim in its response;
3. runs controlled, pre-committed counterfactual experiments;
4. receives an evidence record that says what is supported, contradicted or unresolved, each statement scoped.

It does not determine consciousness, sentience, suffering or the truth of experiential self-reports, and it produces
no such probability.

## Current Stage

V1 neural debugger plus the first working SelfLab vertical slice (research prototype). It is verified end to end in
the browser on the real model. Experiment A (source
influence vs claimed source) and Experiment B (CSP-1 counterfactual self-prediction on the carry heads) are
implemented. Both have diagnostic (`debug_pilot`) results on Qwen2.5-1.5B-Instruct. The pre-registered Experiment A v2
held-out confirmation was run; its manipulation check failed, so the primary hypothesis is untested, while
label-order invariance (H2) was contradicted. A real single-case record is on the landing page. Experiment C
(blinded state reporting) is protocol-only.

## Current Branch

`selflab/evidence-workbench`, cut from `claude/zen-knuth-hhtjxh` at `9a4e13e`.
- `main` holds only the initial commit (`550aaba`).
- Nothing is pushed or merged. Pushing requires explicit approval.

## Current System Architecture

```
apps/web (Next.js 16, bound to 127.0.0.1)
  /  /research  /docs
  /app  /app/math  /app/selflab
        │  /api/* rewrite
        ▼
apps/api (FastAPI, 127.0.0.1)
  /api/*            existing debugger + Math Lab
  /api/selflab/*    SelfLab router: POSTs need a loopback client with local Host/Origin, or X-NeuralScope-Token
        │
        ▼
NeuralScopeService ─┬─ ModelAdapter (HFDecoderAdapter: native PyTorch hooks; tokenize_messages;
                    │                hook hit counts; generation margins)
                    ├─ ExperimentStore (SQLite runs + safetensors tensors; runs.messages column)
                    └─ JobManager (ONE worker thread: the single shared model)
SelfLabService ─┬─ SelfLabEngine (commit → plan hash → run/cancel/resume, budgets, cost)
                ├─ SelfLabStore (selflab_* tables; append-only trials/evidence enforced by SQLite triggers)
                ├─ templates: source_influence_case, source_influence_battery, csp1_carry
                │             (registry protocol)
                ├─ measure.py (full continuation log-likelihood, option log-probs, aligned patching)
                ├─ stats.py (cluster bootstrap, sign-flip, Wilson, Holm/BH, scores),
                │  parsing.py (conservative parsers)
                ├─ export.py (reproduction record + privacy preview), investigator.py (deterministic rules)
                └─ datasets: source_corpus (invented entities, 4 families, F4 held out), carry_items
```

Only one process holds real model weights at a time (`scripts/model_lock.py`, fcntl lock).

## Capabilities Implemented

**Debugger, kept intact:**
- generation and capture;
- token × layer explorer, Logit Lens, attention, DLA;
- head/MLP ablation, patching, steering, sweeps, diffs;
- persistence, Math Lab;
- marketing site with a genuine recorded hero trace.

**SelfLab:**
- **Chat.** Chat with the local target, with an optional document and suggestion. Every segment gets its exact token
  span and a delivery receipt ("proves delivery, not causal use").
- **Cases.** Select a claim span → INVESTIGATE CLAIM → a case with the claim, run, context, access level,
  participants and provenance (model revision, chat-template hash, weights fingerprint, library versions, code
  commit).
- **Templates.** Typed templates only; there is no arbitrary code path. COMMIT hashes the spec and the full
  randomised plan before any trial runs.
- **Runs.** RUN uses the single worker. Every trial is persisted, errors included. CANCEL and RESUME work; resume
  refuses on any plan, spec or model/weights mismatch and never re-randomises.
- **Evidence.** Kind, test status and scope are separate fields. Items link to trials, receipts and Lab runs. No
  automatic "validated" badge exists.
- **OPEN IN LAB.** Deterministic replay into `/app?run=` with a reproduced flag. For CSP-1 trials it replays the
  clean run and the intervention child.
- **EXPORT EVIDENCE.** Reproduction record with a privacy preview and an option to exclude free text.
- **Deterministic suggestions** ("What would test this explanation?").
- **API-only mode.** Disabled by default. It inspects pasted data only, with no provider calls.
- **Landing page.** Headline "Inspect the model. Audit its claims." plus honest feature-status markings.

## Work Completed to Date

- 2026-09-29: V1 debugger, Math Lab, E1/E2 carry-head findings.
- 2026-09-30: SelfLab audit and plan (design only).
- 2026-10-17:
  - baseline re-verified, and E2 reproduced exactly on CPU;
  - pre-registered held-out run executed: H1/H3/H8 `control_failed`, H2 contradicted, H4–H6 supported;
  - real single-case record (`slx_574e679faed6`, clean commit `6a23710`) published on the landing page;
  - SelfLab implemented by six parallel workstreams, integrated by the coordinator;
  - 10 validation defects found and fixed (one high-severity network exposure);
  - Experiment A and CSP-1 diagnostic pilots run;
  - Experiment A protocol v2 locked for a held-out run.

Details are in PROJECT_LEDGER.md.

## Important Files and Artifacts

- **SelfLab code:** `packages/neuralscope/neuralscope/selflab/**` and `apps/api/neuralscope_api/selflab_routes.py`.
- **SelfLab interface:** `apps/web/app/app/selflab/**`, `apps/web/components/selflab/**`, `apps/web/lib/api/selflab.ts`.
- **Contract:** `docs/selflab/INTERFACES.md`, schema `selflab-1`.
- **Design and research docs:** `docs/selflab/PRIOR_ART_MATRIX.md`, `protocols/EXP_{A,B,C}*.md`,
  `INSTRUMENTATION_NOTES.md`, `VALIDATION_REPORT.md`, `COMMERCIAL_HYPOTHESIS.md`.
- **Results:** `docs/selflab/results/*.json` (pilot summaries) and `docs/selflab/prereg/*.lock.json`
  (pre-registrations).
- **Scripts:** `scripts/selflab/run_pilot.py`, `scripts/selflab/lock_spec.py`, `scripts/model_lock.py`,
  `scripts/smoke_selflab_browser.cjs`.
- **Local data (gitignored):** `data/selflab/pilots/<run_id>/{spec.json,trials.jsonl,analysis.json,lab/}` and
  `data/repro/`.

## Data Sources and Provenance

- **Models:** Qwen/Qwen2.5-1.5B-Instruct (revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) and
  Qwen/Qwen2.5-0.5B-Instruct (`7ae557604adf67be50417f59c2c2f167def9a775`). Both Apache-2.0, from the HF Hub cache.
- **Experiment A corpus:** generated (`source_corpus@1`): invented entities with seeded random facts. No external data.
- **Carry items:** generated by the E1 generator logic (reproduces the E1 problem set exactly); E1/E2 problems are
  excluded from CSP-1 by default.
- **Records:** every SelfLab record carries a model/tokenizer revision, chat-template sha256, weights fingerprint,
  device and dtype, library versions, code commit and dirty flag, seed, and decoding configuration.

## Decisions and Rationale

- **Build above the microscope.** The SelfLab deletion map was not executed, because the prompt did not authorise it
  and the debugger is the "open in Lab" target. NNsight was not adopted because no blocker requires it.
- **Template-driven, typed experiments only.** No LLM investigator is configured, so suggestions are deterministic
  rules.
- **CPU fp32 is the reference device.** MPS flips rare near-tie answers.
- **Evidence fields stay separate.** Statuses are scoped. `debug_pilot` outcomes are capped at inconclusive.
- **Experiment A v2 analysis rules** were decided after validation, before any held-out data existed:
  - visible-not-causal trials are scored;
  - `hint_induced_abstention` is separated;
  - H1 is compared against the better of the majority label and the prompt heuristic;
  - the manipulation check requires both document directions.
- **Development servers bind to 127.0.0.1.** The validation found the Next dev server forwarding LAN requests to the
  localhost-only API.

## Verification and Test Results

All results below were run on 2026-10-17.

| Check | Command | Result |
|---|---|---|
| Fast suite | `.venv/bin/python -m pytest -m "not slow" -q` | 721 passed, 0 failed |
| Held-out Experiment A v2 (locked plan `9d5b0ef0…`) | command in the lock file | 120/120 trials, 0 errors; plan hash verified; manipulation check failed |
| Slow suite (real Qwen2.5-0.5B, under lock) | `scripts/model_lock.py -- … pytest -m slow -q` | 8 passed |
| Web | `npm run test` / `npm run lint` / `npm run typecheck` | 149 passed / clean / clean |
| Landing page at phone width (390 px) | Playwright check of `scrollWidth` | no horizontal overflow; 0 page errors |
| E2 reproduction | `carry_patch.py --n 40 --seed 0`, CPU | identical to the recorded summary |
| Browser smoke, fixture mode (validation workstream, DEMO API, before the UI fixes) | `scripts/smoke_selflab_browser.cjs` | 10 pass; 2 known defects, later fixed |
| Browser smoke, REAL mode (Qwen2.5-1.5B, CPU; web 127.0.0.1:3010, API 127.0.0.1:8010) | `WEB=… API=… SMOKE_MAX_TRIALS=6 node scripts/smoke_selflab_browser.cjs` | 12 pass, 0 fail, 0 page errors (`docs/selflab/screenshots/selflab-real-smoke-summary.json`) |
| Real case via the UI | chat → claim → COMPARE CONTEXTS on Qwen2.5-1.5B (`docs/selflab/results/exports/slx_b01122c95783_case_export.json`) | answer depends on the document: supported (n=1); claim specificity: contradicted; replay reproduced |
| v2 analysis validation (independent re-implementation, puppet model, null simulation) | `tests/test_selflab_validation_v2.py` | 59 passed; GO for lock; H3 slightly anti-conservative near its 10-item minimum |
| LAN exposure check | `curl http://<LAN-IP>:3010/app/selflab` | connection refused (exit 7); loopback returns 200 |

The pilot numbers are in RESEARCH_LOG.md (2026-10-17) and `docs/selflab/results/`.

## Known Failures and Limitations

- **Small pilots.** Experiment A has 20 items; CSP-1 has 40 held-out problems. Debug-pilot statuses are capped at
  inconclusive.
- **Experiment A v1 approximation.** Document tracking under a conflicting hint borrows the no-hint pair. A
  `doc_alt_hint_conflict` condition is the v2 candidate.
- **H3 may be untestable.** It needs visible-but-not-causal trials, and Qwen2.5-1.5B always followed the document
  (0 such trials in the pilot and in the held-out run).
- **The held-out confirmation is uninformative for H1.** Its manipulation check failed in family F4 (doc_v 0.70,
  doc_alt 0.75 < 0.8), and F4 is now spent.
- **CSP-1 comparators.** The different-model observer was not run, and the joint control-head ablation is not a null
  (see RESEARCH_LOG).
- **Experiment C** is not implemented.
- **Untested at scale.** The case-local template was tested on the demo model and the UI flow, not in a statistical
  run.
- **Not verified:** cross-hardware bitwise determinism.

## Risks and Technical Debt

- **Novelty and usefulness are not established.** Silico, causalab and the OpenAI monitorability evals overlap at the
  workflow and method level.
- **Two duplicated hook layers.** NNsight duplicates our hook layer (kept deliberately).
- **Unguarded reads.** GET endpoints, including exports with transcripts, have no guard and rely on localhost
  binding.
- **The weights fingerprint is a strided sample, not a full hash.**
- **Pilot data is local.** `data/selflab/pilots` holds about 300 MB and is not committed. Committed summaries carry
  hashes, but they do not make local logs tamper-proof.

## Current Blockers

None for local work. These need the user's approval: pushing or merging the branch; any paid API or cloud compute;
an NDIF key for larger models.

## Exact Next Actions

1. Design Experiment A v3 on the **discovery** split only:
   - calibrate family wording so the document manipulation passes its check;
   - add `doc_alt_hint_conflict`;
   - pre-register per-order and order-averaged accuracy.

   Lock it with `scripts/selflab/lock_spec.py`, then run a **new** held-out family. F4 is now used.
2. Then:
   - add `doc_alt_hint_conflict` (Experiment A v3);
   - add a different-model observer for CSP-1 (Qwen2.5-0.5B under the lock);
   - implement Experiment C following `protocols/EXP_C_blinded_state.md`.
