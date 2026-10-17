# SelfLab pre-registrations

Each `*.lock.json` file fixes one run before any of its trials exist. It records:
- the full `ExperimentSpec`;
- the sha256 of the spec and of the deterministic trial plan;
- model provenance: revision, weights fingerprint, chat-template hash, device and dtype;
- the sha256 of every analysis-relevant source file;
- the analysis rule texts;
- the exact command that runs the locked plan.

The protocol:
1. `scripts/selflab/lock_spec.py`, under the model lock, writes the lock file and the run's committed `spec.json`.
   It executes no trial.
2. The lock file is committed to git.
3. `scripts/selflab/run_pilot.py --resume --run-id <id>` re-derives the plan and refuses to run if its hash differs
   from the lock.
4. Results are written to `docs/selflab/results/<id>.json` and committed. Nulls, errors and failed controls are
   included.

Hashes and commit order make deviations detectable. They do not make local files tamper-proof.
