"""SelfLab job engine: commit → run → (cancel | budget stop | resume) → analyse.

Contract (docs/selflab/INTERFACES.md §2, §5):

* **Commit before any trial runs.** The spec is validated through the template's ``params_model``
  (defaults filled in, so the commitment pins them), the plan is built (``build_plan``: deterministic,
  no forward passes), and ``spec_sha256 = sha256_json(spec.model_dump(mode="json"))`` and
  ``plan_sha256 = sha256_json([p.model_dump(mode="json") for p in plan])`` are persisted together with
  the provenance in a ``CommittedSpec``. The DB makes the commitment immutable.
* **Run** on the existing single-worker ``JobManager`` (passed in — this module never creates a second
  worker, so device-heavy work stays serialised with sweeps and benchmarks). The worker iterates the
  stored plan, skips indices already stored, checks cooperative cancel and the budget between trials,
  and persists EVERY trial immediately (template exceptions become ``status="error"`` trials).
* **Budget** (``max_trials``, ``max_seconds``) is cumulative over the experiment's lifetime (all
  sessions): it is part of the commitment, so a resume cannot extend it. A budget stop analyses the
  trials completed so far and marks the analysis ``complete=False``.
* **Resume** re-derives the plan from the stored committed spec and REFUSES (``ConflictError``) when the
  plan hash differs, when the stored spec no longer hashes to ``spec_sha256``, or when the loaded model
  differs from the committed model provenance (id, revision, tokenizer, chat template, dtype, device, demo
  flag, and ``weights_fingerprint`` whenever both records carry one — older records lack it, which is
  noted). Completed trials are never re-run or re-randomised: the trial table is append-only at the
  database level.
* **Cancel** before the worker reaches the job → ``cancelled`` when it does; while running → stops
  between trials. Trial ids are ``f"{experiment_id}:{index:05d}"``.
"""

from __future__ import annotations

import json
import resource
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from neuralscope.selflab.provenance import code_state, model_provenance, now_iso, run_provenance, sha256_json
from neuralscope.selflab.registry import AnalysisResult, RunContext, get_template
from neuralscope.selflab.schemas import (
    Case,
    CommittedSpec,
    EvidenceItem,
    ExperimentRecord,
    ExperimentSpec,
    Trial,
    TrialPlanItem,
)
from neuralscope.selflab.store import (
    ConflictError,
    SelfLabError,
    SelfLabStore,
    UnavailableError,
    trial_id,
)

ENGINE_VERSION = "selflab-engine-1"
SELFLAB_MAX_PROMPT_TOKENS = 1024  # hard cap on Budget.max_prompt_tokens accepted at commit (§5 bounded inputs)
MAX_PLAN_SIZE = 20000
TERMINAL = frozenset({"done", "cancelled", "error"})
ACTIVE = frozenset({"queued", "running", "cancelling"})
RESUMABLE = frozenset({"cancelled", "error"})
# Model-provenance fields that must match the commitment for a resume (mixing models in one table is refused).
_PROVENANCE_MATCH_FIELDS = ("model_id", "revision", "tokenizer_id", "tokenizer_revision", "chat_template_sha256",
                            "dtype", "device", "is_demo")


def new_experiment_id() -> str:
    return f"slx_{uuid.uuid4().hex[:12]}"


def spec_hash(spec: ExperimentSpec) -> str:
    return sha256_json(spec.model_dump(mode="json"))


def plan_hash(plan: list[TrialPlanItem]) -> str:
    return sha256_json([p.model_dump(mode="json") for p in plan])


def template_access_levels(template: Any) -> tuple[str, ...]:
    """Access levels a template can run on. Optional template attribute ``access_levels``; default white-box only."""
    return tuple(getattr(template, "access_levels", ("white_box_local",)))


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)  # macOS reports bytes, Linux kilobytes


def accelerator_memory(device: str | None) -> dict[str, Any] | None:
    """Allocated device memory for the adapter's device, if it is MPS or CUDA (None for CPU)."""
    dev = str(device or "")
    try:
        import torch

        if dev.startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"device": "mps", "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                    "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}
        if dev.startswith("cuda") and torch.cuda.is_available():
            return {"device": dev, "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    except Exception:  # pragma: no cover - backend specific
        return None
    return None


def _reported_forward_passes(trial: Trial) -> int | None:
    m = trial.measurements or {}
    for key in ("n_forward_passes", "forward_passes"):
        v = m.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    c = m.get("cost")
    if isinstance(c, dict):
        for key in ("n_forward_passes", "forward_passes"):
            v = c.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
    return None


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


class SelfLabEngine:
    """Commits, runs, cancels and resumes SelfLab experiments. One instance per store/model."""

    def __init__(self, store: SelfLabStore, jobs: Any, service: Any = None, adapter: Any = None):
        self.store = store
        self.jobs = jobs  # neuralscope.experiments.jobs.JobManager (single worker thread)
        self.service = service  # NeuralScopeService (Lab runs), passed to templates via RunContext
        self.adapter = adapter if adapter is not None else getattr(service, "adapter", None)
        self._cancel: set[str] = set()
        self._active: dict[str, str] = {}  # experiment id → job id (this process)
        self._lock = threading.RLock()

    # ================================================================== helpers
    def _require_adapter(self) -> Any:
        if self.adapter is None:
            raise UnavailableError("no model is loaded")
        return self.adapter

    @staticmethod
    def _template(template_id: str) -> Any:
        try:
            return get_template(template_id)
        except KeyError as e:
            raise SelfLabError(f"experiment template {template_id!r} is not registered on this server") from e

    def _case_for(self, case_id: str | None) -> Case | None:
        return self.store.require_case(case_id) if case_id else None

    def _build_plan(self, template: Any, spec: ExperimentSpec, case: Case | None) -> list[TrialPlanItem]:
        try:
            raw = template.build_plan(spec, case, self._require_adapter())
        except (ValueError, TypeError) as e:  # invalid params for this case → 400 with the template's reason
            raise SelfLabError(f"template {spec.template} cannot plan this spec: {e}") from e
        plan = [p if isinstance(p, TrialPlanItem) else TrialPlanItem.model_validate(p) for p in list(raw)]
        if not plan:
            raise SelfLabError(f"template {spec.template} produced an empty plan")
        if len(plan) > MAX_PLAN_SIZE:
            raise SelfLabError(f"plan has {len(plan)} trials; the limit is {MAX_PLAN_SIZE}")
        idx = [p.index for p in plan]
        if len(set(idx)) != len(idx) or min(idx) < 0:
            raise SelfLabError("plan indices must be unique and non-negative")
        return sorted(plan, key=lambda p: p.index)

    def _normalise_spec(self, template: Any, spec: ExperimentSpec) -> ExperimentSpec:
        pm = getattr(template, "params_model", None)
        if pm is not None:
            try:
                params = pm.model_validate(spec.params)
            except ValidationError as e:
                raise SelfLabError(f"invalid params for template {spec.template}: {e}") from e
            dumped = params.model_dump(mode="json") if isinstance(params, BaseModel) else dict(params)
            spec = spec.model_copy(update={"params": dumped})
        return ExperimentSpec.model_validate(spec.model_dump(mode="json"))

    # ================================================================== commit
    def commit(self, spec: ExperimentSpec | dict, case: Case | None = None) -> ExperimentRecord:
        """Validate, plan, hash and persist the commitment. No trial runs here."""
        if not isinstance(spec, ExperimentSpec):
            try:
                spec = ExperimentSpec.model_validate(spec)
            except ValidationError as e:
                raise SelfLabError(f"invalid experiment spec: {e}") from e
        template = self._template(spec.template)
        if case is None and spec.case_id:
            case = self.store.require_case(spec.case_id)
        if case is not None:
            if spec.case_id is None:
                spec = spec.model_copy(update={"case_id": case.id})
            elif spec.case_id != case.id:
                raise SelfLabError(f"spec.case_id {spec.case_id!r} does not match case {case.id!r}")
        if getattr(template, "requires_case", False) and case is None:
            raise SelfLabError(f"template {spec.template} requires a case (spec.case_id)")
        if case is not None and case.access_level not in template_access_levels(template):
            raise SelfLabError(f"template {spec.template} needs white-box access to the target; case {case.id} is "
                               f"'{case.access_level}' (pasted data only) — white-box templates are refused for it")
        if spec.budget.max_prompt_tokens > SELFLAB_MAX_PROMPT_TOKENS:
            raise SelfLabError(f"budget.max_prompt_tokens {spec.budget.max_prompt_tokens} exceeds the SelfLab limit "
                               f"{SELFLAB_MAX_PROMPT_TOKENS}")
        adapter = self._require_adapter()
        spec = self._normalise_spec(template, spec)
        plan = self._build_plan(template, spec, case)
        prov = run_provenance(adapter, decoding={"strategy": "greedy", "max_new_tokens": spec.budget.max_new_tokens,
                                                 "kv_cache": True}, seed=spec.seed)
        committed = CommittedSpec(spec=spec, spec_sha256=spec_hash(spec), plan_sha256=plan_hash(plan),
                                  n_planned=len(plan), committed_at=now_iso(), provenance=prov)
        eid = new_experiment_id()
        is_fixture = bool(getattr(adapter.info, "is_demo", False)) or bool(case.is_fixture if case else False)
        msg = f"committed: {len(plan)} planned trials"
        if len(plan) > spec.budget.max_trials:
            msg += f" (budget max_trials={spec.budget.max_trials} will stop the run early)"
        rec = ExperimentRecord(id=eid, case_id=case.id if case else None, committed=committed, status="committed",
                               message=msg, created_at=committed.committed_at, is_fixture=is_fixture,
                               cost={"seconds": 0.0, "generated_tokens": 0, "sessions": 0})
        self.store.insert_experiment(rec, plan)
        if case is not None:
            self.store.add_case_experiment(case.id, eid)
        self.store.log_event(eid, "committed", {
            "spec_sha256": committed.spec_sha256, "plan_sha256": committed.plan_sha256, "n_planned": len(plan),
            "code_commit": prov.code_commit, "code_dirty": prov.code_dirty, "engine_version": ENGINE_VERSION})
        return rec

    # ================================================================== state transitions
    def run(self, experiment_id: str) -> ExperimentRecord:
        """Enqueue a committed experiment on the single model worker."""
        with self._lock:
            rec = self.store.transition(experiment_id, {"committed"}, status="queued",
                                        message="queued on the model worker")
            self.store.log_event(experiment_id, "queued", {})
            self._submit(experiment_id)
        return rec

    def cancel(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            rec = self.store.require_experiment(experiment_id)
            if rec.status == "committed":
                rec = self.store.transition(experiment_id, {"committed"}, status="cancelled", finished_at=now_iso(),
                                            message="cancelled before it was queued (no trials ran)")
                self.store.log_event(experiment_id, "cancelled", {"before_queue": True})
                return rec
            if rec.status == "cancelling":
                return rec
            if rec.status in ("queued", "running"):
                self._cancel.add(experiment_id)
                rec = self.store.transition(experiment_id, {"queued", "running"}, status="cancelling",
                                            message="cancel requested: stops between trials")
                self.store.log_event(experiment_id, "cancel_requested", {"from": rec.status})
                return rec
            raise ConflictError(f"experiment {experiment_id} is '{rec.status}' and cannot be cancelled")

    def verify_resumable(self, experiment_id: str) -> dict[str, Any]:
        """Re-derive the plan from the committed spec and compare hashes and model provenance.

        Returns a report; raises ConflictError (after logging the refusal) on any mismatch.
        """
        rec = self.store.require_experiment(experiment_id)
        c = rec.committed
        template = self._template(c.spec.template)
        case = self._case_for(rec.case_id)
        problems: list[str] = []
        recomputed_spec = spec_hash(c.spec)
        if recomputed_spec != c.spec_sha256:
            problems.append(f"stored spec hashes to {recomputed_spec[:16]}…, committed {c.spec_sha256[:16]}…")
        stored_plan_hash = plan_hash(self.store.get_plan(experiment_id))
        if stored_plan_hash != c.plan_sha256:
            problems.append(f"stored plan hashes to {stored_plan_hash[:16]}…, committed {c.plan_sha256[:16]}…")
        try:
            rederived = plan_hash(self._build_plan(template, c.spec, case))
        except SelfLabError as e:
            rederived = None
            problems.append(f"plan could not be re-derived: {e}")
        if rederived is not None and rederived != c.plan_sha256:
            problems.append(f"re-derived plan hashes to {rederived[:16]}…, committed {c.plan_sha256[:16]}… "
                            "(template, corpus or tokenizer changed)")
        model_diffs: dict[str, Any] = {}
        notes: list[str] = []
        if c.provenance is not None:
            cur = model_provenance(self._require_adapter())
            for f in _PROVENANCE_MATCH_FIELDS:
                a, b = getattr(c.provenance.model, f, None), getattr(cur, f, None)
                if a != b:
                    model_diffs[f] = {"committed": a, "loaded": b}
            # weights identity (D-E1): compared whenever BOTH sides carry a fingerprint
            fa = getattr(c.provenance.model, "weights_fingerprint", None)
            fb = getattr(cur, "weights_fingerprint", None)
            if fa is not None and fb is not None:
                if fa != fb:
                    model_diffs["weights_fingerprint"] = {"committed": fa, "loaded": fb}
            else:
                notes.append("weights_fingerprint not compared: missing in the "
                             + ("committed record" if fa is None else "loaded model's provenance")
                             + f"; identity rests on {', '.join(_PROVENANCE_MATCH_FIELDS)}")
            if model_diffs:
                problems.append(f"loaded model differs from the committed model provenance: {sorted(model_diffs)}")
        else:
            notes.append("no model provenance was committed: the loaded model could not be compared")
        report = {"spec_sha256": c.spec_sha256, "plan_sha256": c.plan_sha256, "rederived_plan_sha256": rederived,
                  "model_diffs": model_diffs, "notes": notes, "problems": problems, "ok": not problems}
        if problems:
            self.store.log_event(experiment_id, "resume_refused", report)
            raise ConflictError("resume refused: " + "; ".join(problems))
        return report

    def resume(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            rec = self.store.require_experiment(experiment_id)
            if rec.status not in RESUMABLE:
                raise ConflictError(f"experiment {experiment_id} is '{rec.status}'; resume needs one of "
                                    f"{sorted(RESUMABLE)}")
            report = self.verify_resumable(experiment_id)
            commit, dirty = code_state()
            stored = len(self.store.trial_indices(experiment_id))
            rec = self.store.transition(experiment_id, set(RESUMABLE), status="queued", error=None, finished_at=None,
                                        message=f"resumed: {stored} stored trials are kept; queued")
            self.store.log_event(experiment_id, "resumed", {"n_stored_trials": stored, "plan_sha256_verified":
                                                            report["plan_sha256"], "code_commit": commit,
                                                            "code_dirty": dirty, "identity_notes": report["notes"]})
            self._cancel.discard(experiment_id)
            self._submit(experiment_id)
        return rec

    def recover_interrupted(self) -> list[str]:
        """Mark experiments left queued/running by a previous process as ``error`` (resumable)."""
        out = []
        with self._lock:
            for rec in self.store.list_experiments(limit=100000):
                if rec.status in ACTIVE and rec.id not in self._active:
                    self.store.update_experiment(
                        rec.id, status="error", finished_at=now_iso(),
                        error="interrupted: the server process ended while this experiment was active",
                        message="interrupted; resume to continue (stored trials are kept)")
                    self.store.log_event(rec.id, "interrupted", {"previous_status": rec.status})
                    out.append(rec.id)
        return out

    def _submit(self, experiment_id: str) -> None:
        job = self.jobs.submit("selflab_experiment", lambda p: self._execute(experiment_id, p),
                               meta={"experiment_id": experiment_id})
        self._active[experiment_id] = job["id"]

    def job_id(self, experiment_id: str) -> str | None:
        return self._active.get(experiment_id)

    def wait(self, experiment_id: str, timeout: float = 120.0, poll: float = 0.02) -> ExperimentRecord:
        """Block until the experiment reaches a terminal state (tests and scripts)."""
        t0 = time.time()
        while True:
            rec = self.store.require_experiment(experiment_id)
            if rec.status in TERMINAL and experiment_id not in self._active:
                return rec
            if time.time() - t0 > timeout:
                raise TimeoutError(f"experiment {experiment_id} still '{rec.status}' after {timeout}s")
            time.sleep(poll)

    def run_now(self, experiment_id: str, progress: Callable[[float, str], None] | None = None) -> ExperimentRecord:
        """Run synchronously in the CALLER's thread (scripts/tests only — the API always uses the JobManager)."""
        with self._lock:
            rec = self.store.require_experiment(experiment_id)
            if rec.status == "committed":
                self.store.transition(experiment_id, {"committed"}, status="queued", message="queued (synchronous)")
            elif rec.status in RESUMABLE:
                report = self.verify_resumable(experiment_id)
                self.store.transition(experiment_id, set(RESUMABLE), status="queued", error=None, finished_at=None,
                                      message="resumed (synchronous)")
                self.store.log_event(experiment_id, "resumed", {"synchronous": True,
                                                                "identity_notes": report["notes"]})
            else:
                raise ConflictError(f"experiment {experiment_id} is '{rec.status}'")
            self._active[experiment_id] = "sync"
        self._execute(experiment_id, progress or (lambda f, m: None))
        return self.store.require_experiment(experiment_id)

    # ================================================================== worker
    def _execute(self, experiment_id: str, progress: Callable[[float, str], None]) -> dict[str, Any]:
        try:
            return self._execute_inner(experiment_id, progress)
        except Exception as e:  # engine-level failure: trials already stored are kept
            tb = traceback.format_exc(limit=6)
            try:
                self.store.update_experiment(experiment_id, status="error", finished_at=now_iso(),
                                             error=f"{type(e).__name__}: {e}", message="engine error")
                self.store.log_event(experiment_id, "engine_error", {"error": f"{type(e).__name__}: {e}",
                                                                     "traceback": tb})
            except Exception:  # pragma: no cover
                pass
            return {"experiment_id": experiment_id, "status": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            with self._lock:
                self._active.pop(experiment_id, None)
                self._cancel.discard(experiment_id)

    def _execute_inner(self, experiment_id: str, progress: Callable[[float, str], None]) -> dict[str, Any]:
        with self._lock:
            rec = self.store.require_experiment(experiment_id)
            if experiment_id in self._cancel or rec.status == "cancelling":
                self.store.update_experiment(experiment_id, status="cancelled", finished_at=now_iso(),
                                             message="cancelled before start (no trials ran in this session)")
                self.store.log_event(experiment_id, "cancelled", {"before_start": True})
                return {"experiment_id": experiment_id, "status": "cancelled"}
            if rec.status != "queued":
                self.store.log_event(experiment_id, "worker_skipped", {"status": rec.status})
                return {"experiment_id": experiment_id, "status": rec.status}
            cost = dict(rec.cost or {})
            cost["sessions"] = int(cost.get("sessions", 0)) + 1
            rec = self.store.update_experiment(experiment_id, status="running", started_at=rec.started_at or now_iso(),
                                               message="running", cost=cost)
        commit, dirty = code_state()
        self.store.log_event(experiment_id, "started", {"session": cost["sessions"], "code_commit": commit,
                                                        "code_dirty": dirty})

        spec = rec.committed.spec
        template = self._template(spec.template)
        case = self._case_for(rec.case_id)
        plan = self.store.get_plan(experiment_id)
        n_planned = len(plan)
        done = self.store.trial_indices(experiment_id)
        counts = self.store.trial_counts(experiment_id)
        n_done = sum(counts.values())
        n_errors = counts.get("error", 0)
        budget = spec.budget
        prior_seconds = float(cost.get("seconds", 0.0))
        session_t0 = time.time()
        holder = {"n_done": n_done}

        def is_cancelled() -> bool:
            return experiment_id in self._cancel

        def sub_progress(frac: float, msg: str) -> None:
            progress(min(1.0, (holder["n_done"] + max(0.0, min(1.0, float(frac)))) / n_planned), msg)

        ctx = RunContext(adapter=self.adapter, service=self.service, experiment_id=experiment_id, spec=spec,
                         case=case, is_cancelled=is_cancelled, progress=sub_progress, scratch={})
        device = getattr(getattr(self.adapter, "info", None), "device", None)
        stop: str | None = None
        n_this_session = 0
        for item in plan:
            if item.index in done:
                continue
            if is_cancelled():
                stop = "cancelled"
                break
            if n_done >= budget.max_trials:
                stop = "budget:max_trials"
                break
            if prior_seconds + (time.time() - session_t0) >= budget.max_seconds:
                stop = "budget:max_seconds"
                break
            trial = self._run_one(template, item, ctx, experiment_id)
            self.store.insert_trial(trial)  # append-only; a duplicate would be an engine bug → engine error
            done.add(item.index)
            n_done += 1
            n_this_session += 1
            holder["n_done"] = n_done
            n_errors += int(trial.status == "error")
            cost["seconds"] = round(prior_seconds + time.time() - session_t0, 4)
            cost["trial_seconds"] = round(float(cost.get("trial_seconds", 0.0)) + float(trial.latency_s or 0.0), 4)
            cost["generated_tokens"] = int(cost.get("generated_tokens", 0)) + int(trial.n_generated_tokens or 0)
            fp = _reported_forward_passes(trial)
            if fp is not None:
                cost["forward_passes"] = int(cost.get("forward_passes", 0)) + fp
                cost["forward_passes_reported_trials"] = int(cost.get("forward_passes_reported_trials", 0)) + 1
            cost["peak_rss_bytes"] = max(int(cost.get("peak_rss_bytes", 0)), peak_rss_bytes())
            mem = accelerator_memory(device)
            if mem is not None:
                cost["device_memory"] = mem
            msg = f"trial {n_done}/{n_planned}" + (f" ({n_errors} errors)" if n_errors else "")
            self.store.update_experiment(experiment_id, n_done=n_done, n_errors=n_errors,
                                         progress=round(n_done / n_planned, 6), cost=cost, message=msg)
            progress(n_done / n_planned, msg)

        cost["seconds"] = round(prior_seconds + time.time() - session_t0, 4)
        cost["peak_rss_bytes"] = max(int(cost.get("peak_rss_bytes", 0)), peak_rss_bytes())
        if stop == "cancelled":
            self.store.update_experiment(experiment_id, status="cancelled", finished_at=now_iso(), cost=cost,
                                         message=f"cancelled after {n_done}/{n_planned} trials; resume to continue")
            self.store.log_event(experiment_id, "cancelled", {"n_done": n_done, "n_this_session": n_this_session})
            return {"experiment_id": experiment_id, "status": "cancelled", "n_done": n_done}
        if stop:
            self.store.log_event(experiment_id, "budget_stop", {"reason": stop, "n_done": n_done,
                                                                 "n_planned": n_planned, "seconds": cost["seconds"]})
            cost["budget_stopped"] = stop.split(":", 1)[1]

        complete = len(done) >= n_planned and all(p.index in done for p in plan)
        return self._analyse(experiment_id, template, spec, case, n_planned, n_done, n_errors, cost, stop, complete)

    def _run_one(self, template: Any, item: TrialPlanItem, ctx: RunContext, experiment_id: str) -> Trial:
        expected = trial_id(experiment_id, item.index)
        t0 = time.time()
        try:
            trial = template.run_trial(item, ctx)
            if isinstance(trial, dict):
                trial = Trial.model_validate(trial)
            if not isinstance(trial, Trial):
                raise TypeError(f"run_trial returned {type(trial).__name__}, expected Trial")
            mismatches = [f for f, want in (("id", expected), ("experiment_id", experiment_id),
                                            ("index", item.index), ("item_id", item.item_id),
                                            ("condition", item.condition))
                          if getattr(trial, f) != want]
            if mismatches:
                raise ValueError(f"template returned a trial that does not match plan item {item.index}: "
                                 f"fields {mismatches}")
            if not trial.latency_s:
                trial = trial.model_copy(update={"latency_s": round(time.time() - t0, 4)})
            return trial
        except Exception as e:
            return Trial(id=expected, experiment_id=experiment_id, index=item.index, item_id=item.item_id,
                         group_id=item.group_id, family=item.family, condition=item.condition,
                         pair_key=item.pair_key, status="error", error=f"{type(e).__name__}: {e}"[:4000],
                         parsed={"engine_error": {"type": type(e).__name__,
                                                  "traceback": traceback.format_exc(limit=4)[-4000:]}},
                         latency_s=round(time.time() - t0, 4), created_at=now_iso())

    def _analyse(self, experiment_id: str, template: Any, spec: ExperimentSpec, case: Case | None, n_planned: int,
                 n_done: int, n_errors: int, cost: dict[str, Any], stop: str | None, complete: bool) -> dict[str, Any]:
        trials = self.store.list_trials(experiment_id)
        try:
            result = template.analyze(spec, case, trials, experiment_id)
            if not isinstance(result, AnalysisResult):
                raise TypeError(f"analyze returned {type(result).__name__}, expected AnalysisResult")
            summary = _jsonable(result.summary)
        except Exception as e:
            self.store.update_experiment(experiment_id, status="error", finished_at=now_iso(), cost=cost,
                                         error=f"analysis failed: {type(e).__name__}: {e}",
                                         message="analysis failed; trials are kept (resume re-runs the analysis)")
            self.store.log_event(experiment_id, "analysis_error", {"error": f"{type(e).__name__}: {e}",
                                                                   "traceback": traceback.format_exc(limit=6)})
            return {"experiment_id": experiment_id, "status": "error"}
        evidence_ids = self._persist_evidence(experiment_id, case, result.evidence)
        analysis = {
            "summary": summary,
            "analysis_version": result.analysis_version,
            "complete": complete,
            "stopped_by": stop or "plan_complete",
            "n_planned": n_planned,
            "n_trials_analyzed": len(trials),
            "n_errors": n_errors,
            "analyzed_at": now_iso(),
            "engine_version": ENGINE_VERSION,
        }
        if complete:
            msg = f"done: {n_done}/{n_planned} trials" + (f" ({n_errors} errors)" if n_errors else "")
        else:
            msg = (f"stopped by budget ({stop}): {n_done}/{n_planned} planned trials ran; "
                   "the analysis covers the completed trials only")
        self.store.update_experiment(experiment_id, status="done", finished_at=now_iso(), cost=cost,
                                     analysis=analysis, evidence_ids=evidence_ids, message=msg,
                                     progress=round(n_done / n_planned, 6))
        self.store.log_event(experiment_id, "done", {"complete": complete, "n_done": n_done,
                                                     "n_evidence": len(evidence_ids),
                                                     "analysis_version": result.analysis_version})
        return {"experiment_id": experiment_id, "status": "done", "n_done": n_done, "evidence_ids": evidence_ids}

    def _persist_evidence(self, experiment_id: str, case: Case | None, items: list[Any]) -> list[str]:
        out: list[str] = []
        for i, ev in enumerate(items or []):
            if not isinstance(ev, EvidenceItem):
                ev = EvidenceItem.model_validate(ev)
            upd: dict[str, Any] = {}
            if ev.experiment_id is None:
                upd["experiment_id"] = experiment_id
            elif ev.experiment_id != experiment_id:
                raise ValueError(f"analysis produced evidence {ev.id} for another experiment ({ev.experiment_id})")
            if ev.case_id is None and case is not None:
                upd["case_id"] = case.id
            eid = ev.id
            if self.store.evidence_exists(eid):
                eid = f"{experiment_id}:{ev.id}"
                if self.store.evidence_exists(eid):
                    eid = f"{experiment_id}:{ev.id}:{i}:{uuid.uuid4().hex[:6]}"
                upd["id"] = eid
            ev = ev.model_copy(update=upd)
            self.store.insert_evidence(EvidenceItem.model_validate(ev.model_dump()))
            out.append(ev.id)
        return out


__all__ = [
    "ENGINE_VERSION", "SELFLAB_MAX_PROMPT_TOKENS", "SelfLabEngine", "plan_hash", "spec_hash",
    "template_access_levels", "peak_rss_bytes", "accelerator_memory",
]
