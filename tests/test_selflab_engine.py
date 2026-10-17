"""SelfLab engine: commit hashes, single-worker runs, budget, cancel, crash + resume without re-running trials.

A small FAKE template (registered in-test, restored afterwards) stands in for SA3's templates so the engine is
tested independently of them. It runs real greedy decoding on the DEMO fixture model (milliseconds).
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field

from neuralscope.experiments.jobs import JobManager
from neuralscope.selflab import registry
from neuralscope.selflab.engine import SelfLabEngine, plan_hash, spec_hash
from neuralscope.selflab.provenance import now_iso, sha256_ids, sha256_json
from neuralscope.selflab.registry import AnalysisResult
from neuralscope.selflab.schemas import (
    Budget,
    EvidenceItem,
    ExperimentSpec,
    Hypothesis,
    Scope,
    Trial,
    TrialPlanItem,
)
from neuralscope.selflab.store import ConflictError, SelfLabError, SelfLabStore

FAKE_ID = "blinded_state"  # a TemplateId literal; the real registration (if any) is restored after each test
WORDS = ["apple", "river", "stone", "cloud", "ember", "maple", "orbit", "tulip"]


class _Crash(BaseException):
    """Simulates the process dying mid-experiment (not an Exception, so nothing converts it to a trial)."""


class FakeParams(BaseModel):
    model_config = {"extra": "forbid"}
    n_items: int = Field(default=4, ge=1, le=200)
    max_new_tokens: int = Field(default=3, ge=1, le=16)
    fail_indices: list[int] = Field(default_factory=list)
    corrupt_output_indices: list[int] = Field(default_factory=list)


class FakeTemplate:
    id: ClassVar[str] = FAKE_ID
    title: ClassVar[str] = "Fake test template"
    description: ClassVar[str] = "Repeats a word; used only by the SelfLab backend tests."
    feature_status: ClassVar[str] = "fixture"
    params_model: ClassVar[type[BaseModel]] = FakeParams

    def __init__(self) -> None:
        self.requires_case = False
        self.calls: list[int] = []
        self.crash_on_call: int | None = None  # raise _Crash on this call number (1-based)
        self.sleep = 0.0
        self.plan_salt = ""
        self.on_trial: Any = None  # callback(index, ctx)
        self.analyze_error: str | None = None
        self.plan_error: str | None = None  # build_plan raises ValueError(plan_error)

    def default_spec(self, case) -> ExperimentSpec:
        return ExperimentSpec(template=FAKE_ID, case_id=case.id if case else None, seed=7,
                              params={"n_items": 4}, primary_metric="ok_rate",
                              hypotheses=[Hypothesis(id="H1", statement="all trials ok", metric="ok_rate",
                                                     primary=True)],
                              budget=Budget(max_trials=50, max_new_tokens=8, max_seconds=120.0))

    def build_plan(self, spec, case, adapter) -> list[TrialPlanItem]:
        if self.plan_error:
            raise ValueError(self.plan_error)
        p = FakeParams.model_validate(spec.params)
        rng = random.Random(spec.seed)
        return [TrialPlanItem(index=i, item_id=f"item{i // 2}", group_id=f"g{i // 2}", family="f0",
                              condition="a" if i % 2 == 0 else "b", pair_key=f"item{i // 2}",
                              params={"word": rng.choice(WORDS) + self.plan_salt, "nonce": rng.randrange(10**9)})
                for i in range(p.n_items)]

    @staticmethod
    def turn1_messages(item: TrialPlanItem) -> list[dict]:
        return [{"role": "user", "content": f"Repeat the word {item.params['word']}."}]

    def replay_messages(self, item: TrialPlanItem, case) -> list[dict]:
        return self.turn1_messages(item)

    def run_trial(self, item: TrialPlanItem, ctx) -> Trial:
        self.calls.append(item.index)
        if self.crash_on_call is not None and len(self.calls) == self.crash_on_call:
            raise _Crash(f"simulated crash at plan index {item.index}")
        if self.on_trial:
            self.on_trial(item.index, ctx)
        if self.sleep:
            time.sleep(self.sleep)
        p = FakeParams.model_validate(ctx.spec.params)
        if item.index in p.fail_indices:
            raise RuntimeError(f"template failure at {item.index}")
        msgs = self.turn1_messages(item)
        tc = ctx.adapter.tokenize_messages(msgs)
        gen = ctx.adapter.generate(tc.input_ids, p.max_new_tokens)
        out = list(gen.generated_ids)
        if item.index in p.corrupt_output_indices:
            out = [int(t) + 1 for t in out]  # stored output deliberately differs from what the model produced
        return Trial(id=f"{ctx.experiment_id}:{item.index:05d}", experiment_id=ctx.experiment_id, index=item.index,
                     item_id=item.item_id, group_id=item.group_id, family=item.family, condition=item.condition,
                     pair_key=item.pair_key, status="ok", prompt_sha256=sha256_json(msgs),
                     input_ids_sha256=sha256_ids(tc.input_ids), n_prompt_tokens=len(tc.input_ids),
                     output_text=ctx.adapter.decode(out), output_ids=out, stopped_by=gen.stopped_by,
                     parsed={"nonce": item.params["nonce"], "fresh_random": random.random()},
                     measurements={"n_forward_passes": len(out)}, n_generated_tokens=len(out), created_at=now_iso())

    def analyze(self, spec, case, trials, experiment_id) -> AnalysisResult:
        if self.analyze_error:
            raise RuntimeError(self.analyze_error)
        n_ok = sum(t.status == "ok" for t in trials)
        ev = EvidenceItem(id="ev_fake", experiment_id=experiment_id, case_id=case.id if case else None,
                          kind="direct_observation", status="inconclusive", title="Fake outcome",
                          statement=f"{n_ok} of {len(trials)} trials completed without error.",
                          measurement={"n_ok": n_ok, "n": len(trials)},
                          scope=Scope(checkpoint="demo", task_distribution="fake", intervention="none",
                                      unit="item", n_units=len({t.group_id for t in trials}), n_trials=len(trials)),
                          trial_ids=[t.id for t in trials], analysis="tests.fake:analyze@1", created_at=now_iso())
        return AnalysisResult(summary={"n_ok": n_ok, "n": len(trials)}, evidence=[ev], analysis_version="fake-1")


@pytest.fixture()
def fake_template(monkeypatch) -> FakeTemplate:
    registry._load_builtin()  # import real templates first so they cannot overwrite the fake later
    t = FakeTemplate()
    monkeypatch.setitem(registry._REGISTRY, FAKE_ID, t)
    return t


@pytest.fixture()
def jobs() -> JobManager:
    return JobManager()


@pytest.fixture()
def sstore(settings) -> SelfLabStore:
    return SelfLabStore(settings.db_path)


@pytest.fixture()
def engine(sstore, jobs, service) -> SelfLabEngine:
    return SelfLabEngine(sstore, jobs, service=service)


def _spec(**params) -> ExperimentSpec:
    budget = params.pop("budget", None) or Budget(max_trials=100, max_new_tokens=8, max_seconds=120.0)
    return ExperimentSpec(template=FAKE_ID, seed=params.pop("seed", 7), params=params, budget=budget)


def _snapshot(store: SelfLabStore, eid: str) -> dict[int, dict]:
    return {t.index: t.model_dump(mode="json") for t in store.list_trials(eid)}


# ====================================================================================== commit
def test_commit_hashes_are_stable_and_nothing_runs(engine, fake_template):
    a = engine.commit(_spec(n_items=5))
    b = engine.commit(_spec(n_items=5))
    c = engine.commit(_spec(n_items=5, seed=8))
    assert a.status == b.status == "committed" and a.id != b.id
    assert a.committed.spec_sha256 == b.committed.spec_sha256
    assert a.committed.plan_sha256 == b.committed.plan_sha256
    assert c.committed.plan_sha256 != a.committed.plan_sha256  # seed changes the plan
    # params defaults are filled in BEFORE hashing, so the commitment pins them
    assert a.committed.spec.params == {"n_items": 5, "max_new_tokens": 3, "fail_indices": [],
                                       "corrupt_output_indices": []}
    assert a.committed.spec_sha256 == spec_hash(a.committed.spec) == sha256_json(a.committed.spec.model_dump(mode="json"))
    plan = engine.store.get_plan(a.id)
    assert a.committed.plan_sha256 == plan_hash(plan) and a.committed.n_planned == len(plan) == 5
    assert a.committed.provenance is not None and a.committed.provenance.model.is_demo is True
    assert a.is_fixture is True
    assert fake_template.calls == [] and engine.store.trial_indices(a.id) == set()
    assert [e["event"] for e in engine.store.list_events(a.id)] == ["committed"]


def test_commit_validates(engine, fake_template):
    with pytest.raises(SelfLabError, match="invalid params"):
        engine.commit(_spec(n_items=0))
    with pytest.raises(SelfLabError, match="invalid params"):
        engine.commit(_spec(unknown_param=1))
    with pytest.raises(SelfLabError, match="invalid experiment spec"):
        engine.commit({"template": "telepathy"})
    with pytest.raises(SelfLabError, match="max_prompt_tokens"):
        engine.commit(_spec(budget=Budget(max_prompt_tokens=2048)))
    fake_template.requires_case = True
    with pytest.raises(SelfLabError, match="requires a case"):
        engine.commit(_spec())
    fake_template.requires_case = False
    fake_template.plan_error = "case has no document segment"
    with pytest.raises(SelfLabError, match="cannot plan this spec: case has no document segment"):
        engine.commit(_spec())


# ====================================================================================== run
def test_run_on_job_worker_persists_trials_and_evidence(engine, fake_template, jobs):
    rec = engine.commit(_spec(n_items=4))
    q = engine.run(rec.id)
    assert q.status == "queued"
    job = jobs.get(engine.job_id(rec.id)) if engine.job_id(rec.id) else None
    assert job is None or job["kind"] == "selflab_experiment"
    done = engine.wait(rec.id)
    assert done.status == "done" and done.n_done == 4 and done.n_errors == 0 and done.progress == 1.0
    trials = engine.store.list_trials(rec.id)
    assert [t.id for t in trials] == [f"{rec.id}:{i:05d}" for i in range(4)]
    assert all(t.status == "ok" and t.output_ids for t in trials)
    assert done.analysis["complete"] is True and done.analysis["summary"] == {"n_ok": 4, "n": 4}
    assert done.analysis["analysis_version"] == "fake-1"
    ev = engine.store.list_evidence(experiment_id=rec.id)
    assert [e.id for e in ev] == done.evidence_ids and ev[0].experiment_id == rec.id
    cost = done.cost
    assert cost["seconds"] > 0 and cost["generated_tokens"] == sum(t.n_generated_tokens for t in trials)
    assert cost["forward_passes"] == cost["generated_tokens"] and cost["forward_passes_reported_trials"] == 4
    assert cost["peak_rss_bytes"] > 0 and cost["sessions"] == 1
    with pytest.raises(ConflictError):
        engine.run(rec.id)  # cannot run twice
    with pytest.raises(ConflictError):
        engine.resume(rec.id)  # done is terminal
    with pytest.raises(ConflictError):
        engine.cancel(rec.id)
    events = [e["event"] for e in engine.store.list_events(rec.id)]
    assert events == ["committed", "queued", "started", "done"]


def test_second_experiment_evidence_ids_do_not_collide(engine, fake_template):
    a = engine.run_now(engine.commit(_spec(n_items=2)).id)
    b = engine.run_now(engine.commit(_spec(n_items=2)).id)
    assert a.evidence_ids == ["ev_fake"] and b.evidence_ids == [f"{b.id}:ev_fake"]


def test_error_trials_are_kept(engine, fake_template):
    rec = engine.commit(_spec(n_items=4, fail_indices=[1, 3]))
    done = engine.run_now(rec.id)
    assert done.status == "done" and done.n_done == 4 and done.n_errors == 2
    trials = {t.index: t for t in engine.store.list_trials(rec.id)}
    assert trials[1].status == "error" and "template failure at 1" in trials[1].error
    assert trials[1].parsed["engine_error"]["type"] == "RuntimeError"
    assert trials[0].status == "ok" and trials[2].status == "ok"
    assert done.analysis["summary"] == {"n_ok": 2, "n": 4} and done.analysis["n_errors"] == 2


def test_mismatched_trial_from_template_becomes_error_trial(engine, fake_template):
    real = fake_template.run_trial

    def wrong(item, ctx):
        t = real(item, ctx)
        return t.model_copy(update={"condition": "tampered"}) if item.index == 0 else t

    fake_template.run_trial = wrong
    done = engine.run_now(engine.commit(_spec(n_items=2)).id)
    t0 = engine.store.list_trials(done.id)[0]
    assert t0.status == "error" and "does not match plan item" in t0.error and t0.condition == "a"


def test_budget_max_trials_stops_and_analyses_partial(engine, fake_template):
    rec = engine.commit(_spec(n_items=5, budget=Budget(max_trials=2, max_new_tokens=8)))
    assert "will stop the run early" in rec.message
    done = engine.run_now(rec.id)
    assert done.status == "done" and done.n_done == 2 and done.progress == pytest.approx(0.4)
    assert fake_template.calls == [0, 1]
    assert done.analysis["complete"] is False and done.analysis["stopped_by"] == "budget:max_trials"
    assert done.cost["budget_stopped"] == "max_trials" and "stopped by budget" in done.message
    assert "budget_stop" in [e["event"] for e in engine.store.list_events(rec.id)]


def test_budget_max_seconds(engine, fake_template):
    fake_template.sleep = 0.06
    rec = engine.commit(_spec(n_items=10, budget=Budget(max_trials=100, max_seconds=0.1)))
    done = engine.run_now(rec.id)
    assert done.status == "done" and 1 <= done.n_done < 10
    assert done.analysis["stopped_by"] == "budget:max_seconds" and done.cost["budget_stopped"] == "max_seconds"


# ====================================================================================== cancel
def test_cancel_before_the_worker_reaches_it(engine, fake_template, jobs):
    gate = threading.Event()
    jobs.submit("blocker", lambda p: gate.wait(10))  # occupies the single worker
    rec = engine.commit(_spec(n_items=3))
    engine.run(rec.id)
    c = engine.cancel(rec.id)
    assert c.status == "cancelling"
    assert engine.cancel(rec.id).status == "cancelling"  # idempotent
    gate.set()
    final = engine.wait(rec.id)
    assert final.status == "cancelled" and final.n_done == 0 and fake_template.calls == []
    assert "before start" in final.message


def test_cancel_committed_experiment(engine, fake_template):
    rec = engine.commit(_spec(n_items=3))
    assert engine.cancel(rec.id).status == "cancelled"
    with pytest.raises(ConflictError):
        engine.run(rec.id)  # run needs 'committed'; resume is the way forward
    r = engine.resume(rec.id)
    assert r.status == "queued"
    assert engine.wait(rec.id).status == "done"


def test_cancel_mid_run_then_resume_keeps_completed_trials(engine, fake_template):
    rec = engine.commit(_spec(n_items=6))

    def cancel_at(index, ctx):
        if index == 2:
            engine.cancel(rec.id)  # the user cancels while trial 2 is running

    fake_template.on_trial = cancel_at
    engine.run(rec.id)
    mid = engine.wait(rec.id)
    assert mid.status == "cancelled" and mid.n_done == 3  # trial 2 finished; stopped before trial 3
    before = _snapshot(engine.store, rec.id)
    assert sorted(before) == [0, 1, 2]
    fake_template.on_trial = None
    fake_template.calls.clear()
    engine.resume(rec.id)
    final = engine.wait(rec.id)
    assert final.status == "done" and final.n_done == 6 and final.analysis["complete"] is True
    assert fake_template.calls == [3, 4, 5]  # completed trials were not re-run
    after = _snapshot(engine.store, rec.id)
    assert {i: after[i] for i in before} == before  # nor re-randomised (fresh_random values unchanged)
    assert final.cost["sessions"] == 2


def test_resume_after_crash_does_not_rerun_or_rerandomise(settings, sstore, jobs, service, fake_template):
    e1 = SelfLabEngine(sstore, jobs, service=service)
    rec = e1.commit(_spec(n_items=6))
    fake_template.crash_on_call = 4  # dies while running plan index 3
    with pytest.raises(_Crash):
        e1.run_now(rec.id)
    assert sstore.require_experiment(rec.id).status == "running"  # stale: the "process" died
    before = _snapshot(sstore, rec.id)
    assert sorted(before) == [0, 1, 2]

    # restart: a new store connection + engine recover the interrupted experiment
    s2 = SelfLabStore(settings.db_path)
    e2 = SelfLabEngine(s2, jobs, service=service)
    assert e2.recover_interrupted() == [rec.id]
    assert s2.require_experiment(rec.id).status == "error"
    fake_template.crash_on_call = None
    fake_template.calls.clear()
    e2.resume(rec.id)
    final = e2.wait(rec.id)
    assert final.status == "done" and final.n_done == 6
    assert fake_template.calls == [3, 4, 5]
    after = _snapshot(s2, rec.id)
    assert {i: after[i] for i in before} == before
    assert sorted(after) == list(range(6))
    assert [e["event"] for e in s2.list_events(rec.id)].count("interrupted") == 1


def test_resume_refuses_when_plan_changed(engine, fake_template):
    rec = engine.commit(_spec(n_items=3))
    engine.cancel(rec.id)
    fake_template.plan_salt = "-changed"  # the template now derives a different plan
    with pytest.raises(ConflictError, match="re-derived plan"):
        engine.resume(rec.id)
    assert engine.store.require_experiment(rec.id).status == "cancelled"
    assert engine.store.list_events(rec.id)[-1]["event"] == "resume_refused"
    fake_template.plan_salt = ""
    assert engine.resume(rec.id).status == "queued"
    assert engine.wait(rec.id).status == "done"


def test_resume_refuses_different_model(engine, fake_template, demo_adapter, monkeypatch):
    rec = engine.commit(_spec(n_items=2))
    engine.cancel(rec.id)
    monkeypatch.setattr(demo_adapter.info, "dtype", "bfloat16")
    with pytest.raises(ConflictError, match="loaded model differs"):
        engine.resume(rec.id)


def test_resume_refuses_different_weights_with_same_identity(engine, fake_template, monkeypatch):
    """D-E1: same model id/revision/dtype/device but different weights → refused via weights_fingerprint."""
    import neuralscope.selflab.engine as eng

    rec = engine.commit(_spec(n_items=2))
    fp = rec.committed.provenance.model.weights_fingerprint
    assert fp  # the demo model's weights are fingerprinted at commit
    engine.cancel(rec.id)
    real = eng.model_provenance
    monkeypatch.setattr(eng, "model_provenance",
                        lambda a: real(a).model_copy(update={"weights_fingerprint": "f" * 64}))
    with pytest.raises(ConflictError, match="weights_fingerprint"):
        engine.resume(rec.id)
    ev = engine.store.list_events(rec.id)[-1]
    assert ev["event"] == "resume_refused" and ev["data"]["model_diffs"]["weights_fingerprint"]["committed"] == fp
    monkeypatch.setattr(eng, "model_provenance", real)
    assert engine.resume(rec.id).status == "queued"
    assert engine.wait(rec.id).status == "done"


def test_resume_of_old_record_without_fingerprint_is_noted(engine, fake_template, monkeypatch):
    import neuralscope.selflab.engine as eng

    real = eng.run_provenance

    def old_style(*a, **k):  # a record committed before weights_fingerprint existed
        p = real(*a, **k)
        return p.model_copy(update={"model": p.model.model_copy(update={"weights_fingerprint": None})})

    monkeypatch.setattr(eng, "run_provenance", old_style)
    rec = engine.commit(_spec(n_items=2))
    assert rec.committed.provenance.model.weights_fingerprint is None
    engine.cancel(rec.id)
    engine.resume(rec.id)
    assert engine.wait(rec.id).status == "done"
    resumed = next(e for e in engine.store.list_events(rec.id) if e["event"] == "resumed")
    assert any("weights_fingerprint not compared" in n for n in resumed["data"]["identity_notes"])


def test_analysis_failure_keeps_trials_and_resume_reanalyses(engine, fake_template):
    fake_template.analyze_error = "stats exploded"
    rec = engine.commit(_spec(n_items=3))
    failed = engine.run_now(rec.id)
    assert failed.status == "error" and "stats exploded" in failed.error
    assert engine.store.trial_indices(rec.id) == {0, 1, 2}
    fake_template.analyze_error = None
    fake_template.calls.clear()
    engine.resume(rec.id)
    final = engine.wait(rec.id)
    assert final.status == "done" and fake_template.calls == [] and final.analysis["summary"]["n"] == 3
