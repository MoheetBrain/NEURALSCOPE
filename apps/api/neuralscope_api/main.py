"""NeuralScope HTTP API (FastAPI).

Run:  uvicorn neuralscope_api.server:app --port 8000   (server.py builds the app from env settings)
The model loads in a background thread at startup; /api/health reports progress.
All tensors stay server-side; endpoints return summarised, browser-sized views.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neuralscope.config import Settings, get_settings
from neuralscope.experiments.jobs import JobManager
from neuralscope.experiments.service import NeuralScopeService, ServiceError
from neuralscope.experiments.store import ExperimentStore
from neuralscope.instrumentation import ModelAdapter, load_adapter
from neuralscope.schemas.api import (
    BenchmarkEvalIn,
    BenchmarkIn,
    CompareIn,
    CreateRunIn,
    ExperimentIn,
    ExperimentPatchIn,
    HeadSweepIn,
    HealthOut,
    InterventionIn,
    JobOut,
    MathProblemsIn,
    MathRunIn,
    MathVerifyIn,
    ModelInfoOut,
    PatchSweepIn,
    SetTargetIn,
    TokenizeIn,
    UpdateRunIn,
)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ExperimentStore(settings.data_dir)
        self.jobs = JobManager()
        self.adapter: ModelAdapter | None = None
        self.service: NeuralScopeService | None = None
        self.status = "loading"
        self.error: str | None = None
        self.load_started = time.time()

    def set_adapter(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self.service = NeuralScopeService(adapter, self.store, self.settings)
        self.status = "ready"

    def load(self) -> None:
        try:
            self.set_adapter(load_adapter(self.settings))
        except Exception as e:  # pragma: no cover — surfaced via /api/health
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"


def create_app(settings: Settings | None = None, adapter: ModelAdapter | None = None,
               load_in_background: bool = True) -> FastAPI:
    settings = settings or get_settings()
    state = AppState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if adapter is not None:
            state.set_adapter(adapter)
        elif load_in_background:
            threading.Thread(target=state.load, daemon=True, name="model-loader").start()
        else:
            state.load()
        yield

    app = FastAPI(title="NeuralScope API", version="0.1.0", lifespan=lifespan,
                  description="Causal debugger for learned intelligence — V1 neural debugger + Math Lab.")
    app.state.ns = state
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                       allow_methods=["*"], allow_headers=["*"])

    def svc() -> NeuralScopeService:
        if state.status == "error":
            raise HTTPException(503, f"model failed to load: {state.error}")
        if state.service is None:
            raise HTTPException(503, "model is still loading — poll /api/health")
        return state.service

    @app.exception_handler(ServiceError)
    async def _service_error(_, exc: ServiceError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _key_error(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    # ------------------------------------------------------------------ model
    @app.get("/api/health", response_model=HealthOut)
    def health():
        return HealthOut(status=state.status, model_id=settings.model_id, is_demo=settings.is_demo,
                         error=state.error, trace_level_default=settings.trace_level)

    @app.get("/api/model", response_model=ModelInfoOut)
    def model_info(s: NeuralScopeService = Depends(svc)):
        i = s.info
        return ModelInfoOut(**{**i.__dict__, "n_depths": i.n_depths,
                               "load_seconds": getattr(s.adapter, "load_seconds", None)})

    @app.post("/api/tokenize")
    def tokenize(body: TokenizeIn, s: NeuralScopeService = Depends(svc)):
        tp = s.adapter.tokenize(body.prompt, chat=body.chat, system_prompt=body.system_prompt)
        return {"text": tp.text, "user_span": tp.user_span,
                "tokens": s.token_view(tp.input_ids, len(tp.input_ids), tp.user_span, body.chat),
                "n_tokens": len(tp.input_ids), "limit": settings.max_prompt_tokens, "status": "observed"}

    # ------------------------------------------------------------------ runs
    @app.post("/api/runs")
    def create_run(body: CreateRunIn, s: NeuralScopeService = Depends(svc)):
        rec = s.create_run(**body.model_dump())
        return s.run_detail(rec["id"])

    @app.get("/api/runs")
    def list_runs(limit: int = Query(100, le=500), kind: str | None = None, benchmark_id: str | None = None,
                  s: NeuralScopeService = Depends(svc)):
        return [s.run_summary(r) for r in state.store.list_runs(limit=limit, kind=kind, benchmark_id=benchmark_id)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, s: NeuralScopeService = Depends(svc)):
        return s.run_detail(run_id)

    @app.patch("/api/runs/{run_id}")
    def patch_run(run_id: str, body: UpdateRunIn, s: NeuralScopeService = Depends(svc)):
        s._get_run(run_id)
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        state.store.update_run(run_id, **fields)
        return s.run_detail(run_id)

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        if not state.store.delete_run(run_id):
            raise HTTPException(404, "run not found")
        return {"deleted": run_id}

    @app.get("/api/runs/{run_id}/tokens")
    def run_tokens(run_id: str, s: NeuralScopeService = Depends(svc)):
        return s.run_detail(run_id)["tokens"]

    @app.get("/api/runs/{run_id}/layers")
    def run_layers(run_id: str, s: NeuralScopeService = Depends(svc)):
        rec = s._get_run(run_id)
        i = s.info
        return {"depth_labels": s.depth_labels(), "n_layers": i.n_layers, "n_heads": i.n_heads,
                "n_kv_heads": i.n_kv_heads, "d_model": i.d_model, "trace_level": rec["trace_level"],
                "tensor_meta": rec.get("tensor_meta")}

    @app.post("/api/runs/{run_id}/target")
    def set_target(run_id: str, body: SetTargetIn, s: NeuralScopeService = Depends(svc)):
        rec = s._get_run(run_id)
        tgt = s.resolve_target(rec["input_ids"], rec["prompt_len"], body.target_text, body.target_token_id,
                               body.target_position, body.contrast_text, body.contrast_token_id)
        state.store.update_run(run_id, target=tgt)
        return tgt

    @app.get("/api/runs/{run_id}/matrix")
    def matrix(run_id: str, metric: str = "lens_top1_prob", target_token_id: int | None = None,
               s: NeuralScopeService = Depends(svc)):
        return s.matrix(run_id, metric, target_token_id)

    @app.get("/api/runs/{run_id}/cell/{depth}/{position}")
    def cell(run_id: str, depth: int, position: int, target_token_id: int | None = None,
             s: NeuralScopeService = Depends(svc)):
        return s.cell(run_id, depth, position, target_token_id)

    @app.get("/api/runs/{run_id}/logit-lens")
    def logit_lens(run_id: str, position: int | None = None, target_token_id: int | None = None,
                   k: int = Query(10, ge=1, le=50), s: NeuralScopeService = Depends(svc)):
        return s.logit_lens(run_id, position, target_token_id, k)

    @app.get("/api/runs/{run_id}/attention/{layer}")
    def attention(run_id: str, layer: int, query_position: int | None = None, s: NeuralScopeService = Depends(svc)):
        return s.attention(run_id, layer, query_position)

    @app.get("/api/runs/{run_id}/heads/{layer}/{head}")
    def head(run_id: str, layer: int, head: int, position: int | None = None, target_token_id: int | None = None,
             s: NeuralScopeService = Depends(svc)):
        return s.head_detail(run_id, layer, head, position, target_token_id)

    @app.get("/api/runs/{run_id}/dla")
    def dla(run_id: str, position: int | None = None, target_token_id: int | None = None,
            contrast_token_id: int | None = None, s: NeuralScopeService = Depends(svc)):
        return s.dla(run_id, position, target_token_id, contrast_token_id)

    # ------------------------------------------------------------------ interventions & comparison
    @app.post("/api/interventions")
    def intervene(body: InterventionIn, s: NeuralScopeService = Depends(svc)):
        return s.run_intervention(body.base_run_id, list(body.interventions), body.max_new_tokens, body.notes)

    @app.post("/api/compare")
    def compare(body: CompareIn, s: NeuralScopeService = Depends(svc)):
        return s.compare(body.run_a, body.run_b, body.position_a, body.position_b)

    @app.post("/api/sweeps/heads", response_model=JobOut)
    def head_sweep(body: HeadSweepIn, s: NeuralScopeService = Depends(svc)):
        s._get_run(body.run_id)
        return state.jobs.submit("head_sweep", lambda p: s.head_ablation_sweep(
            body.run_id, body.position, body.target_token_id, body.contrast_token_id, body.factor, p),
            meta={"run_id": body.run_id})

    @app.post("/api/sweeps/patch", response_model=JobOut)
    def patch_sweep(body: PatchSweepIn, s: NeuralScopeService = Depends(svc)):
        s._get_run(body.clean_run_id)
        s._get_run(body.corrupt_run_id)
        return state.jobs.submit("patch_sweep", lambda p: s.patch_sweep(
            body.clean_run_id, body.corrupt_run_id, p, body.positions),
            meta={"clean_run_id": body.clean_run_id, "corrupt_run_id": body.corrupt_run_id})

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    def job(job_id: str):
        j = state.jobs.get(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        return j

    @app.get("/api/jobs")
    def jobs():
        return state.jobs.list()

    # ------------------------------------------------------------------ Math Lab
    @app.post("/api/math/verify")
    def math_verify(body: MathVerifyIn):
        from neuralscope.mathlab import MathProblem, verify, verify_expression

        try:
            if body.problem:
                v = verify(MathProblem.from_dict(body.problem), body.answer)
            elif body.expression:
                v = verify_expression(body.expression, body.answer)
            else:
                raise HTTPException(400, "give either `problem` or `expression`")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {**v.to_dict(), "status_label": "observed", "oracle": "sympy"}

    @app.post("/api/math/problems")
    def math_problems(body: MathProblemsIn):
        from neuralscope.mathlab import default_benchmark, generate_problems

        if body.default_mix and not body.kinds:
            probs = default_benchmark(seed=body.seed, n=body.n)
        else:
            probs = generate_problems(body.n, seed=body.seed, kinds=body.kinds)
        return [p.to_dict() for p in probs]

    @app.post("/api/math/run")
    def math_run(body: MathRunIn, s: NeuralScopeService = Depends(svc)):
        from neuralscope.mathlab import MathProblem, make_problem

        if body.problem:
            prob = MathProblem.from_dict(body.problem)
        elif body.kind and body.operands:
            prob = make_problem(body.kind, body.operands)
        else:
            raise HTTPException(400, "give `problem` or (`kind`, `operands`)")
        rec = s.run_math_problem(prob, trace_level=body.trace_level)
        return s.run_detail(rec["id"])

    @app.post("/api/math/benchmarks", response_model=JobOut)
    def math_benchmark(body: BenchmarkIn, s: NeuralScopeService = Depends(svc)):
        from neuralscope.mathlab import MathProblem, default_benchmark, generate_problems, make_problem

        if body.problems:
            try:
                probs = [MathProblem.from_dict(p) if "question" in p else make_problem(p["kind"], p["operands"])
                         for p in body.problems]
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(400, f"invalid problem: {e}")
            name = body.name or f"Math Lab · controlled set · {len(probs)} problems"
        else:
            probs = generate_problems(body.n, seed=body.seed, kinds=body.kinds) if body.kinds else \
                default_benchmark(seed=body.seed, n=body.n)
            name = body.name or f"Math Lab · {len(probs)} problems · seed {body.seed}"
        return state.jobs.submit("benchmark", lambda p: s.run_benchmark(probs, name=name, seed=body.seed, progress=p),
                                 meta={"n": len(probs), "seed": body.seed})

    @app.get("/api/math/benchmarks")
    def math_benchmarks():
        return [{k: v for k, v in b.items() if k != "spec"} for b in state.store.list_benchmarks()]

    @app.get("/api/math/benchmarks/{bench_id}")
    def math_benchmark_get(bench_id: str):
        b = state.store.get_benchmark(bench_id)
        if not b:
            raise HTTPException(404, "benchmark not found")
        return b

    @app.post("/api/math/benchmarks/{bench_id}/intervention-eval", response_model=JobOut)
    def math_benchmark_eval(bench_id: str, body: BenchmarkEvalIn, s: NeuralScopeService = Depends(svc)):
        if not state.store.get_benchmark(bench_id):
            raise HTTPException(404, "benchmark not found")
        return state.jobs.submit("benchmark_eval", lambda p: s.benchmark_intervention_eval(
            bench_id, list(body.interventions), p), meta={"benchmark_id": bench_id})

    # ------------------------------------------------------------------ experiments
    @app.post("/api/experiments")
    def create_experiment(body: ExperimentIn):
        for rid in body.run_ids:
            if not state.store.get_run(rid):
                raise HTTPException(400, f"run {rid} not found")
        return state.store.save_experiment(body.name, body.run_ids, body.notes, body.payload)

    @app.get("/api/experiments")
    def list_experiments():
        return state.store.list_experiments()

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str, s: NeuralScopeService = Depends(svc)) -> dict[str, Any]:
        e = state.store.get_experiment(exp_id)
        if not e:
            raise HTTPException(404, "experiment not found")
        e["runs"] = [s.run_summary(r) for rid in e["run_ids"] if (r := state.store.get_run(rid))]
        return e

    @app.patch("/api/experiments/{exp_id}")
    def patch_experiment(exp_id: str, body: ExperimentPatchIn):
        e = state.store.get_experiment(exp_id)
        if not e:
            raise HTTPException(404, "experiment not found")
        return state.store.save_experiment(
            body.name if body.name is not None else e["name"],
            body.run_ids if body.run_ids is not None else e["run_ids"],
            body.notes if body.notes is not None else e["notes"],
            body.payload if body.payload is not None else e["payload"], experiment_id=exp_id)

    @app.delete("/api/experiments/{exp_id}")
    def delete_experiment(exp_id: str):
        if not state.store.delete_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        return {"deleted": exp_id}

    return app
