"""Typed request/response schemas for the NeuralScope HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from neuralscope.schemas.interventions import Intervention

EpistemicStatus = Literal["observed", "derived", "inferred", "causal"]


class ModelInfoOut(BaseModel):
    model_id: str
    architecture: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    vocab_size: int
    dtype: str
    device: str
    n_params: int
    tied_embeddings: bool
    norm_type: str
    is_demo: bool
    capabilities: dict[str, bool]
    n_depths: int
    load_seconds: float | None = None


class HealthOut(BaseModel):
    status: Literal["loading", "ready", "error"]
    model_id: str
    is_demo: bool
    error: str | None = None
    trace_level_default: str
    version: str = "0.1.0"


class TokenizeIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    chat: bool = True
    system_prompt: str | None = None


class CreateRunIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    chat: bool = True
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=16, ge=1, le=256)
    trace_level: Literal["minimal", "standard", "full"] | None = None
    target_text: str | None = None
    target_token_id: int | None = None
    target_position: int | None = None
    contrast_text: str | None = None
    notes: str | None = None


class SetTargetIn(BaseModel):
    target_text: str | None = None
    target_token_id: int | None = None
    target_position: int | None = None
    contrast_text: str | None = None
    contrast_token_id: int | None = None


class UpdateRunIn(BaseModel):
    notes: str | None = None
    tags: list[str] | None = None


class InterventionIn(BaseModel):
    base_run_id: str
    interventions: list[Intervention] = Field(min_length=1, max_length=64)
    max_new_tokens: int | None = Field(default=None, ge=1, le=256)
    notes: str | None = None


class CompareIn(BaseModel):
    run_a: str
    run_b: str
    position_a: int | None = None
    position_b: int | None = None


class HeadSweepIn(BaseModel):
    run_id: str
    position: int | None = None
    target_token_id: int | None = None
    contrast_token_id: int | None = None
    factor: float = 0.0


class PatchSweepIn(BaseModel):
    clean_run_id: str
    corrupt_run_id: str
    positions: list[int] | None = None


class MathVerifyIn(BaseModel):
    expression: str | None = Field(default=None, max_length=200)
    answer: str = Field(max_length=500)
    problem: dict | None = None


class MathProblemsIn(BaseModel):
    n: int = Field(default=25, ge=1, le=200)
    seed: int = 0
    kinds: list[Literal["add", "sub", "mul", "linear_equation", "compare_fractions", "mod"]] | None = None
    default_mix: bool = True


class MathRunIn(BaseModel):
    problem: dict | None = None
    kind: Literal["add", "sub", "mul", "linear_equation", "compare_fractions", "mod"] | None = None
    operands: dict | None = None
    trace_level: Literal["minimal", "standard", "full"] = "standard"


class BenchmarkIn(BaseModel):
    n: int = Field(default=25, ge=1, le=100)
    seed: int = 0
    kinds: list[Literal["add", "sub", "mul", "linear_equation", "compare_fractions", "mod"]] | None = None
    name: str | None = None


class BenchmarkEvalIn(BaseModel):
    interventions: list[Intervention] = Field(min_length=1, max_length=64)


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    run_ids: list[str] = Field(default_factory=list)
    notes: str = ""
    payload: dict = Field(default_factory=dict)


class ExperimentPatchIn(BaseModel):
    name: str | None = None
    notes: str | None = None
    run_ids: list[str] | None = None
    payload: dict | None = None


class JobOut(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "error"]
    progress: float
    message: str
    result: dict | list | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    meta: dict = Field(default_factory=dict)
