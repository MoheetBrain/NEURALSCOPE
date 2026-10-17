"""Experiment-template contract between the experiment runners and the job engine — FROZEN ``selflab-1``.

A template is the ONLY way an experiment reaches execution. It turns a validated
``ExperimentSpec`` into a deterministic trial plan, executes one planned trial at a time
through the shared model adapter, and analyses completed trials into evidence.
Templates never execute arbitrary code, shell commands or network calls supplied by a user,
an investigator model or the target's output.

Engine responsibilities (``selflab/engine.py``): commit + hash, plan verification on resume,
iteration, budget/cancel checks between trials, persistence of every trial (errors included).
Template responsibilities (``selflab/templates/*``): params validation, planning, one-trial
execution, analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol

from pydantic import BaseModel

from neuralscope.selflab.schemas import (
    Case,
    EvidenceItem,
    ExperimentSpec,
    FeatureStatus,
    Trial,
    TrialPlanItem,
)


@dataclass
class RunContext:
    """What a template may use while executing a trial. Read-mostly; one shared model."""

    adapter: Any  # neuralscope.instrumentation.ModelAdapter (the single loaded model)
    service: Any  # neuralscope.experiments.service.NeuralScopeService (Lab runs, patch sweeps)
    experiment_id: str
    spec: ExperimentSpec
    case: Case | None
    is_cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[float, str], None] = lambda f, m: None
    scratch: dict[str, Any] = field(default_factory=dict)  # per-experiment cache (e.g. tokenised prefixes)


@dataclass
class AnalysisResult:
    summary: dict[str, Any]  # numbers for the UI/bottom panel; JSON-serialisable
    evidence: list[EvidenceItem]
    analysis_version: str


class ExperimentTemplate(Protocol):
    id: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    feature_status: ClassVar[FeatureStatus]
    requires_case: ClassVar[bool]
    params_model: ClassVar[type[BaseModel]]

    def default_spec(self, case: Case | None) -> ExperimentSpec:
        """A complete, valid spec with pre-registered hypotheses, controls and budget."""

    def build_plan(self, spec: ExperimentSpec, case: Case | None, adapter: Any) -> list[TrialPlanItem]:
        """Deterministic in (spec, case, tokenizer). Must not run the model's forward pass."""

    def run_trial(self, item: TrialPlanItem, ctx: RunContext) -> Trial:
        """Execute ONE trial. Must return a Trial even on failure (status='error', error=...)."""

    def analyze(self, spec: ExperimentSpec, case: Case | None, trials: list[Trial],
                experiment_id: str) -> AnalysisResult:
        """Pure function of the stored trials (no model calls)."""


_REGISTRY: dict[str, ExperimentTemplate] = {}


def register(t: ExperimentTemplate) -> ExperimentTemplate:
    _REGISTRY[t.id] = t
    return t


def get_template(template_id: str) -> ExperimentTemplate:
    _load_builtin()
    if template_id not in _REGISTRY:
        raise KeyError(f"unknown experiment template {template_id!r}")
    return _REGISTRY[template_id]


def list_templates() -> list[ExperimentTemplate]:
    _load_builtin()
    return list(_REGISTRY.values())


def _load_builtin() -> None:
    # Import side effects register the built-in templates. Missing modules (work in progress) are skipped.
    import importlib

    for mod in ("source_influence", "csp1", "blinded_state"):
        try:
            importlib.import_module(f"neuralscope.selflab.templates.{mod}")
        except ModuleNotFoundError as e:
            if not e.name or not e.name.endswith(mod):
                raise
