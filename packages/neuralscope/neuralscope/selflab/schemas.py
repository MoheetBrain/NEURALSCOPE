"""SelfLab shared schemas — FROZEN interface ``selflab-1``.

Every SelfLab workstream (protocol, instrumentation, experiments, backend, interface,
validation) codes against these types. Changing a field is a coordinator decision;
add new optional fields rather than renaming existing ones.

Vocabulary (see docs/selflab/INTERFACES.md):

* TARGET           the model whose claim is being audited (white-box: loaded locally).
* INVESTIGATOR     a model or rule set that proposes experiments. Its text is a hypothesis.
* ACTIVATION DECODER  an external reader of activations (lens, probe, oracle). Its output is
                   an instrument reading about the target, never evidence the target introspects.
* STATISTICAL SCORER  deterministic analysis code (``analysis`` field names module:function@version).

Evidence is never collapsed into one label. Each ``EvidenceItem`` carries an EVIDENCE KIND,
a TEST STATUS and a SCOPE as separate fields. ``UNKNOWN`` outcomes are first-class:
``status="inconclusive"`` with the reason in ``limitations``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "selflab-1"

AccessLevel = Literal["white_box_local", "api_only"]
ParticipantRole = Literal["target", "investigator", "activation_decoder", "statistical_scorer", "observer"]
EvidenceKind = Literal["direct_observation", "deterministic_derivation", "model_report", "interpretive_hypothesis"]
TestStatus = Literal["untested", "supported", "contradicted", "inconclusive", "control_failed"]
Split = Literal["discovery", "debug_pilot", "heldout_confirmation"]
TrialStatus = Literal["ok", "malformed", "refusal", "abstention", "truncated", "error"]
JobStatus = Literal["committed", "queued", "running", "cancelling", "cancelled", "done", "error"]
ClaimType = Literal["source_attribution", "verification", "process_description", "self_prediction", "other"]
TemplateId = Literal["source_influence_case", "source_influence_battery", "csp1_carry", "blinded_state"]
FeatureStatus = Literal["implemented", "research_prototype", "fixture", "api_only_view", "unsupported", "roadmap"]

# The four source-report categories of Experiment A (constrained report).
SourceCategory = Literal["document", "user_suggestion", "other_or_unknown", "insufficient_evidence"]
SOURCE_CATEGORIES: tuple[str, ...] = ("document", "user_suggestion", "other_or_unknown", "insufficient_evidence")


class _M(BaseModel):
    model_config = {"extra": "forbid"}


# ------------------------------------------------------------------------------------------ provenance
class ModelProvenance(_M):
    model_id: str
    revision: str | None = None  # resolved HF snapshot commit, if known
    tokenizer_id: str
    tokenizer_revision: str | None = None
    chat_template_sha256: str | None = None
    dtype: str
    device: str
    backend: str = "neuralscope.hf_decoder (native PyTorch hooks)"
    attn_implementation: str = "eager"
    library_versions: dict[str, str] = Field(default_factory=dict)
    is_demo: bool = False
    # sha256 over a deterministic sample of parameter tensors (provenance.weights_fingerprint); None on old records
    weights_fingerprint: str | None = None


class RunProvenance(_M):
    model: ModelProvenance
    code_commit: str | None = None
    code_dirty: bool | None = None
    analysis_version: str
    decoding: dict[str, Any] = Field(default_factory=dict)  # {"strategy": "greedy", "max_new_tokens": 24, ...}
    seed: int | None = None
    platform: str | None = None
    created_at: str


# ------------------------------------------------------------------------------------------ participants
class Participant(_M):
    role: ParticipantRole
    id: str  # e.g. "Qwen/Qwen2.5-1.5B-Instruct", "selflab.stats@1", "rules:template-suggester@1"
    description: str
    sees: list[str] = Field(default_factory=list)  # access manifest: names of fields/inputs visible to it
    enabled: bool = True


# ------------------------------------------------------------------------------------------ context
class Message(_M):
    role: Literal["system", "user", "assistant"]
    content: str


class ContextSegment(_M):
    """A labelled character span inside one message, with its token span in the target's input (receipt)."""

    kind: Literal["document", "user_hint", "question", "system", "other"]
    message_index: int
    char_start: int
    char_end: int
    token_start: int | None = None  # [start, end) absolute positions in the target input_ids
    token_end: int | None = None
    sha256: str


class Receipt(_M):
    """An independently checkable record. A receipt of delivery proves access, never causal use."""

    id: str
    kind: Literal["context_delivery", "tool_call", "retrieval", "api_usage"]
    statement: str  # template-generated text
    proves: str
    does_not_prove: str
    data: dict[str, Any] = Field(default_factory=dict)


class ClaimSpan(_M):
    text: str
    char_start: int  # within the assistant response text
    char_end: int
    token_start: int | None = None  # absolute positions within the Lab run's input_ids (generated part)
    token_end: int | None = None


class Case(_M):
    id: str
    created_at: str
    title: str
    access_level: AccessLevel
    run_id: str | None = None  # Lab run that produced the response (white-box only)
    messages: list[Message]  # exact context given to the target, NOT including the response
    response_text: str
    claim: ClaimSpan
    claim_type: ClaimType
    segments: list[ContextSegment] = Field(default_factory=list)
    receipts: list[Receipt] = Field(default_factory=list)
    participants: list[Participant] = Field(default_factory=list)
    provenance: RunProvenance | None = None
    experiment_ids: list[str] = Field(default_factory=list)
    is_fixture: bool = False  # True for demo-model / fixture data: the UI marks it visibly
    notes: str | None = None
    schema_version: str = SCHEMA_VERSION


# ------------------------------------------------------------------------------------------ experiments
class Budget(_M):
    max_trials: int = Field(default=400, ge=1, le=5000)
    max_new_tokens: int = Field(default=24, ge=1, le=64)
    max_seconds: float = Field(default=3600.0, gt=0, le=6 * 3600)
    max_prompt_tokens: int = Field(default=768, ge=16, le=2048)


class Hypothesis(_M):
    id: str  # "H1"
    statement: str
    metric: str
    direction: Literal["greater", "less", "two_sided", "equivalence"] = "two_sided"
    threshold: float | None = None
    primary: bool = False


class ExperimentSpec(_M):
    """A bounded, typed experiment. Natural-language requests compile to this — never to code."""

    template: TemplateId
    version: str = "1"
    case_id: str | None = None
    split: Split = "debug_pilot"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    params: dict[str, Any] = Field(default_factory=dict)  # validated by the template's params model
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    primary_metric: str = ""
    exclusions: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    notes: str | None = None


class TrialPlanItem(_M):
    """One planned trial. The full plan is a deterministic function of (spec, case, seed)."""

    index: int
    item_id: str
    group_id: str  # independent unit for clustered inference (entity / problem)
    family: str  # prompt-template family (held-out splits are by family)
    condition: str
    pair_key: str | None = None  # trials sharing a pair_key form a counterfactual set
    label_order: list[str] | None = None  # permutation of option categories shown to the target
    params: dict[str, Any] = Field(default_factory=dict)


class CommittedSpec(_M):
    spec: ExperimentSpec
    spec_sha256: str
    plan_sha256: str
    n_planned: int
    committed_at: str
    provenance: RunProvenance | None = None


class Trial(_M):
    id: str  # f"{experiment_id}:{index:05d}" — stable across resume
    experiment_id: str
    index: int
    item_id: str
    group_id: str
    family: str
    condition: str
    pair_key: str | None = None
    status: TrialStatus
    prompt_sha256: str = ""
    input_ids_sha256: str = ""
    n_prompt_tokens: int = 0
    output_text: str = ""
    output_ids: list[int] = Field(default_factory=list)
    stopped_by: str | None = None
    parsed: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] = Field(default_factory=dict)  # SEPARATE fields, never one label
    measurements: dict[str, Any] = Field(default_factory=dict)  # log-likelihoods with explicit token counts
    run_ids: list[str] = Field(default_factory=list)  # Lab runs (materialised on demand or for patching)
    latency_s: float = 0.0
    n_generated_tokens: int = 0
    error: str | None = None
    created_at: str


class Alternative(_M):
    explanation: str
    discriminating_test: str
    state: Literal["ruled_out", "not_ruled_out", "untested"] = "untested"
    evidence_ids: list[str] = Field(default_factory=list)


class Scope(_M):
    checkpoint: str
    task_distribution: str
    intervention: str
    unit: str  # what an independent unit is ("invented entity", "2-digit addition problem", "single case")
    n_units: int
    n_trials: int
    controls: list[str] = Field(default_factory=list)
    tested_generalization: str = "none"
    split: Split = "debug_pilot"


class EvidenceItem(_M):
    id: str
    case_id: str | None = None
    experiment_id: str | None = None
    kind: EvidenceKind
    status: TestStatus
    title: str
    statement: str  # template-generated from measurements; never free text from a model
    measurement: dict[str, Any] = Field(default_factory=dict)  # effect sizes, CIs, n — numbers only
    scope: Scope | None = None
    trial_ids: list[str] = Field(default_factory=list)
    receipt_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)  # Lab runs holding the genuine tensors
    analysis: str = ""  # "neuralscope.selflab.stats:paired_effect@1"
    supports: list[str] = Field(default_factory=list)  # hypothesis ids / claim id
    contradicts: list[str] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_at: str


class ExperimentRecord(_M):
    id: str
    case_id: str | None = None
    committed: CommittedSpec
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    n_done: int = 0
    n_errors: int = 0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cost: dict[str, Any] = Field(default_factory=dict)  # seconds, generated tokens, forward passes, peak memory
    analysis: dict[str, Any] | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    is_fixture: bool = False
