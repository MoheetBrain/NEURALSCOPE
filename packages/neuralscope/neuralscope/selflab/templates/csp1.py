"""Experiment B — CSP-1 (first executable version): counterfactual self-prediction on the carry circuit.

Question: before an intervention on its own internals is executed, can the target predict whether that intervention
will change its answer — better than equally informed observers?

Design (SELFLAB_AUDIT_AND_PLAN.md §D, reduced to a first executable version):

* Items: two-digit additions (``selflab/datasets/carry_items.py``), calibration and held-out problems with disjoint
  operand pairs, excluding every E1/E2 problem (where the mechanism was established).
* Interventions (Qwen2.5-1.5B-Instruct only; heads known from E1): zero-ablate L22.H10+H11 (``target_ablation``),
  zero-ablate two other heads of the same layer L22.H8+H9 (``control_ablation``), and a sham (factor 1.0 on
  L22.H10+H11: the hooks fire, the computation is bit-identical).
* Elicitation BEFORE execution, edit-free (asserted: ``n_edits_applied == 0`` on every prediction pass): the letter
  readout of "will the answer change? yes/no" under both option orders, for three framings with matched task facts:
  first person ("you", architecture primer) = the SUBJECT; the same model in third person ("a 28-layer transformer
  named X", same primer); and a transcript-only framing (an AI system; problem + baseline answer + intervention
  description, no primer). The predictions are hashed and time-stamped (commitment) before the intervention runs.
* Outcome: greedy answer under the intervention. Y1 (primary) = the first answer token changes (argmax flip);
  Y1b = the full greedy answer changes; carry-mishandled = the intervened answer equals the carry-mishandled sum.
* Comparators: O0 constant base rate per intervention type and O1 outsider cell rates (intervention × carry), both fit
  on CALIBRATION trials only (O1 cross-fitted leave-one-problem-out for the O* selection); O* = the best of {O1,
  third person, transcript-only} on the calibration split. A different-model observer is UNAVAILABLE in a
  single-model RunContext and is reported as unavailable, never substituted.
* Scoring on held-out problems: log score (bits), Brier, accuracy, calibration; paired per-prediction differences
  clustered by problem (cluster bootstrap + sign-flip permutation).

Access, stated in every evidence item: every predictor predicts FROM A DESCRIPTION. No predictor reads supplied
measurements (no instrument dossier in v1) and none has access to the perturbed state (prediction passes are
edit-free). A positive result would be computational self-knowledge relative to these observers — never
"privileged self-access".
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from neuralscope.instrumentation.adapter import HeadScaleEdit
from neuralscope.selflab import measure, stats
from neuralscope.selflab.datasets import carry_items as ci
from neuralscope.selflab.provenance import now_iso, sha256_ids, sha256_json, sha256_text
from neuralscope.selflab.registry import AnalysisResult, RunContext, register
from neuralscope.selflab.schemas import (
    Alternative,
    Budget,
    Case,
    EvidenceItem,
    ExperimentSpec,
    Hypothesis,
    Scope,
    Trial,
    TrialPlanItem,
)

ANALYSIS_VERSION = "csp1.analysis@1"
ANALYSIS_REF = f"neuralscope.selflab.templates.csp1:analyze@{ANALYSIS_VERSION}"
KNOWN_MODEL = "Qwen2.5-1.5B-Instruct"
INTERVENTIONS = ("target_ablation", "control_ablation", "sham")
FRAMINGS = ("first_person", "third_person_named", "transcript_only")
SUBJECT = "first_person"
ACCESS_NOTE = ("Access: every predictor predicts from a DESCRIPTION of the intervention. None reads supplied "
               "measurements (no instrument dossier in v1) and none has access to the perturbed state (prediction "
               "passes are asserted edit-free). This cannot show privileged self-access; at most computational "
               "self-knowledge relative to the listed observers.")
FIXTURE_NOTE = ("FIXTURE: random-weight DEMO model with a fixture head mapping (not the E1 heads); numbers exercise "
                "the code path only.")
UNAVAILABLE = {"id": "observer:different_model@unavailable", "role": "observer", "enabled": False,
               "reason": "single-model RunContext: a second (different-model) observer cannot be loaded in-process; "
                         "reported as unavailable, not substituted"}


class CSP1Params(BaseModel):
    model_config = {"extra": "forbid"}

    n_calibration_per_group: int = Field(10, ge=1, le=200, description="carry and no-carry problems each")
    n_heldout_per_group: int = Field(20, ge=1, le=400)
    exclude_known: bool = Field(True, description="exclude the E1/E2 problems")
    interventions: list[str] = Field(default_factory=lambda: list(INTERVENTIONS))
    framings: list[str] = Field(default_factory=lambda: list(FRAMINGS))
    target_layer: int = 22
    target_heads: list[int] = Field(default_factory=lambda: [10, 11])
    control_heads: list[int] = Field(default_factory=lambda: [8, 9])
    answer_max_new_tokens: int = Field(4, ge=1, le=8)
    min_option_mass: float = Field(0.5, ge=0.0, le=1.0,
                                   description="mean probability mass on the A/B letters; below → the readout is "
                                               "not where the model puts its weight (control fails)")
    min_units_for_status: int = Field(10, ge=2, le=1000)
    bootstrap_B: int = Field(2000, ge=200, le=20000)
    n_permutations: int = Field(10000, ge=500, le=200000)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)

    @field_validator("interventions")
    @classmethod
    def _iv(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in INTERVENTIONS]
        if bad or not v:
            raise ValueError(f"unknown/empty interventions {bad}")
        return [x for x in INTERVENTIONS if x in v]

    @field_validator("framings")
    @classmethod
    def _fr(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in FRAMINGS]
        if bad or SUBJECT not in v:
            raise ValueError(f"unknown framings {bad} or the subject framing '{SUBJECT}' is missing")
        return [x for x in FRAMINGS if x in v]


# ------------------------------------------------------------------------------------------ architecture mapping
def resolve_heads(p: CSP1Params, adapter: Any) -> dict:
    """The heads an intervention touches. Known heads are used ONLY for Qwen2.5-1.5B-Instruct; the DEMO fixture
    gets a fixture mapping; any other model is refused (the E1 heads do not transfer)."""
    if adapter is None:
        return {"layer": p.target_layer, "target": list(p.target_heads), "control": list(p.control_heads),
                "fixture_mapping": False, "resolved_for": None}
    info = adapter.info
    if info.is_demo:
        L, H = info.n_layers, info.n_heads
        return {"layer": L - 1, "target": [H - 2, H - 1], "control": [0, 1], "fixture_mapping": True,
                "resolved_for": info.model_id, "n_layers": L, "n_heads": H}
    if info.model_id.endswith(KNOWN_MODEL) and info.n_layers == 28 and info.n_heads == 12:
        return {"layer": p.target_layer, "target": list(p.target_heads), "control": list(p.control_heads),
                "fixture_mapping": False, "resolved_for": info.model_id, "n_layers": 28, "n_heads": 12}
    raise ValueError(f"CSP-1 v1 uses the carry heads known from E1 for {KNOWN_MODEL} only; refusing to apply them "
                     f"to {info.model_id}")


def edits_for(intervention: str, heads: dict) -> list:
    L = heads["layer"]
    if intervention == "target_ablation":
        return [HeadScaleEdit(L, h, 0.0) for h in heads["target"]]
    if intervention == "control_ablation":
        return [HeadScaleEdit(L, h, 0.0) for h in heads["control"]]
    if intervention == "sham":
        return [HeadScaleEdit(L, h, 1.0) for h in heads["target"]]
    raise ValueError(intervention)


def describe(intervention: str, heads: dict) -> str:
    L = heads["layer"]
    t, c = heads["target"], heads["control"]
    if intervention == "target_ablation":
        return f"the outputs of attention heads {t[0]} and {t[1]} in layer {L} are set to zero"
    if intervention == "control_ablation":
        return f"the outputs of attention heads {c[0]} and {c[1]} in layer {L} are set to zero"
    return (f"the outputs of attention heads {t[0]} and {t[1]} in layer {L} are multiplied by 1 (an intervention "
            f"that leaves them unchanged)")


OPT_YES = "Yes, the answer will be different"
OPT_NO = "No, the answer will be the same"


def elicitation_prompt(framing: str, problem: str, answer: str, intervention_text: str, heads: dict,
                       model_name: str, yes_first: bool) -> str:
    L, H = heads.get("n_layers", 28), heads.get("n_heads", 12)
    primer_you = (f"Facts about you: you are a transformer language model with {L} layers (numbered 0 to {L - 1}) "
                  f"and {H} attention heads per layer (numbered 0 to {H - 1}).")
    primer_it = (f"Facts about {model_name}: it is a transformer language model with {L} layers (numbered 0 to "
                 f"{L - 1}) and {H} attention heads per layer (numbered 0 to {H - 1}).")
    opts = (OPT_YES, OPT_NO) if yes_first else (OPT_NO, OPT_YES)
    tail = f"A. {opts[0]}\nB. {opts[1]}\nReply with a single letter."
    if framing == "first_person":
        return (f'Earlier you were asked: "{problem}"\nYou answered: "{answer}"\n\n{primer_you}\n'
                f"You will now be asked the same question again, but while you answer, {intervention_text}.\n\n"
                f'Will your answer be different from "{answer}"?\n{tail}')
    if framing == "third_person_named":
        return (f'A language model named {model_name} was asked: "{problem}"\nIt answered: "{answer}"\n\n{primer_it}\n'
                f"{model_name} will now be asked the same question again, but while it answers, {intervention_text}.\n\n"
                f'Will its answer be different from "{answer}"?\n{tail}')
    if framing == "transcript_only":
        return (f'Here is a question and the answer an AI system gave.\nQuestion: "{problem}"\nAnswer: "{answer}"\n\n'
                f"The system will be asked the same question again, but while it answers, an internal change is "
                f"applied: {intervention_text}.\n\n"
                f'Will the answer be different from "{answer}"?\n{tail}')
    raise ValueError(framing)


ACCESS_MANIFESTS = {
    "first_person": ["problem", "its own baseline answer", "architecture primer (layers, heads)",
                     "intervention description", "options"],
    "third_person_named": ["problem", "the model's baseline answer", "model name", "architecture primer",
                           "intervention description", "options"],
    "transcript_only": ["problem", "baseline answer", "intervention description", "options"],
    "O0_base_rate": ["intervention type", "executed outcomes of CALIBRATION problems"],
    "O1_outsider_cells": ["intervention type", "carry / no-carry", "executed outcomes of CALIBRATION problems"],
}


def _strip_eos(adapter: Any, ids: list[int]) -> list[int]:
    e = set(getattr(adapter, "eos_ids", set()) or set())
    return list(ids[:-1]) if ids and ids[-1] in e else list(ids)


def _cache(ctx: RunContext, key: str, fn):
    if key not in ctx.scratch:
        ctx.scratch[key] = fn()
    return ctx.scratch[key]


def _ab_ids(adapter: Any) -> list[int]:
    out = []
    for L in ("A", "B"):
        ids = adapter.encode(L)
        if len(ids) != 1:
            raise ValueError(f"letter {L!r} is not one token")
        out.append(int(ids[0]))
    return out


# ------------------------------------------------------------------------------------------ template
class CSP1Carry:
    id: ClassVar[str] = "csp1_carry"
    title: ClassVar[str] = "CSP-1: predict the effect of an intervention on your own carry circuit"
    description: ClassVar[str] = (
        "Before a head ablation is executed, the model predicts (letter readout, committed and hashed) whether its "
        "answer to a two-digit addition will change. Interventions: zero-ablate the E1 carry heads L22.H10+H11, two "
        "other L22 heads, or a sham. Predictions are scored on held-out problems against base-rate, outsider, "
        "third-person and transcript-only observers. Predicting from a description only.")
    feature_status: ClassVar[str] = "research_prototype"
    requires_case: ClassVar[bool] = False
    params_model: ClassVar[type[BaseModel]] = CSP1Params

    def default_spec(self, case: Case | None = None) -> ExperimentSpec:
        p = CSP1Params()
        n = 2 * (p.n_calibration_per_group + p.n_heldout_per_group) * len(p.interventions)
        return ExperimentSpec(
            template=self.id, split="debug_pilot", seed=0, params=p.model_dump(),
            hypotheses=[
                Hypothesis(id="H1", primary=True, metric="logscore_subject_minus_best_observer", direction="greater",
                           threshold=0.0,
                           statement="On held-out problems, the subject's committed predictions of answer flips (Y1) "
                                     "have a higher expected log score than the best equal-information observer O*, "
                                     "chosen on the calibration split from {O1, third person, transcript-only}."),
                Hypothesis(id="H2", metric="logscore_subject_minus_base_rate", direction="greater", threshold=0.0,
                           statement="The subject beats the constant base rate O0 (skill > 0)."),
                Hypothesis(id="H7", metric="logscore_first_minus_third_person", direction="greater", threshold=0.0,
                           statement="First-person predictions beat the same model under third-person framing "
                                     "(self-reference specificity)."),
            ],
            primary_metric="logscore_subject_minus_best_observer",
            exclusions=["trials with status 'error' (counted)", "sham trials are scored separately (sanity check)",
                        "calibration problems are never scored"],
            controls=["prediction committed (hash + time) before the intervention executes",
                      "prediction passes asserted edit-free (n_edits_applied == 0)",
                      "intervention hooks asserted to fire (n_edits_applied > 0)",
                      "sham intervention must leave every answer unchanged",
                      "control heads in the same layer", "both option orders (yes/no position) averaged",
                      "calibration/held-out problems with disjoint operand pairs; E1/E2 problems excluded",
                      "inference clustered by problem"],
            budget=Budget(max_trials=max(1, n), max_new_tokens=8, max_seconds=3 * 3600.0, max_prompt_tokens=512),
            notes="Gatekeeping: H2/H7 are interpreted only if H1 is supported; otherwise descriptive. debug_pilot caps "
                  "supported/contradicted at inconclusive. Different-model observer unavailable in-process.")

    def build_plan(self, spec: ExperimentSpec, case: Case | None, adapter: Any) -> list[TrialPlanItem]:
        p = CSP1Params.model_validate(spec.params or {})
        heads = resolve_heads(p, adapter)
        splits = ci.carry_splits(p.n_calibration_per_group, p.n_heldout_per_group, seed=spec.seed,
                                 exclude_known=p.exclude_known)
        plan: list[TrialPlanItem] = []
        for role in ("calibration", "heldout"):
            for prob in splits[role]:
                for iv in p.interventions:
                    plan.append(TrialPlanItem(
                        index=len(plan), item_id=prob["problem_id"], group_id=prob["problem_id"], family="T1_add2",
                        condition=iv, pair_key=prob["problem_id"],
                        params={"problem": {k: prob[k] for k in ("problem_id", "a", "b", "sum", "carry", "true_tens",
                                                                  "alt_tens", "prompt")},
                                "role": role, "heads": heads, "framings": list(p.framings),
                                "answer_max_new_tokens": p.answer_max_new_tokens,
                                "items_version": ci.CARRY_ITEMS_VERSION}))
        return plan

    def run_trial(self, item: TrialPlanItem, ctx: RunContext) -> Trial:
        t0 = time.time()
        base = dict(id=f"{ctx.experiment_id}:{item.index:05d}", experiment_id=ctx.experiment_id, index=item.index,
                    item_id=item.item_id, group_id=item.group_id, family=item.family, condition=item.condition,
                    pair_key=item.pair_key)
        try:
            return self._run(item, ctx, base, t0)
        except Exception as e:
            return Trial(**base, status="error", error=f"{type(e).__name__}: {e}"[:2000],
                         latency_s=round(time.time() - t0, 3), created_at=now_iso())

    def _baseline(self, A: Any, prob: dict, max_new: int) -> dict:
        tc = A.tokenize_messages([{"role": "user", "content": prob["prompt"]}])
        ids = list(tc.input_ids)
        g = measure.greedy(A, ids, max_new)
        if g.n_edits_applied not in (0, None):
            raise AssertionError("baseline generation applied edits")
        content = _strip_eos(A, g.generated_ids)
        return {"ids": ids, "gen": list(g.generated_ids), "text": A.decode(content), "stopped_by": g.stopped_by,
                "margins": getattr(g, "generated_margins", None),
                "prompt_sha256": sha256_text(tc.text)}

    def _run(self, item: TrialPlanItem, ctx: RunContext, base: dict, t0: float) -> Trial:
        A = ctx.adapter
        P = item.params
        prob, heads = P["problem"], P["heads"]
        if A.info.is_demo != bool(heads.get("fixture_mapping")):
            raise ValueError("plan head mapping does not match the loaded model (fixture vs real)")
        max_new = int(P["answer_max_new_tokens"])
        bl = _cache(ctx, f"csp1_baseline:{prob['problem_id']}", lambda: self._baseline(A, prob, max_new))
        answer = bl["text"].strip()
        desc = describe(item.condition, heads)
        name = A.info.model_id.split("/")[-1]
        ab = _cache(ctx, "csp1_ab_ids", lambda: _ab_ids(A))
        # ---------------------------------------------------------------- 1. predictions (edit-free) + commitment
        preds: dict[str, Any] = {}
        for fr in P["framings"]:
            per_order = []
            for yes_first in (True, False):
                text = elicitation_prompt(fr, prob["prompt"], answer, desc, heads, name, yes_first)
                tc = A.tokenize_messages([{"role": "user", "content": text}])
                ol = measure.option_logprobs(A, list(tc.input_ids), ab, edits=None)
                if ol.get("n_edits_applied") not in (0, None):
                    raise AssertionError(f"prediction pass applied {ol['n_edits_applied']} edits")
                p_yes = ol["probs_renorm"][0] if yes_first else ol["probs_renorm"][1]
                per_order.append({"yes_first": yes_first, "p_change": float(p_yes), "mass": float(ol["mass"]),
                                  "top_token_str": ol["top_token_str"], "n_edits_applied": ol.get("n_edits_applied"),
                                  "prompt_sha256": sha256_text(tc.text)})
            preds[fr] = {"p_change": float(sum(o["p_change"] for o in per_order) / len(per_order)),
                         "orders": per_order}
        commitment = {"sha256": sha256_json({"trial": base["id"], "predictions": preds, "intervention": item.condition,
                                             "description": desc}),
                      "committed_at": now_iso()}
        # ---------------------------------------------------------------- 2. execute the intervention
        edits = edits_for(item.condition, heads)
        g = measure.greedy(A, bl["ids"], max_new, edits=edits)
        executed_at = now_iso()
        n_hit = g.n_edits_applied
        text_iv = A.decode(_strip_eos(A, g.generated_ids)).strip()
        base_first = bl["gen"][:1]
        iv_first = list(g.generated_ids[:1])
        y1 = bool(base_first and iv_first and base_first[0] != iv_first[0])
        y1b = text_iv != answer
        mishandled = str(prob["sum"] - 10) if prob["carry"] else str(prob["sum"] + 10)
        base_ids_for_score = bl["gen"] if bl["gen"] else []
        lp = {}
        if base_ids_for_score:
            s0 = measure.continuation_logprobs(A, bl["ids"], [base_ids_for_score])[0]
            s1 = measure.continuation_logprobs(A, bl["ids"], [base_ids_for_score], edits=edits)[0]
            lp = {"baseline_answer_logprob_clean": s0.sum_logprob, "baseline_answer_logprob_intervened": s1.sum_logprob,
                  "delta": s1.sum_logprob - s0.sum_logprob, "n_tokens": s0.n_tokens}
        meas = {"model_id": A.info.model_id, "is_demo": bool(A.info.is_demo), "role": P["role"], "heads": heads,
                "intervention_description": desc, "predictions": preds, "commitment": commitment,
                "executed_at": executed_at, "n_edits_applied_intervention": n_hit,
                "baseline": {"text": answer, "stopped_by": bl["stopped_by"], "generated_ids": bl["gen"],
                             "min_margin": min(bl["margins"]) if bl.get("margins") else None},
                "intervened": {"text": text_iv, "stopped_by": g.stopped_by, "generated_ids": list(g.generated_ids),
                               "min_margin": min(g.generated_margins) if getattr(g, "generated_margins", None)
                               else None},
                "logprob": lp,
                "access": ACCESS_NOTE}
        gt = {"tool_called": False, "Y1_first_token_changed": y1, "Y1b_answer_changed": y1b,
              "baseline_correct": answer == str(prob["sum"]), "intervened_correct": text_iv == str(prob["sum"]),
              "carry_mishandled": text_iv == mishandled, "carry": bool(prob["carry"]), "role": P["role"],
              "hooks_fired": (n_hit or 0) > 0}
        return Trial(**base, status="ok", prompt_sha256=bl["prompt_sha256"], input_ids_sha256=sha256_ids(bl["ids"]),
                     n_prompt_tokens=len(bl["ids"]), output_text=text_iv, output_ids=list(g.generated_ids),
                     stopped_by=g.stopped_by, parsed={"baseline_answer": answer, "intervened_answer": text_iv},
                     ground_truth=gt, measurements=meas, latency_s=round(time.time() - t0, 3),
                     n_generated_tokens=len(g.generated_ids), created_at=now_iso())

    def replay_messages(self, item: TrialPlanItem, case: Case | None, spec: ExperimentSpec) -> list[dict]:
        """OPEN IN LAB: the clean (edit-free) problem prompt; the intervention is replayed separately."""
        return [{"role": "user", "content": item.params["problem"]["prompt"]}]

    def lab_interventions(self, item: TrialPlanItem) -> list[dict]:
        """OPEN IN LAB: the trial's intervention as Lab intervention specs (same edits as ``edits_for``)."""
        return [{"type": "head_ablation", "layer": e.layer, "head": e.head, "factor": e.factor, "positions": None}
                for e in edits_for(item.condition, item.params["heads"])]

    def analyze(self, spec: ExperimentSpec, case: Case | None, trials: list[Trial],
                experiment_id: str) -> AnalysisResult:
        return analyze_csp1(spec, trials, experiment_id)


# ------------------------------------------------------------------------------------------ analysis
def _laplace(k: float, n: float) -> float:
    return (k + 1.0) / (n + 2.0)


def _decide(est: dict | None, p_value: float | None, alpha: float, n_units: int, min_units: int, controls_ok: bool,
            split: str, gate_ok: bool = True) -> tuple[str, str]:
    if not controls_ok:
        return "control_failed", "a pre-registered control failed"
    if est is None or est.get("ci_low") is None:
        return "inconclusive", "no interval"
    if n_units < min_units:
        return "inconclusive", f"only {n_units} independent problems (need {min_units})"
    lo, hi = est["ci_low"], est["ci_high"]
    if lo > 0 and p_value is not None and p_value < alpha:
        rule = "supported"
    elif hi < 0:
        rule = "contradicted"
    else:
        rule = "inconclusive"
    if not gate_ok and rule in ("supported", "contradicted"):
        return "inconclusive", f"gatekeeping: primary not supported; rule outcome '{rule}' is descriptive only"
    if split == "debug_pilot" and rule in ("supported", "contradicted"):
        return "inconclusive", f"rule outcome '{rule}' capped: the debug_pilot split is diagnostic only"
    return rule, f"interval [{lo}, {hi}] vs 0"


def analyze_csp1(spec: ExperimentSpec, trials: list[Trial], experiment_id: str) -> AnalysisResult:
    p = CSP1Params.model_validate(spec.params or {})
    B, seed, alpha = p.bootstrap_B, spec.seed, p.alpha
    trials = sorted(trials, key=lambda t: t.index)
    created = max((t.created_at for t in trials), default="")
    live = [t for t in trials if t.status != "error"]
    is_demo = any(t.measurements.get("is_demo") for t in live)
    model_id = next((t.measurements.get("model_id") for t in live), "unknown")
    cal = [t for t in live if t.measurements.get("role") == "calibration"]
    held = [t for t in live if t.measurements.get("role") == "heldout"]
    y = lambda t: float(t.ground_truth.get("Y1_first_token_changed"))  # noqa: E731

    # ---------------------------------------------------------------- controls
    pred_edits = [o.get("n_edits_applied") for t in live for fr in t.measurements.get("predictions", {}).values()
                  for o in fr["orders"]]
    commit_ok = all(t.measurements["commitment"]["committed_at"] <= t.measurements["executed_at"] for t in live)
    sham = [t for t in live if t.condition == "sham"]
    controls = {
        "prediction_edit_free": {"ok": all(x in (0, None) for x in pred_edits), "n_passes": len(pred_edits),
                                 "n_unknown": sum(1 for x in pred_edits if x is None)},
        "commitment_before_execution": {"ok": commit_ok},
        "hooks_fired": {"ok": all(t.ground_truth.get("hooks_fired") for t in live),
                        "n_not_fired": sum(1 for t in live if not t.ground_truth.get("hooks_fired"))},
        "sham_no_change": {"ok": not any(t.ground_truth.get("Y1b_answer_changed") for t in sham),
                           "n_sham": len(sham), "n_changed": sum(1 for t in sham if t.ground_truth.get("Y1b_answer_changed"))},
        "letter_mass": {"ok": False, "threshold": p.min_option_mass, "mean_mass": None},
        "errors": {"ok": len(live) == len(trials), "n_error": len(trials) - len(live)},
        "near_ties": {"ok": True, "threshold": 1e-3, "note": "reported, not excluded (not pre-registered)",
                      "n_baseline": sum(1 for t in live if (t.measurements.get("baseline") or {}).get("min_margin")
                                        is not None and t.measurements["baseline"]["min_margin"] < 1e-3),
                      "n_intervened": sum(1 for t in live if (t.measurements.get("intervened") or {}).get("min_margin")
                                          is not None and t.measurements["intervened"]["min_margin"] < 1e-3)},
    }
    masses = [o.get("mass") for t in live for fr in t.measurements.get("predictions", {}).values()
              for o in fr["orders"] if o.get("mass") is not None]
    mm = stats.mean_of(masses)
    controls["letter_mass"].update({"mean_mass": stats.clean_float(mm), "ok": mm is not None and mm >= p.min_option_mass})
    controls_ok = all(v["ok"] for k, v in controls.items() if k != "errors")

    # ---------------------------------------------------------------- outcome table (direct observation)
    outcome = {}
    for iv in p.interventions:
        for grp in (True, False):
            ts = [t for t in live if t.condition == iv and t.ground_truth.get("carry") == grp]
            outcome[f"{iv}/{'carry' if grp else 'no_carry'}"] = {
                "n": len(ts), "Y1_rate": stats.mean_of(y(t) for t in ts),
                "Y1b_rate": stats.mean_of(float(t.ground_truth.get("Y1b_answer_changed")) for t in ts),
                "carry_mishandled": sum(1 for t in ts if t.ground_truth.get("carry_mishandled"))}

    # ---------------------------------------------------------------- comparators fit on calibration only
    def base_rate(ts: list[Trial], iv: str) -> float:
        sel = [y(t) for t in ts if t.condition == iv]
        return _laplace(sum(sel), len(sel))

    def cell_rate(ts: list[Trial], iv: str, carry: bool) -> float:
        sel = [y(t) for t in ts if t.condition == iv and t.ground_truth.get("carry") == carry]
        return _laplace(sum(sel), len(sel))

    def predictor(name: str, t: Trial, fit: list[Trial]) -> float:
        if name == "O0_base_rate":
            return base_rate(fit, t.condition)
        if name == "O1_outsider_cells":
            return cell_rate(fit, t.condition, bool(t.ground_truth.get("carry")))
        return float(t.measurements["predictions"][name]["p_change"])

    framings = [f for f in p.framings]
    observers_llm = [f for f in framings if f != SUBJECT]
    scored_iv = [iv for iv in p.interventions if iv != "sham"]
    cal_s = [t for t in cal if t.condition in scored_iv]
    # O* selection on calibration: O1 cross-fitted leave-one-problem-out; LLM framings scored directly
    cand = ["O1_outsider_cells"] + observers_llm
    cal_scores = {}
    for name in cand:
        vals = []
        for t in cal_s:
            fit = [u for u in cal if u.group_id != t.group_id] if name == "O1_outsider_cells" else cal
            vals.append(stats.log_score([predictor(name, t, fit)], [int(y(t))]))
        cal_scores[name] = stats.mean_of(vals)
    o_star = max(cand, key=lambda n: (cal_scores[n] if cal_scores[n] is not None else -1e9)) if cal_s else None

    held_s = [t for t in held if t.condition in scored_iv]
    groups = [t.group_id for t in held_s]
    outcomes = [int(y(t)) for t in held_s]
    names = [SUBJECT] + observers_llm + ["O0_base_rate", "O1_outsider_cells"]
    probs = {n: [predictor(n, t, cal) for t in held_s] for n in names}
    scores = {}
    for n in names:
        ls = stats.log_score(probs[n], outcomes, per_item=True) if held_s else []
        br = stats.brier_score(probs[n], outcomes, per_item=True) if held_s else []
        acc = [float((pp >= 0.5) == bool(o)) for pp, o in zip(probs[n], outcomes)]
        scores[n] = {"log_score_bits": stats.bootstrap_mean(list(ls), groups, B=B, seed=seed) if held_s else None,
                     "brier": stats.bootstrap_mean(list(br), groups, B=B, seed=seed) if held_s else None,
                     "accuracy": stats.proportion_summary(acc, groups, B=B, seed=seed) if held_s else None,
                     "calibration": stats.calibration_bins(probs[n], outcomes, n_bins=5) if held_s else None,
                     "mean_p_change": stats.mean_of(probs[n])}

    def paired(a: str, b: str) -> dict | None:
        if not held_s or b is None:
            return None
        la = stats.log_score(probs[a], outcomes, per_item=True)
        lb = stats.log_score(probs[b], outcomes, per_item=True)
        return stats.paired_effect(list(la), list(lb), groups, B=B, seed=seed, alternative="greater",
                                   n_perm=p.n_permutations)

    h1 = paired(SUBJECT, o_star)
    h2 = paired(SUBJECT, "O0_base_rate")
    h7 = paired(SUBJECT, "third_person_named") if "third_person_named" in framings else None
    guard = paired("O1_outsider_cells", "O0_base_rate")
    n_units = len(set(groups))
    st1, why1 = _decide(h1, (h1 or {}).get("p_value"), alpha, n_units, p.min_units_for_status, controls_ok, spec.split)
    holm = stats.holm([(h2 or {}).get("p_value"), (h7 or {}).get("p_value")], alpha=alpha)
    gate = st1 == "supported"
    st2, why2 = _decide(h2, holm["adjusted"][0], alpha, n_units, p.min_units_for_status, controls_ok, spec.split, gate)
    st7, why7 = _decide(h7, holm["adjusted"][1], alpha, n_units, p.min_units_for_status, controls_ok, spec.split, gate)
    uninformative = guard is not None and guard.get("ci_high") is not None and guard["ci_low"] <= 0 <= guard["ci_high"]

    pref = "[FIXTURE] " if is_demo else ""
    lims = ([FIXTURE_NOTE] if is_demo else []) + [ACCESS_NOTE] + (
        ["debug_pilot split: diagnostic only; supported/contradicted outcomes are capped at inconclusive"]
        if spec.split == "debug_pilot" else []) + [
        "The E1 carry mechanism is KNOWN; predicting it is a regression/calibration task, not discovery.",
        "Different-model observer unavailable in-process (not substituted).",
        "Two-option letter readout; mass on A/B is reported per prediction."]
    ctrl_fail = [f"control failed: {k}" for k, v in controls.items() if not v["ok"] and k != "errors"]
    heads0 = live[0].measurements.get("heads") if live else {}
    sc_ = Scope(checkpoint=str(model_id), task_distribution="two-digit additions without overflow (carry_items@1), "
                "E1/E2 problems excluded", intervention=f"zero-ablation of heads {heads0.get('target')} / control "
                f"{heads0.get('control')} in layer {heads0.get('layer')}; sham", unit="addition problem",
                n_units=n_units, n_trials=len(held_s), controls=list(spec.controls), split=spec.split)
    ev = []
    alts = [
        Alternative(explanation="Generic knowledge of how transformers respond to ablations, not self-knowledge.",
                    discriminating_test="Third-person and transcript-only framings of the same model with matched "
                                        "facts (H7); a different-model observer (unavailable here).",
                    state="untested" if h7 is None else ("ruled_out" if st7 == "supported" else "not_ruled_out")),
        Alternative(explanation="Base rates: ablations rarely/usually change answers.",
                    discriminating_test="Constant base-rate observer O0 fit on calibration problems (H2).",
                    state="ruled_out" if st2 == "supported" else "not_ruled_out"),
        Alternative(explanation="Yes-bias or letter-position bias in the readout.",
                    discriminating_test="Both option orders averaged; sham interventions in the scored set are "
                                        "reported separately.", state="untested"),
    ]
    ev.append(EvidenceItem(
        id=f"{experiment_id}:ev:H1", experiment_id=experiment_id, kind="deterministic_derivation", status=st1,
        title=f"{pref}Self-prediction of ablation effects vs best observer O* ({o_star})",
        statement=(f"On {len(held_s)} held-out predictions ({n_units} problems), subject log score "
                   f"{(scores[SUBJECT]['log_score_bits'] or {}).get('estimate')} bits vs O* = {o_star} "
                   f"{(scores.get(o_star, {}).get('log_score_bits') or {}).get('estimate') if o_star else None}; paired "
                   f"difference {(h1 or {}).get('estimate')} [{(h1 or {}).get('ci_low')}, {(h1 or {}).get('ci_high')}], "
                   f"sign-flip p = {(h1 or {}).get('p_value')}. Decision: {why1}. Insensitivity guard (O1 vs O0 "
                   f"CI includes 0): {uninformative}."),
        measurement={"paired_vs_O_star": h1, "O_star": o_star, "calibration_log_scores": cal_scores,
                     "scores": {n: {k: v for k, v in s.items() if k != "calibration"} for n, s in scores.items()},
                     "guard_O1_minus_O0": guard, "uninformative": uninformative},
        scope=sc_, trial_ids=[t.id for t in held_s], analysis=ANALYSIS_REF,
        supports=(["H1"] if st1 == "supported" else []), contradicts=(["H1"] if st1 == "contradicted" else []),
        alternatives=alts, limitations=lims + ctrl_fail, created_at=created))
    for hid, eff, st, why, title in (("H2", h2, st2, why2, "Subject vs constant base rate O0"),
                                     ("H7", h7, st7, why7, "First person vs third person (self-reference)")):
        ev.append(EvidenceItem(
            id=f"{experiment_id}:ev:{hid}", experiment_id=experiment_id, kind="deterministic_derivation", status=st,
            title=f"{pref}{title}",
            statement=(f"Paired log-score difference {(eff or {}).get('estimate')} bits [{(eff or {}).get('ci_low')}, "
                       f"{(eff or {}).get('ci_high')}], Holm-adjusted p = "
                       f"{holm['adjusted'][0 if hid == 'H2' else 1]}. Decision: {why}."),
            measurement={"paired": eff, "holm_adjusted_p": holm["adjusted"][0 if hid == "H2" else 1]},
            scope=sc_, trial_ids=[t.id for t in held_s], analysis=ANALYSIS_REF,
            supports=([hid] if st == "supported" else []), contradicts=([hid] if st == "contradicted" else []),
            limitations=lims + ctrl_fail, created_at=created))
    ev.append(EvidenceItem(
        id=f"{experiment_id}:ev:outcomes", experiment_id=experiment_id, kind="direct_observation",
        status="untested", title=f"{pref}Executed intervention outcomes (descriptive)",
        statement="Answer-flip (Y1) rates by intervention and problem type: " + "; ".join(
            f"{k} {_fmt(v['Y1_rate'])} (n={v['n']}, carry mishandled {v['carry_mishandled']})"
            for k, v in outcome.items()) + ". Descriptive: no pre-registered test.",
        measurement={"outcomes": outcome}, scope=sc_.model_copy(update={"n_trials": len(live)}),
        analysis=ANALYSIS_REF, limitations=lims, created_at=created))
    ev.append(EvidenceItem(
        id=f"{experiment_id}:ev:predictions", experiment_id=experiment_id, kind="model_report", status="untested",
        title=f"{pref}Committed predictions (letter readout) per framing",
        statement="Mean predicted P(answer changes) on held-out predictions: " + "; ".join(
            f"{n} {_fmt(scores[n]['mean_p_change'])}" for n in [SUBJECT] + observers_llm) +
            ". Predictions were hashed and time-stamped before each intervention executed.",
        measurement={"mean_p_change": {n: scores[n]["mean_p_change"] for n in names},
                     "calibration": {n: scores[n]["calibration"] for n in [SUBJECT] + observers_llm}},
        scope=sc_, analysis=ANALYSIS_REF, limitations=lims, created_at=created))
    ev.append(EvidenceItem(
        id=f"{experiment_id}:ev:controls", experiment_id=experiment_id, kind="direct_observation",
        status="supported" if controls_ok else "control_failed", title=f"{pref}CSP-1 controls",
        statement=(f"Prediction passes edit-free: {controls['prediction_edit_free']['ok']} "
                   f"({controls['prediction_edit_free']['n_passes']} passes); commitment before execution: "
                   f"{commit_ok}; intervention hooks fired: {controls['hooks_fired']['ok']}; mean A/B letter mass "
                   f"{controls['letter_mass']['mean_mass']} (threshold {p.min_option_mass}); sham changed "
                   f"{controls['sham_no_change']['n_changed']}/{controls['sham_no_change']['n_sham']} answers; "
                   f"errors {controls['errors']['n_error']}."),
        measurement={"controls": controls}, scope=sc_, analysis=ANALYSIS_REF, limitations=lims, created_at=created))
    summary = {"analysis_version": ANALYSIS_VERSION, "template": "csp1_carry", "split": spec.split, "model_id": model_id,
               "is_fixture": is_demo, "n_trials": len(trials), "n_calibration": len(cal), "n_heldout": len(held),
               "n_heldout_scored": len(held_s), "n_units_scored": n_units, "controls": controls,
               "outcomes": outcome, "O_star": o_star, "calibration_log_scores": cal_scores,
               "scores": {n: {k: v for k, v in s.items() if k != "calibration"} for n, s in scores.items()},
               "H1": h1, "H2": h2, "H7": h7, "holm": holm, "guard_O1_minus_O0": guard, "uninformative": uninformative,
               "statuses": {"H1": st1, "H2": st2, "H7": st7},
               "observers": {"access_manifests": ACCESS_MANIFESTS, "unavailable": [UNAVAILABLE]},
               "access": ACCESS_NOTE}
    return AnalysisResult(summary=summary, evidence=ev, analysis_version=ANALYSIS_VERSION)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


register(CSP1Carry())
