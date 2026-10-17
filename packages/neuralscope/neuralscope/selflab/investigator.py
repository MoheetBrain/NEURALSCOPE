"""Deterministic, rule-based investigator: which registered templates could test a case's claim, and why.

Participant id ``rules:template-suggester@1``. Its output is a HYPOTHESIS about which experiment is
informative (``kind="interpretive_hypothesis"``, ``status="untested"``) — never evidence about the
target. Claim text, documents and target output are treated as data: the rules only look at the
case's ``claim_type``, the KINDS of its context segments, its access level, and (for the arithmetic
rule) whether the claim text contains digits or arithmetic words. Nothing in the text is executed.

An LLM investigator is not part of this build: it is reported as ``unsupported`` until one is
explicitly configured AND approved.
"""

from __future__ import annotations

import re
from typing import Any

from neuralscope.selflab.schemas import Case, ClaimType

SUGGESTER_ID = "rules:template-suggester@1"
CLAIM_TYPE_RULES_ID = "rules:claim-type@1"
LLM_INVESTIGATOR = {
    "id": "llm-investigator",
    "status": "unsupported",
    "reason": "not configured/approved: no LLM investigator is configured on this server, and none may run "
              "without explicit approval",
}
KNOWN_TEMPLATE_IDS = ("source_influence_case", "source_influence_battery", "csp1_carry", "blinded_state")

_CLAIM_TYPE_PATTERNS: tuple[tuple[ClaimType, re.Pattern[str]], ...] = (
    ("source_attribution", re.compile(
        r"\b(according to|the (document|text|passage|article|source|context)|you (said|told|mentioned|suggested|"
        r"wrote)|your (hint|suggestion|message)|based on|from the (document|text|context|passage)|i (read|found|"
        r"saw) (it|this|that)|source)\b", re.I)),
    ("verification", re.compile(
        r"\b(i (checked|verified|double[- ]checked|confirmed|re-?checked)|verified|confirm(ed)?|"
        r"i('m| am) (sure|certain|confident))\b", re.I)),
    ("process_description", re.compile(
        r"\b(i (calculated|computed|added|carried|reasoned|worked|used|first|then|multiplied|subtracted)|"
        r"step( by step)?|carry|carried|by adding|my reasoning|i thought)\b", re.I)),
    ("self_prediction", re.compile(
        r"\b(i (would|will|could|can|cannot|can't|might) (answer|say|get|respond|know)|if you asked me|"
        r"i'?d (say|answer))\b", re.I)),
)
_ARITH = re.compile(r"(\d\s*[-+*x×/]\s*\d|\bcarr(y|ied|ies)\b|\b(add|added|adding|sum|plus|digit)s?\b)", re.I)


def guess_claim_type(claim_text: str) -> ClaimType:
    """Deterministic keyword rules (``rules:claim-type@1``). A default the user can override."""
    for ct, pat in _CLAIM_TYPE_PATTERNS:
        if pat.search(claim_text or ""):
            return ct
    return "other"


def _segment_kinds(case: Case) -> set[str]:
    return {s.kind for s in case.segments}


def suggest(case: Case, templates: list[Any]) -> dict[str, Any]:
    """Rank registered templates for ``case``. Pure function of (case metadata, registry)."""
    registered = {t.id: t for t in templates}
    kinds = _segment_kinds(case)
    has_doc = "document" in kinds
    has_hint = "user_hint" in kinds
    ct = case.claim_type
    arithmetic = bool(_ARITH.search(case.claim.text or ""))
    api_only = case.access_level == "api_only"

    rules: list[tuple[str, bool, str, list[str]]] = []  # (template, applicable, reason, missing)
    if has_doc:
        reason = ("The claim attributes the answer to a source and the context contains a document: removing or "
                  "replacing the document tests whether the answer actually follows it."
                  if ct == "source_attribution" else
                  "The context contains a document: testing whether the answer depends on it (remove / replace "
                  "branches) is a cheap first control for this claim.")
        if has_hint:
            reason += " A user hint is also present, so a hint branch can separate document use from cue-following."
        rules.append(("source_influence_case", True, reason, []))
    else:
        rules.append(("source_influence_case", False,
                      "Needs a case whose context contains a document segment (compose one in the chat panel).",
                      ["document segment"]))
    if ct == "source_attribution":
        rules.append(("source_influence_battery", True,
                      "Population-level test of source-report accuracy on controlled invented-entity items. It "
                      "measures how reliable this model's source reports are in general — not this case.", []))
    else:
        rules.append(("source_influence_battery", False,
                      "Relevant when the claim is a source attribution (claim_type 'source_attribution').",
                      ["claim_type=source_attribution"]))
    if ct in ("process_description", "verification", "self_prediction") and arithmetic:
        rules.append(("csp1_carry", True,
                      "The claim describes or predicts an arithmetic procedure; CSP-1 compares carry-related "
                      "self-reports with measured effects of interventions on the computation.", []))
    else:
        rules.append(("csp1_carry", False,
                      "Relevant when the claim describes, verifies or predicts an arithmetic computation.",
                      ["arithmetic process / self-prediction claim"]))
    if ct == "self_prediction":
        rules.append(("blinded_state", True,
                      "The claim predicts the target's own behaviour; a blinded-state test checks the prediction "
                      "against the behaviour without showing the target its earlier output.", []))
    else:
        rules.append(("blinded_state", False, "Relevant when the claim is a self-prediction.",
                      ["claim_type=self_prediction"]))

    suggestions = []
    for order, (tid, applicable, reason, missing) in enumerate(rules):
        t = registered.get(tid)
        entry: dict[str, Any] = {"template": tid, "rank_rule": order, "reason": reason, "missing": list(missing)}
        if t is None:
            entry.update(applicable=False, available=False, feature_status="roadmap", title=tid,
                         note="template not registered on this server (not available in this build)")
        else:
            levels = tuple(getattr(t, "access_levels", ("white_box_local",)))
            ok_access = case.access_level in levels
            entry.update(available=True, title=getattr(t, "title", tid),
                         feature_status=getattr(t, "feature_status", "research_prototype"),
                         requires_case=bool(getattr(t, "requires_case", False)),
                         applicable=bool(applicable and ok_access))
            if applicable and not ok_access:
                entry["missing"] = [*entry["missing"], "white-box access to the target"]
                entry["note"] = ("API-only case: only pasted-data inspection is available; this template needs the "
                                 "target loaded locally")
        suggestions.append(entry)
    for tid, t in sorted(registered.items()):
        if tid not in KNOWN_TEMPLATE_IDS:
            suggestions.append({"template": tid, "rank_rule": len(rules), "applicable": False, "available": True,
                                "title": getattr(t, "title", tid),
                                "feature_status": getattr(t, "feature_status", "research_prototype"),
                                "reason": "No suggestion rule covers this template.", "missing": []})
    suggestions.sort(key=lambda s: (not s["applicable"], not s.get("available", False), s["rank_rule"],
                                    s["template"]))
    return {
        "participant": SUGGESTER_ID,
        "kind": "interpretive_hypothesis",
        "status": "untested",
        "is_hypothesis": True,
        "disclaimer": "Rule-based suggestions of experiments that could test the claim. They are hypotheses about "
                      "what would be informative, not evidence about the target.",
        "case_id": case.id,
        "inputs_seen": {"claim_type": ct, "segment_kinds": sorted(kinds), "access_level": case.access_level,
                        "claim_mentions_arithmetic": arithmetic},
        "suggestions": suggestions,
        "llm_investigator": dict(LLM_INVESTIGATOR),
    }
