"""Reproduction record for one SelfLab experiment.

Contents: committed spec + hashes (with a recomputation check), provenance, code commit, plan, the
full trial table (every status, errors included), analysis summary, evidence, events (audit log),
the case, and TENSOR REFERENCES ONLY (Lab run ids). Raw tensors never leave the server.

``preview=True`` returns what WOULD be exported — field list, byte sizes, counts — and flags every
free-text field (target outputs, user documents, questions, hints, notes) so the user can review it
before exporting. ``exclude_free_text=True`` replaces those values with ``{"redacted": true, "sha256",
"n_chars"}`` placeholders so hashes stay checkable. Detection is by key AND value type (string / list of
strings for ``TEXT_KEYS``, int lists for ``ID_KEYS``, any string leaf for ``CONTAINER_KEYS``) — a heuristic:
review the preview before sharing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from neuralscope.selflab.engine import ENGINE_VERSION, plan_hash, spec_hash
from neuralscope.selflab.provenance import code_state, now_iso, sha256_json, sha256_text
from neuralscope.selflab.schemas import SCHEMA_VERSION
from neuralscope.selflab.store import SelfLabStore

EXPORT_VERSION = "selflab-export-1"

# Free text is identified by KEY and by VALUE TYPE, so numeric fields that share a name (e.g. a count called
# "error", a probability keyed "document") are never flagged or redacted.
# 1. TEXT_KEYS: flagged when the value is a non-empty string or a list of strings.
TEXT_KEYS: dict[str, str] = {
    "content": "chat messages (user questions, pasted documents, hints, system prompts)",
    "response_text": "target output",
    "output_text": "target output",
    "free_text": "target output",
    "report_text": "target output",
    "raw_text": "target output",
    "answer_text": "target output",
    "replay_output_text": "target output",
    "stored_output_text": "target output",
    "token_strs": "token strings (may include target output or documents)",
    "text": "claim / span / candidate text",
    "document": "user-supplied document",
    "document_text": "document text",
    "replacement_text": "user-supplied document replacement",
    "hint": "user-supplied hint",
    "hint_text": "hint text",
    "question": "user-supplied question",
    "notes": "user/free-text notes",
    "title": "case title (the default title quotes the claim, i.e. target output)",
    "traceback": "error traceback (may echo inputs)",
    "error": "error message (may echo inputs)",
}
# 2. ID_KEYS: token-id lists that encode text exactly; flagged when the value is a non-empty list of ints.
ID_KEYS: dict[str, str] = {
    "output_ids": "target output (token ids encode the text exactly)",
    "free_text_ids": "target output (token ids encode the text exactly)",
    "generated_ids": "target output (token ids encode the text exactly)",
    "response_token_ids": "target output (token ids encode the text exactly)",
}
# 3. CONTAINER_KEYS: the whole value is free text when it contains any non-empty string leaf.
CONTAINER_KEYS: dict[str, str] = {
    "messages": "chat messages",
    "turn1_messages": "chat messages",
    "branch_messages": "chat messages of a counterfactual branch (may contain the user's document)",
    "parsed": "parsed target output",
    "supplied": "user-supplied (unverified) receipt",
}
FREE_TEXT_KEYS: dict[str, str] = {**TEXT_KEYS, **ID_KEYS, **CONTAINER_KEYS}


def _has_string_leaf(v: Any) -> bool:
    if isinstance(v, str):
        return bool(v)
    if isinstance(v, dict):
        return any(_has_string_leaf(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_string_leaf(x) for x in v)
    return False


def is_free_text(key: str | None, v: Any) -> bool:
    if key is None:
        return False
    if key in TEXT_KEYS:
        return (isinstance(v, str) and bool(v)) or (
            isinstance(v, list) and bool(v) and all(isinstance(x, str) for x in v) and any(v))
    if key in ID_KEYS:
        return isinstance(v, list) and bool(v) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    if key in CONTAINER_KEYS:
        return _has_string_leaf(v)
    return False


def _placeholder(v: Any) -> dict[str, Any]:
    if isinstance(v, str):
        return {"redacted": True, "sha256": sha256_text(v), "n_chars": len(v)}
    return {"redacted": True, "sha256": sha256_json(v), "n_chars": len(json.dumps(v, default=str))}


def redact(obj: Any, key: str | None = None) -> Any:
    if is_free_text(key, obj):
        return _placeholder(obj)
    if isinstance(obj, dict):
        return {k: redact(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, None) for v in obj]
    return obj


def free_text_paths(obj: Any, path: str = "") -> dict[str, dict[str, Any]]:
    """Aggregate every non-empty free-text field by normalised path (list indices → ``[]``)."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "bytes": 0, "source": ""})

    def walk(o: Any, p: str, k: str | None) -> None:
        if is_free_text(k, o):
            e = out[p]
            e["count"] += 1
            e["bytes"] += len(json.dumps(o, ensure_ascii=False, default=str).encode("utf-8"))
            e["source"] = FREE_TEXT_KEYS[k]  # type: ignore[index]
            return
        if isinstance(o, dict):
            for kk, vv in o.items():
                walk(vv, f"{p}.{kk}" if p else kk, kk)
        elif isinstance(o, list):
            for vv in o:
                walk(vv, f"{p}[]", None)

    walk(obj, path, None)
    return dict(out)


# Sections that are template-generated by contract (EvidenceItem.statement, AnalysisResult.summary) and are not
# scanned or redacted: short status labels there (e.g. a hypothesis named "free_text") are not free text.
NOT_SCANNED = frozenset({"analysis", "evidence"})


def _redact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: (v if k in NOT_SCANNED else redact(v, k)) for k, v in record.items()}


def _size(o: Any) -> int:
    return len(json.dumps(o, ensure_ascii=False, default=str).encode("utf-8"))


def _trial_row(t: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": t["id"], "index": t["index"], "item_id": t["item_id"], "group_id": t["group_id"],
        "family": t["family"], "condition": t["condition"], "pair_key": t.get("pair_key"), "status": t["status"],
        "n_prompt_tokens": t.get("n_prompt_tokens"), "n_generated_tokens": t.get("n_generated_tokens"),
        "latency_s": t.get("latency_s"), "prompt_sha256": t.get("prompt_sha256"),
        "input_ids_sha256": t.get("input_ids_sha256"),
        "output_sha256": sha256_text(t.get("output_text") or ""), "stopped_by": t.get("stopped_by"),
        "error": (t.get("error") or None) and str(t["error"])[:200], "run_ids": t.get("run_ids") or [],
        "stored_sha256": row["data_sha256"] if row else None,
    }


def build_export(store: SelfLabStore, experiment_id: str, preview: bool = False,
                 exclude_free_text: bool = False) -> dict[str, Any]:
    rec = store.require_experiment(experiment_id)
    c = rec.committed
    plan = store.get_plan(experiment_id)
    trials = [t.model_dump(mode="json") for t in store.list_trials(experiment_id)]
    rows = {r["id"]: r for r in store.trial_rows(experiment_id)}
    evidence = [e.model_dump(mode="json") for e in store.list_evidence(experiment_id=experiment_id)]
    case = store.get_case(rec.case_id) if rec.case_id else None
    events = store.list_events(experiment_id)
    lab_links = store.experiment_trial_runs(experiment_id)

    recomputed_spec = spec_hash(c.spec)
    recomputed_plan = plan_hash(plan)
    integrity_ok = []
    for t in trials:
        r = rows.get(t["id"])
        integrity_ok.append(bool(r and sha256_json(t) == r["data_sha256"]))
    tensor_refs: list[dict[str, Any]] = []
    if case is not None and case.run_id:
        tensor_refs.append({"run_id": case.run_id, "kind": "case_run", "lab_url": f"/app?run={case.run_id}"})
    for t in trials:
        for rid in t.get("run_ids") or []:
            tensor_refs.append({"run_id": rid, "kind": "trial_run", "trial_id": t["id"], "lab_url": f"/app?run={rid}"})
    for link in lab_links:
        tensor_refs.append({"run_id": link["run_id"], "kind": "lab_replay", "trial_id": link["trial_id"],
                            "reproduced": link.get("reproduced"), "lab_url": f"/app?run={link['run_id']}"})
    for ev in evidence:
        for rid in ev.get("run_ids") or []:
            tensor_refs.append({"run_id": rid, "kind": "evidence_run", "evidence_id": ev["id"],
                                "lab_url": f"/app?run={rid}"})
    commit_now, dirty_now = code_state()
    prov = c.provenance.model_dump(mode="json") if c.provenance else None
    record: dict[str, Any] = {
        "export_version": EXPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "exported_at": now_iso(),
        "experiment": {k: v for k, v in rec.model_dump(mode="json").items() if k not in ("committed", "analysis")},
        "committed_spec": c.spec.model_dump(mode="json"),
        "hashes": {
            "spec_sha256": c.spec_sha256, "plan_sha256": c.plan_sha256,
            "recomputed_spec_sha256": recomputed_spec, "recomputed_plan_sha256": recomputed_plan,
            "spec_hash_ok": recomputed_spec == c.spec_sha256, "plan_hash_ok": recomputed_plan == c.plan_sha256,
            "trials_sha256": sha256_json(trials), "n_trials_integrity_ok": sum(integrity_ok),
            "n_trials": len(trials),
            "method": "sha256 of canonical JSON (sorted keys, no whitespace); spec/plan via model_dump(mode='json')",
            "note": "hashes support integrity checks of local records; they do not make local logs tamper-proof",
        },
        "committed_at": c.committed_at,
        "n_planned": c.n_planned,
        "provenance": prov,
        "code": {"commit_at_commitment": prov.get("code_commit") if prov else None,
                 "dirty_at_commitment": prov.get("code_dirty") if prov else None,
                 "commit_at_export": commit_now, "dirty_at_export": dirty_now},
        "plan": [p.model_dump(mode="json") for p in plan],
        "trial_table": [_trial_row(t, rows.get(t["id"])) for t in trials],
        "trials": trials,
        "trial_status_counts": store.trial_counts(experiment_id),
        "analysis": rec.analysis,
        "evidence": evidence,
        "case": case.model_dump(mode="json") if case else None,
        "events": events,
        "tensor_references": tensor_refs,
        "raw_tensors_included": False,
        "tensor_note": "Raw tensors stay server-side (data/activations/<run_id>.safetensors); open a run with "
                       "/app?run=<run_id>. Runs are deterministic and can be re-captured from their input_ids.",
        "is_fixture": rec.is_fixture,
    }
    flagged = {p: v for k, val in record.items() if k not in NOT_SCANNED
               for p, v in free_text_paths(val, k).items()}
    if preview:
        sections = []
        for key, val in record.items():
            sec_flags = {p: v for p, v in flagged.items() if p == key or p.startswith(key + ".")
                         or p.startswith(key + "[]")}
            entry: dict[str, Any] = {"path": key, "bytes": _size(val), "free_text": bool(sec_flags)}
            if isinstance(val, list):
                entry["count"] = len(val)
            if sec_flags:
                entry["free_text_fields"] = sorted(sec_flags)
            sections.append(entry)
        return {
            "preview": True, "experiment_id": experiment_id, "export_version": EXPORT_VERSION,
            "fields": sections, "total_bytes": _size(record),
            "free_text_fields": [{"path": p, **v} for p, v in sorted(flagged.items())],
            "free_text_bytes": sum(v["bytes"] for v in flagged.values()),
            "exclude_free_text_bytes": _size(_redact_record(record)),
            "not_scanned": {k: "template-generated by contract (numbers, labels and template statements only)"
                            for k in sorted(NOT_SCANNED)},
            "redaction": ("key- and type-based: string values under " + ", ".join(sorted(TEXT_KEYS)) +
                          "; token-id lists under " + ", ".join(sorted(ID_KEYS)) + "; whole containers under " +
                          ", ".join(sorted(CONTAINER_KEYS)) + " are replaced by {redacted, sha256, n_chars}. "
                          "This is a heuristic: review the listed fields before sharing."),
            "raw_tensors_included": False, "n_tensor_references": len(tensor_refs),
            "n_trials": len(trials), "trial_status_counts": record["trial_status_counts"],
            "is_fixture": rec.is_fixture,
        }
    if exclude_free_text:
        record = _redact_record(record)
        record["free_text_excluded"] = True
        record["redacted_fields"] = sorted(flagged)
    else:
        record["free_text_excluded"] = False
    record["export_sha256"] = sha256_json({k: v for k, v in record.items() if k != "export_sha256"})
    return record
