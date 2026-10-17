"""SelfLab service: chat runs with delivery receipts, cases, participants, templates, Lab replay, evidence.

Text is data. Questions, documents, hints, target outputs and claim text are stored, hashed, tokenised
and displayed — never interpreted as instructions or executed. The only executable things are the
registered experiment templates, which receive typed, validated ``ExperimentSpec``s.

Access levels:
* ``white_box_local`` — the target is the locally loaded model (the only level that runs experiments).
* ``api_only`` — DISABLED unless ``NEURALSCOPE_SELFLAB_API_ONLY=1``. Even then it is a pasted-transcript
  inspection view: no provider calls are made and white-box templates are refused.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any

from neuralscope.selflab.compose import chat_user_message
from neuralscope.selflab.engine import SELFLAB_MAX_PROMPT_TOKENS, SelfLabEngine, template_access_levels
from neuralscope.selflab.investigator import (
    CLAIM_TYPE_RULES_ID,
    LLM_INVESTIGATOR,
    SUGGESTER_ID,
    guess_claim_type,
    suggest,
)
from neuralscope.selflab.provenance import model_provenance, now_iso, run_provenance, sha256_ids, sha256_text
from neuralscope.selflab.registry import get_template, list_templates
from neuralscope.selflab.schemas import (
    SCHEMA_VERSION,
    Case,
    ClaimSpan,
    ClaimType,
    ContextSegment,
    EvidenceItem,
    ExperimentRecord,
    ExperimentSpec,
    Message,
    Participant,
    Receipt,
    RunProvenance,
    Trial,
)
from neuralscope.selflab.store import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    SelfLabError,
    SelfLabStore,
    UnavailableError,
)

SERVICE_VERSION = "selflab-service-1"
API_ONLY_ENV = "NEURALSCOPE_SELFLAB_API_ONLY"
STATS_SCORER_ID = "selflab.stats@1"

# Input bounds (enforced again by the API request models).
MAX_NEW_TOKENS = 64
MAX_MESSAGES = 16
MAX_MESSAGE_CHARS = 8000
MAX_TOTAL_CHARS = 24000
MAX_QUESTION_CHARS = 2000
MAX_DOCUMENT_CHARS = 8000
MAX_HINT_CHARS = 1000
MAX_SYSTEM_CHARS = 2000
MAX_RESPONSE_CHARS = 8000
MAX_RECEIPTS = 16
# Where open-in-lab looks for a trial's turn-1 messages (after the optional template hook ``replay_messages``):
# trial.measurements, then trial.parsed, then the plan item's params.
REPLAY_MESSAGE_KEYS = ("turn1_messages", "messages", "branch_messages")
CHAT_TRACE_LEVEL = "minimal"  # long chat prompts: store the residual stream only; the Lab upgrades lazily

RECEIPT_PROVES = "The segment's tokens were present in the target's input at the stated positions (delivery/access)."
RECEIPT_DOES_NOT_PROVE = ("That the target used, attended to, relied on or understood the segment; delivery is "
                          "necessary for causal use but is not evidence of it.")


def api_only_enabled() -> bool:
    return os.environ.get(API_ONLY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def new_case_id() -> str:
    import uuid

    return f"case_{uuid.uuid4().hex[:12]}"


# ====================================================================================== participants
def white_box_participants(adapter: Any) -> list[Participant]:
    info = adapter.info
    demo = bool(getattr(info, "is_demo", False))
    return [
        Participant(role="target", id=info.model_id,
                    description=("DEMO random-weight fixture model: its outputs carry no meaning." if demo else
                                 "Local white-box model: weights loaded in this process, activations accessible."),
                    sees=["messages rendered with the chat template (exact input_ids)", "its own generated tokens"]),
        Participant(role="statistical_scorer", id=STATS_SCORER_ID,
                    description="Deterministic analysis code; every number it reports is derived from stored trials.",
                    sees=["stored trials", "ground-truth fields", "measurements", "committed spec"]),
        Participant(role="investigator", id=SUGGESTER_ID,
                    description="Deterministic rules mapping claim type and context-segment kinds to experiment "
                                "templates. Its suggestions are hypotheses, never evidence.",
                    sees=["claim_type", "context segment kinds", "access_level", "registered templates"]),
    ]


def api_only_participants() -> list[Participant]:
    return [
        Participant(role="target", id="unknown (pasted transcript)",
                    description="The model that produced a pasted transcript. It is not loaded and is never called.",
                    sees=["unknown: only the pasted messages are known"], enabled=False),
        Participant(role="investigator", id=SUGGESTER_ID,
                    description="Deterministic rules; for API-only cases no white-box template is applicable.",
                    sees=["claim_type", "context segment kinds", "access_level", "registered templates"]),
    ]


def disabled_participants() -> list[dict[str, Any]]:
    return [
        {**LLM_INVESTIGATOR, "role": "investigator", "enabled": False},
        {"id": "activation-decoders", "role": "activation_decoder", "enabled": False, "status": "unsupported",
         "reason": "none configured: no lens, probe or activation oracle is attached as a participant"},
    ]


# ====================================================================================== helpers
def _strip_trailing_special(adapter: Any, ids: list[int]) -> list[int]:
    out = list(ids)
    while out and adapter.is_special(int(out[-1])):
        out.pop()
    return out


def _char_offsets(adapter: Any, ids: list[int]) -> list[tuple[int, int]]:
    """Per-token [start, end) character offsets within ``decode(ids)`` (prefix decoding; monotone)."""
    ends = [0]
    for i in range(1, len(ids) + 1):
        ends.append(max(ends[-1], len(adapter.decode(ids[:i]))))
    return [(ends[i], ends[i + 1]) for i in range(len(ids))]


def map_chars_to_generated_tokens(adapter: Any, content_ids: list[int], response_text: str, char_start: int,
                                  char_end: int, prompt_len: int) -> tuple[int, int] | None:
    """Absolute [start, end) token positions (in the run's input_ids) covering a char span of the response."""
    if adapter.decode(content_ids) != response_text:
        return None
    idx = [i for i, (s, e) in enumerate(_char_offsets(adapter, content_ids)) if e > char_start and s < char_end
           and e > s]
    if not idx:
        return None
    return prompt_len + idx[0], prompt_len + idx[-1] + 1


def control_token_strings(adapter: Any) -> list[str]:
    """Strings the tokenizer turns into SPECIAL (chat-control) tokens, e.g. ``<|im_start|>``, ``<|im_end|>``."""
    tok = getattr(adapter, "tokenizer", None)
    out = {str(t) for t in (getattr(tok, "all_special_tokens", None) or []) if t}
    for t in (getattr(tok, "added_tokens_decoder", None) or {}).values():
        if getattr(t, "special", False) and getattr(t, "content", None):
            out.add(str(t.content))
    return sorted(out, key=lambda x: (-len(x), x))


def reject_control_strings(adapter: Any, fields: dict[str, str | None]) -> None:
    """Refuse user text containing chat-control token strings: they would be tokenised as REAL control tokens
    and could forge conversation turns (validation defect D-I1). Text is data, never control."""
    specials = control_token_strings(adapter)
    for name, text in fields.items():
        if not text:
            continue
        found = [(text.find(sp), sp) for sp in specials if sp in text]
        if found:
            hit = min(found)[1]  # the first one in the text
            raise SelfLabError(f"{name} contains the chat-control token string {hit!r}; it would be tokenised as a "
                               "real control token and could forge conversation turns. Remove it.")


def _check_span(response_text: str, char_start: int, char_end: int) -> str:
    """The selected claim text. Whitespace-only spans are allowed (the DEMO fixture model emits newlines) and noted."""
    if not 0 <= char_start < char_end <= len(response_text):
        raise SelfLabError(f"claim span [{char_start}, {char_end}) is outside the response "
                           f"(length {len(response_text)})")
    return response_text[char_start:char_end]


def _span_notes(claim: str, claim_type_given: bool) -> list[str]:
    notes = []
    if not claim_type_given:
        notes.append(f"claim_type inferred by {CLAIM_TYPE_RULES_ID} (deterministic keyword rules).")
    if not claim.strip():
        notes.append("The selected claim span is whitespace only.")
    return notes


def _default_title(claim: str) -> str:
    c = " ".join(claim.split())
    return f"Claim: “{c[:72]}{'…' if len(c) > 72 else ''}”"


# ====================================================================================== API-only cases
def create_api_only_case(store: SelfLabStore, messages: list[dict], response_text: str, char_start: int,
                         char_end: int, receipts: list[dict] | None = None, claim_type: ClaimType | None = None,
                         title: str | None = None) -> Case:
    """A case built from a PASTED transcript. No model calls; unavailable unless explicitly enabled."""
    if not api_only_enabled():
        raise ForbiddenError(f"access_level 'api_only' is disabled; set {API_ONLY_ENV}=1 to enable pasted-transcript "
                             "inspection (no provider calls are made in either mode)")
    if not messages:
        raise SelfLabError("an API-only case needs the pasted messages")
    if len(messages) > MAX_MESSAGES:
        raise SelfLabError(f"at most {MAX_MESSAGES} messages")
    if len(response_text) > MAX_RESPONSE_CHARS:
        raise SelfLabError(f"response_text exceeds {MAX_RESPONSE_CHARS} characters")
    msgs = [Message.model_validate(m) for m in messages]
    claim = _check_span(response_text, char_start, char_end)
    rcpts: list[Receipt] = []
    for i, r in enumerate((receipts or [])[:MAX_RECEIPTS]):
        rcpts.append(Receipt(
            id=f"rcpt_pasted_{i}", kind="api_usage",
            statement="User-supplied receipt attached to a pasted transcript. NeuralScope did not verify it.",
            proves="Nothing independently verified: the content was supplied with a pasted transcript.",
            does_not_prove="That the stated delivery, tool call or usage happened, or that the model used it.",
            data={"supplied": r, "verified": False}))
    ct = claim_type or guess_claim_type(claim)
    case = Case(
        id=new_case_id(), created_at=now_iso(), title=title or _default_title(claim), access_level="api_only",
        run_id=None, messages=msgs, response_text=response_text,
        claim=ClaimSpan(text=claim, char_start=char_start, char_end=char_end), claim_type=ct, segments=[],
        receipts=rcpts, participants=api_only_participants(), provenance=None, is_fixture=False,
        notes=" ".join(["API-only view: pasted data inspection only. No model calls are made and white-box "
                        "templates are refused.", *_span_notes(claim, claim_type is not None)]))
    return store.insert_case(case)


# ====================================================================================== service
class SelfLabService:
    """Everything SelfLab does with the loaded model. One instance per (NeuralScopeService, store)."""

    def __init__(self, ns: Any, store: SelfLabStore, jobs: Any, recover: bool = True):
        self.ns = ns  # NeuralScopeService
        self.adapter = ns.adapter
        self.store = store
        self.engine = SelfLabEngine(store, jobs, service=ns, adapter=ns.adapter)
        self.recovered = self.engine.recover_interrupted() if recover else []

    # ------------------------------------------------------------------ info
    @property
    def is_fixture(self) -> bool:
        return bool(getattr(self.adapter.info, "is_demo", False))

    def participants(self) -> list[Participant]:
        return white_box_participants(self.adapter)

    def provenance(self, max_new_tokens: int | None = None) -> RunProvenance:
        dec = {"strategy": "greedy", "kv_cache": True}
        if max_new_tokens is not None:
            dec["max_new_tokens"] = max_new_tokens
        return run_provenance(self.adapter, decoding=dec, seed=None)

    # ------------------------------------------------------------------ Lab runs
    def _create_run(self, messages: list[dict], max_new_tokens: int, kind: str, notes: str) -> dict[str, Any]:
        if not hasattr(self.adapter, "tokenize_messages"):
            raise UnavailableError("the loaded adapter does not support tokenize_messages")
        return self.ns.create_run(messages=messages, max_new_tokens=int(max_new_tokens),
                                  max_prompt_tokens=SELFLAB_MAX_PROMPT_TOKENS, trace_level=CHAT_TRACE_LEVEL,
                                  kind=kind, notes=notes, tags=["selflab"])

    def _receipts(self, run_id: str, input_ids: list[int], messages: list[dict],
                  segments: list[ContextSegment]) -> list[Receipt]:
        out = []
        for i, seg in enumerate(segments):
            if seg.kind not in ("document", "user_hint"):
                continue
            text = messages[seg.message_index]["content"][seg.char_start:seg.char_end]
            ok_hash = sha256_text(text) == seg.sha256
            a, b = seg.token_start, seg.token_end
            decoded = self.adapter.decode(input_ids[a:b]) if a is not None and b is not None else ""
            verified = bool(a is not None and b is not None and ok_hash and text in decoded)
            label = "document" if seg.kind == "document" else "user hint"
            if verified:
                statement = (f"The {label} ({len(text)} characters, sha256 {seg.sha256[:12]}…) occupied tokens "
                             f"[{a}, {b}) of the target's input in run {run_id}; decoding those tokens reproduces "
                             "the text verbatim.")
            else:
                statement = (f"Delivery of the {label} (sha256 {seg.sha256[:12]}…) could NOT be verified in run "
                             f"{run_id}: its tokens were not located, or decoding them does not reproduce the text.")
            out.append(Receipt(
                id=f"rcpt_{run_id}_{i}", kind="context_delivery", statement=statement, proves=RECEIPT_PROVES,
                does_not_prove=RECEIPT_DOES_NOT_PROVE,
                data={"verified": verified, "run_id": run_id, "segment_kind": seg.kind,
                      "message_index": seg.message_index, "char_start": seg.char_start, "char_end": seg.char_end,
                      "token_start": a, "token_end": b, "n_tokens": (b - a) if a is not None and b is not None else 0,
                      "segment_sha256": seg.sha256, "segment_hash_ok": ok_hash,
                      "decoded_sha256": sha256_text(decoded) if decoded else None,
                      "input_ids_sha256": sha256_ids(input_ids), "check": "segment_text in decode(input_ids[a:b])"}))
        return out

    # ------------------------------------------------------------------ chat
    def chat(self, question: str, document: str | None = None, hint: str | None = None, system: str | None = None,
             messages: list[dict] | None = None, max_new_tokens: int = 24) -> dict[str, Any]:
        """Compose the user turn, run the target (a Lab run), return the response with segments and receipts."""
        if not question or not question.strip():
            raise SelfLabError("question is empty")
        if len(question) > MAX_QUESTION_CHARS or len(document or "") > MAX_DOCUMENT_CHARS or \
                len(hint or "") > MAX_HINT_CHARS or len(system or "") > MAX_SYSTEM_CHARS:
            raise SelfLabError("question/document/hint/system exceeds its length limit")
        if not 1 <= int(max_new_tokens) <= MAX_NEW_TOKENS:
            raise SelfLabError(f"max_new_tokens must be within 1..{MAX_NEW_TOKENS}")
        prior = [Message.model_validate(m).model_dump() for m in (messages or [])]
        if len(prior) + 2 > MAX_MESSAGES:
            raise SelfLabError(f"at most {MAX_MESSAGES} messages per chat")
        if any(len(m["content"]) > MAX_MESSAGE_CHARS for m in prior):
            raise SelfLabError(f"a message exceeds {MAX_MESSAGE_CHARS} characters")
        if system is not None and system.strip():
            if prior and prior[0]["role"] == "system":
                raise SelfLabError("give the system prompt either as `system` or as the first message, not both")
            prior = [{"role": "system", "content": system}] + prior
        if any(m["role"] == "system" for m in prior[1:]):
            raise SelfLabError("a system message is only allowed first")
        reject_control_strings(self.adapter, {"question": question, "document": document, "hint": hint,
                                              **{f"messages[{i}].content": m["content"] for i, m in enumerate(prior)}})
        content, segments = chat_user_message(question, document, hint, message_index=len(prior))
        msgs = prior + [{"role": "user", "content": content}]
        if sum(len(m["content"]) for m in msgs) > MAX_TOTAL_CHARS:
            raise SelfLabError(f"the chat exceeds {MAX_TOTAL_CHARS} characters in total")
        rec = self._create_run(msgs, max_new_tokens, kind="selflab_chat",
                               notes="SelfLab chat run (composed with selflab.compose.chat_user_message)")
        run_id = rec["id"]
        P = int(rec["prompt_len"])
        prompt_ids = list(rec["input_ids"][:P])
        tc = self.adapter.tokenize_messages(msgs, add_generation_prompt=True)
        tokenization_consistent = list(tc.input_ids) == prompt_ids
        segs: list[ContextSegment] = []
        for seg in segments:
            try:
                span = tc.char_span_to_tokens(seg.message_index, seg.char_start, seg.char_end)
            except ValueError:
                span = None
            if span is not None and not tokenization_consistent:
                span = None
            segs.append(seg.model_copy(update={"token_start": span[0] if span else None,
                                               "token_end": span[1] if span else None}))
        receipts = self._receipts(run_id, prompt_ids, msgs, segs)
        gen = list(rec["generated_ids"])
        content_ids = _strip_trailing_special(self.adapter, gen)
        response_text = self.adapter.decode(content_ids)
        gen_info = rec.get("generation") or {}
        prov = self.provenance(gen_info.get("max_new_tokens", max_new_tokens))
        data = {
            "run_id": run_id, "messages": msgs, "response_text": response_text, "response_token_ids": content_ids,
            "generated_ids": gen, "prompt_len": P, "segments": [s.model_dump(mode="json") for s in segs],
            "receipts": [r.model_dump(mode="json") for r in receipts], "provenance": prov.model_dump(mode="json"),
            "is_fixture": self.is_fixture or bool(rec.get("is_demo")), "stopped_by": gen_info.get("stopped_by"),
            "max_new_tokens": gen_info.get("max_new_tokens", max_new_tokens),
            "tokenization_consistent": tokenization_consistent, "trace_consistent": rec.get("trace_consistent"),
            "n_prompt_tokens": P, "n_generated_tokens": len(gen), "service_version": SERVICE_VERSION,
        }
        self.store.insert_chat(run_id, data)
        return {**{k: v for k, v in data.items() if k not in ("response_token_ids", "generated_ids")},
                "lab_url": f"/app?run={run_id}", "schema_version": SCHEMA_VERSION}

    # ------------------------------------------------------------------ cases
    def create_case(self, run_id: str, char_start: int, char_end: int, claim_type: ClaimType | None = None,
                    title: str | None = None) -> Case:
        rec = self.ns.store.get_run(run_id)
        if rec is None:
            raise NotFoundError(f"run {run_id} not found")
        if rec.get("model_id") != self.adapter.info.model_id:
            raise ConflictError(f"run {run_id} was produced by {rec.get('model_id')}; the loaded model is "
                                f"{self.adapter.info.model_id}")
        chat = self.store.get_chat(run_id)
        P = int(rec["prompt_len"])
        if chat is not None:
            messages = chat["messages"]
            response_text = chat["response_text"]
            content_ids = list(chat["response_token_ids"])
            segments = [ContextSegment.model_validate(s) for s in chat["segments"]]
            receipts = [Receipt.model_validate(r) for r in chat["receipts"]]
            prov = RunProvenance.model_validate(chat["provenance"])
        else:  # an ordinary Lab run: no composed segments, no receipts
            messages = rec.get("messages") or (
                ([{"role": "system", "content": rec["system_prompt"]}] if rec.get("system_prompt") else []) +
                [{"role": "user", "content": rec["prompt"]}])
            content_ids = _strip_trailing_special(self.adapter, list(rec.get("generated_ids") or []))
            response_text = self.adapter.decode(content_ids)
            segments, receipts = [], []
            prov = self.provenance((rec.get("generation") or {}).get("max_new_tokens"))
        claim = _check_span(response_text, char_start, char_end)
        span = map_chars_to_generated_tokens(self.adapter, content_ids, response_text, char_start, char_end, P)
        ct = claim_type or guess_claim_type(claim)
        note_parts = _span_notes(claim, claim_type is not None)
        if chat is None:
            note_parts.append("Built from an ordinary Lab run: no composed segments/receipts.")
        if span is None:
            note_parts.append("The claim could not be mapped to generated-token positions.")
        notes = " ".join(note_parts) or None
        case = Case(
            id=new_case_id(), created_at=now_iso(), title=title or _default_title(claim),
            access_level="white_box_local", run_id=run_id, messages=[Message.model_validate(m) for m in messages],
            response_text=response_text,
            claim=ClaimSpan(text=claim, char_start=char_start, char_end=char_end,
                            token_start=span[0] if span else None, token_end=span[1] if span else None),
            claim_type=ct, segments=segments, receipts=receipts, participants=self.participants(), provenance=prov,
            is_fixture=self.is_fixture or bool(rec.get("is_demo")), notes=notes)
        return self.store.insert_case(case)

    def case_detail(self, case_id: str) -> dict[str, Any]:
        case = self.store.require_case(case_id)
        return case_detail(self.store, case)

    # ------------------------------------------------------------------ templates & experiments
    def commit(self, spec: ExperimentSpec | dict) -> ExperimentRecord:
        return self.engine.commit(spec)

    def suggest(self, case_id: str) -> dict[str, Any]:
        return suggest(self.store.require_case(case_id), list_templates())

    # ------------------------------------------------------------------ open in Lab
    def _open_intervention_in_lab(self, trial: Trial, msgs: list[dict], specs: list[dict]) -> dict[str, Any]:
        """Trials whose outcome was produced under an intervention (e.g. CSP-1): replay the clean prompt as a Lab
        run, then the same intervention as a Lab intervention child of it, and compare both to the stored ids.

        Stored ids: the intervened output is ``trial.output_ids``; the clean baseline is
        ``trial.measurements["baseline"]["generated_ids"]`` when the template records it (otherwise the baseline is
        not compared and ``baseline_reproduced`` is None). ``reproduced`` requires the intervened output to match and
        the baseline not to mismatch.
        """
        base_stored = list(((trial.measurements or {}).get("baseline") or {}).get("generated_ids") or [])
        iv_stored = list(trial.output_ids)
        n = max(1, min(max(len(base_stored), len(iv_stored), 1), MAX_NEW_TOKENS))
        run = self._create_run(msgs, n, kind="selflab_replay",
                               notes=f"SelfLab deterministic replay of trial {trial.id} (clean prompt)")
        child = self.ns.run_intervention(run["id"], specs, max_new_tokens=n,
                                         notes=f"SelfLab replay of trial {trial.id} intervention")
        child_rec = self.ns.store.get_run(child["run"]["id"]) or {}
        base_replay = list(run["generated_ids"])
        iv_replay = list(child_rec.get("free_generated_ids") or [])

        def compare(stored: list[int], replay: list[int]) -> tuple[bool | None, int | None]:
            if not stored:
                return None, None
            k = min(len(stored), len(replay))
            first = next((i for i in range(k) if stored[i] != replay[i]), None)
            if first is None and len(replay) < len(stored):
                first = len(replay)
            return first is None, first

        base_ok, base_first = compare(base_stored, base_replay)
        iv_ok, iv_first = compare(iv_stored, iv_replay)
        reproduced = iv_ok is True and base_ok is not False
        diff = None if reproduced else {
            "compared": "token_ids", "baseline_reproduced": base_ok, "intervention_reproduced": iv_ok,
            "baseline_first_mismatch_index": base_first, "intervention_first_mismatch_index": iv_first,
            "stored_output_text": trial.output_text,
            "replay_output_text": self.adapter.decode(_strip_trailing_special(self.adapter, iv_replay)),
            "reason": ("the trial stored no intervened output to compare" if iv_ok is None else "outputs differ")}
        for rid, kind in ((run["id"], "lab_replay_baseline"), (child["run"]["id"], "lab_replay_intervention")):
            self.store.link_trial_run(trial.id, rid, {"kind": kind, "reproduced": reproduced, "compared": "token_ids"})
        return {"run_id": child["run"]["id"], "baseline_run_id": run["id"], "reproduced": reproduced,
                "baseline_reproduced": base_ok, "intervention_reproduced": iv_ok, "diff": diff,
                "trial_id": trial.id, "compared": "token_ids", "interventions": specs,
                "lab_url": f"/app?run={child['run']['id']}", "existing_run_ids": list(trial.run_ids),
                "is_fixture": self.is_fixture}

    def _replay_messages(self, template: Any, trial: Trial, rec: ExperimentRecord) -> list[dict] | None:
        case = self.store.get_case(rec.case_id) if rec.case_id else None
        plan = {p.index: p for p in self.store.get_plan(rec.id)}
        item = plan.get(trial.index)
        fn = getattr(template, "replay_messages", None)
        if callable(fn) and item is not None:
            # optional template hook: replay_messages(item, case) or replay_messages(item, case, spec)
            try:
                n_params = len(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                n_params = 2
            msgs = fn(item, case, rec.committed.spec) if n_params >= 3 else fn(item, case)
            if msgs:
                return [{"role": m["role"], "content": m["content"]} for m in msgs]
        for src in (trial.measurements, trial.parsed):
            for key in REPLAY_MESSAGE_KEYS:
                v = (src or {}).get(key)
                if isinstance(v, list) and v and all(isinstance(m, dict) and "role" in m for m in v):
                    return [{"role": m["role"], "content": m["content"]} for m in v]
        if item is not None:
            for key in REPLAY_MESSAGE_KEYS:
                v = item.params.get(key)
                if isinstance(v, list) and v:
                    return [{"role": m["role"], "content": m["content"]} for m in v]
        return None

    def open_in_lab(self, trial_id: str) -> dict[str, Any]:
        """Materialise a Lab run by deterministic replay of the trial's turn-1 messages and compare outputs."""
        trial = self.store.require_trial(trial_id)
        rec = self.store.require_experiment(trial.experiment_id)
        template = get_template(rec.committed.spec.template)
        msgs = self._replay_messages(template, trial, rec)
        if not msgs:
            raise ConflictError(f"trial {trial_id} has no replayable turn-1 messages (its template did not record "
                                "them; see INTERFACES §5 open-in-lab)")
        lab_fn = getattr(template, "lab_interventions", None)  # optional hook: lab_interventions(item) -> specs
        if callable(lab_fn):
            item = {p.index: p for p in self.store.get_plan(rec.id)}.get(trial.index)
            if item is None:
                raise ConflictError(f"trial {trial_id} has no plan item")
            specs = [dict(x) for x in (lab_fn(item) or [])]
            if specs:  # no intervention for this item → the plain replay below applies
                return self._open_intervention_in_lab(trial, msgs, specs)
        stored = list(trial.output_ids)
        n = len(stored) if stored else rec.committed.spec.budget.max_new_tokens
        n = max(1, min(int(n), MAX_NEW_TOKENS))
        run = self._create_run(msgs, n, kind="selflab_replay",
                               notes=f"SelfLab deterministic replay of trial {trial_id} (turn 1)")
        replay = list(run["generated_ids"])
        replay_text = self.adapter.decode(_strip_trailing_special(self.adapter, replay))
        if stored:
            compared = "token_ids"
            k = min(len(stored), len(replay))
            first = next((i for i in range(k) if stored[i] != replay[i]), None)
            prefix_ok = first is None
            reproduced = prefix_ok and len(replay) >= len(stored)
            if first is None and len(replay) < len(stored):
                first = len(replay)
        elif trial.output_text:
            compared = "text"
            reproduced = replay_text == trial.output_text or replay_text.strip() == trial.output_text.strip()
            first = None if reproduced else 0
        else:
            compared = "none"
            reproduced = False
            first = None
        diff = None
        if not reproduced:
            diff = {"compared": compared, "first_mismatch_index": first, "stored_n_tokens": len(stored),
                    "replay_n_tokens": len(replay), "stored_output_text": trial.output_text,
                    "replay_output_text": replay_text,
                    "reason": ("the trial stored no output to compare" if compared == "none" else
                               "replay was shorter than the stored output (server max_new_tokens cap)"
                               if first is not None and first >= len(replay) else "outputs differ")}
        self.store.link_trial_run(trial_id, run["id"], {"kind": "lab_replay", "reproduced": reproduced,
                                                        "compared": compared})
        return {"run_id": run["id"], "reproduced": reproduced, "diff": diff, "trial_id": trial_id,
                "compared": compared, "lab_url": f"/app?run={run['id']}", "existing_run_ids": list(trial.run_ids),
                "is_fixture": self.is_fixture}


# ====================================================================================== model-free helpers
def case_detail(store: SelfLabStore, case: Case) -> dict[str, Any]:
    exps = store.list_experiments(case_id=case.id, limit=1000)
    return {"case": case.model_dump(mode="json"), "experiments": [e.model_dump(mode="json") for e in exps],
            "evidence": [e.model_dump(mode="json") for e in store.list_evidence(case_id=case.id)]}


def templates_payload(store: SelfLabStore, case_id: str | None = None) -> list[dict[str, Any]]:
    case = store.require_case(case_id) if case_id else None
    out = []
    for t in sorted(list_templates(), key=lambda t: t.id):
        levels = list(template_access_levels(t))
        pm = getattr(t, "params_model", None)
        entry: dict[str, Any] = {
            "id": t.id, "title": getattr(t, "title", t.id), "description": getattr(t, "description", ""),
            "feature_status": getattr(t, "feature_status", "research_prototype"),
            "requires_case": bool(getattr(t, "requires_case", False)), "access_levels": levels,
            "params_schema": pm.model_json_schema() if pm is not None else {}, "default_spec": None,
        }
        if case is not None and case.access_level not in levels:
            entry["default_spec_error"] = f"not available for '{case.access_level}' cases"
        elif case is not None or not entry["requires_case"]:
            try:
                entry["default_spec"] = t.default_spec(case).model_dump(mode="json")
            except Exception as e:  # e.g. the case has no document segment
                entry["default_spec_error"] = f"{type(e).__name__}: {e}"
        else:
            entry["default_spec_error"] = "requires ?case_id="
        out.append(entry)
    return out


def evidence_payload(store: SelfLabStore, case_id: str | None = None,
                     experiment_id: str | None = None) -> list[EvidenceItem]:
    return store.list_evidence(case_id=case_id, experiment_id=experiment_id)


@functools.lru_cache(maxsize=1)
def _examples_cached() -> tuple[dict[str, Any], ...]:
    from neuralscope.selflab.datasets.source_corpus import preset_case

    p = preset_case(seed=0)
    return ({
        "title": f"Invented entity: {p['entity']} ({str(p['attribute']).replace('_', ' ')})",
        "document": p["document"],
        "hint": None,
        "question": p["question"],
        "replacement_text": p["replacement_text"],
        "source": "neuralscope.selflab.datasets.source_corpus.preset_case(seed=0)",
        "corpus_version": p.get("corpus_version"),
        "item_id": p.get("item_id"),
        "note": "Invented entity with a randomly assigned value; replacement_text is the same document with the "
                "value swapped for a controlled alternative (position-aligned).",
    },)


def examples_payload() -> dict[str, Any]:
    """Load-example presets for the chat panel: genuine corpus presets (never hand-written numbers)."""
    try:
        return {"examples": [dict(e) for e in _examples_cached()], "examples_error": None}
    except Exception as e:  # a corpus problem must not break /status
        return {"examples": [], "examples_error": f"{type(e).__name__}: {e}"}


def status_payload(sl: SelfLabService | None, model_status: str, model_error: str | None,
                   settings_is_demo: bool) -> dict[str, Any]:
    ready = sl is not None
    try:
        templates = [{"id": t.id, "title": getattr(t, "title", t.id),
                      "feature_status": getattr(t, "feature_status", "research_prototype"),
                      "requires_case": bool(getattr(t, "requires_case", False))}
                     for t in sorted(list_templates(), key=lambda t: t.id)]
    except Exception as e:  # pragma: no cover - a broken template module must not break /status
        templates = [{"error": f"{type(e).__name__}: {e}"}]
    api_only = api_only_enabled()
    return {
        "schema_version": SCHEMA_VERSION,
        "service_version": SERVICE_VERSION,
        "model": {"status": model_status, "error": model_error,
                  "provenance": model_provenance(sl.adapter).model_dump(mode="json") if ready else None},
        "is_fixture": sl.is_fixture if ready else bool(settings_is_demo),
        "access_levels": {
            "white_box_local": {"enabled": ready, "feature_status": "research_prototype",
                                "note": "the locally loaded model; experiments run on the single model worker"},
            "api_only": {"enabled": api_only, "feature_status": "api_only_view" if api_only else "unsupported",
                         "note": ("pasted-transcript inspection only; no provider calls; white-box templates refused"
                                  if api_only else f"disabled; set {API_ONLY_ENV}=1 to enable pasted-transcript "
                                                   "inspection (no provider calls in either mode)")},
        },
        "participants": [p.model_dump(mode="json") for p in white_box_participants(sl.adapter)] if ready else [],
        "disabled_participants": disabled_participants(),
        "features": {
            "chat": "research_prototype" if ready else "unsupported",
            "context_delivery_receipts": "implemented",
            "cases": "implemented",
            "experiments": "research_prototype",
            "open_in_lab": "research_prototype",
            "export": "implemented",
            "investigate_suggest": "implemented",
            "llm_investigator": "unsupported",
            "activation_decoders": "unsupported",
            "api_only": "api_only_view" if api_only else "unsupported",
        },
        "templates": templates,
        "limits": {"max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": SELFLAB_MAX_PROMPT_TOKENS,
                   "max_messages": MAX_MESSAGES, "max_message_chars": MAX_MESSAGE_CHARS,
                   "max_document_chars": MAX_DOCUMENT_CHARS, "max_question_chars": MAX_QUESTION_CHARS,
                   "max_hint_chars": MAX_HINT_CHARS, "max_total_chars": MAX_TOTAL_CHARS},
        "recovered_interrupted_experiments": list(sl.recovered) if ready else [],
        **examples_payload(),
    }
