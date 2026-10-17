"""Experiment A corpus: invented entities with randomly assigned categorical facts (INTERFACES.md §4).

Every entity is a generated pseudo-word ("Vornesk"), so the target has no prior knowledge of it. Every fact is
drawn by a seeded RNG from a closed set of common single-token words. For one item:

* ``v``      the value the document states in ``doc_v`` (and the hint-bearing document conditions);
* ``v_alt``  the controlled replacement the document states in ``doc_alt`` (≠ v);
* ``w``      the value the user suggests in ``doc_v_hint_conflict`` and ``no_doc_hint`` (≠ v, ≠ v_alt).

``doc_v`` and ``doc_alt`` prompts are POSITION-ALIGNED: with a tokenizer, ``make_items`` renders both prompts in
every family through the chat template and requires equal length and a single contiguous run of differing
positions. Draws that fail are redrawn deterministically (attempt counter in the RNG key) and logged.

Rendered text never contains condition names, item ids, seeds, family ids, or words such as "control",
"experiment", "condition" or "test". ``tests/test_selflab_datasets.py`` checks this for every condition × family.

Nothing here loads model weights or runs a forward pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from neuralscope.selflab.compose import Part, compose
from neuralscope.selflab.schemas import SOURCE_CATEGORIES, ContextSegment

CORPUS_VERSION = "source_corpus@1"
_NS = "neuralscope.selflab.source_corpus.v1"

SYSTEM_PROMPT: str = "You are a helpful assistant."
ANSWER_ABSTAIN: str = "insufficient evidence"
CONDITIONS: tuple[str, ...] = ("doc_v", "doc_alt", "no_doc", "doc_v_hint_conflict", "no_doc_hint", "doc_v_hint_agree")

# condition -> (which value the document states, which value the user suggests); None = absent
_CONDITION_TABLE: dict[str, tuple[str | None, str | None]] = {
    "doc_v": ("v", None),
    "doc_alt": ("v_alt", None),
    "no_doc": (None, None),
    "doc_v_hint_conflict": ("v", "w"),
    "no_doc_hint": (None, "w"),
    "doc_v_hint_agree": ("v", "v"),
}

# Substrings that must never appear in rendered text (checked case-insensitively; also used to filter entity names).
LEAK_SUBSTRINGS: tuple[str, ...] = (
    "control", "experiment", "condition", "test", "trial", "seed", "counterfactual", "hint",
    "doc_", "no_doc", "v_alt", "selflab", "neuralscope", "family",
)


# ------------------------------------------------------------------------------------------ attributes
@dataclass(frozen=True)
class Attribute:
    id: str
    values: tuple[str, ...]
    fact: str  # sentence with {E} (entity) and {V} (value); {V} appears exactly once
    phrase: str  # noun phrase with {E}, used in questions ("the colour of the flag of {E}")


# Every value below is ONE Qwen2.5 token in all three forms " v" (document context), "v" and "V" (bare answer forms;
# checked in tests). The value sets are pairwise disjoint so a distractor value can never coincide with v/v_alt/w.
ATTRIBUTES: dict[str, Attribute] = {a.id: a for a in (
    Attribute("flag_colour", ("red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "pink"),
              "The flag of {E} is {V}.", "the colour of the flag of {E}"),
    Attribute("market_day", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
              "The weekly market in {E} is held on {V}.", "the day on which the weekly market in {E} is held"),
    Attribute("festival_month", ("January", "February", "March", "April", "May", "June", "July", "August",
                                 "September", "October", "November", "December"),
              "The harvest festival of {E} takes place in {V}.", "the month of the harvest festival of {E}"),
    Attribute("emblem_animal", ("fox", "wolf", "bear", "crow", "mouse", "cat", "dog", "snake"),
              "The animal on the emblem of {E} is the {V}.", "the animal on the emblem of {E}"),
    Attribute("coin_metal", ("iron", "silver", "gold", "steel"),
              "The old coins of {E} were made of {V}.", "the metal that the old coins of {E} were made of"),
    Attribute("river_direction", ("north", "south", "east", "west"),
              "The main river of {E} flows {V}.", "the direction in which the main river of {E} flows"),
    Attribute("seal_shape", ("circle", "square", "triangle", "star", "diamond"),
              "The seal of {E} is shaped like a {V}.", "the shape of the seal of {E}"),
)}


# ------------------------------------------------------------------------------------------ families
@dataclass(frozen=True)
class Family:
    """A paraphrase family: document framing, hint wording, question wording, instruction, part order.

    A family is the held-out unit (INTERFACES §4): held-out families are used only in ``heldout_confirmation``.
    """

    id: str
    doc_header: str  # rendered as a separate "other" part before the document ("" = none)
    doc_intro: str  # first document sentence, template with {E}
    doc_style: str  # "prose" | "bullets"
    question: str  # template with {P} (attribute phrase)
    hint: str  # template with {W} (suggested value)
    instruction: str  # must name ANSWER_ABSTAIN exactly
    order: tuple[str, ...]  # order of "document", "question", "hint", "instruction"
    doc_noun: str  # how the source question refers to the document
    source_stem: str  # turn-2 question stem


FAMILIES: dict[str, Family] = {f.id: f for f in (
    Family(
        id="F1", doc_header="Document:", doc_intro="{E} is a small town.", doc_style="prose",
        question="What is {P}?", hint="I think it is {W}.",
        instruction=f'Reply with only the answer, or exactly "{ANSWER_ABSTAIN}" if the information is not available.',
        order=("document", "question", "hint", "instruction"), doc_noun="document",
        source_stem="Which of the following was your previous answer based on?",
    ),
    Family(
        id="F2", doc_header="Here is a short passage:",
        doc_intro="In the old regional records, {E} is described as a quiet village.", doc_style="prose",
        question="Can you tell me {P}?", hint="My colleague said it is {W}.",
        instruction=f'Answer with just the value. If the information is not available, reply exactly "{ANSWER_ABSTAIN}".',
        order=("document", "question", "hint", "instruction"), doc_noun="passage",
        source_stem="What was your last answer based on? Choose one option.",
    ),
    Family(
        id="F3", doc_header="Reference text:", doc_intro="{E} is a settlement on a plain.", doc_style="prose",
        question="Question: what is {P}?", hint="A friend told me the answer is {W}.",
        instruction=f'Give only the value as your reply. If it cannot be determined, reply exactly "{ANSWER_ABSTAIN}".',
        order=("document", "question", "hint", "instruction"), doc_noun="reference text",
        source_stem="Which source did your previous answer come from?",
    ),
    Family(
        id="F4", doc_header="Notes:", doc_intro="{E} is a hill town.", doc_style="bullets",
        question="I would like to know {P}.", hint="I remember hearing that the answer is {W}.",
        instruction=f'Respond with one word only, or with exactly "{ANSWER_ABSTAIN}" if the information is not available.',
        order=("question", "document", "hint", "instruction"), doc_noun="notes",
        source_stem="Think about the answer you just gave. Which of these was it based on?",
    ),
)}
DEFAULT_HELDOUT_FAMILIES: tuple[str, ...] = ("F4",)

_LETTERS = ("A", "B", "C", "D")


def families_for_split(split: str, heldout: tuple[str, ...] = DEFAULT_HELDOUT_FAMILIES) -> tuple[str, ...]:
    """Families allowed in a split: held-out families only in ``heldout_confirmation``, never elsewhere."""
    if split == "heldout_confirmation":
        return tuple(f for f in FAMILIES if f in heldout)
    if split in ("discovery", "debug_pilot"):
        return tuple(f for f in FAMILIES if f not in heldout)
    raise ValueError(f"unknown split {split!r}")


# ------------------------------------------------------------------------------------------ items
@dataclass(frozen=True)
class SourceItem:
    item_id: str
    entity: str
    attribute: str
    value_set: tuple[str, ...]
    v: str
    v_alt: str
    w: str
    # additive optional fields (same for every condition of the item, so doc_v/doc_alt differ only at the value)
    distractors: tuple[tuple[str, str], ...] = field(default=())  # (attribute id, value) facts about the entity
    target_index: int = 0  # position of the target fact among the fact sentences


_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "v", "z", "br", "dr", "gr", "kr", "pl", "tr", "vr", "th")
_VOWELS = ("a", "e", "i", "o", "u", "a", "e", "o")
_CODAS = ("", "", "", "n", "r", "l", "sk", "nd", "rn", "m", "x")


_ALL_VALUES_LOWER: tuple[str, ...] = tuple(sorted({v.lower() for a in ATTRIBUTES.values() for v in a.values}))


def _pseudo_word(rng: random.Random) -> str:
    """An invented name; never contains a leak substring or any attribute value (so values occur once per text)."""
    while True:
        n_syll = rng.choice((2, 2, 3))
        w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(n_syll))
        if (5 <= len(w) <= 10 and not any(s in w for s in LEAK_SUBSTRINGS)
                and not any(v in w for v in _ALL_VALUES_LOWER)):
            return w.capitalize()


def _encode_messages(tokenizer: Any, messages: list[dict]) -> list[int]:
    """Token ids of the exact model input. Accepts an SA2 adapter (``tokenize_messages``) or an HF tokenizer."""
    if hasattr(tokenizer, "tokenize_messages"):
        return list(tokenizer.tokenize_messages(messages, add_generation_prompt=True).input_ids)
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        r = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(r, "keys"):
            r = r["input_ids"]
        return [int(t) for t in r]
    return list(tokenizer.encode(messages[-1]["content"]))


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def check_alignment(item: SourceItem, tokenizer: Any, families: tuple[str, ...] | None = None) -> dict:
    """Render ``doc_v`` and ``doc_alt`` in each family; require equal token length and ONE contiguous diff run.

    Returns ``{"ok": bool, "per_family": {fid: {"n_tokens_v", "n_tokens_alt", "diff_positions"}}}``.
    """
    out: dict[str, Any] = {"ok": True, "per_family": {}}
    for fid in families or tuple(FAMILIES):
        ids_v = _encode_messages(tokenizer, render_turn1(item, "doc_v", fid)[0])
        ids_a = _encode_messages(tokenizer, render_turn1(item, "doc_alt", fid)[0])
        rec: dict[str, Any] = {"n_tokens_v": len(ids_v), "n_tokens_alt": len(ids_a), "diff_positions": []}
        ok = len(ids_v) == len(ids_a)
        if ok:
            diff = [i for i, (x, y) in enumerate(zip(ids_v, ids_a)) if x != y]
            rec["diff_positions"] = diff
            ok = bool(diff) and diff == list(range(diff[0], diff[-1] + 1))
        out["per_family"][fid] = rec
        out["ok"] = out["ok"] and ok
    return out


def _attribute_for(seed: int, i: int) -> str:
    ids = sorted(ATTRIBUTES)
    block = random.Random(f"{_NS}:attr:{seed}:{i // len(ids)}")
    block.shuffle(ids)
    return ids[i % len(ids)]


def _draw(seed: int, i: int, attempt: int, attribute: str) -> SourceItem:
    rng = random.Random(f"{_NS}:item:{seed}:{i}:{attempt}")
    attr = ATTRIBUTES[attribute]
    v, v_alt, w = rng.sample(attr.values, 3)
    others = [a for a in sorted(ATTRIBUTES) if a != attribute]
    n_distr = rng.choice((1, 2))
    distractors = tuple((a, rng.choice(ATTRIBUTES[a].values)) for a in rng.sample(others, n_distr))
    return SourceItem(
        item_id=f"src-s{seed}-{i:04d}", entity=_pseudo_word(rng), attribute=attribute, value_set=attr.values,
        v=v, v_alt=v_alt, w=w, distractors=distractors, target_index=rng.randrange(n_distr + 1),
    )


def make_items_with_log(n: int, seed: int, tokenizer: Any = None,
                        max_attempts: int = 50) -> tuple[list[SourceItem], list[dict]]:
    """Generate ``n`` items deterministically from ``seed``; return ``(items, redraw_log)``.

    Items are prefix-stable (``make_items(5, s)`` is a prefix of ``make_items(10, s)``). A draw is rejected and
    redrawn (attempt + 1) when its entity name repeats an earlier one, or — with a tokenizer — when the entity is a
    single token (could be a real word) or the doc_v/doc_alt prompts are not position-aligned in some family.
    Every rejection is a log entry.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    items: list[SourceItem] = []
    log: list[dict] = []
    seen: set[str] = set()
    for i in range(n):
        attribute = _attribute_for(seed, i)
        for attempt in range(max_attempts):
            it = _draw(seed, i, attempt, attribute)
            reason = None
            if it.entity in seen:
                reason = "duplicate_entity"
            elif tokenizer is not None:
                if len(_encode_text(tokenizer, " " + it.entity)) < 2:
                    reason = "entity_single_token"
                else:
                    chk = check_alignment(it, tokenizer)
                    if not chk["ok"]:
                        reason = "value_misaligned"
            if reason is None:
                items.append(it)
                seen.add(it.entity)
                break
            log.append({"index": i, "attempt": attempt, "reason": reason, "entity": it.entity,
                        "attribute": it.attribute, "v": it.v, "v_alt": it.v_alt, "w": it.w})
        else:
            raise ValueError(f"item {i}: no acceptable draw in {max_attempts} attempts")
    return items, log


def make_items(n: int, seed: int, tokenizer: Any = None) -> list[SourceItem]:
    return make_items_with_log(n, seed, tokenizer)[0]


# ------------------------------------------------------------------------------------------ rendering
def _fill(template: str, **kw: str) -> str:
    out = template
    for k, val in kw.items():
        out = out.replace("{" + k + "}", val)
    return out


def render_document(item: SourceItem, family: str, value: str) -> str:
    """The document body (no header) stating ``value`` for the target attribute, plus the item's distractors."""
    fam = FAMILIES[family]
    facts = [_fill(ATTRIBUTES[a].fact, E=item.entity, V=val) for a, val in item.distractors]
    facts.insert(item.target_index, _fill(ATTRIBUTES[item.attribute].fact, E=item.entity, V=value))
    sentences = [_fill(fam.doc_intro, E=item.entity)] + facts
    if fam.doc_style == "bullets":
        return "\n".join("- " + s for s in sentences)
    return " ".join(sentences)


def render_question(item: SourceItem, family: str) -> str:
    return _fill(FAMILIES[family].question, P=_fill(ATTRIBUTES[item.attribute].phrase, E=item.entity))


def render_turn1(item: SourceItem, condition: str, family: str) -> tuple[list[dict], list[ContextSegment], dict]:
    """Turn-1 messages for one (item, condition, family): system + user (composed, message_index=1)."""
    if condition not in _CONDITION_TABLE:
        raise ValueError(f"unknown condition {condition!r}")
    fam = FAMILIES[family]
    doc_key, hint_key = _CONDITION_TABLE[condition]
    doc_value = getattr(item, doc_key) if doc_key else None
    hint_value = getattr(item, hint_key) if hint_key else None
    doc_text = render_document(item, family, doc_value) if doc_value is not None else None
    hint_text = _fill(fam.hint, W=hint_value) if hint_value is not None else None
    question_text = render_question(item, family)

    parts: list[Part] = []
    for slot in fam.order:
        if slot == "document" and doc_text is not None:
            if fam.doc_header:
                parts.append(Part("other", fam.doc_header))
            parts.append(Part("document", doc_text))
        elif slot == "hint" and hint_text is not None:
            parts.append(Part("user_hint", hint_text))
        elif slot == "question":
            parts.append(Part("question", question_text))
        elif slot == "instruction":
            parts.append(Part("other", fam.instruction))
    content, segments = compose(parts, message_index=1)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    meta = {
        "document_value": doc_value, "hint_value": hint_value, "family": family, "condition": condition,
        "entity": item.entity, "attribute": item.attribute, "item_id": item.item_id,
        "document_text": doc_text, "hint_text": hint_text, "question_text": question_text,
        "corpus_version": CORPUS_VERSION,
    }
    return messages, segments, meta


def _option_texts(family: str) -> dict[str, str]:
    noun = FAMILIES[family].doc_noun
    return {
        "document": f"The {noun} included in my message",
        "user_suggestion": "The suggestion in my message about what the answer might be",
        "other_or_unknown": "Your own background knowledge, or something else",
        "insufficient_evidence": "Nothing: the information was not available, so you gave no value",
    }


def render_source_question(label_order: list[str], family: str) -> tuple[str, dict[str, str]]:
    """Turn-2 user content: stem, options A..D in ``label_order``, "Reply with a single letter."."""
    if sorted(label_order) != sorted(SOURCE_CATEGORIES) or len(label_order) != len(SOURCE_CATEGORIES):
        raise ValueError(f"label_order must be a permutation of {SOURCE_CATEGORIES}")
    opts = _option_texts(family)
    letter_map = {_LETTERS[i]: cat for i, cat in enumerate(label_order)}
    lines = [FAMILIES[family].source_stem]
    lines += [f"{letter}. {opts[cat]}" for letter, cat in letter_map.items()]
    lines.append("Reply with a single letter.")
    return "\n".join(lines), letter_map


def render_turn2(turn1_messages: list[dict], answer_text: str, label_order: list[str],
                 family: str) -> tuple[list[dict], dict[str, str]]:
    """Convenience: turn 1 + the assistant's answer VERBATIM + the constrained source question."""
    content, letter_map = render_source_question(label_order, family)
    msgs = [dict(m) for m in turn1_messages]
    msgs += [{"role": "assistant", "content": answer_text}, {"role": "user", "content": content}]
    return msgs, letter_map


def label_orders(rng_or_seed: random.Random | int, n_items: int) -> list[tuple[list[str], list[str]]]:
    """For each item: a uniformly random permutation of SOURCE_CATEGORIES and its reversal."""
    rng = rng_or_seed if isinstance(rng_or_seed, random.Random) else random.Random(f"{_NS}:labels:{rng_or_seed}")
    out = []
    for _ in range(n_items):
        perm = list(SOURCE_CATEGORIES)
        rng.shuffle(perm)
        out.append((perm, list(reversed(perm))))
    return out


# ------------------------------------------------------------------------------------------ UI preset
def preset_case(seed: int = 0) -> dict:
    """An example for the SelfLab chat panel (composed there with ``compose.chat_user_message``).

    The question asks for a short answer AND one sentence naming the source, so the user can select a
    source-attribution claim span. ``replacement_text`` is the same document with v replaced by v_alt
    (position-aligned for the Qwen2.5 tokenizer; same token count).
    """
    item = make_items(1, seed)[0]
    phrase = _fill(ATTRIBUTES[item.attribute].phrase, E=item.entity)
    return {
        "question": (f"What is {phrase}? Give the answer in a few words, then add one sentence saying "
                     f"which source your answer was based on."),
        "document": render_document(item, "F1", item.v),
        "replacement_text": render_document(item, "F1", item.v_alt),
        "expected_values": {"v": item.v, "v_alt": item.v_alt},
        "hint": _fill(FAMILIES["F1"].hint, W=item.w),
        "hint_value": item.w,
        "entity": item.entity,
        "attribute": item.attribute,
        "item_id": item.item_id,
        "corpus_version": CORPUS_VERSION,
    }
