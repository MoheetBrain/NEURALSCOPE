r"""Conservative parsers for SelfLab target outputs. They never guess.

Epistemic role (same contract as ``neuralscope.mathlab.parser``): a parser decides only *what the
output commits to*, never *whether it is right*. When the text does not commit to exactly one
option/value the parser returns ``ambiguous``, ``unparseable`` or ``refusal`` instead of a label,
because a sloppy parse would silently turn hedged or adversarial text into a scored self-report.
Parsed labels are compared with causal ground truth later, so a false "ok" is worse than a miss.

The parsers treat target text as DATA. Nothing in it is executed or followed.

``parse_source_report(text, letter_map)``
-----------------------------------------
``letter_map`` maps the option letters shown to the target ("A".."D") to source categories.
Decision procedure (``rule`` records which step produced the value):

0. Clean: drop chat control tokens (``<|im_end|>`` ...), markdown emphasis, surrounding quotes and
   whitespace. Empty → ``unparseable``. Longer than ``MAX_REPORT_CHARS`` → ``unparseable``.
1. Adversarial structure first: a disjunction of letters ("A or B", "A/B", "A and B") or a negated
   letter ("not A", "isn't B") → ``ambiguous``.
2. Letter extraction, structured forms only (a bare capital "A" in prose is an English article,
   not an answer): the whole text is one letter ("B", "(B)", "B)", "B.", "[B]", "**B**");
   the text starts with "B)", "B.", "B:", "(B)", "B -" followed by option text; or an explicit
   marker anywhere ("Answer: B", "answer is B", "option B", "choice (B)", "letter B"); or "(B)" anywhere.
   Several different letters → ``ambiguous``. One letter: if the rest of the text names a category
   phrase of a DIFFERENT category (e.g. "B) the document" when B is the user's suggestion) →
   ``ambiguous``; a negated phrase does not count. Otherwise ``ok`` (``rule="letter"``).
3. No letter: category phrases (``CATEGORY_PHRASES``; the option texts of the source corpus plus
   common paraphrases). Exactly one category named, not negated → ``ok`` with ``letter=None``
   (``rule="phrase"``). Several categories → ``ambiguous``.
4. Nothing usable: a refusal/hedge phrase ("I cannot say", "I don't know", "not sure") →
   ``refusal``; otherwise ``unparseable``. A refusal phrase together with a letter or phrase →
   ``ambiguous`` (the output both hedges and commits).

``parse_answer(text, value_set, abstain_phrase)``
-------------------------------------------------
Lower-cases, strips control tokens and punctuation, then finds whole-word mentions of the values
(longest values first, so "dark green" is not also counted as "green") and of the abstain phrase.
Exactly one distinct value and no abstention → ``category="value"``; only the abstain phrase →
``"abstain"``; a negated value ("not red") → ``ambiguous``; several values, or a value plus the
abstain phrase → ``ambiguous`` (category ``unparseable``); none → ``unparseable``.

Known limitation: neither parser can tell that a generation was cut off by the token limit
(``"Bl"`` from ``"Blue"`` is simply unparseable; ``"B) The sugg"`` still commits to "B"). Callers
record ``stopped_by`` and count truncated outputs separately.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Mapping, Sequence

PARSING_VERSION = "selflab.parsing@1"
MAX_REPORT_CHARS = 2000
MAX_ANSWER_CHARS = 2000

ReportStatus = Literal["ok", "ambiguous", "unparseable", "refusal"]

__all__ = ["PARSING_VERSION", "CATEGORY_PHRASES", "REFUSAL_PATTERNS", "parse_source_report", "parse_answer", "is_refusal",
           "clean_text"]

_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]{1,64}\|>|</?s>")
_EMPHASIS_RE = re.compile(r"[*_`#]+")

# Phrases that NAME a source category in free text. Kept specific on purpose: a generic word such
# as "other" or "text" would match ordinary prose. Includes every option wording used by the
# source corpus families (document/passage/reference text/notes).
CATEGORY_PHRASES: dict[str, tuple[str, ...]] = {
    "document": ("document", "passage", "reference text", "the notes", "notes included", "the text you provided",
                 "the provided text", "the text included"),
    "user_suggestion": ("suggestion", "you suggested", "suggested", "your colleague", "your friend",
                        "what you said", "what you told me", "you remembered", "remember hearing"),
    "other_or_unknown": ("background knowledge", "own knowledge", "prior knowledge", "general knowledge",
                         "my knowledge", "something else", "my training", "training data", "unknown source"),
    "insufficient_evidence": ("insufficient evidence", "insufficient information", "not enough information",
                              "not enough evidence", "was not available", "wasn't available", "no value",
                              "nothing: the information"),
}

REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\bcannot say\b", r"\bcan'?t say\b", r"\bcannot (?:answer|determine|tell|know|be sure)\b",
    r"\bcan'?t (?:answer|determine|tell|know|be sure)\b", r"\bi (?:do not|don'?t) know\b", r"\bunable to\b",
    r"\bnot sure\b", r"\bunsure\b", r"\bi'?m sorry\b", r"\bas an ai\b", r"\bno way to know\b", r"\bunclear\b",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_NEGATION_WORDS = r"(?:not|never|isn'?t|wasn'?t|aren'?t|weren'?t|didn'?t|doesn'?t|no)"


def clean_text(text: str | None) -> str:
    """Drop control tokens and markdown emphasis; normalise whitespace and curly quotes."""
    if not text:
        return ""
    t = _SPECIAL_TOKEN_RE.sub(" ", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = _EMPHASIS_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _result(status: ReportStatus, category: str | None = None, letter: str | None = None, rule: str | None = None,
            note: str = "") -> dict:
    return {"category": category if status == "ok" else None, "letter": letter if status == "ok" else None,
            "status": status, "rule": rule, "note": note, "parser": PARSING_VERSION}


def _letter_class(letters: Iterable[str]) -> str:
    return "".join(sorted({re.escape(x) for x in letters}))


def _negated(text: str, start: int, window: int = 24) -> bool:
    """True when a negation word closes in on the match from the left (within ``window`` chars, same clause)."""
    left = text[max(0, start - window):start]
    left = re.split(r"[.;:!?,]", left)[-1]
    return re.search(rf"\b{_NEGATION_WORDS}\b[^.;:!?,]*$", left, re.IGNORECASE) is not None


def _find_phrases(text: str, phrases: Mapping[str, Sequence[str]]) -> tuple[set[str], set[str]]:
    """Categories named (non-negated) and categories only named under negation."""
    low = text.lower()
    named, negated = set(), set()
    for cat, ps in phrases.items():
        for p in ps:
            for m in re.finditer(rf"(?<![a-z]){re.escape(p.lower())}(?![a-z])", low):
                (negated if _negated(low, m.start()) else named).add(cat)
    return named, negated - named


def parse_source_report(text: str | None, letter_map: Mapping[str, str], *,
                        category_phrases: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Parse a constrained source report. Returns ``{"category", "letter", "status", "rule", "note", "parser"}``.

    ``status`` ∈ {"ok", "ambiguous", "unparseable", "refusal"}; ``category``/``letter`` are None unless "ok".
    """
    if not letter_map:
        raise ValueError("letter_map is empty")
    phrases = category_phrases or CATEGORY_PHRASES
    letters = [str(k).upper() for k in letter_map]
    lmap = {str(k).upper(): v for k, v in letter_map.items()}
    t = clean_text(text)
    if not t:
        return _result("unparseable", rule="empty", note="empty output")
    if len(t) > MAX_REPORT_CHARS:
        return _result("unparseable", rule="too_long", note=f"longer than {MAX_REPORT_CHARS} chars; not scanned")
    L = _letter_class(letters)
    stripped = t.strip(" \t\"'.,;:!?")

    # 1. adversarial structure
    if re.search(rf"(?<![A-Za-z])\(?([{L}])\)?\s*(?:or|and|/|&|,)\s*\(?([{L}])\)?(?![A-Za-z])", t):
        return _result("ambiguous", rule="letter_disjunction", note="several options offered together")
    neg = re.search(rf"\b{_NEGATION_WORDS}\s+(?:option\s+|answer\s+|letter\s+)?\(?([{L}])\)?(?![A-Za-z])", t,
                    re.IGNORECASE)
    if neg and neg.group(1).upper() in letters and (neg.group(1).isupper() or "(" in neg.group(0)):
        return _result("ambiguous", rule="negated_letter", note="an option letter is negated")

    # 2. letters in structured forms
    found: list[tuple[str, int, int]] = []  # (letter, start, end) of the letter token incl. punctuation
    maybe_article = False  # a leading bare "A " may be the English article, not option A
    m = re.fullmatch(rf"[\(\[]?\s*([{L}])\s*[\)\]]?", stripped, re.IGNORECASE)
    if m:
        found.append((m.group(1).upper(), 0, len(t)))
    else:
        lead = re.match(rf"\s*(?:\(([{L}])\)|\[([{L}])\]|([{L}])\s*[\).:,\]]|([{L}])\s+-\s)", t)
        if lead:
            g = next(x for x in lead.groups() if x)
            found.append((g.upper(), lead.start(), lead.end()))
        else:
            bare = re.match(rf"\s*([{L}])\s+(?=\S)", t)  # "B because ...": a bare leading capital letter
            if bare:
                found.append((bare.group(1), bare.start(), bare.end()))
                maybe_article = bare.group(1) == "A"
        # structured letters ANYWHERE: "B." / "C)" / "D:" followed by whitespace or the end ("B. C.", "Also D.")
        for mm in re.finditer(rf"(?<![A-Za-z0-9])([{L}])[\).:](?=\s|$)", t):
            found.append((mm.group(1), mm.start(), mm.end()))
        for mm in re.finditer(rf"\b(?:answer|option|choice|letter)\s*(?:is|was|would be|:|=|-)?\s*[\(\[]?([{L}])"
                              rf"[\)\]]?(?![A-Za-z])", t, re.IGNORECASE):
            if mm.group(1).isupper() or "(" in mm.group(0) or "[" in mm.group(0):
                found.append((mm.group(1).upper(), mm.start(), mm.end()))
        for mm in re.finditer(rf"\(([{L}])\)", t):
            found.append((mm.group(1).upper(), mm.start(), mm.end()))
    distinct = sorted({f[0] for f in found})
    refusal = _REFUSAL_RE.search(t) is not None
    if len(distinct) > 1:
        return _result("ambiguous", rule="several_letters", note=f"letters {distinct} all appear")
    if len(distinct) == 1:
        letter = distinct[0]
        cat = lmap[letter]
        rest = t
        for _, s, e in sorted(found, key=lambda f: -f[1]):
            rest = rest[:s] + " " + rest[e:]
        named, _ = _find_phrases(rest, phrases)
        if named - {cat}:
            return _result("ambiguous", rule="letter_phrase_conflict",
                           note=f"letter {letter} ({cat}) but text names {sorted(named - {cat})}")
        if maybe_article and len(found) == 1 and named != {cat}:
            return _result("ambiguous", rule="article_or_letter",
                           note="leading 'A' may be an article; the text does not confirm option A")
        if refusal:
            return _result("ambiguous", rule="letter_with_hedge", note="a letter together with a refusal/hedge")
        return _result("ok", category=cat, letter=letter, rule="letter",
                       note="letter and option text agree" if named else "single letter")

    # 3. category phrases without a letter
    named, negated_only = _find_phrases(t, phrases)
    if len(named) > 1:
        return _result("ambiguous", rule="several_categories", note=f"text names {sorted(named)}")
    if len(named) == 1:
        if refusal:
            return _result("ambiguous", rule="phrase_with_hedge", note="a category phrase together with a hedge")
        return _result("ok", category=next(iter(named)), letter=None, rule="phrase",
                       note="category named without a letter")
    if negated_only:
        return _result("ambiguous", rule="negated_category", note=f"only negated mentions of {sorted(negated_only)}")
    # 4. nothing usable
    if refusal:
        return _result("refusal", rule="refusal", note="declines or hedges without choosing")
    return _result("unparseable", rule="no_option", note="no option letter or category phrase")


# ------------------------------------------------------------------------------------------ answers
def _strip_punct(s: str) -> str:
    s = re.sub(r"[^\w\s'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _value_pattern(v: str) -> re.Pattern:
    """Lower-case values (``red``) match case-insensitively. Values with capitals (``May``, ``March``, ``Monday``)
    match only in their canonical or all-caps spelling inside longer text, so "may vary" is not the month May.
    A whole output that is just the value matches in any case (handled by the caller)."""
    cv = _strip_punct(v)
    if cv == cv.lower():
        return re.compile(rf"(?<![\w]){re.escape(cv)}(?![\w])", re.IGNORECASE)
    return re.compile(rf"(?<![\w])(?:{re.escape(cv)}|{re.escape(cv.upper())})(?![\w])")


def parse_answer(text: str | None, value_set: Sequence[str], abstain_phrase: str) -> dict:
    """Parse a short answer against a closed value set.

    Returns ``{"normalized", "category", "status", "values_found", "rule", "note", "parser"}`` with
    ``category`` ∈ {"value", "abstain", "unparseable"} and ``status`` ∈ {"ok", "ambiguous", "unparseable"}.
    ``normalized`` is the canonical value from ``value_set`` (its original spelling), the abstain phrase, or None.
    """
    base = {"parser": PARSING_VERSION}
    t = _strip_punct(clean_text(text or ""))  # case kept
    if not t:
        return {**base, "normalized": None, "category": "unparseable", "status": "unparseable", "values_found": [],
                "rule": "empty", "note": "empty output"}
    if len(t) > MAX_ANSWER_CHARS:
        return {**base, "normalized": None, "category": "unparseable", "status": "unparseable", "values_found": [],
                "rule": "too_long", "note": f"longer than {MAX_ANSWER_CHARS} chars"}
    for v in value_set:  # the whole output is exactly one value (any case)
        if t.lower() == _strip_punct(v).lower():
            return {**base, "normalized": v, "category": "value", "status": "ok", "values_found": [v],
                    "rule": "whole_text", "note": ""}
    work = f" {t} "
    abst = _strip_punct(abstain_phrase)
    abst_re = re.compile(rf"(?<![\w]){re.escape(abst)}(?![\w])", re.IGNORECASE) if abst else None
    abstained = bool(abst_re and abst_re.search(work))
    if abstained:
        work = abst_re.sub(" ", work)
    found: list[str] = []
    negated: list[str] = []
    for v in sorted(value_set, key=lambda x: -len(_strip_punct(x))):
        if not _strip_punct(v):
            continue
        pat = _value_pattern(v)
        hits = list(pat.finditer(work))
        if hits:
            if any(_negated(work, h.start()) for h in hits):
                negated.append(v)
            else:
                found.append(v)
            work = pat.sub(" ", work)
    if negated:
        return {**base, "normalized": None, "category": "unparseable", "status": "ambiguous",
                "values_found": found + negated, "rule": "negated_value", "note": f"negated mention of {negated}"}
    if len(found) > 1:
        return {**base, "normalized": None, "category": "unparseable", "status": "ambiguous", "values_found": found,
                "rule": "several_values", "note": f"several values mentioned: {found}"}
    if found and abstained:
        return {**base, "normalized": None, "category": "unparseable", "status": "ambiguous", "values_found": found,
                "rule": "value_and_abstain", "note": "a value and the abstain phrase both appear"}
    if found:
        return {**base, "normalized": found[0], "category": "value", "status": "ok", "values_found": found,
                "rule": "single_value", "note": ""}
    if abstained:
        return {**base, "normalized": abstain_phrase, "category": "abstain", "status": "ok", "values_found": [],
                "rule": "abstain_phrase", "note": ""}
    return {**base, "normalized": None, "category": "unparseable", "status": "unparseable", "values_found": [],
            "rule": "no_value", "note": "no value from the set and no abstain phrase"}


def is_refusal(text: str | None) -> bool:
    """True when the cleaned text contains a refusal/hedge phrase (``REFUSAL_PATTERNS``)."""
    return _REFUSAL_RE.search(clean_text(text)) is not None
