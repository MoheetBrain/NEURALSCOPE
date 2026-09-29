r"""Conservative extraction of a model's final numeric answer from generated text.

Epistemic role
--------------
The parser sits between raw model output and the oracle. Its only job is to
decide *what value the model committed to*, never *whether that value is right*.
It is deliberately conservative: when the text does not commit to exactly one
value it returns ``"ambiguous"`` or ``"unparseable"`` instead of guessing,
because a sloppy parse could silently turn a hedged or unfinished output into a
CORRECT/INCORRECT label, and those labels are later used as ground truth when
comparing the neural traces of correct and incorrect answers.

The parser never evaluates expressions found in model text. It recognises
numerals only -- integers, thousands-separated integers (``1,022``), decimals,
``p/q`` and ``\frac{p}{q}`` -- and, for a purely arithmetic chain such as
``37 + 58 = 95``, takes the value after the final ``=`` without computing the
left-hand side.

Decision procedure
------------------
``ParsedAnswer.rule`` records which rule produced the value.

0. ``whole_text``: after removing chat special tokens (``<|im_end|>`` ...),
   whitespace, trailing punctuation and markdown/LaTeX decoration (``**``,
   ``$``, ``\(\)``, ``\boxed{}`` ...), the whole output is a single numeral.
1. Explicit markers, collected over the whole text:

   * ``boxed``: ``\boxed{...}``
   * ``answer``: "the answer is", "Answer:", "the sum/product/result is",
     "the larger fraction is", "N is the answer", and -- for rational answers
     only -- "N is larger"
   * ``x_equals``: "x = 6", "x=6", "x = 18/3 = 6"
   * ``first_line``: a bare numeral on the first line of a multi-line output
   * ``equation``: the whole output is an arithmetic chain (``73 * 14 = 1022``)
   * ``copula``: a single sentence ending in "... is 95" / "... equals 95"

   A hedged marker ("the answer is 95 or 96") makes the result ``ambiguous``.
   Markers that disagree with each other make it ``ambiguous``. Otherwise the
   value of the highest-priority marker (its last occurrence) is returned.
2. ``lone_number`` fallback (no usable marker): if the text holds exactly one
   distinct numeral, it is accepted only when it stands at the very start or
   end of the text, is not a list enumerator ("1. First, ..."), and the text
   has no hedging or negation words. Several different numerals give
   ``ambiguous``; none gives ``unparseable``.

Known limitation: the parser cannot tell that a generation was cut off by a
token limit (``"10"`` truncated from ``"1022"`` looks like a finished answer).
Callers should record whether generation stopped at EOS and treat truncated
outputs separately.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import islice
from typing import Any, Literal

from ._types import ANSWER_TYPES

__all__ = ["ParsedAnswer", "ParseStatus", "parse_answer", "MAX_PARSE_CHARS"]

ParseStatus = Literal["ok", "unparseable", "ambiguous"]

MAX_PARSE_CHARS = 20_000
"""Outputs longer than this are refused (``unparseable``) rather than scanned."""


@dataclass(frozen=True)
class ParsedAnswer:
    """The value a model output commits to, or why no single value was found.

    Attributes:
        status: ``"ok"`` when exactly one value was extracted, otherwise
            ``"unparseable"`` (no usable numeral) or ``"ambiguous"`` (several
            candidate values, conflicting markers, or hedging).
        value: Canonical string of the parsed value when ``status == "ok"``:
            integers as ``"1022"`` / ``"-3"`` (thousands commas and integral
            decimals such as ``"6.0"`` are normalised), fractions as written but
            whitespace-free (``"19/27"``, ``"38/54"``), other decimals as
            normalised decimal strings (``"6.5"``). ``None`` otherwise.
        raw_span: The exact substring of the model text that was parsed.
        note: Human-readable explanation of the decision.
        span: ``(start, end)`` character offsets of ``raw_span`` in the input.
        rule: Name of the rule that produced the value (see module docstring).
    """

    status: ParseStatus
    value: str | None
    raw_span: str | None
    note: str
    span: tuple[int, int] | None = None
    rule: str | None = None

    @property
    def ok(self) -> bool:
        """True when a single value was extracted."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        data = asdict(self)
        data["span"] = list(self.span) if self.span is not None else None
        return data


# --------------------------------------------------------------------------- #
# Lexical building blocks
# --------------------------------------------------------------------------- #

# Chat-template control tokens (Qwen: <|im_end|>, <|endoftext|>, <|im_start|>).
_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]{1,64}\|>|</?s>")

# Unicode minus signs are normalised to ASCII '-'. Every replacement is exactly
# one character, so offsets into the cleaned text are valid for the original.
_MINUS_TRANSLATION = str.maketrans({"\u2212": "-", "\u2013": "-", "\ufe63": "-", "\uff0d": "-"})

# A single numeral. The lookarounds stop it from matching a fragment of a longer
# token: "3" in "3x", "02" in "1,02", "5" in "5.5.5", "2" in "1/2/3".
# A leading '-' counts as a sign only if it is not a binary minus ("37-58").
_NUMERAL_RE = re.compile(
    r"""
    (?P<sign>(?<![\w)\]}])-)?
    (?<![\w.,/\\\u2014])
    (?:
        \\[dt]?frac\s*\{\s*(?P<fn>-?[0-9]+)\s*\}\s*\{\s*(?P<fd>[0-9]+)\s*\}
      | (?P<sn>[0-9]+)[ \t]*/[ \t]*(?P<sd>[0-9]+)
      | (?P<dec>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]+)
      | (?P<th>[0-9]{1,3}(?:,[0-9]{3})+)
      | (?P<int>[0-9]+)
    )
    (?![\w/]|[.,][0-9])
    """,
    re.VERBOSE,
)

_DECOR_CHARS = frozenset("*_`$\"'\u201c\u201d\u2018\u2019")
_TRAILING_PUNCT = frozenset(".,;:!")
_BRACE_WRAPPER_RE = re.compile(r"\\(?:boxed|fbox|text|textbf|mathbf|mathrm)\s*\{")
_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_CLOSING_DECOR_RE = re.compile(r"(?:[*_`$\"'\u201d\u2019}]|\\\)|\\\])*")

# "x =" / "x is" prefix inside an answer region ("the answer is x = 6").
_X_PREFIX_RE = re.compile(r"x\s*(?:=|is\b)\s*:?", re.IGNORECASE)

# Characters allowed in the left-hand side of an arithmetic chain "a op b = c".
_ARITH_RE = re.compile(
    r"(?:[0-9\s+\-*/().,=%\[\]{}\u00d7\u00f7\u00b7\u22c5]|\\times|\\cdot|\\div|\\left|\\right"
    r"|\\[dt]?frac|\bmod\b|\\bmod|\\pmod)+"
)

# After a candidate answer: another number offered as an alternative -> hedged.
_HEDGE_AFTER_RE = re.compile(r"\s*\(?\s*(?:,|\bor\b|\band\b|\bto\b|~|\u2014|\+/-|\u00b1)\s*-?[0-9]", re.IGNORECASE)
# After a candidate answer: an operator, i.e. the numeral is not a final value.
_OPERATOR_AFTER_RE = re.compile(
    r"\s*(?:[+\-*/^=%<>\u00d7\u00f7\u2248]|\\times|\\cdot|\\div|\\pm|x\s*[0-9(]|mod\b)",
    re.IGNORECASE,
)

_CLAUSE_END_RE = re.compile(r"\n|[.!?;](?=\s|$)")
_SENTENCE_BREAK_RE = re.compile(r"\n|[.!?;]\s+\S")

_ANSWER_MARKER_RE = re.compile(
    r"\b(?:final\s+answer|answer|result|sum|total|product|difference|remainder|quotient"
    r"|solution|value\s+of\s+x|(?:larger|largest|greater|bigger)\s+(?:fraction|one|number|value))"
    r"(?:\s*\*\*|\s*__)?\s*(?:is\b\s*:?|:|=(?!=)|equals\b)",
    re.IGNORECASE,
)
_REVERSE_ANSWER_RE = re.compile(
    r"(?:[*_`$\"'}]|\\\)|\\\])*\s*(?:is|was)\s+(?:the\s+|my\s+)?(?:final\s+|correct\s+)?answer\b",
    re.IGNORECASE,
)
_REVERSE_LARGER_RE = re.compile(
    r"(?:[*_`$\"'}]|\\\)|\\\])*\s*is\s+(?:the\s+)?(?:larger|largest|greater|bigger)\b",
    re.IGNORECASE,
)
_X_MARKER_RE = re.compile(r"(?<![\w.\\])x\s*(?:=(?!=)|is\s+equal\s+to\b|equals\b|is\b)\s*:?", re.IGNORECASE)
_COPULA_BEFORE_RE = re.compile(
    r"\b(?:is|equals|gives|makes|gets?)\s*:?\s*(?:[*_`$\"']|\\\(|\\\[|\\boxed\s*\{)*\s*$",
    re.IGNORECASE,
)
_HEDGE_WORDS_RE = re.compile(
    r"\b(?:not|no|never|maybe|perhaps|probably|possibly|approximately|approx|about|around"
    r"|roughly|nearly|almost|guess|unsure|uncertain|might|could|cannot|either|or|between)\b"
    r"|n['\u2019]t\b|\u2248|~|\?",
    re.IGNORECASE,
)
_ENUMERATOR_AFTER_RE = re.compile(r"[.)]\s+\S")

_MARKER_PRIORITY = ("boxed", "answer", "x_equals", "first_line", "equation", "copula")
_MAX_MARKERS = 32  # more occurrences of one marker type than this -> refuse (bounded work)
_MAX_NUMERAL_CHARS = 200  # longer "numerals" are refused (also avoids int() digit limits)


@dataclass(frozen=True)
class _Numeral:
    start: int
    end: int
    value: Fraction | None  # None when invalid (zero denominator)
    canonical: str | None
    error: str | None = None


@dataclass(frozen=True)
class _Finding:
    rule: str
    kind: Literal["ok", "hedged", "unusable", "invalid"]
    numeral: _Numeral | None
    note: str


# --------------------------------------------------------------------------- #
# Numerals
# --------------------------------------------------------------------------- #


def _numeral_from_match(m: re.Match[str], offset: int) -> _Numeral:
    start, end = m.start() + offset, m.end() + offset
    if end - start > _MAX_NUMERAL_CHARS:
        return _Numeral(start, end, None, None, "numeral too long")
    negative = m.group("sign") is not None
    den_s = m.group("fd") if m.group("fd") is not None else m.group("sd")
    if den_s is not None:
        num_s = m.group("fn") if m.group("fn") is not None else m.group("sn")
        num, den = int(num_s), int(den_s)
        if den == 0:
            return _Numeral(start, end, None, None, "zero denominator")
        if negative:
            num = -num
        canonical = f"{'-' if num < 0 else ''}{abs(num)}/{den}"
        return _Numeral(start, end, Fraction(num, den), canonical)
    if m.group("dec") is not None:
        dec = Decimal(m.group("dec").replace(",", ""))
        if negative:
            dec = -dec
        value = Fraction(dec)
        if value.denominator == 1:
            return _Numeral(start, end, value, str(value.numerator))
        return _Numeral(start, end, value, format(dec.normalize(), "f"))
    digits = (m.group("th") or m.group("int")).replace(",", "")
    n = -int(digits) if negative else int(digits)
    return _Numeral(start, end, Fraction(n), str(n))


def _detached_minus(text: str, start: int) -> bool:
    """True if ``text[start:]`` is preceded by ``-`` plus whitespace that is not a binary minus.

    ``"is - 12"`` or a bullet ``"- 12"`` cannot be read safely as either 12 or
    -12, whereas in ``"37 - 12"`` the minus is an operator between operands.
    """
    i = start - 1
    if i < 0 or not text[i].isspace():
        return False
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0 or text[i] != "-":
        return False
    i -= 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return True
    operand_end = text[i].isdigit() or text[i] in ")]}" or (text[i] in "xX" and (i == 0 or not text[i - 1].isalpha()))
    return not operand_end


def _make_numeral(text: str, m: re.Match[str], offset: int) -> _Numeral:
    num = _numeral_from_match(m, offset)
    if num.error is None and m.group("sign") is None and _detached_minus(text, num.start):
        return _Numeral(num.start, num.end, None, None, "minus sign separated from the number by whitespace")
    return num


def _full_numeral(text: str, s: int, e: int) -> _Numeral | None:
    m = _NUMERAL_RE.fullmatch(text[s:e])
    return _make_numeral(text, m, s) if m else None


def _all_numerals(text: str, s: int, e: int) -> list[_Numeral]:
    return [_make_numeral(text, m, s) for m in _NUMERAL_RE.finditer(text[s:e])]


# --------------------------------------------------------------------------- #
# Span helpers
# --------------------------------------------------------------------------- #


def _matching_close(text: str, open_idx: int, limit: int, open_ch: str = "{", close_ch: str = "}") -> int:
    """Index of the bracket closing ``text[open_idx]`` (before ``limit``), or -1."""
    depth = 0
    for i in range(open_idx, limit):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


_MAX_WRAPPER_PEELS = 8  # bounds the (linear-scan) bracket matching per _strip call


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    """Shrink ``[s, e)`` past whitespace, trailing punctuation and decoration.

    Balanced ``\\boxed{...}`` / ``\\text{...}`` / ``(...)`` wrappers spanning the
    whole range are peeled, at most ``_MAX_WRAPPER_PEELS`` layers deep.
    """
    peels = 0
    while True:
        s0, e0 = s, e
        while s < e and (text[s].isspace() or text[s] in _DECOR_CHARS):
            s += 1
        while e > s and (text[e - 1].isspace() or text[e - 1] in _DECOR_CHARS or text[e - 1] in _TRAILING_PUNCT):
            e -= 1
        if text.startswith(("\\(", "\\["), s, e):
            s += 2
        if e - s >= 2 and text.endswith(("\\)", "\\]"), s, e):
            e -= 2
        if peels < _MAX_WRAPPER_PEELS:
            wrapper = _BRACE_WRAPPER_RE.match(text, s, e)
            if wrapper and _matching_close(text, wrapper.end() - 1, e) == e - 1:
                s, e = wrapper.end(), e - 1
                peels += 1
            elif s < e and text[s] == "(" and _matching_close(text, s, e, "(", ")") == e - 1:
                s, e = s + 1, e - 1
                peels += 1
        if (s, e) == (s0, e0):
            return s, e


def _is_blank(text: str, s: int, e: int) -> bool:
    s, e = _strip(text, s, e)
    return s >= e


# --------------------------------------------------------------------------- #
# Answer regions
# --------------------------------------------------------------------------- #


def _numeral_finding(rule: str, num: _Numeral, note: str) -> _Finding:
    if num.error is not None:
        return _Finding(rule, "invalid", num, f"{note}: {num.error}")
    return _Finding(rule, "ok", num, note)


def _leading_value(text: str, s: int, e: int, rule: str) -> _Finding:
    """A numeral at the start of ``[s, e)`` that is a final value, not an operand."""
    s, e = _strip(text, s, e)
    sub = text[s:e]
    m = _NUMERAL_RE.match(sub)
    if m is None:
        return _Finding(rule, "unusable", None, "no numeral directly after the marker")
    num = _make_numeral(text, m, s)
    rest = sub[m.end() :]
    closing = _CLOSING_DECOR_RE.match(rest)
    if closing is not None:
        rest = rest[closing.end() :]
    if _HEDGE_AFTER_RE.match(rest):
        return _Finding(rule, "hedged", num, f"alternative value offered after {sub[: m.end()]!r}")
    if _OPERATOR_AFTER_RE.match(rest):
        return _Finding(rule, "unusable", num, "numeral is followed by an operator (not a final value)")
    return _numeral_finding(rule, num, "numeral directly after the marker")


def _region_value(text: str, s: int, e: int, rule: str) -> _Finding:
    """Interpret ``[s, e)`` as an answer region: a numeral, ``x = N`` or ``... = N``."""
    s, e = _strip(text, s, e)
    prefix = _X_PREFIX_RE.match(text[s:e])
    if prefix:
        s, e = _strip(text, s + prefix.end(), e)
    if s >= e:
        return _Finding(rule, "unusable", None, "empty answer region")
    num = _full_numeral(text, s, e)
    if num is not None:
        return _numeral_finding(rule, num, "region is exactly one numeral")
    eq = text.rfind("=", s, e)
    if eq != -1:
        hs, he = _strip(text, s, eq)
        head = text[hs:he]
        if head and _ARITH_RE.fullmatch(head) and any(ch.isdigit() for ch in head):
            tail = _leading_value(text, eq + 1, e, rule)
            if tail.kind == "ok":
                return _Finding(rule, "ok", tail.numeral, "final value of an arithmetic chain")
            return tail
    return _leading_value(text, s, e, rule)


def _clause_end(text: str, start: int, limit: int) -> int:
    m = _CLAUSE_END_RE.search(text, start, limit)
    return m.start() if m else limit


# --------------------------------------------------------------------------- #
# Marker rules
# --------------------------------------------------------------------------- #


def _find_markers(text: str, s: int, e: int, answer_type: str) -> list[_Finding]:
    findings: list[_Finding] = []

    def occurrences(pattern: re.Pattern[str], rule: str) -> list[re.Match[str]]:
        found = list(islice(pattern.finditer(text, s, e), _MAX_MARKERS + 1))
        if len(found) > _MAX_MARKERS:
            findings.append(_Finding(rule, "invalid", None, f"more than {_MAX_MARKERS} {rule} markers"))
            return []
        return found

    for m in occurrences(_BOXED_RE, "boxed"):
        close = _matching_close(text, m.end() - 1, e)
        if close == -1:
            findings.append(_Finding("boxed", "invalid", None, "unterminated \\boxed{ (truncated output?)"))
        else:
            findings.append(_region_value(text, m.end(), close, "boxed"))

    for m in occurrences(_ANSWER_MARKER_RE, "answer"):
        findings.append(_region_value(text, m.end(), _clause_end(text, m.end(), e), "answer"))
    reverse = [_REVERSE_ANSWER_RE] + ([_REVERSE_LARGER_RE] if answer_type == "rational" else [])
    for num in _all_numerals(text, s, e):
        if any(r.match(text, num.end, e) for r in reverse):
            findings.append(_numeral_finding("answer", num, "numeral declared as the answer"))

    for m in occurrences(_X_MARKER_RE, "x_equals"):
        findings.append(_region_value(text, m.end(), _clause_end(text, m.end(), e), "x_equals"))

    newline = text.find("\n", s, e)
    if newline != -1:
        first = _full_numeral(text, *_strip(text, s, newline))
        if first is not None:
            findings.append(_numeral_finding("first_line", first, "first line is a bare numeral"))

    if "=" in text[s:e] and _ARITH_RE.fullmatch(text[s:e]):
        chain = _region_value(text, s, e, "equation")
        if chain.kind in ("ok", "invalid", "hedged"):
            findings.append(chain)

    if not _SENTENCE_BREAK_RE.search(text, s, e):
        nums = _all_numerals(text, s, e)
        if nums and _is_blank(text, nums[-1].end, e) and _COPULA_BEFORE_RE.search(text[s : nums[-1].start]):
            if _HEDGE_WORDS_RE.search(text, s, e):
                findings.append(_Finding("copula", "hedged", nums[-1], "sentence contains hedging/negation"))
            else:
                findings.append(_numeral_finding("copula", nums[-1], "single sentence ending in 'is N'"))

    return [f for f in findings if f.kind != "unusable"]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _result(
    original: str, status: ParseStatus, note: str, num: _Numeral | None = None, rule: str | None = None
) -> ParsedAnswer:
    if num is None or status != "ok":
        return ParsedAnswer(status=status, value=None, raw_span=None, note=note, span=None, rule=rule)
    return ParsedAnswer(
        status="ok",
        value=num.canonical,
        raw_span=original[num.start : num.end],
        note=note,
        span=(num.start, num.end),
        rule=rule,
    )


def _fmt_values(findings: list[_Finding]) -> str:
    return ", ".join(f"{f.rule}={f.numeral.canonical}" for f in findings if f.numeral is not None)


def parse_answer(text: str | None, answer_type: str = "integer") -> ParsedAnswer:
    """Extract the single numeric answer a model output commits to.

    Args:
        text: Raw generated text (may include chat special tokens).
        answer_type: ``"integer"`` or ``"rational"``. Rational mode additionally
            accepts "N is larger" as an answer declaration (fraction comparison).
            Both modes parse integers, decimals and fractions; deciding whether
            the value is *right* is the oracle's job, not the parser's.

    Returns:
        A :class:`ParsedAnswer`. ``status == "ok"`` only when exactly one value
        is committed to; see the module docstring for the full procedure.
    """
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type must be one of {ANSWER_TYPES}, got {answer_type!r}")
    if text is None:
        return _result("", "unparseable", "no text")
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if len(text) > MAX_PARSE_CHARS:
        return _result(text, "unparseable", f"text longer than {MAX_PARSE_CHARS} characters")

    cleaned = _SPECIAL_TOKEN_RE.sub(lambda m: " " * len(m.group(0)), text).translate(_MINUS_TRANSLATION)
    s, e = _strip(cleaned, 0, len(cleaned))
    if s >= e:
        return _result(text, "unparseable", "empty output")

    # Rule 0: the whole output is one numeral.
    whole = _full_numeral(cleaned, s, e)
    if whole is not None:
        if whole.error is not None:
            return _result(text, "unparseable", f"whole text is a malformed numeral: {whole.error}")
        return _result(text, "ok", "whole output is a single numeral", whole, "whole_text")

    # Rule 1: explicit markers.
    findings = _find_markers(cleaned, s, e, answer_type)
    hedged = [f for f in findings if f.kind == "hedged"]
    if hedged:
        return _result(text, "ambiguous", f"hedged answer ({hedged[0].rule}): {hedged[0].note}")
    usable = [f for f in findings if f.kind == "ok"]
    invalid = [f for f in findings if f.kind == "invalid"]
    if usable:
        if invalid:
            return _result(text, "ambiguous", f"markers include a malformed value: {invalid[0].note}")
        if len({f.numeral.value for f in usable if f.numeral is not None}) > 1:
            return _result(text, "ambiguous", f"answer markers disagree: {_fmt_values(usable)}")
        top_rule = min((f.rule for f in usable), key=_MARKER_PRIORITY.index)
        chosen = [f for f in usable if f.rule == top_rule][-1]
        rules = sorted({f.rule for f in usable}, key=_MARKER_PRIORITY.index)
        note = f"{chosen.note} ({top_rule} marker"
        note += f"; agrees with {', '.join(r for r in rules if r != top_rule)})" if len(rules) > 1 else ")"
        return _result(text, "ok", note, chosen.numeral, top_rule)
    if invalid:
        return _result(text, "unparseable", invalid[0].note)

    # Rule 2: lone-number fallback.
    nums = _all_numerals(cleaned, s, e)
    if not nums:
        return _result(text, "unparseable", "no numeral found")
    malformed = [n for n in nums if n.error is not None]
    if malformed:
        bad = malformed[0]
        return _result(text, "ambiguous", f"cannot read numeral {text[bad.start : bad.end]!r}: {bad.error}")
    distinct = {n.value for n in nums}
    if len(distinct) > 1:
        shown = ", ".join(n.canonical or text[n.start : n.end] for n in nums[:6])
        return _result(text, "ambiguous", f"several different numerals and no answer marker: {shown}")
    if _HEDGE_WORDS_RE.search(cleaned, s, e):
        return _result(text, "ambiguous", "lone numeral but the text contains hedging/negation")
    for num in (nums[0], nums[-1]):
        at_start = _is_blank(cleaned, s, num.start)
        at_end = _is_blank(cleaned, num.end, e)
        if at_start and not at_end and _ENUMERATOR_AFTER_RE.match(cleaned, num.end, e):
            return _result(text, "unparseable", "leading numeral looks like a list enumerator")
        if at_start or at_end:
            where = "end" if at_end else "start"
            return _result(text, "ok", f"lone numeral at the {where} of the text (no marker)", num, "lone_number")
    return _result(text, "unparseable", "lone numeral embedded in prose, no answer marker")
