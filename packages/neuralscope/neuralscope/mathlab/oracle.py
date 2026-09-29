"""Deterministic SymPy oracle: computes true answers and labels model outputs.

Epistemic role
--------------
The oracle's CORRECT / INCORRECT label is an *observed ground-truth fact about
the model's output*: given the problem, did the text the model emitted commit
to the mathematically true value? It is computed symbolically (exact integer /
rational arithmetic, ``sympy.solve``, ``sympy.Mod``) and never by another model.

A correctness label says nothing about the model's internal mechanism. A
CORRECT answer may come from memorisation, a shortcut or a lucky guess; an
INCORRECT one may come from a nearly-right algorithm with one carry error.
Claims about *why* an answer is right or wrong require interventions on the
network, not this label.

Labels are conservative. An output that does not commit to exactly one value
(see :mod:`.parser`) is labelled ``"unparseable"`` or ``"ambiguous"``, never
correct, and should be excluded from correct-vs-incorrect comparisons.

Free-form expressions (:func:`verify_expression`) are validated against a strict
character whitelist *before* reaching ``sympy.parse_expr``; exponentiation is
rejected outright and no names other than ``x`` are resolvable. ``eval`` is
never called on user or model text by this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import sympy
from sympy.parsing.sympy_parser import auto_number, parse_expr, rationalize

from ._types import OPERAND_KEYS, PROBLEM_KINDS
from .parser import ParsedAnswer, parse_answer

if TYPE_CHECKING:
    from .problems import MathProblem

__all__ = [
    "Verification",
    "VerificationStatus",
    "compute_oracle_answer",
    "verify",
    "verify_expression",
    "MAX_EXPRESSION_LENGTH",
]

VerificationStatus = Literal["correct", "incorrect", "unparseable", "ambiguous"]

MAX_EXPRESSION_LENGTH = 200
"""Free-form expressions longer than this are rejected before any parsing."""

_X = sympy.Symbol("x")


@dataclass(frozen=True)
class Verification:
    """The oracle's verdict on one answer.

    Attributes:
        correct: True only when the answer parsed to a single value that equals
            the oracle value exactly (SymPy rational equality).
        status: ``"correct"``, ``"incorrect"``, or -- when no single value could
            be extracted -- the parser's ``"unparseable"`` / ``"ambiguous"``.
        parsed_answer: Canonical parsed value (``ParsedAnswer.value``), if any.
        oracle_answer: Canonical oracle answer string.
        note: Human-readable explanation.
    """

    correct: bool
    status: VerificationStatus
    parsed_answer: str | None
    oracle_answer: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Oracle answers
# --------------------------------------------------------------------------- #


def _int_operands(kind: str, operands: Mapping[str, Any]) -> dict[str, int]:
    if kind not in PROBLEM_KINDS:
        raise ValueError(f"unknown problem kind {kind!r}; expected one of {PROBLEM_KINDS}")
    if not isinstance(operands, Mapping):
        raise TypeError(f"operands must be a mapping, got {type(operands).__name__}")
    expected = OPERAND_KEYS[kind]
    if set(operands) != set(expected):
        raise ValueError(f"{kind} requires operands {expected}, got {tuple(operands)}")
    out: dict[str, int] = {}
    for key in expected:
        value = operands[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"operand {key!r} must be an int, got {value!r}")
        out[key] = int(value)
    return out


def compute_oracle_answer(kind: str, operands: Mapping[str, Any]) -> str:
    """Compute the canonical answer string for a problem with SymPy.

    * ``add`` / ``sub`` / ``mul``: exact ``sympy.Integer`` arithmetic.
    * ``linear_equation``: ``sympy.solve(Eq(a*x + b, c), x)``; exactly one
      solution is required (``a != 0``). Non-integer solutions are returned as
      ``"p/q"`` (problem construction restricts itself to integer solutions).
    * ``compare_fractions``: compares ``Rational(p, q)`` with ``Rational(r, s)``
      and returns the larger fraction *exactly as written* (``"p/q"``).
      Denominators must be positive and the values must differ.
    * ``mod``: ``sympy.Mod(a, b)`` with ``a >= 0`` and ``b > 0`` (negative
      operands are rejected because "mod" conventions differ between languages).

    Raises:
        ValueError / TypeError: for unknown kinds or invalid operands.
    """
    ops = _int_operands(kind, operands)
    if kind == "add":
        return str(sympy.Integer(ops["a"]) + sympy.Integer(ops["b"]))
    if kind == "sub":
        return str(sympy.Integer(ops["a"]) - sympy.Integer(ops["b"]))
    if kind == "mul":
        return str(sympy.Integer(ops["a"]) * sympy.Integer(ops["b"]))
    if kind == "linear_equation":
        a, b, c = (sympy.Integer(ops[k]) for k in ("a", "b", "c"))
        if a == 0:
            raise ValueError("linear_equation requires a != 0")
        solutions = sympy.solve(sympy.Eq(a * _X + b, c), _X)
        if len(solutions) != 1 or not solutions[0].is_Rational:
            raise ValueError(f"expected one rational solution, got {solutions}")
        return str(solutions[0])
    if kind == "compare_fractions":
        p, q, r, s = (ops[k] for k in ("p", "q", "r", "s"))
        if q <= 0 or s <= 0:
            raise ValueError("compare_fractions requires positive denominators")
        left, right = sympy.Rational(p, q), sympy.Rational(r, s)
        if left == right:
            raise ValueError(f"{p}/{q} and {r}/{s} are equal; the comparison has no answer")
        return f"{p}/{q}" if left > right else f"{r}/{s}"
    # kind == "mod"
    a, b = ops["a"], ops["b"]
    if b <= 0 or a < 0:
        raise ValueError("mod requires a >= 0 and b > 0")
    return str(sympy.Mod(sympy.Integer(a), sympy.Integer(b)))


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #


def _to_rational(value: str) -> sympy.Rational:
    """Exact rational from a canonical numeric string ("95", "-3/4", "6.5")."""
    frac = Fraction(value)
    return sympy.Rational(frac.numerator, frac.denominator)


def _judge(parsed: ParsedAnswer, oracle_answer: str, rejected_option: str | None = None) -> Verification:
    """Compare a parsed answer with the oracle value.

    ``rejected_option`` is the wrong choice of a two-way comparison (the smaller
    fraction); it only refines the note of an incorrect verdict.
    """
    if parsed.status != "ok" or parsed.value is None:
        return Verification(
            correct=False,
            status="ambiguous" if parsed.status == "ambiguous" else "unparseable",
            parsed_answer=None,
            oracle_answer=oracle_answer,
            note=f"not scored: {parsed.note}",
        )
    model_value = _to_rational(parsed.value)
    oracle_value = _to_rational(oracle_answer)
    if model_value == oracle_value:
        same = parsed.value == oracle_answer
        note = "exact match" if same else f"equal in value to {oracle_answer}"
        return Verification(True, "correct", parsed.value, oracle_answer, f"{note} [{parsed.rule}]")
    note = f"{parsed.value} != {oracle_answer}"
    if rejected_option is not None:
        if model_value == _to_rational(rejected_option):
            note += f"; chose the smaller fraction {rejected_option}"
        else:
            note += "; matches neither fraction"
    return Verification(False, "incorrect", parsed.value, oracle_answer, f"{note} [{parsed.rule}]")


def _smaller_fraction(operands: Mapping[str, int], larger: str) -> str:
    left = f"{operands['p']}/{operands['q']}"
    right = f"{operands['r']}/{operands['s']}"
    return right if larger == left else left


def verify(problem: MathProblem, model_output: str | None) -> Verification:
    """Label a model output for ``problem`` as correct / incorrect / not scorable.

    The stored ``problem.oracle_answer`` is first recomputed from the operands;
    a mismatch (stale or tampered data) raises ``ValueError`` rather than
    producing a label against a wrong reference.

    Equality is exact rational equality, so for integer answers ``"6"``,
    ``"6.0"`` and ``"12/2"`` are all correct; for fraction comparisons any
    fraction equal in *value* to the larger one (``"19/27"``, ``"38/54"``) is
    correct and the smaller fraction is incorrect.
    """
    recomputed = compute_oracle_answer(problem.kind, problem.operands)
    if recomputed != problem.oracle_answer:
        raise ValueError(
            f"problem {problem.id!r}: stored oracle_answer {problem.oracle_answer!r} != recomputed {recomputed!r}"
        )
    parsed = parse_answer(model_output, problem.answer_type)
    rejected = (
        _smaller_fraction(problem.operands, problem.oracle_answer) if problem.kind == "compare_fractions" else None
    )
    return _judge(parsed, problem.oracle_answer, rejected)


# --------------------------------------------------------------------------- #
# Free-form expressions
# --------------------------------------------------------------------------- #

_EXPRESSION_WHITELIST_RE = re.compile(r"(?:[0-9\s+\-*/().=x,]|mod|compare)+")
_COMPARE_RE = re.compile(r"\s*compare\s*\(\s*(-?[0-9]+)\s*/\s*([0-9]+)\s*,\s*(-?[0-9]+)\s*/\s*([0-9]+)\s*\)\s*")
_ARITH_CHARS_RE = re.compile(r"[0-9\s+\-*/().]+")
_ARITH_X_CHARS_RE = re.compile(r"[0-9\s+\-*/().x]+")
_IMPLICIT_MUL_RE = re.compile(r"(?<=[0-9)])\s*(?=[x(])")


def _safe_expr(source: str, allow_x: bool) -> sympy.Expr:
    """Parse a whitelisted arithmetic string (optionally in ``x``) with SymPy.

    The string must consist only of digits, whitespace, ``+ - * / ( ) .`` and
    (if ``allow_x``) the letter ``x``. ``**`` and ``//`` are rejected. The only
    names visible to SymPy's parser are ``Integer``, ``Rational``, ``Float`` and
    ``x``; builtins are disabled.
    """
    src = source.strip()
    pattern = _ARITH_X_CHARS_RE if allow_x else _ARITH_CHARS_RE
    if not src or not pattern.fullmatch(src):
        raise ValueError(f"invalid arithmetic expression {source!r}")
    if "**" in src or "//" in src:
        raise ValueError("exponentiation and floor division are not allowed")
    src = _IMPLICIT_MUL_RE.sub("*", src)  # "3x" -> "3*x", "2(x+1)" -> "2*(x+1)"
    global_dict: dict[str, Any] = {
        "__builtins__": {},
        "Integer": sympy.Integer,
        "Rational": sympy.Rational,
        "Float": sympy.Float,
    }
    local_dict: dict[str, Any] = {"x": _X} if allow_x else {}
    try:
        expr = parse_expr(
            src,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=(auto_number, rationalize),
            evaluate=True,
        )
    except Exception as exc:  # input is whitelisted; any failure is a syntax problem
        raise ValueError(f"could not parse {source!r}: {exc}") from exc
    if not isinstance(expr, sympy.Expr):
        raise ValueError(f"{source!r} is not an arithmetic expression")
    if expr.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
        raise ValueError(f"{source!r} is undefined (division by zero?)")
    if not expr.free_symbols <= {_X}:
        raise ValueError(f"unexpected symbols in {source!r}")
    return expr


def _safe_integer(source: str) -> int:
    value = _safe_expr(source, allow_x=False)
    if not value.is_Integer:
        raise ValueError(f"{source!r} is not an integer")
    return int(value)


def _evaluate_expression(expression: str) -> tuple[str, str, str | None]:
    """Return ``(answer_type, oracle_answer, rejected_option)`` for an expression."""
    if not isinstance(expression, str):
        raise TypeError(f"expression must be a str, got {type(expression).__name__}")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    expr = expression.strip()
    if not expr:
        raise ValueError("empty expression")
    if not _EXPRESSION_WHITELIST_RE.fullmatch(expr):
        raise ValueError(f"expression contains disallowed characters or words: {expression!r}")
    if "**" in expr or "//" in expr:
        raise ValueError("exponentiation and floor division are not allowed")

    if "compare" in expr:
        m = _COMPARE_RE.fullmatch(expr)
        if m is None:
            raise ValueError("expected 'compare(p/q, r/s)'")
        p, q, r, s = (int(g) for g in m.groups())
        operands = {"p": p, "q": q, "r": r, "s": s}
        larger = compute_oracle_answer("compare_fractions", operands)
        return "rational", larger, _smaller_fraction(operands, larger)
    if "," in expr:
        raise ValueError("',' is only allowed inside compare(...)")
    if "mod" in expr:
        parts = expr.split("mod")
        if len(parts) != 2 or "=" in expr or "x" in expr:
            raise ValueError("expected '<integer> mod <integer>'")
        a, b = _safe_integer(parts[0]), _safe_integer(parts[1])
        return "integer", compute_oracle_answer("mod", {"a": a, "b": b}), None
    if "=" in expr:
        sides = expr.split("=")
        if len(sides) != 2 or "x" not in expr:
            raise ValueError("an equation needs exactly one '=' and the unknown x")
        lhs, rhs = _safe_expr(sides[0], allow_x=True), _safe_expr(sides[1], allow_x=True)
        try:
            degree = sympy.Poly(lhs - rhs, _X).degree()
        except sympy.PolynomialError as exc:
            raise ValueError("equation must be linear in x") from exc
        if degree != 1:
            raise ValueError("equation must be linear in x with a unique solution")
        solutions = sympy.solve(sympy.Eq(lhs, rhs), _X)
        if len(solutions) != 1 or not solutions[0].is_Rational:
            raise ValueError(f"expected one rational solution, got {solutions}")
        solution = solutions[0]
        return ("integer" if solution.is_Integer else "rational"), str(solution), None
    if "x" in expr:
        raise ValueError("the unknown x is only allowed in an equation")
    value = _safe_expr(expr, allow_x=False)
    if not value.is_Rational:
        raise ValueError(f"{expression!r} does not evaluate to a rational number")
    return ("integer" if value.is_Integer else "rational"), str(value), None


def verify_expression(expression: str, answer: str | None) -> Verification:
    """Verify an answer against a free-form expression.

    Supported forms (the whitelist admits digits, whitespace, ``+ - * / ( ) .
    = x ,`` and the words ``mod`` / ``compare``; nothing else):

    * arithmetic: ``"37 + 58"``, ``"73*14"``, ``"(12-5)*3"``, ``"7/2"``
    * linear equation in ``x``: ``"3x + 7 = 25"`` (solved for ``x``)
    * ``"47 mod 6"`` (non-negative dividend, positive divisor)
    * ``"compare(17/23, 19/27)"`` (answer = the larger fraction)

    Raises:
        ValueError: for anything outside the whitelist or grammar, including
            exponentiation (``**``), inputs over ``MAX_EXPRESSION_LENGTH``
            characters, division by zero, and non-linear equations.
    """
    answer_type, oracle_answer, rejected = _evaluate_expression(expression)
    return _judge(parse_answer(answer, answer_type), oracle_answer, rejected)
