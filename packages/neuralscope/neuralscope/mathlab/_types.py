"""Shared type aliases and constants for the Math Lab.

Kept in a private leaf module so that :mod:`.problems` and :mod:`.oracle` can
both depend on them without importing each other at module load time.
"""

from __future__ import annotations

from typing import Literal, get_args

ProblemKind = Literal["add", "sub", "mul", "linear_equation", "compare_fractions", "mod"]
"""The closed set of problem families the Math Lab can generate and verify."""

AnswerType = Literal["integer", "rational"]
"""Type of the oracle answer: an integer, or an exact rational (fraction)."""

PROBLEM_KINDS: tuple[str, ...] = get_args(ProblemKind)
ANSWER_TYPES: tuple[str, ...] = get_args(AnswerType)

OPERAND_KEYS: dict[str, tuple[str, ...]] = {
    "add": ("a", "b"),
    "sub": ("a", "b"),
    "mul": ("a", "b"),
    "linear_equation": ("a", "b", "c"),  # a*x + b = c
    "compare_fractions": ("p", "q", "r", "s"),  # p/q  vs  r/s
    "mod": ("a", "b"),  # a mod b
}
"""Exact operand keys required for each problem kind (order is canonical)."""
