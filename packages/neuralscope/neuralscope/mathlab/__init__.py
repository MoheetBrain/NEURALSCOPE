"""NeuralScope Math Lab: verifiable math problems and a deterministic oracle.

Math Lab pairs mathematically verifiable problems with a neural model's
generated answers. A SymPy oracle labels every answer CORRECT or INCORRECT (or
declines to label it when the output does not commit to a single value).

Epistemic role: a correctness label is an *observed ground-truth fact about the
model's output*. It is the reference against which correct and incorrect neural
traces are compared, and it says nothing on its own about the model's internal
mechanism. For that reason the answer parser is conservative: sloppy parsing
must never turn an uncommitted output into a CORRECT label.

This package is pure Python (SymPy only) and never loads a neural model.
"""

from __future__ import annotations

from .alignment import first_divergence
from .oracle import Verification, compute_oracle_answer, verify, verify_expression
from .parser import ParsedAnswer, parse_answer
from .problems import (
    DEFAULT_BENCHMARK_SPEC,
    DEMO_PROBLEM,
    PROBLEM_KINDS,
    MathProblem,
    ProblemKind,
    default_benchmark,
    generate_problems,
    make_problem,
)

__all__ = [
    "MathProblem",
    "ProblemKind",
    "PROBLEM_KINDS",
    "make_problem",
    "generate_problems",
    "default_benchmark",
    "DEFAULT_BENCHMARK_SPEC",
    "DEMO_PROBLEM",
    "parse_answer",
    "ParsedAnswer",
    "verify",
    "verify_expression",
    "compute_oracle_answer",
    "Verification",
    "first_divergence",
]
