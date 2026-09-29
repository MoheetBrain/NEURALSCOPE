"""Math Lab problems: verifiable arithmetic stimuli and deterministic generation.

Epistemic role
--------------
A :class:`MathProblem` is a *stimulus*: a short question whose true answer is
computed symbolically from its operands by :mod:`.oracle` (SymPy), with no
dependence on any neural model. Pairing a problem with a model's generated
answer yields an observed CORRECT/INCORRECT fact about that output; that label
is what lets NeuralScope contrast the internal traces of right and wrong
answers. It is not, by itself, evidence about *how* the model computed anything.

Prompts are short, ASCII-only (``*`` rather than ``x`` for multiplication) and
end by asking for only the answer, so that a chat-formatted instruct model
(Qwen2.5-1.5B-Instruct, greedy decoding) tends to emit a bare number. Qwen's
tokenizer splits every digit into its own token, so the digit structure of an
operand or answer maps directly onto token positions.

Generation is fully deterministic: it uses a private ``random.Random(seed)`` and
never touches the global random state.
"""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Callable

from ._types import ANSWER_TYPES, OPERAND_KEYS, PROBLEM_KINDS, AnswerType, ProblemKind
from .oracle import compute_oracle_answer

__all__ = [
    "MathProblem",
    "ProblemKind",
    "PROBLEM_KINDS",
    "make_problem",
    "generate_problems",
    "default_benchmark",
    "DEFAULT_BENCHMARK_SPEC",
    "DEMO_PROBLEM",
]

_KIND_PREFIX: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "linear_equation": "lin",
    "compare_fractions": "cmp",
    "mod": "mod",
}

MAX_DIGITS = 12
"""Largest operand digit count accepted in a difficulty string."""

_MAX_SAMPLING_ATTEMPTS = 10_000
_MAX_DEDUP_ATTEMPTS = 200


@dataclass(frozen=True)
class MathProblem:
    """One verifiable problem.

    Attributes:
        id: Identifier, unique within a generated set
            (e.g. ``"add-3d3d-00007-s42"``).
        kind: Problem family (see :data:`ProblemKind`).
        question: User-facing prompt text sent to the model.
        expression: Canonical machine form, e.g. ``"37 + 58"``,
            ``"3*x + 7 = 25"``, ``"compare(17/23, 19/27)"``, ``"47 mod 6"``.
            Always accepted by :func:`.oracle.verify_expression`.
        operands: Integer operands, keys per kind: ``a, b`` (add/sub/mul/mod),
            ``a, b, c`` for ``a*x + b = c``, ``p, q, r, s`` for ``p/q`` vs ``r/s``.
        oracle_answer: Canonical answer string (``"95"``, ``"-12"``, ``"19/27"``).
        answer_type: ``"integer"`` or ``"rational"``.
        difficulty: Digit-shape label, e.g. ``"2d+2d"``, ``"5d+5d"``, ``"2dx2d"``.
    """

    id: str
    kind: ProblemKind
    question: str
    expression: str
    operands: dict[str, int] = field(hash=False)
    oracle_answer: str
    answer_type: AnswerType
    difficulty: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (operands are copied)."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> MathProblem:
        """Rebuild a problem from :meth:`to_dict` output.

        Unknown keys are ignored. The stored oracle answer is taken as given;
        :func:`.oracle.verify` recomputes and cross-checks it before labelling.
        """
        missing = [f for f in _PROBLEM_FIELDS if f not in d]
        if missing:
            raise ValueError(f"MathProblem.from_dict: missing keys {missing}")
        if d["kind"] not in PROBLEM_KINDS:
            raise ValueError(f"unknown problem kind {d['kind']!r}")
        if d["answer_type"] not in ANSWER_TYPES:
            raise ValueError(f"unknown answer_type {d['answer_type']!r}")
        if not isinstance(d["operands"], Mapping):
            raise TypeError("operands must be a mapping")
        return MathProblem(
            id=str(d["id"]),
            kind=d["kind"],
            question=str(d["question"]),
            expression=str(d["expression"]),
            operands=dict(d["operands"]),
            oracle_answer=str(d["oracle_answer"]),
            answer_type=d["answer_type"],
            difficulty=str(d["difficulty"]),
        )


_PROBLEM_FIELDS = ("id", "kind", "question", "expression", "operands", "oracle_answer", "answer_type", "difficulty")


# --------------------------------------------------------------------------- #
# Building a single problem
# --------------------------------------------------------------------------- #


def _ndigits(value: int) -> int:
    return len(str(abs(value)))


def _default_id(kind: str, operands: Mapping[str, int]) -> str:
    parts = [f"n{-v}" if v < 0 else str(v) for v in (operands[k] for k in OPERAND_KEYS[kind])]
    return "-".join([_KIND_PREFIX[kind], *parts])


def make_problem(kind: ProblemKind, operands: Mapping[str, int], id: str | None = None) -> MathProblem:
    """Build a problem (question, expression, oracle answer) from operands.

    Constraints beyond what the oracle needs keep prompts well formed:
    ``add``/``sub``/``mul`` operands are non-negative; ``linear_equation`` needs
    ``a >= 2`` and an integer solution; ``compare_fractions`` needs positive
    denominators, non-negative numerators and unequal values; ``mod`` needs
    ``a >= 0`` and ``b >= 1``.

    Args:
        kind: Problem family.
        operands: Integer operands (see :class:`MathProblem`).
        id: Identifier; defaults to a content-derived id such as ``"add-37-58"``.

    Raises:
        ValueError / TypeError: on unknown kinds or invalid operands.
    """
    # The oracle validates kind, operand keys and types (and raises on bad input).
    oracle_answer = compute_oracle_answer(kind, operands)
    ops = {k: int(operands[k]) for k in OPERAND_KEYS[kind]}
    answer_type: AnswerType = "integer"

    if kind in ("add", "sub", "mul"):
        a, b = ops["a"], ops["b"]
        if a < 0 or b < 0:
            raise ValueError(f"{kind} operands must be non-negative")
        symbol = {"add": "+", "sub": "-", "mul": "*"}[kind]
        question = f"What is {a} {symbol} {b}? Give only the answer."
        expression = f"{a} {symbol} {b}"
        difficulty = f"{_ndigits(a)}d{'x' if kind == 'mul' else symbol}{_ndigits(b)}d"
    elif kind == "linear_equation":
        a, b, c = ops["a"], ops["b"], ops["c"]
        if a < 2:
            raise ValueError("linear_equation requires a >= 2")
        if (c - b) % a != 0:
            raise ValueError(f"{a}x + {b} = {c} has no integer solution")
        if b > 0:
            lhs_q, lhs_e = f"{a}x + {b}", f"{a}*x + {b}"
        elif b < 0:
            lhs_q, lhs_e = f"{a}x - {-b}", f"{a}*x - {-b}"
        else:
            lhs_q, lhs_e = f"{a}x", f"{a}*x"
        question = f"Solve for x: {lhs_q} = {c}. Reply with only the integer value of x, no explanation."
        expression = f"{lhs_e} = {c}"
        difficulty = f"a{_ndigits(a)}d,x{_ndigits(int(oracle_answer))}d"
    elif kind == "compare_fractions":
        p, q, r, s = ops["p"], ops["q"], ops["r"], ops["s"]
        if p < 0 or r < 0:
            raise ValueError("compare_fractions numerators must be non-negative")
        question = f"Which is larger: {p}/{q} or {r}/{s}? Reply with only the larger fraction, no explanation."
        expression = f"compare({p}/{q}, {r}/{s})"
        difficulty = f"{max(_ndigits(p), _ndigits(r))}d/{max(_ndigits(q), _ndigits(s))}d"
        answer_type = "rational"
    else:  # mod
        a, b = ops["a"], ops["b"]
        question = f"What is {a} mod {b}? Give only the answer."
        expression = f"{a} mod {b}"
        difficulty = f"{_ndigits(a)}dmod{_ndigits(b)}d"

    return MathProblem(
        id=id if id is not None else _default_id(kind, ops),
        kind=kind,
        question=question,
        expression=expression,
        operands=ops,
        oracle_answer=oracle_answer,
        answer_type=answer_type,
        difficulty=difficulty,
    )


# --------------------------------------------------------------------------- #
# Random operand sampling
# --------------------------------------------------------------------------- #

_DIFFICULTY_RE: dict[str, re.Pattern[str]] = {
    "add": re.compile(r"([0-9]+)d\+([0-9]+)d"),
    "sub": re.compile(r"([0-9]+)d-([0-9]+)d"),
    "mul": re.compile(r"([0-9]+)dx([0-9]+)d"),
    "linear_equation": re.compile(r"a([0-9]+)d,x([0-9]+)d"),
    "compare_fractions": re.compile(r"([0-9]+)d/([0-9]+)d"),
    "mod": re.compile(r"([0-9]+)dmod([0-9]+)d"),
}
"""Difficulty grammar per kind. ``Nd`` = exactly N digits.

* add ``"3d+3d"``, sub ``"3d-3d"``, mul ``"3dx2d"``, mod ``"3dmod1d"``
* linear_equation ``"a1d,x2d"``: coefficient ``a`` has 1 digit (>= 2), the
  solution ``x`` has 2 digits (random sign); ``b`` is non-zero, ``|b| <= 99``
* compare_fractions ``"2d/2d"``: numerators / denominators of the given widths,
  each fraction in lowest terms, proper when widths are equal
"""


def _normalize_difficulty(difficulty: str) -> str:
    return re.sub(r"\s+", "", str(difficulty)).lower().replace("×", "x").replace("%", "mod")


def _parse_difficulty(kind: str, difficulty: str) -> tuple[int, int]:
    if kind not in PROBLEM_KINDS:
        raise ValueError(f"unknown problem kind {kind!r}")
    m = _DIFFICULTY_RE[kind].fullmatch(_normalize_difficulty(difficulty))
    if m is None:
        raise ValueError(f"bad difficulty {difficulty!r} for {kind}; expected like {_DIFFICULTY_RE[kind].pattern}")
    d1, d2 = int(m.group(1)), int(m.group(2))
    if not (1 <= d1 <= MAX_DIGITS and 1 <= d2 <= MAX_DIGITS):
        raise ValueError(f"digit counts must be in 1..{MAX_DIGITS}: {difficulty!r}")
    return d1, d2


def _rand_digits(rng: random.Random, digits: int, minimum: int = 1) -> int:
    """Uniform integer with exactly ``digits`` digits and value >= ``minimum``."""
    lo = max(10 ** (digits - 1), minimum)
    hi = 10**digits - 1
    if lo > hi:
        raise ValueError(f"no {digits}-digit integer >= {minimum}")
    return rng.randint(lo, hi)


def _retry(sample: Callable[[], dict[str, int] | None]) -> dict[str, int]:
    for _ in range(_MAX_SAMPLING_ATTEMPTS):
        out = sample()
        if out is not None:
            return out
    raise ValueError("could not sample operands satisfying the difficulty constraints")


def _sample_fraction(rng: random.Random, num_digits: int, den_digits: int) -> tuple[int, int]:
    def attempt() -> dict[str, int] | None:
        q = _rand_digits(rng, den_digits, minimum=2)
        p = _rand_digits(rng, num_digits)
        if math.gcd(p, q) != 1 or (num_digits == den_digits and p >= q):
            return None
        return {"p": p, "q": q}

    frac = _retry(attempt)
    return frac["p"], frac["q"]


def _sample_operands(kind: str, d1: int, d2: int, rng: random.Random) -> dict[str, int]:
    if kind == "add":
        return {"a": _rand_digits(rng, d1), "b": _rand_digits(rng, d2)}
    if kind == "sub":

        def sub() -> dict[str, int] | None:
            a, b = _rand_digits(rng, d1), _rand_digits(rng, d2)
            return {"a": a, "b": b} if a != b else None

        return _retry(sub)
    if kind == "mul":
        return {"a": _rand_digits(rng, d1, minimum=2), "b": _rand_digits(rng, d2, minimum=2)}
    if kind == "mod":

        def mod() -> dict[str, int] | None:
            a, b = _rand_digits(rng, d1), _rand_digits(rng, d2, minimum=2)
            return {"a": a, "b": b} if a > b else None

        return _retry(mod)
    if kind == "linear_equation":
        a = _rand_digits(rng, d1, minimum=2)
        x = _rand_digits(rng, d2) * rng.choice((1, -1))
        b = rng.randint(1, 99) * rng.choice((1, -1))
        return {"a": a, "b": b, "c": a * x + b}

    # compare_fractions: distinct values, and "non-trivial": one fraction must not
    # dominate the other on both numerator and denominator, which would allow a
    # surface heuristic ("bigger top, smaller bottom") instead of a real comparison.
    def compare() -> dict[str, int] | None:
        p, q = _sample_fraction(rng, d1, d2)
        r, s = _sample_fraction(rng, d1, d2)
        if Fraction(p, q) == Fraction(r, s):
            return None
        if not ((p < r and q < s) or (p > r and q > s)):
            return None
        return {"p": p, "q": q, "r": r, "s": s}

    return _retry(compare)


# --------------------------------------------------------------------------- #
# Problem sets
# --------------------------------------------------------------------------- #

DEFAULT_BENCHMARK_SPEC: list[dict[str, Any]] = [
    {"kind": "add", "difficulty": "2d+2d", "weight": 3},  # easy anchor, mostly correct
    {"kind": "add", "difficulty": "3d+3d", "weight": 3},
    {"kind": "add", "difficulty": "5d+5d", "weight": 3},  # long carry chains
    {"kind": "sub", "difficulty": "3d-3d", "weight": 3},  # ~half the results negative
    {"kind": "mul", "difficulty": "2dx2d", "weight": 3},
    {"kind": "mul", "difficulty": "3dx2d", "weight": 3},  # errors likely
    {"kind": "linear_equation", "difficulty": "a1d,x2d", "weight": 3},
    {"kind": "compare_fractions", "difficulty": "2d/2d", "weight": 2},
    {"kind": "mod", "difficulty": "3dmod1d", "weight": 2},
]
"""25-problem mix meant to give a ~1.5B instruct model both right and wrong
answers. Weights are integer counts summing to 25, so ``default_benchmark()``
reproduces them exactly; other ``n`` are apportioned proportionally."""


@dataclass(frozen=True)
class _SpecEntry:
    kind: str
    difficulty: str
    digits: tuple[int, int]
    weight: Fraction


def _normalize_spec(spec: Sequence[Mapping[str, Any]]) -> list[_SpecEntry]:
    entries: list[_SpecEntry] = []
    for raw in spec:
        if not isinstance(raw, Mapping) or "kind" not in raw or "difficulty" not in raw:
            raise ValueError(f"spec entries need 'kind' and 'difficulty': {raw!r}")
        kind = raw["kind"]
        weight = raw.get("weight", 1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float, Fraction)) or not weight > 0:
            raise ValueError(f"spec weight must be a positive number: {raw!r}")
        digits = _parse_difficulty(kind, raw["difficulty"])
        entries.append(_SpecEntry(kind, _normalize_difficulty(raw["difficulty"]), digits, Fraction(weight)))
    return entries


def _apportion(n: int, weights: Sequence[Fraction]) -> list[int]:
    """Largest-remainder apportionment of ``n`` items; ties go to earlier entries."""
    total = sum(weights)
    quotas = [n * w / total for w in weights]
    counts = [math.floor(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def _slug(difficulty: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", difficulty.lower())


def generate_problems(
    n: int,
    seed: int = 0,
    kinds: Sequence[ProblemKind] | None = None,
    spec: Sequence[Mapping[str, Any]] | None = None,
) -> list[MathProblem]:
    """Generate ``n`` problems deterministically.

    Args:
        n: Number of problems.
        seed: Seed for a private ``random.Random``; identical
            ``(n, seed, kinds, spec)`` always yield identical problems.
        kinds: Restrict to these kinds (filters ``spec``, or the default spec).
        spec: Mix as ``[{"kind": ..., "difficulty": ..., "weight": ...}, ...]``;
            defaults to :data:`DEFAULT_BENCHMARK_SPEC`. ``n`` is split across
            entries in proportion to weight (largest remainder method).

    Returns:
        Problems grouped in spec order, with ids ``"<kind>-<difficulty>-<index>-s<seed>"``
        (e.g. ``"add-3d3d-00007-s42"``), unique within the set. Duplicate
        operand sets are avoided whenever the difficulty admits enough of them.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    entries = _normalize_spec(spec if spec is not None else DEFAULT_BENCHMARK_SPEC)
    if kinds is not None:
        unknown = [k for k in kinds if k not in PROBLEM_KINDS]
        if unknown:
            raise ValueError(f"unknown problem kinds {unknown}")
        entries = [e for e in entries if e.kind in set(kinds)]
    if not entries:
        raise ValueError("no spec entries left to generate from")

    rng = random.Random(seed)
    problems: list[MathProblem] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for entry, count in zip(entries, _apportion(n, [e.weight for e in entries])):
        for _ in range(count):
            for _attempt in range(_MAX_DEDUP_ATTEMPTS):
                operands = _sample_operands(entry.kind, *entry.digits, rng)
                key = (entry.kind, tuple(sorted(operands.items())))
                if key not in seen:
                    break
            seen.add(key)
            pid = f"{_KIND_PREFIX[entry.kind]}-{_slug(entry.difficulty)}-{len(problems):05d}-s{seed}"
            problem = make_problem(entry.kind, operands, id=pid)  # type: ignore[arg-type]
            if problem.difficulty != entry.difficulty:  # internal consistency check
                raise AssertionError(f"sampled {problem.difficulty!r} for spec {entry.difficulty!r}")
            problems.append(problem)
    return problems


def default_benchmark(seed: int = 0, n: int = 25) -> list[MathProblem]:
    """The default mixed benchmark (:data:`DEFAULT_BENCHMARK_SPEC`)."""
    return generate_problems(n, seed=seed, spec=copy.deepcopy(DEFAULT_BENCHMARK_SPEC))


DEMO_PROBLEM: MathProblem = make_problem("add", {"a": 37, "b": 58}, id="demo-37-plus-58")
""""What is 37 + 58? Give only the answer." -> "95"."""
