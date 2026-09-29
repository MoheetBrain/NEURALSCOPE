"""Tests for neuralscope.mathlab: problems, parser, oracle, alignment."""

from __future__ import annotations

import json
import random
import re
import time
from fractions import Fraction

import pytest

from neuralscope.mathlab import (
    DEFAULT_BENCHMARK_SPEC,
    DEMO_PROBLEM,
    PROBLEM_KINDS,
    MathProblem,
    ParsedAnswer,
    Verification,
    compute_oracle_answer,
    default_benchmark,
    first_divergence,
    generate_problems,
    make_problem,
    parse_answer,
    verify,
    verify_expression,
)

ALL_KINDS = list(PROBLEM_KINDS)


def _independent_answer(kind: str, ops: dict) -> str:
    """Recompute an answer with plain ints / fractions.Fraction (no SymPy)."""
    if kind == "add":
        return str(ops["a"] + ops["b"])
    if kind == "sub":
        return str(ops["a"] - ops["b"])
    if kind == "mul":
        return str(ops["a"] * ops["b"])
    if kind == "linear_equation":
        x = Fraction(ops["c"] - ops["b"], ops["a"])
        assert x.denominator == 1, "generated linear equations must have integer solutions"
        return str(x.numerator)
    if kind == "compare_fractions":
        left, right = Fraction(ops["p"], ops["q"]), Fraction(ops["r"], ops["s"])
        assert left != right
        return f"{ops['p']}/{ops['q']}" if left > right else f"{ops['r']}/{ops['s']}"
    if kind == "mod":
        return str(ops["a"] % ops["b"])
    raise AssertionError(kind)


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "operands", "expected"),
    [
        ("add", {"a": 37, "b": 58}, "95"),
        ("mul", {"a": 73, "b": 14}, "1022"),
        ("add", {"a": 12345, "b": 67890}, "80235"),
        ("linear_equation", {"a": 3, "b": 7, "c": 25}, "6"),
        ("linear_equation", {"a": 5, "b": -12, "c": -242}, "-46"),
        # 17/23 ~ 0.7391 > 19/27 ~ 0.7037: the FIRST fraction is larger here.
        ("compare_fractions", {"p": 17, "q": 23, "r": 19, "s": 27}, "17/23"),
        ("compare_fractions", {"p": 19, "q": 27, "r": 17, "s": 23}, "17/23"),
        ("compare_fractions", {"p": 1, "q": 3, "r": 2, "s": 5}, "2/5"),
        ("mod", {"a": 47, "b": 6}, "5"),
        ("mod", {"a": 48, "b": 6}, "0"),
        ("sub", {"a": 123, "b": 456}, "-333"),
        ("sub", {"a": 456, "b": 123}, "333"),
    ],
)
def test_oracle_known_answers(kind, operands, expected):
    assert compute_oracle_answer(kind, operands) == expected


def test_oracle_fraction_answer_is_written_form_not_reduced():
    # The larger fraction is returned exactly as written in the question.
    assert compute_oracle_answer("compare_fractions", {"p": 2, "q": 4, "r": 1, "s": 3}) == "2/4"


@pytest.mark.parametrize(
    ("kind", "operands", "exc"),
    [
        ("pow", {"a": 1, "b": 2}, ValueError),
        ("add", {"a": 1}, ValueError),
        ("add", {"a": 1, "b": 2, "c": 3}, ValueError),
        ("add", {"a": 1.5, "b": 2}, TypeError),
        ("add", {"a": True, "b": 2}, TypeError),
        ("add", {"a": "1", "b": 2}, TypeError),
        ("linear_equation", {"a": 0, "b": 1, "c": 2}, ValueError),
        ("compare_fractions", {"p": 1, "q": 2, "r": 2, "s": 4}, ValueError),  # equal values
        ("compare_fractions", {"p": 1, "q": 0, "r": 2, "s": 3}, ValueError),
        ("mod", {"a": 5, "b": 0}, ValueError),
        ("mod", {"a": -5, "b": 3}, ValueError),
    ],
)
def test_oracle_rejects_invalid_operands(kind, operands, exc):
    with pytest.raises(exc):
        compute_oracle_answer(kind, operands)


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #


def test_demo_problem():
    assert DEMO_PROBLEM.id == "demo-37-plus-58"
    assert DEMO_PROBLEM.question == "What is 37 + 58? Give only the answer."
    assert DEMO_PROBLEM.expression == "37 + 58"
    assert DEMO_PROBLEM.oracle_answer == "95"
    assert DEMO_PROBLEM.answer_type == "integer"
    assert DEMO_PROBLEM.difficulty == "2d+2d"
    assert DEMO_PROBLEM.operands == {"a": 37, "b": 58}


@pytest.mark.parametrize(
    ("kind", "operands", "question", "expression", "answer", "difficulty"),
    [
        ("add", {"a": 37, "b": 58}, "What is 37 + 58? Give only the answer.", "37 + 58", "95", "2d+2d"),
        ("sub", {"a": 123, "b": 456}, "What is 123 - 456? Give only the answer.", "123 - 456", "-333", "3d-3d"),
        ("mul", {"a": 73, "b": 14}, "What is 73 * 14? Give only the answer.", "73 * 14", "1022", "2dx2d"),
        (
            "linear_equation",
            {"a": 3, "b": 7, "c": 25},
            "Solve for x: 3x + 7 = 25. Reply with only the integer value of x, no explanation.",
            "3*x + 7 = 25",
            "6",
            "a1d,x1d",
        ),
        (
            "linear_equation",
            {"a": 3, "b": -7, "c": 20},
            "Solve for x: 3x - 7 = 20. Reply with only the integer value of x, no explanation.",
            "3*x - 7 = 20",
            "9",
            "a1d,x1d",
        ),
        (
            "linear_equation",
            {"a": 4, "b": 0, "c": -48},
            "Solve for x: 4x = -48. Reply with only the integer value of x, no explanation.",
            "4*x = -48",
            "-12",
            "a1d,x2d",
        ),
        (
            "compare_fractions",
            {"p": 17, "q": 23, "r": 19, "s": 27},
            "Which is larger: 17/23 or 19/27? Reply with only the larger fraction, no explanation.",
            "compare(17/23, 19/27)",
            "17/23",
            "2d/2d",
        ),
        ("mod", {"a": 47, "b": 6}, "What is 47 mod 6? Give only the answer.", "47 mod 6", "5", "2dmod1d"),
    ],
)
def test_make_problem_templates(kind, operands, question, expression, answer, difficulty):
    p = make_problem(kind, operands)
    assert p.question == question
    assert p.expression == expression
    assert p.oracle_answer == answer
    assert p.difficulty == difficulty
    assert p.answer_type == ("rational" if kind == "compare_fractions" else "integer")
    assert p.question.isascii()
    # The canonical expression is always accepted by the free-form verifier.
    assert verify_expression(p.expression, p.oracle_answer).correct


def test_make_problem_default_id_is_deterministic():
    assert make_problem("add", {"a": 37, "b": 58}).id == "add-37-58"
    assert make_problem("linear_equation", {"a": 3, "b": -7, "c": 20}).id == "lin-3-n7-20"
    assert make_problem("add", {"a": 1, "b": 2}, id="custom").id == "custom"


@pytest.mark.parametrize(
    ("kind", "operands"),
    [
        ("linear_equation", {"a": 3, "b": 7, "c": 26}),  # x = 19/3, not an integer
        ("linear_equation", {"a": 1, "b": 7, "c": 26}),  # a < 2
        ("add", {"a": -1, "b": 2}),
        ("compare_fractions", {"p": 3, "q": 6, "r": 1, "s": 2}),  # equal
        ("mod", {"a": 7, "b": 0}),
    ],
)
def test_make_problem_rejects_bad_operands(kind, operands):
    with pytest.raises(ValueError):
        make_problem(kind, operands)


def test_problem_dict_roundtrip_and_json():
    for p in default_benchmark(seed=3):
        d = p.to_dict()
        assert json.loads(json.dumps(d)) == d
        q = MathProblem.from_dict(json.loads(json.dumps(d)))
        assert q == p
        assert hash(q) == hash(p)
    with pytest.raises(ValueError):
        MathProblem.from_dict({"id": "x"})
    bad = DEMO_PROBLEM.to_dict() | {"kind": "pow"}
    with pytest.raises(ValueError):
        MathProblem.from_dict(bad)


def test_to_dict_does_not_share_operands():
    d = DEMO_PROBLEM.to_dict()
    d["operands"]["a"] = 999
    assert DEMO_PROBLEM.operands["a"] == 37


def test_generate_is_deterministic():
    a = generate_problems(25, seed=1)
    b = generate_problems(25, seed=1)
    assert a == b
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_generate_does_not_touch_global_random():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    generate_problems(50, seed=9)
    assert random.random() == expected


def test_different_seeds_differ():
    a = [p.expression for p in generate_problems(25, seed=1)]
    b = [p.expression for p in generate_problems(25, seed=2)]
    assert a != b


def test_ids_unique_and_well_formed():
    problems = generate_problems(300, seed=42, kinds=ALL_KINDS)
    ids = [p.id for p in problems]
    assert len(set(ids)) == len(ids)
    for p in problems:
        assert re.fullmatch(r"(add|sub|mul|lin|cmp|mod)-[0-9a-z]+-\d{5}-s42", p.id), p.id


def test_no_duplicate_operands_when_space_allows():
    problems = generate_problems(200, seed=5)
    keys = {(p.kind, tuple(sorted(p.operands.items()))) for p in problems}
    assert len(keys) == len(problems)


def test_oracle_agrees_with_independent_arithmetic_on_500_problems():
    problems = generate_problems(500, seed=2024, kinds=ALL_KINDS)
    assert len(problems) == 500
    assert {p.kind for p in problems} == set(ALL_KINDS)
    for p in problems:
        assert p.oracle_answer == _independent_answer(p.kind, p.operands), p
        assert verify(p, p.oracle_answer).correct, p
        assert verify_expression(p.expression, p.oracle_answer).correct, p


def test_generated_problem_invariants():
    for p in generate_problems(400, seed=11, kinds=ALL_KINDS):
        ops = p.operands
        if p.kind == "linear_equation":
            assert ops["a"] >= 2 and ops["b"] != 0
            assert (ops["c"] - ops["b"]) % ops["a"] == 0
        if p.kind == "compare_fractions":
            p_, q, r, s = ops["p"], ops["q"], ops["r"], ops["s"]
            assert Fraction(p_, q) != Fraction(r, s)
            # Non-trivial: neither fraction dominates on both numerator and denominator.
            assert (p_ < r and q < s) or (p_ > r and q > s)
            assert p.oracle_answer in (f"{p_}/{q}", f"{r}/{s}")
            assert p.oracle_answer in p.question
        if p.kind == "mod":
            assert ops["b"] >= 2
        assert p.question.endswith(
            ("Give only the answer.", "Reply with only the integer value of x, no explanation.", "Reply with only the larger fraction, no explanation.")
        )


def test_default_benchmark():
    bench = default_benchmark()
    assert len(bench) == 25
    assert {p.kind for p in bench} == set(ALL_KINDS)
    expected = {(e["kind"], e["difficulty"]): e["weight"] for e in DEFAULT_BENCHMARK_SPEC}
    got: dict[tuple[str, str], int] = {}
    for p in bench:
        got[(p.kind, p.difficulty)] = got.get((p.kind, p.difficulty), 0) + 1
    assert got == expected
    assert sum(e["weight"] for e in DEFAULT_BENCHMARK_SPEC) == 25
    assert default_benchmark(seed=0) == bench
    assert default_benchmark(seed=1) != bench
    # Subtraction problems include negative results (for the chosen seed).
    subs = [p for p in default_benchmark(seed=0) if p.kind == "sub"]
    assert any(p.oracle_answer.startswith("-") for p in subs)


def test_default_benchmark_difficulty_shapes():
    for p in default_benchmark(seed=7):
        if p.difficulty == "5d+5d":
            assert all(10_000 <= v <= 99_999 for v in p.operands.values())
        if p.difficulty == "3dx2d":
            assert 100 <= p.operands["a"] <= 999 and 10 <= p.operands["b"] <= 99


def test_generate_with_custom_spec_and_kinds():
    spec = [{"kind": "add", "difficulty": "4d + 1d", "weight": 1}, {"kind": "mul", "difficulty": "1dx1d", "weight": 3}]
    problems = generate_problems(8, seed=0, spec=spec)
    assert [p.kind for p in problems] == ["add"] * 2 + ["mul"] * 6
    assert {p.difficulty for p in problems} == {"4d+1d", "1dx1d"}
    only_mod = generate_problems(10, seed=0, kinds=["mod"])
    assert len(only_mod) == 10 and {p.kind for p in only_mod} == {"mod"}
    assert generate_problems(0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": 5, "kinds": ["pow"]},
        {"n": 5, "spec": [{"kind": "add", "difficulty": "2d*2d"}]},
        {"n": 5, "spec": [{"kind": "add", "difficulty": "2d+2d", "weight": 0}]},
        {"n": 5, "spec": [{"kind": "add", "difficulty": "99d+2d"}]},
        {"n": 5, "spec": [{"kind": "add", "difficulty": "2d+2d"}], "kinds": ["mul"]},
    ],
)
def test_generate_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_problems(**kwargs)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("95", "95"),
        (" 95.", "95"),
        ("95\n", "95"),
        ("95<|im_end|>", "95"),
        ("95<|endoftext|>", "95"),
        ("1,022", "1022"),
        ("12,345,678", "12345678"),
        ("x = 6", "6"),
        ("x=6", "6"),
        ("X = 6.", "6"),
        ("The answer is 95", "95"),
        ("The answer is 95.", "95"),
        ("The final answer is 1,022.", "1022"),
        ("Answer: 95", "95"),
        ("**Answer:** 95", "95"),
        ("**95**", "95"),
        ("\\boxed{95}", "95"),
        ("The final answer is $\\boxed{95}$.", "95"),
        ("\\[\\boxed{1022}\\]", "1022"),
        ("$95$", "95"),
        ("-12", "-12"),
        ("\u221212", "-12"),  # unicode minus
        ("The answer is -12.", "-12"),
        ("x = -6", "-6"),
        ("6.0", "6"),
        ("1,022.00", "1022"),
        ("6.5", "6.5"),
        ("007", "7"),
        ("19/27", "19/27"),
        ("19 / 27", "19/27"),
        ("\\frac{19}{27}", "19/27"),
        ("\\dfrac{38}{54}", "38/54"),
        ("-\\frac{3}{4}", "-3/4"),
        ("37 + 58 = 95", "95"),
        ("73 \\times 14 = 1022", "1022"),
        ("123 - 456 = -333", "-333"),
        ("47 mod 6 = 5", "5"),
        ("x = 18/3 = 6", "6"),
        ("The answer is 37 + 58 = 95.", "95"),
        ("To solve, subtract 7: 3x = 18. Then x = 6.", "6"),
        ("x = 6\n\nCheck: 3(6) + 7 = 25", "6"),
        ("95\n\nExplanation: 37 + 58 = 95", "95"),
        ("The sum of 37 and 58 is 95.", "95"),
        ("So the remainder is 5.", "5"),
        ("95 is the answer.", "95"),
        ("Sure! 95", "95"),
        ("The answer is 1,022, which is a four-digit number.", "1022"),
        ("<|im_start|>assistant\n95<|im_end|>", "95"),
    ],
)
def test_parse_ok(text, value):
    parsed = parse_answer(text)
    assert parsed.status == "ok", parsed
    assert parsed.value == value
    assert parsed.raw_span is not None and parsed.raw_span in text
    assert text[parsed.span[0] : parsed.span[1]] == parsed.raw_span


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("17/23", "17/23"),
        ("17/23 is larger than 19/27.", "17/23"),
        ("The larger fraction is 17/23.", "17/23"),
        ("The larger fraction is 17/23, since 17*27 = 459 > 19*23 = 437.", "17/23"),
        ("The answer is \\(\\frac{19}{27}\\).", "19/27"),
    ],
)
def test_parse_ok_rational(text, value):
    parsed = parse_answer(text, "rational")
    assert parsed.status == "ok", parsed
    assert parsed.value == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "<|im_end|>",
        "I don't know",
        "Step 1: add the ones digits",  # lone numeral inside prose
        "1. First, add the ones digits",  # list enumerator
        "5/0",
        "1,02",  # malformed thousands grouping
        "95.5.5",
        "\\boxed{95",  # unterminated (truncated) box
        "\\boxed{5/0}",
    ],
)
def test_parse_unparseable(text):
    parsed = parse_answer(text)
    assert parsed.status == "unparseable", parsed
    assert parsed.value is None


@pytest.mark.parametrize(
    "text",
    [
        "95 or 96",
        "The answer is 95 or 96.",
        "The answer is 95, 96",
        "\\boxed{95} The answer is 96",
        "\\boxed{95 or 96}",
        "x = 5. Actually x = 6",
        "The sum is 95. The sum is 96.",
        "3x + 7 = 25",  # restating the equation is not an answer
        "17/23 > 19/27",
        "It might be 95",
        "The answer is not 95.",
        "95?",
        "x \u2248 6",
        "The answer is - 12",  # detached minus: sign cannot be read safely
        "- 12",
    ],
)
def test_parse_ambiguous(text):
    parsed = parse_answer(text)
    assert parsed.status == "ambiguous", parsed
    assert parsed.value is None


def test_parse_is_larger_marker_only_for_rational():
    assert parse_answer("17/23 is larger than 19/27", "rational").value == "17/23"
    assert parse_answer("17/23 is larger than 19/27", "integer").status == "ambiguous"


def test_parse_rejects_bad_arguments():
    with pytest.raises(ValueError):
        parse_answer("95", "float")
    with pytest.raises(TypeError):
        parse_answer(95)  # type: ignore[arg-type]
    assert parse_answer(None).status == "unparseable"
    assert parse_answer("9" * 30_000).status == "unparseable"


def test_parsed_answer_to_dict():
    d = parse_answer("The answer is 95.").to_dict()
    assert d["status"] == "ok" and d["value"] == "95" and d["span"] == [14, 16]
    assert json.loads(json.dumps(d)) == d
    assert isinstance(parse_answer("95"), ParsedAnswer)


def test_parser_never_evaluates_expressions():
    # "12 + 3" is not a committed value; the parser must not compute 15.
    assert parse_answer("12 + 3").status != "ok"
    assert parse_answer("The answer is 12 + 3").status != "ok"


@pytest.mark.parametrize(
    "text",
    [
        "(" * 9000 + "1" + ")" * 9000,
        "\\boxed{" * 2400 + "1" + "}" * 2400,
        "\\boxed{1} " * 1900,
        "x = 1. " * 2800,
        "The answer is 1. " * 1100,
        "- " * 9000 + "1",
        "9" * 19_999,
        "1," * 9000,
    ],
)
def test_parser_bounded_on_pathological_inputs(text):
    start = time.perf_counter()
    parsed = parse_answer(text)
    assert time.perf_counter() - start < 2.0
    assert parsed.status in ("ok", "unparseable", "ambiguous")


def test_parser_fuzz_invariants():
    rng = random.Random(0)
    alphabet = list("0123456789 .,/-+*=x()$\\\n") + [
        "\\boxed{",
        "\\frac{",
        "}",
        "answer is ",
        "x = ",
        "**",
        "<|im_end|>",
        " or ",
        "−",
        "mod ",
    ]
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        for answer_type in ("integer", "rational"):
            parsed = parse_answer(text, answer_type)
            if parsed.status == "ok":
                assert parsed.value is not None
                assert text[parsed.span[0] : parsed.span[1]] == parsed.raw_span
                Fraction(parsed.value)  # canonical value is always an exact number
            else:
                assert parsed.value is None and parsed.raw_span is None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

MUL = make_problem("mul", {"a": 73, "b": 14})
CMP = make_problem("compare_fractions", {"p": 17, "q": 23, "r": 19, "s": 27})
LIN = make_problem("linear_equation", {"a": 3, "b": 7, "c": 25})
SUB = make_problem("sub", {"a": 123, "b": 456})
ADD6 = make_problem("add", {"a": 2, "b": 4})


@pytest.mark.parametrize(
    ("problem", "output", "status"),
    [
        (MUL, "1022", "correct"),
        (MUL, "1,022", "correct"),
        (MUL, "The answer is 1022.<|im_end|>", "correct"),
        (MUL, "\\boxed{1022}", "correct"),
        (MUL, "1012", "incorrect"),
        (MUL, "1,012", "incorrect"),
        (MUL, "-1022", "incorrect"),
        (MUL, "I don't know", "unparseable"),
        (MUL, "", "unparseable"),
        (MUL, "1022 or 1012", "ambiguous"),
        (ADD6, "6", "correct"),
        (ADD6, "6.0", "correct"),
        (ADD6, "12/2", "correct"),
        (ADD6, "6.5", "incorrect"),
        (LIN, "x = 6", "correct"),
        (LIN, "x=6", "correct"),
        (LIN, "6", "correct"),
        (LIN, "x = 5", "incorrect"),
        (LIN, "3x + 7 = 25", "ambiguous"),
        (SUB, "-333", "correct"),
        (SUB, "\u2212333", "correct"),
        (SUB, "333", "incorrect"),
        (CMP, "17/23", "correct"),
        (CMP, "34/46", "correct"),  # same value as the larger fraction
        (CMP, "\\frac{17}{23}", "correct"),
        (CMP, "17/23 is larger.", "correct"),
        (CMP, "19/27", "incorrect"),  # the smaller one
        (CMP, "38/54", "incorrect"),  # value of the smaller one
        (CMP, "0.739", "incorrect"),  # a decimal is never exactly the fraction
    ],
)
def test_verify(problem, output, status):
    v = verify(problem, output)
    assert v.status == status, v
    assert v.correct == (status == "correct")
    assert v.oracle_answer == problem.oracle_answer
    if status in ("unparseable", "ambiguous"):
        assert v.parsed_answer is None


def test_verify_notes_smaller_fraction():
    assert "smaller fraction" in verify(CMP, "19/27").note
    assert "neither" in verify(CMP, "1/2").note


def test_verify_demo_and_to_dict():
    v = verify(DEMO_PROBLEM, "95")
    assert isinstance(v, Verification)
    assert v.to_dict() == {
        "correct": True,
        "status": "correct",
        "parsed_answer": "95",
        "oracle_answer": "95",
        "note": v.note,
    }
    assert json.loads(json.dumps(v.to_dict())) == v.to_dict()


def test_verify_missing_output_is_unparseable():
    v = verify(DEMO_PROBLEM, None)
    assert v.status == "unparseable" and not v.correct


def test_verify_detects_tampered_oracle_answer():
    tampered = MathProblem.from_dict(DEMO_PROBLEM.to_dict() | {"oracle_answer": "96"})
    with pytest.raises(ValueError):
        verify(tampered, "96")


@pytest.mark.parametrize(
    ("expression", "answer", "correct"),
    [
        ("37 + 58", "95", True),
        ("73*14", "1022", True),
        ("73*14", "1,022", True),
        ("73*14", "1012", False),
        ("12345 + 67890", "80235", True),
        ("(12-5)*3", "21", True),
        ("3x + 7 = 25", "6", True),
        ("3x + 7 = 25", "x = 6", True),
        ("3*x + 7 = 25", "5", False),
        ("2(x + 1) = 10", "4", True),
        ("25 = 3x + 7", "6", True),
        ("47 mod 6", "5", True),
        ("47 mod 6", "4", False),
        ("compare(17/23, 19/27)", "17/23", True),
        ("compare(17/23, 19/27)", "19/27", False),
        ("123 - 456", "-333", True),
        ("7/2", "3.5", True),
        ("7/2", "7/2", True),
        ("1.5 + 1.5", "3", True),
    ],
)
def test_verify_expression(expression, answer, correct):
    assert verify_expression(expression, answer).correct is correct


def test_verify_expression_unparseable_answer():
    v = verify_expression("37 + 58", "no idea")
    assert v.status == "unparseable" and not v.correct and v.oracle_answer == "95"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "open('x')",
        "1+1; import os",
        "x**x**x**99999",
        "2**10",
        "2 ** 3",
        "9//2",
        "1" * 201,
        "1+" * 100 + "1",
        "",
        "   ",
        "y + 1",
        "x + 1",  # free variable outside an equation
        "x*x = 4",  # not linear
        "0*x = 0",  # no unique solution
        "1/x = 2",
        "x = x",
        "1 = 2 = 3",
        "1/0",
        "47 mod 0",
        "compare(1/2, 2/4)",  # equal fractions
        "compare(1/2)",
        "1, 2",
        "lambda: 1",
        "exec('1')",
        "sympy.Integer(5)",
        "1e5",
        "5!",
        "(1 + 2",
        "()",
        "1 2",
        "1 +",
    ],
)
def test_verify_expression_rejects_unsafe_or_invalid(expression):
    with pytest.raises(ValueError):
        verify_expression(expression, "1")


def test_verify_expression_rejects_non_string():
    with pytest.raises(TypeError):
        verify_expression(123, "1")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model", "oracle", "expected"),
    [
        ("1012", "1022", {"index": 2, "model_char": "1", "oracle_char": "2", "common_prefix": "10"}),
        ("1022", "1022", {"index": None, "model_char": None, "oracle_char": None, "common_prefix": "1022"}),
        ("95", "950", {"index": 2, "model_char": None, "oracle_char": "0", "common_prefix": "95"}),
        ("9500", "950", {"index": 3, "model_char": "0", "oracle_char": None, "common_prefix": "950"}),
        (None, "95", {"index": 0, "model_char": None, "oracle_char": "9", "common_prefix": ""}),
        ("", "95", {"index": 0, "model_char": None, "oracle_char": "9", "common_prefix": ""}),
        ("12", "-12", {"index": 0, "model_char": "1", "oracle_char": "-", "common_prefix": ""}),
        ("19/27", "17/23", {"index": 1, "model_char": "9", "oracle_char": "7", "common_prefix": "1"}),
    ],
)
def test_first_divergence(model, oracle, expected):
    assert first_divergence(model, oracle) == expected


def test_first_divergence_on_parsed_values():
    parsed = parse_answer("The answer is 1,012.")
    assert first_divergence(parsed.value, MUL.oracle_answer)["index"] == 2
