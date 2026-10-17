"""Experiment B (CSP-1 carry regression) items: 2-digit addition problems with carry / no-carry labels.

Reuses the problem logic of ``scripts/experiments/carry_heads.py`` (E1) and ``carry_patch.py`` (E2) WITHOUT
importing or modifying them (they import torch and load a model at call time):

* same prompt template ``"What is {a} + {b}? Give only the answer."``;
* same pools: two-digit a, b with a + b < 100; carry iff a_u + b_u >= 10; no-carry problems only when the
  "spurious carry" digit a_t + b_t + 1 <= 9 exists (so every item has an alternative tens digit);
* ``make_carry_problems(n, seed)`` with no exclusions reproduces E1's ``build_problems(n, seed)`` exactly
  (same RNG calls in the same order), so ``make_carry_problems(40, 0)`` is the E1 problem set.

``carry_splits`` returns calibration and held-out problems with DISJOINT unordered operand pairs, and by default
excludes every problem used in E1 (n=40, seed 0) and E2 (40 pairs, seed 0), on which the L22.H10/H11 mechanism was
established. CSP-1 on this task is a regression/calibration task, not a discovery.
"""

from __future__ import annotations

import random

PROMPT_TEMPLATE = "What is {a} + {b}? Give only the answer."
CARRY_ITEMS_VERSION = "carry_items@1"


def _row(a: int, b: int) -> dict:
    at, au, bt, bu = a // 10, a % 10, b // 10, b % 10
    carry = au + bu >= 10
    return {
        "problem_id": f"add-{a:02d}+{b:02d}",
        "a": a, "b": b, "sum": a + b, "carry": carry,
        "true_tens": (a + b) // 10,
        "alt_tens": at + bt if carry else at + bt + 1,  # the digit you get by mis-handling the carry
        "prompt": PROMPT_TEMPLATE.format(a=a, b=b),
    }


def carry_pools() -> tuple[list[dict], list[dict]]:
    """(carry, no_carry) pools in E1's enumeration order."""
    carry, nocarry = [], []
    for a in range(10, 100):
        for b in range(10, 100):
            if a + b >= 100:
                continue
            at, au, bt, bu = a // 10, a % 10, b // 10, b % 10
            if au + bu >= 10:
                carry.append(_row(a, b))
            elif at + bt + 1 <= 9:
                nocarry.append(_row(a, b))
    return carry, nocarry


def _key(p: dict) -> tuple[int, int]:
    return (min(p["a"], p["b"]), max(p["a"], p["b"]))


def make_carry_problems(n_per_group: int, seed: int, exclude_pairs: set[tuple[int, int]] | None = None) -> list[dict]:
    """``n_per_group`` carry then ``n_per_group`` no-carry problems. With no exclusions this equals E1's
    ``build_problems(n_per_group, seed)``. ``exclude_pairs`` holds unordered (min, max) operand pairs."""
    carry, nocarry = carry_pools()
    if exclude_pairs:
        carry = [p for p in carry if _key(p) not in exclude_pairs]
        nocarry = [p for p in nocarry if _key(p) not in exclude_pairs]
    rng = random.Random(seed)
    return rng.sample(carry, n_per_group) + rng.sample(nocarry, n_per_group)


def e1_problem_pairs(n_per_group: int = 40, seed: int = 0) -> set[tuple[int, int]]:
    """Unordered operand pairs of the E1 problem set (``docs/research/E1_carry_heads.json``)."""
    return {_key(p) for p in make_carry_problems(n_per_group, seed)}


def e2_problem_pairs(n_pairs: int = 40, seed: int = 0) -> set[tuple[int, int]]:
    """Unordered operand pairs of the E2 N and C problems (replicates ``carry_patch.pairs``)."""
    rng = random.Random(seed)
    out: set[tuple[int, int]] = set()
    count = 0
    nu = [(u1, u2) for u1 in range(10) for u2 in range(10) if u1 + u2 < 10]
    cu = [(u1, u2) for u1 in range(10) for u2 in range(10) if u1 + u2 >= 10]
    while count < n_pairs:
        t1, t2 = rng.randint(1, 8), rng.randint(1, 8)
        if t1 + t2 + 1 > 9:
            continue
        (a1, a2), (c1, c2) = rng.choice(nu), rng.choice(cu)
        for x, y in ((10 * t1 + a1, 10 * t2 + a2), (10 * t1 + c1, 10 * t2 + c2)):
            out.add((min(x, y), max(x, y)))
        count += 1
    return out


def carry_splits(n_calibration_per_group: int, n_heldout_per_group: int, seed: int,
                 exclude_known: bool = True) -> dict[str, list[dict]]:
    """Calibration and held-out problems with disjoint unordered operand pairs (and no a+b / b+a leakage).

    ``exclude_known`` removes the E1/E2 problems from both splits. Each row gets a ``split`` key.
    """
    excluded = (e1_problem_pairs() | e2_problem_pairs()) if exclude_known else set()
    carry, nocarry = carry_pools()
    rng = random.Random(f"carry_items.splits:{seed}")
    out: dict[str, list[dict]] = {"calibration": [], "heldout": []}
    for pool in (carry, nocarry):
        avail = [p for p in pool if _key(p) not in excluded]
        rng.shuffle(avail)
        used: set[tuple[int, int]] = set()
        for split, n in (("calibration", n_calibration_per_group), ("heldout", n_heldout_per_group)):
            taken = 0
            for p in avail:
                if taken == n:
                    break
                if _key(p) in used:
                    continue
                used.add(_key(p))
                out[split].append({**p, "split": split})
                taken += 1
            if taken < n:
                raise ValueError(f"not enough problems for {split}: wanted {n}, got {taken}")
    return out
