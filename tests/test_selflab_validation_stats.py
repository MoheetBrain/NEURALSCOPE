"""SA6 independent validation — statistics against synthetic data with KNOWN outcomes.

Adversarial complements to tests/test_selflab_stats.py (SA3):

* deterministic repeats are NOT independent units: duplicating every observation k times must leave the
  clustered estimate, interval and ``n_units`` exactly unchanged;
* cluster-bootstrap coverage is near nominal on clustered data, and a naive (unclustered) interval is shown
  to under-cover on the same data — so the test would catch an analysis that forgot ``groups``;
* the paired effect counts units, not pairs, and is invariant to how many paraphrases/label orders a unit has;
* multiplicity corrections are monotone, never below the raw p-value, and Holm is uniformly more conservative
  than BH.

Pure numpy; no model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

S = pytest.importorskip("neuralscope.selflab.stats")


def _clustered(rng, G, m, *, sd_u=1.0, sd_e=0.3, binary=False):
    vals, groups = [], []
    for g in range(G):
        u = rng.normal(0, sd_u)
        for _ in range(m):
            if binary:
                vals.append(float(rng.random() < 1.0 / (1.0 + math.exp(-u))))
            else:
                vals.append(u + rng.normal(0, sd_e))
            groups.append(f"item-{g}")
    return vals, groups


# ------------------------------------------------------------------------------------------ repeats ≠ units
def test_deterministic_repeats_do_not_shrink_the_clustered_interval():
    """A deterministic model asked the same item k times yields k identical rows. The clustered summary must
    be EXACTLY the summary of one row per item (same estimate, same CI, same n_units) — repeats add nothing."""
    rng = np.random.default_rng(7)
    G = 25
    single = [bool(rng.random() < 0.4) for _ in range(G)]
    groups1 = [f"e{g}" for g in range(G)]
    for k in (2, 5, 12):
        dup = [s for s in single for _ in range(k)]
        groups_k = [g for g in groups1 for _ in range(k)]
        a = S.proportion_summary(single, groups1, B=1000, seed=3)
        b = S.proportion_summary(dup, groups_k, B=1000, seed=3)
        assert b["n_units"] == G == a["n_units"]
        assert b["n_obs"] == G * k
        assert b["estimate"] == a["estimate"]
        assert (b["ci_low"], b["ci_high"]) == (a["ci_low"], a["ci_high"])
        # The Wilson interval (independence assumption) DOES shrink — it must not be the reported CI.
        assert b["wilson_high"] - b["wilson_low"] < a["wilson_high"] - a["wilson_low"]
        assert (b["ci_low"], b["ci_high"]) != (b["wilson_low"], b["wilson_high"])


def test_paired_effect_counts_units_not_pairs():
    rng = np.random.default_rng(11)
    G = 14
    base = rng.normal(0.3, 1.0, size=G)
    a, b, groups = [], [], []
    for g in range(G):
        reps = int(rng.integers(1, 9))  # unequal numbers of paraphrases / label orders per unit
        for _ in range(reps):
            a.append(base[g] + 1.0)
            b.append(1.0)
            groups.append(g)
    out = S.paired_effect(a, b, groups, B=500, seed=0)
    assert out["n_units"] == G
    assert out["n_obs"] == len(a)
    # each unit weighted once: the estimate is the plain mean of the unit effects, independent of reps
    assert out["estimate"] == pytest.approx(float(base.mean()), abs=1e-6)
    assert out["test"]["n_units"] == G
    assert out["test"]["method"] == "sign_flip_exact"  # G <= 16


def test_sign_flip_cannot_be_significant_with_too_few_units():
    """With G units the smallest attainable exact two-sided p is 2/2^G: 3 identical huge effects can never
    reach p < 0.05 — a guard against 'significant' results from a handful of items repeated many times."""
    out = S.sign_flip_test([5.0, 5.0, 5.0], alternative="two_sided")
    assert out["p_value"] == pytest.approx(2 / 8)
    out = S.sign_flip_test([5.0] * 4, alternative="greater")
    assert out["p_value"] == pytest.approx(1 / 16)
    rows = [5.0] * 3
    groups = ["a", "b", "c"]
    pe = S.paired_effect([r for r in rows for _ in range(50)], [0.0] * 150, [g for g in groups for _ in range(50)],
                         B=200)
    assert pe["n_units"] == 3 and pe["p_value"] >= 0.25


# ------------------------------------------------------------------------------------------ coverage
@pytest.mark.parametrize("binary", [False, True])
def test_cluster_bootstrap_coverage_near_nominal_and_naive_fails(binary):
    """Seeded simulation (R=200, G=30 units × 6 correlated observations). Truth = 0 (Gaussian) or 0.5
    (logit-normal Bernoulli; E[sigmoid(u)] = 0.5 by symmetry). Clustered coverage must be near 0.95; the naive
    interval (each observation its own unit) must visibly under-cover on the same data."""
    rng = np.random.default_rng(2026)
    truth = 0.5 if binary else 0.0
    R, G, m = 200, 30, 6
    cov_c = cov_n = 0
    for r in range(R):
        vals, groups = _clustered(rng, G, m, binary=binary)
        c = S.bootstrap_mean(vals, groups, B=800, seed=r)
        n = S.bootstrap_mean(vals, None, B=800, seed=r)
        assert c["n_units"] == G and n["n_units"] == G * m
        cov_c += c["ci_low"] <= truth <= c["ci_high"]
        cov_n += n["ci_low"] <= truth <= n["ci_high"]
    cov_c /= R
    cov_n /= R
    # Monte Carlo SE at p=0.95, R=200 is ~0.015; percentile bootstrap with 30 units under-covers slightly.
    assert 0.90 <= cov_c <= 0.99, cov_c
    assert cov_n <= 0.90 and cov_n < cov_c - 0.05, (cov_n, cov_c)


def test_bootstrap_difference_null_is_not_significant_on_paired_clusters():
    """Two arms that are identical within each unit: the clustered difference CI must be exactly [0, 0]."""
    rng = np.random.default_rng(5)
    vals, arm, groups = [], [], []
    for g in range(20):
        u = rng.normal()
        for a in ("doc_v", "doc_alt"):
            vals.append(u)
            arm.append(a)
            groups.append(g)
    out = S.bootstrap_difference(vals, arm, groups, "doc_v", "doc_alt", B=300)
    assert out["estimate"] == 0.0 and out["ci_low"] == 0.0 and out["ci_high"] == 0.0
    assert out["n_units"] == 20


def test_none_and_nan_are_never_returned_as_numbers():
    out = S.bootstrap_mean([], [], B=50)
    assert out["estimate"] is None and out["ci_low"] is None
    one = S.bootstrap_mean([1.0, 1.0, 1.0], ["a", "a", "a"], B=50)
    assert one["n_units"] == 1 and one["ci_low"] is None and one["ci_high"] is None and one.get("note")
    ps = S.proportion_summary([None, None], ["a", "b"], B=50)
    assert ps["estimate"] is None and ps["n_excluded"] == 2
    assert S.clean_float(float("nan")) is None and S.clean_float(float("inf")) is None


# ------------------------------------------------------------------------------------------ multiplicity
def test_multiplicity_corrections_are_monotone_and_conservative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = list(rng.uniform(0, 0.2, size=int(rng.integers(2, 9))))
        h = S.holm(p)["adjusted"]
        q = S.benjamini_hochberg(p)["adjusted"]
        for raw, ha, qa in zip(p, h, q):
            assert ha >= raw - 1e-8 and qa >= raw - 1e-8  # outputs are rounded to 8 places
            assert ha >= qa - 1e-9  # Holm (FWER) is at least as conservative as BH (FDR)
            assert ha <= 1.0 and qa <= 1.0
        order = np.argsort(p)
        assert all(h[order[i]] <= h[order[i + 1]] + 1e-12 for i in range(len(p) - 1))
    # one comparison: no correction
    assert S.holm([0.03])["adjusted"] == [0.03]
    # None passes through and does not count as a comparison
    assert S.holm([0.01, None, 0.04])["adjusted"] == [0.02, None, 0.04]


def test_uncorrected_family_would_inflate_fwer_but_holm_controls_it():
    """Known-outcome sanity check that motivates reporting adjusted p-values: 6 null comparisons."""
    rng = np.random.default_rng(1)
    raw_any = holm_any = 0
    R = 2000
    for _ in range(R):
        p = rng.uniform(size=6)
        raw_any += bool((p <= 0.05).any())
        holm_any += any(S.holm(list(p))["reject"])
    assert raw_any / R > 0.2  # ≈ 1 − 0.95^6 = 0.26
    assert holm_any / R <= 0.065
