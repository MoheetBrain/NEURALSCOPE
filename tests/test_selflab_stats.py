"""SelfLab statistics: validated on synthetic data with KNOWN answers (numpy only, no model)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuralscope.selflab import stats as S

REQUIRED = {"n_units", "n_obs", "estimate", "ci_low", "ci_high", "method", "assumptions"}


# ------------------------------------------------------------------------------------------ Wilson
@pytest.mark.parametrize("k,n,lo,hi", [
    (5, 10, 0.236593, 0.763407),   # textbook reference values (z = 1.959964)
    (0, 10, 0.0, 0.277533),
    (10, 10, 0.722467, 1.0),
    (81, 263, 0.255289, 0.366210),  # hand-computed: centre 0.310750, half-width 0.055461
])
def test_wilson_reference_values(k, n, lo, hi):
    a, b = S.wilson_ci(k, n)
    assert a == pytest.approx(lo, abs=2e-5)
    assert b == pytest.approx(hi, abs=2e-5)


def test_wilson_degenerate():
    assert S.wilson_ci(0, 0) == (None, None)
    with pytest.raises(ValueError):
        S.wilson_ci(3, 2)


def test_wilson_coverage_simulation():
    rng = np.random.default_rng(1)
    p, n, reps = 0.3, 40, 2000
    ks = rng.binomial(n, p, size=reps)
    cover = np.mean([(lambda ci: ci[0] <= p <= ci[1])(S.wilson_ci(int(k), n)) for k in ks])
    assert 0.92 <= cover <= 0.98


# ------------------------------------------------------------------------------------------ cluster bootstrap
def _clustered(rng, G=30, m=5, mu=0.0, sd_u=1.0, sd_e=0.5):
    vals, groups = [], []
    for g in range(G):
        u = rng.normal(0, sd_u)
        for _ in range(m):
            vals.append(mu + u + rng.normal(0, sd_e))
            groups.append(f"g{g}")
    return np.array(vals), groups


def test_summary_dict_contract():
    rng = np.random.default_rng(0)
    x, g = _clustered(rng)
    out = S.bootstrap_mean(x, g, B=200)
    assert REQUIRED <= set(out)
    assert out["n_units"] == 30 and out["n_obs"] == 150
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]


def test_cluster_bootstrap_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    x, g = _clustered(rng)
    a = S.bootstrap_mean(x, g, B=300, seed=7)
    b = S.bootstrap_mean(x, g, B=300, seed=7)
    c = S.bootstrap_mean(x, g, B=300, seed=8)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_cluster_bootstrap_coverage_vs_naive_under_clustering():
    """Repeated observations of one unit are NOT independent evidence: the cluster bootstrap keeps
    near-nominal coverage while the naive i.i.d. interval badly under-covers (design effect ≈ 4)."""
    rng = np.random.default_rng(123)
    reps, mu = 250, 1.5
    cover_cluster = cover_naive = 0
    for r in range(reps):
        x, g = _clustered(rng, G=30, m=5, mu=mu)
        out = S.bootstrap_mean(x, g, B=400, seed=r)
        cover_cluster += out["ci_low"] <= mu <= out["ci_high"]
        se = x.std(ddof=1) / math.sqrt(len(x))
        cover_naive += abs(x.mean() - mu) <= 1.96 * se
    cc, cn = cover_cluster / reps, cover_naive / reps
    assert 0.88 <= cc <= 0.98, cc
    assert cn < 0.80, cn


def test_generic_statistic_matches_fast_path():
    rng = np.random.default_rng(5)
    x, g = _clustered(rng, G=20, m=3)
    fast = S.bootstrap_mean(x, g, B=500, seed=1)
    slow = S.cluster_bootstrap(g, lambda idx: float(x[idx].mean()), B=500, seed=1)
    assert fast["estimate"] == pytest.approx(slow["estimate"])
    # different RNG draws (multinomial vs integer indices) but the same sampling distribution
    assert fast["ci_low"] == pytest.approx(slow["ci_low"], abs=0.25)
    assert fast["ci_high"] == pytest.approx(slow["ci_high"], abs=0.25)


def test_single_group_has_no_interval():
    out = S.bootstrap_mean([1.0, 2.0, 3.0], ["a", "a", "a"], B=100)
    assert out["n_units"] == 1 and out["ci_low"] is None and out["estimate"] == pytest.approx(2.0)


def test_proportion_summary_reports_both_intervals():
    succ = [1, 1, 0, 1, 0, 0, 1, 1, None]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "d"]
    out = S.proportion_summary(succ, groups, B=500)
    assert out["k"] == 5 and out["n_obs"] == 8 and out["n_units"] == 4 and out["n_excluded"] == 1
    assert out["wilson_low"] is not None and out["ci_low"] is not None
    plain = S.proportion_summary([True, False, True])
    assert plain["method"] == "wilson" and plain["estimate"] == pytest.approx(2 / 3, abs=1e-6)


def test_bootstrap_difference_detects_planted_shift():
    rng = np.random.default_rng(9)
    vals, arm, groups = [], [], []
    for gi in range(40):
        u = rng.normal(0, 1)
        for a, shift in (("A", 1.0), ("B", 0.0)):
            vals.append(u + shift + rng.normal(0, 0.3))
            arm.append(a)
            groups.append(gi)
    out = S.bootstrap_difference(vals, arm, groups, "A", "B", B=500)
    assert out["ci_low"] > 0.7 and out["ci_high"] < 1.3


# ------------------------------------------------------------------------------------------ permutation test
def test_sign_flip_exact_small_example():
    # all positive, G = 5: only the identity (and nothing else) is as extreme in the 'greater' direction
    out = S.sign_flip_test([1, 2, 3, 4, 5], alternative="greater")
    assert out["method"] == "sign_flip_exact"
    assert out["p_value"] == pytest.approx(1 / 32)
    two = S.sign_flip_test([1, 2, 3, 4, 5], alternative="two_sided")
    assert two["p_value"] == pytest.approx(2 / 32)


def test_sign_flip_pvalues_uniform_under_null():
    rng = np.random.default_rng(42)
    ps = []
    for r in range(400):
        d = rng.normal(0, 1, size=25)  # G = 25 → Monte Carlo branch
        ps.append(S.sign_flip_test(d, n_perm=400, seed=r)["p_value"])
    ps = np.array(ps)
    assert 0.02 <= np.mean(ps <= 0.05) <= 0.09
    assert 0.44 <= ps.mean() <= 0.56
    # roughly flat histogram: every decile holds 5–15 % of p-values
    hist, _ = np.histogram(ps, bins=10, range=(0, 1))
    assert hist.min() >= 0.05 * len(ps) and hist.max() <= 0.15 * len(ps)


def test_sign_flip_detects_planted_effect():
    rng = np.random.default_rng(7)
    hits = sum(S.sign_flip_test(rng.normal(0.8, 1, size=30), n_perm=1000, seed=i,
                                alternative="greater")["p_value"] < 0.01 for i in range(50))
    assert hits >= 45


def test_paired_effect_clusters_by_unit():
    rng = np.random.default_rng(3)
    a, b, g = [], [], []
    for unit in range(20):
        base = rng.normal(0, 2)
        for rep in range(4):  # 4 correlated pairs per unit
            a.append(base + 0.5 + rng.normal(0, 0.1))
            b.append(base + rng.normal(0, 0.1))
            g.append(unit)
    out = S.paired_effect(a, b, g, B=500, n_perm=2000)
    assert REQUIRED <= set(out)
    assert out["n_units"] == 20 and out["n_obs"] == 80
    assert out["estimate"] == pytest.approx(0.5, abs=0.06)
    assert out["ci_low"] < 0.5 < out["ci_high"]
    assert out["p_value"] < 0.001


# ------------------------------------------------------------------------------------------ multiplicity
def test_holm_reference():
    out = S.holm([0.01, 0.04, 0.03, 0.005])
    assert out["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert out["reject"] == [True, False, False, True]


def test_bh_reference():
    out = S.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert out["adjusted"] == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert all(out["reject"])
    out2 = S.benjamini_hochberg([0.5, None, 0.01])
    assert out2["adjusted"][1] is None and out2["adjusted"][2] == pytest.approx(0.02)


def test_holm_controls_fwer_under_null():
    rng = np.random.default_rng(0)
    fw = np.mean([any(S.holm(list(rng.uniform(size=8)))["reject"]) for _ in range(2000)])
    assert fw <= 0.065


# ------------------------------------------------------------------------------------------ scoring rules
def test_brier_and_log_score_known_values():
    assert S.brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert S.brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert S.log_score([0.5, 0.5], [1, 0]) == pytest.approx(-1.0)  # bits
    assert S.log_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    # multiclass
    P = [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]
    assert S.brier_score(P, [0, 2]) == pytest.approx(((0.09 + 0.04 + 0.01) + (0.01 + 0.01 + 0.04)) / 2)
    assert S.log_score(P, [0, 2]) == pytest.approx((math.log2(0.7) + math.log2(0.8)) / 2)
    # confident miss is clipped, not -inf
    assert math.isfinite(S.log_score([0.0], [1]))


def test_calibration_bins_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(11)
    p = rng.uniform(size=20000)
    y_cal = rng.uniform(size=p.size) < p
    y_mis = rng.uniform(size=p.size) < p ** 3
    good = S.calibration_bins(p, y_cal, n_bins=10)
    bad = S.calibration_bins(p, y_mis, n_bins=10)
    assert good["ece"] < 0.02
    assert bad["ece"] > 0.15
    assert sum(b["n"] for b in good["bins"]) == 20000
    q = S.calibration_bins(p, y_cal, n_bins=5, strategy="quantile")
    assert all(b["n"] > 3000 for b in q["bins"])
