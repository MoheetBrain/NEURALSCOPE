"""SelfLab statistics: numpy + stdlib only (scipy is deliberately not a dependency).

Inference unit
--------------
Every SelfLab experiment has an INDEPENDENT UNIT (an invented entity, an addition problem, a
single case). Trials that share a unit are NOT independent evidence: a deterministic model asked
the same thing twice gives the same answer twice, and paraphrases or label orders of one item are
strongly correlated. All interval estimates and tests here therefore resample or permute whole
groups (``group_id``), never individual observations, unless a function says otherwise.

Every summary helper returns a plain, JSON-serialisable dict with at least::

    {"n_units", "n_obs", "estimate", "ci_low", "ci_high", "method", "assumptions"}

``estimate``/``ci_*`` are ``None`` (never NaN) when they cannot be computed, with the reason in
``note``. Validation on synthetic data with known answers lives in ``tests/test_selflab_stats.py``
(bootstrap coverage under clustering, uniform permutation p-values under the null, power against a
planted effect, reference values for Wilson/Holm/BH, calibrated vs miscalibrated forecasts).
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Any, Callable, Hashable, Iterable, Literal, Sequence

import numpy as np

STATS_VERSION = "selflab.stats@1"
Alternative = Literal["greater", "less", "two_sided"]

__all__ = [
    "STATS_VERSION", "wilson_ci", "proportion_summary", "cluster_bootstrap", "bootstrap_mean",
    "bootstrap_difference", "stratified_risk_difference", "paired_effect", "sign_flip_test", "holm", "benjamini_hochberg",
    "brier_score", "log_score", "calibration_bins", "group_means", "clean_float",
]


# ------------------------------------------------------------------------------------------ helpers
def clean_float(x: Any, nd: int = 6) -> float | None:
    """Round to ``nd`` places; NaN/inf/None → None (JSON-safe)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _z(conf: float) -> float:
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + conf / 2.0)


def _group_index(groups: Sequence[Hashable]) -> tuple[list[Hashable], list[np.ndarray]]:
    """Unique groups (first-appearance order) and the row indices of each."""
    order: dict[Hashable, list[int]] = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    keys = list(order)
    return keys, [np.asarray(order[k], dtype=np.int64) for k in keys]


def _summary(*, n_units: int, n_obs: int, estimate: Any, ci: tuple[Any, Any] | None, method: str,
             assumptions: list[str], **extra: Any) -> dict[str, Any]:
    lo, hi = (ci if ci is not None else (None, None))
    out = {"n_units": int(n_units), "n_obs": int(n_obs), "estimate": clean_float(estimate),
           "ci_low": clean_float(lo), "ci_high": clean_float(hi), "method": method,
           "assumptions": list(assumptions), "analysis": STATS_VERSION}
    out.update(extra)
    return out


# ------------------------------------------------------------------------------------------ Wilson
def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion k/n (independent trials assumed)."""
    if n <= 0:
        return None, None
    if k < 0 or k > n:
        raise ValueError("need 0 <= k <= n")
    z = _z(conf)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------------------------------ bootstrap core
def cluster_bootstrap(groups: Sequence[Hashable], statistic: Callable[[np.ndarray], float], *, B: int = 2000,
                      seed: int = 0, conf: float = 0.95) -> dict[str, Any]:
    """Percentile cluster bootstrap of an ARBITRARY statistic.

    ``statistic(idx)`` receives an integer index array into the original observations (rows of
    whole resampled groups, concatenated; a group drawn twice appears twice) and returns a float.
    Groups are resampled with replacement, ``B`` times, with a seeded ``numpy`` Generator.
    Non-finite bootstrap statistics (e.g. a resample with no observation of some condition) are
    dropped and counted in ``n_boot_invalid``.
    """
    keys, rows = _group_index(groups)
    G = len(keys)
    n_obs = int(sum(len(r) for r in rows))
    full = np.arange(n_obs, dtype=np.int64) if n_obs else np.zeros(0, dtype=np.int64)
    est = float(statistic(np.concatenate(rows) if rows else full)) if n_obs else float("nan")
    if G < 2:
        return _summary(n_units=G, n_obs=n_obs, estimate=est, ci=None, method="cluster_bootstrap_percentile",
                        assumptions=["groups are independent"], B=B, seed=seed, conf=conf, n_boot_invalid=0,
                        note="fewer than 2 independent units: no interval")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(B, G))
    boot = np.empty(B, dtype=float)
    for b in range(B):
        idx = np.concatenate([rows[g] for g in draws[b]])
        try:
            boot[b] = float(statistic(idx))
        except (ZeroDivisionError, ValueError, FloatingPointError):
            boot[b] = float("nan")
    ok = boot[np.isfinite(boot)]
    alpha = 1.0 - conf
    ci = (float(np.quantile(ok, alpha / 2)), float(np.quantile(ok, 1 - alpha / 2))) if len(ok) >= max(20, B // 10) \
        else None
    return _summary(n_units=G, n_obs=n_obs, estimate=est, ci=ci, method="cluster_bootstrap_percentile",
                    assumptions=["independent units (groups) sampled from the task distribution",
                                 "observations within a group may be arbitrarily dependent",
                                 "percentile interval; may under-cover with few groups"],
                    B=B, seed=seed, conf=conf, n_boot_invalid=int(B - len(ok)),
                    boot_se=clean_float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
                    note=None if ci is not None else "too few valid bootstrap replicates for an interval")


def _fast_mean_boot(sums: np.ndarray, counts: np.ndarray, B: int, seed: int) -> np.ndarray:
    """Vectorised cluster bootstrap of a pooled (ratio) mean: Σ_g w_g s_g / Σ_g w_g n_g, w ~ Multinomial."""
    G = len(sums)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    num, den = w @ sums, w @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def bootstrap_mean(values: Sequence[float], groups: Sequence[Hashable] | None = None, *, B: int = 2000,
                   seed: int = 0, conf: float = 0.95, weighting: Literal["observation", "unit"] = "observation"
                   ) -> dict[str, Any]:
    """Mean of ``values`` with a cluster-bootstrap percentile CI over ``groups``.

    ``weighting="observation"``: the pooled mean over observations (ratio estimator).
    ``weighting="unit"``: the mean of per-group means (every independent unit counts once).
    ``groups=None`` treats every observation as its own unit (only valid if they are independent).
    """
    x = np.asarray(values, dtype=float)
    if groups is None:
        groups = list(range(len(x)))
    if len(groups) != len(x):
        raise ValueError("values and groups differ in length")
    keep = np.isfinite(x)
    x = x[keep]
    groups = [g for g, k in zip(groups, keep) if k]
    keys, rows = _group_index(groups)
    G = len(keys)
    if G == 0:
        return _summary(n_units=0, n_obs=0, estimate=None, ci=None, method="cluster_bootstrap_percentile",
                        assumptions=[], B=B, seed=seed, conf=conf, note="no observations")
    sums = np.array([x[r].sum() for r in rows])
    counts = np.array([len(r) for r in rows], dtype=float)
    if weighting == "unit":
        means = sums / counts
        sums, counts = means, np.ones(G)
    est = float(sums.sum() / counts.sum())
    if G < 2:
        return _summary(n_units=G, n_obs=len(x), estimate=est, ci=None, method="cluster_bootstrap_percentile",
                        assumptions=["groups are independent"], B=B, seed=seed, conf=conf, weighting=weighting,
                        note="fewer than 2 independent units: no interval")
    boot = _fast_mean_boot(sums, counts, B, seed)
    ok = boot[np.isfinite(boot)]
    alpha = 1.0 - conf
    ci = (float(np.quantile(ok, alpha / 2)), float(np.quantile(ok, 1 - alpha / 2)))
    return _summary(n_units=G, n_obs=len(x), estimate=est, ci=ci, method="cluster_bootstrap_percentile",
                    assumptions=["independent units (groups) sampled from the task distribution",
                                 "observations within a group may be arbitrarily dependent",
                                 "percentile interval; may under-cover with few groups"],
                    B=B, seed=seed, conf=conf, weighting=weighting,
                    boot_se=clean_float(np.std(ok, ddof=1)) if len(ok) > 1 else None)


def proportion_summary(successes: Sequence[bool | int | None], groups: Sequence[Hashable] | None = None, *,
                       B: int = 2000, seed: int = 0, conf: float = 0.95) -> dict[str, Any]:
    """A proportion with BOTH a Wilson interval (treats observations as independent; reported for
    reference) and, when ``groups`` is given, a cluster-bootstrap interval (the one to use).
    ``None`` entries are excluded and counted in ``n_excluded``."""
    pairs = [(bool(s), (groups[i] if groups is not None else i)) for i, s in enumerate(successes) if s is not None]
    n_excl = len(successes) - len(pairs)
    k = sum(1 for s, _ in pairs if s)
    n = len(pairs)
    w = wilson_ci(k, n, conf)
    if groups is None:
        return _summary(n_units=n, n_obs=n, estimate=(k / n if n else None), ci=w, method="wilson",
                        assumptions=["observations are independent Bernoulli trials"], k=k,
                        n_excluded=n_excl, conf=conf, wilson_low=clean_float(w[0]), wilson_high=clean_float(w[1]))
    bs = bootstrap_mean([float(s) for s, _ in pairs], [g for _, g in pairs], B=B, seed=seed, conf=conf)
    bs.update({"k": k, "n_excluded": n_excl, "wilson_low": clean_float(w[0]), "wilson_high": clean_float(w[1]),
               "wilson_assumption": "Wilson treats every observation as independent; with repeated observations "
                                    "per unit it is too narrow — use ci_low/ci_high (cluster bootstrap)."})
    return bs


def bootstrap_difference(values: Sequence[float], arm: Sequence[Hashable], groups: Sequence[Hashable], a: Hashable,
                         b: Hashable, *, B: int = 2000, seed: int = 0, conf: float = 0.95) -> dict[str, Any]:
    """Difference of pooled means mean(arm==a) − mean(arm==b), clusters (groups) resampled jointly so the
    within-unit correlation between arms is preserved."""
    x = np.asarray(values, dtype=float)
    arm_a = np.array([z == a for z in arm])
    arm_b = np.array([z == b for z in arm])
    fin = np.isfinite(x)

    def stat(idx: np.ndarray) -> float:
        xa = x[idx][arm_a[idx] & fin[idx]]
        xb = x[idx][arm_b[idx] & fin[idx]]
        if len(xa) == 0 or len(xb) == 0:
            return float("nan")
        return float(xa.mean() - xb.mean())

    out = cluster_bootstrap(groups, stat, B=B, seed=seed, conf=conf)
    out["contrast"] = f"{a} − {b}"
    out["n_obs_a"] = int((arm_a & fin).sum())
    out["n_obs_b"] = int((arm_b & fin).sum())
    return out


def stratified_risk_difference(values: Sequence[float], arm: Sequence[Hashable], strata: Sequence[Hashable],
                               groups: Sequence[Hashable], a: Hashable, b: Hashable, *, B: int = 2000, seed: int = 0,
                               conf: float = 0.95, max_dropped_fraction: float = 0.05) -> dict[str, Any]:
    """Stratified difference of proportions P(value | arm a) − P(value | arm b), computed WITHIN each stratum and
    combined with Mantel–Haenszel risk-difference weights w_s = n_a·n_b / (n_a + n_b) (trial counts).

    Only strata containing BOTH arms contribute, so a quantity that is constant within a stratum (e.g. a constant or
    transcript-copying reporter) gives exactly 0 in every stratum and overall. The interval is a percentile cluster
    bootstrap over ``groups`` of the whole stratified statistic (strata usable in a resample are recomputed in that
    resample). A resample in which no stratum has both arms is undefined and DROPPED; the number and fraction of
    dropped resamples are reported, and ``ci`` is withheld (``ci_ok=False``) when more than
    ``max_dropped_fraction`` were dropped.
    """
    x = np.asarray(values, dtype=float)
    A = np.array([z == a for z in arm])
    Bm = np.array([z == b for z in arm])
    S = list(strata)
    G = list(groups)
    if not (len(x) == len(A) == len(S) == len(G)):
        raise ValueError("values, arm, strata and groups must have equal length")
    keys = list(dict.fromkeys(S))
    s_idx = {k: np.array([i for i, z in enumerate(S) if z == k], dtype=np.int64) for k in keys}

    def combine(idx: np.ndarray) -> float:
        num = den = 0.0
        sel = np.zeros(len(x), dtype=np.int64)
        np.add.at(sel, idx, 1)  # multiplicity of each original row in the resample
        for k in keys:
            rows = s_idx[k]
            wa, wb = sel[rows] * A[rows], sel[rows] * Bm[rows]
            na, nb = float(wa.sum()), float(wb.sum())
            if na == 0 or nb == 0:
                continue
            d = float((wa * x[rows]).sum() / na - (wb * x[rows]).sum() / nb)
            w = na * nb / (na + nb)
            num += w * d
            den += w
        return num / den if den > 0 else float("nan")

    per = {}
    for k in keys:
        rows = s_idx[k]
        ra, rb = rows[A[rows]], rows[Bm[rows]]
        na, nb = len(ra), len(rb)
        pa = float(x[ra].mean()) if na else None
        pb = float(x[rb].mean()) if nb else None
        per[str(k)] = {"n_trials_a": na, "n_trials_b": nb, "n_items_a": len({G[i] for i in ra}),
                       "n_items_b": len({G[i] for i in rb}), "p_a": clean_float(pa), "p_b": clean_float(pb),
                       "difference": clean_float(pa - pb) if na and nb else None,
                       "weight": clean_float(na * nb / (na + nb)) if na and nb else None, "usable": bool(na and nb)}
    usable = [k for k in keys if per[str(k)]["usable"]]
    items_b = {G[i] for k in usable for i in s_idx[k][Bm[s_idx[k]]]}
    items_a = {G[i] for k in usable for i in s_idx[k][A[s_idx[k]]]}
    base = {"strata": per, "n_usable_strata": len(usable), "n_items_a_usable": len(items_a),
            "n_items_b_usable": len(items_b), "weights": "Mantel-Haenszel w_s = n_a*n_b/(n_a+n_b) (trials)",
            "contrast": f"{a} − {b}", "max_dropped_fraction": max_dropped_fraction}
    if not usable:
        return {**_summary(n_units=len(set(G)), n_obs=len(x), estimate=None, ci=None,
                           method="stratified_risk_difference+cluster_bootstrap",
                           assumptions=["no stratum contains both arms: not estimable"]),
                **base, "n_boot_dropped": None, "dropped_fraction": None, "ci_ok": False,
                "note": "no stratum contains both arms"}
    out = cluster_bootstrap(G, combine, B=B, seed=seed, conf=conf)
    dropped = int(out.get("n_boot_invalid") or 0)
    frac = dropped / B if B else 0.0
    ci_ok = out.get("ci_low") is not None and frac <= max_dropped_fraction
    out.update(base)
    out.update({"method": "stratified_risk_difference+cluster_bootstrap", "n_boot_dropped": dropped,
                "dropped_fraction": clean_float(frac), "ci_ok": ci_ok,
                "assumptions": ["independent units (groups)", "within-stratum comparison of the two arms",
                                "Mantel-Haenszel weights; strata lacking an arm contribute nothing"]})
    if not ci_ok:
        out["ci_withheld"] = {"ci_low": out["ci_low"], "ci_high": out["ci_high"]}
        out["ci_low"] = out["ci_high"] = None
        out["note"] = (f"{dropped}/{B} bootstrap resamples ({frac:.1%}) had no stratum with both arms "
                       f"(> {max_dropped_fraction:.0%}): interval withheld")
    return out


# ------------------------------------------------------------------------------------------ paired effects
def group_means(values: Sequence[float], groups: Sequence[Hashable]) -> tuple[list[Hashable], np.ndarray]:
    """Per-group mean of the finite values (groups with no finite value are dropped)."""
    x = np.asarray(values, dtype=float)
    keys, rows = _group_index(groups)
    out_k, out_v = [], []
    for k, r in zip(keys, rows):
        v = x[r]
        v = v[np.isfinite(v)]
        if len(v):
            out_k.append(k)
            out_v.append(float(v.mean()))
    return out_k, np.asarray(out_v, dtype=float)


def sign_flip_test(unit_effects: Sequence[float], *, alternative: Alternative = "two_sided", n_perm: int = 10000,
                   seed: int = 0, exact_max_units: int = 16) -> dict[str, Any]:
    """Clustered sign-flip permutation test of H0: unit-level effects are symmetric about 0.

    Input is ONE effect per independent unit (e.g. the mean paired difference within an item).
    Exact (all 2^G sign patterns) when G ≤ ``exact_max_units``; otherwise Monte Carlo with
    p = (1 + #{T* ≥ T}) / (1 + n_perm), which is valid (never anti-conservative).
    """
    d = np.asarray([v for v in unit_effects if v is not None and math.isfinite(float(v))], dtype=float)
    G = len(d)
    if G == 0:
        return {"p_value": None, "statistic": None, "n_units": 0, "method": "sign_flip", "alternative": alternative,
                "note": "no units"}
    t_obs = float(d.mean())

    def tail(t: np.ndarray) -> np.ndarray:
        if alternative == "greater":
            return t >= t_obs - 1e-12
        if alternative == "less":
            return t <= t_obs + 1e-12
        return np.abs(t) >= abs(t_obs) - 1e-12

    if G <= exact_max_units:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
        t = signs @ d / G
        p = float(tail(t).mean())
        method, n_used = "sign_flip_exact", int(len(t))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([1.0, -1.0]), size=(n_perm, G))
        t = signs @ d / G
        p = float((1 + tail(t).sum()) / (1 + n_perm))
        method, n_used = "sign_flip_monte_carlo", int(n_perm)
    return {"p_value": clean_float(p, 8), "statistic": clean_float(t_obs), "n_units": G, "method": method,
            "alternative": alternative, "n_permutations": n_used, "seed": seed,
            "assumptions": ["units independent", "under H0 each unit's effect is symmetric about 0"],
            "analysis": STATS_VERSION}


def paired_effect(a: Sequence[float], b: Sequence[float], groups: Sequence[Hashable], *, B: int = 2000,
                  seed: int = 0, conf: float = 0.95, alternative: Alternative = "two_sided",
                  n_perm: int = 10000) -> dict[str, Any]:
    """Paired clustered effect of a − b.

    Each position i is one PAIR (same unit, two conditions). Differences are averaged within each
    group first; the estimate is the mean of per-group mean differences (each independent unit
    weighted once). CI: bootstrap over units. Test: clustered sign-flip permutation.
    """
    a_ = np.asarray(a, dtype=float)
    b_ = np.asarray(b, dtype=float)
    if len(a_) != len(b_) or len(a_) != len(groups):
        raise ValueError("a, b and groups must have equal length")
    diff = a_ - b_
    keys, d = group_means(diff, groups)
    n_pairs = int(np.isfinite(diff).sum())
    ci_out = bootstrap_mean(d, list(range(len(d))), B=B, seed=seed, conf=conf) if len(d) else None
    test = sign_flip_test(d, alternative=alternative, n_perm=n_perm, seed=seed)
    return _summary(n_units=len(d), n_obs=n_pairs, estimate=(float(d.mean()) if len(d) else None),
                    ci=((ci_out["ci_low"], ci_out["ci_high"]) if ci_out else None),
                    method="paired_unit_mean_difference+cluster_bootstrap+sign_flip",
                    assumptions=["pairs are matched within a unit", "units independent",
                                 "estimate weights each unit equally"],
                    B=B, seed=seed, conf=conf, p_value=test["p_value"], test=test,
                    sd_unit_effect=clean_float(np.std(d, ddof=1)) if len(d) > 1 else None)


# ------------------------------------------------------------------------------------------ multiplicity
def holm(pvalues: Sequence[float | None], alpha: float = 0.05) -> dict[str, Any]:
    """Holm–Bonferroni step-down adjusted p-values (FWER). ``None`` entries are passed through."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    p = np.array([float(pvalues[i]) for i in idx])
    m = len(p)
    adj_full: list[float | None] = [None] * len(pvalues)
    if m:
        order = np.argsort(p, kind="mergesort")
        adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
        for r, o in enumerate(order):
            adj_full[idx[o]] = float(adj[r])
    return {"adjusted": [clean_float(x, 8) for x in adj_full],
            "reject": [None if x is None else bool(x <= alpha) for x in adj_full], "alpha": alpha, "method": "holm"}


def benjamini_hochberg(pvalues: Sequence[float | None], alpha: float = 0.05) -> dict[str, Any]:
    """Benjamini–Hochberg step-up adjusted p-values (q-values, FDR under independence/PRDS)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    p = np.array([float(pvalues[i]) for i in idx])
    m = len(p)
    adj_full: list[float | None] = [None] * len(pvalues)
    if m:
        order = np.argsort(p, kind="mergesort")
        ranked = p[order] * m / (np.arange(m) + 1)
        adj = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
        for r, o in enumerate(order):
            adj_full[idx[o]] = float(adj[r])
    return {"adjusted": [clean_float(x, 8) for x in adj_full],
            "reject": [None if x is None else bool(x <= alpha) for x in adj_full], "alpha": alpha,
            "method": "benjamini_hochberg"}


# ------------------------------------------------------------------------------------------ scoring rules
def _as_prob_matrix(probs: Sequence[Any], outcomes: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes)
    if P.ndim == 1:  # binary: probability of the event (outcome 1/True)
        P = np.stack([1.0 - P, P], axis=1)
        y = y.astype(int)
    else:
        y = y.astype(int)
    if len(P) != len(y):
        raise ValueError("probs and outcomes differ in length")
    return P, y


def brier_score(probs: Sequence[Any], outcomes: Sequence[Any], *, per_item: bool = False) -> Any:
    """Brier score (lower is better). Binary: mean (p − y)². Multiclass (probs [n, K], outcomes = class index):
    mean Σ_k (p_k − 1[y=k])² (range 0..2)."""
    P = np.asarray(probs, dtype=float)
    if P.ndim == 1:
        y = np.asarray(outcomes, dtype=float)
        s = (P - y) ** 2
    else:
        _, y = _as_prob_matrix(probs, outcomes)
        onehot = np.zeros_like(P)
        onehot[np.arange(len(y)), y] = 1.0
        s = ((P - onehot) ** 2).sum(axis=1)
    return s if per_item else (float(s.mean()) if len(s) else None)


def log_score(probs: Sequence[Any], outcomes: Sequence[Any], *, base: float = 2.0, eps: float = 1e-12,
              per_item: bool = False) -> Any:
    """Logarithmic score: mean log_base p(realised outcome) (≤ 0; higher is better; bits by default).
    Probabilities are clipped to [eps, 1] so a confident miss costs at most log(eps)."""
    P, y = _as_prob_matrix(probs, outcomes)
    p = np.clip(P[np.arange(len(y)), y], eps, 1.0)
    s = np.log(p) / math.log(base)
    return s if per_item else (float(s.mean()) if len(s) else None)


def calibration_bins(probs: Sequence[float], outcomes: Sequence[Any], *, n_bins: int = 10,
                     strategy: Literal["uniform", "quantile"] = "uniform") -> dict[str, Any]:
    """Binned reliability table and expected calibration error for binary forecasts.

    ECE = Σ_b (n_b / N) |mean_pred_b − frac_pos_b|; MCE = max over non-empty bins. Bin counts are reported
    because a binned ECE from a handful of forecasts is mostly noise (and is biased upward).
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    N = len(p)
    if N == 0:
        return {"ece": None, "mce": None, "bins": [], "n": 0, "n_bins": n_bins, "strategy": strategy}
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        m = which == b
        n_b = int(m.sum())
        if n_b == 0:
            bins.append({"lo": clean_float(edges[b]), "hi": clean_float(edges[b + 1]), "n": 0,
                         "mean_pred": None, "frac_pos": None})
            continue
        mp, fp = float(p[m].mean()), float(y[m].mean())
        gap = abs(mp - fp)
        ece += n_b / N * gap
        mce = max(mce, gap)
        bins.append({"lo": clean_float(edges[b]), "hi": clean_float(edges[b + 1]), "n": n_b,
                     "mean_pred": clean_float(mp), "frac_pos": clean_float(fp)})
    return {"ece": clean_float(ece), "mce": clean_float(mce), "bins": bins, "n": N, "n_bins": len(edges) - 1,
            "strategy": strategy, "note": "binned ECE is biased upward for small n; read with bin counts"}


def mean_of(xs: Iterable[float | None]) -> float | None:
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(v) / len(v)) if v else None
