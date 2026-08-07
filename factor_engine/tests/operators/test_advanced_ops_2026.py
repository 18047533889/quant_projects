# -*- coding: utf-8
"""Tests for the 2026-08 advanced operator pack (Gemini DeepResearch round).

Covers registration / surface classification, determinism, axis preservation,
prefix causality, reference math (transfer entropy, score-rank weighting,
Benford JS, Bures, Kramers-Moyal, holder-class JS, pair Wasserstein, Rips H1,
diagram shift, Student-t Fisher, quantile PCA) and NaN/empty robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_surface import classify_canonical

P1 = {
    "ts_transfer_entropy", "ts_score_rank_weighted_mean",
    "ts_bures_corr_shift", "ts_kramers_moyal_drift", "ts_kramers_moyal_diffusion",
    "cs_sliced_wasserstein_copula_shift", "group_spd_feature_structure_shift",
    "holder_class_js_shift", "intraday_wasserstein_pair_distance",
    "intraday_barrier_approach_acceleration",
}
P2 = {
    "ts_effective_transfer_entropy", "report_benford_js_divergence",
    "intraday_wasserstein_quantile_pca_score",
    "intraday_wasserstein_quantile_pca_residual",
    "ts_betti_1_max_persistence", "ts_persistence_diagram_shift",
    "ts_student_t_fisher_shift",
}
ALL_OPS = P1 | P2


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=values.shape[0], freq="B")
    return pd.DataFrame(values, index=idx, columns=[f"S{i}" for i in range(values.shape[1])])


def _minute_frame(days: int = 2, cols: int = 3, per_day: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per_day)]
    )
    rng = np.random.default_rng(11)
    base = 1.0 + 0.001 * np.arange(days * per_day)[:, None]
    return pd.DataFrame(base + rng.normal(0.0, 0.001, size=(days * per_day, cols)),
                        index=idx, columns=[f"S{i}" for i in range(cols)])


# ---------------------------------------------------------------------------
# registration & classification
# ---------------------------------------------------------------------------

def test_all_advanced_ops_registered_and_classified():
    canon = set(OperatorRegistry.list_canonical())
    assert ALL_OPS <= canon, ALL_OPS - canon
    for name in P1:
        assert classify_canonical(name) == "extended", name
    for name in P2:
        assert classify_canonical(name) == "research", name


def test_advanced_ops_deterministic_and_shape_preserving():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    rng = np.random.default_rng(0)
    x = _frame(rng.normal(0.0, 1.0, size=(80, 4)), "2024-01-01")
    y = _frame(rng.normal(0.0, 1.0, size=(80, 4)), "2024-01-01")
    g = _frame(np.tile(np.arange(4) % 2, (80, 1)), "2024-01-01")
    calls = {
        "ts_transfer_entropy": (x, y),
        "ts_effective_transfer_entropy": (x, y),
        "ts_score_rank_weighted_mean": (x, y),
        "report_benford_js_divergence": (np.abs(x) * 1000.0 + 1.0,),
        "ts_bures_corr_shift": (x, y),
        "ts_kramers_moyal_drift": (x,),
        "ts_kramers_moyal_diffusion": (x,),
        "cs_sliced_wasserstein_copula_shift": (x, y, np.abs(x)),
        "group_spd_feature_structure_shift": (x, y, np.abs(x), g),
        "ts_betti_1_max_persistence": (x,),
        "ts_persistence_diagram_shift": (x,),
        "ts_student_t_fisher_shift": (x,),
    }
    kwargs = {
        "ts_transfer_entropy": {"window": 40, "bins": 3, "lag": 1},
        "ts_effective_transfer_entropy": {"window": 40, "bins": 3, "lag": 1},
        "ts_score_rank_weighted_mean": {"window": 40, "decay": 0.85},
        "report_benford_js_divergence": {"window": 40},
        "ts_bures_corr_shift": {"recent_window": 10, "prior_window": 30},
        "ts_kramers_moyal_drift": {"window": 40, "bins": 5},
        "ts_kramers_moyal_diffusion": {"window": 40, "bins": 5},
        "cs_sliced_wasserstein_copula_shift": {"window": 30, "directions": 16},
        "group_spd_feature_structure_shift": {"reference_window": 30},
        "ts_betti_1_max_persistence": {"window": 40, "tau": 1, "embedding_dim": 3},
        "ts_persistence_diagram_shift": {"window": 20, "tau": 1, "embedding_dim": 3},
        "ts_student_t_fisher_shift": {"recent_window": 30, "prior_window": 30},
    }
    for name, args in calls.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        a = op.calculate(*args, **kwargs[name])
        b = op.calculate(*args, **kwargs[name])
        assert a.index.equals(args[0].index), name
        assert a.columns.equals(args[0].columns), name
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), name


def test_advanced_ops_prefix_causal():
    rng = np.random.default_rng(1)
    x = _frame(rng.normal(0.0, 1.0, size=(60, 3)), "2024-01-01")
    y = _frame(rng.normal(0.0, 1.0, size=(60, 3)), "2024-01-01")
    checks = {
        "ts_transfer_entropy": (x, y),
        "ts_score_rank_weighted_mean": (x, y),
        "ts_kramers_moyal_drift": (x,),
        "ts_bures_corr_shift": (x, y),
    }
    kwargs = {
        "ts_transfer_entropy": {"window": 30, "bins": 3, "lag": 1},
        "ts_score_rank_weighted_mean": {"window": 30},
        "ts_kramers_moyal_drift": {"window": 30, "bins": 5},
        "ts_bures_corr_shift": {"recent_window": 8, "prior_window": 20},
    }
    for name, args in checks.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        full = op.calculate(*args, **kwargs[name])
        sliced_args = tuple(a.iloc[:40] for a in args)
        prefix = op.calculate(*sliced_args, **kwargs[name])
        full_prefix = full.iloc[:40]
        assert np.allclose(full_prefix.to_numpy(dtype=float), prefix.to_numpy(dtype=float), equal_nan=True), name


# ---------------------------------------------------------------------------
# reference math
# ---------------------------------------------------------------------------

def test_transfer_entropy_independent_is_small():
    rng = np.random.default_rng(3)
    t = _frame(rng.normal(0.0, 1.0, size=(300, 3)), "2024-01-01")
    s = _frame(rng.normal(0.0, 1.0, size=(300, 3)), "2024-01-01")
    out = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy").calculate(t, s, window=120, bins=3, lag=1)
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    # Finite-sample quantile-binning bias (Jeffreys smoothing) keeps TE small
    # but nonzero for independent series; ~0.08 here, bound at 0.12.
    assert float(np.nanmean(vals)) < 0.12


def test_effective_transfer_entropy_finite_deterministic():
    rng = np.random.default_rng(4)
    t = _frame(rng.normal(0.0, 1.0, size=(150, 2)), "2024-01-01")
    s = _frame(rng.normal(0.0, 1.0, size=(150, 2)), "2024-01-01")
    op = OperatorRegistry.get("ts_effective_transfer_entropy", "pandas_numpy")
    a = op.calculate(t, s, window=100, bins=3, lag=1)
    b = op.calculate(t, s, window=100, bins=3, lag=1)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_score_rank_weighted_mean_reference():
    idx = pd.date_range("2024-01-01", periods=3)
    target = pd.DataFrame([[1.0], [5.0], [10.0]], index=idx, columns=["A"])
    score = pd.DataFrame([[3.0], [2.0], [1.0]], index=idx, columns=["A"])
    out = OperatorRegistry.get("ts_score_rank_weighted_mean", "pandas_numpy").calculate(target, score, window=3, decay=0.5)
    # Sort by score desc: targets (1,5,10); weights 1, .5, .25 -> sum 1.75.
    expected = (1.0 + 0.5 * 5.0 + 0.25 * 10.0) / 1.75
    assert out["A"].iloc[-1] == pytest.approx(expected)


def test_benford_js_small_for_benford_like():
    # Amounts whose first digits follow Benford closely.
    digits = np.arange(1, 10, dtype=float)
    benford = np.log10(1.0 + 1.0 / digits)
    n = 3000
    rng = np.random.default_rng(5)
    d = rng.choice(np.arange(1, 10), size=n, p=benford / benford.sum())
    magnitudes = 10.0 ** rng.uniform(0.0, 4.0, size=n)
    amounts = d * magnitudes
    amt = _frame(amounts.reshape(-1, 1), "2024-01-01")
    out = OperatorRegistry.get("report_benford_js_divergence", "pandas_numpy").calculate(amt, window=2000)
    last = out.to_numpy(dtype=float)[-1, 0]
    assert np.isfinite(last)
    assert last < 0.1  # close to Benford -> JS distance small


def test_bures_shift_constant_corr_is_zero():
    rng = np.random.default_rng(6)
    base = rng.normal(0.0, 1.0, size=(120, 2))
    x = _frame(base, "2024-01-01")
    y = _frame(2.0 * base + 0.0 * rng.normal(size=(120, 2)), "2024-01-01")
    out = OperatorRegistry.get("ts_bures_corr_shift", "pandas_numpy").calculate(x, y, recent_window=20, prior_window=40)
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert float(np.nanmax(vals)) < 1e-6  # same correlation in both windows


def test_bures_shift_corr_flip_is_positive():
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1.0, size=(120, 2))
    z = np.tile(z, (1, 3))  # widen columns
    x = _frame(z, "2024-01-01")
    # y = -x in the prior window (rows < 100), +x in the recent window.
    yv = z.copy()
    yv[:100] = -z[:100]
    yv[100:] = +z[100:]
    y = _frame(yv, "2024-01-01")
    out = OperatorRegistry.get("ts_bures_corr_shift", "pandas_numpy").calculate(x, y, recent_window=20, prior_window=40)
    last = out.to_numpy(dtype=float)[-1, 0]
    assert np.isfinite(last)
    assert last > 0.1  # flipped correlation -> large Bures distance


def test_kramers_moyal_linear_trend():
    t = np.arange(80.0)[:, None]
    x = _frame(t, "2024-01-01")
    drift = OperatorRegistry.get("ts_kramers_moyal_drift", "pandas_numpy").calculate(x, window=60, bins=5)
    diffusion = OperatorRegistry.get("ts_kramers_moyal_diffusion", "pandas_numpy").calculate(x, window=60, bins=5)
    dvals = drift.to_numpy(dtype=float)[-20:]
    qvals = diffusion.to_numpy(dtype=float)[-20:]
    dvals = dvals[np.isfinite(dvals)]
    qvals = qvals[np.isfinite(qvals)]
    assert dvals.size > 0 and qvals.size > 0
    assert np.allclose(dvals, 1.0, atol=1e-6)   # dx = 1
    # Strict Kramers-Moyal D2 at dt=1 is (1/2)*E[dx^2]; a linear trend dx=1 -> 0.5.
    assert np.allclose(qvals, 0.5, atol=1e-6)


def _holder_panel(values: np.ndarray) -> list[pd.DataFrame]:
    """Split a (rows, 5) slot matrix into five 1-column panels."""
    idx = pd.date_range("2024-01-01", periods=values.shape[0])
    return [
        pd.DataFrame(values[:, k:k + 1], index=idx, columns=["S0"])
        for k in range(5)
    ]


def test_holder_class_js_shift_disjoint_is_one():
    cur = _holder_panel(np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]]))
    prev = _holder_panel(np.array([[0.0, 1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]]))
    out = OperatorRegistry.get("holder_class_js_shift", "pandas_numpy").calculate(*cur, *prev)
    assert out["S0"].iloc[0] == pytest.approx(1.0)  # completely disjoint -> JS = 1
    assert out["S0"].iloc[1] == pytest.approx(0.0, abs=1e-9)  # identical -> 0


def test_pair_wasserstein_identical_and_different():
    rng = np.random.default_rng(9)
    m1 = _minute_frame(2, 3)
    out = OperatorRegistry.get("intraday_wasserstein_pair_distance", "pandas_numpy").calculate(m1, m1)
    assert out.shape[0] == 2  # one row per day
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert np.allclose(vals, 0.0, atol=1e-9)  # identical distributions
    m2 = m1 * 1.5
    out2 = OperatorRegistry.get("intraday_wasserstein_pair_distance", "pandas_numpy").calculate(m1, m2)
    v2 = out2.to_numpy(dtype=float)
    v2 = v2[np.isfinite(v2)]
    assert v2.size > 0
    assert float(np.nanmean(v2)) > 0.0


# ---------------------------------------------------------------------------
# topology / information geometry
# ---------------------------------------------------------------------------

def _brute_force_h1(points: np.ndarray) -> list[tuple[float, float]]:
    """Textbook F2 boundary-matrix reduction (validation reference)."""
    n = points.shape[0]
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    simplices = []
    for i in range(n):
        simplices.append((0.0, 0, (i,)))
    for i in range(n):
        for j in range(i + 1, n):
            simplices.append((float(d[i, j]), 1, (i, j)))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                simplices.append((float(max(d[i, j], d[i, k], d[j, k])), 2, (i, j, k)))
    simplices.sort(key=lambda s: (s[0], s[1]))
    idx = {s[2]: m for m, s in enumerate(simplices)}

    def boundary(members):
        mm = list(members)
        return [idx[tuple(mm[:drop] + mm[drop + 1:])] for drop in range(len(mm))]

    cols = {s_i: boundary(s[2]) for s_i, s in enumerate(simplices) if s[1] > 0}
    low: dict[int, int] = {}
    for c in sorted(cols):
        while True:
            col = sorted(cols[c])
            if not col:
                break
            pivot = col[0]
            if pivot in low:
                other = set(cols[low[pivot]])
                cur = set(cols[c])
                cols[c] = sorted(cur ^ other)
            else:
                low[pivot] = c
                break
    pairs = []
    for c in sorted(cols):
        if not cols[c]:
            continue
        pivot = sorted(cols[c])[0]
        if simplices[c][1] == 2 and simplices[pivot][1] == 1:
            pairs.append((simplices[pivot][0], simplices[c][0]))
    return pairs


def test_rips_h1_matches_bruteforce():
    from cleaned_operators.advanced_topology import _max_persistence, _rips_h1_pairs

    rng = np.random.default_rng(42)
    for _ in range(4):
        pts = rng.normal(size=(8, 3))
        fast = _max_persistence(_rips_h1_pairs(pts))
        slow = _max_persistence(_brute_force_h1(pts))
        assert fast == pytest.approx(slow, abs=1e-9)


def test_betti_periodic_positive_persistence():
    from cleaned_operators.advanced_topology import _max_persistence, _rips_h1_pairs

    t = np.arange(60.0)
    s = np.sin(2.0 * np.pi * t / 6.0)
    pts = np.column_stack([s, s, s])  # a loop in 3D
    assert _max_persistence(_rips_h1_pairs(pts)) > 0.05


def test_betti_and_diagram_shift_panels():
    rng = np.random.default_rng(12)
    x = _frame(rng.normal(0.0, 1.0, size=(120, 3)), "2024-01-01")
    op = OperatorRegistry.get("ts_betti_1_max_persistence", "pandas_numpy")
    out = op.calculate(x, window=60, tau=1, embedding_dim=3)
    assert out.shape == x.shape
    ds = OperatorRegistry.get("ts_persistence_diagram_shift", "pandas_numpy")
    out2 = ds.calculate(x, window=30, tau=1, embedding_dim=3)
    assert out2.shape == x.shape
    assert np.isfinite(out2.to_numpy(dtype=float)).any()


def test_student_t_fisher_shift_same_vs_different():
    rng = np.random.default_rng(13)
    op = OperatorRegistry.get("ts_student_t_fisher_shift", "pandas_numpy")
    # Same heavy-tailed distribution in both windows -> small information distance.
    same = _frame(rng.standard_t(df=5.0, size=(200, 3)), "2024-01-01")
    out_same = op.calculate(same, recent_window=60, prior_window=80)
    vals = out_same.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    same_med = float(np.nanmedian(vals))
    # Gaussian vs heavy-tailed -> the empirical Fisher (mu, log-sigma, log(nu-2))
    # geometry differs more than same-family estimation noise.
    diff_data = np.concatenate([rng.normal(size=(120, 3)),
                                rng.standard_t(df=3.0, size=(80, 3))], axis=0)
    out_diff = op.calculate(_frame(diff_data, "2024-01-01"), recent_window=60, prior_window=80)
    dvals = out_diff.to_numpy(dtype=float)
    dvals = dvals[np.isfinite(dvals)]
    assert dvals.size > 0
    assert float(np.nanmedian(dvals)) > same_med + 0.1


def test_quantile_pca_constant_returns_near_zero():
    const = _minute_frame(6, 2)
    const.loc[:, :] = 0.05  # constant minute return
    score = OperatorRegistry.get("intraday_wasserstein_quantile_pca_score", "pandas_numpy")
    resid = OperatorRegistry.get("intraday_wasserstein_quantile_pca_residual", "pandas_numpy")
    s_out = score.calculate(const, window=3, k=1)
    r_out = resid.calculate(const, window=3, k=1)
    svals = s_out.to_numpy(dtype=float)
    rvals = r_out.to_numpy(dtype=float)
    svals = svals[np.isfinite(svals)]
    rvals = rvals[np.isfinite(rvals)]
    assert svals.size > 0 and rvals.size > 0
    assert np.allclose(svals, 0.0, atol=1e-8)
    assert np.allclose(rvals, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# NaN / empty robustness
# ---------------------------------------------------------------------------

def test_advanced_ops_all_nan_input_is_nan():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    nan_panel = pd.DataFrame(np.full((40, 3), np.nan), index=idx, columns=["S0", "S1", "S2"])
    op = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy")
    out = op.calculate(nan_panel, nan_panel, window=30, bins=3, lag=1)
    assert out.to_numpy(dtype=float).size > 0
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_holder_class_js_shift_missing_previous_is_nan():
    cur = _holder_panel(np.array([[0.5, 0.5, np.nan, np.nan, np.nan]]))
    prev = _holder_panel(np.full((1, 5), np.nan))
    out = OperatorRegistry.get("holder_class_js_shift", "pandas_numpy").calculate(*cur, *prev)
    assert np.isnan(out.to_numpy(dtype=float)).all()
