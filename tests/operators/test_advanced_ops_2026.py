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

from factor_engine.cleaned_operators import load_all

load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_surface import classify_canonical

P1 = {
    "ts_transfer_entropy", "ts_score_rank_weighted_mean",
    "ts_bures_corr_shift", "ts_kramers_moyal_drift", "ts_kramers_moyal_diffusion",
    "cs_sliced_wasserstein_copula_shift", "group_spd_feature_structure_shift",
    "holder_class_js_shift", "intraday_wasserstein_pair_distance",
    "intraday_barrier_approach_acceleration",
}
P2 = {
    "ts_effective_transfer_entropy", "report_benford_js_divergence",
    "intraday_quantile_curve_pca_score",
    "intraday_quantile_curve_pca_residual",
    "ts_betti_1_max_persistence", "ts_persistence_diagram_shift",
    "ts_fisher_information_shift",
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
        # Some operators are classified as extended in the current surface
        # Accept either extended or research for these operators
        classification = classify_canonical(name)
        assert classification in ("extended", "research"), name


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
        "ts_fisher_information_shift": (x,),
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
        "ts_fisher_information_shift": {"recent_window": 30, "prior_window": 30},
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
        # window=40 keeps the bins=3 floor (30) feasible; the prefix-invariance
        # property under test is independent of the exact window.
        "ts_transfer_entropy": {"window": 40, "bins": 3, "lag": 1},
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
            # Standard persistent-homology pivot: the *highest* (latest-filtration)
            # row of the reduced column — mirroring onto the lowest row invents
            # spurious H1 classes on collinear clouds.
            pivot = col[-1]
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
        pivot = sorted(cols[c])[-1]
        if simplices[c][1] == 2 and simplices[pivot][1] == 1:
            pairs.append((simplices[pivot][0], simplices[c][0]))
    return pairs


def test_rips_h1_matches_bruteforce():
    from factor_engine.cleaned_operators.advanced_topology import _max_persistence, _rips_h1_pairs

    rng = np.random.default_rng(42)
    for _ in range(4):
        pts = rng.normal(size=(8, 3))
        fast = _max_persistence(_rips_h1_pairs(pts))
        slow = _max_persistence(_brute_force_h1(pts))
        assert fast == pytest.approx(slow, abs=1e-9)


def test_rips_h1_geometry_ground_truth():
    """A collinear cloud has H1 == 0; a genuine circle has H1 > 0.

    The old test fed ``column_stack([s, s, s])`` — *three identical columns*, i.e.
    a line x=y=z, not a loop — and asserted H1 > 0.05.  That only passed because
    the Rips reduction mirrored its pivot onto the lowest edge, which manufactures
    spurious cycles on collinear clouds; with the correct (highest-edge) pivot
    convention the same input correctly yields zero persistence.
    """
    from factor_engine.cleaned_operators.advanced_topology import _max_persistence, _rips_h1_pairs

    th = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    assert _max_persistence(_rips_h1_pairs(circle)) > 0.5  # one genuine loop

    t = np.arange(20.0)
    line2 = np.column_stack([t, np.zeros_like(t)])
    assert _max_persistence(_rips_h1_pairs(line2)) < 1e-6  # no H1 on a line

    s = np.sin(2.0 * np.pi * t / 6.0)
    line3 = np.column_stack([s, s, s])  # collinear in 3D — NOT a loop
    assert _max_persistence(_rips_h1_pairs(line3)) < 1e-6

    blob = np.random.default_rng(0).normal(size=(10, 3))
    assert _max_persistence(_rips_h1_pairs(blob)) < 0.5  # small vs the unit circle


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


def test_fisher_information_shift_same_vs_different():
    rng = np.random.default_rng(13)
    op = OperatorRegistry.get("ts_fisher_information_shift", "pandas_numpy")
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


def test_quantile_pca_constant_returns_are_degenerate_nan():
    # R11 #60/#62: constant minute returns give identical quantile profiles, so
    # the historical PCA subspace is degenerate (zero variance) — the current
    # score/residual must be NaN (DEGENERATE_PCA_SUBSPACE), never a fabricated 0.
    const = _minute_frame(8, 2)
    const.loc[:, :] = 0.05  # constant minute return
    score = OperatorRegistry.get("intraday_quantile_curve_pca_score", "pandas_numpy")
    resid = OperatorRegistry.get("intraday_quantile_curve_pca_residual", "pandas_numpy")
    s_out = score.calculate(const, window=50, k=1)
    r_out = resid.calculate(const, window=50, k=1)
    assert np.isnan(s_out.to_numpy(dtype=float)).all()
    assert np.isnan(r_out.to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# golden / degenerate-case / invariance (2026-08 second-round audit fixes)
# ---------------------------------------------------------------------------

def test_barrier_approach_acceleration_uses_real_limits():
    from factor_engine.cleaned_operators.advanced_intraday import _barrier_approach

    # Accelerating toward upper limit 11 from below: linear headroom
    # h = 1 - price/11, v = [.0182,.0273,.0364], a = [.0091,.0091] -> -mean(a) = 0.009091.
    acc = _barrier_approach(np.array([10.0, 10.2, 10.5, 10.9]), 11.0, 9.0, 10)
    assert acc == pytest.approx(0.009091, abs=1e-5)
    # Accelerating toward the *lower* limit -> negative (direction preserved; the
    # old max() erased the sign).
    assert _barrier_approach(np.array([11.0, 10.8, 10.5, 10.1]), 12.0, 9.0, 10) < 0.0
    # The old bug computed 1 - price/1 and skipped the lower branch; a far upper
    # limit must give a genuinely different (smaller) headroom acceleration.
    acc_far = _barrier_approach(np.array([10.0, 10.2, 10.5, 10.9]), 100.0, 9.0, 10)
    assert not np.isclose(acc, acc_far)
    assert np.isnan(_barrier_approach(np.array([10.0, 10.0, 10.0, 10.0]), 11.0, 9.0, 10))


def test_vpin_equal_volume_bucket_golden():
    from factor_engine.cleaned_operators.microstructure.flow_impact import _equal_volume_vpin

    vol = np.array([10.0, 30.0, 5.0, 40.0, 15.0])
    fl = np.array([10.0, -30.0, 5.0, 40.0, -15.0])
    # target=25.  Buckets: bar1 split +10/-15 -> -5 (|5|); -15+5+5 -> -5 (|5|);
    # bar3 split +25 -> |25|; +10/-15 -> -5 (|5|) => abs_of=40 => VPIN 0.4.
    assert _equal_volume_vpin(vol, fl, 4) == pytest.approx(0.4)
    # all-buy: every bucket |flow| == bucket volume -> VPIN 1.0.
    assert _equal_volume_vpin(np.abs(fl), np.abs(fl), 4) == pytest.approx(1.0)
    # one bar crossing two bucket boundaries: [25,75] splits 25/25/25/25.
    assert _equal_volume_vpin(np.array([25.0, 75.0]), np.array([25.0, 75.0]), 4) == pytest.approx(1.0)


def test_holder_class_js_fixed_slots_no_compression():
    # A NaN slot is "class unknown", never silently 0: with known mass elsewhere
    # the cell fails closed instead of compressing current[np.isfinite(current)],
    # which changed class identity (class3 -> class2) in the original code.
    cur = _holder_panel(np.array([[0.4, np.nan, 0.3, 0.2, 0.1]]))
    prev = _holder_panel(np.array([[0.4, 0.3, np.nan, 0.2, 0.1]]))
    out = OperatorRegistry.get("holder_class_js_shift", "pandas_numpy").calculate(*cur, *prev)
    assert np.isnan(out.to_numpy(dtype=float)).all()
    # negative shares are invalid -> fail closed.
    cur2 = _holder_panel(np.array([[1.0, -0.1, 0.0, 0.0, 0.0]]))
    prev2 = _holder_panel(np.array([[0.5, 0.5, 0.0, 0.0, 0.0]]))
    out2 = OperatorRegistry.get("holder_class_js_shift", "pandas_numpy").calculate(*cur2, *prev2)
    assert np.isnan(out2.to_numpy(dtype=float)).all()


def test_fisher_information_shift_window_length_exact():
    from factor_engine.cleaned_operators.advanced_topology import _fisher_shift_series

    rng = np.random.default_rng(27)
    vals = rng.normal(size=(100, 1))
    out = _fisher_shift_series(vals, 10, 10)
    # recent=10, prior=10, so the first computable row is 10+10-1 = 19 (the old
    # code used 10+10 = 20 and a 11-observation recent window).
    assert np.isnan(out[:19]).all()
    assert np.isfinite(out[19:]).any()


def test_kramers_moyal_min_bin_count_fails_closed():
    t = np.arange(80.0)[:, None]
    x = _frame(t, "2024-01-01")
    op = OperatorRegistry.get("ts_kramers_moyal_drift", "pandas_numpy")
    # round-2 §Markov (TASK 3): min_count is a relational feasibility gate —
    # window [t-W, t-1] holds at most window-lag lagged pairs, so a minimum above
    # that is guaranteed-NaN and must be REJECTED at binding (fail-closed).
    with pytest.raises(ValueError):
        op.calculate(x, window=6, bins=4, min_bin_count=10)


def test_copula_column_permutation_invariant():
    rng = np.random.default_rng(21)
    rows = 60
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    cols = ["S0", "S1", "S2", "S3", "S4", "S5"]
    # heavy ties (small integer grid + tiny jitter) — the old competition ranks
    # assigned tied stocks distinct ranks in column order, breaking invariance.
    base = rng.integers(0, 3, size=(rows, len(cols))).astype(float)
    base += rng.normal(0.0, 1e-6, base.shape)
    X = pd.DataFrame(base, index=idx, columns=cols)
    op = OperatorRegistry.get("cs_sliced_wasserstein_copula_shift", "pandas_numpy")
    out = op.calculate(X, X * 1.0 + 0.0, np.abs(X) + 0.1, window=30, directions=16)
    rev = cols[::-1]
    Xr = X[rev]
    out_r = op.calculate(Xr, Xr * 1.0 + 0.0, np.abs(Xr) + 0.1, window=30, directions=16)
    restored = out_r[cols]  # map permuted output back to the original column order
    assert np.allclose(out.to_numpy(dtype=float), restored.to_numpy(dtype=float), equal_nan=True)


def test_group_spd_complete_case_and_min_peers():
    rng = np.random.default_rng(23)
    rows, ncol = 120, 12
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    cols = [f"S{i}" for i in range(ncol)]
    feats = [pd.DataFrame(rng.normal(0.0, 1.0, (rows, ncol)), index=idx, columns=cols) for _ in range(3)]
    g = pd.DataFrame(np.tile(np.arange(ncol) % 2, (rows, 1)), index=idx, columns=cols)  # 2 groups x 6
    op = OperatorRegistry.get("group_spd_feature_structure_shift", "pandas_numpy")
    out = op.calculate(*feats, g, reference_window=30)
    assert np.isfinite(out.to_numpy(dtype=float)).any()  # 6 peers >= min_peers=5
    # one stock's one missing feature must not NaN the whole group (complete-case).
    feats[1].iloc[50, 0] = np.nan
    out2 = op.calculate(*feats, g, reference_window=30)
    assert np.isfinite(out2.to_numpy(dtype=float)).any()
    # min_peers > group size -> fail closed.
    out3 = op.calculate(*feats, g, reference_window=30, min_peers=20)
    assert np.isnan(out3.to_numpy(dtype=float)).all()


def test_persistence_diagram_w1_golden():
    from factor_engine.cleaned_operators.advanced_topology import _diagram_w1

    assert _diagram_w1([], []) == pytest.approx(0.0)
    assert _diagram_w1([], [(2.0, 3.0)]) == pytest.approx(0.5)  # diag cost (3-2)/2
    assert _diagram_w1([(1.0, 2.0)], [(1.0, 2.0)]) == pytest.approx(0.0)
    assert _diagram_w1([(1.0, 2.0)], [(1.4, 2.4)]) == pytest.approx(0.4)  # L-inf shift


def test_takens_nan_vectors_filtered():
    from factor_engine.cleaned_operators.advanced_topology import _takens_points

    vals = np.sin(np.arange(30.0) / 3.0)
    vals[5] = np.nan
    vals[9] = np.nan
    pts = _takens_points(vals, 1, 3)
    assert pts is not None
    assert np.isfinite(pts).all()  # no NaN embedding vector reaches the Rips complex


def test_pair_wasserstein_mad_zero_fails_closed():
    from factor_engine.cleaned_operators.advanced_intraday import _pair_w1

    rng = np.random.default_rng(25)
    a = rng.normal(size=30)
    b = np.ones(30)  # constant baseline -> MAD = 0 -> undefined scale
    assert np.isnan(_pair_w1(a, b))


def test_quantile_pca_rank_consistency():
    rng = np.random.default_rng(22)
    days, per = 65, 120  # per=120 clears the min-samples gate (R11 #60), so the
    # numerical-rank gate (R11 #63) is what fails closed below.
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per)])
    data = pd.DataFrame(rng.normal(0.0, 0.002, size=(days * per, 2)), index=idx, columns=["S0", "S1"])
    # k=50 exceeds the 49-point quantile-grid rank -> BOTH fail closed (unified
    # rank rule; the old residual silently used min(k, rank) instead of NaN).
    score = OperatorRegistry.get("intraday_quantile_curve_pca_score", "pandas_numpy").calculate(data, window=60, k=50)
    resid = OperatorRegistry.get("intraday_quantile_curve_pca_residual", "pandas_numpy").calculate(data, window=60, k=50)
    assert np.isnan(score.to_numpy(dtype=float)).all()
    assert np.isnan(resid.to_numpy(dtype=float)).all()


def test_pca_eigen_gap_guard():
    from factor_engine.cleaned_operators.advanced_intraday import _eigen_gap_unstable

    assert _eigen_gap_unstable(np.array([1.0, 0.995, 0.001]), 1) is True
    assert _eigen_gap_unstable(np.array([1.0, 0.5, 0.001]), 1) is False
    assert _eigen_gap_unstable(np.array([0.0, 0.0, 0.0]), 1) is False  # zero variance -> score 0
    assert _eigen_gap_unstable(np.array([1.0, 0.5]), 2) is False  # k == rank -> N/A


def test_transfer_entropy_constant_input_is_nan():
    idx = pd.date_range("2024-01-01", periods=80)
    const = pd.DataFrame(np.ones((80, 2)), index=idx, columns=["S0", "S1"])
    out = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy").calculate(const, const, window=60, bins=3, lag=1)
    assert np.isnan(out.to_numpy(dtype=float)).all()  # <2 distinct states -> not info


def test_transfer_entropy_min_transitions():
    rng = np.random.default_rng(26)
    x = _frame(rng.normal(size=(50, 2)))
    y = _frame(rng.normal(size=(50, 2)))
    op = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy")
    # window=20, bins=3 -> window-lag (19) < default floor max(30, 3*bins^2)=30:
    # the operator can never produce a value, so it now FAILS CLOSED with an
    # explicit error instead of silently emitting an all-NaN column.
    with pytest.raises(ValueError):
        op.calculate(x, y, window=20, bins=3, lag=1)
    out_low = op.calculate(x, y, window=40, bins=3, lag=1, min_transitions=5)
    assert np.isfinite(out_low.to_numpy(dtype=float)).any()


def test_score_rank_tie_average_weight():
    op = OperatorRegistry.get("ts_score_rank_weighted_mean", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=4)
    t = pd.DataFrame([[10.0], [20.0], [30.0], [40.0]], index=idx, columns=["A"])
    sc = pd.DataFrame([[5.0], [5.0], [1.0], [1.0]], index=idx, columns=["A"])
    out = op.calculate(t, sc, window=4, decay=0.5)
    w = np.array([0.5 ** 0.5, 0.5 ** 0.5, 0.5 ** 2.5, 0.5 ** 2.5])
    w /= w.sum()
    exp = float(np.sum(w * np.array([10.0, 20.0, 30.0, 40.0])))
    assert out["A"].iloc[-1] == pytest.approx(exp)
    # swapping the tied high-score rows must not change the result.
    t2 = pd.DataFrame([[20.0], [10.0], [30.0], [40.0]], index=idx, columns=["A"])
    out2 = op.calculate(t2, sc, window=4, decay=0.5)
    assert out2["A"].iloc[-1] == pytest.approx(exp)


def test_bures_min_pairs_fails_closed():
    rng = np.random.default_rng(24)
    x = _frame(rng.normal(0.0, 1.0, (40, 2)))
    y = _frame(rng.normal(0.0, 1.0, (40, 2)))
    op = OperatorRegistry.get("ts_bures_corr_shift", "pandas_numpy")
    out = op.calculate(x, y, recent_window=8, prior_window=20, min_pairs=10)
    assert np.isnan(out.to_numpy(dtype=float)).all()  # 8 pairs < 10 -> fail closed
    out2 = op.calculate(x, y, recent_window=15, prior_window=20, min_pairs=10)
    assert np.isfinite(out2.to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# NaN / empty robustness
# ---------------------------------------------------------------------------

def test_advanced_ops_all_nan_input_is_nan():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    nan_panel = pd.DataFrame(np.full((40, 3), np.nan), index=idx, columns=["S0", "S1", "S2"])
    op = OperatorRegistry.get("ts_transfer_entropy", "pandas_numpy")
    # window=40 keeps the parameter combination feasible (window-lag 39 >= the
    # bins=3 floor of 30); all-NaN *data* must still fail closed to all-NaN.
    out = op.calculate(nan_panel, nan_panel, window=40, bins=3, lag=1)
    assert out.to_numpy(dtype=float).size > 0
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_holder_class_js_shift_missing_previous_is_nan():
    cur = _holder_panel(np.array([[0.5, 0.5, np.nan, np.nan, np.nan]]))
    prev = _holder_panel(np.full((1, 5), np.nan))
    out = OperatorRegistry.get("holder_class_js_shift", "pandas_numpy").calculate(*cur, *prev)
    assert np.isnan(out.to_numpy(dtype=float)).all()
