# -*- coding: utf-8 -*-
"""R11 round-3 file-disjoint audit — level / matrix-profile / RQA fixes.

Covers the five items owned by this batch (structural_levels, candle_state_space,
rqa_ext, recurrence_analysis):

* #14  structural pivot levels must not be forward-filled forever: ``max_pivot_age``
       caps the usable age of a confirmed pivot; beyond it the level -> NaN.
* #63  matrix-profile motif age is START-to-START, not end-to-start (the old
       ``r - best_s`` was systematically L-1 bars too old).
* #64  a constant subsequence z-normalises to all zeros -> shape collision;
       constant subsequences now emit NaN (shape-motif semantics).
* #65  Mahalanobis must not mix median-centering with ordinary covariance; the
       kernel now uses the consistent classical mean + covariance pair.
* #66  Mahalanobis / KNN high-dimensional sample floor: N >= 5p (after dropping
       constant dimensions) or NaN.
* #67  RQA DET/LAM (and the other line-structure statistics) require an effective
       length >= 0.8*window; a short-sample recompute after a gap is NaN.
* #68  ``ts_rqa_determinism_fixed_rr`` / ``ts_rqa_laminarity_fixed_rr``: epsilon
       is calibrated to a target recurrence rate FIRST, then DET/LAM measure line
       topology at that controlled density.

The modules are imported directly (which registers both backends) instead of a
full ``load_all()`` so the focused suite is independent of the global surface
finalize pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Direct import registers every owned operator (pandas_numpy + polars bridges).
import cleaned_operators.structural_levels  # noqa: F401
import cleaned_operators.candle_state_space  # noqa: F401
import cleaned_operators.rqa_ext  # noqa: F401
import cleaned_operators.recurrence_analysis  # noqa: F401

from cleaned_operators.candle_state_space import (
    _local_density_series,
    _mahalanobis_series,
    _matrix_profile_series,
    _z_normalize,
)
from cleaned_operators.rqa_ext import _rqa_epsilon_for_target_rr, _rqa_stats_window_fixed_rr
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.rolling_pack import frame_like


def _get(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _calc(name: str, *frames: pd.DataFrame, **params: object) -> np.ndarray:
    return np.asarray(_get(name).calculate(*frames, **params), dtype=float)


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# #14 pivot staleness — structural_levels
# ---------------------------------------------------------------------------
def test_item14_max_pivot_age_na_turns_stale_pivot_into_nan():
    # A single early pivot (peak at bar 5 confirmed row 8) followed by a long
    # drift with NO newer pivot.  With window=30 the pivot stays inside the
    # trailing window the whole time, so without a staleness cap it keeps
    # contributing.  max_pivot_age=2 must stop using it once age > 2 and the
    # level must become NaN (no newer pivot to fall back on).
    x = np.array([
        100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 109.5, 109.0, 109.5,
        112.0, 115.0, 118.0, 121.0, 117.0, 113.0, 110.0, 108.0, 105.0,
        103.0, 101.0, 100.0,
    ])
    base = dict(window=30, prominence=0.02, confirmation=3, min_coverage_fraction=0.0)
    loose = _calc("ts_structural_level_density", _frame(x), **base)
    capped = _calc("ts_structural_level_density", _frame(x), **{**base, "max_pivot_age": 2})
    # The pivot is usable from confirmed row 8; age = t - (pivot_at + conf).
    # With max_pivot_age=2 the cap is exceeded for t >= 8 + 2 + 1 = 11.
    assert np.all(np.isnan(capped[11:])), "stale pivot kept being used beyond max_pivot_age"
    # Default (no cap) is backward compatible — still emits on the stale pivot.
    assert np.isfinite(loose[-1]), "default behaviour changed unexpectedly"


def test_item14_max_pivot_age_exposed_on_all_three_level_ops():
    for name in (
        "ts_structural_level_density",
        "ts_nearest_structural_level_distance",
        "ts_structural_level_strength",
    ):
        assert "max_pivot_age" in _get(name).metadata.param_names, name


# ---------------------------------------------------------------------------
# #63 matrix-profile motif age is start-to-start
# ---------------------------------------------------------------------------
def test_item63_motif_age_start_to_start():
    L = 4
    pattern = np.array([3.0, -1.0, 2.0, 0.5])
    p1, p2 = 10, 30  # distinctive pattern starts
    r = p2 + L - 1   # current subsequence END
    x = np.zeros(50)
    rng = np.random.default_rng(1)
    x[:] = rng.normal(0.0, 1e-8, 50)
    x[p1 : p1 + L] = pattern
    x[p2 : p2 + L] = pattern
    _, age, _, _ = _matrix_profile_series(
        x[:, None], window=30, subsequence_length=L, history=25
    )
    # True start-to-start age = (r - L + 1) - p1 = 30 - 10 = 20, normalised by h=25.
    expected = (r - L + 1 - p1) / 25.0
    old_wrong = (r - p1) / 25.0
    assert abs(float(age[r, 0]) - expected) < 1e-9
    assert abs(float(age[r, 0]) - old_wrong) > 1e-3


def test_item63_motif_age_operator_is_finite_on_walk():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0.0, 1.0, 120))
    out = _calc("ts_matrix_profile_motif_age", _frame(x), window=60, subsequence_length=8, history=40)
    assert np.isfinite(out).mean() > 0.5


# ---------------------------------------------------------------------------
# #64 constant subsequence -> NaN (shape-motif semantics)
# ---------------------------------------------------------------------------
def test_item64_constant_z_normalize_is_nan():
    flat = _z_normalize(np.array([10.0, 10.0, 10.0, 10.0]))
    assert np.all(np.isnan(flat)), "constant subsequence must not collapse to zeros"


def test_item64_constant_current_subsequence_nan():
    # A fully-constant series: every window is flat -> every reading NaN.
    xc = np.ones((60, 1)) * 5.0
    nov, _, _, _ = _matrix_profile_series(xc, window=40, subsequence_length=6, history=30)
    assert np.all(np.isnan(nov))
    # [10,10,10,10] and [100,100,100,100] must NOT match each other as the same
    # shape — the NaN policy means neither produces a shape distance.
    out = _calc("ts_matrix_profile_novelty", _frame(np.full(40, 10.0)), window=30, subsequence_length=6, history=20)
    assert np.all(np.isnan(out))


def test_item64_flat_history_blocks_matching():
    # A step after a long flat block: the current window has a defined shape but
    # every historical candidate is flat (degenerate) -> no comparable match,
    # so the reading is NaN rather than matching a different-level flat segment.
    x2 = np.concatenate([np.ones(25), np.linspace(0.0, 1.0, 60)])[:, None]
    nov, _, _, _ = _matrix_profile_series(x2, window=40, subsequence_length=6, history=30)
    flat_rows = np.isnan(nov[5:25, 0])  # windows fully inside the flat block
    assert flat_rows.all()


# ---------------------------------------------------------------------------
# #65 / #66 Mahalanobis + KNN — estimator consistency and sample floor
# ---------------------------------------------------------------------------
def _reference_mahalanobis(f1, f2, f3, f4, window, shrinkage):
    """Reference implementation of the (now classical-consistent) kernel."""
    feat = (f1, f2, f3, f4)
    p = len(feat)
    rows = f1.shape[0]
    out = np.full(rows, np.nan, dtype=float)
    for r in range(rows):
        i0 = max(0, r - window + 1)
        hist = np.stack([feat[j][i0:r, 0] for j in range(p)], axis=1)
        cur = np.stack([feat[j][r, 0] for j in range(p)])
        if not np.all(np.isfinite(cur)):
            continue
        finite = np.all(np.isfinite(hist), axis=1)
        valid = hist[finite].astype(float)
        if valid.shape[0] == 0:
            continue
        keep_cols = np.where(np.std(valid, axis=0) > 1e-12)[0]
        if keep_cols.size == 0 or valid.shape[0] < 5 * keep_cols.size:
            continue
        valid = valid[:, keep_cols]
        z = cur[keep_cols]
        mu = valid.mean(axis=0)
        cov = np.atleast_2d(np.cov(valid, rowvar=False, ddof=1))
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        shrunk = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
        prec = np.linalg.pinv(shrunk + 1e-12 * np.eye(keep_cols.size))
        d = z - mu
        out[r] = float(np.sqrt(max(0.0, float(d @ prec @ d))))
    return out


def test_item65_mahalanobis_uses_classical_mean_not_median():
    # Skewed history where the median differs materially from the mean.  The
    # kernel must equal the mean + ordinary-covariance reference exactly (a
    # half-robust median centre would produce a different distance).
    rng = np.random.default_rng(7)
    n = 60
    g1 = rng.gamma(2.0, 1.0, (n, 1)) + 1.0
    g2 = rng.gamma(3.0, 0.7, (n, 1)) + 2.0
    g3 = rng.gamma(1.5, 1.2, (n, 1)) + 0.5
    g4 = rng.gamma(2.5, 0.9, (n, 1)) + 1.5
    out = _mahalanobis_series(g1, g2, g3, g4, window=40, shrinkage=0.5)[:, 0]
    ref = _reference_mahalanobis(g1, g2, g3, g4, window=40, shrinkage=0.5)
    finite = np.isfinite(ref)
    assert finite.sum() > 10
    np.testing.assert_allclose(out[finite], ref[finite], rtol=1e-9, atol=1e-12)


def test_item66_mahalanobis_and_knn_sample_floor():
    # 4 features but only ~6 finite history rows: N < 5p (20) -> covariance /
    # k-NN are meaningless -> every reading NaN.
    rng = np.random.default_rng(2)
    small = np.full((30, 1), np.nan)
    small[:6, :] = rng.normal(0.0, 1.0, (6, 1))
    args = [small.copy() for _ in range(4)]
    mah = _mahalanobis_series(*args, window=10, shrinkage=0.5)
    assert np.all(np.isnan(mah)), "N < 5p must be NaN for mahalanobis"
    knn = _local_density_series(*args, window=10, k=2)
    assert np.all(np.isnan(knn)), "N < 5p must be NaN for KNN density"


def test_item66_mahalanobis_finite_with_enough_samples():
    rng = np.random.default_rng(5)
    n = 80
    arrs = [rng.normal(0.0, 1.0, (n, 1)) for _ in range(4)]
    out = _mahalanobis_series(*arrs, window=40, shrinkage=0.5)
    assert np.isfinite(out).mean() > 0.5
    knn = _local_density_series(*arrs, window=40, k=5)
    assert np.isfinite(knn).mean() > 0.5


# ---------------------------------------------------------------------------
# #67 RQA effective-length floor (>= 0.8*window)
# ---------------------------------------------------------------------------
def _gap_series(n=120):
    rng = np.random.default_rng(0)
    y = np.full(n, np.nan)
    y[:40] = rng.normal(0.0, 1.0, 40)
    y[40] = np.nan  # gap
    y[41:] = rng.normal(0.0, 1.0, n - 41)
    return y


def test_item67_rqa_ext_determinism_no_short_sample_after_gap():
    y = _gap_series()
    out = _calc("ts_recurrence_determinism", _frame(y), window=60, min_periods=10)
    # Right after the gap (row 42) only ~1 contiguous bar is available -> NaN,
    # NOT a short-sample recompute.
    assert np.isnan(out[42, 0])
    # Once >= ceil(0.8*60)=48 contiguous bars accumulate (~row 90) it recovers.
    assert np.isfinite(out[90:, 0]).mean() > 0.8


def test_item67_rqa_ext_laminarity_floor_and_recovery():
    y = _gap_series()
    out = _calc("ts_recurrence_laminarity", _frame(y), window=60, min_periods=10)
    assert np.isnan(out[42, 0])
    assert np.isfinite(out[90:, 0]).mean() > 0.8


def test_item67_recurrence_analysis_line_stats_floor():
    # recurrence_analysis line-structure ops (diagonal entropy, trapping time,
    # divergence) share the same effective-length floor.
    y = _gap_series()
    for name in (
        "ts_recurrence_diagonal_entropy",
        "ts_recurrence_trapping_time",
        "ts_recurrence_divergence",
    ):
        out = _calc(name, _frame(y), window=60, min_periods=10)
        assert np.isnan(out[42, 0]), name
        assert np.isfinite(out[90:, 0]).mean() > 0.8, name


def test_item67_determinism_still_discriminates_periodic_vs_noise():
    # The floor must not break the classic semantics on fully-finite windows.
    periodic = np.tile([1.0, 0.5, -0.5, -1.0], 30)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(120)
    d_per = _calc("ts_recurrence_determinism", _frame(periodic), window=60, dim=1, min_line=4)
    d_noi = _calc("ts_recurrence_determinism", _frame(noise), window=60, dim=1, min_line=4)
    assert np.isfinite(d_per[-1, 0]) and np.isfinite(d_noi[-1, 0])
    assert d_per[-1, 0] > d_noi[-1, 0]


# ---------------------------------------------------------------------------
# #68 fixed-recurrence-rate DET / LAM
# ---------------------------------------------------------------------------
def test_item68_fixed_rr_epsilon_targeting_hits_target():
    v = np.sin(np.linspace(0.0, 20 * np.pi, 60))
    stats = _rqa_stats_window_fixed_rr(v, dim=1, delay=1, target_rr=0.05, min_line=4, theiler=0)
    assert np.isfinite(stats["determinism"]) and np.isfinite(stats["laminarity"])
    assert abs(stats["rate"] - 0.05) < 0.03, stats["rate"]


def test_item68_fixed_rr_epsilon_routine():
    rng = np.random.default_rng(4)
    P = rng.normal(0.0, 1.0, (40, 2))
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    res = _rqa_epsilon_for_target_rr(D, theiler=0, target_rr=0.2)
    assert res is not None
    eps, n = res
    n_el = n
    # eps is the ceil(0.2*n)-th smallest distance, so RR ~ 0.2.
    iu, ju = np.triu_indices(40, k=1)
    d = D[iu, ju]
    frac = float((d <= eps).mean())
    assert abs(frac - 0.2) < 0.01


def test_item68_fixed_rr_canonicals_registered_and_classify_daily():
    for name in ("ts_rqa_determinism_fixed_rr", "ts_rqa_laminarity_fixed_rr"):
        assert name in OperatorRegistry.list_canonical(), name
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends and "polars" in backends, name
        op = _get(name)
        assert "target_rr" in op.metadata.param_names, name
        # target_rr is a non-searchable estimator-resolution knob.
        spec = op.metadata.param_specs.get("target_rr")
        assert spec is not None and spec.searchable is False, name


def test_item68_fixed_rr_operators_run():
    rng = np.random.default_rng(6)
    x = np.cumsum(rng.normal(0.0, 1.0, 120))
    det = _calc("ts_rqa_determinism_fixed_rr", _frame(x), window=60, target_rr=0.05)
    lam = _calc("ts_rqa_laminarity_fixed_rr", _frame(x), window=60, target_rr=0.05)
    assert np.isfinite(det[90:, 0]).mean() > 0.8
    assert np.isfinite(lam[90:, 0]).mean() > 0.8
