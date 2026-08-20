"""Tests for true-time-axis temporal metrics (QE-METRIC-P0-06).

Pairwise-finite lag alignment: for lag k, only pairs (t, t-k) where BOTH
original positions are finite contribute. NaN-compression before lagging is
the bug being guarded against.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.temporal import (
    compute_autocorrelation,
    compute_half_life,
)


def compressed_autocorr(series, max_lag, min_obs):
    """The OLD (buggy) implementation: NaN-compress then lag."""
    valid = series[~np.isnan(series)]
    if len(valid) < min_obs:
        return np.full(max_lag + 1, np.nan)
    acf = np.full(max_lag + 1, np.nan)
    acf[0] = 1.0
    mean_ts = np.mean(valid)
    var_ts = np.var(valid, ddof=0)
    if var_ts == 0:
        return acf
    for lag in range(1, max_lag + 1):
        if len(valid) - lag < min_obs:
            break
        cov = np.mean((valid[:-lag] - mean_ts) * (valid[lag:] - mean_ts))
        acf[lag] = cov / var_ts
    return acf


class TestAutocorrelationTrueAxis:
    def test_gap_changes_lag1_autocorr(self):
        """A large interior NaN gap must yield a DIFFERENT (correct)
        lag-1 autocorr vs the compressed version: compression splices the
        block boundary into one adjacent pair (last of A, first of B) that
        never existed in time; the true axis excludes it."""
        rng = np.random.default_rng(42)
        block_a = rng.normal(0.0, 1.0, 90)
        block_b = rng.normal(0.0, 1.0, 90)
        # Extremes exactly at the boundary: the spliced pair (A[-1], B[0])
        # dominates compressed variance but is invisible to the true axis.
        block_a[-1] = 50.0
        block_b[0] = 50.0
        series = np.concatenate([
            block_a,
            np.full(20, np.nan),  # large interior gap
            block_b,
        ])
        assert series.shape[0] == 200

        acf_true = compute_autocorrelation(series, max_lag=1, min_obs=30)
        acf_compressed = compressed_autocorr(series, max_lag=1, min_obs=30)

        # True axis: all pairs are white-noise pairs -> rho ~ 0.
        assert abs(acf_true[1]) < 0.15
        # Compressed: the one spurious (50, 50) pair inflates rho.
        assert acf_compressed[1] > 0.3
        assert abs(acf_true[1] - acf_compressed[1]) > 0.3

    def test_gap_pairs_never_cross_the_gap(self):
        """With a full NaN gap of length G, no lag-1 pair spans the gap:
        the pairwise estimator only uses within-block adjacency."""
        # Two independent blocks with opposite-sign lag-1 dependence.
        block_a = np.array([1.0, -1.0] * 30)   # strong negative lag-1
        block_b = np.array([1.0, 1.0, -1.0, -1.0] * 15)  # positive-ish
        gap = np.full(10, np.nan)
        series = np.concatenate([block_a, gap, block_b])

        acf = compute_autocorrelation(series, max_lag=1, min_obs=30)

        # Hand-computable: pairs only within blocks. All finite pairs:
        # block_a: 59 pairs alternating (+1,-1): corr = -1
        # block_b: 59 pairs: pattern ++--++--: 30 same-sign, 29 opposite
        # combined correlation must be strictly negative (block_a dominates
        # variance) but NOT -1 (which compression-style mingling could give).
        assert acf[1] < -0.3
        assert acf[1] > -1.0 + 1e-9

    def test_hand_computable_pairs(self):
        """Exact pairwise correlation on a small hand-built series."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                           np.nan, np.nan, np.nan, np.nan,
                           9.0, 10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0, 16.0])
        # Lag-1 pairs: (1,2)..(7,8) and (9,10)..(15,16) -> 14 pairs, all
        # consecutive integers: correlation of (x, x+1) over x in 1..7,9..15
        # is exactly 1.
        acf = compute_autocorrelation(series, max_lag=2, min_obs=10)

        assert acf[1] == pytest.approx(1.0, abs=1e-12)
        # Lag-2 pairs: (1,3)..(6,8) and (9,11)..(14,16): also perfect.
        assert acf[2] == pytest.approx(1.0, abs=1e-12)

    def test_all_pairs_required_finite_at_both_positions(self):
        """A pair with NaN at either original position is excluded, even
        when the lag skips over it."""
        series = np.arange(30, dtype=float)
        series[10] = np.nan
        # Lag-2 pair (11, 9) uses positions 9 and 11 — both finite, included.
        # Pair (12, 10): position 10 is NaN -> excluded.
        acf = compute_autocorrelation(series, max_lag=2, min_obs=10)

        # Remaining pairs are still perfectly linear -> rho = 1.
        assert acf[2] == pytest.approx(1.0, abs=1e-10)

    def test_insufficient_pairs_nan(self):
        """Too few valid pairs at a lag -> NaN (fail-closed)."""
        series = np.arange(40, dtype=float)
        series[20:] = np.nan  # only 20 finite values
        acf = compute_autocorrelation(series, max_lag=5, min_obs=30)

        assert np.all(np.isnan(acf))

    def test_constant_series_nan_lags(self):
        """Constant series: correlation undefined at lags > 0 -> NaN."""
        series = np.full(50, 3.0)
        acf = compute_autocorrelation(series, max_lag=3, min_obs=10)

        assert acf[0] == 1.0
        assert np.all(np.isnan(acf[1:]))


class TestHalfLifeTrueAxis:
    def test_half_life_ar1_oracle(self):
        """Hand-computable AR(1): x_t = phi * x_{t-1} exactly (no noise).
        phi = 0.5 -> half-life = -log(2)/log(0.5) = 1."""
        phi = 0.5
        T = 100
        series = np.array([[phi ** t] for t in range(T)])

        hl = compute_half_life(series, min_periods=60)

        assert hl[0] == pytest.approx(-np.log(2) / np.log(phi), rel=1e-9)

    def test_half_life_gap_does_not_fabricate_pairs(self):
        """An interior NaN gap must NOT create an (x_end, x_start) pair
        across the gap; half-life differs from the compressed estimate."""
        phi = 0.9
        T = 150
        x = np.zeros(T)
        x[0] = 1.0
        for t in range(1, T):
            x[t] = phi * x[t - 1] + 0.0  # noiseless AR(1)

        # Insert gap and RESET the process after it to +1 (a fresh shock):
        # compression would pair x[-1 before gap] with x[after gap]=1.
        gapped = x.copy()
        gapped[60:90] = np.nan
        gapped[90] = 1.0
        for t in range(91, T):
            gapped[t] = phi * gapped[t - 1]

        hl = compute_half_life(gapped.reshape(-1, 1), min_periods=60)
        # Noiseless segments each give phi exactly; pairs across the gap
        # (0.9^59, 1.0) would bias phi if compressed. True estimator:
        expected_phi = phi
        assert hl[0] == pytest.approx(
            -np.log(2) / np.log(expected_phi), rel=1e-6
        )

    def test_half_life_insufficient_periods(self):
        series = np.random.default_rng(1).normal(size=(30, 1))
        hl = compute_half_life(series, min_periods=60)
        assert np.isnan(hl[0])

    def test_half_life_white_noise_nan(self):
        series = np.random.default_rng(2).normal(size=(100, 1))
        hl = compute_half_life(series, min_periods=60)
        # phi ~ 0 -> NaN (out of (0,1) range)
        assert np.isnan(hl[0]) or hl[0] > 0

    def test_half_life_multiple_factors(self):
        phi_slow = 0.9
        phi_fast = 0.2
        T = 200
        slow = np.array([[phi_slow ** t] for t in range(T)])
        fast = np.array([[phi_fast ** t] for t in range(T)])
        ic = np.hstack([slow, fast])

        hl = compute_half_life(ic, min_periods=60)

        assert hl[0] == pytest.approx(-np.log(2) / np.log(phi_slow), rel=1e-6)
        assert hl[1] == pytest.approx(-np.log(2) / np.log(phi_fast), rel=1e-6)
        assert hl[0] > hl[1]

    def test_half_life_gap_vs_compressed_differ(self):
        """Explicit divergence: pairwise vs compressed estimators give
        different phi on a gapped series."""
        # Blocks of [1.0, 0.5] separated by single-NaN gaps. Every REAL
        # adjacent pair is (1.0, 0.5) -> phi_true = 0.5 exactly (half-life
        # exactly 1). Compression splices (0.5, 1.0) pairs across gaps,
        # dragging phi toward ~0.8.
        K = 60  # blocks (>= min_periods - 1 real pairs)
        blocks = []
        for k in range(K):
            blocks.append(np.array([1.0, 0.5]))
            if k < K - 1:
                blocks.append(np.array([np.nan]))
        series = np.concatenate(blocks)

        hl = compute_half_life(series.reshape(-1, 1), min_periods=60)

        # Pairwise phi (the true estimator's internal quantity).
        valid = np.isfinite(series)
        pair_mask = valid[1:] & valid[:-1]
        x = series[:-1][pair_mask]
        y = series[1:][pair_mask]
        phi_true = np.sum(x * y) / np.sum(x * x)

        # Compressed phi (the OLD estimator's quantity).
        compressed = series[valid]
        xc = compressed[:-1]
        yc = compressed[1:]
        phi_compressed = np.sum(xc * yc) / np.sum(xc * xc)

        # phi_true is exactly 0.5; compression inflates it toward 0.8.
        assert phi_true == pytest.approx(0.5, abs=1e-12)
        assert abs(phi_true - phi_compressed) > 0.2
        # And the reported half-life comes from the pairwise phi:
        # -log(2)/log(0.5) = 1.0.
        assert hl[0] == pytest.approx(1.0, rel=1e-9)
