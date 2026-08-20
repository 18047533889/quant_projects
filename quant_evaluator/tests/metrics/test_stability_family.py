"""Tests for the IC stability family (QE-METRIC-P0-07).

compute_ic_stability is EXPERIMENTAL (weak first/second-half correlation
proxy). The richer family is:
  - compute_rolling_ic_stats
  - compute_yearly_quarterly_dispersion
  - compute_change_point_score
All NaN-aware and fail-closed.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.ic_summary import (
    compute_ic_stability,
    compute_rolling_ic_stats,
    compute_yearly_quarterly_dispersion,
    compute_change_point_score,
)


def make_ic(T=300, seed=0, mu=0.05, sigma=0.1):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=T)


class TestICStabilityExperimental:
    def test_docstring_marks_experimental(self):
        assert "EXPERIMENTAL" in compute_ic_stability.__doc__
        assert "experimental" in compute_ic_stability.__doc__

    def test_not_registered_as_stable_metric(self):
        """compute_ic_stability must not appear in the metric registry."""
        from quant_evaluator.registry import metrics as registry_metrics

        names = set(registry_metrics.METRIC_SPECS.keys()) if hasattr(
            registry_metrics, "METRIC_SPECS"
        ) else set()
        # Any registry binding of ic_stability would be a spec entry.
        for name in names:
            assert "stability" not in name.lower() or "rolling" in name.lower(), (
                f"experimental stability metric registered: {name}"
            )


class TestRollingICStats:
    def test_perfectly_stable_positive_ic(self):
        """Consistently positive IC -> positive_ic_ratio 1, long runs."""
        T = 300
        ic = np.abs(make_ic(T=T, seed=1)) + 0.01  # all positive
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)

        assert stats["positive_ic_ratio"][0] == pytest.approx(1.0)
        assert stats["sign_survival"][0] == pytest.approx(float(T))
        assert stats["recent_vs_full"][0] == pytest.approx(0.0, abs=0.05)

    def test_decaying_ic_recent_vs_full_negative(self):
        """IC that decays to zero -> recent mean below full mean."""
        T = 400
        t = np.arange(T)
        ic = 0.1 * (1.0 - t / T) + make_ic(T=T, seed=2, mu=0.0, sigma=0.01)
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)

        assert stats["recent_vs_full"][0] < 0

    def test_improving_ic_recent_vs_full_positive(self):
        T = 400
        t = np.arange(T)
        ic = 0.1 * (t / T) + make_ic(T=T, seed=3, mu=0.0, sigma=0.01)
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)

        assert stats["recent_vs_full"][0] > 0

    def test_worst_rolling_ic(self):
        """A single bad regime is captured by worst_rolling_ic."""
        ic = make_ic(T=300, seed=4, mu=0.05, sigma=0.02)
        ic[100:160] = -0.3  # bad regime
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)

        assert stats["worst_rolling_ic"][0] < -0.2

    def test_rolling_shapes_and_nan_prefix(self):
        T = 200; F = 2
        rng = np.random.default_rng(5)
        ic = rng.normal(0.03, 0.1, size=(T, F))
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)

        assert stats["rolling_ic_mean"].shape == (T, F)
        assert stats["rolling_ic_ir"].shape == (T, F)
        for key in ("positive_ic_ratio", "sign_survival",
                    "worst_rolling_ic", "recent_vs_full"):
            assert stats[key].shape == (F,)
        # Early rows lack min_periods finite observations.
        assert np.all(np.isnan(stats["rolling_ic_mean"][:10, :]))

    def test_nan_aware(self):
        """NaN entries are excluded, not treated as zeros."""
        T = 300
        ic = make_ic(T=T, seed=6, mu=0.05, sigma=0.1)
        ic_with_nan = ic.copy()
        ic_with_nan[50:100] = np.nan
        clean = compute_rolling_ic_stats(ic, window=60, min_periods=20)
        gapped = compute_rolling_ic_stats(ic_with_nan, window=60, min_periods=20)

        # Full-sample positive ratio uses only finite values.
        assert gapped["positive_ic_ratio"][0] == pytest.approx(
            clean["positive_ic_ratio"][0], abs=0.15
        )
        # Inside the gap the rolling mean is still computable from the
        # finite tail of the window (NaN-aware, not NaN-propagating); but
        # the earliest fully-NaN rows are NaN.
        assert np.all(np.isnan(gapped["rolling_ic_mean"][:10, :]))
        # Windows fully covering the gap still produce finite values:
        assert np.isfinite(gapped["rolling_ic_mean"][-1, 0])

    def test_insufficient_data_fail_closed(self):
        ic = make_ic(T=15, seed=7)
        stats = compute_rolling_ic_stats(ic, window=60, min_periods=20)
        for key in ("positive_ic_ratio", "sign_survival",
                    "worst_rolling_ic", "recent_vs_full"):
            assert np.isnan(stats[key][0])
        assert np.all(np.isnan(stats["rolling_ic_mean"]))


class TestYearlyQuarterlyDispersion:
    def test_position_based_grouping(self):
        """No time_index -> quarters of 63, years of 252 positions."""
        T = 504
        t = np.arange(T)
        ic = 0.05 * np.sin(t / 63.0 * np.pi)  # year-structured variation
        out = compute_yearly_quarterly_dispersion(ic)

        assert out["n_years"][0] == 2
        assert out["n_quarters"][0] == 8
        assert out["yearly_mean_ic_std"][0] >= 0
        assert out["quarterly_mean_ic_std"][0] > out["yearly_mean_ic_std"][0]

    def test_datetime_labels(self):
        """datetime64 labels group by calendar year and quarter."""
        idx = np.arange("2022-01-01", "2024-01-01", dtype="datetime64[D]")[:504]
        rng = np.random.default_rng(8)
        ic = rng.normal(0.03, 0.1, size=len(idx))
        out = compute_yearly_quarterly_dispersion(ic, time_index=idx)

        assert out["n_years"][0] == 2
        assert out["n_quarters"][0] >= 6

    def test_insufficient_groups_nan(self):
        ic = make_ic(T=100, seed=9)  # < 1 position-year
        out = compute_yearly_quarterly_dispersion(ic)
        assert np.isnan(out["yearly_mean_ic_std"][0])
        assert np.isfinite(out["quarterly_mean_ic_std"][0])
        assert out["n_years"][0] == 1

    def test_time_index_length_mismatch_raises(self):
        ic = make_ic(T=100, seed=10)
        with pytest.raises(ValueError, match="time_index"):
            compute_yearly_quarterly_dispersion(
                ic, time_index=np.arange(50)
            )


class TestChangePointScore:
    def test_regime_shift_detected(self):
        """IC that flips from +0.2 to -0.2 mid-sample has a large score."""
        T = 400
        ic = np.concatenate([
            np.full(T // 2, 0.2) + make_ic(T // 2, seed=11, mu=0.0, sigma=0.02),
            np.full(T - T // 2, -0.2) + make_ic(T - T // 2, seed=12, mu=0.0, sigma=0.02),
        ])
        score = compute_change_point_score(ic, window=60)

        # Rolling mean must traverse ~[-0.2, 0.2]; normalized by full std
        # (~0.2), the max one-step shift is sizable.
        assert score[0] > 0.2

    def test_stable_ic_low_score(self):
        """Steady IC -> tiny max one-step rolling-mean shift."""
        ic = make_ic(T=400, seed=13, mu=0.05, sigma=0.1)
        shifted = np.concatenate([
            np.full(200, 0.2) + make_ic(200, seed=11, mu=0.0, sigma=0.02),
            np.full(200, -0.2) + make_ic(200, seed=12, mu=0.0, sigma=0.02),
        ])
        s_stable = compute_change_point_score(ic, window=60)[0]
        s_shift = compute_change_point_score(shifted, window=60)[0]
        assert s_shift > s_stable

    def test_constant_series_nan(self):
        ic = np.full(300, 0.05)
        score = compute_change_point_score(ic, window=60)
        assert np.isnan(score[0])

    def test_insufficient_data_nan(self):
        ic = make_ic(T=20, seed=14)
        score = compute_change_point_score(ic, window=60)
        assert np.isnan(score[0])
