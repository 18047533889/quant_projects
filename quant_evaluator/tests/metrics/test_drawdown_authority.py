"""Tests for the drawdown authority (QE-METRIC-P0-09).

compute_maximum_drawdown must:
  - report the TRUE peak index (last index where wealth attains its running
    maximum at or before the trough), not the trough itself;
  - forward-fill NaN drawdown after a wealth wipeout (<= 0), matching
    metrics/risk/drawdown_analysis.py;
  - accept missing_return_policy in {"zero_fill", "fail"} — "fail" raises on
    any NaN return; "zero_fill" (default, back-compat) silently treats NaN
    as flat days.
"""

import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown


class TestTruePeakIndex:
    def test_peak_is_not_the_trough(self):
        """Gain, then loss, then partial recovery: the peak precedes the
        trough. The old implementation returned the trough index."""
        # wealth: 1.2 -> 0.9 (trough, dd = 0.25) -> 1.1
        returns = np.array([0.2, -0.25, 1.0 / 9.0])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        trough_idx = int(np.nanargmin(dd_series))
        assert trough_idx == 1
        assert peak_idx == 0, "peak must be the pre-drawdown high, not the trough"
        assert max_dd == pytest.approx(0.25, rel=1e-9)
        assert dd_series[trough_idx] == pytest.approx(-0.25, rel=1e-9)

    def test_peak_is_last_running_max_index(self):
        """Equal highs: the peak is the LAST index attaining the running
        max before the trough, not the first."""
        # wealth: 1.1 -> 1.1 (same high) -> 0.88 (dd 0.2)
        returns = np.array([0.1, 0.0, -0.2, 0.0])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        assert max_dd == pytest.approx(0.2, rel=1e-9)
        assert peak_idx == 1, "last index where wealth == running max"

    def test_no_drawdown(self):
        returns = np.array([0.01, 0.02, 0.015])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
        assert max_dd == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(dd_series, 0.0)

    def test_multi_factor_shapes(self):
        """(T, F) input -> per-factor arrays without squeezing."""
        rng = np.random.default_rng(4)
        rets = rng.normal(0.001, 0.02, size=(100, 3))
        rets[:, 1] = np.array([0.1] + [-0.05] * 40 + [0.02] * 59)

        max_dd, dd_series, peak_idx = compute_maximum_drawdown(rets)

        assert max_dd.shape == (3,)
        assert dd_series.shape == (100, 3)
        assert peak_idx.shape == (3,)
        for f in range(3):
            trough = int(np.nanargmin(dd_series[:, f]))
            assert peak_idx[f] <= trough
            assert dd_series[peak_idx[f], f] == pytest.approx(0.0, abs=1e-9)


class TestWealthWipeoutGuard:
    def test_wipeout_nan_forward_fill(self):
        """A return <= -1 wipes out wealth: drawdown is NaN from that point
        onward (never a fake recovery), matching drawdown_analysis.py."""
        returns = np.array([0.1, 0.1, -1.2, 0.5, 0.5])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)

        assert np.isfinite(dd_series[0]) and dd_series[0] == pytest.approx(0.0)
        assert np.isfinite(dd_series[1]) and dd_series[1] == pytest.approx(0.0)
        assert np.all(np.isnan(dd_series[2:]))
        # Fail closed: no fabricated drawdown number from negative wealth.
        assert max_dd == pytest.approx(0.0, abs=1e-12) or np.isnan(max_dd)

    def test_wipeout_from_start_is_nan_not_crash(self):
        """Wealth <= 0 at t=0: entire series invalid; must not raise."""
        returns = np.array([-1.5, 0.2, 0.1])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
        assert np.all(np.isnan(dd_series))
        assert np.isnan(max_dd)


class TestMissingReturnPolicy:
    def test_fail_policy_raises_on_nan(self):
        returns = np.array([0.01, np.nan, 0.02])
        with pytest.raises(ValueError, match="non-finite"):
            compute_maximum_drawdown(returns, missing_return_policy="fail")

    def test_zero_fill_policy_backcompat(self):
        """Default policy: NaN returns are treated as flat days."""
        returns = np.array([0.1, np.nan, -0.1, np.nan, 0.05])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(returns)
        # NaN at index 1 acts as 0: wealth 1.1 -> 1.1 -> 0.99 (dd 0.1)
        assert max_dd == pytest.approx(0.1, rel=1e-9)
        assert np.all(np.isfinite(dd_series))

    def test_invalid_policy_raises(self):
        returns = np.array([0.01, 0.02])
        with pytest.raises(ValueError, match="missing_return_policy"):
            compute_maximum_drawdown(returns, missing_return_policy="bfill")

    def test_fail_policy_passes_on_clean_data(self):
        returns = np.array([0.05, -0.1, 0.08])
        max_dd, dd_series, peak_idx = compute_maximum_drawdown(
            returns, missing_return_policy="fail"
        )
        assert max_dd == pytest.approx(0.1, rel=1e-9)
