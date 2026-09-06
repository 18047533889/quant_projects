"""
QE-R2 (AlphaPROBE refactor): long/short + risk-adjusted return metrics.

Pins the four returns-panel metrics registered into the SINGLE metric
registry authority (``quant_evaluator.registry.metrics``):

- ``long_short_returns``
- ``sharpe_ratio``
- ``sortino_ratio``
- ``win_rate``

Each test resolves the spec through the registry, asserts the spec's
declared semantics (status / implementation identity / direction / units /
parameter defaults), and checks the bound ``compute_fn`` numerically against
a hand-computed oracle (independent NumPy arithmetic, no reuse of the
implementation under test).

Also verifies that the implementations in ``metrics/portfolio_stats.py`` are
NaN-aware, enforce ``min_periods``, and fail closed (``missing_return_policy
="fail"`` raises).

Run with:

    PYTHONPATH=/home/sunhaiwei/quant_projects \\
    python -m pytest quant_evaluator/tests/test_portfolio_stats_registry.py -q --tb=short
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_evaluator.registry.metrics import get_metric, list_metrics


def _returns_series(rng: np.random.Generator, T: int = 60) -> np.ndarray:
    """Deterministic pseudo-return series for oracle tests."""
    return rng.normal(0.001, 0.01, size=T)


# ---------------------------------------------------------------------------
# Registry presence + spec semantics.
# ---------------------------------------------------------------------------


class TestRegistryPresence:
    def test_four_metrics_registered(self):
        for metric_id in (
            "long_short_returns",
            "sharpe_ratio",
            "sortino_ratio",
            "win_rate",
        ):
            assert metric_id in list_metrics(), f"{metric_id} not registered"

    def test_spec_metadata(self):
        cases = {
            "long_short_returns": (
                "series",
                "return",
                "higher_is_better",
            ),
            "sharpe_ratio": ("scalar", "ratio", "higher_is_better"),
            "sortino_ratio": ("scalar", "ratio", "higher_is_better"),
            "win_rate": ("scalar", "fraction", "higher_is_better"),
        }
        for metric_id, (artifact_kind, units, direction) in cases.items():
            spec = get_metric(metric_id)
            assert spec.artifact_kind == artifact_kind, metric_id
            assert spec.units == units, metric_id
            assert spec.direction == direction, metric_id
            assert spec.compute_fn is not None, metric_id
            assert spec.implementation_id != metric_id, metric_id

    def test_implementation_hash_content_derived(self):
        import quant_evaluator.registry.metrics as rm

        # Distinct kernels -> distinct hashes (portfolio_stats functions).
        s = get_metric("sharpe_ratio")
        w = get_metric("win_rate")
        l = get_metric("long_short_returns")
        assert s.implementation_hash != w.implementation_hash
        assert s.implementation_hash != l.implementation_hash
        # Hash is a stable 16-hex content fingerprint.
        assert len(s.implementation_hash) == 16
        int(s.implementation_hash, 16)
        # hash reflects real source module
        assert s.implementation_id.startswith(
            "quant_evaluator.metrics.portfolio_stats."
        )

    def test_requires_forward_returns_input(self):
        for metric_id in (
            "sharpe_ratio",
            "sortino_ratio",
            "win_rate",
            "long_short_returns",
        ):
            spec = get_metric(metric_id)
            assert "forward_returns" in spec.required_inputs, metric_id


# ---------------------------------------------------------------------------
# Numerical oracles (independent hand-computed arithmetic).
# ---------------------------------------------------------------------------


class TestSharpeRatioOracle:
    def test_matches_hand_computed_annualized_sharpe(self):
        ret = np.array(
            [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.015, -0.005, 0.025,
             0.01, -0.015, 0.02, 0.005, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02,
             0.01, -0.01, 0.015, 0.005, -0.005],
            dtype=np.float64,
        )
        fn = get_metric("sharpe_ratio").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        mean = ret.mean()
        std = ret.std(ddof=1)
        expected = mean / std * np.sqrt(252)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_risk_free_rate_uses_per_period_subtraction(self):
        ret = np.array(
            [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.015, -0.005, 0.025,
             0.01, -0.015, 0.02, 0.005, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02,
             0.01, -0.01, 0.015, 0.005, -0.005],
            dtype=np.float64,
        )
        fn = get_metric("sharpe_ratio").compute_fn
        got = float(np.asarray(fn(ret, risk_free_rate=0.02)).ravel()[0])
        ex = ret - 0.02 / 252
        expected = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_nan_aware_and_ddof1(self):
        rng = np.random.default_rng(3)
        ret = _returns_series(rng)
        ret[5:10] = np.nan
        fn = get_metric("sharpe_ratio").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        valid = ret[np.isfinite(ret)]
        expected = valid.mean() / valid.std(ddof=1) * np.sqrt(252)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_min_periods_gate_returns_nan(self):
        ret = np.array([0.01, -0.02, 0.03], dtype=np.float64)
        fn = get_metric("sharpe_ratio").compute_fn
        got = np.asarray(fn(ret, min_periods=20)).ravel()[0]
        assert np.isnan(got)

    def test_zero_std_returns_nan(self):
        ret = np.full(30, 0.01, dtype=np.float64)
        fn = get_metric("sharpe_ratio").compute_fn
        got = np.asarray(fn(ret)).ravel()[0]
        assert np.isnan(got)


class TestSortinoRatioOracle:
    def test_matches_hand_computed(self):
        ret = np.array(
            [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.015, -0.005, 0.025,
             0.01, -0.015, 0.02, 0.005, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02,
             0.01, -0.01, 0.015, 0.005, -0.005],
            dtype=np.float64,
        )
        fn = get_metric("sortino_ratio").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        ex = ret
        mean = ex.mean()
        downside = ex[ex < 0]
        downside_dev = np.sqrt(np.mean(downside ** 2))
        expected = mean / downside_dev * np.sqrt(252)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_uses_negative_excess_returns(self):
        ret = np.array(
            [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.015, -0.005, 0.025,
             0.01, -0.015, 0.02, 0.005, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02,
             0.01, -0.01, 0.015, 0.005, -0.005, 0.012, -0.018, 0.022, 0.006,
             -0.012],
            dtype=np.float64,
        )
        fn = get_metric("sortino_ratio").compute_fn
        got = float(np.asarray(fn(ret, risk_free_rate=0.02)).ravel()[0])
        ex = ret - 0.02 / 252
        mean = ex.mean()
        downside = ex[ex < 0]
        assert downside.size > 0
        downside_dev = np.sqrt(np.mean(downside ** 2))
        expected = mean / downside_dev * np.sqrt(252)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_no_negative_returns_returns_nan(self):
        ret = np.linspace(0.001, 0.01, 30, dtype=np.float64)
        fn = get_metric("sortino_ratio").compute_fn
        got = np.asarray(fn(ret)).ravel()[0]
        assert np.isnan(got)

    def test_min_periods_gate_returns_nan(self):
        ret = np.array([0.01, -0.02, 0.03], dtype=np.float64)
        fn = get_metric("sortino_ratio").compute_fn
        got = np.asarray(fn(ret, min_periods=20)).ravel()[0]
        assert np.isnan(got)


class TestWinRateOracle:
    def test_matches_hand_computed(self):
        ret = np.array(
            [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.015, -0.005, 0.025],
            dtype=np.float64,
        )
        fn = get_metric("win_rate").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        wins = np.sum(ret > 0)
        expected = wins / ret.size
        assert got == pytest.approx(expected, rel=1e-12)

    def test_nan_returns_excluded_from_denominator(self):
        ret = np.array([1.0, np.nan, 2.0, -1.0, np.nan], dtype=np.float64)
        fn = get_metric("win_rate").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        assert got == pytest.approx(2.0 / 3.0, rel=1e-12)

    def test_zero_returns_count_as_not_wins(self):
        ret = np.zeros(5, dtype=np.float64)
        fn = get_metric("win_rate").compute_fn
        got = float(np.asarray(fn(ret)).ravel()[0])
        assert got == 0.0

    def test_all_nan_returns_nan(self):
        ret = np.full(5, np.nan, dtype=np.float64)
        fn = get_metric("win_rate").compute_fn
        got = np.asarray(fn(ret)).ravel()[0]
        assert np.isnan(got)


class TestLongShortReturnsOracle:
    def test_matches_hand_computed_day0(self):
        rng = np.random.default_rng(42)
        T, N = 10, 50
        fv = rng.normal(size=(T, N))
        ret = rng.normal(0.01, 0.05, size=(T, N))
        fn = get_metric("long_short_returns").compute_fn
        long_r, short_r, ls_r = fn(fv, ret)
        assert long_r.shape == (T,)
        assert ls_r.shape == (T,)

        # Hand-compute day 0.
        fv0 = fv[0]
        ret0 = ret[0]
        long_cut = np.quantile(fv0, 0.8)
        short_cut = np.quantile(fv0, 0.2)
        long_mean = ret0[fv0 >= long_cut].mean()
        short_mean = ret0[fv0 <= short_cut].mean()
        # Funded portfolio: equal absolute amount per name, 100% gross.
        # Leg means remain standalone bucket returns, not account contributions.
        n_long = np.count_nonzero(fv0 >= long_cut)
        n_short = np.count_nonzero(fv0 <= short_cut)
        expected = (n_long * long_mean - n_short * short_mean) / (n_long + n_short)
        assert ls_r[0] == pytest.approx(expected, rel=1e-12)
        assert long_r[0] == pytest.approx(long_mean, rel=1e-12)
        assert short_r[0] == pytest.approx(short_mean, rel=1e-12)

    def test_3d_factor_values_broadcast(self):
        rng = np.random.default_rng(11)
        T, N, F = 8, 30, 2
        fv = rng.normal(size=(T, N, F))
        ret = rng.normal(0.0, 0.02, size=(T, N))
        fn = get_metric("long_short_returns").compute_fn
        long_r, short_r, ls_r = fn(fv, ret)
        assert long_r.shape == (T, F)
        assert short_r.shape == (T, F)
        assert ls_r.shape == (T, F)

    def test_missing_return_policy_zero_fill(self):
        rng = np.random.default_rng(1)
        T, N = 6, 20
        fv = rng.normal(size=(T, N))
        ret = rng.normal(0.0, 0.02, size=(T, N))
        ret[0, :5] = np.nan
        fn = get_metric("long_short_returns").compute_fn
        long_zf, _, _ = fn(fv, ret, missing_return_policy="zero_fill")
        long_drop, _, _ = fn(fv, ret, missing_return_policy="drop")
        # zero_fill buckets NaN returns as 0 -> different from drop.
        assert not np.isclose(long_zf[0], long_drop[0]) or np.isnan(long_drop[0])
        assert np.isfinite(long_zf[0])

    def test_missing_return_policy_fail_raises(self):
        rng = np.random.default_rng(2)
        T, N = 6, 20
        fv = rng.normal(size=(T, N))
        ret = rng.normal(0.0, 0.02, size=(T, N))
        ret[0, :5] = np.nan
        fn = get_metric("long_short_returns").compute_fn
        with pytest.raises(ValueError):
            fn(fv, ret, missing_return_policy="fail")

    def test_insufficient_assets_returns_nan(self):
        # A single finite asset can never form both buckets: long/short
        # stays NaN for that day (the <2-observation guard).
        fv = np.array([[1.0, np.nan, np.nan]], dtype=np.float64)
        ret = np.array([[0.01, 0.02, 0.03]], dtype=np.float64)
        fn = get_metric("long_short_returns").compute_fn
        long_r, short_r, ls_r = fn(fv, ret)
        assert np.isnan(long_r[0])
        assert np.isnan(short_r[0])
        assert np.isnan(ls_r[0])
