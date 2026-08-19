"""Focused tests for the QE-P0-7 registry adapter compute_fns.

Every STABLE metric in the registry must have a callable compute_fn that
returns one well-defined value per factor (P0-6/P0-7 contract). These tests
exercise the two adapters bound in this change (block_bootstrap_ci,
factor_turnover_rate) plus the invariant that no STABLE metric lacks one.
"""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.registry_adapters import (
    compute_block_bootstrap_ci_value,
    compute_factor_turnover_rate_value,
)
from quant_evaluator.registry import (
    MetricStatus,
    list_metrics_by_status,
    get_metric,
)


def _make_batch(T=80, N=60, F=3, seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    factor_ids = tuple(f"factor_{i:03d}" for i in range(F))
    batch = FactorBatch(
        factor_ids=factor_ids,
        time_axis=AxisRef(name="time", dtype="int64", size=T),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N),
        values=values,
    )
    bundle = LabelBundle(
        target_id="return",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    return batch, bundle


class TestBlockBootstrapCiValue:
    def test_returns_scalar_per_factor(self):
        _, _ = _make_batch()
        rng = np.random.default_rng(11)
        ic = rng.normal(size=(100, 3))
        result = compute_block_bootstrap_ci_value(ic, num_bootstrap=100)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # half-width of a CI must be non-negative
        assert np.all(result >= 0)

    def test_min_periods_gates_output(self):
        ic = np.random.default_rng(13).normal(size=(30, 2))  # 30 < min_periods=60
        result = compute_block_bootstrap_ci_value(ic, num_bootstrap=50)
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    def test_half_width_matches_underlying_ci(self):
        from quant_evaluator.metrics.robustness import compute_block_bootstrap_ci
        rng = np.random.default_rng(17)
        ic = rng.normal(size=(80, 1))
        lo, hi = compute_block_bootstrap_ci(
            ic, num_bootstrap=100, random_seed=0
        )
        expected = (hi[0] - lo[0]) / 2.0
        result = compute_block_bootstrap_ci_value(
            ic, num_bootstrap=100, random_seed=0
        )
        assert result[0] == pytest.approx(expected)


class TestFactorTurnoverRateValue:
    def test_returns_scalar_per_factor(self):
        batch, _ = _make_batch()
        result = compute_factor_turnover_rate_value(batch)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # membership turnover is a fraction of changed positions
        assert np.all((result >= 0) & (result <= 1))

    def test_min_periods_gates_output(self):
        batch, _ = _make_batch(T=10)  # 9 turnover obs < min_periods=30
        result = compute_factor_turnover_rate_value(batch)
        assert np.all(np.isnan(result))

    def test_deterministic(self):
        batch, _ = _make_batch()
        a = compute_factor_turnover_rate_value(batch)
        b = compute_factor_turnover_rate_value(batch)
        np.testing.assert_array_equal(a, b)


class TestStableMetricContract:
    def test_no_stable_metric_without_compute_fn(self):
        missing = [
            name
            for name in list_metrics_by_status(MetricStatus.STABLE)
            if get_metric(name).compute_fn is None
        ]
        assert missing == [], f"STABLE metrics without compute_fn: {missing}"

    def test_newly_bound_metrics_callable(self):
        for name in ("block_bootstrap_ci", "factor_turnover_rate"):
            spec = get_metric(name)
            assert callable(spec.compute_fn), name
