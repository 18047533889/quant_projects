"""Regression tests for once-per-panel quantile numeric error bounds."""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
import quant_evaluator.metrics.quantile as quantile


def _panel(labels):
    t, n, f = labels.shape[0], labels.shape[1], 3
    times = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = np.broadcast_to(np.arange(n, dtype=np.float64)[None, :, None],
                             (t, n, f)).copy()
    validity = np.ones(values.shape, dtype=bool)
    validity[0, 0, 1] = False
    validity[-1, -1, 2] = False
    batch = FactorBatch(("f0", "f1", "f2"), times, assets, values,
                        validity=validity)
    bundle = LabelBundle(
        "bound-reuse", labels, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=assets,
    )
    return batch, bundle


@pytest.mark.parametrize("extreme", [False, True], ids=["normal", "extreme-cancellation"])
def test_compute_quantile_returns_reuses_one_bound_panel_and_preserves_results(
        monkeypatch, extreme):
    # A per-factor recomputation regresses this contract; incorrect sharing can
    # change risky-bucket repairs and therefore the returned bits.
    t, n = 3, 30
    if extreme:
        labels = np.tile(np.array([1e308, 1e-100, -1e308, -1e-100, 4.0, -4.0]),
                         (t, n // 6)).astype(np.float64)
    else:
        labels = np.linspace(-0.02, 0.02, t * n).reshape(t, n)
    labels[1, 2] = np.nan
    labels[2, 4] = np.inf
    batch, bundle = _panel(labels)

    original_bounds = quantile.label_sum_error_bounds
    calls = []

    def counted_bounds(panel):
        calls.append(panel)
        return original_bounds(panel)

    monkeypatch.setattr(quantile, "label_sum_error_bounds", counted_bounds)
    got_returns, got_counts = quantile.compute_quantile_returns(
        batch, bundle, n_quantiles=5, min_assets=1,
    )
    assert len(calls) == 1

    # Preserve an independent legacy-style fallback reference: each factor is
    # assigned and aggregated separately, with its own bounds computation.
    expected_returns = np.full_like(got_returns, np.nan)
    expected_counts = np.zeros_like(got_counts)
    normalized, label_validity = quantile.normalize_label_panel(bundle, n)
    if label_validity is not None:
        normalized = np.where(label_validity, normalized, np.nan)
    for f in range(batch.num_factors):
        assignments = quantile.assign_quantiles_batch(
            np.where(batch.validity[:, :, f], batch.values[:, :, f], np.nan),
            n_quantiles=5,
        )
        quantile._aggregate_quantile_assignments(
            assignments, normalized, n_quantiles=5, min_assets=1,
            out=(expected_returns[:, :, f:f + 1], expected_counts[:, :, f:f + 1]),
        )
    np.testing.assert_array_equal(got_counts, expected_counts)
    np.testing.assert_array_equal(got_returns, expected_returns)


def test_aggregate_quantile_assignments_keeps_standalone_bounds_fallback(monkeypatch):
    assignments = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=np.int32)
    labels = np.array([[1e308, 1e-100, -1e308, -1e-100]], dtype=np.float64)
    original_bounds = quantile.label_sum_error_bounds
    calls = []

    def counted_bounds(panel):
        calls.append(panel)
        return original_bounds(panel)

    monkeypatch.setattr(quantile, "label_sum_error_bounds", counted_bounds)
    returns, counts = quantile._aggregate_quantile_assignments(
        assignments, labels, n_quantiles=2, min_assets=1,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(counts, np.array([[[2, 2], [2, 2]]], dtype=np.int32))
    assert returns.shape == (1, 2, 2)
