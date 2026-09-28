"""Regression tests for FactorBatch validity in turnover registry adapters."""

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.metrics.registry_adapters import (
    compute_factor_turnover_rate_value,
    compute_turnover_value,
)
from quant_evaluator.metrics.turnover import (
    _estimate_turnover_from_ranks_reference,
    estimate_turnover_from_ranks,
)


def _batch(values: np.ndarray, validity: np.ndarray) -> FactorBatch:
    return FactorBatch(
        factor_ids=("factor",),
        time_axis=AxisRef("time", "int64", values.shape[0], np.arange(values.shape[0])),
        asset_axis=AxisRef(
            "asset", "str", values.shape[1], np.array([f"A{i}" for i in range(values.shape[1])])
        ),
        values=values[:, :, None],
        validity=validity[:, :, None],
    )


def test_turnover_adapters_mask_invalid_finite_poison_values() -> None:
    clean = np.vstack(
        [
            np.arange(12, dtype=np.float64),
            np.roll(np.arange(12, dtype=np.float64), 2),
            np.roll(np.arange(12, dtype=np.float64), -1),
        ]
    )
    validity = np.ones_like(clean, dtype=bool)
    validity[:, 11] = False

    baseline = clean.copy()
    baseline[:, 11] = np.nan
    poisoned = clean.copy()
    poisoned[:, 11] = np.array([1.0e300, -1.0e300, 1.0e300])

    clean_batch = _batch(baseline, validity)
    poisoned_batch = _batch(poisoned, validity)

    np.testing.assert_allclose(
        compute_turnover_value(poisoned_batch, min_periods=2),
        compute_turnover_value(clean_batch, min_periods=2),
        equal_nan=True,
    )
    membership = compute_factor_turnover_rate_value(
        poisoned_batch, min_periods=1, quantile=0.8
    )
    assert np.isfinite(membership).all()
    np.testing.assert_allclose(
        membership,
        compute_factor_turnover_rate_value(clean_batch, min_periods=1, quantile=0.8),
        equal_nan=True,
    )


def test_rank_turnover_sentinel_stays_above_large_finite_signal() -> None:
    # Adding 1 to either signal rounds back to the same float64 value. A
    # missing-value sentinel tied with that maximum changes every weight.
    for high in (1.0e20, np.finfo(np.float64).max):
        first = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., high, np.nan])
        second = first.copy()
        second[8], second[9] = second[9], second[8]
        values = np.stack((first, second))
        validity = np.ones_like(values, dtype=bool)
        validity[:, -1] = False
        batch = _batch(values, validity)

        actual = estimate_turnover_from_ranks(batch)
        oracle = _estimate_turnover_from_ranks_reference(batch)
        np.testing.assert_array_equal(actual, oracle)
        np.testing.assert_allclose(actual[1, 0], 1.0 / 55.0, rtol=0, atol=1e-15)


def test_rank_turnover_ignores_finite_values_marked_invalid() -> None:
    first = np.arange(11, dtype=np.float64)
    second = np.roll(first, 1)
    values = np.stack((first, second))
    validity = np.ones_like(values, dtype=bool)
    validity[:, -1] = False

    baseline = values.copy()
    baseline[:, -1] = np.nan
    poisoned = values.copy()
    poisoned[:, -1] = [1.0e6, -1.0e6]

    clean_batch = _batch(baseline, validity)
    poisoned_batch = _batch(poisoned, validity)
    clean = estimate_turnover_from_ranks(clean_batch)
    actual = estimate_turnover_from_ranks(poisoned_batch)

    np.testing.assert_array_equal(actual, clean)
    np.testing.assert_array_equal(actual, _estimate_turnover_from_ranks_reference(poisoned_batch))
    np.testing.assert_allclose(actual[1, 0], 9.0 / 55.0, rtol=0, atol=1e-15)
