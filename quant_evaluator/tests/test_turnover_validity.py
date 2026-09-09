"""Regression tests for FactorBatch validity in turnover registry adapters."""

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.metrics.registry_adapters import (
    compute_factor_turnover_rate_value,
    compute_turnover_value,
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
