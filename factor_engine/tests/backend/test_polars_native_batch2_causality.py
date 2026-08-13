"""Causality checks for the repaired native Polars pivot operators."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "cleaned_operators"
    / "polars_native"
    / "ts_advanced_batch2.py"
)
_SPEC = importlib.util.spec_from_file_location("_ts_advanced_batch2_causality", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TSLastPivotHighPolarsNative = _MODULE.TSLastPivotHighPolarsNative
TSLastPivotLowPolarsNative = _MODULE.TSLastPivotLowPolarsNative


@pytest.mark.parametrize(
    ("operator_type", "values", "pivot_value"),
    [
        (TSLastPivotHighPolarsNative, [1, 2, 3, 7, 3, 2, 1, 4], 7.0),
        (TSLastPivotLowPolarsNative, [7, 6, 5, 1, 5, 6, 7, 4], 1.0),
    ],
)
def test_last_pivot_is_emitted_only_at_confirmation(
    operator_type, values, pivot_value
) -> None:
    result = operator_type()._calculate_series(
        pl.Series("x", values, dtype=pl.Float64), window=3
    ).to_numpy()

    assert np.isnan(result[:6]).all()
    np.testing.assert_allclose(result[6:], [pivot_value, pivot_value])


@pytest.mark.parametrize(
    ("operator_type", "values"),
    [
        (TSLastPivotHighPolarsNative, [1, 3, 2, 5, 2, 1, 4, 7, 3, 2, 6, 1]),
        (TSLastPivotLowPolarsNative, [8, 6, 7, 3, 6, 8, 5, 2, 6, 7, 3, 9]),
    ],
)
def test_last_pivot_prefix_is_invariant_under_future_poison(
    operator_type, values
) -> None:
    split = 9
    original = pl.Series("x", values, dtype=pl.Float64)
    poisoned = original.to_numpy().copy()
    poisoned[split:] = [1e12, -1e12, 1e12]
    operator = operator_type()

    baseline = operator._calculate_series(original, window=3).to_numpy()
    poison_result = operator._calculate_series(
        pl.Series("x", poisoned), window=3
    ).to_numpy()
    prefix_result = operator._calculate_series(original.head(split), window=3).to_numpy()

    np.testing.assert_allclose(baseline[:split], poison_result[:split], equal_nan=True)
    np.testing.assert_allclose(baseline[:split], prefix_result, equal_nan=True)
