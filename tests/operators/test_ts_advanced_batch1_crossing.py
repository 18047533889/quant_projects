# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.crossing import _crossing_acceleration_series
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _native_operator_class():
    module_path = (
        Path(__file__).parents[2]
        / "cleaned_operators"
        / "polars_native"
        / "ts_advanced_batch1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_ts_advanced_batch1_crossing", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.object(OperatorRegistry, "register"):
        spec.loader.exec_module(module)
    return module.TSCrossingAccelerationPolarsNative


def test_crossing_acceleration_matches_canonical_reference() -> None:
    x = pd.Series([1.0, 1.5, 3.0, 0.0, -2.0, 2.0], name="x")
    y = pd.Series([2.0, 2.0, 2.0, 1.0, 0.0, 0.0], name="y")

    actual = _native_operator_class()().calculate(x, y, window=4)
    expected = _crossing_acceleration_series(
        x.to_numpy(dtype=float).reshape(-1, 1),
        y.to_numpy(dtype=float).reshape(-1, 1),
        4,
    )[:, 0]

    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
    assert actual.index.equals(x.index)
    assert actual.name == x.name


def test_crossing_acceleration_prefix_poison_preserves_values_and_nan_mask() -> None:
    x = pd.Series(
        [-2.0, -1.0, 1.0, 2.0, np.nan, -1.0, 2.0, 3.0, -2.0, 1.0],
        name="x",
    )
    y = pd.Series(np.zeros(len(x)), name="y")
    split = 6
    op = _native_operator_class()()

    baseline = op.calculate(x, y, window=4)
    poisoned_x = x.copy()
    poisoned_y = y.copy()
    poisoned_x.iloc[split:] = [1e100, -1e100, np.nan, 1e100]
    poisoned_y.iloc[split:] = [-1e100, 1e100, 1e100, np.nan]
    poisoned = op.calculate(poisoned_x, poisoned_y, window=4)

    baseline_prefix = baseline.iloc[:split]
    poisoned_prefix = poisoned.iloc[:split]
    np.testing.assert_array_equal(
        baseline_prefix.isna().to_numpy(),
        poisoned_prefix.isna().to_numpy(),
    )
    np.testing.assert_allclose(
        baseline_prefix.to_numpy(),
        poisoned_prefix.to_numpy(),
        equal_nan=True,
    )
    assert baseline_prefix.isna().any()
