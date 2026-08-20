"""Direct-import parity tests for the Polars-native conditional time-series rank."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


class _Metadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ParamSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SeriesOperator:
    pass


def _register_operator(**_kwargs):
    return lambda cls: cls


base = types.ModuleType("cleaned_operators.base")
base.SeriesOperator = _SeriesOperator
base.register_operator = _register_operator
base.OperatorMetadata = _Metadata
base.ParamSpec = _ParamSpec
base.ParamRole = types.SimpleNamespace(
    HORIZON="horizon", SUPPORT_POLICY="support_policy", SCALAR="scalar"
)
package = types.ModuleType("cleaned_operators")
package.__path__ = [str(ROOT / "cleaned_operators")]
sys.modules["cleaned_operators"] = package
sys.modules["cleaned_operators.base"] = base

_SPEC = spec_from_file_location(
    "ts_batch1_rank_direct", ROOT / "cleaned_operators" / "polars_native" / "ts_batch1.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _series(values):
    return pl.Series("value", values, dtype=pl.Float64)


def _oracle(values, condition, window, min_periods=5):
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    out = []
    for i, current in enumerate(values):
        if current is None or not np.isfinite(current):
            out.append(np.nan)
            continue
        selected = [
            value
            for value, flag in zip(values[max(0, i - w + 1) : i + 1], condition[max(0, i - w + 1) : i + 1])
            if flag is not None
            and np.isfinite(flag)
            and flag == 1.0
            and value is not None
            and np.isfinite(value)
        ]
        if len(selected) < mp:
            out.append(np.nan)
            continue
        less = sum(value < current for value in selected)
        equal = sum(value == current for value in selected)
        out.append((less + 0.5 * equal) / len(selected))
    return np.asarray(out, dtype=float)


def test_known_outputs_and_heavy_ties_match_canonical_oracle():
    values = [3.0, 1.0, 2.0, 2.0, 2.0, 4.0]
    condition = [1.0] * len(values)
    expected = _oracle(values, condition, window=4, min_periods=2)
    actual = _MODULE.TSRankIfPolarsNative()._calculate_series(
        _series(values), _series(condition), 4, min_periods=2
    ).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(actual[3], 0.5)


def test_condition_false_and_null_exclude_rows_and_false_current_is_null():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    condition = [1.0, 0.0, None, 1.0, 1.0]
    expected = _oracle(values, condition, window=5, min_periods=2)
    actual = _MODULE.TSRankIfPolarsNative()._calculate_series(
        _series(values), _series(condition), 5, min_periods=2
    ).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])


def test_nan_and_inf_are_nonfinite_and_window_one_uses_canonical_floor():
    values = [1.0, np.nan, np.inf, 2.0, 3.0]
    condition = [1.0] * len(values)
    expected = _oracle(values, condition, window=1, min_periods=2)
    actual = _MODULE.TSRankIfPolarsNative()._calculate_series(
        _series(values), _series(condition), 1, min_periods=2
    ).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_window_longer_than_history_and_future_poison_are_causal():
    values = [5.0, 1.0, 3.0, 2.0]
    condition = [1.0] * len(values)
    poisoned = values + [1.0e300]
    expected = _oracle(values, condition, window=20, min_periods=2)
    actual = _MODULE.TSRankIfPolarsNative()._calculate_series(
        _series(values), _series(condition), 20, min_periods=2
    ).to_numpy()
    poisoned_prefix = _MODULE.TSRankIfPolarsNative()._calculate_series(
        _series(poisoned), _series(condition + [1.0]), 20, min_periods=2
    ).to_numpy()[: len(values)]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(poisoned_prefix, actual, equal_nan=True)
    assert np.isfinite(actual[-1])
