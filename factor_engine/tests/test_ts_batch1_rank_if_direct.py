"""Direct-import parity tests for the Polars-native conditional time-series rank."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import enum

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


class _Metadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ParamSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Canonical base.py `RelationalParamSpec` is `(expression, message=None)`.
class _RelationalParamSpec:
    def __init__(self, expression, message=None):
        self.expression = expression
        self.message = message


class _SeriesOperator:
    def calculate(self, *args, **kwargs):
        return None


class _TransformOperator(_SeriesOperator):
    pass


def _register_operator(**_kwargs):
    return lambda cls: cls


_MissingDefault = type("_MissingDefault", (), {})
_PREV_PKG = sys.modules.get("factor_engine.cleaned_operators")
_PREV_BASE = sys.modules.get("factor_engine.cleaned_operators.base")


# Standalone enum mirroring the canonical ParamRole members referenced by the
# module under test (see cleaned_operators/base.py).  Using a real enum (not a
# SimpleNamespace) keeps the shim semantically faithful to the canonical
# definition.
class _ParamRole(str, enum.Enum):
    HORIZON = "horizon"
    SUPPORT_POLICY = "support_policy"
    SCALAR = "scalar"
    NUMERICAL = "numerical"
    STATE_THRESHOLD = "state_threshold"
    # canonical alias so both spellings resolve (see cleaned_operators/base.py)
    THRESHOLD = "state_threshold"
    ECONOMIC = "economic"
    POLICY = "policy"
    ESTIMATOR_RESOLUTION = "estimator_resolution"
    REGULARIZATION = "regularization"
    MODEL_ORDER = "model_order"
    MISSING_POLICY = "missing_policy"
    MARKET_POLICY = "market_policy"
    SOURCE_POLICY = "source_policy"
    SESSION_POLICY = "session_policy"


base = types.ModuleType("factor_engine.cleaned_operators.base")
base.SeriesOperator = _SeriesOperator
base.register_operator = _register_operator
base.OperatorMetadata = _Metadata
base.ParamSpec = _ParamSpec
base.RelationalParamSpec = _RelationalParamSpec
base.strict_bool_param = lambda value, name: bool(value)
base.strict_int_param = lambda value, name, **kw: int(value)
base.strict_int_runtime = lambda value, name, **kw: int(value)
base._coerce_declared_numeric_string = lambda value, **kw: value
base._normalise_integer = lambda value, name, **kw: int(value)
base.BroadcastSpec = _Metadata
base.validate_operator_call = lambda *a, **k: None
base.ParamRole = _ParamRole
# ts_batch1.py transitively imports cleaned_operators.common.{daily_panel,
# scalar_compare} which need the canonical base.py surface (pandas Operator,
# TwoVarOperator, TransformOperator, ScalarOperator, MISSING); carry them so
# the shimmed package boundary stays import-complete (parity with the real
# base.py surface the registry's framework-identification reads).
base.Operator = _SeriesOperator
base.TwoVarOperator = _SeriesOperator
base.ScalarOperator = _SeriesOperator
base.TransformOperator = _TransformOperator
base.MISSING = _MissingDefault()
_PREV_PKG = sys.modules.get("factor_engine.cleaned_operators")
_PREV_BASE = sys.modules.get("factor_engine.cleaned_operators.base")
package = types.ModuleType("cleaned_operators")
package.__path__ = [str(ROOT / "cleaned_operators")]
sys.modules["cleaned_operators"] = package
sys.modules["factor_engine.cleaned_operators.base"] = base
sys.modules["factor_engine.cleaned_operators"] = package

_SPEC = spec_from_file_location(
    "ts_batch1_rank_direct", ROOT / "cleaned_operators" / "polars_native" / "ts_batch1.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

# R49.5: restore the real modules so this test's fake registration does not
# pollute later tests in the same pytest process (test-isolation hygiene).
if _PREV_PKG is None:
    sys.modules.pop("factor_engine.cleaned_operators", None)
else:
    sys.modules["factor_engine.cleaned_operators"] = _PREV_PKG
if _PREV_BASE is None:
    sys.modules.pop("factor_engine.cleaned_operators.base", None)
else:
    sys.modules["factor_engine.cleaned_operators.base"] = _PREV_BASE


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
