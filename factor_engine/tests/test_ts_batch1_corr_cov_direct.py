"""Focused direct-import regression tests for rolling pairwise statistics."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
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
    HORIZON="horizon",
    SUPPORT_POLICY="support_policy",
    SCALAR="scalar",
    NUMERICAL="numerical",
    STATE_THRESHOLD="state_threshold",
    ESTIMATOR_RESOLUTION="estimator_resolution",
    MODEL_ORDER="model_order",
    MISSING_POLICY="missing_policy",
    MARKET_POLICY="market_policy",
    POLICY="policy",
)
package = types.ModuleType("cleaned_operators")
package.__path__ = [str(ROOT / "cleaned_operators")]
sys.modules["cleaned_operators"] = package
sys.modules["cleaned_operators.base"] = base

_SPEC = spec_from_file_location(
    "ts_batch1_direct", ROOT / "cleaned_operators" / "polars_native" / "ts_batch1.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _series(values):
    return pl.Series("value", values, dtype=pl.Float64)


def test_known_sample_corr_and_cov():
    x = _series([1, 2, 4])
    y = _series([2, 1, 3])

    corr = _MODULE.TSCorrPolarsNative()._calculate_series(x, y, 3).to_numpy()
    cov = _MODULE.TSCovPolarsNative()._calculate_series(x, y, 3).to_numpy()

    np.testing.assert_allclose(corr[-1], 0.6546536707, rtol=0, atol=1e-10)
    np.testing.assert_allclose(cov[-1], 1.0, rtol=0, atol=1e-12)
    assert np.isnan(corr[0])
    np.testing.assert_allclose(corr[1], -1.0, atol=1e-12)
    assert np.isnan(cov[0])


def test_matches_pandas_pairwise_complete_and_ddof():
    x = _series([1.0, np.nan, 4.0, np.inf, 8.0])
    y = _series([2.0, 3.0, np.nan, 5.0, 10.0])
    expected_corr = pd.Series(x.to_numpy()).rolling(3, min_periods=2).corr(
        pd.Series(y.to_numpy())
    ).to_numpy()
    expected_cov = pd.Series(x.to_numpy()).rolling(3, min_periods=2).cov(
        pd.Series(y.to_numpy()), ddof=1
    ).to_numpy()

    actual_corr = _MODULE.TSCorrPolarsNative()._calculate_series(
        x, y, 3, min_periods=2
    ).to_numpy()
    actual_cov = _MODULE.TSCovPolarsNative()._calculate_series(
        x, y, 3, min_periods=2, ddof=1
    ).to_numpy()
    np.testing.assert_allclose(actual_corr, expected_corr, equal_nan=True)
    np.testing.assert_allclose(actual_cov, expected_cov, equal_nan=True)

    cov_population = _MODULE.TSCovPolarsNative()._calculate_series(
        _series([1.0, 2.0, 4.0]), _series([2.0, 1.0, 3.0]), 3, ddof=0
    ).to_numpy()
    np.testing.assert_allclose(cov_population[-1], 2 / 3, atol=1e-12)


def test_constant_inputs_and_future_poison_are_causal():
    x = _series([1.0, 2.0, 4.0, 7.0])
    y = _series([2.0, 1.0, 3.0, 9.0])
    poisoned_y = _series([2.0, 1.0, 3.0, 9.0e300])

    op = _MODULE.TSCorrPolarsNative()
    original = op._calculate_series(x, y, 3).to_numpy()
    poisoned = op._calculate_series(x, poisoned_y, 3).to_numpy()
    np.testing.assert_allclose(original[:3], poisoned[:3], equal_nan=True)

    constant = _MODULE.TSCorrPolarsNative()._calculate_series(
        _series([1.0, 1.0, 1.0]), _series([2.0, 3.0, 4.0]), 3
    ).to_numpy()
    assert np.isnan(constant[-1])


def test_window_larger_than_history_and_min_periods():
    x = _series([1.0, 2.0])
    y = _series([2.0, 4.0])
    result = _MODULE.TSCovPolarsNative()._calculate_series(
        x, y, 5, min_periods=2
    ).to_numpy()
    np.testing.assert_allclose(result[-1], 1.0, atol=1e-12)
    assert np.isnan(
        _MODULE.TSCovPolarsNative()._calculate_series(
            x, y, 5, min_periods=3
        ).to_numpy()[-1]
    )
