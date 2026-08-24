import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.common.time_series import TSZScore, TSZScorePolars
from factor_engine.backend.operator_errors import OperatorParameterError


def _load_module(monkeypatch):
    import factor_engine.cleaned_operators.base as base
    import factor_engine.cleaned_operators.base_polars as base_polars

    class _ImportProbeParamRole:
        pass

    for role in ParamRole:
        setattr(_ImportProbeParamRole, role.name, role)
    _ImportProbeParamRole.THRESHOLD = ParamRole.STATE_THRESHOLD
    monkeypatch.setattr(base, "ParamRole", _ImportProbeParamRole)
    monkeypatch.setattr(base_polars, "ParamRole", _ImportProbeParamRole, raising=False)
    monkeypatch.setattr(base_polars, "register_operator", lambda **_kwargs: lambda cls: cls)
    path = Path(__file__).resolve().parents[2] / "cleaned_operators/common/polars_ts_stats.py"
    spec = importlib.util.spec_from_file_location("_window_local_polars_ts_stats", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _windows(values: np.ndarray, window: int):
    for i in range(values.size):
        yield values[max(0, i - window + 1) : i + 1]


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _oracle_skew(values: np.ndarray, window: int) -> np.ndarray:
    out = []
    for chunk in _windows(values, window):
        v = _finite(chunk)
        if v.size < window or v.size < 3:
            out.append(np.nan)
            continue
        centered = v - np.mean(v)
        ss = np.sum(centered**2)
        out.append(np.nan if ss == 0 else v.size * np.sum(centered**3) / ((v.size - 1) * (v.size - 2) * (ss / (v.size - 1)) ** 1.5))
    return np.asarray(out)


def _oracle_trim(values: np.ndarray, window: int, trim: float) -> np.ndarray:
    out = []
    for chunk in _windows(values, window):
        v = np.sort(_finite(chunk))
        if v.size < window:
            out.append(np.nan)
            continue
        cut = int(np.floor(trim * v.size))
        out.append(np.nan if v.size == 0 or cut * 2 >= v.size else np.mean(v[cut : v.size - cut]))
    return np.asarray(out)


def _oracle_qn(values: np.ndarray, window: int) -> np.ndarray:
    corrections = {2: 0.399356, 3: 0.99365, 4: 0.51321, 5: 0.84401, 6: 0.61220, 7: 0.85877, 8: 0.66993, 9: 0.87344, 10: 0.72014, 11: 0.88906, 12: 0.75743}
    out = []
    for chunk in _windows(values, window):
        v = _finite(chunk)
        n = v.size
        if n < window or n < 2:
            out.append(np.nan)
            continue
        h = n // 2 + 1
        k = h * (h - 1) // 2
        diffs = np.asarray([abs(v[i] - v[j]) for i in range(n) for j in range(i + 1, n)])
        if n in corrections:
            dn = corrections[n]
        elif n % 2:
            dn = 1 / (1 + 1.60188 / n - 2.1284 / n**2 - 5.172 / n**3)
        else:
            dn = 1 / (1 + 3.67561 / n + 1.9654 / n**2 + 6.987 / n**3 - 77 / n**4)
        out.append(2.21914446598508 * dn * np.partition(diffs, k - 1)[k - 1])
    return np.asarray(out)


def _oracle_es(values: np.ndarray, window: int, alpha: float) -> np.ndarray:
    out = []
    for chunk in _windows(values, window):
        v = _finite(chunk)
        if v.size < window or v.size == 0:
            out.append(np.nan)
            continue
        threshold = np.quantile(v, alpha)
        out.append(np.mean(v[v <= threshold]))
    return np.asarray(out)


def _zscore_oracle(values: np.ndarray, window: int) -> np.ndarray:
    result = []
    for end in range(len(values)):
        raw = values[max(0, end - window + 1) : end + 1]
        finite = raw[np.isfinite(raw)]
        current = values[end]
        if not np.isfinite(current) or finite.size < 2:
            result.append(np.nan)
            continue
        std = float(np.std(finite, ddof=1))
        result.append(0.0 if std == 0.0 else float((current - np.mean(finite)) / std))
    return np.asarray(result)


def test_active_ts_zscore_polars_matches_pandas_reference_for_nonfinite_values():
    values = np.asarray([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 4.0, 4.0, 5.0])
    result = TSZScorePolars().calculate(
        pl.DataFrame({"x": values}), window=5
    )["x"].to_numpy()
    expected = _zscore_oracle(values, 5)
    np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert np.isnan(result[2])
    assert np.isnan(result[4])
    assert np.isnan(result[5])
    assert result[6] == 0.0


def test_active_ts_zscore_polars_matches_pandas_warmup_and_zero_std():
    values = np.asarray([7.0, 7.0, 7.0, 8.0])
    result = TSZScorePolars().calculate(
        pl.DataFrame({"x": values}), window=3
    )["x"].to_numpy()
    expected = _zscore_oracle(values, 3)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    assert np.isnan(result[0])
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_active_ts_zscore_polars_zero_std_does_not_fill_nonfinite_current():
    values = np.asarray([1.0, 1.0, np.inf])
    pandas_result = TSZScore().calculate(
        pd.DataFrame({"x": values}), window=3
    )["x"].to_numpy()
    polars_result = TSZScorePolars().calculate(
        pl.DataFrame({"x": values}), window=3
    )["x"].to_numpy()
    assert np.isnan(pandas_result[-1])
    assert np.isnan(polars_result[-1])


def test_active_ts_zscore_polars_public_api_matches_pandas_reference():
    assert TSZScorePolars.metadata.param_names == ["x", "window"]
    with pytest.raises(OperatorParameterError, match="min_periods"):
        TSZScorePolars().calculate(
            pl.DataFrame({"x": [1.0, 2.0]}), window=2, min_periods=2
        )


def test_ts_zscore_prior_results_are_future_invariant():
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    first = TSZScorePolars().calculate(
        pl.DataFrame({"x": values}), window=3
    )["x"].to_numpy()
    extended = TSZScorePolars().calculate(
        pl.DataFrame({"x": np.r_[values, 10_000.0]}), window=3
    )["x"].to_numpy()
    np.testing.assert_array_equal(first, extended[: values.size])


def test_legacy_ts_zscore_native_rejects_infinite_current_value(monkeypatch):
    module = _load_module(monkeypatch)
    operator = module.TSZScoreNative()
    result = operator.calculate(
        pl.DataFrame({"x": [1.0, 2.0, np.inf]}), window=3
    )["x"].to_numpy()
    assert np.isnan(result[-1])


def test_bootstrap_does_not_load_window_local_ts_zscore_native():
    code = """
from factor_engine.cleaned_operators import _LOAD_MODULES, load_all
load_all()
assert "factor_engine.cleaned_operators.common.polars_ts_stats" not in _LOAD_MODULES
import sys
assert "factor_engine.cleaned_operators.common.polars_ts_stats" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


@pytest.fixture
def stats_module(monkeypatch):
    return _load_module(monkeypatch)


@pytest.mark.parametrize(
    ("operator_name", "kwargs", "oracle"),
    [
        ("TSSkewNative", {"window": 10}, _oracle_skew),
        ("TSTrimmedMeanNative", {"window": 10, "trim_pct": 0.25}, lambda x, window, trim_pct: _oracle_trim(x, window, trim_pct)),
        ("TSQnScaleNative", {"window": 10}, _oracle_qn),
        ("TSExpectedShortfallNative", {"window": 10, "alpha": 0.25}, lambda x, window, alpha: _oracle_es(x, window, alpha)),
    ],
)
def test_registered_common_polars_stats_match_independent_window_oracle(operator_name, kwargs, oracle, stats_module):
    operator = getattr(stats_module, operator_name)
    values = np.asarray([1.0, 8.0, 2.0, 100.0, 3.0, np.nan, 4.0, np.inf, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0])
    frame = pl.DataFrame({"x": values})
    result = operator()._calculate_series(frame, **kwargs)["x"].to_numpy()
    expected = oracle(values, **kwargs)
    np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_window_local_center_and_threshold_are_not_pointwise_rolling_approximations(stats_module):
    TSSkewNative = stats_module.TSSkewNative
    TSExpectedShortfallNative = stats_module.TSExpectedShortfallNative
    values = np.asarray([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])
    frame = pl.DataFrame({"x": values})
    skew = TSSkewNative()._calculate_series(frame, window=10)["x"].to_numpy()
    expected = _oracle_skew(values, 10)
    np.testing.assert_allclose(skew, expected, equal_nan=True)

    es_values = np.asarray([-10.0, -2.0, 0.0, 1.0, 2.0, 100.0])
    es = TSExpectedShortfallNative()._calculate_series(pl.DataFrame({"x": es_values}), window=10, alpha=0.25)["x"].to_numpy()
    np.testing.assert_allclose(es, _oracle_es(es_values, 10, 0.25), equal_nan=True)


def test_constants_ties_and_invalid_parameters_fail_closed(stats_module):
    TSSkewNative = stats_module.TSSkewNative
    TSQnScaleNative = stats_module.TSQnScaleNative
    TSTrimmedMeanNative = stats_module.TSTrimmedMeanNative
    TSExpectedShortfallNative = stats_module.TSExpectedShortfallNative
    constant = pl.DataFrame({"x": [2.0] * 10})
    assert math.isnan(TSSkewNative()._calculate_series(constant, window=10)["x"][-1])
    assert TSQnScaleNative()._calculate_series(constant, window=10)["x"][-1] == 0.0
    assert TSTrimmedMeanNative()._calculate_series(constant, window=10, trim_pct=0.25)["x"][-1] == 2.0

    with pytest.raises(ValueError):
        TSExpectedShortfallNative()._calculate_series(constant, window=10, alpha=0.0)
    with pytest.raises(ValueError):
        TSTrimmedMeanNative()._calculate_series(constant, window=10, trim_pct=0.5)


def test_future_values_do_not_change_prior_result(stats_module):
    TSTrimmedMeanNative = stats_module.TSTrimmedMeanNative
    prefix = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = TSTrimmedMeanNative()._calculate_series(pl.DataFrame({"x": prefix}), window=5, trim_pct=0.25)["x"].to_numpy()
    extended = TSTrimmedMeanNative()._calculate_series(pl.DataFrame({"x": np.r_[prefix, 10_000.0]}), window=5, trim_pct=0.25)["x"].to_numpy()
    np.testing.assert_array_equal(first, extended[: prefix.size])
