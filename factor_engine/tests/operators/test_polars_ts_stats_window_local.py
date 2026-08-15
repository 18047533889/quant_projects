import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

from cleaned_operators.base import ParamRole


def _load_module(monkeypatch):
    import cleaned_operators.base as base
    import cleaned_operators.base_polars as base_polars

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
    sys.modules[spec.name] = module
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


def _oracle_zscore(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    out = []
    for chunk in _windows(values, window):
        finite = _finite(chunk)
        current = chunk[-1]
        if finite.size < min_periods or np.isnan(current) or finite.size < 2:
            out.append(np.nan)
            continue
        std = np.std(finite, ddof=1)
        out.append(0.0 if std == 0.0 else (current - np.mean(finite)) / std)
    return np.asarray(out)


def test_registered_ts_zscore_matches_pandas_semantics_for_nonfinite_values(stats_module):
    values = np.asarray([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 4.0, 4.0, 5.0])
    frame = pl.DataFrame({"x": values})
    result = stats_module.TSZScoreNative()._calculate_series(
        frame, window=5, min_periods=2
    )["x"].to_numpy()
    expected = _oracle_zscore(values, 5, 2)
    np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert np.isnan(result[2])
    assert np.isposinf(result[4])
    assert np.isneginf(result[5])
    assert result[6] == 0.0


def test_registered_ts_zscore_honors_warmup_and_zero_std(stats_module):
    values = np.asarray([7.0, 7.0, 7.0, 8.0])
    result = stats_module.TSZScoreNative()._calculate_series(
        pl.DataFrame({"x": values}), window=3, min_periods=2
    )["x"].to_numpy()
    expected = _oracle_zscore(values, 3, 2)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    assert np.isnan(result[0])
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_registered_ts_zscore_rejects_invalid_min_periods(stats_module):
    with pytest.raises(ValueError, match="min_periods"):
        stats_module.TSZScoreNative()._calculate_series(
            pl.DataFrame({"x": [1.0, 2.0]}), window=2, min_periods=3
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
    prefix = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = TSTrimmedMeanNative()._calculate_series(pl.DataFrame({"x": prefix}), window=5, trim_pct=0.25)["x"].to_numpy()
    extended = TSTrimmedMeanNative()._calculate_series(pl.DataFrame({"x": np.r_[prefix, 10_000.0]}), window=5, trim_pct=0.25)["x"].to_numpy()
    np.testing.assert_array_equal(first, extended[: prefix.size])
