import math

import polars as pl
import pytest

from cleaned_operators.common.polars_ts_rolling import (
    TSCorrNative,
    TSCovNative,
    TSRegressionInterceptNative,
    TSRegressionR2Native,
    TSRegressionResidNative,
    TSRegressionSlopeNative,
)


def _oracle(x, y, window, *, ddof=1, min_periods=2):
    """Direct finite-pair rolling oracle for the native pairwise operators."""
    results = {key: [] for key in ("corr", "cov", "slope", "intercept", "resid", "r2")}
    for end in range(len(x)):
        pairs = [
            (left, right)
            for left, right in zip(x[max(0, end - window + 1) : end + 1], y[max(0, end - window + 1) : end + 1])
            if left is not None and right is not None and math.isfinite(left) and math.isfinite(right)
        ]
        if len(pairs) < min_periods:
            for values in results.values():
                values.append(None)
            continue
        xs, ys = zip(*pairs)
        count = len(pairs)
        mean_x = sum(xs) / count
        mean_y = sum(ys) / count
        cross = sum((left - mean_x) * (right - mean_y) for left, right in pairs)
        ss_x = sum((left - mean_x) ** 2 for left in xs)
        ss_y = sum((right - mean_y) ** 2 for right in ys)
        results["cov"].append(cross / (count - ddof) if count > ddof else None)
        if ss_x <= 0 or ss_y <= 0:
            results["corr"].append(None)
            results["slope"].append(None)
            results["intercept"].append(None)
            results["resid"].append(None)
            results["r2"].append(None)
            continue
        slope = cross / ss_x
        intercept = mean_y - slope * mean_x
        results["corr"].append(cross / math.sqrt(ss_x * ss_y))
        results["slope"].append(slope)
        results["intercept"].append(intercept)
        endpoint_x, endpoint_y = x[end], y[end]
        results["resid"].append(
            (endpoint_y - mean_y) - slope * (endpoint_x - mean_x)
            if endpoint_x is not None
            and endpoint_y is not None
            and math.isfinite(endpoint_x)
            and math.isfinite(endpoint_y)
            else None
        )
        results["r2"].append(cross**2 / (ss_x * ss_y))
    return results


def _values(frame):
    return frame["a"].to_list()


def _assert_series(actual, expected, *, abs_tol=1e-12, rel_tol=1e-12):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, abs=abs_tol, rel=rel_tol)


def test_pairwise_direct_window_counterexample_matches_oracle():
    x_values = [1.0, 2.0, 100.0]
    y_values = [1.0, 4.0, 1.0]
    x = pl.DataFrame({"a": x_values})
    y = pl.DataFrame({"a": y_values})
    expected = _oracle(x_values, y_values, window=3)

    _assert_series(_values(TSCorrNative()._calculate_series(x, y, window=3, min_periods=2)), expected["corr"])
    _assert_series(_values(TSCovNative()._calculate_series(x, y, window=3, min_periods=2)), expected["cov"])
    _assert_series(_values(TSRegressionSlopeNative()._calculate_series(y, x, window=3, min_periods=2)), expected["slope"])
    _assert_series(_values(TSRegressionInterceptNative()._calculate_series(y, x, window=3, min_periods=2)), expected["intercept"])
    _assert_series(_values(TSRegressionResidNative()._calculate_series(y, x, window=3, min_periods=2)), expected["resid"])
    _assert_series(_values(TSRegressionR2Native()._calculate_series(y, x, window=3, min_periods=2)), expected["r2"])


def test_pairwise_finite_null_and_constant_contract():
    x_values = [1.0, None, 2.0, math.inf, 3.0, 5.0, 5.0]
    y_values = [2.0, 5.0, math.nan, 4.0, 6.0, 7.0, 9.0]
    x = pl.DataFrame({"a": x_values})
    y = pl.DataFrame({"a": y_values})
    expected = _oracle(x_values, y_values, window=3)

    _assert_series(_values(TSCorrNative()._calculate_series(x, y, window=3, min_periods=2)), expected["corr"])
    _assert_series(_values(TSCovNative()._calculate_series(x, y, window=3, min_periods=2)), expected["cov"])
    _assert_series(_values(TSRegressionSlopeNative()._calculate_series(y, x, window=3, min_periods=2)), expected["slope"])
    _assert_series(_values(TSRegressionInterceptNative()._calculate_series(y, x, window=3, min_periods=2)), expected["intercept"])
    _assert_series(_values(TSRegressionResidNative()._calculate_series(y, x, window=3, min_periods=2)), expected["resid"])
    _assert_series(_values(TSRegressionR2Native()._calculate_series(y, x, window=3, min_periods=2)), expected["r2"])

    constant_x = pl.DataFrame({"a": [2.0, 2.0, 2.0]})
    changing_y = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    for operator in (
        TSCorrNative(),
        TSRegressionSlopeNative(),
        TSRegressionInterceptNative(),
        TSRegressionResidNative(),
        TSRegressionR2Native(),
    ):
        if isinstance(operator, TSCorrNative):
            result = operator._calculate_series(constant_x, changing_y, window=3)
        else:
            result = operator._calculate_series(changing_y, constant_x, window=3)
        assert _values(result) == [None, None, None]



def test_large_offset_values_keep_centered_moments_stable():
    x_values = [1.0e10 + value for value in (1.0, 2.0, 4.0, 8.0)]
    y_values = [1.0e10 + value for value in (3.0, 7.0, 6.0, 11.0)]
    x = pl.DataFrame({"a": x_values})
    y = pl.DataFrame({"a": y_values})
    expected = _oracle(x_values, y_values, window=4)

    _assert_series(_values(TSCorrNative()._calculate_series(x, y, window=4, min_periods=2)), expected["corr"])
    _assert_series(_values(TSCovNative()._calculate_series(x, y, window=4, min_periods=2)), expected["cov"])
    _assert_series(_values(TSRegressionSlopeNative()._calculate_series(y, x, window=4, min_periods=2)), expected["slope"])
    _assert_series(_values(TSRegressionInterceptNative()._calculate_series(y, x, window=4, min_periods=2)), expected["intercept"])
    _assert_series(_values(TSRegressionResidNative()._calculate_series(y, x, window=4, min_periods=2)), expected["resid"])
    _assert_series(_values(TSRegressionR2Native()._calculate_series(y, x, window=4, min_periods=2)), expected["r2"])


def test_large_offset_covariance_matches_centered_numpy_reference() -> None:
    """Pairwise covariance/regression preserve low-order differences at 1e12 offsets."""

    x_values = [1e12 + value for value in (1.0, 2.0, 4.0, 8.0)]
    y_values = [3e12 + value for value in (2.0, 4.0, 8.0, 16.0)]
    x = pl.DataFrame({"a": x_values})
    y = pl.DataFrame({"a": y_values})
    expected_cov = [None, 1.0, 14.0 / 3.0, 115.0 / 6.0]
    expected_slope = [None, 2.0, 2.0, 2.0]

    _assert_series(_values(TSCovNative()._calculate_series(x, y, window=4, min_periods=2)), expected_cov, abs_tol=1e-8, rel_tol=1e-10)
    _assert_series(_values(TSRegressionSlopeNative()._calculate_series(y, x, window=4, min_periods=2)), expected_slope, abs_tol=1e-8, rel_tol=1e-8)


def test_large_offset_anchor_rollout_keeps_final_correlation():
    x_values = [0.0, 10_000_000_001.0, 10_000_000_002.0, 10_000_000_004.0, 10_000_000_008.0]
    y_values = [0.0, 10_000_000_003.0, 10_000_000_007.0, 10_000_000_006.0, 10_000_000_011.0]
    expected = _oracle(x_values, y_values, window=4)

    actual = _values(TSCorrNative()._calculate_series(
        pl.DataFrame({"a": x_values}), pl.DataFrame({"a": y_values}), window=4
    ))
    _assert_series(actual, expected["corr"])


def test_missing_matching_column_fails_closed_for_pairwise_operators():
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    missing = pl.DataFrame({"b": [2.0, 3.0, 4.0]})
    for operator, args in (
        (TSCorrNative(), (left, missing)),
        (TSCovNative(), (left, missing)),
        (TSRegressionSlopeNative(), (left, missing)),
        (TSRegressionInterceptNative(), (left, missing)),
        (TSRegressionResidNative(), (left, missing)),
        (TSRegressionR2Native(), (left, missing)),
    ):
        with pytest.raises(ValueError, match="missing.*column"):
            operator._calculate_series(*args, window=3)


def test_pairwise_matching_columns_retain_aligned_arithmetic():
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    right = pl.DataFrame({"a": [2.0, 4.0, 8.0]})
    result = TSCorrNative()._calculate_series(left, right, window=3, min_periods=2)
    assert result["a"].to_list() == [None, 1.0, pytest.approx(0.9819805060619659)]


def test_bootstrap_module_list_selects_repaired_ts_cov_as_exact_authority():
    import cleaned_operators
    from cleaned_operators import load_all
    from cleaned_operators.common.polars_ts_rolling import TSCovNative as ExpectedTSCovNative
    from cleaned_operators.registry import OperatorRegistry

    assert "cleaned_operators.common.polars_ts_rolling" in cleaned_operators._LOAD_MODULES
    load_all()
    operator = OperatorRegistry.get("ts_cov", backend="polars")
    assert type(operator) is ExpectedTSCovNative
    assert type(operator).__module__ == "cleaned_operators.common.polars_ts_rolling"
    assert type(operator).__name__ == "TSCovNative"
    assert not any(
        entry.get("canonical") == "ts_cov"
        and entry.get("backend") == "polars"
        and entry.get("old_source") == "factor_dsl_np"
        for entry in OperatorRegistry.overwrite_log()
    )


def test_bootstrap_module_list_activates_repaired_native_module():
    import cleaned_operators
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    assert "cleaned_operators.common.polars_ts_rolling" in cleaned_operators._LOAD_MODULES
    load_all()
    operator = OperatorRegistry.get("ts_regression_slope", backend="polars")
    assert type(operator).__module__ == "cleaned_operators.common.polars_ts_rolling"
    assert type(operator).__name__ == "TSRegressionSlopeNative"


def test_covariance_honors_ddof_and_min_periods_with_finite_pairs():
    x_values = [1.0, None, 3.0, 5.0]
    y_values = [2.0, 4.0, 8.0, 10.0]
    result = TSCovNative()._calculate_series(
        pl.DataFrame({"a": x_values}),
        pl.DataFrame({"a": y_values}),
        window=4,
        ddof=0,
        min_periods=2,
    )
    _assert_series(_values(result), _oracle(x_values, y_values, 4, ddof=0, min_periods=2)["cov"])
