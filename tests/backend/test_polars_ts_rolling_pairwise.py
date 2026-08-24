import math

import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_ts_rolling import (
    TSCorrNative,
    TSCovNative,
    TSRegressionInterceptNative,
    TSRegressionR2Native,
    TSRegressionResidNative,
    TSRegressionSlopeNative,
    TSEwmCorrNative,
    TSEwmCovNative,
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


def test_trend_slope_uses_local_positions_for_prefix_and_missing_values():
    """Trend slope matches direct trailing-window OLS on finite observations."""
    from factor_engine.cleaned_operators.common.polars_ts_rolling import TSTrendSlopeNative

    values = [1.0, 3.0, None, 7.0, float("nan"), 11.0]
    window = 4
    expected = []
    for end in range(len(values)):
        start = max(0, end - window + 1)
        pairs = [
            (index, value)
            for index, value in enumerate(values[start : end + 1], start=start)
            if value is not None and math.isfinite(value)
        ]
        if len(pairs) < 2:
            expected.append(None)
            continue
        times, observations = zip(*pairs)
        time_mean = sum(times) / len(times)
        value_mean = sum(observations) / len(observations)
        centered_time = [time - time_mean for time in times]
        denominator = sum(delta * delta for delta in centered_time)
        expected.append(
            sum(delta * (value - value_mean) for delta, value in zip(centered_time, observations))
            / denominator
            if denominator
            else None
        )

    actual = TSTrendSlopeNative()._calculate_series(
        pl.DataFrame({"a": values}), window=window
    )["a"].to_list()
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want)


def test_trend_slope_avoids_row_index_column_collision():
    from factor_engine.cleaned_operators.common.polars_ts_rolling import TSTrendSlopeNative

    actual = TSTrendSlopeNative()._calculate_series(
        pl.DataFrame({"__ts_row": [10.0, 12.0, 15.0], "a": [1.0, 3.0, 6.0]}),
        window=3,
    )

    assert actual.columns == ["__ts_row", "a"]
    assert actual["a"].to_list() == [None, pytest.approx(2.0), pytest.approx(2.5)]


def test_trend_slope_excludes_positive_and_negative_infinity():
    from factor_engine.cleaned_operators.common.polars_ts_rolling import TSTrendSlopeNative

    actual = TSTrendSlopeNative()._calculate_series(
        pl.DataFrame({"a": [1.0, float("inf"), 3.0, float("-inf"), 5.0]}),
        window=5,
    )["a"].to_list()

    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(1.0)
    assert actual[3] == pytest.approx(1.0)
    assert actual[4] == pytest.approx(1.0)


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


def test_ewm_corr_missing_matching_column_fails_closed():
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    missing = pl.DataFrame({"b": [2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="missing.*column"):
        TSEwmCorrNative()._calculate_series(left, missing, window=3)


def test_ewm_cov_missing_matching_column_fails_closed():
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    missing = pl.DataFrame({"b": [2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="missing.*column"):
        TSEwmCovNative()._calculate_series(left, missing, window=3)


# ---------------------------------------------------------------------------
# EWM pairwise oracles (R20-P0-EWM-PAIRWISE)
#
# Reference: pandas ``Series.ewm(span=window, adjust=False).corr/cov`` — the
# same reference the polars long emitter pins.  The oracles below are computed
# INDEPENDENTLY of the kernel (the kernel pre-masks Inf and pairwise-masks
# invalid rows before delegating; the oracle replicates that contract on raw
# lists) so a kernel regression against pandas fails loudly.
# ---------------------------------------------------------------------------


def _ewm_oracle(x_values, y_values, window, *, corr, bias=False):
    """Brute-force pandas ewm().corr()/cov() oracle (independent of the kernel).

    Replicates the pairwise-finite contract on raw lists: a row with either
    operand invalid (None/NaN/Inf) is dropped from both streams, Inf is masked
    to NaN, and pandas NaN output maps to None.
    """
    import math

    import numpy as np
    import pandas as pd

    def _finite(value):
        if value is None:
            return math.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.nan
        return value if math.isfinite(value) else math.nan

    xs = pd.Series([_finite(v) for v in x_values], dtype=float)
    ys = pd.Series([_finite(v) for v in y_values], dtype=float)
    pair = xs.notna() & ys.notna()
    xs = xs.where(pair, np.nan)
    ys = ys.where(pair, np.nan)
    ewm = xs.ewm(span=window, adjust=False, ignore_na=False, min_periods=2)
    out = ewm.corr(ys) if corr else ewm.cov(ys, bias=bias)
    return [None if pd.isna(v) else float(v) for v in out.tolist()]


def _ewm_oracle_alpha(x_values, y_values, alpha, *, corr):
    """Same oracle addressed by alpha (span/alpha alias guard)."""
    import math

    import numpy as np
    import pandas as pd

    def _finite(value):
        if value is None:
            return math.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.nan
        return value if math.isfinite(value) else math.nan

    xs = pd.Series([_finite(v) for v in x_values], dtype=float)
    ys = pd.Series([_finite(v) for v in y_values], dtype=float)
    pair = xs.notna() & ys.notna()
    xs = xs.where(pair, np.nan)
    ys = ys.where(pair, np.nan)
    ewm = xs.ewm(alpha=alpha, adjust=False, ignore_na=False, min_periods=2)
    out = ewm.corr(ys) if corr else ewm.cov(ys)
    return [None if pd.isna(v) else float(v) for v in out.tolist()]


def _ewm_oracle_var(x_values, window):
    """Brute-force pandas ewm().var() oracle (cov(x,x) identity guard)."""
    import math

    import numpy as np
    import pandas as pd

    def _finite(value):
        if value is None:
            return math.nan
        return value if math.isfinite(value) else math.nan

    xs = pd.Series([_finite(v) for v in x_values], dtype=float)
    out = xs.ewm(span=window, adjust=False, ignore_na=False, min_periods=2).var()
    return [None if pd.isna(v) else float(v) for v in out.tolist()]


def _assert_ewm(actual, expected, *, rel_tol=1e-10, abs_tol=1e-12):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None, f"expected None, got {got!r}"
        else:
            assert got is not None, f"expected {want!r}, got None"
            assert got == pytest.approx(want, rel=rel_tol, abs=abs_tol)


def _run_ewm_pair(x_values, y_values, window):
    x = pl.DataFrame({"a": x_values})
    y = pl.DataFrame({"a": y_values})
    corr = TSEwmCorrNative()._calculate_series(x, y, window=window)["a"].to_list()
    cov = TSEwmCovNative()._calculate_series(x, y, window=window)["a"].to_list()
    return corr, cov


def test_ewm_pairwise_matches_pandas_on_random_series():
    """Old-divergence case: the pre-fix kernel diverged from pandas on cov."""
    import numpy as np

    rng = np.random.default_rng(11)
    xs = rng.normal(0, 2, 12).tolist()
    ys = (0.5 * np.asarray(xs) + rng.normal(0, 1, 12)).tolist()

    corr, cov = _run_ewm_pair(xs, ys, 6)
    # Old kernel: cov[1] = 0.197 vs pandas 0.674; corr matched only by luck of
    # scale-free cancellation.  The new kernel must match pandas exactly.
    _assert_ewm(corr, _ewm_oracle(xs, ys, 6, corr=True))
    _assert_ewm(cov, _ewm_oracle(xs, ys, 6, corr=False))


def test_ewm_pairwise_nan_holes_match_pandas():
    """Rows missing either operand must not update the EWM pair state (pandas
    carries the previous output forward at invalid rows)."""
    import numpy as np

    rng = np.random.default_rng(5)
    xs = rng.normal(0, 2, 12).tolist()
    ys = (0.5 * np.asarray(xs) + rng.normal(0, 1, 12)).tolist()
    holed_x = [None if i in (4, 9) else v for i, v in enumerate(xs)]
    holed_y = [None if i == 7 else v for i, v in enumerate(ys)]

    corr, cov = _run_ewm_pair(holed_x, holed_y, 5)
    _assert_ewm(corr, _ewm_oracle(holed_x, holed_y, 5, corr=True))
    _assert_ewm(cov, _ewm_oracle(holed_x, holed_y, 5, corr=False))


def test_ewm_pairwise_finite_pair_observations_like_pandas():
    x_values = [1.0, None, 3.0, 4.0]
    y_values = [2.0, 5.0, None, 8.0]

    corr, cov = _run_ewm_pair(x_values, y_values, 4)
    _assert_ewm(corr, [None, None, None, 1.0])
    _assert_ewm(cov, _ewm_oracle(x_values, y_values, 4, corr=False))
    assert cov[3] == pytest.approx(9.0, rel=1e-10)


def test_ewm_pairwise_inf_is_invalid_not_propagated():
    """Inf must be treated as an invalid observation (pandas would propagate
    Inf as valid — the kernel pre-masks it to keep the pairwise-finite
    contract)."""
    x_values = [1.0, float("inf"), 3.0, 4.0, 5.0]
    y_values = [2.0, 4.0, 6.0, 8.0, 10.0]

    corr, cov = _run_ewm_pair(x_values, y_values, 4)
    # Row 1 is dropped from both streams: no output yet (warmup T>=2), rows 2+
    # are the perfect-linear correlation.
    _assert_ewm(corr, [None, None, 1.0, 1.0, 1.0])
    _assert_ewm(cov, _ewm_oracle(x_values, y_values, 4, corr=False))


def test_ewm_pairwise_constant_series_corr_is_null_cov_is_zero():
    """Zero-variance stream: corr is null (guard), cov collapses to 0."""
    x_values = [2.0, 2.0, 2.0, 2.0]
    y_values = [1.0, 2.0, 3.0, 4.0]

    corr, cov = _run_ewm_pair(x_values, y_values, 4)
    _assert_ewm(corr, [None, None, None, None])
    _assert_ewm(cov, [None, 0.0, 0.0, 0.0], abs_tol=1e-12)


def test_ewm_pairwise_exact_linear_correlation_is_plus_minus_one():
    for y_values in ([2.0, 4.0, 6.0, 8.0], [-2.0, -4.0, -6.0, -8.0]):
        x_values = [1.0, 2.0, 3.0, 4.0]
        corr, cov = _run_ewm_pair(x_values, y_values, 4)
        sign = 1.0 if y_values[0] > 0 else -1.0
        _assert_ewm(corr, [None, sign, sign, sign])
        _assert_ewm(cov, _ewm_oracle(x_values, y_values, 4, corr=False))


def test_ewm_pairwise_warmup_requires_two_valid_pairs():
    """pandas ewmcov is NaN until two valid pairs exist (bias-correction
    denominator is zero at T=1) — first row is always null."""
    x_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_values = [2.0, 4.0, 7.0, 8.0, 11.0]

    corr, cov = _run_ewm_pair(x_values, y_values, 3)
    _assert_ewm(corr, _ewm_oracle(x_values, y_values, 3, corr=True))
    _assert_ewm(cov, _ewm_oracle(x_values, y_values, 3, corr=False))
    assert corr[0] is None and cov[0] is None

    # A leading invalid pair delays the warmup to the second valid pair.
    leading_nan_x = [None] + x_values
    corr, cov = _run_ewm_pair(leading_nan_x, y_values, 3)
    assert corr[0] is None and corr[1] is None and corr[2] is not None
    assert cov[0] is None and cov[1] is None and cov[2] is not None


def test_ewm_pairwise_window_two_minimum_matches_pandas():
    import numpy as np

    rng = np.random.default_rng(21)
    xs = rng.normal(0, 1, 10).tolist()
    ys = rng.normal(0, 1, 10).tolist()

    corr, cov = _run_ewm_pair(xs, ys, 2)
    _assert_ewm(corr, _ewm_oracle(xs, ys, 2, corr=True))
    _assert_ewm(cov, _ewm_oracle(xs, ys, 2, corr=False))


def test_ewm_pairwise_rejects_window_below_two():
    """window < 2 fails closed (strict_integer minimum=2)."""
    x = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pl.DataFrame({"a": [2.0, 3.0, 4.0]})
    for operator in (TSEwmCorrNative(), TSEwmCovNative()):
        with pytest.raises((ValueError, TypeError)):
            operator._calculate_series(x, y, window=1)


def test_ewm_pairwise_xx_corr_is_one_cov_is_ewm_var():
    """ewm_corr(x, x) == 1 wherever finite; ewm_cov(x, x) == ewm_var(x)."""
    import numpy as np

    rng = np.random.default_rng(13)
    xs = rng.normal(-1, 3, 15).tolist()
    ys = [None if i in (2, 8) else v for i, v in enumerate(xs)]  # holes -> warmup restarts

    corr, cov = _run_ewm_pair(xs, xs, 6)
    for i, got in enumerate(corr):
        assert got is None or got == pytest.approx(1.0, rel=1e-10, abs=1e-12)

    corr_h, cov_h = _run_ewm_pair(ys, ys, 6)
    var_h = _ewm_oracle_var(ys, 6)
    _assert_ewm(cov_h, var_h)
    for got in corr_h:
        assert got is None or got == pytest.approx(1.0, rel=1e-10, abs=1e-12)


def test_ewm_pairwise_span_alpha_alias():
    """span=w must equal alpha=2/(w+1) (the pandas mapping the kernel pins)."""
    import numpy as np

    rng = np.random.default_rng(17)
    xs = rng.normal(0, 2, 12).tolist()
    ys = (0.3 * np.asarray(xs) + rng.normal(0, 1, 12)).tolist()

    corr, cov = _run_ewm_pair(xs, ys, 7)
    _assert_ewm(corr, _ewm_oracle_alpha(xs, ys, 2.0 / 8, corr=True))
    _assert_ewm(cov, _ewm_oracle_alpha(xs, ys, 2.0 / 8, corr=False))


def test_ewm_pairwise_cov_is_unbiased_default_bias_false():
    """Kernel cov uses the pandas default bias=False (unbiased); the biased
    stream is a different number, so the oracle must pin the unbiased one."""
    import numpy as np

    rng = np.random.default_rng(19)
    xs = rng.normal(0, 2, 10).tolist()
    ys = rng.normal(0, 1, 10).tolist()

    corr, cov = _run_ewm_pair(xs, ys, 5)
    biased = _ewm_oracle(xs, ys, 5, corr=False, bias=True)
    assert any(w is not None and b is not None and w != pytest.approx(b, rel=1e-6)
               for w, b in zip(cov, biased))
    _assert_ewm(cov, _ewm_oracle(xs, ys, 5, corr=False))


def test_ewm_pairwise_rejects_window_below_two():
    x = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pl.DataFrame({"a": [2.0, 3.0, 4.0]})
    for operator in (TSEwmCorrNative(), TSEwmCovNative()):
        with pytest.raises((ValueError, TypeError)):
            operator._calculate_series(x, y, window=1)


def test_ewm_pairwise_multiple_columns_align_independently():
    import numpy as np

    rng = np.random.default_rng(7)
    n = 12
    x = pl.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 5, n),
    })
    y = pl.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 5, n),
    })

    corr = TSEwmCorrNative()._calculate_series(x, y, window=5)
    cov = TSEwmCovNative()._calculate_series(x, y, window=5)
    for column in ("a", "b"):
        xs = x[column].to_list()
        ys = y[column].to_list()
        _assert_ewm(corr[column].to_list(), _ewm_oracle(xs, ys, 5, corr=True))
        _assert_ewm(cov[column].to_list(), _ewm_oracle(xs, ys, 5, corr=False))


def test_ewm_pairwise_honors_span_alias():
    """``span`` (the pandas-side parameter name) must drive the kernel, not be
    silently absorbed while ``window`` keeps its default."""
    import numpy as np

    rng = np.random.default_rng(3)
    xs = rng.normal(0, 2, 10).tolist()
    ys = (0.7 * np.asarray(xs) + rng.normal(0, 1, 10)).tolist()
    x = pl.DataFrame({"a": xs})
    y = pl.DataFrame({"a": ys})

    corr_span = TSEwmCorrNative()._calculate_series(x, y, span=3)["a"].to_list()
    cov_span = TSEwmCovNative()._calculate_series(x, y, span=3)["a"].to_list()
    _assert_ewm(corr_span, _ewm_oracle(xs, ys, 3, corr=True))
    _assert_ewm(cov_span, _ewm_oracle(xs, ys, 3, corr=False))


def test_ewm_pairwise_delegation_is_declared_not_polars_native():
    """The kernels delegate to pandas — the physical spec must say so."""
    from factor_engine.backend.contracts import ExecutionKind

    for cls in (TSEwmCorrNative, TSEwmCovNative):
        spec = getattr(cls, "_physical_spec", None)
        assert spec is not None, f"{cls.__name__} must declare _physical_spec"
        assert spec.execution_kind is not ExecutionKind.POLARS_NATIVE_EXPR
        assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
        assert "pandas_ewm_span_adjustfalse" in spec.semantic_contract_hash


def test_pairwise_matching_columns_retain_aligned_arithmetic():
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    right = pl.DataFrame({"a": [2.0, 4.0, 8.0]})
    result = TSCorrNative()._calculate_series(left, right, window=3, min_periods=2)
    assert result["a"].to_list() == [None, 1.0, pytest.approx(0.9819805060619659)]


def test_bootstrap_module_list_selects_repaired_ts_cov_as_exact_authority():
    import factor_engine.cleaned_operators
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.common.polars_ts_rolling import TSCovNative as ExpectedTSCovNative
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert "factor_engine.cleaned_operators.common.polars_ts_rolling" in cleaned_operators._LOAD_MODULES
    load_all()
    operator = OperatorRegistry.get("ts_cov", backend="polars")
    assert type(operator) is ExpectedTSCovNative
    assert type(operator).__module__ == "factor_engine.cleaned_operators.common.polars_ts_rolling"
    assert type(operator).__name__ == "TSCovNative"
    assert not any(
        entry.get("canonical") == "ts_cov"
        and entry.get("backend") == "polars"
        and entry.get("old_source") == "factor_dsl_np"
        for entry in OperatorRegistry.overwrite_log()
    )


def test_bootstrap_module_list_activates_repaired_native_module():
    import factor_engine.cleaned_operators
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert "factor_engine.cleaned_operators.common.polars_ts_rolling" in cleaned_operators._LOAD_MODULES
    load_all()
    operator = OperatorRegistry.get("ts_regression_slope", backend="polars")
    assert type(operator).__module__ == "factor_engine.cleaned_operators.common.polars_ts_rolling"
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
