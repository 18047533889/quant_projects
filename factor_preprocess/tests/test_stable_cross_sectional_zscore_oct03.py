"""Independent high-precision numeric contract tests for stable FP z-scores."""
from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import polars as pl
import pytest


def _decimal_zscore(values, *, ddof=1, constant_value=0.0):
    """Reference exact represented float64 values, not rounded decimal literals."""
    x = np.asarray(values, dtype=np.float64)
    result = np.full(x.shape, np.nan, dtype=np.float64)
    present = ~np.isnan(x)
    if not present.any():
        return result
    if not np.isfinite(x[present]).all():
        result[present] = constant_value
        return result
    finite = x[present]
    if finite.size <= ddof:
        result[present] = constant_value
        return result
    with localcontext() as ctx:
        # Float64 subnormal exact decimal expansions require over 1,000 digits.
        # Low precision can invent variance even for identical represented values.
        ctx.prec = 1200
        dx = [Decimal.from_float(float(value)) for value in finite]
        mean = sum(dx, Decimal(0)) / Decimal(len(dx))
        numerator = sum(((value - mean) ** 2 for value in dx), Decimal(0))
        variance = numerator / Decimal(len(dx) - ddof)
        std = variance.sqrt()
        if not std:
            result[present] = constant_value
            return result
        result[present] = np.asarray(
            [float((value - mean) / std) for value in dx], dtype=np.float64
        )
    return result


def _pandas_frame(frame):
    return frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()


def _grouped_decimal_reference(frame, *, ddof=1, constant_value=0.0):
    pdf = _pandas_frame(frame)
    values = pdf["value"].to_numpy(dtype=np.float64, na_value=np.nan)
    out = np.full(len(pdf), np.nan, dtype=np.float64)
    groups = pdf.groupby("group", sort=False, dropna=False).indices
    for positions in groups.values():
        out[positions] = _decimal_zscore(
            values[positions], ddof=ddof, constant_value=constant_value
        )
    return out


def _assert_close(actual, expected, *, atol=2e-15):
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=atol,
                               equal_nan=True)


def test_large_offset_float64_fixture_matches_exact_represented_value_oracle():
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    base = np.float64(1e16)
    values = np.array([base, base + 2., base + 4., base + 6.], dtype=np.float64)
    expected = np.array([-1.3416407864998738, -0.4472135954999579,
                         0.4472135954999579, 1.3416407864998738])
    np.testing.assert_array_equal(_decimal_zscore(values, ddof=0), expected)
    _assert_close(cs_zscore(values, ddof=0), expected)


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_ordinary_random_finite_values_match_decimal_oracle_for_ddof(ddof):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    rng = np.random.default_rng(20261003 + ddof)
    values = rng.normal(loc=1e7, scale=3.5, size=71)
    values[[2, 15, 54]] = np.nan
    _assert_close(cs_zscore(values, ddof=ddof),
                  _decimal_zscore(values, ddof=ddof))


@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_singleton_and_ddof_at_or_above_finite_count_use_constant_value(ddof):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    values = np.array([np.nan, 7.25, np.nan]) if ddof in (0, 1) else np.array([2., 4., np.nan, np.nan])
    actual = cs_zscore(values, ddof=ddof, constant_value=-3.5)
    _assert_close(actual, _decimal_zscore(values, ddof=ddof, constant_value=-3.5))
    finite = np.isfinite(values)
    if finite.sum() <= ddof:
        np.testing.assert_array_equal(actual[finite], -3.5)


@pytest.mark.parametrize("values", [
    np.array([np.nan, np.nan]),
    np.array([np.nan, np.inf, 3., -np.inf]),
    np.array([4., 4., 4., np.nan]),
])
def test_all_missing_infinity_and_constant_policy(values):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    actual = cs_zscore(values, ddof=1, constant_value=2.75)
    expected = _decimal_zscore(values, ddof=1, constant_value=2.75)
    np.testing.assert_array_equal(actual, expected)


def test_int64_above_float_exactness_boundary_uses_float64_reference_values():
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    values = np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64)
    expected = _decimal_zscore(values.astype(np.float64), ddof=0)
    _assert_close(cs_zscore(values, ddof=0), expected)


@pytest.mark.parametrize("values", [
    np.array([-np.finfo(np.float64).max, np.finfo(np.float64).max]),
    np.array([-np.finfo(np.float64).max, 0., np.finfo(np.float64).max]),
    np.array([0., np.nextafter(0., 1.), 2 * np.nextafter(0., 1.)]),
    np.array([np.nextafter(0., 1.)] * 4),
])
@pytest.mark.parametrize("ddof", [0, 1])
def test_extreme_spans_and_subnormals_match_decimal_oracle(values, ddof):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        actual = cs_zscore(values, ddof=ddof, constant_value=1.25)
    expected = _decimal_zscore(values, ddof=ddof, constant_value=1.25)
    _assert_close(actual, expected, atol=5e-15)


@pytest.mark.parametrize("axis", [0, -1])
def test_noncontiguous_multidimensional_input_matches_axiswise_decimal_oracle(axis):
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    rng = np.random.default_rng(731)
    source = rng.normal(size=(7, 8, 9))
    source[2, 3, 4] = np.nan
    values = source[:, ::2, ::2]
    assert not values.flags.c_contiguous
    expected = np.apply_along_axis(
        lambda row: _decimal_zscore(row, ddof=1, constant_value=-2.0),
        axis, values,
    )
    _assert_close(cs_zscore(values, axis=axis, ddof=1, constant_value=-2.0), expected)


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_grouped_adapter_preserves_duplicate_rows_and_null_groups(kind):
    from factor_preprocess.backends.polars_backend import cs_zscore_polars

    pdf = pd.DataFrame(
        {"group": [None, None, 1, 1, None, None, 1, 1],
         "value": [10., 12., 3., 5., np.nan, 14., 7., 9.]},
        index=[8, 8, 3, 3, 8, 8, 3, 3],
    )
    if kind == "pandas":
        frame = pdf
    else:
        frame = pl.DataFrame(
            {"group": pdf["group"].tolist(), "value": pdf["value"].tolist()}
        )
    actual = cs_zscore_polars(frame, "value", group_col="group", ddof=0)
    values = (actual.to_numpy(dtype=float, na_value=np.nan)
              if isinstance(actual, pd.Series)
              else np.array([np.nan if x is None else x for x in actual.to_list()]))
    _assert_close(values, _grouped_decimal_reference(frame, ddof=0))
    assert len(values) == len(pdf)
    if kind == "pandas":
        assert actual.index.equals(pdf.index)


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_polars_adapter_large_offsets_and_nonfinite_group_policy(kind):
    from factor_preprocess.backends.polars_backend import cs_zscore_polars

    b = np.float64(1e16)
    pdf = pd.DataFrame(
        {"group": [1, 1, 1, 1, 2, 2, 2],
         "value": [b, b+2., b+4., b+6., 5., np.inf, np.nan]},
        index=[4, 4, 5, 5, 6, 6, 6],
    )
    frame = pdf if kind == "pandas" else pl.DataFrame(pdf.reset_index(drop=True))
    actual = cs_zscore_polars(frame, "value", group_col="group", ddof=0,
                              constant_value=3.25)
    values = (actual.to_numpy(dtype=float, na_value=np.nan)
              if isinstance(actual, pd.Series)
              else np.array([np.nan if x is None else x for x in actual.to_list()]))
    _assert_close(values, _grouped_decimal_reference(
        frame, ddof=0, constant_value=3.25))
    assert values[0] < 0 and values[3] > 0
    np.testing.assert_array_equal(values[4:6], [3.25, 3.25])
    assert np.isnan(values[6])
    if kind == "pandas":
        assert actual.index.equals(pdf.index)


def test_polars_null_nan_and_all_missing_groups_keep_rows_as_nan():
    from factor_preprocess.backends.polars_backend import cs_zscore_polars

    frame = pl.DataFrame(
        {"group": [None, None, 1, 1, 2, 2],
         "value": [None, None, float("nan"), None, 4.0, None]}
    )
    actual = cs_zscore_polars(frame, "value", group_col="group", constant_value=-4.0)
    values = np.array([np.nan if x is None else x for x in actual.to_list()])
    np.testing.assert_array_equal(values, [np.nan, np.nan, np.nan, np.nan, -4., np.nan])
    assert len(values) == frame.height
