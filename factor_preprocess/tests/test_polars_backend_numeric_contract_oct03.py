"""Numeric contract oracles for the Polars cross-sectional backend (2026-10-03)."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import runpy
import sys
from pathlib import Path

from factor_preprocess.backends.polars_backend import (
    cs_rank_polars,
    cs_zscore_polars,
)
from factor_preprocess.transforms.cross_sectional import cs_rank, cs_zscore


def _as_pandas(frame):
    return frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()


def _result_values(result):
    if isinstance(result, pd.Series):
        return result.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(
        [np.nan if value is None else value for value in result.to_list()],
        dtype=float,
    )


def _rank_reference(frame, *, method="average", pct=False):
    pdf = _as_pandas(frame)
    values = pdf["value"].to_numpy()
    expected = np.full(len(pdf), np.nan, dtype=float)
    groups = pdf.groupby("date", sort=False, dropna=False).indices
    for positions in groups.values():
        group_values = values[positions]
        if group_values.dtype.kind == "O":
            missing = np.asarray([pd.isna(value) for value in group_values], dtype=bool)
            present = [value for value, is_missing in zip(group_values, missing) if not is_missing]
            if not present:
                continue
            # Nullable integer objects stay integer typed; do not lose ties by
            # converting large int64 values through float64.
            if all(isinstance(value, (int, np.integer)) for value in present):
                integer_values = np.asarray(present, dtype=np.int64)
                expected[positions[~missing]] = cs_rank(
                    integer_values, method=method, pct=pct
                )
            else:
                numeric_values = np.asarray(
                    [np.nan if is_missing else value for value, is_missing in zip(group_values, missing)],
                    dtype=np.float64,
                )
                expected[positions] = cs_rank(
                    numeric_values, method=method, pct=pct
                )
        else:
            expected[positions] = cs_rank(group_values, method=method, pct=pct)
    return expected


def _zscore_reference(frame, *, ddof=1, constant_value=0.0):
    pdf = _as_pandas(frame)
    values = pdf["value"].to_numpy(dtype=np.float64, na_value=np.nan)
    expected = np.full(len(pdf), np.nan, dtype=float)
    groups = pdf.groupby("date", sort=False, dropna=False).indices
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for positions in groups.values():
            expected[positions] = cs_zscore(
                values[positions], ddof=ddof, constant_value=constant_value
            )
    return expected


def _both_inputs(pdf):
    # Preserve actual IEEE NaN in the Polars test input; null is tested too.
    return [pdf, pl.DataFrame(pdf.to_dict(orient="list"))]


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
@pytest.mark.parametrize("pct", [False, True])
@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_rank_matches_fp_for_ties_nonfinite_and_singleton(method, pct, kind):
    pdf = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 1, 1, 2, 2],
            "value": [4.0, 2.0, 2.0, np.nan, np.inf, -np.inf, 9.0, np.nan],
        },
        index=[7, 7, 3, 3, 9, 9, 1, 1],
    )
    frame = pdf if kind == "pandas" else pl.DataFrame(pdf.to_dict(orient="list"))
    actual = cs_rank_polars(frame, "value", group_col="date", method=method, pct=pct)
    np.testing.assert_allclose(
        _result_values(actual),
        _rank_reference(frame, method=method, pct=pct),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert len(actual) == len(pdf)
    if kind == "pandas":
        assert actual.index.equals(pdf.index)
    # Output values remain in caller row order, even with duplicate labels.
    assert _result_values(actual)[0] == _rank_reference(frame, method=method, pct=pct)[0]


@pytest.mark.parametrize("kind", ["pandas", "polars"])
@pytest.mark.parametrize("pct", [False, True])
def test_integer_rank_keeps_large_values_exact(kind, pct):
    pdf = pd.DataFrame(
        {
            "date": [1, 1, 1, 2],
            "value": np.asarray(
                [2**53, 2**53 + 1, 2**53 + 1, 2**53 + 2],
                dtype=np.int64,
            ),
        }
    )
    frame = pdf if kind == "pandas" else pl.DataFrame(pdf)
    actual = cs_rank_polars(frame, "value", group_col="date", pct=pct)
    np.testing.assert_allclose(
        _result_values(actual),
        _rank_reference(frame, pct=pct),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_rank_polars_null_and_nan_are_both_missing_but_inf_is_not_ranked():
    frame = pl.DataFrame(
        {"date": [1, 1, 1, 1, 1], "value": [1.0, 2.0, float("nan"), None, float("inf")]}
    )
    actual = cs_rank_polars(frame, "value", group_col="date", pct=True)
    np.testing.assert_allclose(
        _result_values(actual),
        [0.0, 1.0, np.nan, np.nan, np.nan],
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_all_null_polars_value_column_is_valid_all_missing_numeric_input():
    frame = pl.DataFrame({"date": [1, 1], "value": [None, None]})
    assert frame.schema["value"] == pl.Null
    np.testing.assert_array_equal(
        _result_values(cs_rank_polars(frame, "value", group_col="date", pct=True)),
        [np.nan, np.nan],
    )
    np.testing.assert_array_equal(
        _result_values(cs_zscore_polars(frame, "value", group_col="date")),
        [np.nan, np.nan],
    )


@pytest.mark.parametrize("method", ["bogus", "", None])
def test_rank_rejects_invalid_method(method):
    frame = pl.DataFrame({"date": [1], "value": [1.0]})
    with pytest.raises(ValueError, match="rank method"):
        cs_rank_polars(frame, "value", group_col="date", method=method)


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_zscore_matches_fp_for_nan_null_constant_singleton_and_infinity(kind):
    pdf = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "value": [
                1.0, 2.0, np.nan, 4.0, 5.0, 5.0, 5.0,
                7.0, np.nan, 1.0, 2.0, np.inf,
            ],
        }
    )
    if kind == "pandas":
        frame = pdf
    else:
        frame = pl.DataFrame(
            {
                "date": pdf["date"].tolist(),
                "value": [1.0, 2.0, float("nan"), None, 5.0, 5.0, 5.0,
                          7.0, float("nan"), 1.0, 2.0, float("inf")],
            }
        )
    actual = cs_zscore_polars(
        frame, "value", group_col="date", ddof=1, constant_value=3.25
    )
    np.testing.assert_allclose(
        _result_values(actual),
        _zscore_reference(frame, ddof=1, constant_value=3.25),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_zscore_omits_all_missing_values_and_preserves_nan_null_outputs(kind):
    pdf = pd.DataFrame({"date": [1, 1, 1, 2, 2], "value": [np.nan, np.nan, 4.0, 8.0, np.nan]})
    if kind == "pandas":
        frame = pdf
    else:
        frame = pl.DataFrame(
            {"date": [1, 1, 1, 2, 2], "value": [float("nan"), None, 4.0, 8.0, None]}
        )
    actual = cs_zscore_polars(frame, "value", group_col="date", constant_value=2.0)
    np.testing.assert_allclose(
        _result_values(actual),
        _zscore_reference(frame, constant_value=2.0),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_randomized_1000_row_polars_expressions_match_public_fp_oracles():
    rng = np.random.default_rng(20261003)
    values = rng.normal(size=1000)
    values[[5, 81, 327]] = np.nan
    values[[14, 702]] = np.inf
    pdf = pd.DataFrame(
        {"date": np.arange(1000) % 17, "value": values},
        index=np.arange(1000) % 23,
    )
    frame = pl.DataFrame(pdf.reset_index(drop=True))
    for method in ("average", "min", "max", "dense", "ordinal"):
        for pct in (False, True):
            actual = cs_rank_polars(frame, "value", group_col="date", method=method, pct=pct)
            np.testing.assert_allclose(
                _result_values(actual),
                _rank_reference(frame, method=method, pct=pct),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
    actual_z = cs_zscore_polars(frame, "value", group_col="date")
    np.testing.assert_allclose(
        _result_values(actual_z), _zscore_reference(frame),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("transform", [cs_rank_polars, cs_zscore_polars])
def test_non_numeric_value_dtype_is_rejected(transform):
    frame = pl.DataFrame({"date": [1, 1], "value": ["1", "2"]})
    with pytest.raises(TypeError, match="numeric dtype"):
        transform(frame, "value", group_col="date")


def test_module_imports_when_optional_polars_dependency_is_unavailable(monkeypatch):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "factor_preprocess"
        / "backends"
        / "polars_backend.py"
    )
    monkeypatch.setitem(sys.modules, "polars", None)
    module = runpy.run_path(str(module_path))
    assert module["POLARS_AVAILABLE"] is False
