"""FE composite parity tests for FP effective-rank OLS semantics."""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_neutralization import (
    OLS_NEUTRALIZATION_RECIPE,
    execute_ols_neutralize,
    get_fe_neutralization_executor,
)
from factor_preprocess.neutralization import ols_neutralize


def _frame(x, y=None, *, index=None):
    lengths = {len(column) for column in x.values()}
    if len(lengths) != 1:
        raise ValueError("all exposure columns must have the same number of rows")
    n = lengths.pop()
    if y is None:
        y = np.array([0.2, 1.7, -0.4, 3.1, 2.0, -1.2, 4.4, 0.8, 5.2, -2.1, 3.7, 1.3])[:n]
    if len(y) != n:
        raise ValueError("values and exposure columns must have the same number of rows")
    ids = [f"a{i}" for i in range(n)]
    values = pd.DataFrame({"date": [1] * n, "asset_id": ids, "value": y}, index=index)
    exposures = pd.DataFrame({"date": [1] * n, "asset_id": ids})
    for column, data in x.items():
        exposures[column] = data
    return values, exposures


@pytest.mark.parametrize("case", ["duplicated", "zero", "near_collinear"])
def test_fe_composite_matches_public_ols_for_rank_deficient_designs(case):
    x = np.arange(12, dtype=float)
    if case == "duplicated":
        predictors = {"x": x, "duplicate": 2 * x + 1}
    elif case == "zero":
        predictors = {"x": x, "zero": np.zeros_like(x)}
    else:
        perturb = np.resize(np.array([-1.0, 1.0]), len(x)) * 1e-14
        predictors = {"x": x, "near_duplicate": x + perturb}
    values, exposures = _frame(predictors)
    actual = execute_ols_neutralize(values, exposures, min_observations=4)
    expected = ols_neutralize(values, exposures, min_observations=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_fe_composite_matches_support_boundary_and_preserves_strict_fe_operator():
    from factor_engine.cleaned_operators.polars_native.cs_neutralize_ols_native_20260930 import (
        _ols_residual_strict,
    )
    values, exposures = _frame({"x": np.array([0.0, 1.0, 2.0])}, y=[2.0, 1.0, 5.0])
    values, exposures = values.iloc[:3], exposures.iloc[:3]
    actual = execute_ols_neutralize(values, exposures, min_observations=2)
    expected = ols_neutralize(values, exposures, min_observations=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    # Generic FE strict contract remains rank/column gated for its callers.
    with pytest.raises(ValueError, match="min_obs must exceed fitted coefficient count"):
        _ols_residual_strict(
            np.array([2.0, 1.0, 5.0]), [np.array([0.0, 1.0, 2.0])],
            weights=None, add_intercept=True, min_obs=2,
        )
    strict_rank_deficient = _ols_residual_strict(
        np.array([2.0, 1.0, 5.0, 9.0]),
        [np.arange(4, dtype=float), 2 * np.arange(4, dtype=float) + 1],
        weights=None, add_intercept=True, min_obs=4,
    )
    assert np.isnan(strict_rank_deficient).all()


def test_fe_composite_left_joins_keys_and_restores_duplicate_index_positions():
    values = pd.DataFrame(
        {"date": [1, 1, 1, 1], "asset_id": ["b", "a", "b", "missing"],
         "value": [4.0, 2.0, 5.0, 99.0]}, index=[7, 7, 3, 7]
    )
    exposures = pd.DataFrame(
        {"date": [1, 1], "asset_id": ["a", "b"], "size": [0.0, 1.0]}
    )
    actual = execute_ols_neutralize(values, exposures, min_observations=2)
    expected = ols_neutralize(values, exposures, min_observations=2)
    assert actual.index.equals(values.index)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isnan(actual.iloc[-1])


def test_fe_composite_matches_null_key_semantics_and_rejects_duplicate_exposure_keys():
    values = pd.DataFrame(
        {"date": [1, 1, 1, 1], "asset_id": [None, "a", "b", "c"],
         "value": [2.0, 1.0, 3.0, 4.0]}, index=[5, 5, 2, 5]
    )
    exposures = pd.DataFrame(
        {"date": [1, 1, 1, 1], "asset_id": [None, "a", "b", "c"],
         "size": [0.0, 1.0, 2.0, 3.0]}
    )
    actual = execute_ols_neutralize(values, exposures, min_observations=2)
    expected = ols_neutralize(values, exposures, min_observations=2)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    with pytest.raises(ValueError, match="unique"):
        execute_ols_neutralize(values, pd.concat([exposures, exposures.iloc[:1]]))


def test_fe_composite_handles_null_dates_and_fully_explained_roundoff():
    values, exposures = _frame({"x": np.arange(12, dtype=float)})
    exposures["date"] = pd.array([pd.NA] + [1] * 11, dtype="Int64")
    values["date"] = pd.array([pd.NA] + [1] * 11, dtype="Int64")
    values.loc[values.index[1:], "value"] = 0.1 + 0.3 * exposures.loc[exposures.index[1:], "x"]
    actual = execute_ols_neutralize(values, exposures, min_observations=3)
    expected = ols_neutralize(values, exposures, min_observations=3)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isnan(actual.iloc[0])
    np.testing.assert_array_equal(actual.iloc[1:].to_numpy(), np.zeros(11))


def test_recipe_resolver_is_explicit_and_fail_closed():
    from factor_preprocess.errors import GovernanceError
    assert get_fe_neutralization_executor("ols_neutralize", OLS_NEUTRALIZATION_RECIPE) is not None
    with pytest.raises(GovernanceError, match="not registered"):
        get_fe_neutralization_executor("other", OLS_NEUTRALIZATION_RECIPE)
    with pytest.raises(GovernanceError, match="not registered"):
        get_fe_neutralization_executor("ols_neutralize", "unknown:v9")


def test_all_zero_predictors_without_intercept_preserve_signal():
    values = pd.DataFrame(
        {"date": [1] * 4, "asset_id": list("abcd"), "value": [1.0, -2.0, 3.0, 4.0]},
        index=[9, 9, 3, 9],
    )
    exposures = values[["date", "asset_id"]].assign(zero=0.0)
    actual = execute_ols_neutralize(
        values, exposures, min_observations=2, add_intercept=False
    )
    expected = ols_neutralize(
        values, exposures, min_observations=2, add_intercept=False
    )
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("case", ["empty", "all_missing", "no_joined_finite", "all_null_dates"])
def test_empty_or_no_finite_observations_match_native(case):
    if case == "empty":
        values = pd.DataFrame({"date": pd.Series(dtype="int64"),
                               "asset_id": pd.Series(dtype="object"),
                               "value": pd.Series(dtype="float64")})
        exposures = pd.DataFrame({"date": pd.Series(dtype="int64"),
                                  "asset_id": pd.Series(dtype="object"),
                                  "size": pd.Series(dtype="float64")})
    elif case == "all_null_dates":
        values = pd.DataFrame({"date": pd.array([pd.NA, pd.NA], dtype="Int64"),
                               "asset_id": ["a", "b"], "value": [1.0, 2.0]})
        exposures = pd.DataFrame({"date": pd.array([pd.NA, pd.NA], dtype="Int64"),
                                  "asset_id": ["a", "b"], "size": [1.0, 2.0]})
    elif case == "all_missing":
        values = pd.DataFrame({"date": [1, 1], "asset_id": ["a", "b"],
                               "value": [np.nan, np.nan]})
        exposures = pd.DataFrame({"date": [1, 1], "asset_id": ["a", "b"],
                                  "size": [1.0, 2.0]})
    else:
        values = pd.DataFrame({"date": [1, 1], "asset_id": ["a", "b"],
                               "value": [1.0, 2.0]})
        exposures = pd.DataFrame({"date": [1, 1], "asset_id": ["c", "d"],
                                  "size": [1.0, 2.0]})
    actual = execute_ols_neutralize(values, exposures, min_observations=2)
    expected = ols_neutralize(values, exposures, min_observations=2)
    pd.testing.assert_series_equal(actual, expected)


def test_zero_minimum_and_large_finite_inputs_match_native_contract():
    values = pd.DataFrame({"date": [1, 1, 1, 1], "asset_id": list("abcd"),
                           "value": [1.0e308, -1.0e308, 1.0e308, -1.0e308]})
    exposures = values[["date", "asset_id"]].assign(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        near=np.array([0.0, 1.0, 2.0, 3.0]) +
             np.array([0.0, 1.0, -1.0, 1.0]) * 1e-14,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        actual = execute_ols_neutralize(values, exposures, min_observations=0)
        expected = ols_neutralize(values, exposures, min_observations=0)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
