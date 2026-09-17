from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "rank(ts_regression_tstat(ret, log_returns(volume), 20, 10, 1))",
            "rank(ts_regression_tstat(ret, log_returns(volume), 20, 10, True))",
        ),
        (
            "ts_regression_intercept(y, x, window=20, min_periods=10, add_intercept=0)",
            "ts_regression_intercept(y, x, window=20, min_periods=10, add_intercept=False)",
        ),
        (
            "ts_regression_intercept(y, x, 20, 10, add_intercept=1)",
            "ts_regression_intercept(y, x, 20, 10, add_intercept=True)",
        ),
    ],
)
def test_exact_regression_boolean_literals_are_migrated(source: str, expected: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == expected
    assert len(result.changes) == 1
    assert ".add_intercept" in result.changes[0]


@pytest.mark.parametrize(
    "source",
    [
        "ts_regression_tstat(y, x, 20, 10, 2)",
        "ts_regression_tstat(y, x, 20, 10, flag)",
        "ts_regression_tstat(y, x, 20, 10, 1, extra)",
        "ts_regression_tstat(y, x, 20, 10, 1, add_intercept=True)",
        "ts_regression_intercept(y, x, add_intercept=1, **settings)",
        "namespace.ts_regression_tstat(y, x, 20, 10, 1)",
        "ts_regression_slope(y, x, 20, 10, 1)",
    ],
)
def test_other_numbers_dynamic_values_collisions_and_operators_are_preserved(source: str) -> None:
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == source
    assert result.changes == ()


def test_literal_one_matches_historical_truthy_intercept_semantics() -> None:
    from factor_engine.cleaned_operators.overhaul.regression import (
        pd_reg_intercept,
        pd_reg_tstat,
    )

    index = pd.date_range("2024-01-02", periods=30, freq="B")
    x = pd.DataFrame({"A": np.linspace(-2.0, 3.0, 30)}, index=index)
    y = 1.5 + 2.25 * x + pd.DataFrame(
        {"A": np.sin(np.arange(30)) * 0.1}, index=index
    )
    pd.testing.assert_frame_equal(
        pd_reg_tstat(y, x, 20, 10, 1),
        pd_reg_tstat(y, x, 20, 10, True),
    )
    pd.testing.assert_frame_equal(
        pd_reg_intercept(y, x, 20, 10, 1),
        pd_reg_intercept(y, x, 20, 10, True),
    )
