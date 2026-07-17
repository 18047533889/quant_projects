# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "ts_last_if",
            (
                pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0]}),
                pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0, 1.0]}),
                3,
            ),
        ),
        ("ts_true_streak", (pd.DataFrame({"A": [1.0, 1.0, 0.0, 1.0, 1.0]}),)),
        ("ts_argmax", (pd.DataFrame({"A": [1.0, 3.0, np.nan, 3.0, 2.0]}), 4, 1)),
        ("ts_bottomk_mean", (pd.DataFrame({"A": [4.0, 1.0, np.nan, 3.0, 2.0]}), 4, 2, 2)),
    ],
)
def test_final_expression_polars_matches_pandas(name: str, args) -> None:
    pl = pytest.importorskip("polars")
    expected = OperatorRegistry.get(name, backend="pandas_numpy").calculate(*args)
    converted = [
        pl.DataFrame({str(column): value[column].to_numpy() for column in value.columns})
        if isinstance(value, pd.DataFrame) else value
        for value in args
    ]
    actual_pl = OperatorRegistry.get(name, backend="polars").calculate(*converted)
    actual = pd.DataFrame(
        {column: actual_pl[str(column)].to_numpy() for column in expected.columns},
        index=expected.index,
    )
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float), expected.to_numpy(dtype=float),
        rtol=1e-10, atol=1e-10, equal_nan=True,
    )


def test_rolling_ols_outputs_share_one_fit_pass() -> None:
    from cleaned_operators.layer_regression_fusion import (
        clear_rolling_ols_cache,
        rolling_ols_cache_info,
    )

    y = pd.DataFrame({"A": [2.0, 4.0, 6.0, 8.0, 11.0, 12.0]})
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    clear_rolling_ols_cache()
    outputs = []
    for name in (
        "ts_regression_slope", "ts_regression_intercept", "ts_regression_resid",
        "ts_regression_r2", "ts_regression_tstat",
    ):
        outputs.append(
            OperatorRegistry.get(name, backend="pandas_numpy").calculate(y, x, 5, 3)
        )
    assert rolling_ols_cache_info()["stats_computations"] == 1
    assert all(frame.shape == y.shape for frame in outputs)
