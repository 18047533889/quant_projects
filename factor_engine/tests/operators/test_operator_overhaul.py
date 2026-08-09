# -*- coding: utf-8 -*-
"""Regression tests for the audited operator overhaul."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def test_no_polars_backend_is_a_pandas_bridge() -> None:
    for canonical, catalog in OperatorRegistry.catalog().items():
        metadata = dict(catalog.get("backend_meta") or {})
        source = str((metadata.get("polars") or {}).get("source", ""))
        assert "bridge" not in source.lower(), (canonical, source)
        assert source != "daily_panel_polars", (canonical, source)


def test_duplicate_canonicals_are_aliases() -> None:
    canonicals = set(OperatorRegistry.list_canonical())
    for removed in (
        "MACD",
        "Slope",
        "slope",
        "beta",
        "rolling_beta",
        "cs_rank_01",
        "rank_pct",
        "c_percentile",
        "is_inf",
        "ts_top_n_avg",
        "ts_top_n_std",
        "ts_bottom_n_avg",
        "ts_bottom_n_sum",
        "tm_top_n_avg",
        "tm_top_n_sum",
    ):
        assert removed not in canonicals
        assert removed in OperatorRegistry._aliases

    assert OperatorRegistry._aliases["MACD"] == "MACD_line"
    assert OperatorRegistry._aliases["Beta"] == "ts_beta"
    assert OperatorRegistry._aliases["rolling_beta"] == "ts_beta"


def test_cs_bucket_uses_full_bucket_range() -> None:
    x = pd.DataFrame([np.arange(1.0, 11.0)], columns=list("ABCDEFGHIJ"))
    result = OperatorRegistry.get("cs_bucket").calculate(x, 10)
    assert result.iloc[0].tolist() == list(np.arange(1.0, 11.0))

    singleton = pd.DataFrame({"A": [5.0], "B": [np.nan]})
    single_result = OperatorRegistry.get("cs_bucket").calculate(singleton, 10)
    assert single_result.loc[0, "A"] == 6.0
    assert np.isnan(single_result.loc[0, "B"])


def test_conditional_operators_reject_infinite_values() -> None:
    x = pd.DataFrame({"A": [1.0, np.inf, 3.0, 4.0]})
    # NEW-047/253: the legal ConditionBool set is strictly {0, 1, NaN}.  An Inf
    # in the CONDITION is a contract violation and must raise — it must NOT be
    # silently treated as True by one family and False by another.
    condition = pd.DataFrame({"A": [1.0, 1.0, np.inf, 1.0]})
    with pytest.raises(ValueError):
        OperatorRegistry.get("ts_sum_if").calculate(x, condition, 4, 1)
    # An Inf in the DATA (not the condition) is a missing observation: excluded
    # from the selection, never counted toward a sum.
    condition_ok = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]})
    result = OperatorRegistry.get("ts_sum_if").calculate(x, condition_ok, 4, 1)
    # row windows: [1] -> 1; [1,inf] -> 1; [1,inf,3] -> 1+3; [1,inf,3,4] -> 1+3+4
    assert result["A"].tolist() == [1.0, 1.0, 4.0, 8.0]


def test_argmax_argmin_are_distance_to_most_recent_tie() -> None:
    x = pd.DataFrame({"A": [1.0, 3.0, 3.0, 2.0, 0.0, 0.0]})
    argmax = OperatorRegistry.get("ts_argmax").calculate(x, 4)
    argmin = OperatorRegistry.get("ts_argmin").calculate(x, 4)
    assert argmax["A"].iloc[2] == 0.0
    assert argmax["A"].iloc[3] == 1.0
    assert argmin["A"].iloc[-1] == 0.0


def test_time_slope_recomputes_window_time_coordinates() -> None:
    x = pd.DataFrame({"A": [1.0, np.nan, 5.0, 7.0]})
    result = OperatorRegistry.get("ts_time_slope").calculate(x, 4, 2)
    # Valid coordinates are t=[0,2,3], not a truncated fixed weight vector.
    t = np.array([0.0, 2.0, 3.0])
    y = np.array([1.0, 5.0, 7.0])
    expected = np.dot(t - t.mean(), y - y.mean()) / np.dot(t - t.mean(), t - t.mean())
    assert result["A"].iloc[-1] == pytest.approx(expected)


def test_regression_residual_is_current_residual_not_rolling_mean() -> None:
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.DataFrame({"A": [2.0, 4.0, 6.0, 8.0, 11.0]})
    residual = OperatorRegistry.get("ts_regression_resid").calculate(y, x, 5, 3)
    design = np.column_stack((np.ones(5), x["A"].to_numpy()))
    beta = np.linalg.lstsq(design, y["A"].to_numpy(), rcond=None)[0]
    expected = y["A"].iloc[-1] - design[-1] @ beta
    assert residual["A"].iloc[-1] == pytest.approx(expected)


def test_period_operators_do_not_count_forward_filled_rows_as_new_periods() -> None:
    periods = pd.DataFrame({"A": ["2024Q1"] * 3 + ["2024Q2"] * 3 + ["2024Q3"] * 3 + ["2024Q4"] * 3})
    quarterly = pd.DataFrame({"A": [10.0] * 3 + [20.0] * 3 + [30.0] * 3 + [40.0] * 3})
    lagged = OperatorRegistry.get("period_lag").calculate(quarterly, periods, 1)
    assert lagged["A"].iloc[3] == 10.0
    assert lagged["A"].iloc[5] == 10.0

    ttm = OperatorRegistry.get("ttm_from_quarterly").calculate(quarterly, periods, 4, True)
    assert ttm["A"].iloc[-1] == 100.0
    assert ttm["A"].iloc[8] != 100.0 or np.isnan(ttm["A"].iloc[8])


def test_macd_alias_matches_composed_line() -> None:
    x = pd.DataFrame({"A": np.linspace(10.0, 30.0, 60)})
    alias = OperatorRegistry.get("MACD").calculate(x, 12, 26, 9)
    line = OperatorRegistry.get("MACD_line").calculate(x, 12, 26)
    pd.testing.assert_frame_equal(alias, line)


@pytest.mark.parametrize(
    "operator,args",
    [
        ("ts_count_if", (pd.DataFrame({"A": [1.0, 0.0, np.nan, 1.0]}), 3, 1)),
        (
            "ts_sum_if",
            (
                pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}),
                pd.DataFrame({"A": [1.0, 0.0, 1.0, 1.0]}),
                3,
                1,
            ),
        ),
        ("ts_argmax", (pd.DataFrame({"A": [1.0, 3.0, 2.0, 3.0]}), 3, 1)),
        ("ts_topk_mean", (pd.DataFrame({"A": [1.0, 4.0, 2.0, 3.0]}), 4, 2, 2)),
    ],
)
def test_native_polars_matches_pandas(operator, args) -> None:
    pl = pytest.importorskip("polars")
    if "polars" not in OperatorRegistry.backends_for(operator):
        from cleaned_operators.operator_surface import classify_canonical
        assert classify_canonical(operator) in {"daily", "extended"}
        pytest.skip("operator has no certified pure-Polars implementation")
    pandas_result = OperatorRegistry.get(operator, backend="pandas_numpy").calculate(*args)
    polars_args = [
        pl.DataFrame({c: arg[c].to_numpy() for c in arg.columns})
        if isinstance(arg, pd.DataFrame)
        else arg
        for arg in args
    ]
    polars_result = OperatorRegistry.get(operator, backend="polars").calculate(*polars_args)
    np.testing.assert_allclose(
        polars_result.select(pandas_result.columns.tolist()).to_numpy(),
        pandas_result.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-8,
        atol=1e-8,
    )


def test_future_mutation_does_not_change_prefix() -> None:
    x = pd.DataFrame({"A": np.arange(1.0, 21.0)})
    baseline = OperatorRegistry.get("ts_tail_mean").calculate(x, 5, 0.2, "lower", 3)
    changed = x.copy()
    changed.iloc[-4:] = -1e9
    mutated = OperatorRegistry.get("ts_tail_mean").calculate(changed, 5, 0.2, "lower", 3)
    np.testing.assert_allclose(
        baseline.iloc[:-4].to_numpy(),
        mutated.iloc[:-4].to_numpy(),
        equal_nan=True,
    )
