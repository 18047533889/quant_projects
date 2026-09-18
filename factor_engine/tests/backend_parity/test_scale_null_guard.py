# -*- coding: utf-8
"""scale NULL guard：全空截面缺失行不输出 0。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.backend.cross_section_spec import polars_scale_expr
from factor_engine.cleaned_operators.common.cs_broadcast import broadcast_row_stat


def test_polars_scale_expr_null_row_not_zero():
    import polars as pl

    df = pl.DataFrame(
        {
            "ts": ["2024-01-02"] * 3,
            "inst": ["A", "B", "C"],
            "v": [None, None, None],
        }
    )
    expr = polars_scale_expr("v", 1.0, partition_cols=("ts",), order_by="inst")
    got = df.select(expr.alias("s"))["s"].to_list()
    assert all(x is None for x in got)


def test_polars_scale_expr_valid_row_zero_sum():
    import polars as pl

    df = pl.DataFrame(
        {
            "ts": ["2024-01-02"] * 2,
            "inst": ["A", "B"],
            "v": [0.0, 0.0],
        }
    )
    expr = polars_scale_expr("v", 1.0, partition_cols=("ts",), order_by="inst")
    got = df.select(expr.alias("s"))["s"].to_list()
    assert got == [0.0, 0.0]


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("magnitude", [1.0, 1e308, 1e-308])
def test_scale_finite_members_do_not_overflow_or_poison_each_other(invalid, magnitude):
    import polars as pl

    df = pl.DataFrame({
        "ts": [0, 0, 0, 1, 1, 1],
        "inst": ["A", "B", "C"] * 2,
        "v": [magnitude, invalid, -magnitude, 0.0, invalid, 0.0],
    })
    expr = polars_scale_expr("v", 2.0, partition_cols=("ts",), order_by="inst")
    got = df.select(expr.alias("s"))["s"].to_numpy()
    np.testing.assert_allclose(got, [1.0, np.nan, -1.0, 0.0, np.nan, 0.0], equal_nan=True)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_scale_real_long_path_with_missing_members_and_extremes(fused, invalid):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
    from factor_engine.api.columns import col
    from tests.backend_parity.test_three_backend_parity import _run, _result_series
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=3), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    values = [1e308, invalid, -1e308] * 3
    source = InMemorySeriesSource(data={"x": pd.Series(values, index=idx)})
    inner = F("ts_mean")(col("x"), 1) if fused else col("x")
    result = _run(source, F("scale")(inner), "polars_long")
    assert result.get("used_polars_long_path") is True
    np.testing.assert_allclose(
        _result_series(result).to_numpy(),
        [0.5, np.nan, -0.5] * 3, rtol=1e-12, atol=0, equal_nan=True,
    )
