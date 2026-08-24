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
