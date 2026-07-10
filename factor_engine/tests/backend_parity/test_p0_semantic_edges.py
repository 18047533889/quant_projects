# -*- coding: utf-8
"""P0 语义 edge parity：ties rank、定义域、NaN/null、常数截面。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def edge_source():
    load_all()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    insts = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    close = pd.Series(
        [
            1.0,
            1.0,
            2.0,
            np.nan,
            -1.0,
            0.0,
            np.nan,
            4.0,
        ],
        index=idx,
    )
    exp = pd.Series([2.0, 2.5, -1.0, 0.5, 1.0, 1.0, 3.0, 0.5], index=idx)
    grp = pd.Series([1, 1, 2, 2] * len(dates), index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close, "exp": exp, "grp": grp})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


F = make_cleaned_call_factory


def _assert_all_backends(source, expr_builder, *, rtol=1e-5, atol=1e-5):
    expr = expr_builder()
    pd_out = _run(source, expr, "pandas")
    long_out = _run(source, expr, "polars_long")
    pd.testing.assert_series_equal(
        pd_out.astype(float),
        long_out.astype(float),
        check_names=False,
        rtol=rtol,
        atol=atol,
    )


def test_rank_pct_tie_average(edge_source):
    """并列值使用 average rank（非 min rank）。"""
    out = _run(edge_source, F("rank_pct")(col("close")), "pandas")
    day1 = out.loc[pd.Timestamp("2024-01-02")]
    assert day1.loc["A"] == pytest.approx(0.5)
    assert day1.loc["B"] == pytest.approx(0.5)
    assert day1.loc["C"] == pytest.approx(1.0)
    assert math.isnan(day1.loc["D"])
    _assert_all_backends(edge_source, lambda: F("rank_pct")(col("close")))


def test_group_rank_tie_average(edge_source):
    _assert_all_backends(
        edge_source,
        lambda: F("group_rank")(col("close"), col("grp")),
    )


def test_log_sqrt_power_domain(edge_source):
    _assert_all_backends(edge_source, lambda: F("log")(col("close")))
    _assert_all_backends(edge_source, lambda: F("sqrt")(col("close")))
    _assert_all_backends(
        edge_source,
        lambda: F("power")(col("close"), col("exp")),
    )


def test_is_nan_is_finite_nan_to_num(edge_source):
    _assert_all_backends(edge_source, lambda: F("is_nan")(col("close")))
    _assert_all_backends(edge_source, lambda: F("is_finite")(col("close")))
    _assert_all_backends(edge_source, lambda: F("nan_to_num")(col("close")))


def test_normalize_constant_cross_section(edge_source):
    _assert_all_backends(edge_source, lambda: F("normalize")(col("grp")))


def test_bfill_polars_long_raises(edge_source):
    from backend.polars_long_policy import UnsupportedCausalOperatorError

    expr = F("bfill")(col("close"))
    with pytest.raises(UnsupportedCausalOperatorError):
        FactorEngine(backend=build_backend("polars_long"), data_source=edge_source).run(
            Factor(name="t", expr=expr)
        )


def test_bfill_duckdb_raises(edge_source):
    from backend.polars_long_policy import UnsupportedCausalOperatorError

    expr = F("bfill")(col("close"))
    with pytest.raises(UnsupportedCausalOperatorError):
        FactorEngine(backend=build_backend("duckdb_sql"), data_source=edge_source).run(
            Factor(name="t", expr=expr)
        )
