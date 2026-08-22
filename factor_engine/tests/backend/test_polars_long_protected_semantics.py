# -*- coding: utf-8
"""protected_div / protected_log PolarsLong 语义与 SQL/Pandas 对齐。"""
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
from backend.polars_long_policy import POLARS_LONG_MAP_GROUPS, POLARS_LONG_NATIVE
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def edge_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-06"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=idx)
    y = pd.Series([0.0, 1e-15, -1e-15, 1.0, np.nan, 2.0], index=idx)
    z = pd.Series([0.0, -1.0, 1e-15, 2.0, 3.0, np.nan], index=idx)
    return InMemorySeriesSource(data={"x": x, "y": y, "z": z})


def _run(source, expr, backend_name: str):
    return FactorEngine(backend=build_backend(backend_name), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    [
        ("protected_div", lambda: make_cleaned_call_factory("protected_div")(col("x"), col("y"))),
    ],
)
def test_protected_ops_polars_long_matches_pandas(edge_source, factory_name, expr_builder):
    expr = expr_builder()
    pd_out = _run(edge_source, expr, "pandas")
    long_out = _run(edge_source, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_protected_div_small_denominator_returns_default(edge_source):
    expr = make_cleaned_call_factory("protected_div")(col("x"), col("y"))
    out = _run(edge_source, expr, "polars_long")
    idx = edge_source.data["y"].index
    assert out.loc[idx[0]] == 0.0
    assert out.loc[idx[1]] == 0.0
    assert out.loc[idx[2]] == 0.0


def test_removed_protected_log_is_not_a_production_primitive(edge_source):
    from cleaned_operators.operator_surface import DAILY_CANONICALS, INTERNAL_ONLY_CANONICALS

    assert "protected_log" not in DAILY_CANONICALS | INTERNAL_ONLY_CANONICALS


def test_require_native_accepts_ts_rank(edge_source):
    import os

    expr = make_cleaned_call_factory("ts_rank")(col("x"), 3)
    os.environ["FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE"] = "1"
    try:
        out = FactorEngine(backend=build_backend("polars_long"), data_source=edge_source).run(
            Factor(name="t", expr=expr)
        )["result"]
        assert len(out) > 0
    finally:
        os.environ.pop("FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE", None)
