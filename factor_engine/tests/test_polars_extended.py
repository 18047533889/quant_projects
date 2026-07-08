# -*- coding: utf-8
"""扩展 Polars 算子数值对齐。"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api import rank, ts_mean, ts_var, zscore
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def panel_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8, freq="D"), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(range(10, 10 + len(idx)), index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close})


def test_ts_var_polars_matches_pandas(panel_source):
    load_all()
    expr = ts_var(col("close"), 3)
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    pd.testing.assert_series_equal(
        eng_pd.run(Factor(name="t", expr=expr))["result"],
        eng_pl.run(Factor(name="t", expr=expr))["result"],
        check_names=False,
        rtol=1e-4,
        atol=1e-4,
    )


def test_if_else_polars_matches_pandas(panel_source):
    load_all()
    from api.cleaned_ops import make_cleaned_call_factory

    if_else = make_cleaned_call_factory("if_else")
    expr = if_else(rank(col("close")), col("close"), col("close") * 0.5)
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    pd.testing.assert_series_equal(
        eng_pd.run(Factor(name="t", expr=expr))["result"],
        eng_pl.run(Factor(name="t", expr=expr))["result"],
        check_names=False,
        rtol=1e-5,
        atol=1e-5,
    )
