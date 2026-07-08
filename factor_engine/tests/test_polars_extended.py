# -*- coding: utf-8
"""扩展 Polars 算子：分组 / TA / 扩展窗口。"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api import MACD, RSI, rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def panel_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8, freq="D"), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(range(10, 10 + len(idx)), index=idx, dtype=float)
    industry = pd.Series([1, 1, 2] * (len(idx) // 3), index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close, "industry": industry})


def test_polars_backends_registered():
    load_all()
    for name in (
        "group_rank",
        "group_mean",
        "group_zscore",
        "ts_ema",
        "WMA",
        "RSI",
        "MACD",
        "prev",
        "expanding_mean",
    ):
        assert "polars" in OperatorRegistry.backends_for(name), name


def test_group_rank_polars_matches_pandas(panel_source):
    from api.cleaned_ops import make_cleaned_call_factory

    group_rank = make_cleaned_call_factory("group_rank")
    expr = group_rank(col("close"), col("industry"))
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    pd.testing.assert_series_equal(
        eng_pd.run(Factor(name="t", expr=expr))["result"],
        eng_pl.run(Factor(name="t", expr=expr))["result"],
        check_names=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_rsi_polars_matches_pandas(panel_source):
    expr = RSI(col("close"), 5)
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    pd.testing.assert_series_equal(
        eng_pd.run(Factor(name="t", expr=expr))["result"],
        eng_pl.run(Factor(name="t", expr=expr))["result"],
        check_names=False,
        rtol=1e-4,
        atol=1e-4,
    )


def test_macd_polars_matches_pandas(panel_source):
    expr = MACD(col("close"))
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    pd.testing.assert_series_equal(
        eng_pd.run(Factor(name="t", expr=expr))["result"],
        eng_pl.run(Factor(name="t", expr=expr))["result"],
        check_names=False,
        rtol=1e-4,
        atol=1e-4,
    )
