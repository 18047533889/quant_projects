"""PolarsBackend 与 PandasBackend 数值对齐（时序 Polars + 截面 pandas 回退）。"""

from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api import exp, rank, ts_delay, ts_mean, zscore
from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

abs_ = make_cleaned_call_factory("abs")
sin = make_cleaned_call_factory("sin")
cos = make_cleaned_call_factory("cos")


def _panel():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([1.0, 2.0, 4.0, 10.0, 20.0], index=idx)
    return x


@pytest.fixture
def source():
    return InMemorySeriesSource(data={"x": _panel()})


def _run(engine: FactorEngine, expr):
    return engine.run(Factor(name="t", expr=expr))["result"]


@pytest.mark.parametrize(
    "expr",
    [
        rank(col("x")),
        ts_mean(col("x"), 2),
        col("x") + 1.0,
        -1 * col("x"),
        sin(col("x")),
        cos(col("x")),
        exp(col("x")),
        zscore(col("x")),
        ts_delay(col("x"), 1),
        abs_(col("x")),
    ],
)
def test_polars_matches_pandas(source, expr):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)

    a = _run(eng_pd, expr)
    b = _run(eng_pl, expr)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-12, atol=1e-12)


def test_polars_rank_ts_mean_composite_runs(source):
    """混合路径（ts_mean→polars, rank→pandas）可运行；截面秩与全 pandas 路径可能略有差异。"""
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
    result = _run(eng_pl, rank(ts_mean(col("x"), 2)))
    assert len(result) == 5


def test_polars_lazy_matches_eager(source):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        expr = ts_mean(col("x"), 2)
        eng_e = FactorEngine(backend=build_backend("polars"), data_source=source)
        eng_l = FactorEngine(backend=build_backend("polars_lazy"), data_source=source)
        a = _run(eng_e, expr)
        b = _run(eng_l, expr)
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)

    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-12, atol=1e-12)
