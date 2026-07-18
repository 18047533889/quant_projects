# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER10：价量 / 截面百分位 parity CI。"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skip(reason="legacy Polars rollout tier superseded by six-way evidence certification")

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED_TIER10,
    POLARS_PRODUCTION_SAFE,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 21.0, 20.5, 22.0], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    x = pd.Series([1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "ret": ret, "x": x})


def _run_pair(source, expr):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
        a = eng_pd.run(Factor(name="t", expr=expr))["result"]
        b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
    return a, b


def _parity_exprs():
    return {
        "log_returns": make_cleaned_call_factory("log_returns")(col("close")),
        "volatility": make_cleaned_call_factory("volatility")(col("ret"), 2),
        "rank_pct": make_cleaned_call_factory("rank_pct")(col("x")),
        "cs_pct_rank": make_cleaned_call_factory("cs_pct_rank")(col("x")),
        "cs_quantile": make_cleaned_call_factory("cs_quantile")(col("x"), 0.5),
    }


def test_parity_tier10_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER10 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER10))
def test_polars_parity_verified_tier10(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
