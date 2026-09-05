# -*- coding: utf-8
"""production 路径：plan / run_many DSL 门禁。"""

from __future__ import annotations

import os

import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor, FactorExecutionScopeHint
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators.operator_policy import normalize_bars_market
from factor_engine.runtime.engine import FactorEngine, PhysicalPlanRequiredError, _assert_backend_plan_authority
from factor_engine.runtime.production_policy import (
    ProductionPolicyViolation,
    assert_production_plan_ops,
    assert_production_factors,
)
from factor_engine.runtime.warmup_service import prepare_run_warmup
from tests.helpers import InMemorySeriesSource


shuffle = make_cleaned_call_factory("shuffle")
def _prod_factor(name, expr, source_expr):
    """Production factors in tests must carry an explicit market (R40 #174)."""
    return Factor(
        name=name,
        expr=expr,
        source_expr=source_expr,
        semantic_identity=FactorExecutionScopeHint(market="ashare"),
    )



def test_hybrid_backend_rejects_logical_plan_without_physical_consumer():
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.backend.hybrid_backend import HybridBackend

    with pytest.raises(PhysicalPlanRequiredError, match="PhysicalRegionPlan"):
        _assert_backend_plan_authority(HybridBackend(), PlanNode("column"))


def test_normalize_bars_market_maps_ashare_to_cn():
    assert normalize_bars_market("ashare") == "CN"
    assert normalize_bars_market("us") == "US"


def test_production_compile_blocks_shuffle():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    idx = __import__("pandas").MultiIndex.from_product(
        [__import__("pandas").date_range("2024-01-01", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = __import__("pandas").Series([1.0, 2.0, 3.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        eng = FactorEngine(backend=build_backend("pandas"), data_source=source, run_mode="production")
        factor = _prod_factor("bad", shuffle(col("close"), 1), "shuffle(close, 1)")
        with pytest.raises(KeyError, match="unknown operator canonical"):
            eng.compile(factor)
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)


def test_production_run_many_validates_source_expr():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.api import ts_mean

    ensure_cleaned_loaded()
    idx = __import__("pandas").MultiIndex.from_product(
        [__import__("pandas").date_range("2024-01-01", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = __import__("pandas").Series([1.0, 2.0, 3.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        eng = FactorEngine(backend=build_backend("pandas"), data_source=source, run_mode="production")
        bad = _prod_factor("bad", shuffle(col("close"), 1), "shuffle(close, 1)")
        with pytest.raises(ProductionPolicyViolation, match="DSL 语法/兼容校验失败"):
            assert_production_factors([bad], mode="production", context="run_many")
        good = _prod_factor("ok", ts_mean(col("close"), 2), "ts_mean(close, 2)")
        assert_production_factors([good], mode="production", context="run_many")
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)


def test_production_rejects_source_expr_mismatch():
    from factor_engine.api import ts_mean

    factor = _prod_factor("mismatch", ts_mean(col("close"), 2), "ts_mean(close, 3)")
    with pytest.raises(ProductionPolicyViolation, match="does not match"):
        assert_production_factors([factor], mode="production")


def test_warmup_uses_market_bars_per_day():
    from factor_engine.ir.analyzer import Analyzer

    class _DS:
        dataset = "ashare_stock_minute"
        bar_freq = "5m"

        def get_date_bounds(self):
            return "2024-01-02", "2024-01-03"

    class _Eng:
        data_source = _DS()
        cache = None

    factor = Factor(name="t", expr=col("close"), freq="5m", universe="ASHARE")
    analysis = Analyzer().lower(col("close"))
    ctx = prepare_run_warmup(_Eng(), factor, analysis, auto_warmup=False, trim_warmup=True, market=None)
    assert ctx.bars_per_day == 48
