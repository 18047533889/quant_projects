"""Phase F：production fallback 告警、cost_summary、cutover patch。"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from factor_engine.api import rank
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.backend.polars_backend import PolarsBackend
from factor_engine.planner.cost_summary import summarize_plans
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.production_policy import (
    ProductionPolicyViolation,
    assert_no_production_pandas_fallbacks,
    record_production_pandas_fallback,
)
from tests.helpers import InMemorySeriesSource


def _panel():
    dates = pd.bdate_range("2024-01-02", periods=5)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    return {"close": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)}


def test_record_production_pandas_fallback():
    ctx = ExecutionContext(data_source=object(), runtime_stats={})
    record_production_pandas_fallback(
        ctx,
        op="rank",
        requested_backend="polars",
        actual_backend="pandas_numpy",
        mode="production",
    )
    assert ctx.runtime_stats["production_pandas_fallbacks"]


def test_strict_polars_fallback_raises(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_STRICT_POLARS", "1")
    ctx = ExecutionContext(
        data_source=object(),
        runtime_stats={
            "production_pandas_fallbacks": [
                {"op": "rank", "requested": "polars", "actual": "pandas_numpy"}
            ]
        },
    )
    with pytest.raises(ProductionPolicyViolation):
        assert_no_production_pandas_fallbacks(ctx, mode="production")


def test_cost_summary_tier_histogram():
    col_node = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    win = PlanNode(op="literal", attrs={"value": 3}, inputs=[])
    root = PlanNode(op="ts_mean", attrs={"window": 3}, inputs=[col_node, win])
    summary = summarize_plans({"f1": root})
    assert summary["factor_count"] == 1
    assert "tier_histogram" in summary


def test_run_many_includes_cost_summary():
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=_panel()),
    )
    factors = [
        Factor(name="a", expr=rank(col("close"))),
        Factor(name="b", expr=col("close")),
    ]
    out = eng.run_many(factors)
    assert "cost_summary" in out
    assert out["cost_summary"]["factor_count"] == 2
