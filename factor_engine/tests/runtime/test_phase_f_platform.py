"""Phase F：production fallback 告警、cost_summary、cutover patch。"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from api import rank
from api.columns import col
from api.factor import Factor
from backend.context import ExecutionContext
from backend.pandas_backend import PandasBackend
from backend.polars_backend import PolarsBackend
from planner.cost_summary import summarize_plans
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from runtime.production_policy import (
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


def test_emit_cutover_patch_script():
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2].parent
        / "data_access"
        / "scripts"
        / "emit_bucket_cutover_patch.py"
    )
    if not script.exists():
        pytest.skip("emit script missing")
    sys.path.insert(0, str(script.parents[1].parent))
    from data_access.scripts.emit_bucket_cutover_patch import build_cutover_patch

    patch = build_cutover_patch(
        dataset="ashare_stock_minute",
        config=None,
        target_root="/data/bucket/minute",
    )
    assert "partition_columns" in patch
    assert "bucket" in patch["yaml_snippet"]
