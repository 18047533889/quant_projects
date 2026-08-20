"""Phase A：run_many 生产路径、rolling CSE、cache 模块、production 读门禁。"""

from __future__ import annotations

import pandas as pd
import pytest

from api import rank, ts_mean, ts_std, ts_std_dev
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from ir.analyzer import Analyzer
from planner.cse import apply_cse
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from planner.rolling_cse import apply_rolling_cse, rolling_semantic_key
from runtime.config import PipelineConfig
from runtime.engine import FactorEngine
from runtime.production_policy import ProductionPolicyViolation, assert_columns_explicit
from tests.helpers import InMemorySeriesSource


def _panel(n: int = 6) -> dict:
    dates = pd.bdate_range("2024-01-02", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    s = pd.Series([float(i + 1) for i in range(len(idx))], index=idx)
    return {"close": s}


def _walk(root: PlanNode):
    for c in root.inputs:
        yield from _walk(c)
    yield root


def _plan(expr):
    return Optimizer().optimize(Lowerer().to_logical_plan(Analyzer().lower(expr).ir))


def test_rolling_cse_merges_std_and_std_dev():
    col_node = PlanNode(op="column", attrs={"name": "close"}, inputs=[])
    win = PlanNode(op="literal", attrs={"value": 20}, inputs=[])
    std = PlanNode(op="ts_std", attrs={"window": 20}, inputs=[col_node, win])
    std_dev = PlanNode(op="ts_std_dev", attrs={"window": 20, "min_periods": 1}, inputs=[col_node, win])
    assert rolling_semantic_key(std) == rolling_semantic_key(std_dev)
    new_roots, rolling_shared = apply_rolling_cse([std, std_dev], existing_shared={})
    assert len(rolling_shared) >= 1
    assert any(n.op == "plan_ref" for n in _walk(new_roots[1]))


def test_rolling_cse_shared_across_factors():
    sub = ts_mean(col("close"), 3)
    f1 = _plan(sub)
    f2 = _plan(rank(sub))
    roots, shared = apply_cse([f1, f2])
    assert len(shared) >= 1
    roots2, shared2 = apply_rolling_cse(roots, existing_shared=shared)
    assert len(shared2) >= len(shared)


def test_production_run_requires_all_flags(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data=_panel()),
        run_mode="production",
    )
    f = Factor(
        name="m",
        expr=rank(ts_mean(col("close"), 2)),
        source_expr="rank(ts_mean(close, 2))",
    )
    with pytest.raises(ProductionPolicyViolation):
        eng.run(f, input_dq_check=True, auto_warmup=False, pit_enforce=False)
    out = eng.run(
        f,
        input_dq_check=True,
        auto_warmup=True,
        pit_enforce=True,
    )
    assert "result" in out
    assert out["analysis"].referenced_columns


def test_pipeline_config_batched_engine():
    assert PipelineConfig(batched_engine=True).batched_engine is True


def test_store_production_requires_columns(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    from data_access.read.query_budget import resolve_query_budget, validate_query_request

    budget = resolve_query_budget(None)
    assert budget.require_columns is True
    with pytest.raises(Exception):
        validate_query_request(budget, columns=None, time_range=None)


def test_assert_columns_explicit_production():
    with pytest.raises(ProductionPolicyViolation):
        assert_columns_explicit(None, mode="production")


def test_expression_and_panel_cache_modules():
    from cache.expression_cache import ExpressionCache
    from cache.panel_cache import PanelCache, series_panel_cache_key

    expr = ExpressionCache()
    expr.set("sid1", 42)
    assert expr.get("sid1") == 42

    s = pd.Series([1.0, 2.0])
    panel = PanelCache()
    panel.set_for_series(s, "panel")
    assert panel.get_for_series(s) == "panel"
    assert series_panel_cache_key(s) == series_panel_cache_key(s)
