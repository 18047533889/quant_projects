# -*- coding: utf-8 -*-
"""SQL lowerer / Polars SAFE / hybrid speed-path regression tests."""
from __future__ import annotations

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.sql_lowerer import lower_to_physical_plan


def test_lowerer_does_not_extract_bare_column_under_python_parent() -> None:
    """Non-SQL root with a column leaf must not DuckDB-round-trip the leaf."""
    from factor_engine.cleaned_operators import load_all

    load_all()

    # Synthetic non-SQL parent; column is production SQL-safe but leaf-only.
    plan = PlanNode(
        op="__python_only__",
        inputs=(PlanNode(op="column", inputs=(), attrs={"name": "Close"}),),
        attrs={},
    )
    physical = lower_to_physical_plan(plan, mode="production")
    assert not physical.fully_sql
    assert physical.sql_subtrees == {}
    assert physical.root.op == "__python_only__"
    assert physical.root.inputs[0].op == "column"


def test_lowerer_still_extracts_non_leaf_sql_subtree() -> None:
    from factor_engine.cleaned_operators import load_all

    load_all()

    mean = PlanNode(
        op="ts_mean",
        inputs=(PlanNode(op="column", inputs=(), attrs={"name": "Close"}),),
        attrs={"window": 5},
    )
    plan = PlanNode(op="__python_only__", inputs=(mean,), attrs={})
    physical = lower_to_physical_plan(plan, mode="production")
    assert not physical.fully_sql
    assert len(physical.sql_subtrees) == 1
    sub = next(iter(physical.sql_subtrees.values()))
    assert sub.op == "ts_mean"


def test_bare_column_root_still_fully_sql() -> None:
    from factor_engine.cleaned_operators import load_all

    load_all()

    plan = PlanNode(op="column", inputs=(), attrs={"name": "Close"})
    physical = lower_to_physical_plan(plan, mode="production")
    assert physical.fully_sql
    assert physical.sql_subtrees == {}


def test_cs_aggregates_remain_polars_production_safe_after_rename() -> None:
    from factor_engine.cleaned_operators import load_all, operator_policy
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    for name in ("cs_mean", "cs_sum", "cs_count", "cs_std"):
        assert name in operator_policy.POLARS_PRODUCTION_SAFE, name
        assert "polars" in OperatorRegistry.backends_for(name), name


def test_hybrid_auto_upgrades_to_long_when_scan_available() -> None:
    from factor_engine.backend.factory import build_backend
    from factor_engine.backend.hybrid_backend import HybridBackend
    from factor_engine.backend.hybrid_long_backend import HybridLongBackend

    assert isinstance(build_backend("auto"), HybridBackend)
    assert isinstance(build_backend("auto_long"), HybridLongBackend)


def test_hybrid_auto_execute_uses_long_path_when_scan_present(monkeypatch) -> None:
    from factor_engine.backend.hybrid_backend import HybridBackend
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.planner.logical_plan import PlanNode

    calls: list[str] = []

    class _DS:
        def scan_polars_long(self, columns):  # pragma: no cover - not executed
            raise AssertionError("scan should not run in this unit test")

    class _Long:
        runtime_backend_label = "hybrid_long"

        def execute(self, plan, ctx):
            calls.append("long")
            return "ok-long"

    backend = HybridBackend()
    monkeypatch.setattr(backend, "_long_backend", lambda: _Long())
    monkeypatch.setattr(backend, "_sql", type("S", (), {"execute": staticmethod(lambda p, c: calls.append("wide") or "ok-wide")})())

    ctx = ExecutionContext(data_source=_DS())
    plan = PlanNode(op="column", inputs=(), attrs={"name": "Close"})
    assert backend.execute(plan, ctx) == "ok-long"
    assert calls == ["long"]


def test_hybrid_auto_execute_stays_wide_without_scan(monkeypatch) -> None:
    from factor_engine.backend.hybrid_backend import HybridBackend
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.planner.logical_plan import PlanNode

    calls: list[str] = []

    class _DS:
        pass

    backend = HybridBackend()
    monkeypatch.setattr(
        backend,
        "_sql",
        type("S", (), {"execute": staticmethod(lambda p, c: calls.append("wide") or "ok-wide")})(),
    )
    ctx = ExecutionContext(data_source=_DS())
    plan = PlanNode(op="column", inputs=(), attrs={"name": "Close"})
    assert backend.execute(plan, ctx) == "ok-wide"
    assert calls == ["wide"]
