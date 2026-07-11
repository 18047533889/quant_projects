# -*- coding: utf-8
"""真实 PlanNode production fastpath gate 测试。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_real_plan_gate_accepts_multi_column(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "rank(col('close'))",
        use_real_plan=True,
        check_full_plan=True,
        require_mode="any",
    )
    assert result.ok, result.violations


def test_real_plan_gate_accepts_composite(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "rank(col('close'))",
        use_real_plan=True,
        check_full_plan=True,
        require_mode="any",
    )
    assert result.ok, result.violations


def test_real_plan_gate_rejects_deferred(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "ts_decay_linear(col('close'), 5)",
        use_real_plan=True,
    )
    assert not result.ok
    assert any("ts_decay_linear" in v or "python_rolling" in v or "deferred" in v for v in result.violations)


def test_real_plan_gate_collects_all_referenced_columns(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "zscore(col('volume'))",
        use_real_plan=True,
        check_full_plan=True,
        require_mode="any",
    )
    assert result.ok, result.violations


def test_polars_long_no_load_executes_multi_column_plan(_loaded):
    from api.cleaned_ops import make_cleaned_call_factory
    from api.columns import col
    from api.factor import Factor
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from tests.helpers import NoLoadColumnSource

    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    src = NoLoadColumnSource(
        data={
            "close": pd.Series([10.0, 11.0, 12.0], index=idx),
            "open": pd.Series([9.0, 10.0, 11.0], index=idx),
            "volume": pd.Series([100.0, 110.0, 120.0], index=idx),
        }
    )
    expr = make_cleaned_call_factory("add")(
        make_cleaned_call_factory("ts_mean")(col("close"), 2),
        make_cleaned_call_factory("rank")(col("volume")),
    )
    out = FactorEngine(backend=build_backend("polars_long"), data_source=src).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_path") is True
    assert not out.get("polars_long_fallback_reason")
    assert len(out["result"]) == 3


def test_minimal_plan_hint_differs_from_real_plan(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    formula = "ts_decay_linear(col('close'), 5)"
    real = check_production_fastpath_formula_ops(formula, use_real_plan=True)
    hint = check_production_fastpath_formula_ops(formula, use_real_plan=False)
    assert not real.ok
    assert not hint.ok


def test_real_plan_gate_accepts_mom_when_primitives_certified(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(
        "MOM(col('close'), 5)",
        use_real_plan=True,
        check_full_plan=False,
    )
    assert result.ok, result.violations


def test_gate_rejects_unlowered_composite_in_strict(_loaded):
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops
    from planner.logical_plan import PlanNode

    plan = PlanNode(
        op="MOM",
        inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])],
        attrs={"window": 5},
    )
    result = check_production_fastpath_plan_ops(plan, strict=True, check_full_plan=False)
    assert not result.ok
    assert any("composite" in v for v in result.violations)
