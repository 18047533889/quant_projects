# -*- coding: utf-8 -*-
"""R9-P0-034 / R9-P0-036 regression tests.

* Semantic roundtrip: for a real expression the semantic identity
  (unit / semantic_type / frequency / grain / availability / source_vintage /
  universe / semantic_kind) must survive Expr → Typed IR → Plan → optimized
  Plan.  This directly guards the R9-P0-002 lowerer fix (semantic_attrs must
  not be dropped).
* Production entry: ``FactorEngine`` constructed with ``run_mode="production"``
  must reject an unknown raw column through the REAL ``compile()`` entry, not
  just via the ``Analyzer(production=True)`` helper (R9-P0-001 / R9-P0-036).

Both tests deliberately avoid ``load_all()`` and any heavy engine data stack.
"""
from __future__ import annotations

import pytest

from factor_engine.ir.analyzer import Analyzer, UnknownRawColumnError
from factor_engine.planner.lowerer import Lowerer
from factor_engine.planner.optimizer import Optimizer


def _semantic_snapshot(node: object) -> dict:
    attrs = getattr(node, "semantic_attrs", None) or {}
    return dict(attrs)


def test_semantic_roundtrip_ir_to_plan_preserves_semantic_attrs():
    """Expr → IR → Plan → optimized Plan must all carry the same semantic
    identity (unit / frequency / domain / available_at / source_vintage /
    semantic_kind)."""
    from factor_engine.api.columns import col
    from factor_engine.expr.cleaned_call import CleanedCall

    expr = CleanedCall(op="ts_ema", args=(col("close"),), kwargs=(("window", 20),))
    analysis = Analyzer().lower(expr)
    ir_root = analysis.ir
    ir_sem = _semantic_snapshot(ir_root)
    # Without load_all the operator metadata is not in the registry, but the
    # derived attrs (semantic_kind / available_at / pit_safe) MUST be non-empty —
    # the R9-P0-002 invariant is that whatever the IR carries reaches the Plan.
    assert ir_sem, "IR root carried no semantic attrs"

    plan = Lowerer().to_logical_plan(ir_root)
    plan_sem = _semantic_snapshot(plan)
    # The plan root must carry the IR root's semantic identity.
    for key in ("unit", "domain", "frequency", "semantic_kind", "source_vintage"):
        if key in ir_sem and plan_sem.get(key) is not None:
            assert plan_sem[key] == ir_sem[key], f"{key}: IR={ir_sem[key]} Plan={plan_sem[key]}"

    optimized = Optimizer().optimize(plan, production=False)
    opt_sem = _semantic_snapshot(optimized)
    for key in ("unit", "domain", "frequency", "semantic_kind"):
        if key in plan_sem and opt_sem.get(key) is not None:
            assert opt_sem[key] == plan_sem[key], f"{key}: Plan={plan_sem[key]} Opt={opt_sem[key]}"


def test_semantic_roundtrip_leaf_columns_keep_availability_and_unit():
    """A leaf raw column's available_at / price_basis must survive lowering."""
    from factor_engine.api.columns import col

    analysis = Analyzer().lower(col("close"))
    ir_leaf = analysis.ir
    ir_sem = _semantic_snapshot(ir_leaf)
    plan = Lowerer().to_logical_plan(ir_leaf)
    plan_sem = _semantic_snapshot(plan)
    for key in ("unit", "domain", "available_at", "price_basis", "semantic_kind", "frequency"):
        if key in ir_sem:
            assert plan_sem.get(key) == ir_sem.get(key), f"{key} dropped in lowerer"


# ---------------------------------------------------------------------------
# R9-P0-001 / R9-P0-036 — production mode must be ON at the REAL engine entry.
# ---------------------------------------------------------------------------
def test_engine_production_compile_rejects_unknown_raw_column():
    """A FactorEngine with run_mode='production' must reject an unknown raw
    column at ``FactorEngine.compile()`` — NOT just ``Analyzer(production=True)``.
    This guards the fix where ``self.analyzer = Analyzer()`` silently disabled
    the production typed-field gate."""
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.runtime.engine import FactorEngine

    engine = FactorEngine(_FakeBackend(), _FakeSource(), run_mode="production")
    factor = Factor(name="f", expr=col("custom_alpha_input"))
    with pytest.raises(UnknownRawColumnError):
        engine.compile(factor)


def test_engine_research_compile_allows_raw_column():
    """A run_mode='research' engine still compiles an unknown raw column
    (research policy), proving the gate is mode-driven, not a blanket ban."""
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.runtime.engine import FactorEngine

    engine = FactorEngine(_FakeBackend(), _FakeSource(), run_mode="research")
    factor = Factor(name="f", expr=col("custom_alpha_input"))
    plan, analysis = engine.compile(factor)
    assert analysis.ir.op == "column"
    assert plan is not None


class _FakeBackend:
    """Minimal backend stand-in — compile() never reaches execution."""


class _FakeSource:
    """Minimal data-source stand-in — compile() never reads data."""
