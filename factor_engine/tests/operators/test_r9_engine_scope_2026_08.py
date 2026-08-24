# -*- coding: utf-8 -*-
"""R9-P0-011 / R9-P0-013 engine execution-scope regression tests (2026-08-09).

Covers:

* R9-P0-011 — ``FactorExecutionScope`` must enter the execution contract.
  A factor declaring a scoped universe (e.g. ``universe="CSI300"``) whose plan
  computes a cross-sectional operator (rank/zscore/neutralize/group/
  CS-regression/kNN) must NOT silently compute over the FULL source.
  ``FactorPlan`` now carries ``execution_scope`` and
  ``assert_execution_scope_contract`` is the fail-closed gate wired into the
  engine ``_dag_from_factors`` / ``run`` paths.
* R9-P0-013 — ``_scope_from_factor`` must not raise AttributeError for a minimal
  ``Factor`` that only has ``name/expr/freq/universe``; the
  ``decision_time_policy`` ``getattr(..., "")`` default is correct and is left
  as-is.

NOTE (2026-08-09): constructing a live ``FactorEngine`` here is not feasible in
this workspace — ``cleaned_operators.load_all()`` (a hard dependency of
``PandasBackend`` / the DSL-operator parser path) blocks on an environment lock
held by a concurrent session.  The review explicitly allows testing the gate
function directly instead, so the fail-closed contract is exercised here with
hand-built ``PlanNode`` / ``FactorExecutionScope`` values (no operator registry
lookup required).
"""
from __future__ import annotations

import inspect

import pytest

from factor_engine.planner.dag import FactorExecutionScope, FactorPlan
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import (
    _plan_has_cross_sectional_ops,
    _scope_from_factor,
    assert_execution_scope_contract,
)
from factor_engine.runtime.production_policy import ProductionPolicyViolation


def _column(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _rank_plan() -> PlanNode:
    return PlanNode(op="rank", inputs=(_column(),))


def _zscore_plan() -> PlanNode:
    return PlanNode(op="zscore", inputs=(_column(),))


def _group_plan() -> PlanNode:
    return PlanNode(op="group_mean", attrs={"group": "industry"}, inputs=(_column(),))


def _cs_plan() -> PlanNode:
    return PlanNode(op="cs_regression", inputs=(_column(), _column("ret_1")))


def _neutralize_plan() -> PlanNode:
    return PlanNode(op="neutralize", inputs=(_column(), _column("industry")))


def _ts_plan() -> PlanNode:
    return PlanNode(op="ts_mean", attrs={"window": 2}, inputs=(_column(),))


# --- R9-P0-011: cross-sectional op detection -------------------------------


@pytest.mark.parametrize(
    "plan",
    [_rank_plan(), _zscore_plan(), _group_plan(), _cs_plan(), _neutralize_plan()],
    ids=["rank", "zscore", "group_mean", "cs_regression", "neutralize"],
)
def test_plan_has_cross_sectional_ops_detected(plan):
    assert _plan_has_cross_sectional_ops(plan) is True


def test_plan_has_cross_sectional_ops_false_for_ts_and_none():
    assert _plan_has_cross_sectional_ops(_ts_plan()) is False
    assert _plan_has_cross_sectional_ops(None) is False
    assert _plan_has_cross_sectional_ops(_column()) is False


# --- R9-P0-011: fail-closed execution-contract gate ------------------------


def test_scoped_universe_rank_on_full_source_raises():
    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            FactorExecutionScope(universe_id="CSI300", market="A"),
            _rank_plan(),
            factor_name="cs300_rank",
        )
    assert "CSI300" in str(exc.value)
    assert "cross-sectional" in str(exc.value)


def test_all_universe_allows_cross_sectional():
    # universe="ALL" (the default) is the whole market -> cross-section is correct.
    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="ALL", market="A"),
        _rank_plan(),
        factor_name="f",
    )


def test_scoped_universe_without_cross_sectional_allowed():
    # A time-series-only plan on a scoped universe has no cross-section risk.
    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="CSI300", market="A"),
        _ts_plan(),
        factor_name="cs300_ts",
    )


def test_scoped_universe_allowed_when_data_source_scoped():
    class _ScopedSource:
        instrument_filter = ["600000.SH", "600036.SH"]

    # The data layer already restricts the instrument set -> the cross-section
    # runs over the scoped pool, not the full source.
    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="CSI300", market="A"),
        _rank_plan(),
        factor_name="f",
        data_source=_ScopedSource(),
    )


def test_scoped_universe_allowed_when_long_source_wraps_scoped_inner():
    class _ScopedInner:
        instrument_filter = ["600000.SH"]

    class _LongWrapper:
        inner = _ScopedInner()

    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="CSI300", market="A"),
        _rank_plan(),
        factor_name="f",
        data_source=_LongWrapper(),
    )


@pytest.mark.parametrize(
    "scope",
    [
        FactorExecutionScope(universe_id="ASHARE_ALL", market="A"),
        FactorExecutionScope(universe_id="A", market="A"),
        FactorExecutionScope(universe_id="US", market="US"),
    ],
    ids=["ASHARE_ALL", "A==market", "US==market"],
)
def test_whole_market_universe_labels_allowed(scope):
    # Whole-market universe labels do not represent a scoped subset, so a
    # cross-sectional op over the (whole-market) source is correct.
    # R13-P1-02: ``ASHARE_DAILY`` is NO LONGER whole-market (infer_market branch
    # removed) — a named pool is a scoped subset, conservatively fail-closed.
    assert_execution_scope_contract(scope, _rank_plan(), factor_name="f")


def test_market_mismatch_still_raises():
    # universe="US" on an A-market source is a genuine mismatch -> fail-closed.
    with pytest.raises(ProductionPolicyViolation):
        assert_execution_scope_contract(
            FactorExecutionScope(universe_id="US", market="A"),
            _rank_plan(),
            factor_name="us_on_a",
        )


# --- R9-P0-011: FactorPlan carries execution_scope -------------------------


def test_factor_plan_carries_execution_scope():
    scope = FactorExecutionScope(universe_id="CSI300", market="A", frequency="1d")
    fp = FactorPlan(factor_name="cs300", root=_rank_plan(), execution_scope=scope)
    assert fp.execution_scope is scope
    assert fp.execution_scope.universe_id == "CSI300"


def test_factor_plan_default_execution_scope_is_all():
    fp = FactorPlan(factor_name="plain", root=_ts_plan())
    assert isinstance(fp.execution_scope, FactorExecutionScope)
    assert fp.execution_scope.universe_id == "ALL"


def test_gate_wired_into_dag_and_run_paths():
    """The fail-closed gate is wired into the batch and single-factor paths."""
    from factor_engine.runtime import engine as engine_mod

    src_dag = inspect.getsource(engine_mod.FactorEngine._dag_from_factors)
    src_run = inspect.getsource(engine_mod.FactorEngine.run)
    assert "assert_execution_scope_contract(" in src_dag
    assert "assert_execution_scope_contract(" in src_run
    assert "execution_scope=" in src_dag  # FactorPlan carries the scope


# --- R9-P0-013: _scope_from_factor robustness ------------------------------


def test_scope_from_factor_minimal_factor_no_attribute_error():
    from factor_engine.api.factor import Factor
    from factor_engine.expr.base import Expr

    factor = Factor(name="min", expr=Expr(), freq="1d", universe="CSI300")
    scope = _scope_from_factor(factor)
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI300"
    # R13-P1-01: no market attribute -> keep "" (never silently default "A").
    assert scope.market == ""
    assert scope.calendar_id == ""
    assert scope.source_scope_hash == ""
    assert scope.decision_time_policy == ""


def test_scope_from_factor_none_universe_is_all():
    from factor_engine.api.factor import Factor
    from factor_engine.expr.base import Expr

    factor = Factor(name="plain", expr=Expr())
    scope = _scope_from_factor(factor)
    assert scope.universe_id == "ALL"
    # R9-P0-013: a Factor without decision_time_policy keeps the "" default.
    assert scope.decision_time_policy == ""


# --- R9-P0-011: engine-path fail-closed (requires loaded operator registry) --
# ``tests/operators/conftest.py`` forces ``cleaned_operators.load_all()`` as a
# session autouse fixture, so within a pytest session of this directory the
# registry is already frozen when these run.  The guard below is a safety net
# for direct-file invocation: skip instead of hanging on ``load_all``.


def _registry_loaded() -> bool:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    try:
        return OperatorRegistry.lifecycle() == "frozen"
    except Exception:
        return False


def _build_research_engine():
    import numpy as np
    import pandas as pd

    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["A", "B", "C"]],
        names=["date", "inst"],
    )
    close = pd.Series(np.arange(len(idx), dtype=float), index=idx, name="close")
    return FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
        run_mode="research",
    )


def test_engine_run_many_scoped_rank_fails_closed():
    if not _registry_loaded():
        pytest.skip("operator registry not loaded; cannot build a FactorEngine")
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.api.factor import Factor

    eng = _build_research_engine()
    factor = Factor(name="cs300_rank", expr=parse_expr("rank(close)"), universe="CSI300")
    with pytest.raises(ProductionPolicyViolation):
        eng.run_many([factor])


def test_engine_run_scoped_rank_fails_closed():
    if not _registry_loaded():
        pytest.skip("operator registry not loaded; cannot build a FactorEngine")
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.api.factor import Factor

    eng = _build_research_engine()
    factor = Factor(name="cs300_rank", expr=parse_expr("rank(close)"), universe="CSI300")
    with pytest.raises(ProductionPolicyViolation):
        eng.run(factor)


def test_engine_compile_many_carries_execution_scope():
    if not _registry_loaded():
        pytest.skip("operator registry not loaded; cannot build a FactorEngine")
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.api.factor import Factor

    eng = _build_research_engine()
    # Non-cross-sectional factor on a scoped universe: compile_many must succeed
    # and carry the execution_scope into the FactorPlan.
    factor = Factor(name="cs300_ts", expr=parse_expr("ts_mean(close, 2)"), universe="CSI300")
    dag = eng.compile_many([factor])
    assert dag.roots[0].execution_scope.universe_id == "CSI300"
    assert dag.roots[0].factor_name == "cs300_ts"
