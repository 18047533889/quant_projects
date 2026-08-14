# -*- coding: utf-8 -*-
"""FE-P0-031 / FE-P0-032 execution scope production gates (2026-08-14).

FE-P0-031: Production cross-sectional operators must require validated
           UniverseSnapshotIdentity; universe missing/defaulted to "ALL"
           must fail-closed.

FE-P0-032: Production execution scope must not default frequency/calendar
           when absent; grain/frequency/calendar must be explicit or from
           validated canonical contract.

Owned files:
- runtime/engine.py (_scope_from_factor, assert_execution_scope_contract)
- This test file

These tests use hand-built Factor/PlanNode objects to avoid dependency on
operator registry locks in concurrent environments.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from api.factor import Factor
from expr.base import Expr
from planner.dag import FactorExecutionScope
from planner.logical_plan import PlanNode
from runtime.engine import _scope_from_factor, assert_execution_scope_contract
from runtime.production_policy import ProductionPolicyViolation


def _column(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _rank_plan() -> PlanNode:
    return PlanNode(op="rank", inputs=(_column(),))


def _ts_plan() -> PlanNode:
    return PlanNode(op="ts_mean", attrs={"window": 5}, inputs=(_column(),))


def _set_production_mode():
    """Force production mode via env."""
    os.environ["FACTOR_ENGINE_RUN_MODE"] = "production"


def _set_research_mode():
    """Force research mode via env."""
    os.environ["FACTOR_ENGINE_RUN_MODE"] = "research"


@pytest.fixture(autouse=True)
def reset_run_mode():
    """Reset run_mode env after each test."""
    original = os.environ.get("FACTOR_ENGINE_RUN_MODE")
    yield
    if original is None:
        os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)
    else:
        os.environ["FACTOR_ENGINE_RUN_MODE"] = original


# --- FE-P0-031: production cross-sectional must have explicit universe -------


def test_p0_031_production_cross_sectional_missing_universe_raises():
    """Production + cross-sectional + universe missing → fail-closed."""
    _set_production_mode()
    scope = FactorExecutionScope(universe_id="", market="A", frequency="1d")
    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            scope, _rank_plan(), factor_name="rank_no_univ"
        )
    assert "FE-P0-031" in str(exc.value)
    assert "UniverseSnapshotIdentity" in str(exc.value)
    assert "rank_no_univ" in str(exc.value)


def test_p0_031_production_cross_sectional_all_universe_raises():
    """Production + cross-sectional + universe="ALL" → fail-closed."""
    _set_production_mode()
    scope = FactorExecutionScope(universe_id="ALL", market="A", frequency="1d")
    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            scope, _rank_plan(), factor_name="rank_all"
        )
    assert "FE-P0-031" in str(exc.value)
    assert "ALL" in str(exc.value)


def test_p0_031_production_cross_sectional_explicit_universe_allowed():
    """Production + cross-sectional + explicit scoped universe + scoped data_source → allowed."""
    _set_production_mode()

    class _ScopedSource:
        instrument_filter = ["600000.SH", "600036.SH"]

    scope = FactorExecutionScope(universe_id="CSI300", market="A", frequency="1d")
    # No exception
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="rank_csi300", data_source=_ScopedSource()
    )


def test_p0_031_research_cross_sectional_all_universe_allowed():
    """Research mode allows cross-sectional with ALL universe (backward compat)."""
    _set_research_mode()
    scope = FactorExecutionScope(universe_id="ALL", market="A", frequency="1d")
    # No exception — research mode is permissive
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="rank_all_research"
    )


def test_p0_031_production_timeseries_all_universe_allowed():
    """Production + time-series only + ALL → allowed (no cross-sectional risk)."""
    _set_production_mode()
    scope = FactorExecutionScope(universe_id="ALL", market="A", frequency="1d")
    # No exception — no cross-sectional op
    assert_execution_scope_contract(
        scope, _ts_plan(), factor_name="ts_mean_all"
    )


# --- FE-P0-032: production must not default frequency ------------------------


def test_p0_032_production_missing_frequency_in_semantic_identity_raises():
    """Production + semantic_identity without frequency → fail-closed."""
    _set_production_mode()

    class _SemanticIdentity:
        market = "A"
        universe_id = "CSI300"
        frequency = None  # missing
        calendar_id = ""
        decision_time_policy = ""
        source_scope_hash = ""

    class _FactorWithSemanticIdentity:
        name = "no_freq"
        expr = Expr()
        semantic_identity = _SemanticIdentity()
        freq = None

    with pytest.raises(ValueError) as exc:
        _scope_from_factor(_FactorWithSemanticIdentity())
    assert "FE-P0-032" in str(exc.value)
    assert "frequency" in str(exc.value).lower()
    assert "no_freq" in str(exc.value)


def test_p0_032_production_missing_frequency_no_semantic_identity_raises():
    """Production + factor.freq missing (no semantic_identity) → fail-closed."""
    _set_production_mode()

    class _MinimalFactor:
        name = "plain_no_freq"
        expr = Expr()
        universe = "CSI300"
        freq = None
        semantic_identity = None

    with pytest.raises(ValueError) as exc:
        _scope_from_factor(_MinimalFactor())
    assert "factor.freq is missing" in str(exc.value) or "frequency" in str(exc.value).lower()
    assert "refusing to guess" in str(exc.value) or "P0-019" in str(exc.value) or "P0-032" in str(exc.value)


def test_p0_032_production_explicit_frequency_allowed():
    """Production + explicit frequency → allowed."""
    _set_production_mode()
    factor = Factor(name="with_freq", expr=Expr(), freq="1d", universe="CSI300")
    scope = _scope_from_factor(factor)
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI300"


def test_p0_032_research_missing_frequency_defaults_to_1d():
    """Research mode defaults missing frequency to '1d' (backward compat)."""
    _set_research_mode()

    class _MinimalFactor:
        name = "research_no_freq"
        expr = Expr()
        universe = "CSI300"
        freq = None
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI300"


def test_p0_032_production_frequency_from_semantic_identity_allowed():
    """Production + frequency in semantic_identity → allowed."""
    _set_production_mode()

    class _SemanticIdentity:
        market = "A"
        universe_id = "CSI500"
        frequency = "1d"
        calendar_id = "ASHARE"
        decision_time_policy = ""
        source_scope_hash = "abc123"

    class _FactorWithSemanticIdentity:
        name = "semantic_freq"
        expr = Expr()
        semantic_identity = _SemanticIdentity()
        freq = None  # overridden by semantic_identity

    scope = _scope_from_factor(_FactorWithSemanticIdentity())
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI500"
    assert scope.market == "A"


# --- Combined FE-P0-031 + FE-P0-032 scenario ---------------------------------


def test_p0_031_032_production_cross_sectional_missing_both_raises():
    """Production + cross-sectional + missing universe + missing frequency → both fail."""
    _set_production_mode()

    class _BrokenFactor:
        name = "broken"
        expr = Expr()
        freq = None
        universe = None
        semantic_identity = None

    # frequency check happens first in _scope_from_factor
    with pytest.raises(ValueError) as exc:
        _scope_from_factor(_BrokenFactor())
    assert "freq" in str(exc.value).lower()


def test_p0_031_032_production_complete_scope_allowed():
    """Production + explicit universe + explicit frequency → allowed."""
    _set_production_mode()

    class _ScopedSource:
        universe = "CSI300"

    factor = Factor(name="complete", expr=Expr(), freq="1d", universe="CSI300")
    scope = _scope_from_factor(factor)
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI300"

    # cross-sectional with scoped data_source
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="complete", data_source=_ScopedSource()
    )


# --- Negative tests: research mode backward compatibility --------------------


def test_research_mode_preserves_convenience_defaults():
    """Research mode still allows ALL/1d defaults for convenience."""
    _set_research_mode()

    class _MinimalFactor:
        name = "research_minimal"
        expr = Expr()
        freq = None
        universe = None
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.universe_id == "ALL"

    # cross-sectional on ALL is allowed in research
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="research_minimal"
    )
