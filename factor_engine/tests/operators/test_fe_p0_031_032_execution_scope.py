"""FE-P0-031/032: production execution scope fail-closed gates.

FE-P0-031: Production cross-sectional operators require validated
UniverseSnapshotIdentity (digest/members/policy_version), not a string name.
String universe like "CSI300" cannot prove membership snapshot or effective interval.

FE-P0-032: Production execution scope requires explicit frequency AND calendar_id
(grain/session determines bar boundaries). Minute/hour factors defaulted to daily
would cause under-read in warmup/history.
"""
import pytest

from planner.dag import FactorExecutionScope
from runtime.engine import _scope_from_factor, assert_execution_scope_contract
from runtime.production_policy import ProductionPolicyViolation


class Expr:
    """Minimal expression stub."""


def _set_production_mode(monkeypatch):
    """Set production mode via environment."""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")


def _set_research_mode(monkeypatch):
    """Set research mode via environment."""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "research")


def _rank_plan():
    """Minimal plan with cross-sectional operator (rank)."""
    from planner.logical_plan import PlanNode

    return PlanNode(
        op="rank",
        inputs=(),
        attrs={"kind": "cross_sectional"},
    )


def _ts_plan():
    """Minimal plan with time-series operator (ts_mean)."""
    from planner.logical_plan import PlanNode

    return PlanNode(
        op="ts_mean",
        inputs=(),
        attrs={"window": 5, "kind": "time_series"},
    )


def _scoped_data_source():
    """Mock scoped data_source with explicit universe scope."""

    class _ScopedSource:
        universe = "CSI500"

        def get_universe_filter(self):
            return {"universe": "CSI500"}

    return _ScopedSource()


# ============================================================================
# FE-P0-031: UniverseSnapshotIdentity validation
# ============================================================================


def test_p0_031_production_cross_sectional_missing_universe_raises(monkeypatch):
    """Production + cross-sectional + missing universe → fail."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "no_univ"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = None
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.universe_id == "ALL"

    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            scope, _rank_plan(), factor_name="no_univ", data_source=None
        )
    assert "ALL" in str(exc.value)
    assert "FE-P0-031" in str(exc.value)
    assert "UniverseSnapshotIdentity" in str(exc.value)


def test_p0_031_production_cross_sectional_all_universe_raises(monkeypatch):
    """Production + cross-sectional + explicit 'ALL' → fail."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "all_univ"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = "ALL"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())

    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            scope, _rank_plan(), factor_name="all_univ", data_source=None
        )
    assert "ALL" in str(exc.value)
    assert "FE-P0-031" in str(exc.value)


def test_p0_031_production_cross_sectional_string_universe_raises(monkeypatch):
    """Production + cross-sectional + string universe (e.g. 'CSI300') → fail.

    FE-P0-031: A mere universe string cannot prove membership snapshot/digest/
    effective interval. Must be validated UniverseSnapshotIdentity.
    """
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "string_univ"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = "CSI300"  # string name, not validated snapshot
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.universe_id == "CSI300"

    # Without scoped data_source, string universe is rejected
    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            scope, _rank_plan(), factor_name="string_univ", data_source=None
        )
    assert "CSI300" in str(exc.value)
    assert "FE-P0-031" in str(exc.value)
    assert "membership snapshot" in str(exc.value).lower()


def test_p0_031_production_cross_sectional_scoped_source_allowed(monkeypatch):
    """Production + cross-sectional + scoped data_source → allowed (interim gate)."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "scoped_univ"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = "CSI500"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())

    # With scoped data_source, currently allowed (documented interim solution)
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="scoped_univ", data_source=_scoped_data_source()
    )


def test_p0_031_research_cross_sectional_all_universe_allowed(monkeypatch):
    """Research + cross-sectional + ALL universe → allowed (backward compat)."""
    _set_research_mode(monkeypatch)

    class _MinimalFactor:
        name = "research_all"
        expr = Expr()
        freq = "1d"
        universe = "ALL"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())

    # Research mode allows ALL for convenience
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="research_all", data_source=None
    )


def test_p0_031_production_timeseries_all_universe_allowed(monkeypatch):
    """Production + time-series (no cross-sectional) + ALL → allowed."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "ts_all"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = "ALL"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())

    # Time-series operators don't require validated universe
    assert_execution_scope_contract(
        scope, _ts_plan(), factor_name="ts_all", data_source=None
    )


# ============================================================================
# FE-P0-032: frequency + calendar_id validation
# ============================================================================


def test_p0_032_production_missing_frequency_raises(monkeypatch):
    """Production + missing frequency → fail."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "no_freq"
        expr = Expr()
        freq = None
        calendar_id = "ASHARE"
        semantic_identity = None

    with pytest.raises(ProductionPolicyViolation) as exc:
        _scope_from_factor(_MinimalFactor())
    assert "freq" in str(exc.value).lower()
    assert "FE-P0-032" in str(exc.value)


def test_p0_032_production_missing_calendar_raises(monkeypatch):
    """Production + missing calendar_id → fail."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "no_calendar"
        expr = Expr()
        freq = "1d"
        calendar_id = None
        calendar = None
        semantic_identity = None

    with pytest.raises(ProductionPolicyViolation) as exc:
        _scope_from_factor(_MinimalFactor())
    assert "calendar" in str(exc.value).lower()
    assert "FE-P0-032" in str(exc.value)


def test_p0_032_production_semantic_identity_missing_frequency_raises(monkeypatch):
    """Production + semantic_identity without frequency → fail."""
    _set_production_mode(monkeypatch)

    class _SemanticIdentity:
        market = "A"
        universe_id = "CSI300"
        frequency = None  # missing
        calendar_id = "ASHARE"

    class _FactorWithSemanticIdentity:
        name = "semantic_no_freq"
        expr = Expr()
        semantic_identity = _SemanticIdentity()
        freq = None

    with pytest.raises(ProductionPolicyViolation) as exc:
        _scope_from_factor(_FactorWithSemanticIdentity())
    assert "frequency" in str(exc.value).lower()
    assert "FE-P0-032" in str(exc.value)


def test_p0_032_production_semantic_identity_missing_calendar_raises(monkeypatch):
    """Production + semantic_identity without calendar_id → fail."""
    _set_production_mode(monkeypatch)

    class _SemanticIdentity:
        market = "A"
        universe_id = "CSI300"
        frequency = "1d"
        calendar_id = None  # missing

    class _FactorWithSemanticIdentity:
        name = "semantic_no_calendar"
        expr = Expr()
        semantic_identity = _SemanticIdentity()
        calendar_id = None

    with pytest.raises(ProductionPolicyViolation) as exc:
        _scope_from_factor(_FactorWithSemanticIdentity())
    assert "calendar" in str(exc.value).lower()
    assert "FE-P0-032" in str(exc.value)


def test_p0_032_production_explicit_frequency_and_calendar_allowed(monkeypatch):
    """Production + explicit frequency + calendar_id → allowed."""
    _set_production_mode(monkeypatch)

    class _MinimalFactor:
        name = "explicit_scope"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.calendar_id == "ASHARE"


def test_p0_032_research_missing_frequency_defaults_to_1d(monkeypatch):
    """Research mode defaults missing frequency to '1d' (backward compat)."""
    _set_research_mode(monkeypatch)

    class _MinimalFactor:
        name = "research_no_freq"
        expr = Expr()
        freq = None
        universe = "CSI300"
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.universe_id == "CSI300"


def test_p0_032_research_missing_calendar_defaults_to_empty(monkeypatch):
    """Research mode defaults missing calendar to '' (backward compat)."""
    _set_research_mode(monkeypatch)

    class _MinimalFactor:
        name = "research_no_calendar"
        expr = Expr()
        freq = "1d"
        calendar_id = None
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.calendar_id == ""


def test_p0_032_production_frequency_and_calendar_from_semantic_identity_allowed(monkeypatch):
    """Production + frequency/calendar in semantic_identity → allowed."""
    _set_production_mode(monkeypatch)

    class _SemanticIdentity:
        market = "A"
        universe_id = "CSI500"
        frequency = "1d"
        calendar_id = "ASHARE"
        decision_time_policy = ""
        source_scope_hash = "abc123"

    class _FactorWithSemanticIdentity:
        name = "semantic_complete"
        expr = Expr()
        semantic_identity = _SemanticIdentity()
        freq = None  # overridden by semantic_identity
        calendar_id = None  # overridden by semantic_identity

    scope = _scope_from_factor(_FactorWithSemanticIdentity())
    assert scope.frequency == "1d"
    assert scope.calendar_id == "ASHARE"
    assert scope.universe_id == "CSI500"
    assert scope.market == "A"


# ============================================================================
# Combined scenarios
# ============================================================================


def test_p0_031_032_production_cross_sectional_missing_frequency_and_universe_raises(
    monkeypatch,
):
    """Production + cross-sectional + missing both → frequency check fails first."""
    _set_production_mode(monkeypatch)

    class _BrokenFactor:
        name = "broken"
        expr = Expr()
        freq = None
        calendar_id = "ASHARE"
        universe = None
        semantic_identity = None

    # frequency check happens first in _scope_from_factor
    with pytest.raises(ProductionPolicyViolation) as exc:
        _scope_from_factor(_BrokenFactor())
    assert "freq" in str(exc.value).lower()


def test_p0_031_032_production_complete_scope_with_scoped_source_allowed(monkeypatch):
    """Production + explicit frequency/calendar + scoped data_source → allowed."""
    _set_production_mode(monkeypatch)

    class _CompleteFactor:
        name = "complete"
        expr = Expr()
        freq = "1d"
        calendar_id = "ASHARE"
        universe = "CSI500"
        semantic_identity = None

    scope = _scope_from_factor(_CompleteFactor())
    assert scope.frequency == "1d"
    assert scope.calendar_id == "ASHARE"
    assert scope.universe_id == "CSI500"

    # Cross-sectional with scoped data_source is allowed
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="complete", data_source=_scoped_data_source()
    )


def test_research_mode_preserves_convenience_defaults(monkeypatch):
    """Research mode allows ALL/1d/empty-calendar defaults for convenience."""
    _set_research_mode(monkeypatch)

    class _MinimalFactor:
        name = "research_minimal"
        expr = Expr()
        freq = None
        calendar_id = None
        universe = None
        semantic_identity = None

    scope = _scope_from_factor(_MinimalFactor())
    assert scope.frequency == "1d"
    assert scope.universe_id == "ALL"
    assert scope.calendar_id == ""

    # cross-sectional on ALL is allowed in research
    assert_execution_scope_contract(
        scope, _rank_plan(), factor_name="research_minimal", data_source=None
    )
