# -*- coding: utf-8 -*-
"""R13 execution-scope fixes (2026-08-09): P1-01 .. P1-04.

Covers:

* (a) ``_is_whole_market_universe`` no longer treats named pools
  (``US_NASDAQ100`` / ``ASHARE_CSI300``) as whole market — P1-02 removed the
  ``infer_market`` string-inference branch; only ``ALL`` / universe==market /
  ``*_ALL`` are whole market.
* (b) ``assert_execution_scope_contract`` rejects a data source whose explicitly
  declared universe mismatches the factor's scoped universe (P1-03), while
  keeping the ``instrument_filter`` pass-through.
* (c) ``_scope_from_factor`` no longer silently defaults ``market`` to ``"A"``
  and prefers ``Factor.semantic_identity`` (P1-01).
* (d) ``compute_source_scope_hash`` yields a 16-hex SHA-256 that differs across
  data-source configs (P1-04).

These gate functions are tested directly with hand-built ``PlanNode`` /
``FactorExecutionScope`` / stub sources — no operator-registry load required.
"""
from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError

from factor_engine.api.factor import Factor, FactorSemanticIdentity
from factor_engine.expr.base import Expr
from factor_engine.planner.dag import FactorExecutionScope
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import (
    _is_whole_market_universe,
    _scope_from_factor,
    assert_execution_scope_contract,
    compute_source_scope_hash,
)
from factor_engine.runtime.production_policy import ProductionPolicyViolation


def _column(name: str = "close") -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _rank_plan() -> PlanNode:
    return PlanNode(op="rank", inputs=(_column(),))


# --- (a) whole-market universe (P1-02) -------------------------------------


@pytest.mark.parametrize(
    "univ",
    ["US_NASDAQ100", "ASHARE_CSI300", "CSI300", "ASHARE_DAILY", "US_MASSIVE"],
    ids=["US_NASDAQ100", "ASHARE_CSI300", "CSI300", "ASHARE_DAILY", "US_MASSIVE"],
)
def test_whole_market_universe_scoped_pools_not_whole(univ):
    """Named pools must NOT be treated as whole market (fail-closed)."""
    scope = FactorExecutionScope(universe_id=univ, market="A")
    assert _is_whole_market_universe(scope) is False


@pytest.mark.parametrize(
    ("univ", "market"),
    [
        ("ALL", "A"),
        ("", "A"),
        ("ASHARE_ALL", "A"),
        ("US", "US"),
        ("A", "A"),
    ],
    ids=["ALL", "empty", "ASHARE_ALL", "US==market", "A==market"],
)
def test_whole_market_universe_all_labels_whole(univ, market):
    scope = FactorExecutionScope(universe_id=univ, market=market)
    assert _is_whole_market_universe(scope) is True


def test_whole_market_universe_scoped_pool_cross_sectional_fails_closed():
    """A named pool + cross-sectional op on an unfiltered source must reject."""
    with pytest.raises(ProductionPolicyViolation):
        assert_execution_scope_contract(
            FactorExecutionScope(universe_id="US_NASDAQ100", market="US"),
            _rank_plan(),
            factor_name="us_nasdaq100_rank",
        )


# --- (b) data-source universe mismatch (P1-03) -----------------------------


def test_data_source_scoped_universe_mismatch_rejected():
    class _Source:
        universe = "CSI500"

    with pytest.raises(ProductionPolicyViolation) as exc:
        assert_execution_scope_contract(
            FactorExecutionScope(universe_id="CSI300", market="A"),
            _rank_plan(),
            factor_name="cs300_on_cs500",
            data_source=_Source(),
        )
    assert "universe mismatch" in str(exc.value)


def test_data_source_scoped_direct():
    from factor_engine.runtime.engine import _data_source_scoped

    class _NoScope:
        pass

    class _Filtered:
        instrument_filter = ["600000.SH"]

    class _Universe:
        universe = "CSI300"

    assert _data_source_scoped(None) is False
    assert _data_source_scoped(_NoScope()) is False
    assert _data_source_scoped(_Filtered()) is True
    assert _data_source_scoped(_Universe()) is True


def test_data_source_scoped_no_universe_instrument_filter_allowed():
    """Data layer already restricts instruments -> source is authoritative."""

    class _Source:
        instrument_filter = ["600000.SH", "600036.SH"]

    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="CSI300", market="A"),
        _rank_plan(),
        factor_name="cs300_filtered",
        data_source=_Source(),
    )


def test_data_source_scoped_matching_universe_allowed():
    class _Source:
        universe = "CSI300"

    assert_execution_scope_contract(
        FactorExecutionScope(universe_id="CSI300", market="A"),
        _rank_plan(),
        factor_name="cs300_on_cs300",
        data_source=_Source(),
    )


def test_data_source_scoped_universe_mismatch_inner_rejected():
    """Wrapper sources recurse into inner; explicit conflicting universe rejects."""

    class _Inner:
        universe = "CSI500"

    class _Wrapper:
        inner = _Inner()

    with pytest.raises(ProductionPolicyViolation):
        assert_execution_scope_contract(
            FactorExecutionScope(universe_id="CSI300", market="A"),
            _rank_plan(),
            factor_name="cs300_wrapped",
            data_source=_Wrapper(),
        )


# --- (c) _scope_from_factor semantic identity / no default "A" (P1-01) ------


def test_scope_from_factor_us_factor_no_market_not_default_a():
    factor = Factor(name="us_factor", expr=Expr(), universe="US_NASDAQ100")
    scope = _scope_from_factor(factor)
    assert scope.market != "A"
    assert scope.market == ""


def test_scope_from_factor_uses_semantic_identity():
    factor = Factor(
        name="us_semantic",
        expr=Expr(),
        semantic_identity=FactorSemanticIdentity(
            market="US", universe_id="US_NASDAQ100", frequency="1d"
        ),
    )
    scope = _scope_from_factor(factor)
    assert scope.market == "US"
    assert scope.universe_id == "US_NASDAQ100"
    assert scope.frequency == "1d"


def test_scope_from_factor_semantic_identity_source_hash_wins():
    factor = Factor(
        name="hashed",
        expr=Expr(),
        semantic_identity=FactorSemanticIdentity(
            market="US",
            universe_id="US_NASDAQ100",
            source_scope_hash="abcdef1234567890",
        ),
    )
    scope = _scope_from_factor(factor)
    assert scope.source_scope_hash == "abcdef1234567890"


def test_scope_from_factor_backward_compat_without_semantic_identity():
    factor = Factor(name="plain", expr=Expr(), universe="CSI300")
    scope = _scope_from_factor(factor)
    assert scope.universe_id == "CSI300"
    assert scope.frequency == "1d"
    assert scope.market == ""


# --- (d) compute_source_scope_hash (P1-04) ---------------------------------


def test_compute_source_scope_hash_differs_by_config():
    h1 = compute_source_scope_hash(data_source_config={"dataset": "ashare_daily"})
    h2 = compute_source_scope_hash(data_source_config={"dataset": "us_daily"})
    assert h1 != h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_source_scope_hash_stable_across_dict_order():
    cfg = {"dataset": "ashare_daily", "fields": {"close": "close_px"}}
    h1 = compute_source_scope_hash(data_source_config=cfg, market="A")
    h2 = compute_source_scope_hash(data_source_config=dict(cfg), market="A")
    assert h1 == h2


def test_compute_source_scope_hash_sensitive_to_semantic_fields():
    cfg = {"dataset": "ashare_daily"}
    base = compute_source_scope_hash(data_source_config=cfg, market="A")
    other_market = compute_source_scope_hash(data_source_config=cfg, market="US")
    other_pit = compute_source_scope_hash(
        data_source_config=cfg, market="A", pit_mode="enforce"
    )
    assert base != other_market
    assert base != other_pit


def test_compute_source_scope_hash_rejects_unknown_values():
    class Unknown:
        def __str__(self):
            return "same-looking-value"

    with pytest.raises(ValidationError, match="Cannot encode"):
        compute_source_scope_hash(data_source_config={"opaque": Unknown()})


def test_compute_source_scope_hash_preserves_typed_collision_distinction():
    int_hash = compute_source_scope_hash(data_source_config={"value": 1})
    string_hash = compute_source_scope_hash(data_source_config={"value": "1"})
    assert int_hash != string_hash


def test_compute_source_scope_hash_accepts_typed_source_scope_id():
    from factor_engine.planner.physical_factor_dag import SourceScopeId

    scope = SourceScopeId(dataset="prices", snapshot_id="s1", market="A")
    digest = compute_source_scope_hash(data_source_config={"scope": scope})
    assert len(digest) == 64
def test_scope_from_factor_with_data_source_computes_source_hash():
    class _DS:
        dataset = "ashare_daily"
        instrument_filter = ["600000.SH"]

    factor = Factor(name="src", expr=Expr(), universe="CSI300")
    scope = _scope_from_factor(factor, data_source=_DS())
    assert scope.source_scope_hash != ""
    scope2 = _scope_from_factor(factor, data_source=_DS())
    assert scope.source_scope_hash == scope2.source_scope_hash


def test_physical_region_plan_hash_is_full_sha256():
    from factor_engine.planner.backend_region import PhysicalRegionPlan

    digest = PhysicalRegionPlan.compute_plan_hash((), (), "logical")
    assert len(digest) == 64
    assert digest == PhysicalRegionPlan.compute_plan_hash((), (), "logical")


def test_scope_from_factor_different_data_sources_differ():
    class _DS_A:
        dataset = "ashare_daily"

    class _DS_U:
        dataset = "us_daily"

    factor = Factor(name="src", expr=Expr(), universe="CSI300")
    h_a = _scope_from_factor(factor, data_source=_DS_A()).source_scope_hash
    h_u = _scope_from_factor(factor, data_source=_DS_U()).source_scope_hash
    assert h_a != h_u
