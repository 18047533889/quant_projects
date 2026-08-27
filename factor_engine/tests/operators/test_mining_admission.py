# -*- coding: utf-8 -*-
"""Tests for the mining discovery layer + admission matrix + promote_or_delete.

The discovery layer is the single authority for "what can AlphaProbe /
AlphaMiner actually call".  These tests pin the classification invariants, not
brittle counts (the certification state moves when evidence is regenerated).
"""
from __future__ import annotations

from types import MappingProxyType

import json
from pathlib import Path

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.operator_catalog import (
    MiningRole,
    RoleSource,
    assign_mining_role,
    assign_mining_role_ex,
    get_mining_operators,
    mining_eligible,
    mining_role_manifest,
)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


_ROLE_EXPECTATIONS = {
    "ts_mean": MiningRole.ALPHA,
    "rank": MiningRole.ALPHA,
    "KAMA": MiningRole.ALPHA_HIGH_COST,
    "Supertrend": MiningRole.ALPHA_HIGH_COST,
    "state_latch": MiningRole.STATE,
    "event_refractory": MiningRole.EVENT,
    "ts_multi_regression_resid": MiningRole.DIAGNOSTIC,
    "ts_ar_forecast": MiningRole.DIAGNOSTIC,
    "ts_ar_prior_forecast": MiningRole.ALPHA,  # causal replacement stays mineable
    "arg": MiningRole.DENIED,
    "tan": MiningRole.DENIED,
    "holder_concentration_change": MiningRole.SOURCE_TRANSFORM,
    "micro_bvc_vpin": MiningRole.RESEARCH,
    "cube": MiningRole.LEGACY,
    "identity": MiningRole.INTERNAL,
    "fin_total_operating_accruals": MiningRole.FUNDAMENTAL_PIT,
    # R63 orthogonality: raw trig -> INTERNAL (MOVE_INTERNAL verdict), in-sample
    # AR/poly2 diagnostics -> RESEARCH (RESEARCH_TOOL verdict), recipe/fundamental
    # period transforms -> RECIPE_INTERNAL (DIRECT_RECIPE verdict, non-terminal).
    "sin": MiningRole.INTERNAL,
    "cos": MiningRole.INTERNAL,
    "asin": MiningRole.INTERNAL,
    "acos": MiningRole.INTERNAL,
    "ts_ar_fitted_value": MiningRole.RESEARCH,
    "ts_ar_in_sample_resid": MiningRole.RESEARCH,
    "ts_poly2_coeff": MiningRole.RESEARCH,
    "ts_poly2_resid": MiningRole.RESEARCH,
    "period_average": MiningRole.RECIPE_INTERNAL,
    "quarter_from_cumulative": MiningRole.RECIPE_INTERNAL,
    "ttm_from_cumulative": MiningRole.RECIPE_INTERNAL,
}


def test_role_assignment_is_stable() -> None:
    roles = mining_role_manifest()
    assert len(roles) == len(OperatorRegistry._catalog)
    for canonical, expected in _ROLE_EXPECTATIONS.items():
        if canonical not in OperatorRegistry._catalog:
            continue  # already unregistered (future/random kernels)
        assert roles[canonical] == expected, canonical


def test_every_registered_canonical_has_a_role() -> None:
    roles = mining_role_manifest()
    unknown = [c for c, r in roles.items() if r is None]
    assert unknown == []


def test_denied_never_eligible() -> None:
    for canonical in ("arg", "tan", "cot", "sec", "csc", "cosh", "sinh"):
        catalog = OperatorRegistry._catalog.get(canonical) or {}
        role = assign_mining_role(canonical, catalog)
        assert role is MiningRole.DENIED
        assert not mining_eligible(canonical, catalog=catalog, role=role)


def test_eligible_implies_certified_invariant() -> None:
    # MINING_ELIGIBLE_WITHOUT_CERTIFICATION == ∅ by construction.
    for canonical in OperatorRegistry._catalog:
        catalog = OperatorRegistry._catalog[canonical]
        role = assign_mining_role(canonical, catalog)
        eligible = mining_eligible(canonical, catalog=catalog, role=role)
        if eligible:
            assert catalog.get("production_certified") is True, canonical
            assert role in (
                MiningRole.ALPHA,
                MiningRole.ALPHA_HIGH_COST,
                MiningRole.STATE,
                MiningRole.CONDITION,
                MiningRole.EVENT,
                MiningRole.GROUP_STATE,
                MiningRole.GLOBAL_STATE,
                MiningRole.INTRADAY_EOD,
                MiningRole.FUNDAMENTAL_PIT,
            )


def test_get_mining_operators_filters() -> None:
    alpha = get_mining_operators(roles=["alpha"], admission="all")
    assert alpha
    assert all(op.role is MiningRole.ALPHA for op in alpha)
    states = get_mining_operators(roles=["state", "event"], admission="all")
    assert states
    assert all(op.role in (MiningRole.STATE, MiningRole.EVENT) for op in states)
    # max_cost filter is honored (cost_tier is monotone in the filter).
    cheap = get_mining_operators(admission="all", max_cost=1)
    assert cheap
    assert all(op.cost_tier <= 1 for op in cheap)


def test_terminal_allowlist_exact() -> None:
    """R50: DirectUse is now the SINGLE terminal / AST-position authority.

    ``get_mining_operators`` derives ``terminal_allowed`` and
    ``allowed_ast_positions`` from the DirectUse contract
    (``build_direct_use_operator``), not from the role.  A DIRECT_INTERMEDIATE /
    DIRECT_RECIPE / DIRECT_CONTROL_FLOW canonical has its own semantic role and is
    NOT a terminal, so ``role in _TERMINAL_ROLES`` is no longer the contract.  The
    invariant that every row satisfies ``terminal_allowed == ("terminal" in
    allowed_ast_positions)`` holds because both come from the same DirectUse
    row.  A canonical whose legacy role is supporting but whose DirectUse verdict
    is DIRECT_* (e.g. fin_expectation_revision* -> DIRECT_ALPHA terminal while the
    role machinery says DENIED) is resolved by DirectUse — the single authority.
    """
    from factor_engine.mining.direct_use import build_direct_use_operator

    du_by = {
        r.canonical: r
        for r in (
            build_direct_use_operator(c, OperatorRegistry._catalog.get(c) or {})
            for c in sorted(OperatorRegistry._catalog)
        )
    }

    for op in get_mining_operators(admission="all"):
        # NEW single-authority invariant: both fields come from the same
        # DirectUse row, so they are always self-consistent.
        assert op.terminal_allowed == ("terminal" in op.allowed_ast_positions), (
            f"{op.canonical}: terminal_allowed={op.terminal_allowed} but positions="
            f"{op.allowed_ast_positions}"
        )
        # terminal_usable is the semantic gate behind terminal_allowed.
        assert op.terminal_allowed == op.terminal_usable, op.canonical
        # the four orthogonal fields are populated.
        assert isinstance(op.mining_visible, bool), op.canonical
        assert isinstance(op.production_terminal_usable, bool), op.canonical
        assert isinstance(op.production_admitted, bool), op.canonical

        # get_mining_operators must match the DirectUse authority on every row.
        row = du_by.get(op.canonical)
        assert row is not None, op.canonical
        assert op.allowed_ast_positions == row.allowed_ast_positions, op.canonical
        assert op.terminal_allowed == row.terminal_usable, op.canonical
        assert op.terminal_usable == row.terminal_usable, op.canonical
        assert op.production_terminal_usable == row.production_terminal_usable, op.canonical
        assert op.production_admitted == row.production_admitted, op.canonical
        assert op.mining_visible == row.mining_visible, op.canonical

        # Supporting / non-direct DirectUse verdicts have EMPTY positions
        # (research_tool / move_internal / delete_*).  A canonical with a
        # DIRECT_* verdict gets its positions from the contract, even if the
        # legacy role classifies it as a supporting role.
        if op.role in (
            MiningRole.RECIPE_INTERNAL,
            MiningRole.SOURCE_TRANSFORM,
            MiningRole.INTERNAL,
            MiningRole.DIAGNOSTIC,
            MiningRole.RESEARCH,
            MiningRole.LEGACY,
            MiningRole.DENIED,
            MiningRole.UNRESOLVED,
        ):
            if not row.direct_use_status.value.startswith("direct_"):
                assert op.allowed_ast_positions == (), op.canonical

    # Spot-check the DirectUse authority directly on a few representatives.
    # DIRECT_INTERMEDIATE / DIRECT_RECIPE -> role RECIPE_INTERNAL, NOT terminal.
    for c in ("ATR_WILDER", "cos", "sin", "ttm_from_cumulative", "period_average"):
        if c in du_by:
            assert "terminal" not in du_by[c].allowed_ast_positions, c
            assert du_by[c].terminal_usable is False, c
    # DIRECT_ALPHA -> terminal-eligible with "terminal" in positions.
    for c in ("ts_mean", "rank"):
        if c in du_by:
            assert "terminal" in du_by[c].allowed_ast_positions, c
            assert du_by[c].terminal_usable is True, c


def test_no_fallback_role_sources() -> None:
    """R15-INC-222: a role produced by the FALLBACK catch-all is banned in the
    final registry.  Every canonical has an explicit or verified-rule role."""
    for canonical in OperatorRegistry._catalog:
        catalog = OperatorRegistry._catalog[canonical]
        _role, source = assign_mining_role_ex(canonical, catalog)
        assert source is not RoleSource.FALLBACK, canonical


def test_continuous_event_statistics_not_event_mask() -> None:
    """R15-INC-015/224: ``event_interval_memory`` / ``event_fano_factor`` /
    ``event_local_variation`` are continuous statistics, not EventBool masks —
    they must be mineable numeric roles, not EVENT."""
    for canonical in (
        "event_interval_memory",
        "event_local_variation",
        "event_fano_factor",
        "event_fano_excess",
        "event_cumulative_return_past",
    ):
        if canonical not in OperatorRegistry._catalog:
            continue
        role = assign_mining_role(canonical, OperatorRegistry._catalog[canonical])
        assert role is not MiningRole.EVENT, canonical
        assert role in (
            MiningRole.ALPHA,
            MiningRole.ALPHA_HIGH_COST,
            MiningRole.STATE,
        ), canonical


def test_continuous_state_statistics_not_state_slot() -> None:
    """R15-INC-014: ``state_episode_mfe`` / ``threshold_cycle_period`` /
    ``candle_gap_atr`` are continuous measures, not discrete Bool state slots —
    they must stay terminal-eligible, not STATE."""
    for canonical in (
        "state_episode_mfe",
        "state_episode_mae",
        "state_episode_efficiency",
        "ts_threshold_cycle_period",
        "ts_threshold_cycle_asymmetry",
        "candle_gap_atr",
    ):
        if canonical not in OperatorRegistry._catalog:
            continue
        role = assign_mining_role(canonical, OperatorRegistry._catalog[canonical])
        assert role not in (MiningRole.STATE, MiningRole.EVENT), canonical
        assert role in (
            MiningRole.ALPHA,
            MiningRole.ALPHA_HIGH_COST,
            MiningRole.CONDITION,
        ), canonical


def test_market_filter_fail_closed() -> None:
    """R15-INC-002/223: A-share-only operators must not appear in a US context,
    and an unknown market is rejected (not silently ignored)."""
    from factor_engine.mining.operator_catalog import validate_market, market_support

    with pytest.raises(ValueError, match="unknown market"):
        get_mining_operators(market="cn", admission="all")
    us = {op.canonical for op in get_mining_operators(market="us", admission="all")}
    ashare = {op.canonical for op in get_mining_operators(market="ashare", admission="all")}
    # every operator valid in US is a subset of the both-market pool
    assert us <= ashare
    # a pure A-share state-machine op never appears in the US pool
    for canonical in ("ashare_limit_up_streak", "ashare_days_since_limit_up"):
        if canonical not in OperatorRegistry._catalog:
            continue
        assert canonical not in us, canonical
        assert canonical in ashare, canonical
    assert market_support("ashare_limit_up_streak") == ("ashare",)


def test_target_frequency_matches_output_grain() -> None:
    """R15-INC-003/223: INTRADAY_EOD is minute input / DAILY output — it must be
    eligible for target=daily (with minute source), never for target=minute."""
    from factor_engine.mining.operator_catalog import mining_eligible

    intraday = [
        op for op in get_mining_operators(
            roles=["intraday_eod"], admission="all"
        )
        if op.canonical in OperatorRegistry._catalog
        and op.canonical not in {"intraday_wasserstein_pair_distance"}
    ]
    assert intraday
    for op in intraday:
        # R63: minute-grain session_intraday event/flag primitives
        # (intra_neighbor_event_class, intra_range_gap_flag) stay at minute grain;
        # every other intraday_eod row is minute-input → daily-output.
        minute_grain = op.canonical in {"intra_neighbor_event_class", "intra_range_gap_flag"}
        expected = "minute" if minute_grain else "daily"
        assert op.output_grain == expected, op.canonical
    # Deterministic gate test on a certified local copy (the live catalog is
    # evidence-stale → 0 certified, so eligibility cannot be asserted directly).
    candidate = next(
        op.canonical for op in intraday
        if op.canonical not in {"intra_neighbor_event_class", "intra_range_gap_flag"}
    )
    catalog = dict(OperatorRegistry._catalog[candidate])
    catalog["production_certified"] = True
    catalog["tags"] = list(catalog.get("tags") or ()) + ["cost:5"]
    assert mining_eligible(
        candidate, catalog=catalog, role=MiningRole.INTRADAY_EOD,
        available_sources=["minute_bar"], target_frequency="daily",
    )
    assert not mining_eligible(
        candidate, catalog=catalog, role=MiningRole.INTRADAY_EOD,
        available_sources=["minute_bar"], target_frequency="minute",
    )


def test_source_context_unknown_is_fail_closed() -> None:
    """R15-INC-004: an omitted source context is UNKNOWN capability, never
    "everything available".  Explicit empty set and explicit set differ."""
    from factor_engine.mining.operator_catalog import source_status

    catalog = OperatorRegistry._catalog["ts_mean"]
    unk = source_status("ts_mean", catalog, None)
    empty = source_status("ts_mean", catalog, ())
    full = source_status("ts_mean", catalog, ["daily_bar"])
    assert unk.unknown is True
    assert empty.unknown is False and empty.missing == ("daily_bar",)
    assert full.missing == () and full.satisfied == ("daily_bar",)


def test_unresolved_role_never_mines() -> None:
    """R15-INC-001: a registered canonical that matches no verified rule is
    UNRESOLVED and never eligible — it is never silently promoted to ALPHA."""
    from factor_engine.mining.operator_catalog import MiningOperator

    # register a throwaway canonical with a bare surface (no family rule, no
    # daily/extended surface mapping can fire for 'zz_unresolved_probe_*')
    registered = "zz_unresolved_probe_never_exists"
    catalog = OperatorRegistry._catalog
    # the registry may be frozen (MappingProxyType) after load_all(); thaw it for
    # the throwaway registration, then re-freeze on exit.
    thawed = False
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry as _OR, _BOOTSTRAP_TOKEN as _BT

        if isinstance(catalog, MappingProxyType):
            _OR.thaw_for_bootstrap(_BT)
            catalog = _OR._catalog
            thawed = True
    except Exception:
        pass
    old = catalog.get(registered)
    catalog[registered] = {
        "surface": "unclassified",
        "param_names": ["x"],
        "input_grain": "daily",
        "tags": ["cost:1"],
        "production_certified": True,
    }
    try:
        role, source = assign_mining_role_ex(registered, catalog[registered])
        assert role is MiningRole.UNRESOLVED
        assert source is RoleSource.FALLBACK
        assert not mining_eligible(registered, catalog=catalog[registered], role=role)
    finally:
        if old is None:
            catalog.pop(registered, None)
        else:
            catalog[registered] = old
        if thawed:
            _OR.finalize()
            _OR.freeze()


def test_admission_matrix_no_vague_answers() -> None:
    from factor_engine.audit.operator_admission_matrix import generate_admission_matrix

    records = generate_admission_matrix()
    assert len(records) >= 1000
    seen: set[str] = set()
    for rec in records:
        assert rec.canonical not in seen
        seen.add(rec.canonical)
        assert rec.factor_role
        assert rec.target_mining_lane
        if not rec.production_certified:
            assert rec.blocker_codes, f"{rec.canonical} has no blocker"
        assert rec.recommended_action, f"{rec.canonical} has no recommended action"
        assert "research only" not in rec.recommended_action.lower()
        assert "unsupported" not in rec.recommended_action.lower()
        assert "not production" not in rec.recommended_action.lower()


def test_promote_or_delete_audit_invariants() -> None:
    from scripts.audit_all_registered_operators import audit_all_registered_operators

    result = audit_all_registered_operators()
    invariants = result["invariants"]
    assert invariants["UNCLASSIFIED_FACTOR_CANONICALS"] == []
    assert invariants["MINING_ELIGIBLE_WITHOUT_CERTIFICATION"] == []
    # Every bucket A..J disjoint; every canonical appears exactly once.
    buckets = result["buckets"]
    all_members = [c for members in buckets.values() for c in members]
    assert len(all_members) == len(set(all_members))
    assert len(all_members) == result["counts"]["registered public total"]
    counts = result["counts"]
    assert counts["unclassified total"] == 0


def test_mining_manifest_export_and_cold_start(tmp_path: Path) -> None:
    from scripts.export_mining_manifest import export_mining_manifest, validate_cold_start

    paths = export_mining_manifest(tmp_path, admission="all")
    manifest = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert manifest["count"] >= 1000
    assert paths["csv"].exists()
    assert paths["md"].exists()
    # eligible entries carry full role/position metadata
    eligible = [e for e in manifest["operators"] if e["mining_eligible"]]
    for entry in eligible:
        assert entry["mining_role"]
        assert entry["allowed_ast_positions"]
        assert entry["production_certified"] is True
    # cold-start validation: every public DSL name must be in the manifest
    violations = validate_cold_start(tmp_path)
    assert violations == []
