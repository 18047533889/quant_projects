# -*- coding: utf-8 -*-
"""Tests for the mining discovery layer + admission matrix + promote_or_delete.

The discovery layer is the single authority for "what can AlphaProbe /
AlphaMiner actually call".  These tests pin the classification invariants, not
brittle counts (the certification state moves when evidence is regenerated).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.operator_catalog import (
    MiningRole,
    RoleSource,
    _TERMINAL_ROLES,
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
    """R15-INC-221: only the factor-value roles (ALPHA / ALPHA_HIGH_COST /
    INTRADAY_EOD / FUNDAMENTAL_PIT) may be terminal.  INTERNAL / DIAGNOSTIC /
    RESEARCH / LEGACY / DENIED must be False — the pre-R15 exclusion list let
    them fall through as terminal.  Supporting roles also have EMPTY positions."""
    for op in get_mining_operators(admission="all"):
        expected = op.role in _TERMINAL_ROLES
        assert op.terminal_allowed is expected, (
            f"{op.canonical}: terminal_allowed={op.terminal_allowed} "
            f"role={op.role} but expected {expected}"
        )
        if not expected:
            assert "terminal" not in op.allowed_ast_positions, op.canonical
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
            assert op.allowed_ast_positions == (), op.canonical


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
        assert op.output_grain == "daily", op.canonical
    # Deterministic gate test on a certified local copy (the live catalog is
    # evidence-stale → 0 certified, so eligibility cannot be asserted directly).
    candidate = intraday[0].canonical
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
