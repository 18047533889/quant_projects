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

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from mining.operator_catalog import (
    MiningRole,
    assign_mining_role,
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


def test_non_terminal_roles_never_terminal() -> None:
    for op in get_mining_operators(admission="all"):
        if op.role in (
            MiningRole.STATE,
            MiningRole.CONDITION,
            MiningRole.EVENT,
            MiningRole.GROUP_STATE,
            MiningRole.GLOBAL_STATE,
        ):
            assert op.terminal_allowed is False
            assert "terminal" not in op.allowed_ast_positions
        else:
            assert op.terminal_allowed is True


def test_admission_matrix_no_vague_answers() -> None:
    from audit.operator_admission_matrix import generate_admission_matrix

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
