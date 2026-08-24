# -*- coding: utf-8 -*-
"""R20-RELALPHA-PROMOTION: pin the _RELATIVE_ALPHA_OPS promotion of the six
dimensionless ex-self / prior-beta / group-relative canonicals.

The canonicals were landed by R20-EXSELF-CS-DIRECTUSE (ex_self_zscore,
ex_self_rank_pct, ex_self_mad_z) and R20-GROUPSTATE-BETA-DIRECTUSE
(beta_residual_z, beta_divergence_pct, relative_strength_group_pct) — each
already resolving DIRECT_ALPHA through the verified MiningRole fallback and
sitting in the alpha_direct lane with mining/composition/terminal usable True.

This slice adds them to ``_RELATIVE_ALPHA_OPS`` (mining/direct_use.py), the
set every sibling R20 family is promoted through, so the verdict is pinned to
the explicit promotion table instead of depending on role-fallback drift.

Contract under test:
* the six canonicals are members of ``_RELATIVE_ALPHA_OPS``;
* each resolves DIRECT_ALPHA via the EXPLICIT promotion path (not the
  role fallback) with alpha_direct lane and mining/composition/terminal True;
* ``ex_self_mean_gap`` (unit-bearing level gap — NOT scale-invariant) is
  deliberately NOT promoted and keeps its fallback path — pinned;
* the audited unit-bearing / duplicate neighbours stay OUT of the set.
"""
from __future__ import annotations

from factor_engine.cleaned_operators import load_all

load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.mining.direct_use import (  # noqa: E402
    _RELATIVE_ALPHA_OPS,
    build_direct_use_operator,
    resolve_direct_use,
)

_PROMOTED = (
    "ex_self_zscore", "ex_self_rank_pct", "ex_self_mad_z",
    "beta_residual_z", "beta_divergence_pct", "relative_strength_group_pct",
)


def test_six_canonicals_are_promoted_members():
    for name in _PROMOTED:
        assert name in _RELATIVE_ALPHA_OPS, name


def test_promotion_resolves_explicit_direct_alpha():
    for name in _PROMOTED:
        contract = resolve_direct_use(name)
        assert contract.status.value == "direct_alpha", name
        # the explicit promotion table produced the contract, not the
        # MiningRole fallback (reason mentions the dimensionless-alpha lane)
        assert "dimensionless alpha" in contract.retention_reason, (
            name, contract.retention_reason,
        )
        row = build_direct_use_operator(name, OperatorRegistry._catalog.get(name) or {})
        assert row.mining_lane == "alpha_direct", name
        assert row.mining_visible and row.composition_usable and row.terminal_usable, name


def test_ex_self_mean_gap_deliberately_not_promoted():
    """ex_self_mean_gap carries x's unit (a level gap) — NOT scale-invariant;
    it stays off the promotion set and keeps the role-fallback path (which
    still classifies it direct_alpha as a verified-role alpha, but never via
    the dimensionless-promotion reason)."""
    assert "ex_self_mean_gap" not in _RELATIVE_ALPHA_OPS
    contract = resolve_direct_use("ex_self_mean_gap")
    assert "dimensionless alpha" not in (contract.retention_reason or "")


def test_audited_neighbours_stay_out():
    # unit-bearing / level / duplicate neighbours audited by the two landing
    # slices must never enter the dimensionless promotion set
    for name in ("peer_residual_z", "within_group_rank_pct",
                 "rolling_beta_to_market", "tail_beta", "group_zscore",
                 "benchmark_excess_return", "group_ex_self_mean"):
        assert name not in _RELATIVE_ALPHA_OPS, name
