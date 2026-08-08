# -*- coding: utf-8 -*-
"""Explicit machine-readable policies for compatibility and pandas-first promotion."""
from __future__ import annotations

_APPLIED = False


def apply_lqtp_policy_patch() -> None:
    """Attach reviewed PIT/scope policy before layer governance enriches metadata."""
    global _APPLIED
    if _APPLIED:
        return
    from cleaned_operators import operator_policy
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS

    # The only new numerical primitive is the stateful three-argument SMA.
    operator_policy._EXPLICIT_POLICIES["ts_sma_cn"] = {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
    }

    # Promotion is reviewed independently from backend portability. Extended /
    # research operators no longer become temporally unsafe merely because
    # DuckDB or Polars support is absent.
    special_scopes = {
        "cs_resid": "cs",
        "round": "elementwise",
        "scale": "cs",
        "sigmoid": "elementwise",
        "true_range": "ts",
    }
    reviewed_scopes = {"ts", "cs", "group", "fundamental_period", "session_intraday"}
    for canonical in sorted(PANDAS_FIRST_PRODUCTION_CANONICALS):
        current = dict(operator_policy._EXPLICIT_POLICIES.get(canonical) or {})
        fallback = special_scopes.get(
            canonical,
            "ts" if canonical.startswith("ts_") else "elementwise",
        )
        existing_scope = current.get("scope")
        # 2026-08 final pack: canonicals with an explicitly reviewed non-fallback
        # scope (relation_* -> cs, group_* -> group, intra_* -> session_intraday)
        # keep it; the prefix fallback only applies when no deliberate scope was
        # declared.
        scope = existing_scope if existing_scope in reviewed_scopes else fallback
        current["scope"] = scope
        # Review P0-A03: an extended/research surface NEVER grants PIT safety by
        # blanket.  ``pit_safe`` is a lifecycle result written only by
        # ``reconcile_operator_certification`` (evidence-driven) after the
        # evidence overlay binds the per-operator artifact.  Sealing it True here
        # would give extended-only operators pit_safe without any certificate.
        if "pit_safe" not in current:
            current["pit_safe"] = False
        if scope == "ts":
            current.setdefault("min_periods", 1)
        operator_policy._EXPLICIT_POLICIES[canonical] = current

    _APPLIED = True
