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

    # Compatibility primitives introduced by this change.
    operator_policy._EXPLICIT_POLICIES["safe_log_null"] = {
        "scope": "elementwise",
        "pit_safe": True,
    }
    operator_policy._EXPLICIT_POLICIES["ts_sma_cn"] = {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
    }

    # The promotion list is reviewed independently from backend portability.
    # Some extended/research canonicals were previously fail-closed to
    # pit_safe=False only because they were outside the old three-backend daily
    # surface.  Make temporal semantics explicit here rather than inferring
    # safety from backend coverage.
    special_scopes = {
        "cs_resid": "cs",
        "round": "elementwise",
        "scale": "cs",
        "sigmoid": "elementwise",
        "safe_log_null": "elementwise",
        "true_range": "ts",
    }
    for canonical in sorted(PANDAS_FIRST_PRODUCTION_CANONICALS):
        scope = special_scopes.get(
            canonical,
            "ts" if canonical.startswith("ts_") else "elementwise",
        )
        current = dict(operator_policy._EXPLICIT_POLICIES.get(canonical) or {})
        current["scope"] = scope
        current["pit_safe"] = True
        if scope == "ts":
            current.setdefault("min_periods", 1)
        operator_policy._EXPLICIT_POLICIES[canonical] = current

    _APPLIED = True
