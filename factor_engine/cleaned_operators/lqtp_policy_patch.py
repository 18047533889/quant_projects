# -*- coding: utf-8 -*-
"""Explicit machine-readable policies for LQTP compatibility primitives."""
from __future__ import annotations

_APPLIED = False


def apply_lqtp_policy_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from cleaned_operators import operator_policy

    operator_policy._EXPLICIT_POLICIES["safe_log_null"] = {
        "scope": "elementwise",
        "pit_safe": True,
    }
    operator_policy._EXPLICIT_POLICIES["ts_sma_cn"] = {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
    }
    _APPLIED = True
