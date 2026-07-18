#!/usr/bin/env python3
"""Detect whether factor_engine DSL can be sent to LQTP RunFactor as-is."""
from __future__ import annotations

import re

# factor_engine operator names that differ from LQTP DSL naming.
# When any of these appear, materialize locally and upload values instead.
FE_ONLY_OPERATOR_PATTERNS: tuple[str, ...] = (
    r"\bprotected_div\s*\(",
    r"\bts_median\s*\(",
    r"\badd\s*\(",
    r"\bnot_\s*\(",
    r"\btanh\s*\(",
    r"\bADX\s*\(",
    r"\bATR\s*\(",
    r"\bRSI\s*\(",
    r"\bROC\s*\(",
    r"\bSMA\s*\(",
)


def fe_only_operators(dsl: str) -> list[str]:
    """Return FE-only operator names found in *dsl*."""
    hits: list[str] = []
    for pattern in FE_ONLY_OPERATOR_PATTERNS:
        if re.search(pattern, dsl):
            name = pattern.replace(r"\b", "").replace(r"\s*\(", "")
            hits.append(name)
    return hits


def is_lqtp_native_dsl(dsl: str) -> bool:
    """True when *dsl* uses LQTP-compatible operator naming and can be passed directly."""
    text = (dsl or "").strip()
    if not text:
        return False
    return not fe_only_operators(text)


def eval_route_for_entry(*, status: str, dsl: str) -> str:
    """Catalog eval route: lqtp_dsl | local_dsl | local_python."""
    if status in {"python", "hard"} or not (dsl or "").strip():
        return "local_python"
    if status == "ready" and is_lqtp_native_dsl(dsl):
        return "lqtp_dsl"
    return "local_dsl"
