# -*- coding: utf-8 -*-
"""Operator-specific IEEE edge evidence requirements.

This module does not manufacture certification. It reports which production
operators still lack required NaN/Inf evidence so routing can remain fail-closed.

review #5 R5-15: the previous gate was a *vacuous pass* — an operator with no
declared required edge dimensions returned ``production_edge_evidence_complete
== True`` ("nothing required, therefore passed") without anyone deciding it is
edge-insensitive.  The gate now exposes an explicit tri-state
(:func:`edge_evidence_status`) — ``complete`` / ``incomplete`` / ``undeclared``
— and a strict mode (:data:`edge_gate_strict`) under which an undeclared
operator does NOT pass.  ``NAN_REQUIRED`` and ``INF_REQUIRED`` are now genuinely
independent sets instead of the same frozenset.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Operators whose output must be validated against NaN inputs (every numeric
# reducer / windowed statistic / conditional that mixes NaN and finite values).
NAN_REQUIRED = frozenset({
    "cs_mean", "cs_std", "cs_sum", "normalize", "zscore", "scale", "winsorize",
    "group_mean", "group_std", "group_zscore", "group_normalize", "group_rank",
    "ts_mean", "ts_std", "ts_var", "ts_corr", "ts_cov", "ts_beta", "ts_zscore",
    "ts_sharpe", "volatility", "maximum", "minimum", "where", "coalesce",
    # R5-15: arithmetic / ratio / log ops are NaN-sensitive too.
    "add", "subtract", "multiply", "divide", "protected_div", "safe_div",
    "log", "log_returns", "ts_pct", "returns", "ratio",
})

# Operators whose output can be corrupted by Inf (Inf propagates into a finite
#-looking average / slope / ratio and must be fail-closed, not silently mixed).
INF_REQUIRED = frozenset({
    "divide", "protected_div", "safe_div", "ratio", "ts_pct", "returns",
    "log", "log_returns", "ts_mean", "ts_std", "ts_beta", "ts_corr",
    "volatility", "zscore", "normalize", "winsorize",
})

# Operators explicitly declared edge-insensitive: they only move/reshape values
# or compare booleans and cannot turn NaN/Inf into a wrong finite output, so no
# edge evidence is required.  Anything NOT here (and not in the required sets)
# is *undeclared* — a fail-closed state under strict mode, never a pass.
EDGE_IMMUNE = frozenset({
    "reindex", "shift_forward", "alias", "identity", "first_not_null",
    "last_not_null", "cum_sum", "cum_prod", "cum_max", "cum_min",
})

# R5-15: strict gate — when True, an operator with undeclared edge requirements
# does NOT count as edge-verified (no vacuous pass).  Production certification
# should run with this enabled and resolve every undeclared operator into either
# a required set (with evidence) or EDGE_IMMUNE.  Default is False to keep the
# existing catalog surface operational until the evidence is regenerated.
edge_gate_strict: bool = False


def _factor_engine_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(canonical: str) -> str:
    try:
        from cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(canonical)
    except Exception:
        return canonical


@lru_cache(maxsize=1)
def load_primitive_evidence() -> dict:
    paths = sorted(_factor_engine_root().rglob("primitive_verified.json"))
    if not paths:
        return {}
    return json.loads(paths[0].read_text(encoding="utf-8"))


def edge_requirements_declared(canonical: str) -> bool:
    name = _resolve(canonical)
    return bool(
        required_edge_dimensions(name)
        or name in EDGE_IMMUNE
    )


def required_edge_dimensions(canonical: str) -> frozenset[str]:
    name = _resolve(canonical)
    required: set[str] = set()
    if name in NAN_REQUIRED:
        required.add("nan")
    if name in INF_REQUIRED:
        required.add("inf")
    return frozenset(required)


def missing_edge_dimensions(canonical: str, evidence: dict | None = None) -> frozenset[str]:
    payload = evidence if evidence is not None else load_primitive_evidence()
    name = _resolve(canonical)
    missing: set[str] = set()
    required = required_edge_dimensions(name)
    if "nan" in required and name not in set(payload.get("duckdb_nan_edge_verified", [])):
        missing.add("nan")
    if "inf" in required and name not in set(payload.get("duckdb_inf_edge_verified", [])):
        missing.add("inf")
    return frozenset(missing)


def edge_evidence_status(
    canonical: str, evidence: dict | None = None
) -> str:
    """R5-15: honest tri-state edge gate.

    * ``complete`` — all required NaN/Inf dimensions have verified evidence, or
      the operator is explicitly ``EDGE_IMMUNE``;
    * ``incomplete`` — at least one required dimension lacks evidence;
    * ``undeclared`` — the operator has no declared edge requirement (it is
      neither in a required set nor ``EDGE_IMMUNE``); under strict mode this is
      a fail-closed state, never a pass.
    """
    name = _resolve(canonical)
    if name in EDGE_IMMUNE:
        return "complete"
    missing = missing_edge_dimensions(name, evidence)
    if missing:
        return "incomplete"
    required = required_edge_dimensions(name)
    if required:
        return "complete"
    return "undeclared"


def production_edge_evidence_complete(canonical: str, evidence: dict | None = None) -> bool:
    status = edge_evidence_status(canonical, evidence)
    if status == "complete":
        return True
    if status == "incomplete":
        return False
    # undeclared: strict mode fails closed; legacy mode preserves the previous
    # vacuous pass for operators already on the surface (see module docstring).
    if edge_gate_strict:
        return False
    return True
