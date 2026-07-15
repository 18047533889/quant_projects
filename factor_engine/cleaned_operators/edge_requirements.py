# -*- coding: utf-8 -*-
"""Operator-specific IEEE edge evidence requirements.

This module does not manufacture certification. It reports which production
operators still lack required NaN/Inf evidence so routing can remain fail-closed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

NAN_REQUIRED = frozenset({
    "c_mean", "c_std", "c_sum", "normalize", "zscore", "scale", "winsorize",
    "group_mean", "group_std", "group_zscore", "group_normalize", "group_rank",
    "ts_mean", "ts_std", "ts_var", "ts_corr", "ts_cov", "ts_beta", "ts_zscore",
    "ts_sharpe", "volatility", "maximum", "minimum", "where", "coalesce",
})
INF_REQUIRED = NAN_REQUIRED


def _factor_engine_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_primitive_evidence() -> dict:
    paths = sorted(_factor_engine_root().rglob("primitive_verified.json"))
    if not paths:
        return {}
    return json.loads(paths[0].read_text(encoding="utf-8"))


def required_edge_dimensions(canonical: str) -> frozenset[str]:
    required: set[str] = set()
    if canonical in NAN_REQUIRED:
        required.add("nan")
    if canonical in INF_REQUIRED:
        required.add("inf")
    return frozenset(required)


def missing_edge_dimensions(canonical: str, evidence: dict | None = None) -> frozenset[str]:
    payload = evidence if evidence is not None else load_primitive_evidence()
    missing: set[str] = set()
    required = required_edge_dimensions(canonical)
    if "nan" in required and canonical not in set(payload.get("duckdb_nan_edge_verified", [])):
        missing.add("nan")
    if "inf" in required and canonical not in set(payload.get("duckdb_inf_edge_verified", [])):
        missing.add("inf")
    return frozenset(missing)


def production_edge_evidence_complete(canonical: str, evidence: dict | None = None) -> bool:
    return not missing_edge_dimensions(canonical, evidence)
