# -*- coding: utf-8
"""Production fast path evidence backed by the certified artifact.

The certification script is the single authority for executed parity.  Do not
reconstruct evidence by importing pytest parameter tables here: those tables
contain aliases and are intentionally reorganised over time, while the
certified artifact contains final canonicals and provenance hashes.
"""
from __future__ import annotations

from backend.evidence_provenance import evidence_artifact_valid, load_verified_artifact

_META_OPS: frozenset[str] = frozenset(
    {"column", "literal", "materialized_series", "plan_ref", "if_else"}
)

def _certified_set(field: str) -> frozenset[str]:
    """Return a fail-closed canonical set from a valid certified artifact."""
    if not evidence_artifact_valid():
        return frozenset()
    values = load_verified_artifact().get(field) or []
    return frozenset(str(value) for value in values)


def polars_executed_parity_canonicals() -> frozenset[str]:
    """Canonicals with certified Polars reference and edge parity."""
    return _certified_set("polars_reference_parity") & _certified_set("polars_edge_verified")


def duckdb_executed_parity_canonicals() -> frozenset[str]:
    """Canonicals with certified DuckDB parity, edge coverage and real SQL."""
    return (
        _certified_set("duckdb_reference_parity")
        & _certified_set("duckdb_edge_verified")
        & _certified_set("duckdb_real_sql_verified")
    )


def missing_polars_parity_evidence() -> list[str]:
    """列出 PolarsLong production-safe 但缺少 parity case 的 canonical。

    返回:
        按字母排序的缺失 canonical 名称列表。
    """
    from backend.polars_long_production import POLARS_LONG_NATIVE_PRODUCTION_SAFE

    required = POLARS_LONG_NATIVE_PRODUCTION_SAFE - _META_OPS
    have = polars_executed_parity_canonicals()
    return sorted(c for c in required if c not in have)


def missing_duckdb_parity_evidence() -> list[str]:
    """列出 DuckDB production-safe 但缺少 parity case 的 canonical。

    返回:
        按字母排序的缺失 canonical 名称列表。
    """
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    required = SQL_PRODUCTION_SAFE_CANONICALS - _META_OPS
    have = duckdb_executed_parity_canonicals()
    return sorted(c for c in required if c not in have)


def evidence_summary() -> dict[str, int | list[str]]:
    """汇总 Polars/DuckDB parity 证据链覆盖情况。

    返回:
        含 required、evidence 计数及 missing 列表的摘要字典。
    """
    from backend.polars_long_production import POLARS_LONG_NATIVE_PRODUCTION_SAFE
    from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    pol_req = POLARS_LONG_NATIVE_PRODUCTION_SAFE - _META_OPS
    duck_req = SQL_PRODUCTION_SAFE_CANONICALS - _META_OPS
    pol_have = polars_executed_parity_canonicals()
    duck_have = duckdb_executed_parity_canonicals()
    return {
        "polars_required": len(pol_req),
        "polars_evidence": len(pol_have & pol_req),
        "polars_missing": missing_polars_parity_evidence(),
        "duckdb_required": len(duck_req),
        "duckdb_evidence": len(duck_have & duck_req),
        "duckdb_missing": missing_duckdb_parity_evidence(),
    }
