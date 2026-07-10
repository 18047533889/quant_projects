# -*- coding: utf-8
"""Backend fast path coverage matrix（报表 / CI）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_BENCHMARK_JSON = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_cost_baseline.json"


@dataclass(frozen=True)
class FastpathCoverageRow:
    canonical: str
    allow_in_production: bool
    pandas_numpy_status: str
    polars_panel_status: str
    polars_long_tier: str
    polars_long_native_tier: str
    polars_long_native: bool
    polars_long_map_groups: bool
    polars_long_registry: bool
    duckdb_sql_status: str
    clickhouse_sql_status: str
    sql_implemented: bool
    sql_parity_verified: bool
    sql_production_safe: bool
    sql_emitter_ok: bool
    duckdb_sql_production_safe: bool
    clickhouse_sql_production_safe: bool
    polars_long_native_production_safe: bool
    pandas_polars_long_parity: bool
    pandas_duckdb_parity: bool
    benchmark_available: bool
    production_fast_path: bool
    fastpath_block_reason: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "allow_in_production": self.allow_in_production,
            "production_allowed": self.allow_in_production,
            "pandas_numpy_status": self.pandas_numpy_status,
            "polars_panel_status": self.polars_panel_status,
            "polars_long_tier": self.polars_long_tier,
            "polars_long_native_tier": self.polars_long_native_tier,
            "polars_long_native": self.polars_long_native,
            "polars_long_map_groups": self.polars_long_map_groups,
            "polars_long_registry": self.polars_long_registry,
            "duckdb_sql": self.duckdb_sql_status,
            "duckdb_sql_status": self.duckdb_sql_status,
            "clickhouse_sql_status": self.clickhouse_sql_status,
            "sql_implemented": self.sql_implemented,
            "sql_parity_verified": self.sql_parity_verified,
            "sql_production_safe": self.sql_production_safe,
            "sql_emitter_ok": self.sql_emitter_ok,
            "duckdb_sql_production_safe": self.duckdb_sql_production_safe,
            "clickhouse_sql_production_safe": self.clickhouse_sql_production_safe,
            "polars_long_native_production_safe": self.polars_long_native_production_safe,
            "parity": self.pandas_polars_long_parity or self.pandas_duckdb_parity,
            "pandas_polars_long_parity": self.pandas_polars_long_parity,
            "pandas_duckdb_parity": self.pandas_duckdb_parity,
            "benchmark": self.benchmark_available,
            "benchmark_available": self.benchmark_available,
            "production_fast_path": self.production_fast_path,
            "fastpath_block_reason": self.fastpath_block_reason,
        }


def _benchmark_canonicals() -> frozenset[str]:
    if not _BENCHMARK_JSON.is_file():
        return frozenset()
    try:
        import json

        data = json.loads(_BENCHMARK_JSON.read_text(encoding="utf-8"))
        ops = data.get("operators") or data.get("canonicals") or {}
        if isinstance(ops, dict):
            return frozenset(str(k) for k in ops)
        if isinstance(ops, list):
            return frozenset(str(x) for x in ops)
    except Exception:
        return frozenset()
    return frozenset()


def _fastpath_block_reason(
    *,
    canon: str,
    allow_in_production: bool,
    duckdb_ok: bool,
    polars_native_ok: bool,
    polars_tier: str,
) -> str:
    from backend.polars_long_production import POLARS_LONG_FASTPATH_DEFERRED
    from backend.sql_tiers import SQL_PRODUCTION_DEFERRED_CANONICALS

    if canon in {"column", "literal", "materialized_series", "plan_ref"}:
        return ""
    if canon in POLARS_LONG_FASTPATH_DEFERRED or canon in SQL_PRODUCTION_DEFERRED_CANONICALS:
        return "deferred_complex_op"
    if duckdb_ok or polars_native_ok:
        return ""
    if polars_tier in {"map_groups", "registry"}:
        return f"polars_long_{polars_tier}_not_fastpath"
    if polars_tier == "passthrough":
        return "polars_long_passthrough"
    if not allow_in_production:
        return "not_production_allowed"
    reasons: list[str] = []
    if not duckdb_ok:
        reasons.append("no_duckdb_production_safe")
    if not polars_native_ok:
        reasons.append("no_polars_long_native_production_safe")
    return ";".join(reasons) or "no_fastpath_backend"


def build_fastpath_coverage_row(canon: str) -> FastpathCoverageRow:
    from backend.operator_capability import (
        _sql_emitter_ok,
        capability_for,
        polars_long_tier,
        resolve_canonical,
    )
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.polars_long_production import (
        duckdb_triple_parity_verified,
        is_polars_long_native_production_safe,
        polars_long_production_tier,
    )
    from backend.production_fast_path import PRODUCTION_TRIPLE_PARITY_CANONICALS
    from backend.sql_tiers import (
        SQL_IMPLEMENTED_CANONICALS,
        SQL_PARITY_VERIFIED_CANONICALS,
        effective_sql_production_safe,
    )
    from backend.sql_pushdown.clickhouse_capabilities import effective_clickhouse_production_safe
    from cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED
    from cleaned_operators.operator_spec import build_operator_spec

    name = resolve_canonical(canon)
    spec = build_operator_spec(name)
    tier = infer_polars_long_tier(name)
    pandas_cap = capability_for(name, "pandas_numpy")
    polars_cap = capability_for(name, "polars")
    duck_cap = capability_for(name, "duckdb_sql")
    ch_cap = capability_for(name, "clickhouse_sql")

    sql_emitter_ok = _sql_emitter_ok(name)
    duckdb_prod = effective_sql_production_safe(name) and sql_emitter_ok
    ch_prod = effective_clickhouse_production_safe(name) and sql_emitter_ok
    polars_native_prod = is_polars_long_native_production_safe(name)
    pandas_polars_parity = name in POLARS_PARITY_VERIFIED or name in PRODUCTION_TRIPLE_PARITY_CANONICALS
    pandas_duckdb_parity = duckdb_triple_parity_verified(name)
    benchmark_set = _benchmark_canonicals()
    block = _fastpath_block_reason(
        canon=name,
        allow_in_production=bool(spec.allow_in_production),
        duckdb_ok=duckdb_prod,
        polars_native_ok=polars_native_prod,
        polars_tier=tier,
    )
    production_fast_path = bool(spec.allow_in_production) and (duckdb_prod or ch_prod or polars_native_prod) and not block

    native_tier = polars_long_production_tier(name) if tier == "native" else tier

    return FastpathCoverageRow(
        canonical=name,
        allow_in_production=bool(spec.allow_in_production),
        pandas_numpy_status=pandas_cap.status,
        polars_panel_status=polars_cap.status,
        polars_long_tier=tier,
        polars_long_native_tier=native_tier,
        polars_long_native=tier == "native",
        polars_long_map_groups=tier == "map_groups",
        polars_long_registry=tier == "registry",
        duckdb_sql_status=duck_cap.status,
        clickhouse_sql_status=ch_cap.status,
        sql_implemented=name in SQL_IMPLEMENTED_CANONICALS,
        sql_parity_verified=name in SQL_PARITY_VERIFIED_CANONICALS,
        sql_production_safe=duckdb_prod,
        sql_emitter_ok=sql_emitter_ok,
        duckdb_sql_production_safe=duckdb_prod,
        clickhouse_sql_production_safe=ch_prod,
        polars_long_native_production_safe=polars_native_prod,
        pandas_polars_long_parity=pandas_polars_parity,
        pandas_duckdb_parity=pandas_duckdb_parity,
        benchmark_available=name in benchmark_set,
        production_fast_path=production_fast_path,
        fastpath_block_reason=block if not production_fast_path else "",
    )


def build_fastpath_coverage_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[FastpathCoverageRow]:
    from backend.operator_capability import build_capability_matrix, resolve_canonical

    if canonicals is None:
        summaries = build_capability_matrix()
        names = [s.canonical for s in summaries]
    else:
        names = sorted({resolve_canonical(c) for c in canonicals})
    return [build_fastpath_coverage_row(c) for c in sorted(names)]


def summarize_fastpath_coverage(rows: Sequence[FastpathCoverageRow]) -> dict[str, Any]:
    prod_rows = [r for r in rows if r.allow_in_production]
    fast = [r for r in prod_rows if r.production_fast_path]
    blocked = [r for r in prod_rows if not r.production_fast_path]
    return {
        "total": len(rows),
        "production_allowed_count": len(prod_rows),
        "production_fast_path_count": len(fast),
        "production_blocked_count": len(blocked),
        "sql_production_safe_emitter_ok": sum(
            1 for r in rows if r.duckdb_sql_production_safe and r.sql_emitter_ok
        ),
        "clickhouse_sql_production_safe_count": sum(
            1 for r in rows if r.clickhouse_sql_production_safe and r.sql_emitter_ok
        ),
        "polars_long_native_production_safe_count": sum(
            1 for r in rows if r.polars_long_native_production_safe
        ),
        "benchmark_available_count": sum(1 for r in rows if r.benchmark_available),
        "production_fast_path": sorted(r.canonical for r in fast),
        "production_blocked_sample": [
            {"canonical": r.canonical, "reason": r.fastpath_block_reason}
            for r in blocked[:30]
        ],
    }
