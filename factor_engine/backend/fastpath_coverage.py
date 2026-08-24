# -*- coding: utf-8
"""Backend fast path coverage matrix（报表 / CI）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_BENCHMARK_JSON = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_cost_baseline.json"


@dataclass(frozen=True)
class FastpathCoverageRow:
    """单 canonical 的 fast path 覆盖矩阵行。"""

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
    execution_kind: str = "primitive"
    lowering_available: bool = False
    polars_long_fastpath: bool = False
    duckdb_fastpath: bool = False
    dual_backend_fastpath: bool = False
    composite_dual_backend_capable: bool = False
    composite_production_safe: bool = False
    composite_dual_backend_fastpath: bool = False
    effective_production_fastpath: bool = False
    effective_dual_backend_fastpath: bool = False
    dual_backend_parity_verified: bool = False
    production_policy: str = "denied"
    lowered_primitives: tuple[str, ...] | None = None

    def to_csv_row(self) -> dict[str, Any]:
        """导出为 CSV/报表用扁平行字典。

        返回:
            含各 backend 状态、parity 与 fastpath 标志的字段字典。
        """
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
            "parity": self.dual_backend_parity_verified,
            "polars_parity_verified": self.pandas_polars_long_parity,
            "duckdb_parity_verified": self.pandas_duckdb_parity,
            "dual_backend_parity_verified": self.dual_backend_parity_verified,
            "pandas_polars_long_parity": self.pandas_polars_long_parity,
            "pandas_duckdb_parity": self.pandas_duckdb_parity,
            "benchmark": self.benchmark_available,
            "benchmark_available": self.benchmark_available,
            "production_fast_path": self.production_fast_path,
            "fastpath_block_reason": self.fastpath_block_reason,
            "execution_kind": self.execution_kind,
            "lowering_available": self.lowering_available,
            "polars_long_fastpath": self.polars_long_fastpath,
            "duckdb_fastpath": self.duckdb_fastpath,
            "dual_backend_fastpath": self.dual_backend_fastpath,
            "composite_dual_backend_capable": self.composite_dual_backend_capable,
            "composite_production_safe": self.composite_production_safe,
            "composite_dual_backend_fastpath": self.composite_dual_backend_fastpath,
            "effective_production_fastpath": self.effective_production_fastpath,
            "effective_dual_backend_fastpath": self.effective_dual_backend_fastpath,
            "production_policy": self.production_policy,
            "lowered_primitives": ",".join(self.lowered_primitives) if self.lowered_primitives else "",
        }


def _benchmark_canonicals() -> frozenset[str]:
    """从 benchmark JSON 读取已实测的 canonical 集合。"""
    if not _BENCHMARK_JSON.is_file():
        return frozenset()
    try:
        import json

        data = json.loads(_BENCHMARK_JSON.read_text(encoding="utf-8"))
        if data.get("generated_by") == "seed_defaults":
            return frozenset()
        provenance = data.get("provenance") or {}
        if provenance.get("measured") is False:
            return frozenset()
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
    production_policy: str,
    duckdb_ok: bool,
    polars_native_ok: bool,
    polars_tier: str,
    execution_kind: str,
    composite_structurally_capable: bool,
    composite_production_safe: bool,
) -> str:
    """推断 canonical 被 production fast path 阻断的原因（优先级有序）。"""
    from factor_engine.backend.polars_long_production import POLARS_LONG_FASTPATH_DEFERRED
    from factor_engine.backend.sql_tiers import SQL_PRODUCTION_DEFERRED_CANONICALS
    from factor_engine.cleaned_operators.operator_spec import is_production_permanently_forbidden

    if canon in {"column", "literal", "materialized_series", "plan_ref"}:
        return ""
    if not allow_in_production:
        if production_policy == "permanently_forbidden" or is_production_permanently_forbidden(canon):
            return "permanently_forbidden"
        if production_policy == "pending":
            return "composite_policy_pending"
        return "not_production_allowed"
    if production_policy == "permanently_forbidden" or is_production_permanently_forbidden(canon):
        return "permanently_forbidden"
    if canon in POLARS_LONG_FASTPATH_DEFERRED or canon in SQL_PRODUCTION_DEFERRED_CANONICALS:
        return "deferred_complex_op"
    if execution_kind == "composite":
        if composite_production_safe:
            return ""
        if composite_structurally_capable:
            return "composite_evidence_missing"
        return "composite_not_structurally_capable"
    if polars_tier in {"map_groups", "registry", "python_rolling"}:
        return f"polars_long_{polars_tier}_not_fastpath"
    if polars_tier == "passthrough":
        return "polars_long_passthrough"
    if duckdb_ok and polars_native_ok:
        return ""
    reasons: list[str] = []
    if not duckdb_ok:
        reasons.append("no_duckdb_production_safe")
    if not polars_native_ok:
        reasons.append("no_polars_long_native_production_safe")
    return ";".join(reasons) or "no_fastpath_backend"


def build_fastpath_coverage_row(canon: str) -> FastpathCoverageRow:
    """构建单个 canonical 的 fast path 覆盖行。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        聚合各 backend 状态与 parity 证据的 ``FastpathCoverageRow``。
    """
    from factor_engine.backend.operator_capability import (
        _sql_emitter_ok,
        capability_for,
        polars_long_tier,
        resolve_canonical,
    )
    from factor_engine.backend.polars_long_policy import infer_polars_long_tier
    from factor_engine.backend.polars_long_production import (
        duckdb_triple_parity_verified,
        is_polars_long_native_production_safe,
        polars_long_production_tier,
    )
    from factor_engine.backend.primitive_evidence import (
        DUCKDB_REAL_SQL_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    from factor_engine.backend.production_fast_path import PRODUCTION_TRIPLE_PARITY_CANONICALS
    from factor_engine.backend.sql_tiers import (
        SQL_IMPLEMENTED_CANONICALS,
        SQL_PARITY_VERIFIED_CANONICALS,
        effective_sql_production_safe,
    )
    from factor_engine.backend.sql_pushdown.clickhouse_capabilities import effective_clickhouse_production_safe
    from factor_engine.cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec

    from factor_engine.cleaned_operators.operator_spec import build_operator_spec, infer_production_policy

    name = resolve_canonical(canon)
    spec = build_operator_spec(name)
    # Stale evidence/allowlist entries can outlive a runtime canonical during
    # deduplication.  Coverage reporting must fail closed instead of crashing.
    allow_in_production = bool(spec is not None and spec.allow_in_production)
    tier = infer_polars_long_tier(name)
    pandas_cap = capability_for(name, "pandas_numpy")
    polars_cap = capability_for(name, "polars")
    duck_cap = capability_for(name, "duckdb_sql")
    ch_cap = capability_for(name, "clickhouse_sql")

    sql_emitter_ok = _sql_emitter_ok(name)
    duckdb_prod = effective_sql_production_safe(name) and sql_emitter_ok
    ch_prod = effective_clickhouse_production_safe(name) and sql_emitter_ok
    polars_native_prod = is_polars_long_native_production_safe(name)
    pandas_polars_parity = name in POLARS_REFERENCE_PARITY_VERIFIED
    pandas_duckdb_parity = name in DUCKDB_REAL_SQL_VERIFIED
    dual_backend_parity = pandas_polars_parity and pandas_duckdb_parity
    benchmark_set = _benchmark_canonicals()

    from factor_engine.backend.composite_evidence import composite_production_safe as is_composite_production_safe
    from factor_engine.planner.composite_lowering import (
        composite_dual_backend_capable,
        has_composite_lowering,
        infer_execution_kind,
        lowered_primitives,
    )

    execution_kind = infer_execution_kind(name)
    production_policy = infer_production_policy(name)
    lowering_available = has_composite_lowering(name)
    composite_structural = composite_dual_backend_capable(name) if lowering_available else False
    composite_prod_safe = is_composite_production_safe(name) if lowering_available else False
    lowered = lowered_primitives(name) if lowering_available else None

    block = _fastpath_block_reason(
        canon=name,
        allow_in_production=allow_in_production,
        production_policy=production_policy,
        duckdb_ok=duckdb_prod,
        polars_native_ok=polars_native_prod,
        polars_tier=tier,
        execution_kind=execution_kind,
        composite_structurally_capable=composite_structural,
        composite_production_safe=composite_prod_safe,
    )

    native_tier = polars_long_production_tier(name) if tier == "native" else tier

    direct_any = allow_in_production and (duckdb_prod or ch_prod or polars_native_prod)
    direct_dual = allow_in_production and polars_native_prod and duckdb_prod
    production_fast_path = direct_any and not block
    polars_long_fastpath = allow_in_production and polars_native_prod and not block
    duckdb_fastpath = allow_in_production and duckdb_prod and not block
    dual_backend_fastpath = direct_dual and not block
    composite_dual_backend_fastpath = composite_prod_safe and not block
    effective_production_fastpath = production_fast_path or composite_dual_backend_fastpath
    effective_dual_backend_fastpath = dual_backend_fastpath or composite_dual_backend_fastpath

    return FastpathCoverageRow(
        canonical=name,
        allow_in_production=allow_in_production,
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
        dual_backend_parity_verified=dual_backend_parity,
        benchmark_available=name in benchmark_set,
        production_fast_path=production_fast_path,
        fastpath_block_reason=block if not production_fast_path else "",
        execution_kind=execution_kind,
        lowering_available=lowering_available,
        polars_long_fastpath=polars_long_fastpath,
        duckdb_fastpath=duckdb_fastpath,
        dual_backend_fastpath=dual_backend_fastpath,
        composite_dual_backend_capable=composite_structural,
        composite_production_safe=composite_prod_safe,
        composite_dual_backend_fastpath=composite_dual_backend_fastpath,
        effective_production_fastpath=effective_production_fastpath,
        effective_dual_backend_fastpath=effective_dual_backend_fastpath,
        production_policy=production_policy,
        lowered_primitives=lowered,
    )


def build_fastpath_coverage_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[FastpathCoverageRow]:
    """构建 fast path 覆盖矩阵。

    参数:
        canonicals: 可选 canonical 子集；为 ``None`` 时使用全量 capability 矩阵。

    返回:
        ``FastpathCoverageRow`` 列表。
    """
    from factor_engine.backend.operator_capability import build_capability_matrix, resolve_canonical

    if canonicals is None:
        summaries = build_capability_matrix()
        names = [s.canonical for s in summaries]
    else:
        names = sorted({resolve_canonical(c) for c in canonicals})
    return [build_fastpath_coverage_row(c) for c in sorted(names)]


def summarize_fastpath_coverage(rows: Sequence[FastpathCoverageRow]) -> dict[str, Any]:
    """汇总 fast path 覆盖统计（报表 / CI）。

    参数:
        rows: fast path 覆盖行序列。

    返回:
        含 production 允许数、fast path 数及阻断样例的摘要字典。
    """
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
        "dual_backend_fastpath_count": sum(1 for r in rows if r.dual_backend_fastpath),
        "effective_production_fastpath_count": sum(1 for r in rows if r.effective_production_fastpath),
        "effective_dual_backend_fastpath_count": sum(
            1 for r in rows if r.effective_dual_backend_fastpath
        ),
        "composite_lowering_count": sum(1 for r in rows if r.lowering_available),
        "composite_dual_backend_capable_count": sum(
            1 for r in rows if r.composite_dual_backend_capable
        ),
        "composite_production_safe_count": sum(1 for r in rows if r.composite_production_safe),
        "benchmark_available_count": sum(1 for r in rows if r.benchmark_available),
        "production_fast_path": sorted(r.canonical for r in fast),
        "production_blocked_sample": [
            {"canonical": r.canonical, "reason": r.fastpath_block_reason}
            for r in blocked[:30]
        ],
    }
