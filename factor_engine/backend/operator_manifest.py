# -*- coding: utf-8
"""算子 manifest：capability + fastpath 统一视图。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.composite_evidence import COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED


def build_operator_manifest_entry(canon: str) -> dict[str, Any]:
    """构建单个 canonical 的 manifest 条目（capability + fastpath 视图）。"""
    from backend.fastpath_coverage import build_fastpath_coverage_row
    from backend.operator_capability import capability_for, resolve_canonical
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.sql_tiers import effective_sql_production_safe
    from cleaned_operators.operator_spec import build_operator_spec, spec_to_manifest_entry
    from planner.composite_lowering import lowered_primitives as probe_lowered_primitives
    from backend.polars_long_production import polars_long_production_tier

    name = resolve_canonical(canon)
    spec = build_operator_spec(name)
    if spec is None:
        return {"canonical": name, "status": "missing"}
    base = spec_to_manifest_entry(spec)
    row = build_fastpath_coverage_row(name)
    duck = capability_for(name, "duckdb_sql")
    polars = capability_for(name, "polars")
    ch = capability_for(name, "clickhouse_sql")
    tier = infer_polars_long_tier(name)

    polars_long_label = tier
    if tier == "native" and row.polars_long_native_production_safe:
        polars_long_label = "native_production_safe"
    elif tier == "native":
        polars_long_label = "native"
    native_tier = polars_long_production_tier(name) if tier == "native" else tier
    lowered = probe_lowered_primitives(name) if row.lowering_available else None

    return {
        **base,
        "canonical": name,
        "shape_preserving": spec.shape_preserving,
        "production_policy": spec.production_policy,
        "execution_kind": row.execution_kind,
        "lowering_available": row.lowering_available,
        "lowered_primitives": list(lowered) if lowered else [],
        "composite_dual_backend_capable": row.composite_dual_backend_capable,
        "composite_production_safe": row.composite_production_safe,
        "composite_reference_parity": name in COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED,
        "composite_reference_to_lowered_pandas": name in COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED,
        "polars_long_fastpath": row.polars_long_fastpath,
        "duckdb_fastpath": row.duckdb_fastpath,
        "dual_backend_fastpath": row.dual_backend_fastpath,
        "composite_dual_backend_fastpath": row.composite_dual_backend_fastpath,
        "effective_production_fastpath": row.effective_production_fastpath,
        "effective_dual_backend_fastpath": row.effective_dual_backend_fastpath,
        "pandas": capability_for(name, "pandas_numpy").status,
        "polars_panel": polars.status,
        "polars_long": polars_long_label,
        "polars_long_tier": tier,
        "polars_long_native_tier": native_tier,
        "duckdb_sql": duck.status,
        "duckdb_sql_effective_production_safe": effective_sql_production_safe(name),
        "clickhouse_sql": ch.status,
        "production_fast_path": row.production_fast_path,
        "fastpath_block_reason": row.fastpath_block_reason,
        "parity_verified": row.dual_backend_parity_verified,
        "polars_parity_verified": row.pandas_polars_long_parity,
        "duckdb_parity_verified": row.pandas_duckdb_parity,
        "dual_backend_parity_verified": row.dual_backend_parity_verified,
        "benchmark": row.benchmark_available,
        "research_allowed": True,
        "production_allowed": spec.allow_in_production,
        "production_fastpath_allowed": row.effective_production_fastpath,
    }


def build_operator_manifest(
    *,
    production_only: bool = False,
    fastpath_only: bool = False,
) -> list[dict[str, Any]]:
    """构建全量或过滤后的算子 manifest 列表。"""
    from cleaned_operators.operator_spec import iter_operator_specs

    rows: list[dict[str, Any]] = []
    for spec in iter_operator_specs():
        if production_only and not spec.allow_in_production:
            continue
        entry = build_operator_manifest_entry(spec.canonical)
        if fastpath_only and not entry.get("effective_production_fastpath"):
            continue
        rows.append(entry)
    return rows


def _manifest_metadata() -> dict[str, str]:
    """Manifest 生成元数据（schema v2）。"""
    import subprocess
    from datetime import datetime, timezone

    sha = "unknown"
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_commit_sha": sha,
    }


def write_operator_manifest(path: Path, **kwargs: Any) -> None:
    """将算子 manifest 写入 JSON 文件。"""
    data = {
        **_manifest_metadata(),
        "operators": build_operator_manifest(**kwargs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
