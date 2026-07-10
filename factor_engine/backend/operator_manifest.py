# -*- coding: utf-8
"""算子 manifest：capability + fastpath 统一视图。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_operator_manifest_entry(canon: str) -> dict[str, Any]:
    """构建单个 canonical 的 manifest 条目（capability + fastpath 视图）。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        含各 backend 状态、fastpath 标志与 parity 信息的字典。
    """
    from backend.fastpath_coverage import build_fastpath_coverage_row
    from backend.operator_capability import capability_for, resolve_canonical
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.sql_tiers import effective_sql_production_safe
    from cleaned_operators.operator_spec import build_operator_spec, spec_to_manifest_entry

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
    from backend.polars_long_production import polars_long_production_tier

    polars_long_label = tier
    if tier == "native" and row.polars_long_native_production_safe:
        polars_long_label = "native_production_safe"
    elif tier == "native":
        polars_long_label = "native"
    native_tier = polars_long_production_tier(name) if tier == "native" else tier
    return {
        **base,
        "canonical": name,
        "shape_preserving": spec.shape_preserving,
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
        "parity_verified": row.pandas_polars_long_parity or row.pandas_duckdb_parity,
        "benchmark": row.benchmark_available,
        "research_allowed": True,
        "production_allowed": spec.allow_in_production,
        "production_fastpath_allowed": row.production_fast_path,
    }


def build_operator_manifest(
    *,
    production_only: bool = False,
    fastpath_only: bool = False,
) -> list[dict[str, Any]]:
    """构建全量或过滤后的算子 manifest 列表。

    参数:
        production_only: 为 ``True`` 时仅包含 ``allow_in_production`` 算子。
        fastpath_only: 为 ``True`` 时仅包含可走 production fast path 的算子。

    返回:
        manifest 条目字典列表。
    """
    from cleaned_operators.operator_spec import iter_operator_specs

    rows: list[dict[str, Any]] = []
    for spec in iter_operator_specs():
        if production_only and not spec.allow_in_production:
            continue
        entry = build_operator_manifest_entry(spec.canonical)
        if fastpath_only and not entry.get("production_fastpath_allowed"):
            continue
        rows.append(entry)
    return rows


def write_operator_manifest(path: Path, **kwargs: Any) -> None:
    """将算子 manifest 写入 JSON 文件。

    参数:
        path: 输出文件路径。
        **kwargs: 透传给 :func:`build_operator_manifest` 的过滤参数。
    """
    data = {
        "schema_version": 1,
        "operators": build_operator_manifest(**kwargs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
