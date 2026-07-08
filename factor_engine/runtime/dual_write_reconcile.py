# -*- coding: utf-8
"""双写（staging + ClickHouse）对账：失败 run、watermark 与 staging 一致性。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from storage.catalog import FactorCatalog


def reconcile_dual_write_state(
    *,
    factor_id: str,
    lake_root: str | Path,
    catalog: FactorCatalog | None = None,
) -> dict[str, Any]:
    """检查因子是否存在未完成的 ClickHouse 双写（staging 可能已领先 watermark）。"""
    root = Path(lake_root)
    cat = catalog or FactorCatalog(root / "_catalog.sqlite")
    failures = cat.list_dual_write_failures(factor_id=factor_id, limit=20)
    watermark = cat.get_watermark(factor_id)
    latest = failures[0] if failures else None
    staging_written = bool(
        latest and (latest.get("extra") or {}).get("staging_written")
    )
    return {
        "ok": not failures,
        "factor_id": factor_id,
        "lake_root": str(root),
        "watermark": watermark,
        "open_dual_write_failures": len(failures),
        "latest_failure": (
            {
                "run_id": latest.get("run_id"),
                "created_at": latest.get("created_at"),
                "error": (latest.get("extra") or {}).get("dual_write_error"),
                "downstream": (latest.get("extra") or {}).get("dual_write_downstream"),
                "staging_written": staging_written,
            }
            if latest
            else None
        ),
        "recommendation": (
            "重跑 materialize(write_target=staging_clickhouse) 或人工对账 staging/CH"
            if failures
            else None
        ),
    }


def list_open_dual_write_failures(
    *,
    lake_root: str | Path,
    factor_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出 catalog 中未完成的双写失败记录。"""
    cat = FactorCatalog(Path(lake_root) / "_catalog.sqlite")
    rows = cat.list_dual_write_failures(factor_id=factor_id, limit=limit)
    return [
        {
            "run_id": row.get("run_id"),
            "factor_id": row.get("factor_id"),
            "created_at": row.get("created_at"),
            "error": (row.get("extra") or {}).get("dual_write_error"),
            "staging_written": (row.get("extra") or {}).get("staging_written"),
        }
        for row in rows
    ]
