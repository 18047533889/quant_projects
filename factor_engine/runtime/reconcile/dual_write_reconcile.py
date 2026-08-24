# -*- coding: utf-8
"""双写（staging + ClickHouse）对账、修复与运维 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from factor_engine.storage.catalog import FactorCatalog, _parse_json_field
from factor_engine.storage.exceptions import FactorNotFoundError
from factor_engine.storage.result_store import PandasResultStore

LoadSource = Literal["local", "staging"]


def _dual_write_still_open(cat: FactorCatalog, factor_id: str) -> bool:
    """最新 run 为失败且其后无 repair 记录时视为仍开放。"""
    for run in cat.list_runs(factor_id, limit=50):
        extra = _parse_json_field(run.get("extra_json"))
        if not extra and "extra" in run:
            extra = run.get("extra") or {}
        if extra.get("dual_write_repaired"):
            return False
        if extra.get("dual_write_failed"):
            return True
    return False


def _load_series_for_repair(
    factor_id: str,
    lake_root: Path,
    *,
    prefer: LoadSource | None = None,
) -> tuple[pd.Series, LoadSource]:
    """优先 local Parquet，其次 factor_lake_staging。"""
    cat = FactorCatalog(lake_root / "_catalog.sqlite")
    order: list[LoadSource] = ["local", "staging"]
    if prefer == "staging":
        order = ["staging", "local"]
    elif prefer == "local":
        order = ["local", "staging"]

    last_error: Exception | None = None
    for source in order:
        try:
            if source == "local":
                store = PandasResultStore(lake_root, catalog=cat)
                return store.load_factor_series(factor_id), "local"
            from factor_engine.storage.staging_loader import load_factor_series_from_staging

            return load_factor_series_from_staging(factor_id), "staging"
        except Exception as exc:
            last_error = exc
            continue
    raise FileNotFoundError(
        f"无法从 local/staging 加载因子 '{factor_id}': {last_error}"
    ) from last_error


def _merge_watermark_range(
    existing: dict | None,
    series: pd.Series,
) -> tuple[str, str, int]:
    start = pd.Timestamp(series.index.get_level_values(0).min()).isoformat()
    end = pd.Timestamp(series.index.get_level_values(0).max()).isoformat()
    rows = len(series)
    if existing:
        if existing.get("start_date") and str(existing["start_date"]) < start:
            start = str(existing["start_date"])
        if existing.get("end_date") and str(existing["end_date"]) > end:
            end = str(existing["end_date"])
        if existing.get("row_count"):
            rows = max(int(existing["row_count"]), rows)
    return start, end, rows


def reconcile_dual_write_state(
    *,
    factor_id: str,
    lake_root: str | Path,
    catalog: FactorCatalog | None = None,
) -> dict[str, Any]:
    """检查因子是否存在未完成的 ClickHouse 双写（staging 可能已领先 watermark）。"""
    root = Path(lake_root)
    cat = catalog or FactorCatalog(root / "_catalog.sqlite")
    still_open = _dual_write_still_open(cat, factor_id)
    failures = cat.list_dual_write_failures(factor_id=factor_id, limit=20) if still_open else []
    watermark = cat.get_watermark(factor_id)
    latest = failures[0] if failures else None
    staging_written = bool(
        latest and (latest.get("extra") or {}).get("staging_written")
    )
    return {
        "ok": not still_open,
        "factor_id": factor_id,
        "lake_root": str(root),
        "watermark": watermark,
        "open_dual_write_failures": len(failures) if still_open else 0,
        "dual_write_still_open": still_open,
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
            "run_pipeline reconcile dual-write --repair（自动尝试 staging/local）"
            if still_open
            else None
        ),
    }


def list_open_dual_write_failures(
    *,
    lake_root: str | Path,
    factor_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出仍未 repair 的双写失败记录。"""
    cat = FactorCatalog(Path(lake_root) / "_catalog.sqlite")
    rows = cat.list_dual_write_failures(factor_id=factor_id, limit=limit * 5)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        fid = str(row.get("factor_id") or "")
        if not fid or fid in seen:
            continue
        if not _dual_write_still_open(cat, fid):
            continue
        seen.add(fid)
        out.append(
            {
                "run_id": row.get("run_id"),
                "factor_id": fid,
                "created_at": row.get("created_at"),
                "error": (row.get("extra") or {}).get("dual_write_error"),
                "staging_written": (row.get("extra") or {}).get("staging_written"),
            }
        )
        if len(out) >= limit:
            break
    return out


def reconcile_all_dual_write_states(
    *,
    lake_root: str | Path,
    limit: int = 200,
) -> dict[str, Any]:
    """扫描因子湖内所有仍存在开放双写失败的因子。"""
    open_rows = list_open_dual_write_failures(lake_root=lake_root, limit=limit)
    factor_ids = sorted({str(r["factor_id"]) for r in open_rows})
    cat = FactorCatalog(Path(lake_root) / "_catalog.sqlite")
    reports = [
        reconcile_dual_write_state(factor_id=fid, lake_root=lake_root, catalog=cat)
        for fid in factor_ids
    ]
    return {
        "ok": not reports,
        "lake_root": str(lake_root),
        "factors_with_failures": len(reports),
        "reports": reports,
    }


def repair_dual_write_clickhouse(
    *,
    factor_id: str,
    lake_root: str | Path,
    clickhouse_table: str = "factor_values",
    commit_watermark: bool = True,
    prefer_source: LoadSource | None = None,
    compensate_staging: bool = True,
    compensate_after: str | None = None,
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_username: str | None = None,
    ch_password: str | None = None,
    ch_secure: bool | None = None,
) -> dict[str, Any]:
    """从 local Parquet 或 staging 补写 ClickHouse，并可选提交/合并 watermark。"""
    root = Path(lake_root)
    cat = FactorCatalog(root / "_catalog.sqlite")
    info = cat.get_factor_info(factor_id)
    if info is None:
        raise FactorNotFoundError(f"Factor not registered: {factor_id}")

    staging_compensation: dict[str, Any] | None = None
    if compensate_staging:
        from factor_engine.storage.staging_loader import delete_staging_rows

        watermark = cat.get_watermark(factor_id)
        after = compensate_after
        if after is None and watermark and watermark.get("end_date"):
            after = str(watermark["end_date"])
        staging_compensation = delete_staging_rows(
            factor_id,
            after=after,
        )

    try:
        series, load_source = _load_series_for_repair(
            factor_id,
            root,
            prefer=prefer_source,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "factor_id": factor_id,
            "reason": "no_data_source",
            "message": str(exc),
            "recommendation": (
                "确认 factor_lake_staging 或本地 factors/ 目录存在数据后重试 --repair；"
                "或重跑 materialize(staging_clickhouse)"
            ),
        }

    if series is None or len(series) == 0:
        return {"ok": False, "factor_id": factor_id, "reason": "empty_series"}

    from factor_engine.storage.clickhouse_materializer import ClickHouseMaterializer

    ast_hash = str(info.get("ast_hash") or "")
    ch_mat = ClickHouseMaterializer(
        table=clickhouse_table,
        host=ch_host,
        port=ch_port,
        database=ch_database,
        username=ch_username,
        password=ch_password,
        secure=ch_secure,
    )
    ch_summary = ch_mat.materialize(
        factor_id=factor_id,
        result=series,
        factor_version=ast_hash[:16],
        ensure_table=True,
    )

    watermark = cat.get_watermark(factor_id)
    if commit_watermark:
        start, end, rows = _merge_watermark_range(watermark, series)
        cat.update_watermark(
            factor_id=factor_id,
            start_date=start,
            end_date=end,
            row_count=rows,
        )
        watermark = cat.get_watermark(factor_id)

    from factor_engine.runtime.lineage import new_run_id

    cat.record_run(
        {
            "run_id": new_run_id(),
            "factor_id": factor_id,
            "factor_name": info.get("description") or factor_id,
            "ast_hash": ast_hash or "__repair__",
            "operator_catalog_hash": "repair",
            "lookback": 0,
            "referenced_columns": [],
            "dq_passed": True,
            "row_count": len(series),
            "extra": {
                "dual_write_repaired": True,
                "clickhouse_rows": ch_summary.rows_written,
                "repair_source": load_source,
            },
        }
    )

    return {
        "ok": True,
        "factor_id": factor_id,
        "load_source": load_source,
        "staging_compensation": staging_compensation,
        "clickhouse": {
            "table": ch_summary.table,
            "rows_written": ch_summary.rows_written,
            "database": ch_summary.database,
        },
        "watermark": watermark,
        "rows_repaired": len(series),
    }
