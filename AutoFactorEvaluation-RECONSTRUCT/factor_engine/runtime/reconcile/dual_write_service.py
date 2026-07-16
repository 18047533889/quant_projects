# -*- coding: utf-8
"""ClickHouse 双写服务：从 FactorEngine 抽离，供 materialize / reconcile 共用。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from logging_utils import get_logger

logger = get_logger("runtime.dual_write_service")


def append_clickhouse_to_summary(
    summary: dict[str, Any],
    *,
    factor_id: str,
    result: pd.Series,
    ast_hash: str,
    write_target: str,
    data_snapshot_id: str | None = None,
    clickhouse_table: str | None = None,
    preserve_invalid_rows: bool = False,
    value_dtype: str = "float32",
    write_metadata: bool = True,
    ensure_table: bool = True,
    dq_check: bool = False,
    dq_strict: bool = True,
    dq_thresholds=None,
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_username: str | None = None,
    ch_password: str | None = None,
    ch_secure: bool | None = None,
) -> dict[str, Any]:
    """Parquet/staging 落盘后追加 ClickHouse 写入。"""
    from storage.exceptions import DualWriteError
    from storage.write_targets import ClickHouseWriteTarget

    target = ClickHouseWriteTarget(
        table=clickhouse_table or "factor_values",
        host=ch_host,
        port=ch_port,
        database=ch_database,
        username=ch_username,
        password=ch_password,
        secure=ch_secure,
    )
    partial = dict(summary)
    partial["write_target"] = str(write_target).lower()
    partial["primary_write_completed"] = True
    try:
        ch_result = target.write_factor_series(
            factor_id,
            result,
            factor_version=ast_hash[:16],
            data_snapshot_id=data_snapshot_id,
            ensure_table=ensure_table,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
        )
    except Exception as exc:
        partial["partial_write"] = True
        partial["clickhouse_error"] = str(exc)
        raise DualWriteError(
            f"ClickHouse 写入失败（主存储可能已成功）: factor_id={factor_id}, error={exc}",
            summary=partial,
            cause=exc,
        ) from exc

    merged = dict(partial)
    merged["clickhouse"] = {
        "factor_id": ch_result["factor_id"],
        "table": ch_result["table"],
        "rows_written": ch_result["rows_written"],
        "database": ch_result["database"],
    }
    if ch_result.get("dq_report") is not None:
        merged["clickhouse_dq_report"] = ch_result["dq_report"]
    merged["partial_write"] = False
    return merged


def dual_write_clickhouse(
    materializer,
    summary: dict[str, Any],
    *,
    factor_id: str,
    result: pd.Series,
    ast_hash: str,
    write_target: str,
    data_snapshot_id: str | None = None,
    clickhouse_table: str | None = None,
    preserve_invalid_rows: bool = False,
    value_dtype: str = "float32",
    write_metadata: bool = True,
    ensure_table: bool = True,
    dq_check: bool = False,
    dq_strict: bool = True,
    dq_thresholds=None,
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_username: str | None = None,
    ch_password: str | None = None,
    ch_secure: bool | None = None,
) -> dict[str, Any]:
    """ClickHouse 双写；defer watermark 时成功 commit、失败 abort。"""
    from storage.exceptions import DualWriteError

    deferred = bool(summary.get("watermark_deferred"))
    try:
        merged = append_clickhouse_to_summary(
            summary,
            factor_id=factor_id,
            result=result,
            ast_hash=ast_hash,
            write_target=write_target,
            data_snapshot_id=data_snapshot_id,
            clickhouse_table=clickhouse_table,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
            ensure_table=ensure_table,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
        )
    except DualWriteError as exc:
        if deferred:
            materializer.abort_deferred_materialization(
                exc.summary or summary,
                error=str(exc.cause or exc),
            )
        raise

    if deferred:
        return materializer.commit_deferred_materialization(merged)
    return merged
