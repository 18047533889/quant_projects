# -*- coding: utf-8
"""物化编排：Parquet/staging + 可选 ClickHouse 双写。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from api.factor import Factor
from logging_utils import get_logger
from runtime import dual_write_service, lineage_service
from storage.catalog import compute_ir_hash
from storage.materializer import ParquetMaterializer

logger = get_logger("runtime.materialize_service")


def resolve_parquet_write_target(write_target: str) -> str:
    target = str(write_target or "local").lower()
    if target == "staging_clickhouse":
        return "staging"
    if target == "clickhouse":
        return "clickhouse"
    return target


def needs_clickhouse_write(write_target: str) -> bool:
    return str(write_target or "local").lower() in ("clickhouse", "staging_clickhouse")


def execute_materialize(
    engine: Any,
    factor: Factor,
    output: dict[str, Any],
    *,
    target: str,
    lake_root: str | Path | None,
    staging_dataset: str,
    factor_id: str | None,
    author: str | None,
    frequency: str | None,
    description: str | None,
    expression: str | None,
    dq_check: bool,
    dq_strict: bool,
    dq_thresholds,
    write_metadata: bool,
    data_source_config: dict | None,
    resume_materialize: bool,
    isolate_partition_failures: bool,
    preserve_invalid_rows: bool,
    value_dtype: str,
    clickhouse_table: str | None,
    ch_ensure_table: bool,
    ch_host: str | None,
    ch_port: int | None,
    ch_database: str | None,
    ch_username: str | None,
    ch_password: str | None,
    ch_secure: bool | None,
    lineage_mode: str = "full",
    storage_format: str = "long",
    partition_columns: list[str] | None = None,
) -> dict[str, Any]:
    """run 输出 → lineage → ParquetMaterializer → 可选 CH 双写。"""
    analysis = output["analysis"]
    lineage = lineage_service.build_materialize_lineage(
        factor=factor,
        analysis=analysis,
        output=output,
        factor_id=factor_id,
        expression=expression,
        data_source_config=data_source_config,
        data_source=engine.data_source,
        mode=lineage_mode,
    )
    materializer = ParquetMaterializer(
        lake_root=lake_root,
        staging_dataset=staging_dataset,
    )
    ast_hash = compute_ir_hash(analysis.ir)
    parquet_target = resolve_parquet_write_target(target)

    summary = materializer.materialize(
        factor_id=factor_id or factor.name,
        result=output["result"],
        ir_node=analysis.ir,
        author=author,
        frequency=frequency or factor.freq,
        description=description or factor.description,
        expression=expression,
        dq_check=dq_check,
        dq_strict=dq_strict,
        dq_thresholds=dq_thresholds,
        run_lineage={**lineage.to_dict(), "factor_id": factor_id or factor.name},
        write_metadata=write_metadata,
        data_snapshot_id=lineage_service.data_snapshot_id_from_config(data_source_config),
        data_source_config=data_source_config,
        resume=resume_materialize,
        isolate_partition_failures=isolate_partition_failures,
        preserve_invalid_rows=preserve_invalid_rows,
        value_dtype=value_dtype,
        write_target=parquet_target,
        defer_watermark=needs_clickhouse_write(target),
        storage_format=storage_format,
        partition_columns=partition_columns,
    )

    from runtime.incremental_scheduler import record_factor_dependency_from_analysis

    try:
        record_factor_dependency_from_analysis(
            materializer.catalog,
            factor_id=factor_id or factor.name,
            analysis=analysis,
            data_source=engine.data_source,
            frequency=frequency or factor.freq,
        )
    except Exception as exc:
        logger.warning("因子依赖 catalog 写入失败 factor=%s: %s", factor.name, exc)

    if needs_clickhouse_write(target):
        summary = dual_write_service.dual_write_clickhouse(
            materializer,
            summary,
            factor_id=factor_id or factor.name,
            result=output["result"],
            ast_hash=ast_hash,
            write_target=target,
            data_snapshot_id=lineage_service.data_snapshot_id_from_config(data_source_config),
            clickhouse_table=clickhouse_table,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
            ensure_table=ch_ensure_table,
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

    output["materialization"] = {
        **summary,
        "lake_root": str(materializer.lake_root),
    }
    logger.info(
        "完成落盘因子 '%s'，factor_id=%s，rows_written=%s",
        factor.name,
        summary["factor_id"],
        summary["rows_written"],
    )
    return output


def execute_materialize_from_resolved(
    engine: Any,
    factor: Factor,
    output: dict[str, Any],
    opts: Any,
    *,
    lineage_mode: str = "full",
) -> dict[str, Any]:
    """已有 run 输出 + ResolvedMaterializeKwargs → 落盘（批量物化 run_many 路径）。"""
    return execute_materialize(
        engine,
        factor,
        output,
        target=opts.write_target,
        lake_root=opts.lake_root,
        staging_dataset=opts.staging_dataset,
        factor_id=opts.factor_id,
        author=opts.author,
        frequency=opts.frequency,
        description=opts.description,
        expression=opts.expression,
        dq_check=opts.dq_check,
        dq_strict=opts.dq_strict,
        dq_thresholds=opts.dq_thresholds,
        write_metadata=True,
        data_source_config=opts.data_source_config,
        resume_materialize=opts.resume_materialize,
        isolate_partition_failures=opts.isolate_partition_failures,
        preserve_invalid_rows=opts.preserve_invalid_rows,
        value_dtype=opts.value_dtype,
        clickhouse_table=opts.clickhouse_table,
        ch_ensure_table=opts.ch_ensure_table,
        ch_host=opts.clickhouse_host,
        ch_port=opts.clickhouse_port,
        ch_database=opts.clickhouse_database,
        ch_username=opts.clickhouse_username,
        ch_password=opts.clickhouse_password,
        ch_secure=opts.clickhouse_secure,
        lineage_mode=lineage_mode,
        storage_format=opts.storage_format,
        partition_columns=list(opts.partition_columns) if opts.partition_columns else None,
    )


def can_batch_materialize_compute(opts: Any) -> bool:
    """是否可对多因子共享一次 run_many（非增量、非 resume 分区）。"""
    return not opts.resume_materialize and opts.since is None

