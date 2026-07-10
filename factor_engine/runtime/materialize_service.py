# -*- coding: utf-8 -*-
"""物化编排：Parquet/staging + 可选 ClickHouse 双写。

本模块封装 ``FactorEngine.materialize*`` 的落盘逻辑：构建 lineage、
调用 ``ParquetMaterializer``、记录因子依赖 catalog，并按需双写 ClickHouse。
"""

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
    """将用户 write_target 映射为 Parquet 物化器可识别的目标。

    ``staging_clickhouse`` 仅 Parquet 写 staging；``clickhouse`` 跳过 Parquet。
    """
    target = str(write_target or "local").lower()
    if target == "staging_clickhouse":
        return "staging"
    if target == "clickhouse":
        return "clickhouse"
    return target


def needs_clickhouse_write(write_target: str) -> bool:
    """write_target 是否需要 ClickHouse 双写（``clickhouse`` 或 ``staging_clickhouse``）。"""
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
    """将 ``run`` 输出物化到因子湖（及可选 ClickHouse）。

    流程：构建 lineage → ``ParquetMaterializer.materialize`` →
    记录因子依赖 catalog → 按需 ``dual_write_clickhouse``。

    Args:
        engine: 因子引擎实例（提供 ``data_source``）。
        factor: 因子对象。
        output: ``run`` / ``run_incremental`` 返回的字典，须含 ``analysis`` 与 ``result``。
        target: 写入目标（``local``/``staging``/``clickhouse``/``staging_clickhouse`` 等）。
        lake_root: 因子湖根目录。
        staging_dataset: staging 数据集名称。
        factor_id: 落盘因子 ID；缺省用 ``factor.name``。
        author/frequency/description/expression: 元数据字段。
        dq_check/dq_strict/dq_thresholds: 产出 DQ 门禁参数。
        write_metadata: 是否写入 run 元数据。
        data_source_config: 数据源配置快照（lineage 用）。
        resume_materialize: 是否断点续写分区。
        isolate_partition_failures: 单分区失败是否隔离。
        preserve_invalid_rows: 是否保留 inf 等为 ``is_valid=0``。
        value_dtype: 落盘值 dtype。
        clickhouse_* / ch_*: ClickHouse 连接与表参数。
        lineage_mode: lineage 模式（``full`` / ``incremental``）。
        storage_format: Parquet 存储格式（``long`` 等）。
        partition_columns: 自定义分区列。

    Returns:
        原 ``output`` 字典，附加 ``materialization`` 键含落盘摘要。
    """
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
    """已有 run 输出 + ``ResolvedMaterializeKwargs`` → 落盘。

    供 ``materialize_many_from_config`` 等批量路径复用，避免重复解析 kwargs。
    """
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
    """是否可对多因子共享一次 ``run_many`` 计算（非增量、非 resume 分区）。

    增量物化或断点续写分区需逐因子独立执行窗口。
    """
    return not opts.resume_materialize and opts.since is None

