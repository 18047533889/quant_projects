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
from runtime.production_policy import is_production_mode
from storage.catalog import compute_ir_hash
from storage.exceptions import (
    MaterializedButCatalogCommitFailed,
    UnreconstructableDataSource,
)
from storage.materializer import ParquetMaterializer

logger = get_logger("runtime.materialize_service")


def _backend_kind(backend: Any) -> str | None:
    """从 backend 实例反推规范后端名（full definition 用，best-effort）。"""
    name = type(backend).__name__.lower()
    if "hybrid_long" in name:
        return "hybrid_long"
    if "polars_long" in name:
        return "polars_long"
    if "hybrid" in name:
        return "hybrid"
    if "polars" in name:
        return "polars"
    if "duckdb" in name or "pushdown" in name:
        return "sql"
    if "debug" in name:
        return "debug"
    return "pandas"


def _build_full_factor_definition(
    *,
    engine: Any,
    factor: Factor,
    factor_id: str,
    author: str | None,
    frequency: str | None,
    description: str | None,
    expression: str | None,
    data_source_config: dict | None,
    analysis: Any,
    pit_enforce: bool | None = None,
    scope: Any = None,
) -> dict[str, Any]:
    """从 materialize 现场收集因子完整重建规格（R11 #5）。

    事件增量重建引擎时，缺 full definition 的因子会退化成 surface=None /
    dialect=None / market=None / universe=None / run_mode=None / backend=pandas /
    pit_enforce=None —— 即「公式一样，但执行语义已经不是原来的 factor」。
    正常成功 materialize 必须原子持久化完整规格。

    #收官轮 P0：所有语义字段（market/universe/frequency/calendar/decision_policy）
    一律取 canonical ``FactorExecutionScope``（``_scope_from_factor``，优先
    ``factor.semantic_identity``）——绝不能在这里重新拼第二套 ``factor.freq`` /
    ``factor.universe`` 语义，否则会出现「执行 scope=5m、落库 identity=1d」分裂。
    """
    ds = getattr(engine, "data_source", None)
    if pit_enforce is None:
        pit_enforce = getattr(ds, "pit_enforce", None)
    scope_f = getattr(scope, "frequency", None)
    scope_u = getattr(scope, "universe_id", None)
    scope_m = getattr(scope, "market", None)
    scope_c = getattr(scope, "calendar_id", None)
    scope_d = getattr(scope, "decision_time_policy", None)
    market = (
        (scope_m or None)
        or getattr(factor, "market", None)
        or getattr(ds, "market", None)
        or getattr(ds, "market_code", None)
    )
    calendar = (
        (scope_c or None)
        or getattr(ds, "calendar_id", None)
        or getattr(ds, "calendar", None)
        or getattr(factor, "calendar_id", None)
    )
    return {
        "factor_id": factor_id,
        "expression": expression or getattr(factor, "source_expr", None),
        "surface": getattr(factor, "surface", None),
        "dialect": getattr(factor, "dialect", None),
        "dialect_version": getattr(factor, "dialect_version", None),
        "market": market,
        "universe": (scope_u or None) or getattr(factor, "universe", None),
        "frequency": frequency or scope_f or getattr(factor, "freq", None),
        "data_source_config": data_source_config or {},
        "decision_policy": (scope_d or None) or getattr(factor, "decision_time_policy", None),
        "run_mode": getattr(engine, "run_mode", None),
        "calendar": calendar,
        "backend": _backend_kind(getattr(engine, "backend", None)),
        "pit_enforce": pit_enforce,
        "author": author,
        "description": description or getattr(factor, "description", None),
        "ast_hash": compute_ir_hash(analysis.ir),
    }


def _effective_data_source_config(
    data_source_config: dict | None,
    data_source: Any,
) -> dict | None:
    """#收官轮 P0：显式 ``data_source_config`` 优先，否则从 live source 还原。

    程序化 ``materialize()`` 未传 ``data_source_config`` 时，用
    ``data_source.execution_spec()`` 还原正在执行的真实 source contract——否则
    catalog / lineage / semantic identity 里只剩 ``{}``，事件增量 rebuild 无法
    还原完整因子。不可推导时返回 ``None``。
    """
    if data_source_config:
        return data_source_config
    spec_fn = getattr(data_source, "execution_spec", None)
    if not callable(spec_fn):
        return None
    try:
        spec = spec_fn()
    except UnreconstructableDataSource:
        # #收官轮 P1：production 下 Composite child 无法序列化是硬错误，必须
        # 向上抛（调用方 production fail-closed），不能被当作「不可用」吞掉。
        raise
    except Exception:  # pragma: no cover - 推导失败按不可用处理
        return None
    if isinstance(spec, dict) and spec:
        return dict(spec)
    return None


def _build_semantic_identity(
    *,
    engine: Any,
    factor: Factor,
    ir_node: Any,
    ast_hash: str,
    frequency: str | None,
    data_source_config: dict | None,
    run_lineage: dict | None,
    pit_enforce: bool | None,
    scope: Any = None,
) -> Any:
    """从 materialize 现场构建完整 ``FactorSemanticIdentity``（#收官轮 P0）。

    不能让 ``ParquetMaterializer`` 自己根据残缺 ctx 重建身份——那只有哈希、
    没有执行语义。同一公式不同 freq / universe / market / PIT / dialect 的因子
    必须得到不同的 ``factor_version``，production 才能识别语义漂移并拒绝沿用
    同 factor_id（``FactorCatalog.register`` 的
    ``FactorSemanticIdentityMismatchError``）。

    语义字段优先级：canonical ``FactorExecutionScope``（``_scope_from_factor``，
    优先 ``factor.semantic_identity``）> engine / data_source 显式执行语义 >
    factor 元数据 > 兜底 ``None``。无法获取真实语义时宁可显式 ``None`` 也不猜测。

    #收官轮 P0：market / calendar / universe / decision_time_policy / frequency
    必须与执行引擎 ``_scope_from_factor`` 消费同一个 scope——否则因子 A 的
    ``factor.freq=1d`` 但 ``semantic_identity.frequency=5m`` 时，执行按 5m 跑、
    身份却按 1d 落库，factor_version 与实际执行语义 split-brain。
    """
    if ir_node is None and not ast_hash:
        return None
    from runtime.factor_identity import compute_factor_identity

    ds = getattr(engine, "data_source", None)
    lineage_extra = (run_lineage or {}).get("extra") or {}
    pit_effective = bool(
        pit_enforce if pit_enforce is not None else getattr(ds, "pit_enforce", False)
    )
    scope_u = getattr(scope, "universe_id", None)
    scope_m = getattr(scope, "market", None)
    scope_c = getattr(scope, "calendar_id", None)
    scope_d = getattr(scope, "decision_time_policy", None)
    # R20-201..206: storage precision policy 从 lineage 侧接线进 identity ctx。
    # ``compute_factor_identity`` 当前忽略未知 ctx key；当 #183 把
    # ``storage_precision_policy`` 加进 ``FactorSemanticIdentity`` 后，它自动进入
    # identity digest / factor_version / checkpoint 失效判定。
    storage_precision_policy = (
        lineage_extra.get("storage_precision_policy")
        or lineage_extra.get("storage_value_dtype")
        or getattr(getattr(engine, "backend", None), "storage_precision_policy", None)
    )
    ctx: dict[str, Any] = {
        "ir_hash": ast_hash,
        "operator_contract_hash": (run_lineage or {}).get("operator_catalog_hash"),
        "field_contract_hash": (run_lineage or {}).get("field_catalog_hash"),
        "source_contract_hash": lineage_extra.get("source_contract_hash"),
        "source_dependency_hash": lineage_extra.get("source_dependency_hash"),
        "data_source_config": data_source_config,
        "factor": factor,
        "frequency": frequency,
        "storage_precision_policy": storage_precision_policy,
        "market": (
            (scope_m or None)
            or getattr(engine, "market", None)
            or getattr(ds, "market", None)
            or getattr(ds, "market_code", None)
        ),
        "calendar": (
            (scope_c or None)
            or getattr(ds, "calendar_id", None)
            or getattr(ds, "calendar", None)
        ),
        "timezone": getattr(ds, "timezone", None),
        "universe": (scope_u or None) or getattr(factor, "universe", None),
        "price_basis": getattr(ds, "price_basis", None),
        "pit_policy": "enforce" if pit_effective else None,
        "decision_time_policy": (scope_d or None) or getattr(ds, "decision_time_policy", None),
        "dialect": getattr(factor, "dialect", None),
        "dialect_version": getattr(factor, "dialect_version", None),
    }
    return compute_factor_identity(ir_node, ctx=ctx)


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
    deleted_keys: list[tuple] | None = None,
    pit_enforce: bool | None = None,
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
    # #收官轮 P0：程序化 materialize() 未显式传 data_source_config 时，从 live
    # source 的 ``execution_spec()`` 还原完整执行规格，供 lineage / semantic
    # identity / full definition / rebuild 复用（否则 catalog 只存 ``{}``）。
    effective_data_source_config = _effective_data_source_config(
        data_source_config, engine.data_source
    )
    lineage = lineage_service.build_materialize_lineage(
        factor=factor,
        analysis=analysis,
        output=output,
        factor_id=factor_id,
        expression=expression,
        data_source_config=effective_data_source_config,
        data_source=engine.data_source,
        mode=lineage_mode,
    )
    materializer = ParquetMaterializer(
        lake_root=lake_root,
        staging_dataset=staging_dataset,
    )
    ast_hash = compute_ir_hash(analysis.ir)
    parquet_target = resolve_parquet_write_target(target)

    # #收官轮 P0：production 由 orchestrator 从 ``engine.run_mode`` 一锤定音，
    # 显式传给物化器——下游（``_resolve_production``）禁止再重新猜运行模式
    # （否则 engine=production + env=research 时 direct-local 写守卫、semantic
    # identity mismatch 门、incremental tombstone 策略全部失效）。
    production = is_production_mode(engine.run_mode)
    # R20-201..206: storage precision policy 在 orchestrator 层解析一次，写进
    # lineage extra，让 identity ctx / catalog / 事件 rebuild 消费同一份精度契约。
    # 显式 ``value_dtype=None`` 时 production 默认 float64；float32 需 quantization
    # certificate（否则记录 ``float32_legacy`` 诚实标记）。
    from storage.materialize.materializer import storage_precision_policy_for

    _eff_dtype, _precision_policy = storage_precision_policy_for(
        value_dtype,
        production=production,
        lineage_extra=dict(lineage.extra),
    )
    lineage.extra["storage_precision_policy"] = _precision_policy
    lineage.extra["storage_value_dtype"] = _eff_dtype
    # #收官轮 P0：完整语义身份（含 frequency/market/universe/pit/dialect）在
    # orchestrator 层构建后显式传入——物化器不再根据残缺 ctx 重建（那样 freq
    # 1d 与 5m 会同 digest / 同 factor_version）。
    # #收官轮 P0：执行作用域只算一次，且必须与执行引擎共用同一个
    # ``_scope_from_factor``（优先 ``factor.semantic_identity``）——cache scope、
    # factor_version、full definition、事件 rebuild 全部消费它，杜绝
    # 「执行 5m / 落库 1d」的语义分裂。
    from runtime.engine import _scope_from_factor

    scope = _scope_from_factor(factor, data_source=engine.data_source)
    frequency_effective = frequency or getattr(scope, "frequency", None) or factor.freq
    semantic_identity = _build_semantic_identity(
        engine=engine,
        factor=factor,
        ir_node=analysis.ir,
        ast_hash=ast_hash,
        frequency=frequency_effective,
        data_source_config=effective_data_source_config,
        run_lineage={**lineage.to_dict(), "factor_id": factor_id or factor.name},
        pit_enforce=pit_enforce,
        scope=scope,
    )

    summary = materializer.materialize(
        factor_id=factor_id or factor.name,
        result=output["result"],
        ir_node=analysis.ir,
        author=author,
        frequency=frequency_effective,
        description=description or factor.description,
        expression=expression,
        dq_check=dq_check,
        dq_strict=dq_strict,
        dq_thresholds=dq_thresholds,
        run_lineage={**lineage.to_dict(), "factor_id": factor_id or factor.name},
        write_metadata=write_metadata,
        data_snapshot_id=lineage_service.resolve_data_snapshot_id(
            engine.data_source, effective_data_source_config
        ),
        data_source_config=effective_data_source_config,
        resume=resume_materialize,
        isolate_partition_failures=isolate_partition_failures,
        preserve_invalid_rows=preserve_invalid_rows,
        value_dtype=value_dtype,
        write_target=parquet_target,
        defer_watermark=needs_clickhouse_write(target),
        storage_format=storage_format,
        partition_columns=partition_columns,
        deleted_keys=deleted_keys,
        production=production,
        semantic_identity=semantic_identity,
    )

    from runtime.incremental_scheduler import record_factor_dependency_from_analysis

    try:
        record_factor_dependency_from_analysis(
            materializer.catalog,
            factor_id=factor_id or factor.name,
            analysis=analysis,
            data_source=engine.data_source,
            frequency=frequency_effective,
            production=production,
            full_definition=_build_full_factor_definition(
                engine=engine,
                factor=factor,
                factor_id=factor_id or factor.name,
                author=author,
                frequency=frequency_effective,
                description=description,
                expression=expression,
                data_source_config=effective_data_source_config,
                analysis=analysis,
                pit_enforce=pit_enforce,
                scope=scope,
            ),
        )
    except Exception as exc:
        # #收官轮 P0：factor 数据已落盘 + 水位线已推进，但依赖边 / full
        # definition 没写进 catalog —— 调用方拿到的不能再是「完整成功」。否则
        # DataEvent → dependency lookup → reconstruct 完整因子链路直接断裂，
        # 增量重算再也找不到这个因子。production fail-closed：抛 IN_DOUBT
        # 异常并携带已落盘的 summary 供对账；research 保留历史 warning 行为。
        if production:
            raise MaterializedButCatalogCommitFailed(
                f"因子 {factor.name} 数据已落盘但依赖/full-definition catalog "
                f"写入失败（IN_DOUBT，需对账：重试 catalog 写入或标记 "
                f"NEED_RECONCILE，不能视为完整 PUBLISHED success）: {exc}",
                materialization=summary,
            ) from exc
        logger.warning("因子依赖 catalog 写入失败 factor=%s: %s", factor.name, exc)

    if needs_clickhouse_write(target):
        # #收官轮 P0：canonical factor_version 只算一次，Parquet/catalog/ClickHouse
        # 用同一份（identity digest 前缀）。data_snapshot_id 也改用
        # ``effective_data_source_config``（与 Parquet/lineage 同源），不再用调用方
        # 原始 ``data_source_config``（可能为空 → snapshot id 两侧不一致）。
        canonical_factor_version = (
            semantic_identity.identity_digest()[:16]
            if semantic_identity is not None
            else ast_hash[:16]
        )
        # R14 #3：**一个** ``MaterializationDelta`` 是全部 sink 的唯一构造点。
        # ``deleted_keys``（上游 ``materialize_incremental`` 传入的 tombstone 键）
        # 必须同时到达 Parquet 与 ClickHouse——Parquet 已把它写为 NaN/deleted，
        # 这里把同一份 tombstone 传进 CH（``_series_with_tombstones`` 转成 NaN 行
        # 覆盖旧有限值，与 Parquet 同构）。全量 digest / generation /
        # transaction_id 也由 delta 统一携带，杜绝各 sink 各自重新拼状态。
        snapshot_id = lineage_service.resolve_data_snapshot_id(
            engine.data_source, effective_data_source_config
        )
        from runtime.dual_write_service import MaterializationDelta

        delta = MaterializationDelta(
            upserts=output["result"],
            tombstones=tuple(deleted_keys or ()),
            semantic_identity_digest=(
                semantic_identity.identity_digest()
                if semantic_identity is not None
                else None
            ),
            data_snapshot_id=snapshot_id,
        )
        summary = dual_write_service.dual_write_clickhouse(
            materializer,
            summary,
            factor_id=factor_id or factor.name,
            result=output["result"],
            ast_hash=ast_hash,
            factor_version=canonical_factor_version,
            write_target=target,
            data_snapshot_id=snapshot_id,
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
            delta=delta,
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
        deleted_keys=getattr(opts, "deleted_keys", None),
        pit_enforce=getattr(opts, "pit_enforce", None),
    )


def can_batch_materialize_compute(opts: Any) -> bool:
    """是否可对多因子共享一次 ``run_many`` 计算（非增量、非 resume 分区）。

    增量物化或断点续写分区需逐因子独立执行窗口。
    """
    return not opts.resume_materialize and opts.since is None

