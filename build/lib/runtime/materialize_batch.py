# -*- coding: utf-8 -*-
"""R39-PERF-039: 真实批量物化编排 ``execute_materialize_batch``。

把 ``materialize_many_fast`` writer 里“batch 内仍逐 item 完整
``execute_materialize``”的假 batch 改成：**共享工作 hoist 出 per-item 循环**，
依赖 manifest 目录提交收敛到**单个事务**。

诚实范围（对照 R39 §11/§20）：
    - hoisted（一次，全 batch 共享）：
        * ``ParquetMaterializer`` 实例；
        * ``write_target`` 规范化 / production 判定 / ``data_source_config`` 还原 /
          ``data_snapshot_id`` 解析 / ``run_generation``；
        * 依赖 manifest 的**批量目录提交**（``record_factor_manifests_many``：
          N 次 BEGIN/COMMIT → 1 次）。
    - per-item（保留原语义，storage 层仍逐 partition 写）：
        * lineage / ast_hash / execution scope / semantic identity / precision
          policy（本质依赖单因子 analysis，无法向量化）；
        * ``materializer.materialize(...)`` 的物理分区写 + 内部 ``catalog.register``
          + partition checkpoint（storage 格式不在本 cluster 范围——由
          delta-storage 专项处理 R39-P0-PERF-043..058）；
        * ClickHouse 双写（per factor 物理表）。

因此 ``batch_write_transaction_count``（本 API 在 orchestrator 层发起的 catalog
事务数）对共享 generation 的 batch 为 O(1)，严格 ``<< factor_count``（Gate-03）；
而 ``physical_partition_write_rounds`` 仍 == 逐因子分区写（storage 层，已如实
暴露计数，不在本层改写）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from api.factor import Factor
from logging_utils import get_logger

logger = get_logger("runtime.materialize_batch")


@dataclass
class MaterializeItem:
    """单因子批量物化项。

    ``options`` 为 ``execute_materialize`` 同款物化参数（缺失时继承 batch 级
    共享默认，来自第一个 item 或 ``execute_materialize_batch`` 显式默认）。
    """

    factor: Factor
    output: dict[str, Any]  # 须含 ``analysis`` 与 ``result``
    factor_id: str | None = None
    options: dict[str, Any] | None = None


@dataclass
class GenerationTransaction:
    """批量 generation 记录：共享一个 generation_id，完成后原子 publish。"""

    generation_id: str
    started: float = field(default_factory=time.monotonic)
    published_items: int = 0

    def publish(self, count: int = 1) -> None:
        self.published_items += int(count)


@dataclass
class BatchMaterializeCounters:
    """R39 Gate-03 计数：batch 层 catalog/write 事务数 vs 逐因子数。"""

    item_count: int = 0
    #: orchestrator 层发起的 catalog 事务数（依赖 manifest 批量提交=1）。
    batch_write_transaction_count: int = 0
    #: 依赖 manifest 实际写入条数（== item_count）。
    manifest_writes: int = 0
    #: storage 层物理分区写轮数（=Σ 每因子 partitions，storage 范围，如实暴露）。
    physical_partition_write_rounds: int = 0
    #: storage 层 catalog.register 调用次数（== item_count，storage 范围）。
    catalog_register_calls: int = 0
    #: ClickHouse 双写次数（== 需要 CH 的 item 数）。
    clickhouse_writes: int = 0
    #: hoisted 共享上下文次数（materializer / target / production 等）。
    hoisted_context: int = 0
    skipped_empty: int = 0
    #: R39-PERF-030: batch 内多个 item 共享同一 axis（MultiIndex）的**不同**共享
    #: axis 数（同一 index 只记一次；逐因子 parquet 写路径不变）。
    factor_block_shared_axis_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "batch_write_transaction_count": self.batch_write_transaction_count,
            "manifest_writes": self.manifest_writes,
            "physical_partition_write_rounds": self.physical_partition_write_rounds,
            "catalog_register_calls": self.catalog_register_calls,
            "clickhouse_writes": self.clickhouse_writes,
            "hoisted_context": self.hoisted_context,
            "skipped_empty": self.skipped_empty,
            "factor_block_shared_axis_count": self.factor_block_shared_axis_count,
        }


#: ``execute_materialize`` 物化参数缺省值（与 materialize_service 对齐）。
_DEFAULT_OPTS: dict[str, Any] = {
    "target": "local",
    "lake_root": None,
    "staging_dataset": "factor_lake_staging",
    "author": None,
    "frequency": None,
    "description": None,
    "expression": None,
    "dq_check": False,
    "dq_strict": True,
    "dq_thresholds": None,
    "write_metadata": True,
    "data_source_config": None,
    "resume_materialize": False,
    "isolate_partition_failures": True,
    "preserve_invalid_rows": False,
    "value_dtype": "float32",
    "clickhouse_table": None,
    "ch_ensure_table": True,
    "ch_host": None,
    "ch_port": None,
    "ch_database": None,
    "ch_username": None,
    "ch_password": None,
    "ch_secure": None,
    "storage_format": "long",
    "partition_columns": None,
    "deleted_keys": None,
    "pit_enforce": None,
}


def _resolve_item_options(
    item: MaterializeItem, shared: dict[str, Any]
) -> dict[str, Any]:
    opts = dict(shared)
    if item.options:
        opts.update(item.options)
    return opts


def _prepare_one(
    engine: Any,
    item: MaterializeItem,
    opts: dict[str, Any],
    *,
    effective_data_source_config: dict | None,
    production: bool,
    run_generation: str,
) -> dict[str, Any]:
    """per-item 物化前准备：lineage / ast_hash / scope / identity / precision。

    返回 dict 供 ``_write_one`` 与 manifest 批量提交共用（避免两处重建）。
    """
    from runtime import lineage_service
    from runtime.engine import _scope_from_factor
    from storage.catalog import compute_ir_hash

    analysis = item.output["analysis"]
    factor = item.factor
    factor_id = item.factor_id or factor.name
    lineage = lineage_service.build_materialize_lineage(
        factor=factor,
        analysis=analysis,
        output=item.output,
        factor_id=factor_id,
        expression=opts.get("expression"),
        data_source_config=effective_data_source_config,
        data_source=engine.data_source,
        mode="full",
    )
    ast_hash = compute_ir_hash(analysis.ir)
    scope = _scope_from_factor(factor, data_source=engine.data_source)
    frequency_effective = (
        opts.get("frequency") or getattr(scope, "frequency", None) or factor.freq
    )
    from storage.materialize.materializer import storage_precision_policy_for

    _eff_dtype, _precision_policy = storage_precision_policy_for(
        opts.get("value_dtype"),
        production=production,
        lineage_extra=dict(lineage.extra),
    )
    lineage.extra["storage_precision_policy"] = _precision_policy
    lineage.extra["storage_value_dtype"] = _eff_dtype
    from runtime.materialize_service import _build_semantic_identity

    semantic_identity = _build_semantic_identity(
        engine=engine,
        factor=factor,
        ir_node=analysis.ir,
        ast_hash=ast_hash,
        frequency=frequency_effective,
        data_source_config=effective_data_source_config,
        run_lineage={**lineage.to_dict(), "factor_id": factor_id},
        pit_enforce=opts.get("pit_enforce"),
        scope=scope,
    )
    return {
        "factor_id": factor_id,
        "factor": factor,
        "output": item.output,
        "analysis": analysis,
        "ast_hash": ast_hash,
        "scope": scope,
        "frequency_effective": frequency_effective,
        "lineage": lineage,
        "semantic_identity": semantic_identity,
        "opts": opts,
        "precision_policy": _precision_policy,
    }


def _write_one(
    engine: Any,
    materializer: Any,
    prep: dict[str, Any],
    *,
    parquet_target: str,
    production: bool,
    data_snapshot_id: str | None,
    run_generation: str,
    effective_data_source_config: dict | None,
    counters: BatchMaterializeCounters,
) -> dict[str, Any]:
    """per-item 物理写（storage writer）+ 可选 ClickHouse 双写。

    与 ``execute_materialize`` 的 materialize() 参数一一对应，保证 value /
    index / dtype / PIT / factor_version / lineage 语义完全一致。
    """
    opts = prep["opts"]
    output = prep["output"]
    factor_id = prep["factor_id"]
    need_ch = _needs_clickhouse_write(parquet_target, opts)
    summary = materializer.materialize(
        factor_id=factor_id,
        result=output["result"],
        ir_node=prep["analysis"].ir,
        author=opts.get("author"),
        frequency=prep["frequency_effective"],
        description=opts.get("description") or prep["factor"].description,
        expression=opts.get("expression"),
        dq_check=opts.get("dq_check", False),
        dq_strict=opts.get("dq_strict", True),
        dq_thresholds=opts.get("dq_thresholds"),
        run_lineage={**prep["lineage"].to_dict(), "factor_id": factor_id},
        write_metadata=opts.get("write_metadata", True),
        data_snapshot_id=data_snapshot_id,
        data_source_config=effective_data_source_config,
        resume=opts.get("resume_materialize", False),
        isolate_partition_failures=opts.get("isolate_partition_failures", True),
        preserve_invalid_rows=opts.get("preserve_invalid_rows", False),
        value_dtype=opts.get("value_dtype"),
        write_target=parquet_target,
        defer_watermark=need_ch,
        storage_format=str(opts.get("storage_format", "long")),
        partition_columns=opts.get("partition_columns"),
        deleted_keys=opts.get("deleted_keys"),
        production=production,
        run_generation=run_generation,
        semantic_identity=prep["semantic_identity"],
    )
    counters.catalog_register_calls += 1
    counters.physical_partition_write_rounds += len(summary.get("partitions") or [])

    if need_ch:
        from runtime.dual_write_service import (
            MaterializationDelta,
            dual_write_clickhouse,
        )
        from runtime import lineage_service

        semantic_identity = prep["semantic_identity"]
        canonical_factor_version = (
            semantic_identity.identity_digest()[:16]
            if semantic_identity is not None
            else prep["ast_hash"][:16]
        )
        snapshot_id = lineage_service.resolve_data_snapshot_id(
            engine.data_source, effective_data_source_config
        )
        delta = MaterializationDelta(
            upserts=output["result"],
            tombstones=tuple(opts.get("deleted_keys") or ()),
            semantic_identity_digest=(
                semantic_identity.identity_digest()
                if semantic_identity is not None
                else None
            ),
            data_snapshot_id=snapshot_id,
        )
        summary = dual_write_clickhouse(
            materializer,
            summary,
            factor_id=factor_id,
            result=output["result"],
            ast_hash=prep["ast_hash"],
            factor_version=canonical_factor_version,
            write_target=str(opts.get("target", "local")),
            data_snapshot_id=snapshot_id,
            clickhouse_table=opts.get("clickhouse_table"),
            preserve_invalid_rows=opts.get("preserve_invalid_rows", False),
            value_dtype=opts.get("value_dtype"),
            write_metadata=opts.get("write_metadata", True),
            ensure_table=opts.get("ch_ensure_table", True),
            dq_check=opts.get("dq_check", False),
            dq_strict=opts.get("dq_strict", True),
            dq_thresholds=opts.get("dq_thresholds"),
            ch_host=opts.get("ch_host"),
            ch_port=opts.get("ch_port"),
            ch_database=opts.get("ch_database"),
            ch_username=opts.get("ch_username"),
            ch_password=opts.get("ch_password"),
            ch_secure=opts.get("ch_secure"),
            delta=delta,
        )
        counters.clickhouse_writes += 1

    return {
        **summary,
        "lake_root": str(materializer.lake_root),
        "factor_id": factor_id,
        "factor_name": prep["factor"].name,
    }


def _needs_clickhouse_write(parquet_target: str, opts: dict[str, Any]) -> bool:
    """与 ``materialize_service.needs_clickhouse_write`` 对齐。"""
    target = str(opts.get("target") or parquet_target or "local").lower()
    return target in ("clickhouse", "staging_clickhouse")


def _build_manifest_payloads(
    engine: Any,
    preps: list[dict[str, Any]],
    *,
    production: bool,
    data_source: Any,
) -> list[dict[str, Any]]:
    """per-item 依赖 manifest 数据（mirror ``record_factor_dependency_from_analysis``）。

    build 仍 per-item（依赖单因子 analysis），但 DB 写入由
    ``record_factor_manifests_many`` 收敛到单个事务。
    """
    from runtime.dependency_catalog import DependencyCatalog  # noqa: F401
    from runtime.incremental_scheduler import (
        _calendar_version,
        _edges_from_analysis,
        _operator_manifest_from_ir,
        _resolve_source_dataset,
    )
    from runtime.materialize_service import _build_full_factor_definition

    payloads: list[dict[str, Any]] = []
    for prep in preps:
        factor_id = prep["factor_id"]
        analysis = prep["analysis"]
        referenced_columns = getattr(analysis, "referenced_columns", set()) or set()
        lookback = int(getattr(analysis, "lookback", 0))
        source_dataset = _resolve_source_dataset(data_source)
        edges = _edges_from_analysis(
            factor_id, analysis, data_source, production=production
        )
        manifest = _operator_manifest_from_ir(
            getattr(analysis, "ir", None), production=production
        )
        cal_version = _calendar_version(data_source, production=production)
        full_def = _build_full_factor_definition(
            engine=engine,
            factor=prep["factor"],
            factor_id=factor_id,
            author=prep["opts"].get("author"),
            frequency=prep["frequency_effective"],
            description=prep["opts"].get("description"),
            expression=prep["opts"].get("expression"),
            data_source_config=prep["opts"].get("data_source_config")
            or prep.get("effective_data_source_config"),
            analysis=analysis,
            pit_enforce=prep["opts"].get("pit_enforce"),
            scope=prep["scope"],
        )
        if manifest:
            full_def["operator_manifest"] = manifest
        if cal_version:
            full_def["calendar_version"] = cal_version
        payloads.append(
            {
                "factor_id": factor_id,
                "edges": edges,
                "referenced_columns": referenced_columns,
                "lookback": lookback,
                "frequency": prep["frequency_effective"],
                "source_dataset": source_dataset,
                "full_definition": full_def or None,
            }
        )
    return payloads


def _record_shared_axes(preps: list[dict[str, Any]]) -> int:
    """R39-PERF-030: batch 内同 axis 共享观测。

    若 batch 内多个 item 的 ``result`` 共享同一 MultiIndex（同 axis），对每个
    **不同**的共享 axis 构造一个 ``FactorBlockRef`` 并返回共享 axis 数（同一 index
    不重复计数）。逐因子 parquet 写路径保持原样——本观测/载体不改变写入文件。
    """
    from runtime.factor_block_ref import build_factor_block, group_by_shared_axis

    series_by_fid: dict[str, pd.Series] = {}
    for prep in preps:
        result = prep["output"].get("result")
        if isinstance(result, pd.Series):
            series_by_fid[prep["factor_id"]] = result
    if len(series_by_fid) < 2:
        return 0
    dtype = str(preps[0].get("opts", {}).get("value_dtype") or "float32")
    groups = group_by_shared_axis(series_by_fid)
    for axis_ref, fids in groups:
        build_factor_block(
            fids,
            {fid: series_by_fid[fid] for fid in fids},
            index=axis_ref.index,
            dtype=dtype,
        )
    return len(groups)


def execute_materialize_batch(
    engine: Any,
    items: Iterable[MaterializeItem],
    generation: GenerationTransaction | None = None,
    *,
    shared_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """批量物化：共享工作 hoist，依赖 manifest 单事务，逐因子物理写保留。

    Args:
        engine: 因子引擎实例。
        items: ``MaterializeItem`` 列表（共享同一 execution 上下文）。
        generation: 可选的 ``GenerationTransaction``——传入时所有 item 共享
            ``generation.generation_id`` 作为 ``run_generation``，完成后
            ``generation.publish(item_count)``。
        shared_options: batch 级共享物化参数（缺省用第一个 item 的 options 或
            ``_DEFAULT_OPTS``）；per-item ``MaterializeItem.options`` 覆盖。

    Returns:
        ``{"materializations": {factor_name: summary}, "per_item": [...],
          "counters": {...}, "generation": generation_id}``。
    """
    from runtime import lineage_service
    from runtime.materialize_service import (
        _effective_data_source_config,
        resolve_parquet_write_target,
    )
    from runtime.production_policy import is_production_mode
    from storage.materializer import ParquetMaterializer

    items = list(items)
    counters = BatchMaterializeCounters(item_count=len(items))
    if not items:
        return {
            "materializations": {},
            "per_item": [],
            "counters": counters.to_dict(),
            "generation": None,
        }

    # --- hoisted shared context（一次） ---
    shared = dict(_DEFAULT_OPTS)
    if shared_options:
        shared.update(shared_options)
    if items[0].options:
        first = dict(items[0].options)
        for k, v in first.items():
            shared.setdefault(k, v)
    # 调用方只给 ``write_target`` 未给 ``target`` 时（如 ``execute_materialize``
    # 语义），让 ``target`` 跟随 ``write_target``（缺省 target="local" 不算显式）。
    if "target" not in (shared_options or {}) and not any(
        "target" in (i.options or {}) for i in items
    ):
        _wt = shared.get("write_target")
        if _wt:
            shared["target"] = str(_wt)
    target = str(shared.get("target") or "local")
    lake_root = shared.get("lake_root")
    staging_dataset = shared.get("staging_dataset") or "factor_lake_staging"
    production = is_production_mode(engine.run_mode)
    parquet_target = resolve_parquet_write_target(target)
    effective_data_source_config = _effective_data_source_config(
        shared.get("data_source_config"), engine.data_source
    )
    data_snapshot_id = lineage_service.resolve_data_snapshot_id(
        engine.data_source, effective_data_source_config
    )
    materializer = ParquetMaterializer(
        lake_root=lake_root, staging_dataset=staging_dataset
    )
    if generation is not None:
        generation_id = generation.generation_id
    else:
        from runtime.lineage import new_run_id

        generation_id = f"batch-{new_run_id()}"
    run_generation = generation_id
    counters.hoisted_context = 1

    # --- per-item prepare（lineage/identity/scope 依赖单因子，无法向量化） ---
    preps: list[dict[str, Any]] = []
    for item in items:
        opts = _resolve_item_options(item, shared)
        prep = _prepare_one(
            engine,
            item,
            opts,
            effective_data_source_config=effective_data_source_config,
            production=production,
            run_generation=run_generation,
        )
        prep["effective_data_source_config"] = effective_data_source_config
        preps.append(prep)

    # --- per-item 物理写（storage writer 逐 partition；语义与 execute_materialize 一致）。
    # 单因子写失败隔离（isolate_partition_failures 语义）：记录 error，不中断 batch，
    # 失败的因子不进入 manifest 批量提交。
    summaries: dict[str, dict[str, Any]] = {}
    per_item: list[dict[str, Any]] = []
    written_preps: list[dict[str, Any]] = []
    for prep in preps:
        try:
            summary = _write_one(
                engine,
                materializer,
                prep,
                parquet_target=parquet_target,
                production=production,
                data_snapshot_id=data_snapshot_id,
                run_generation=run_generation,
                effective_data_source_config=effective_data_source_config,
                counters=counters,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "batch 单因子写失败 factor=%s: %s", prep["factor"].name, exc
            )
            err = {"error": f"{type(exc).__name__}: {exc}"}
            summaries[prep["factor"].name] = err
            per_item.append(err)
            continue
        if int(summary.get("rows_written") or 0) == 0:
            counters.skipped_empty += 1
        summaries[prep["factor"].name] = summary
        per_item.append(summary)
        written_preps.append(prep)

    # --- R39-PERF-030: 同 axis 共享观测（不改变逐因子写入；同 index 不重复） ---
    counters.factor_block_shared_axis_count = _record_shared_axes(written_preps)

    # --- batched catalog commit：依赖 manifest 单事务（Gate-03 核心） ---
    if written_preps:
        payloads = _build_manifest_payloads(
            engine, written_preps, production=production, data_source=engine.data_source
        )
        from runtime.dependency_catalog import DependencyCatalog

        dep_catalog = DependencyCatalog(materializer.catalog)
        try:
            written = dep_catalog.record_factor_manifests_many(payloads)
            counters.batch_write_transaction_count += 1
            counters.manifest_writes += written
        except Exception as exc:  # noqa: BLE001
            # 与 execute_materialize 的 catalog 提交失败语义一致：production 下
            # 数据已落盘但目录未提交 → IN_DOUBT（fail-closed）；research 保留 warning。
            if production:
                from storage.exceptions import MaterializedButCatalogCommitFailed

                raise MaterializedButCatalogCommitFailed(
                    f"batch materialize: 因子数据已落盘但依赖/full-definition catalog "
                    f"批量提交失败（IN_DOUBT，需对账）: {exc}",
                    materialization=summaries,
                ) from exc
            logger.warning(
                "batch 依赖 manifest 目录提交失败（research，已记录待对账）: %s", exc
            )
            for name, s in summaries.items():
                if "error" not in s:
                    s["catalog_commit_warning"] = str(exc)

    # --- 单 generation publish ---
    if generation is not None:
        generation.publish(len(items))

    logger.info(
        "execute_materialize_batch items=%d batch_write_transactions=%d "
        "manifest_writes=%d partition_rounds=%d",
        len(items),
        counters.batch_write_transaction_count,
        counters.manifest_writes,
        counters.physical_partition_write_rounds,
    )
    return {
        "materializations": summaries,
        "per_item": per_item,
        "counters": counters.to_dict(),
        "generation": generation_id,
    }
