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
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from factor_engine.api.factor import Factor
from factor_engine.util.logging_utils import get_logger

logger = get_logger("factor_engine.runtime.materialize_batch")


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


class WriteState(str, Enum):
    CREATED = "CREATED"
    STAGED = "STAGED"
    COMMITTED = "COMMITTED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    IN_DOUBT = "IN_DOUBT"


@dataclass(frozen=True)
class WriteContext:
    target: str
    lake_root: str | None
    staging_dataset: str
    data_source_config: Any
    storage_format: str
    value_dtype: str
    atomicity: str


@dataclass
class WriteItemReceipt:
    name: str
    state: WriteState = WriteState.CREATED
    rows: int = 0
    error: str | None = None
    run_id: str | None = None
    inventory_digest: str | None = None
    inventory: list[dict[str, Any]] = field(default_factory=list)
    coverage_proof: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "state": self.state.value, "rows": self.rows,
            "error": self.error, "run_id": self.run_id,
            "inventory_digest": self.inventory_digest,
            "inventory": self.inventory,
            "coverage_proof": self.coverage_proof,
        }


@dataclass
class WriteReceipt:
    generation_id: str | None
    expected_items: tuple[str, ...]
    items: dict[str, WriteItemReceipt]
    manifest_digest: str | None = None
    idempotency_key: str | None = None

    @property
    def state(self) -> WriteState:
        states = {item.state for item in self.items.values()}
        if WriteState.IN_DOUBT in states:
            return WriteState.IN_DOUBT
        if WriteState.FAILED in states:
            return WriteState.FAILED
        if states == {WriteState.PUBLISHED}:
            return WriteState.PUBLISHED
        if states == {WriteState.COMMITTED}:
            return WriteState.COMMITTED
        if states == {WriteState.STAGED}:
            return WriteState.STAGED
        return WriteState.CREATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "expected_items": list(self.expected_items),
            "state": self.state.value,
            "manifest_digest": self.manifest_digest,
            "idempotency_key": self.idempotency_key,
            "items": {name: item.to_dict() for name, item in self.items.items()},
        }

    def validate(self, expected_items: Iterable[str] | None = None) -> "WriteReceipt":
        requested = self.expected_items if expected_items is None else expected_items
        if isinstance(requested, (str, bytes)) or not isinstance(requested, (list, tuple)):
            raise ValueError("WriteReceipt expected_items must be a list/tuple of strings")
        if not all(isinstance(name, str) and name for name in requested):
            raise ValueError("WriteReceipt expected_items must contain non-empty strings")
        expected = tuple(requested)
        if len(set(expected)) != len(expected):
            raise ValueError("WriteReceipt expected_items contains duplicates")
        if tuple(self.expected_items) != expected:
            raise ValueError("WriteReceipt expected_items does not match request")
        if set(self.items) != set(expected):
            raise ValueError("WriteReceipt item keys do not exactly match expected_items")
        if expected and (
            not isinstance(self.generation_id, str) or not self.generation_id
            or not isinstance(self.idempotency_key, str) or not self.idempotency_key
        ):
            raise ValueError("non-empty WriteReceipt requires generation and idempotency key")
        if self.manifest_digest is not None and (
            not isinstance(self.manifest_digest, str) or not self.manifest_digest
        ):
            raise ValueError("WriteReceipt manifest_digest must be a non-empty string")
        for key, item in self.items.items():
            if item.name != key:
                raise ValueError("WriteReceipt item name/key mismatch")
            if not isinstance(key, str) or not key:
                raise ValueError("WriteReceipt item keys must be non-empty strings")
            if not isinstance(item.state, WriteState):
                raise ValueError(f"invalid WriteReceipt state for {key!r}")
            if isinstance(item.rows, bool) or not isinstance(item.rows, int) or item.rows < 0:
                raise ValueError(f"invalid WriteReceipt rows for {key!r}")
            if item.state in {WriteState.COMMITTED, WriteState.PUBLISHED} and not self.manifest_digest:
                raise ValueError("committed/published receipt requires manifest_digest")
            if item.state in {WriteState.COMMITTED, WriteState.PUBLISHED}:
                if not item.run_id or not item.inventory_digest or not item.inventory:
                    raise ValueError(
                        "committed/published item requires run_id and byte inventory"
                    )
                if not isinstance(item.run_id, str) or not isinstance(item.inventory_digest, str):
                    raise ValueError("committed/published item evidence must be strings")
                if not isinstance(item.inventory, list) or not all(
                    isinstance(entry, dict) for entry in item.inventory
                ):
                    raise ValueError("committed/published item inventory must be a list of mappings")
                required_inventory = {"path", "rows", "bytes", "sha256"}
                for entry in item.inventory:
                    if set(entry) != required_inventory:
                        raise ValueError("inventory entries require path/rows/bytes/sha256 only")
                    if (
                        not isinstance(entry["path"], str) or not entry["path"]
                        or Path(entry["path"]).is_absolute()
                        or ".." in Path(entry["path"]).parts
                    ):
                        raise ValueError("inventory path must be a confined relative path")
                    for field_name in ("rows", "bytes"):
                        value = entry[field_name]
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError(f"inventory {field_name} must be a nonnegative integer")
                    if (
                        not isinstance(entry["sha256"], str)
                        or len(entry["sha256"]) != 64
                        or any(ch not in "0123456789abcdef" for ch in entry["sha256"])
                    ):
                        raise ValueError("inventory sha256 must be lowercase hexadecimal")
                calculated_digest = hashlib.sha256(
                    json.dumps(
                        item.inventory, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                if item.inventory_digest != calculated_digest:
                    raise ValueError("inventory_digest does not match inventory entries")
            if item.coverage_proof is not None and not isinstance(item.coverage_proof, dict):
                raise ValueError("item coverage_proof must be a mapping")
        return self

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, expected_items: Iterable[str] | None = None
    ) -> "WriteReceipt":
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
            raise ValueError("invalid WriteReceipt payload")
        raw_expected = payload.get("expected_items")
        if not isinstance(raw_expected, (list, tuple)) or not all(
            isinstance(name, str) and name for name in raw_expected
        ):
            raise ValueError("invalid WriteReceipt expected_items")
        items: dict[str, WriteItemReceipt] = {}
        for key, raw in payload["items"].items():
            if not isinstance(key, str) or not key or not isinstance(raw, dict):
                raise ValueError(f"invalid WriteReceipt item {key!r}")
            try:
                state = WriteState(raw.get("state"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid WriteReceipt state for {key!r}") from exc
            rows = raw.get("rows", 0)
            if not isinstance(raw.get("name"), str) or not raw["name"]:
                raise ValueError(f"invalid WriteReceipt item name for {key!r}")
            raw_inventory = raw.get("inventory", [])
            if not isinstance(raw_inventory, list):
                raise ValueError(f"invalid WriteReceipt inventory for {key!r}")
            items[key] = WriteItemReceipt(
                name=raw["name"],
                state=state,
                rows=rows,
                error=None if raw.get("error") is None else str(raw["error"]),
                run_id=raw.get("run_id"),
                inventory_digest=(
                    raw.get("inventory_digest")
                ),
                inventory=list(raw_inventory),
                coverage_proof=raw.get("coverage_proof"),
            )
        receipt = cls(
            generation_id=payload.get("generation_id"),
            expected_items=tuple(raw_expected),
            items=items,
            manifest_digest=payload.get("manifest_digest"),
            idempotency_key=payload.get("idempotency_key"),
        )
        return receipt.validate(expected_items)


@dataclass
class GenerationTransaction:
    """In-process batch generation counter; carries no persistent publish authority.

    ``publish()`` is a compatibility name that records storage-confirmed commit
    counts only. Durable PUBLISHED state is established by the explicit lake
    publication and certified-watermark path.
    """

    generation_id: str
    started: float = field(default_factory=time.monotonic)
    published_items: int = 0
    committed_items: int = 0

    def publish(self, count: int = 1) -> None:
        # Compatibility method: this in-process object has no authority to
        # assert PUBLISHED.  Record only storage-confirmed commits.
        self.committed_items += int(count)


def _canonical_context_value(value: Any) -> Any:
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("WriteContext contains non-finite float")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, dict):
        encoded = [
            (_canonical_context_value(k), _canonical_context_value(v))
            for k, v in value.items()
        ]
        return ("dict", tuple(sorted(encoded, key=repr)))
    if isinstance(value, (list, tuple)):
        return (
            "list" if isinstance(value, list) else "tuple",
            tuple(_canonical_context_value(v) for v in value),
        )
    if isinstance(value, (set, frozenset)):
        encoded = [_canonical_context_value(v) for v in value]
        return (
            "set" if isinstance(value, set) else "frozenset",
            tuple(sorted(encoded, key=repr)),
        )
    raise ValueError(
        f"WriteContext contains unsupported mutable/opaque value {type(value).__name__}"
    )


def _resolve_batch_contexts(
    items: list[MaterializeItem], shared_options: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]], WriteContext]:
    """Resolve and validate every physical write context before side effects."""
    shared = dict(_DEFAULT_OPTS)
    if items and items[0].options:
        shared.update(items[0].options)
    if shared_options:
        shared.update(shared_options)
    resolved = [_resolve_item_options(item, shared) for item in items]
    contexts: list[WriteContext] = []
    batch_explicit_target = "target" in (shared_options or {})
    batch_explicit_write_target = "write_target" in (shared_options or {})
    for item, opts in zip(items, resolved):
        item_options = item.options or {}
        if "target" in item_options or batch_explicit_target:
            effective_target = opts.get("target")
        elif "write_target" in item_options or batch_explicit_write_target:
            effective_target = opts.get("write_target")
        else:
            effective_target = "local"
        contexts.append(WriteContext(
            target=str(effective_target or "local").lower(),
            lake_root=str(opts["lake_root"]) if opts.get("lake_root") is not None else None,
            staging_dataset=str(opts.get("staging_dataset") or "factor_lake_staging"),
            data_source_config=_canonical_context_value(opts.get("data_source_config")),
            storage_format=str(opts.get("storage_format") or "long"),
            value_dtype=str(opts.get("value_dtype") or "float32"),
            atomicity=str(opts.get("atomicity") or "factor"),
        ))
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError("heterogeneous WriteContext in one materialize batch; split before writing")
    if contexts[0].atomicity not in {"factor", "batch"}:
        raise ValueError(f"unsupported write atomicity {contexts[0].atomicity!r}")
    return shared, resolved, contexts[0]


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
    from factor_engine.runtime import lineage_service
    from factor_engine.runtime.engine import _scope_from_factor
    from factor_engine.storage.catalog import compute_ir_hash

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
    from factor_engine.storage.materialize.materializer import storage_precision_policy_for

    _eff_dtype, _precision_policy = storage_precision_policy_for(
        opts.get("value_dtype"),
        production=production,
        lineage_extra=dict(lineage.extra),
    )
    lineage.extra["storage_precision_policy"] = _precision_policy
    lineage.extra["storage_value_dtype"] = _eff_dtype
    from factor_engine.runtime.materialize_service import _build_semantic_identity

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
        from factor_engine.runtime.dual_write_service import (
            MaterializationDelta,
            dual_write_clickhouse,
        )
        from factor_engine.runtime import lineage_service

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
    from factor_engine.runtime.dependency_catalog import DependencyCatalog  # noqa: F401
    from factor_engine.runtime.incremental_scheduler import (
        _calendar_version,
        _edges_from_analysis,
        _operator_manifest_from_ir,
        _resolve_source_dataset,
    )
    from factor_engine.runtime.materialize_service import _build_full_factor_definition

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
    from factor_engine.runtime.factor_block_ref import group_by_shared_axis

    series_by_fid: dict[str, pd.Series] = {}
    for prep in preps:
        result = prep["output"].get("result")
        if isinstance(result, pd.Series):
            series_by_fid[prep["factor_id"]] = result
    if len(series_by_fid) < 2:
        return 0
    groups = group_by_shared_axis(series_by_fid)
    return len(groups)


def _written_inventory(
    factor_id: str, *, materializer: Any, parquet_target: str
) -> tuple[list[dict[str, Any]], str]:
    from factor_engine.storage.materialize.lake_publish import (
        _factor_inventory, _inventory_digest, _resolve_staging_factor_dir,
    )
    if "staging" in str(parquet_target).lower():
        root = _resolve_staging_factor_dir(factor_id)
    elif str(parquet_target).lower() == "local":
        from factor_engine.security.factor_id import factor_dir_for
        root = factor_dir_for(materializer.lake_root, factor_id)
    else:
        raise ValueError(
            f"cannot prove parquet inventory for target={parquet_target!r}"
        )
    inventory = _factor_inventory(root)
    return inventory, _inventory_digest(inventory)


def _write_lock_root(factor_id: str, *, materializer: Any, parquet_target: str):
    if "staging" in str(parquet_target).lower():
        from factor_engine.storage.materialize.lake_publish import _resolve_staging_factor_dir
        return _resolve_staging_factor_dir(factor_id)
    if str(parquet_target).lower() == "local":
        from factor_engine.security.factor_id import factor_dir_for
        return factor_dir_for(materializer.lake_root, factor_id)
    return None


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
    from factor_engine.runtime import lineage_service
    from factor_engine.runtime.materialize_service import (
        _effective_data_source_config,
        resolve_parquet_write_target,
    )
    from factor_engine.runtime.production_policy import is_production_mode
    from factor_engine.storage.materializer import ParquetMaterializer

    items = list(items)
    counters = BatchMaterializeCounters(item_count=len(items))
    if not items:
        receipt = WriteReceipt(
            generation_id=None, expected_items=(), items={}, idempotency_key=None
        )
        return {
            "materializations": {},
            "per_item": [],
            "counters": counters.to_dict(),
            "generation": None,
            "receipt": receipt,
            "write_receipt": receipt.to_dict(),
        }
    item_names = [item.factor_id or item.factor.name for item in items]
    if len(set(item_names)) != len(item_names):
        raise ValueError("duplicate factor ids cannot share one WriteReceipt")

    # Resolve every physical context before constructing a materializer or
    # performing any write.
    shared, resolved_options, write_context = _resolve_batch_contexts(
        items, shared_options
    )
    # 调用方只给 ``write_target`` 未给 ``target`` 时（如 ``execute_materialize``
    # 语义），让 ``target`` 跟随 ``write_target``（缺省 target="local" 不算显式）。
    if "target" not in (shared_options or {}) and not any(
        "target" in (i.options or {}) for i in items
    ):
        _wt = shared.get("write_target")
        if _wt:
            shared["target"] = str(_wt)
    target = write_context.target
    lake_root = write_context.lake_root
    staging_dataset = write_context.staging_dataset
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
        from factor_engine.runtime.lineage import new_run_id

        generation_id = f"batch-{new_run_id()}"
    run_generation = generation_id
    receipt = WriteReceipt(
        generation_id=generation_id,
        expected_items=tuple(item.factor_id or item.factor.name for item in items),
        items={
            item.factor_id or item.factor.name: WriteItemReceipt(
                item.factor_id or item.factor.name
            )
            for item in items
        },
        idempotency_key=generation_id,
    )
    counters.hoisted_context = 1

    # --- per-item prepare（lineage/identity/scope 依赖单因子，无法向量化） ---
    preps: list[dict[str, Any]] = []
    for item, opts in zip(items, resolved_options):
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
            lock_root = _write_lock_root(
                prep["factor_id"], materializer=materializer,
                parquet_target=parquet_target,
            )
            if lock_root is None:
                raise ValueError("cannot establish factor-root write lock")
            from data_access.write.mutation_lock import mutation_lock
            with mutation_lock(lock_root):
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
                if int(summary.get("rows_written") or 0) > 0:
                    inventory, inventory_digest = _written_inventory(
                        prep["factor_id"], materializer=materializer,
                        parquet_target=parquet_target,
                    )
                    run_id = summary.get("run_id")
                    if not run_id:
                        raise ValueError("materializer returned no run_id")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "batch 单因子写失败 factor=%s: %s", prep["factor"].name, exc
            )
            err = {"error": f"{type(exc).__name__}: {exc}"}
            summaries[prep["factor"].name] = err
            per_item.append(err)
            item_receipt = receipt.items[prep["factor_id"]]
            # A writer exception does not prove absence of physical side effects.
            # Fail closed unless a future typed exception explicitly carries such proof.
            item_receipt.state = WriteState.IN_DOUBT
            item_receipt.error = "write outcome uncertain: " + err["error"]
            continue
        if int(summary.get("rows_written") or 0) == 0:
            counters.skipped_empty += 1
            summaries[prep["factor"].name] = summary
            per_item.append(summary)
            # An empty current run must never bind pre-existing factor bytes to
            # its new run/generation. Leave the receipt non-committable.
            item_receipt = receipt.items[prep["factor_id"]]
            item_receipt.state = WriteState.CREATED
            item_receipt.rows = 0
            continue
        summaries[prep["factor"].name] = summary
        per_item.append(summary)
        written_preps.append(prep)
        item_receipt = receipt.items[prep["factor_id"]]
        item_receipt.state = WriteState.STAGED
        item_receipt.rows = int(summary.get("rows_written") or 0)
        item_receipt.run_id = str(run_id)
        item_receipt.inventory = inventory
        item_receipt.inventory_digest = inventory_digest

    written_preps = [
        prep for prep in written_preps
        if receipt.items[prep["factor_id"]].state == WriteState.STAGED
    ]
    # --- R39-PERF-030: 同 axis 共享观测（不改变逐因子写入；同 index 不重复） ---
    counters.factor_block_shared_axis_count = _record_shared_axes(written_preps)

    failed_items = [
        item for item in receipt.items.values() if item.state == WriteState.FAILED
    ]
    uncertain_items = [
        item for item in receipt.items.values() if item.state == WriteState.IN_DOUBT
    ]
    if (failed_items or uncertain_items) and write_context.atomicity == "batch":
        for item in receipt.items.values():
            if item.state == WriteState.STAGED:
                item.state = WriteState.IN_DOUBT
                item.error = "batch atomicity broken before catalog commit; reconciliation required"
        if production:
            from factor_engine.storage.exceptions import MaterializedButCatalogCommitFailed

            raise MaterializedButCatalogCommitFailed(
                "batch atomic materialize has partial physical writes "
                "(IN_DOUBT; do not replay individual items)",
                materialization={
                    **summaries, "_write_receipt": receipt.to_dict()
                },
            )

    # --- batched catalog commit：依赖 manifest 单事务（Gate-03 核心） ---
    if written_preps and not (
        (failed_items or uncertain_items) and write_context.atomicity == "batch"
    ):
        payloads = _build_manifest_payloads(
            engine, written_preps, production=production, data_source=engine.data_source
        )
        receipt.manifest_digest = hashlib.sha256(
            json.dumps(
                _canonical_context_value(payloads),
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest()
        from factor_engine.runtime.dependency_catalog import DependencyCatalog

        dep_catalog = DependencyCatalog(materializer.catalog)
        try:
            written = dep_catalog.record_factor_manifests_many(payloads)
            if int(written) != len(payloads):
                raise RuntimeError(
                    "dependency manifest commit count mismatch: "
                    f"expected={len(payloads)} written={written}"
                )
            counters.batch_write_transaction_count += 1
            counters.manifest_writes += written
            for prep in written_preps:
                receipt.items[prep["factor_id"]].state = WriteState.COMMITTED
        except Exception as exc:  # noqa: BLE001
            # 与 execute_materialize 的 catalog 提交失败语义一致：production 下
            # 数据已落盘但目录未提交 → IN_DOUBT（fail-closed）；research 保留 warning。
            for prep in written_preps:
                item_receipt = receipt.items[prep["factor_id"]]
                item_receipt.state = WriteState.IN_DOUBT
                item_receipt.error = f"{type(exc).__name__}: {exc}"
            if production:
                from factor_engine.storage.exceptions import MaterializedButCatalogCommitFailed

                raise MaterializedButCatalogCommitFailed(
                    f"batch materialize: 因子数据已落盘但依赖/full-definition catalog "
                    f"批量提交失败（IN_DOUBT，需对账）: {exc}",
                    materialization={
                        **summaries, "_write_receipt": receipt.to_dict()
                    },
                ) from exc
            logger.warning(
                "batch 依赖 manifest 目录提交失败（research，已记录待对账）: %s", exc
            )
            for name, s in summaries.items():
                if "error" not in s:
                    s["catalog_commit_warning"] = str(exc)

    # --- 单 generation publish ---
    if generation is not None:
        generation.publish(
            sum(item.state == WriteState.COMMITTED for item in receipt.items.values())
        )

    logger.info(
        "execute_materialize_batch items=%d batch_write_transactions=%d "
        "manifest_writes=%d partition_rounds=%d",
        len(items),
        counters.batch_write_transaction_count,
        counters.manifest_writes,
        counters.physical_partition_write_rounds,
    )
    receipt.validate()
    return {
        "materializations": summaries,
        "per_item": per_item,
        "counters": counters.to_dict(),
        "generation": generation_id,
        "receipt": receipt,
        "write_receipt": receipt.to_dict(),
    }
