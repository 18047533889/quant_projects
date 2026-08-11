"""因子落盘核心：pd.Series → 分区 Parquet，支持幂等增量更新与原子写入。

设计要点
--------
1. **Schema 强转**：因子值默认 ``float32``（可配置 ``value_dtype``），资产列强转为 ``string``。
2. **数据清洗 + 显式 tombstone**：``±inf → NaN``；NaN/Inf 行不再 ``dropna``，而是
   作为显式 tombstone 保留 —— tombstone 标记为 ``value=NaN`` + ``is_valid=0`` +
   ``invalid_reason="inf_or_nan"``。下游 Upsert 的 ``[datetime, asset]`` dedup
   keep="last" 会用 NaN 覆盖旧有限值，读取端把该格映射为 NaN/invalid，从而支持
   valid→null / valid→deleted / 退出 universe / 源修订删行。
3. **幂等 Upsert**：按年分区，旧数据与新数据 Concat 后按 ``[datetime, asset]``
   去重（保留最新），排序后整体覆盖。
4. **原子写入**：先写 ``.data.parquet.tmp``，``os.replace()`` 覆盖正式文件。
5. **元数据联动**：自动注册因子 + 更新水位线（AST Hash 防呆）。

目录拓扑
--------
::

    {lake_root}/
    ├── _catalog.sqlite
    └── factors/
        └── {factor_id}/
            ├── year=2023/
            │   └── data.parquet
            └── year=2024/
                └── data.parquet
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from workspace_paths import default_factor_lake_root

from runtime.factor_identity import (
    NO_FACTOR_IDENTITY,
    FactorIdentityMismatch,
    FactorSemanticIdentity,
    checkpoint_fingerprint,
    checkpoint_fingerprint_matches,
    compute_identity_from_materialize_ctx,
    partition_input_fingerprint,
)

from ..catalog import FactorCatalog, compute_ir_hash
from ..exceptions import FactorNotFoundError, MaterializePartitionError
from ..partition_policy import (
    PartitionPolicy,
    attach_partition_columns,
    checkpoint_year,
    iter_partition_groups,
    partition_key,
    partition_path_segments,
)
from logging_utils import ProgressLogger, get_logger

logger = get_logger("storage.materializer")

from storage.factor_schema import FACTOR_METADATA_COLUMNS as METADATA_COLUMNS

from storage.partition_stats import (
    PartitionCommitStats,
    WritePassDQStats,
    compute_partition_commit_stats,
    compute_write_pass_dq,
    count_parquet_rows_from_footer,
)
from storage.write_amplification import WriteAmplificationTracker

# ---------------------------------------------------------------------------
# R39 §28 hard-gate counters (read by scripts/r39_hard_gates_audit.py).
# ---------------------------------------------------------------------------

#: Gate-05: incremented only when a genuine full-history value-cell re-scan is
#: required (stats + footer are unavailable).  == 0 on the normal write path.
full_factor_rescan_count = 0

#: Gate-04: accumulated historical-rewrite bytes.  == 0 for ordinary incremental
#: writes in delta mode.
historical_rewrite_bytes = 0

#: R39 PERF-051: incremented only when the write path genuinely must re-read the
#: committed watermark (watermark_deferred / external-modification scenarios).
#: == 0 on the normal write path, which now returns the pending watermark the
#: transaction just wrote instead of issuing a post-write ``get_watermark`` query.
post_write_watermark_readback_count = 0


def get_post_write_watermark_readback_count() -> int:
    """Return the module-level post-write watermark read-back counter (PERF-051)."""
    return post_write_watermark_readback_count


def reset_post_write_watermark_readback_count() -> None:
    """Reset the module-level post-write watermark read-back counter (PERF-051)."""
    global post_write_watermark_readback_count

    post_write_watermark_readback_count = 0


@dataclass(frozen=True)
class MaterializeMetadata:
    """因子落盘元数据字段容器。

    参数:
        无
    """

    calc_time: str
    factor_version: str
    data_snapshot_id: str | None = None
    is_valid: int = 1
    invalid_reason: str = ""
    #: R20-207..212: row-level metadata records the RESOLVED source snapshot
    #: (the combined snapshot identity) — not just the caller's explicit
    #: ``data_snapshot_id``, which may be empty while lineage_extra carries the
    #: resolved value.
    resolved_snapshot_id: str | None = None
    #: R20-201..206: storage precision policy captured per row so catalog /
    #: checkpoint / lineage all see the same value dtype contract.
    storage_precision_policy: str = ""


# ---------------------------------------------------------------------------
# R20-210: identity 异常区分
# ---------------------------------------------------------------------------

class IdentityUnavailable(ValueError):
    """因子身份合法不可得：既无 IR 也无有效 ast_hash（无身份可计算）。"""


class IdentityComputationFailed(RuntimeError):
    """因子身份计算代码抛异常：production 必须 hard fail，绝不降级为无指纹。"""


# ---------------------------------------------------------------------------
# R20-201..206: storage precision policy
# ---------------------------------------------------------------------------

#: storage precision policy 常量。production 默认 float64；float32 仅在持有明确
#: quantization certificate 时允许（``storage_precision_policy`` 落进
#: lineage/row metadata/checkpoint，成为语义身份的一部分）。
PRECISION_FLOAT64_DEFAULT = "float64_default"
PRECISION_FLOAT32_CERTIFIED = "float32_certified"
PRECISION_FLOAT32_LEGACY = "float32_legacy"

_FLOAT32_DTYPES = frozenset({"float32", "float", "f4", "float16", "f2"})

#: R32-P0-027: write_mode 严格枚举门。除这四个值之外的任何拼写必须拒绝
#: （历史行为：拼错字符串静默回落为 keep-last upsert，掩盖写入意图错误）。
WRITE_MODES = frozenset({"upsert", "append", "replace_window", "recompute_window"})


def validate_write_mode(write_mode: str) -> str:
    """R32-P0-027: 严格校验 write_mode 枚举，非法值直接抛 ``ValueError``。

    历史上 ``"upsertt"`` / ``"replaced"`` 等拼写错误会静默走 keep-last upsert；
    现在入口严格只允许 upsert/append/replace_window/recompute_window。
    """
    mode = str(write_mode or "upsert").lower()
    if mode not in WRITE_MODES:
        raise ValueError(
            f"write_mode must be one of {sorted(WRITE_MODES)}, got {write_mode!r}"
        )
    if mode == "replace_window":
        # 依赖调用方随后校验 replace_window 窗口元组（_upsert_partition 内）。
        pass
    return mode


#: R32-P0-025: OutputGrainContract —— materializer 只支持精确的
#: ``timestamp × instrument`` 两层 MultiIndex。nlevels>2 的额外 grain（minute/
#: session/event/relation/extra dimension）必须走专门 schema，禁止静默降维丢弃。
OUTPUT_GRAIN_MAX_NLEVELS = 2


def storage_precision_policy_for(
    value_dtype: str | None,
    *,
    production: bool,
    lineage_extra: dict | None = None,
) -> tuple[str, str]:
    """解析 effective value dtype + storage precision policy（R20-201..206）。

    - 显式 ``value_dtype``：使用它；production 下 float32 且无 quantization
      certificate 时记录 ``float32_legacy``（保留兼容，但 policy 显式进入血缘）；
    - 未显式提供：production -> float64，research -> float32。
    """
    lineage_extra = lineage_extra or {}
    float32_certified = bool(
        lineage_extra.get("storage_precision_policy") == PRECISION_FLOAT32_CERTIFIED
        or lineage_extra.get("float32_quantization_certified")
    )
    if value_dtype is None or str(value_dtype).strip() == "":
        effective = "float64" if production else "float32"
        policy = PRECISION_FLOAT64_DEFAULT if production else PRECISION_FLOAT32_LEGACY
        return effective, policy
    effective = str(value_dtype).strip().lower()
    if effective in _FLOAT32_DTYPES:
        policy = (
            PRECISION_FLOAT32_CERTIFIED if float32_certified else PRECISION_FLOAT32_LEGACY
        )
    elif effective == "float64":
        policy = PRECISION_FLOAT64_DEFAULT
    else:
        policy = "explicit"
    return effective, policy


# ---------------------------------------------------------------------------
# 默认作者
# ---------------------------------------------------------------------------

def _get_default_author() -> str:
    """获取默认因子作者标识。
    
    参数:
        无
    
    返回:
        str
    """
    return os.getenv("QUANTSOCIETY_USER", getpass.getuser() or "anonymous")


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------

class ParquetMaterializer:
    """因子结果物化到本地 Parquet 因子湖。
    
    参数:
        lake_root: 因子湖根目录（可选）
        catalog: FactorCatalog 实例（可选）
        staging_dataset: 见函数签名（可选）
    """

    def __init__(
        self,
        lake_root: str | Path | None = None,
        catalog: FactorCatalog | None = None,
        *,
        staging_dataset: str = "factor_lake_staging",
        delta_mode: bool | None = None,
    ) -> None:
        """初始化实例。

        参数:
            lake_root: 因子湖根目录（可选）
            catalog: FactorCatalog 实例（可选）
            staging_dataset: 见函数签名（可选）
            delta_mode: 显式 delta 存储开关；None 时读环境变量
                ``FACTOR_ENGINE_DELTA_STORAGE``（R39 PERF-052，DEFAULT OFF）。

        返回:
            无
        """
        if lake_root is None:
            lake_root = default_factor_lake_root()
        self._lake_root = Path(lake_root)
        self._lake_root.mkdir(parents=True, exist_ok=True)

        if catalog is None:
            catalog = FactorCatalog(self._lake_root / "_catalog.sqlite")
        self._catalog = catalog
        self._staging_dataset = str(staging_dataset or "factor_lake_staging")
        # R39 PERF-052/054/044/05: delta-mode opt-in (DEFAULT OFF), write
        # amplification tracker + full-factor-rescan counter + lock reuse.
        self._delta_mode = delta_mode if delta_mode is not None else self._delta_mode_from_env()
        self.write_amplification = WriteAmplificationTracker(delta_mode=self._delta_mode)
        self.full_factor_rescan_count = 0
        self.post_write_watermark_readback_count = 0
        self._lock_manager = None  # created lazily in delta mode

    @staticmethod
    def _delta_mode_from_env() -> bool:
        """Resolve delta-mode activation from ``FACTOR_ENGINE_DELTA_STORAGE``."""
        raw = os.getenv("FACTOR_ENGINE_DELTA_STORAGE", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _resolve_delta_mode(self, explicit: bool | None) -> bool:
        """Explicit flag > constructor flag > environment (DEFAULT OFF)."""
        if explicit is not None:
            return bool(explicit)
        return self._delta_mode

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def lake_root(self) -> Path:
        """lake_root。
        
        参数:
            无
        
        返回:
            Path
        """
        return self._lake_root

    @property
    def catalog(self) -> FactorCatalog:
        """catalog。
        
        参数:
            无
        
        返回:
            FactorCatalog
        """
        return self._catalog

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def materialize(
        self,
        factor_id: str,
        result: pd.Series,
        *,
        author: str | None = None,
        frequency: str = "1d",
        ast_hash: str | None = None,
        ir_node: Any | None = None,
        description: str | None = None,
        expression: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        run_lineage: dict | None = None,
        write_metadata: bool = True,
        data_snapshot_id: str | None = None,
        data_source_config: dict | None = None,
        isolate_partition_failures: bool = True,
        resume: bool = False,
        preserve_invalid_rows: bool = False,
        value_dtype: str | None = None,
        write_target: str = "local",
        defer_watermark: bool = False,
        partition_columns: list[str] | None = None,
        storage_format: str = "long",
        null_overwrite: bool = False,
        deleted_keys: list[tuple] | None = None,
        production: bool | None = None,
        run_generation: str | None = None,
        force_tombstones: bool | None = None,
        semantic_identity: FactorSemanticIdentity | None = None,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
        delta_mode: bool | None = None,
    ) -> dict:
        """将因子计算结果落盘为分区 Parquet。
        
        参数:
            factor_id: 因子唯一标识
            result: 因子计算结果 Series
            author: 见函数签名（可选）
            frequency: 因子频率（可选）
            ast_hash: 因子 AST 哈希（可选）
            ir_node: 因子 IR 树根节点（可选）
            description: 见函数签名（可选）
            expression: 见函数签名（可选）
            dq_check: 见函数签名（可选）
            dq_strict: 见函数签名（可选）
            dq_thresholds: 见函数签名（可选）
            run_lineage: 见函数签名（可选）
            write_metadata: 见函数签名（可选）
            data_snapshot_id: 见函数签名（可选）
            data_source_config: 见函数签名（可选）
            isolate_partition_failures: 见函数签名（可选）
            resume: 见函数签名（可选）
            preserve_invalid_rows: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
            write_target: 见函数签名（可选）
            defer_watermark: 见函数签名（可选）
            partition_columns: 见函数签名（可选）
            storage_format: 见函数签名（可选）
        
        返回:
            dict
        """
        prep = self._materialize_prepare(
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            ir_node=ir_node,
            description=description,
            expression=expression,
            run_lineage=run_lineage,
            write_metadata=write_metadata,
            data_snapshot_id=data_snapshot_id,
            data_source_config=data_source_config,
            production=production,
            run_generation=run_generation,
            semantic_identity=semantic_identity,
            write_target=write_target,
            value_dtype=value_dtype,
            write_mode=write_mode,
            replace_window=replace_window,
        )
        factor_id = prep["factor_id"]
        author = prep["author"]
        write_mode = prep["write_mode"]
        ast_hash = prep["ast_hash"]
        production = prep["production"]
        run_lineage = prep["run_lineage"]
        effective_value_dtype = prep["effective_value_dtype"]
        precision_policy = prep["precision_policy"]
        identity_digest = prep["identity_digest"]
        lineage_extra = prep["lineage_extra"]
        source_snapshot = prep["source_snapshot"]
        source_dep_hash = prep["source_dep_hash"]
        generation = prep["generation"]


        # --- 0. 可选 DQ 门禁（在清洗前检查原始 result）---
        dq_report = None
        if dq_check:
            from runtime.dq_gates import assert_factor_dq

            dq_report = assert_factor_dq(
                result,
                thresholds=dq_thresholds,
                raise_on_fail=dq_strict,
                preserve_invalid_rows=preserve_invalid_rows,
            )

        # --- 1. 转长表 + 强制 Schema ---
        meta = None
        if write_metadata:
            # R11 #6: 落盘行 factor_version = 语义身份 digest 前缀（缺省 ast_hash
            # 前缀）——不再只用 ast_hash，否则「公式一样但 data_source/执行语义变
            # 了」的因子仍被 catalog 当作同一版本。
            # R20-207..212: row-level metadata 写 RESOLVED source_snapshot（统一
            # 用 data_snapshot_id 兜底 lineage_extra 的 combined snapshot），不再
            # 只写调用方原始 data_snapshot_id。
            meta = MaterializeMetadata(
                calc_time=datetime.now(timezone.utc).isoformat(),
                factor_version=(identity_digest or ast_hash)[:16],
                data_snapshot_id=data_snapshot_id,
                resolved_snapshot_id=source_snapshot,
                storage_precision_policy=precision_policy,
            )
        df = self._normalize_to_long_table(result, metadata=meta, value_dtype=effective_value_dtype)

        return self._materialize_long_df(
            factor_id=factor_id,
            df=df,
            meta=meta,
            dq_report=dq_report,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            description=description,
            expression=expression,
            run_lineage=run_lineage,
            write_metadata=write_metadata,
            data_snapshot_id=data_snapshot_id,
            data_source_config=data_source_config,
            isolate_partition_failures=isolate_partition_failures,
            resume=resume,
            preserve_invalid_rows=preserve_invalid_rows,
            write_target=write_target,
            defer_watermark=defer_watermark,
            partition_columns=partition_columns,
            storage_format=storage_format,
            null_overwrite=null_overwrite,
            deleted_keys=deleted_keys,
            production=production,
            run_generation=run_generation,
            force_tombstones=force_tombstones,
            write_mode=write_mode,
            replace_window=replace_window,
            delta_mode=delta_mode,
            effective_value_dtype=effective_value_dtype,
            precision_policy=precision_policy,
            identity_digest=identity_digest,
            lineage_extra=lineage_extra,
            source_snapshot=source_snapshot,
            source_dep_hash=source_dep_hash,
            generation=generation,
        )

    # ------------------------------------------------------------------
    # R39 PERF-046/047: Arrow block fast-path materialization
    # ------------------------------------------------------------------

    def materialize_block(
        self,
        factor_id: str,
        block: Any,
        *,
        author: str | None = None,
        frequency: str = "1d",
        ast_hash: str | None = None,
        ir_node: Any | None = None,
        description: str | None = None,
        expression: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        run_lineage: dict | None = None,
        write_metadata: bool = True,
        data_snapshot_id: str | None = None,
        data_source_config: dict | None = None,
        isolate_partition_failures: bool = True,
        resume: bool = False,
        preserve_invalid_rows: bool = False,
        value_dtype: str | None = None,
        write_target: str = "local",
        defer_watermark: bool = False,
        partition_columns: list[str] | None = None,
        storage_format: str = "long",
        null_overwrite: bool = False,
        deleted_keys: list[tuple] | None = None,
        production: bool | None = None,
        run_generation: str | None = None,
        force_tombstones: bool | None = None,
        semantic_identity: FactorSemanticIdentity | None = None,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
        delta_mode: bool | None = None,
    ) -> dict:
        """Arrow fast-path materialization (R39 PERF-046/047).

        ``block`` is a ``pa.Table`` or ``pyarrow.ipc.RecordBatchReader`` with
        ``datetime``/``asset``/``value`` columns (metadata columns optional).
        Tombstone/finite masks are generated directly with Arrow ops
        (``is_finite``/``if_else``/``cast``/``dictionary_encode``) rather than
        per-row Python string construction.  Output is byte-identical to
        :meth:`materialize` on the same data (test asserts values/index/watermark
        equality).  The existing ``evaluate_factor_dq`` gate still runs when
        ``dq_check`` is enabled.
        """
        prep = self._materialize_prepare(
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            ir_node=ir_node,
            description=description,
            expression=expression,
            run_lineage=run_lineage,
            write_metadata=write_metadata,
            data_snapshot_id=data_snapshot_id,
            data_source_config=data_source_config,
            production=production,
            run_generation=run_generation,
            semantic_identity=semantic_identity,
            write_target=write_target,
            value_dtype=value_dtype,
            write_mode=write_mode,
            replace_window=replace_window,
        )
        factor_id = prep["factor_id"]
        author = prep["author"]
        write_mode = prep["write_mode"]
        ast_hash = prep["ast_hash"]
        production = prep["production"]
        run_lineage = prep["run_lineage"]
        effective_value_dtype = prep["effective_value_dtype"]
        precision_policy = prep["precision_policy"]
        identity_digest = prep["identity_digest"]
        lineage_extra = prep["lineage_extra"]
        source_snapshot = prep["source_snapshot"]
        source_dep_hash = prep["source_dep_hash"]
        generation = prep["generation"]

        # --- 0. 可选 DQ 门禁（在清洗前检查原始 block 值；evaluate_factor_dq 仍运行）---
        dq_report = None
        if dq_check:
            from runtime.dq_gates import assert_factor_dq

            raw_series = self._block_to_raw_series(block)
            dq_report = assert_factor_dq(
                raw_series,
                thresholds=dq_thresholds,
                raise_on_fail=dq_strict,
                preserve_invalid_rows=preserve_invalid_rows,
            )

        # --- 1. 长表 + 强制 Schema（Arrow 路径：直接生成 tombstone/finite mask）---
        meta = None
        if write_metadata:
            meta = MaterializeMetadata(
                calc_time=datetime.now(timezone.utc).isoformat(),
                factor_version=(identity_digest or ast_hash)[:16],
                data_snapshot_id=data_snapshot_id,
                resolved_snapshot_id=source_snapshot,
                storage_precision_policy=precision_policy,
            )
        df = self._arrow_block_to_long_df(
            block, metadata=meta, value_dtype=effective_value_dtype
        )

        return self._materialize_long_df(
            factor_id=factor_id,
            df=df,
            meta=meta,
            dq_report=dq_report,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            description=description,
            expression=expression,
            run_lineage=run_lineage,
            write_metadata=write_metadata,
            data_snapshot_id=data_snapshot_id,
            data_source_config=data_source_config,
            isolate_partition_failures=isolate_partition_failures,
            resume=resume,
            preserve_invalid_rows=preserve_invalid_rows,
            write_target=write_target,
            defer_watermark=defer_watermark,
            partition_columns=partition_columns,
            storage_format=storage_format,
            null_overwrite=null_overwrite,
            deleted_keys=deleted_keys,
            production=production,
            run_generation=run_generation,
            force_tombstones=force_tombstones,
            write_mode=write_mode,
            replace_window=replace_window,
            delta_mode=delta_mode,
            effective_value_dtype=effective_value_dtype,
            precision_policy=precision_policy,
            identity_digest=identity_digest,
            lineage_extra=lineage_extra,
            source_snapshot=source_snapshot,
            source_dep_hash=source_dep_hash,
            generation=generation,
        )

    def _materialize_prepare(
        self,
        *,
        factor_id: str,
        author: str | None,
        frequency: str,
        ast_hash: str | None,
        ir_node: Any | None,
        description: str | None,
        expression: str | None,
        run_lineage: dict | None,
        write_metadata: bool,
        data_snapshot_id: str | None,
        data_source_config: dict | None,
        production: bool | None,
        run_generation: str | None,
        semantic_identity: FactorSemanticIdentity | None,
        write_target: str,
        value_dtype: str | None,
        write_mode: str,
        replace_window: tuple[str, str] | None,
    ) -> dict:
        """Shared identity / precision / write-target preparation used by both
        :meth:`materialize` and :meth:`materialize_block` (R39 PERF-046)."""
        if author is None:
            author = _get_default_author()

        # R32-P0-036/043: factor_id 统一 domain gate。
        from security.factor_id import validate_factor_id

        factor_id = validate_factor_id(factor_id)

        # R32-P0-027: write_mode 严格枚举门。
        write_mode = validate_write_mode(write_mode)
        if write_mode == "replace_window":
            if replace_window is None or len(replace_window) != 2:
                raise ValueError(
                    f"write_mode='replace_window' requires replace_window=(start, end), "
                    f"got {replace_window!r}"
                )

        # --- AST Hash ---
        if ast_hash is None:
            if ir_node is not None:
                ast_hash = compute_ir_hash(ir_node)
            else:
                ast_hash = NO_FACTOR_IDENTITY

        # R10 #17: production 物化必须提供真实因子身份。
        production = self._resolve_production(production)
        if production and ast_hash == NO_FACTOR_IDENTITY:
            raise FactorIdentityMismatch(
                f"production materialize of factor '{factor_id}' requires a "
                f"factor identity; got sentinel {NO_FACTOR_IDENTITY!r}. "
                f"Pass ast_hash or ir_node."
            )

        # #收官轮 P0（Integration）：write_target 在任何 side effect 之前严格枚举。
        from storage.write_targets import normalize_write_target

        write_target_flags = normalize_write_target(write_target)
        if production and write_target_flags["local"]:
            raise ValueError(
                f"factor_id={factor_id!r}: production 禁止 direct-local factor-lake "
                "write（绕过 DataAccess staging→publish 原子发布/journal/snapshot "
                "manifest）。请用 write_target='staging' + publish_factor_lake。"
            )

        # R10 #47 / R20-210: 断点续写身份。
        checkpoint_identity = semantic_identity
        if checkpoint_identity is None:
            has_any_identity = ir_node is not None and ast_hash != NO_FACTOR_IDENTITY
            if not has_any_identity and ast_hash in (None, NO_FACTOR_IDENTITY):
                checkpoint_identity = None
            else:
                try:
                    checkpoint_identity = compute_identity_from_materialize_ctx(
                        ir_node=ir_node,
                        ast_hash=ast_hash,
                        data_source_config=data_source_config,
                        run_lineage=run_lineage,
                        frequency=frequency,
                    )
                except IdentityUnavailable:
                    checkpoint_identity = None
                except Exception as exc:
                    if production:
                        raise IdentityComputationFailed(
                            f"factor={factor_id}: checkpoint 身份计算代码抛异常，"
                            f"production 拒绝降级为无指纹（R20-210）: {exc!r}"
                        ) from exc
                    logger.debug(
                        "checkpoint 身份计算失败 factor=%s, 降级为无指纹",
                        factor_id,
                        exc_info=True,
                    )
                    checkpoint_identity = None
        identity_digest = (
            checkpoint_identity.identity_digest()
            if checkpoint_identity is not None
            else None
        )
        lineage_extra = (run_lineage or {}).get("extra") or {}
        source_snapshot = data_snapshot_id or lineage_extra.get("data_snapshot_id")
        source_dep_hash = lineage_extra.get("source_dependency_hash")
        if source_dep_hash is None and checkpoint_identity is not None:
            source_dep_hash = checkpoint_identity.source_dependency_hash
        generation = str(run_generation or "0")

        # R20-201..206: storage precision policy。
        effective_value_dtype, precision_policy = storage_precision_policy_for(
            value_dtype,
            production=production,
            lineage_extra=lineage_extra,
        )
        if write_metadata:
            lineage_extra = dict(lineage_extra)
            lineage_extra["storage_precision_policy"] = precision_policy
            lineage_extra["storage_value_dtype"] = effective_value_dtype
            if run_lineage is not None:
                run_lineage = dict(run_lineage)
                run_lineage["extra"] = lineage_extra
        return {
            "factor_id": factor_id,
            "author": author,
            "write_mode": write_mode,
            "ast_hash": ast_hash,
            "production": production,
            "run_lineage": run_lineage,
            "effective_value_dtype": effective_value_dtype,
            "precision_policy": precision_policy,
            "identity_digest": identity_digest,
            "lineage_extra": lineage_extra,
            "source_snapshot": source_snapshot,
            "source_dep_hash": source_dep_hash,
            "generation": generation,
        }

    @staticmethod
    def _coerce_block(block: Any):
        """Coerce ``pa.Table`` / ``RecordBatchReader`` to a ``pa.Table``."""
        import pyarrow as pa

        if isinstance(block, pa.Table):
            return block
        if hasattr(block, "read_all"):
            try:
                return block.read_all()
            finally:
                # Consume/close the stream so pyarrow's C++ objects are torn
                # down deterministically (avoids GC-time teardown aborts).
                close = getattr(block, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
        raise ValueError(
            "materialize_block expects pyarrow.Table or pyarrow.ipc.RecordBatchReader, "
            f"got {type(block).__name__}"
        )

    def _block_to_raw_series(self, block: Any) -> pd.Series:
        """Rebuild the raw (uncleaned) MultiIndex Series for the DQ gate."""
        table = self._coerce_block(block)
        dt = pd.to_datetime(table["datetime"].to_pandas())
        asset = table["asset"].to_pandas().astype(str)
        vals = np.asarray(table["value"].to_numpy(), dtype=float)
        idx = pd.MultiIndex.from_arrays([dt, asset], names=["timestamp", "instrument"])
        return pd.Series(vals, index=idx, name="value")

    @staticmethod
    def _arrow_block_to_long_df(
        block: Any,
        *,
        metadata: "MaterializeMetadata | None" = None,
        value_dtype: str = "float32",
    ) -> pd.DataFrame:
        """Arrow block → normalized long DataFrame (R39 PERF-046/047).

        Tombstone/finite masks are generated directly with Arrow compute ops
        (``is_finite``/``if_else``/``cast``/``dictionary_encode``); constant
        metadata columns use Arrow dictionary encoding rather than per-row
        Python string construction.  The returned frame matches what
        ``_normalize_to_long_table`` + ``_clean`` produce for the same data.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        table = ParquetMaterializer._coerce_block(block)
        cols = set(table.column_names)
        if not {"datetime", "asset", "value"}.issubset(cols):
            raise ValueError(
                "materialize_block table must have datetime/asset/value columns, "
                f"got {table.column_names}"
            )

        dt_arr = table["datetime"]
        if not pa.types.is_timestamp(dt_arr.type):
            dt_arr = pc.cast(dt_arr, pa.timestamp("ns"))
        dt_series = pd.Series(dt_arr.to_pandas(), name="datetime")

        asset_arr = table["asset"]
        if not (pa.types.is_string(asset_arr.type) or pa.types.is_large_string(asset_arr.type)):
            asset_arr = pc.cast(asset_arr, pa.string())
        asset_series = pd.Series(asset_arr.to_pandas(), name="asset").astype("string")

        value_arr = table["value"]
        _f32 = value_dtype in ("float32", "float", "f4", "float16", "f2")
        target_type = pa.float32() if _f32 else pa.float64()
        if not pa.types.is_floating(value_arr.type):
            value_arr = pc.cast(value_arr, target_type)
        # Tombstone/finite mask directly with Arrow ops (PERF-047).
        if pa.types.is_floating(value_arr.type):
            finite = pc.is_finite(value_arr)
        else:
            finite = pc.is_valid(value_arr)
        value_clean = pc.if_else(
            finite, value_arr, pa.scalar(float("nan"), value_arr.type)
        )
        is_valid = pc.if_else(
            finite, pa.scalar(1, pa.int8()), pa.scalar(0, pa.int8())
        )
        invalid_reason = pc.if_else(
            finite, pa.scalar("", pa.string()), pa.scalar("inf_or_nan", pa.string())
        )
        value_series = pd.Series(value_clean.to_pandas(), name="value").astype(
            str(value_dtype or "float32")
        )

        n = table.num_rows
        now_iso = (
            metadata.calc_time
            if metadata is not None
            else datetime.now(timezone.utc).isoformat()
        )
        fv = metadata.factor_version if metadata is not None else ""
        dsid = (metadata.data_snapshot_id or "") if metadata is not None else ""
        rsnap = (metadata.resolved_snapshot_id or "") if metadata is not None else ""
        spol = (metadata.storage_precision_policy or "") if metadata is not None else ""

        def _const(v: str) -> pd.Series:
            arr = pa.array([v] * n, type=pa.string())
            return pd.Series(pc.dictionary_encode(arr).to_pandas(), dtype=object)

        df = pd.DataFrame(
            {
                "datetime": dt_series,
                "asset": asset_series,
                "value": value_series,
                "calc_time": _const(now_iso),
                "factor_version": _const(fv),
                "data_snapshot_id": _const(dsid),
                "resolved_snapshot_id": _const(rsnap),
                "storage_precision_policy": _const(spol),
                "is_valid": pd.Series(is_valid.to_pandas(), dtype="int64"),
                "invalid_reason": pd.Series(invalid_reason.to_pandas(), dtype=object),
            }
        )
        return df

    # ------------------------------------------------------------------
    # R39 PERF-048/049: watermark metrics without a full-history rescan
    # ------------------------------------------------------------------

    def _compute_watermark_metrics(
        self,
        factor_id: str,
        factor_dir: Path,
        stats_by_partition: dict[str, PartitionCommitStats],
        active_df: pd.DataFrame,
    ) -> tuple[dict, int, bool, dict]:
        """Aggregate partition metrics from in-memory commit stats + catalog
        stored stats (PERF-048/049).  Legacy on-disk partitions lacking stats
        are reconciled via Parquet footer metadata (never value-cell loads).

        Returns ``(partition_metrics, total_rows, rescanned, wm_range)`` where
        ``rescanned`` is True only if a genuine full-history value-cell re-scan
        was required (which increments ``full_factor_rescan_count``), and
        ``wm_range`` is ``{start_date, end_date, has_legacy}`` describing the
        post-write full-history date range derived from merged partition stats
        (R39 PERF-051).  ``has_legacy`` is True when at least one on-disk
        partition predates incremental stats (no date info) — callers must then
        fall back to the committed watermark to preserve the range.
        """
        stored = self._catalog.get_partition_stats_all(factor_id)
        merged: dict[str, dict] = {}
        for k, row in stored.items():
            merged[k] = {
                "rows": int(row["rows"]),
                "valid_rows": int(row["valid_rows"]),
                "min_date": row.get("min_date"),
                "max_date": row.get("max_date"),
                "file_bytes": int(row.get("file_bytes") or 0),
            }
        for k, s in stats_by_partition.items():
            merged[k] = {
                "rows": s.rows_after,
                "valid_rows": s.valid_after,
                "min_date": s.min_date,
                "max_date": s.max_date,
                "file_bytes": s.file_bytes,
            }
        # Reconcile on-disk partitions that predate incremental stats.
        if factor_dir.exists():
            for pq_file in factor_dir.rglob("*.parquet"):
                if pq_file.name.startswith("."):
                    continue
                try:
                    rel_dir = pq_file.parent.relative_to(factor_dir)
                except ValueError:
                    continue
                segs = [seg for seg in rel_dir.parts if "=" in seg]
                if not segs:
                    continue
                pkey = "|".join(sorted(segs))
                if pkey in merged:
                    continue
                rows = count_parquet_rows_from_footer(pq_file)
                merged[pkey] = {
                    "rows": rows,
                    "valid_rows": rows,
                    "min_date": None,
                    "max_date": None,
                    "file_bytes": int(pq_file.stat().st_size),
                }
        if not merged:
            # Truly no stats anywhere: one-time full value-cell scan (counted).
            self._bump_full_rescan()
            metrics = self._count_partition_metrics(factor_dir)
            wm_range = {"start_date": None, "end_date": None, "has_legacy": True}
            return metrics, metrics["physical_row_count"], True, wm_range
        total_rows = sum(int(r["rows"]) for r in merged.values())
        valid_rows = sum(int(r["valid_rows"]) for r in merged.values())
        file_bytes = sum(int(r.get("file_bytes") or 0) for r in merged.values())
        if len(active_df):
            date_count = int(active_df["datetime"].nunique())
            asset_count = int(active_df["asset"].nunique())
            non_null = int(active_df["value"].notna().sum())
        else:
            date_count = 0
            asset_count = 0
            non_null = valid_rows
        partition_metrics = {
            "physical_row_count": total_rows,
            "date_count": date_count,
            "asset_count": asset_count,
            "cell_count": total_rows,
            "non_null_cell_count": non_null,
            "valid_rows": valid_rows,
            "file_bytes": file_bytes,
        }
        # R39 PERF-051: full-history date range from merged partition stats —
        # the normal path needs no get_watermark round-trip to compute it.
        starts = [r["min_date"] for r in merged.values() if r.get("min_date")]
        ends = [r["max_date"] for r in merged.values() if r.get("max_date")]
        has_legacy = any(
            not r.get("min_date") or not r.get("max_date") for r in merged.values()
        )
        wm_range = {
            "start_date": min(starts) if starts else None,
            "end_date": max(ends) if ends else None,
            "has_legacy": has_legacy,
        }
        return partition_metrics, total_rows, False, wm_range

    def _bump_full_rescan(self) -> None:
        """Increment the full-history-rescan hard-gate counter (Gate-05)."""
        global full_factor_rescan_count

        self.full_factor_rescan_count += 1
        full_factor_rescan_count += 1
        try:
            from runtime.perf_counters import get_global_counters

            get_global_counters().incr("full_factor_rescan_count")
        except Exception:
            pass

    def _bump_post_write_readback(self) -> None:
        """Increment the post-write watermark read-back counter (R39 PERF-051).

        Only genuine read-backs count: the write path must re-read the committed
        watermark when it cannot know the new value (watermark_deferred /
        external-modification scenarios).  The normal path returns the pending
        watermark the transaction just committed and never bumps this.
        """
        global post_write_watermark_readback_count

        self.post_write_watermark_readback_count += 1
        post_write_watermark_readback_count += 1

    @staticmethod
    def _watermark_result_from_pending(factor_id: str, pending: dict) -> dict:
        """Build a ``get_watermark``-shaped result dict from a pending watermark
        this transaction just committed (R39 PERF-051).

        The numerical/date semantics are identical to a read-back.  ``last_updated``
        is the wall-clock captured at commit time — tests compare it separately
        (it is a wall-clock, not watermark semantics).
        """
        return {
            "factor_id": factor_id,
            "start_date": pending["start_date"],
            "end_date": pending["end_date"],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "row_count": pending.get("row_count"),
        }

    def _materialize_long_df(
        self,
        factor_id: str,
        df: pd.DataFrame,
        *,
        meta: "MaterializeMetadata | None",
        dq_report=None,
        author: str,
        frequency: str,
        ast_hash: str,
        description: str | None = None,
        expression: str | None = None,
        run_lineage: dict | None = None,
        write_metadata: bool = True,
        data_snapshot_id: str | None = None,
        data_source_config: dict | None = None,
        isolate_partition_failures: bool = True,
        resume: bool = False,
        preserve_invalid_rows: bool = False,
        write_target: str = "local",
        defer_watermark: bool = False,
        partition_columns: list[str] | None = None,
        storage_format: str = "long",
        null_overwrite: bool = False,
        deleted_keys: list[tuple] | None = None,
        production: bool | None = None,
        run_generation: str | None = None,
        force_tombstones: bool | None = None,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
        delta_mode: bool | None = None,
        effective_value_dtype: str,
        precision_policy: str,
        identity_digest: str | None,
        lineage_extra: dict,
        source_snapshot: str | None,
        source_dep_hash: str | None,
        generation: str,
    ) -> dict:
        """Shared post-normalization materialize core (used by both the Series
        ``materialize`` path and the Arrow ``materialize_block`` path).

        The body is the historical tail of ``materialize`` (kept byte-identical
        in behavior) plus R39 PERF-048/049/059/044 wiring: each partition commit
        returns :class:`PartitionCommitStats`, aggregate stats are recorded
        incrementally in the catalog, the watermark row count is derived from
        those stats (never a full-history value-cell re-scan), and write
        amplification is tracked.
        """
        from storage.write_targets import normalize_write_target

        # R39: resolve delta-mode activation once for this factor write.
        delta_mode_eff = self._resolve_delta_mode(delta_mode)
        if delta_mode_eff and self._lock_manager is None:
            from storage.delta_store import PartitionLockManager

            self._lock_manager = PartitionLockManager()

        # Review-8 #444/#446: 声明 tombstones 的调用方把历史 key 标记为已删除
        # （value=NaN 覆盖旧值）。deleted_keys 行必须在 _clean 前注入，与普通 NaN
        # 一起走同一 upsert 路径，才能覆盖该 key 的旧有限值。
        tombstoned_rows = 0
        if deleted_keys:
            null_overwrite = True
            df = self._append_tombstones(df, deleted_keys, metadata=meta)
            tombstoned_rows = len(deleted_keys)

        # --- 2. 数据清洗 ---
        df = self._clean(
            df,
            preserve_invalid_rows=preserve_invalid_rows,
            null_overwrite=null_overwrite,
        )

        # R9-P0-027: NaN rows are kept as explicit tombstones (is_valid=0), so a
        # clean result is no longer empty just because every value is NaN.
        #
        # R10 #48: production incremental must distinguish:
        #   * "no computation"  -> genuinely empty frame (no rows) -> skip;
        #   * "computed invalid" -> rows exist but ALL values are NaN/Inf and this
        #     is a production incremental recompute -> write tombstones
        #     (is_valid=0) so stale finite values in the partition are CLEARED,
        #     without requiring the caller to pass null_overwrite=True;
        #   * "deleted" -> explicit deleted_keys (already flips null_overwrite).
        has_valid_value = bool(df["value"].notna().any())
        force_tombstones = self._resolve_force_tombstones(
            force_tombstones, production, run_lineage
        )
        if write_mode == "recompute_window" and not df.empty and not has_valid_value:
            force_tombstones = True
            null_overwrite = True
        if df.empty or (
            not has_valid_value and not null_overwrite and not force_tombstones
        ):
            logger.warning("因子 '%s' 清洗后无有效数据，跳过落盘。", factor_id)
            return {
                "factor_id": factor_id,
                "rows_written": 0,
                "partitions": [],
                "watermark": None,
                "write_target": write_target,
                "force_tombstones": force_tombstones,
                "identity_digest": identity_digest,
                "run_generation": generation,
            }

        try:
            target_flags = normalize_write_target(write_target)
        except ValueError:
            raise
        write_target_flags = target_flags
        target = str(write_target or "local").lower()
        write_local = bool(write_target_flags["local"])
        write_staging = bool(write_target_flags["staging"])
        write_clickhouse = bool(write_target_flags["clickhouse"])
        if write_target_flags.get("staging_dataset"):
            self._staging_dataset = str(write_target_flags["staging_dataset"])

        watermark_deferred = bool(defer_watermark)
        if write_staging and not write_local and not write_clickhouse:
            watermark_deferred = True
            if production:
                logger.warning(
                    "factor_id=%s write_target=%s 是 staging-only，权威水位线已 defer"
                    "（publish 成功后由 publish_factor_lake 推进）",
                    factor_id,
                    target,
                )

        # --- 3. 注册 / Hash 校验（可能抛 FactorHashMismatchError）---
        self._catalog.register(
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            description=description,
            expression=expression,
            data_source_config=data_source_config,
            semantic_identity_digest=identity_digest,
            production=production,
        )

        staging_result = None
        policy = PartitionPolicy.from_config(
            partition_columns=partition_columns,
            storage_format=storage_format,
        )
        if write_staging:
            staging_result = self._upsert_to_data_access_staging(
                factor_id, df, policy=policy
            )

        partitions_written: list[int] = []
        partitions_failed: list[int] = []
        partitions_skipped: list[int] = []
        partition_keys_written: list[str] = []
        partition_keys_failed: list[str] = []
        partition_keys_skipped: list[str] = []

        from runtime.lineage import new_run_id

        checkpoint_run_id = (
            str(run_lineage.get("run_id"))
            if run_lineage and run_lineage.get("run_id")
            else new_run_id()
        )

        from security.factor_id import factor_dir_for

        factor_dir = factor_dir_for(self._lake_root, factor_id)
        work_df = attach_partition_columns(df, policy)

        # R39 PERF-049: per-partition commit stats collected during the loop so
        # the watermark path needs no full-factor value-cell re-scan.
        stats_by_partition: dict[str, PartitionCommitStats] = {}

        if write_local:
            # Phase 5 R17：直接迭代 generator，禁止 ``list(iter_partition_groups(...))``
            progress = ProgressLogger(
                logger,
                desc=f"落盘因子 {factor_id}",
                total=None,
                unit="partition",
            )
            for part_values, partition_df in iter_partition_groups(work_df, policy):
                pkey = partition_key(part_values)
                ck_year = checkpoint_year(part_values)
                partition_dir = factor_dir / partition_path_segments(
                    part_values, column_order=policy.columns
                )
                fp = checkpoint_fingerprint(
                    identity_digest=identity_digest,
                    source_snapshot=source_snapshot,
                    source_dependency_hash=source_dep_hash,
                    partition_input_fingerprint=partition_input_fingerprint(
                        partition_df
                    ),
                    run_generation=generation,
                )
                fp["storage_precision_policy"] = precision_policy
                fp["storage_value_dtype"] = effective_value_dtype
                if resume:
                    checkpoint = self._catalog.get_partition_checkpoint_by_key(
                        factor_id, pkey
                    )
                    if checkpoint is None and policy.columns == ("year",):
                        checkpoint = self._catalog.get_partition_checkpoint(
                            factor_id, ck_year
                        )
                    if self._checkpoint_skippable(
                        checkpoint,
                        fp,
                        production=production,
                        partition_dir=partition_dir,
                    ):
                        partitions_skipped.append(ck_year)
                        partition_keys_skipped.append(pkey)
                        progress.advance(detail=f"{pkey}, resume_skip")
                        continue

                try:
                    commit_stats = self._upsert_partition(
                        factor_dir,
                        part_values,
                        partition_df,
                        policy=policy,
                        write_mode=write_mode,
                        replace_window=replace_window,
                        delta_mode=delta_mode_eff,
                    )
                    if commit_stats is not None:
                        stats_by_partition[pkey] = commit_stats
                        # R39 PERF-044: write-amplification KPI per commit.
                        logical = int(partition_df.memory_usage(deep=True).sum())
                        self.write_amplification.record_partition_write(
                            logical_changed_bytes=logical,
                            physical_new_write_bytes=commit_stats.file_bytes,
                            historical_rewrite_bytes=(
                                0 if delta_mode_eff else commit_stats.file_bytes
                            ),
                            delta_file_count=1 if delta_mode_eff else 0,
                        )
                        _sync_module_wa_counters(self)
                    self._write_checkpoint_fingerprint_file(
                        partition_dir, fp, production=production, run_id=checkpoint_run_id
                    )
                    self._catalog.record_partition_checkpoint(
                        factor_id=factor_id,
                        partition_year=ck_year,
                        partition_key=pkey,
                        run_id=checkpoint_run_id,
                        status="success",
                    )
                    partitions_written.append(ck_year)
                    partition_keys_written.append(pkey)
                    progress.advance(detail=f"{pkey}, rows={len(partition_df)}")
                except Exception as exc:
                    self._catalog.record_partition_checkpoint(
                        factor_id=factor_id,
                        partition_year=ck_year,
                        partition_key=pkey,
                        run_id=checkpoint_run_id,
                        status="failed",
                        error_message=str(exc),
                    )
                    partitions_failed.append(ck_year)
                    partition_keys_failed.append(pkey)
                    progress.advance(detail=f"{pkey}, failed")
                    logger.error(
                        "分区落盘失败 factor_id=%s partition=%s error=%s",
                        factor_id,
                        pkey,
                        exc,
                    )
                    if not isolate_partition_failures:
                        raise

            # Review-8 #440/#441: failure adjudication FIRST.
            if partitions_failed:
                if run_lineage is not None and not watermark_deferred:
                    lineage_payload = dict(run_lineage)
                    if dq_report is not None:
                        lineage_payload["dq_passed"] = dq_report.passed
                    lineage_payload.setdefault("extra", {})
                    if isinstance(lineage_payload["extra"], dict):
                        lineage_payload["extra"]["partitions_failed"] = partitions_failed
                    self._catalog.record_run(lineage_payload)
                raise MaterializePartitionError(
                    f"因子 '{factor_id}' 分区落盘部分失败: {partitions_failed}"
                )

            if not partitions_written and not partitions_skipped:
                return {
                    "factor_id": factor_id,
                    "rows_written": 0,
                    "partitions": [],
                    "partition_keys": [],
                    "partitions_failed": partitions_failed,
                    "partitions_skipped": partitions_skipped,
                    "watermark": self._catalog.get_watermark(factor_id),
                    "checkpoint_run_id": checkpoint_run_id,
                    "write_target": write_target,
                    "storage_format": policy.storage_format,
                }

        # R39 PERF-049: record per-partition aggregate stats incrementally in one
        # batch catalog commit (no full-history re-scan needed downstream).
        if write_local and stats_by_partition:
            with self._catalog.batch_transaction(generation) as tx:
                tx.update_partition_stats_many(
                    [
                        {
                            "factor_id": factor_id,
                            "partition_key": k,
                            "rows": s.rows_after,
                            "valid_rows": s.valid_after,
                            "min_date": s.min_date,
                            "max_date": s.max_date,
                            "file_bytes": s.file_bytes,
                        }
                        for k, s in stats_by_partition.items()
                    ]
                )

        # --- 5. 更新水位线（R9-P0-026: commit watermark LAST）---
        value_columns = [c for c in work_df.columns if c not in policy.columns]
        if write_local and (partitions_written or partitions_skipped):
            active_keys = set(partition_keys_written) | set(partition_keys_skipped)

            def _row_partition_key(row: pd.Series) -> str:
                return partition_key({col: row[col] for col in policy.columns})

            active_df = work_df.loc[
                work_df.apply(_row_partition_key, axis=1).isin(active_keys),
                value_columns,
            ]
        else:
            active_df = work_df[value_columns]

        start_date = active_df["datetime"].min().isoformat()
        end_date = active_df["datetime"].max().isoformat()

        if write_local:
            # R39 PERF-048/049: derive total_rows + partition metrics from the
            # in-memory commit stats (never a full-history value-cell re-scan).
            partition_metrics, total_rows, _rescanned, wm_range = (
                self._compute_watermark_metrics(
                    factor_id, factor_dir, stats_by_partition, active_df
                )
            )
            # R39 PERF-051: when the post-write full-history range is derivable
            # from partition stats (normal path — no legacy partitions), merge it
            # with the in-flight range WITHOUT a get_watermark round-trip.
            if (
                wm_range.get("start_date") is not None
                and wm_range.get("end_date") is not None
                and not wm_range.get("has_legacy")
            ):
                if wm_range["start_date"] < start_date:
                    start_date = wm_range["start_date"]
                if wm_range["end_date"] > end_date:
                    end_date = wm_range["end_date"]
            else:
                # Legacy on-disk partitions predate incremental stats (no date
                # info), or no stats anywhere: preserve the committed range via
                # a read of the existing watermark (pre-write, not a read-back).
                existing_wm = self._catalog.get_watermark(factor_id)
                if existing_wm is not None:
                    if existing_wm["start_date"] < start_date:
                        start_date = existing_wm["start_date"]
                    if existing_wm["end_date"] > end_date:
                        end_date = existing_wm["end_date"]
        else:
            total_rows = len(active_df)
            partition_metrics = {
                "physical_row_count": total_rows,
                "date_count": int(active_df["datetime"].nunique()),
                "asset_count": int(active_df["asset"].nunique()),
                "cell_count": total_rows,
                "non_null_cell_count": int(active_df["value"].notna().sum()),
            }
            # No local partition stats to derive the range from (staging-only
            # writes): merge with the committed watermark via a pre-write read.
            existing_wm = self._catalog.get_watermark(factor_id)
            if existing_wm is not None:
                if existing_wm["start_date"] < start_date:
                    start_date = existing_wm["start_date"]
                if existing_wm["end_date"] > end_date:
                    end_date = existing_wm["end_date"]

        pending_watermark = {
            "start_date": start_date,
            "end_date": end_date,
            "row_count": total_rows,
        }
        pending_lineage = None

        if watermark_deferred:
            # The transaction did NOT commit the watermark (deferred to
            # publish); the committed value is unknown to us → genuine read-back.
            self._bump_post_write_readback()
            watermark = self._catalog.get_watermark(factor_id)
            if run_lineage is not None:
                pending_lineage = dict(run_lineage)
                if dq_report is not None:
                    pending_lineage["dq_passed"] = dq_report.passed
        else:
            # (3) lineage + checkpoint cleanup first ...
            with self._catalog.batch_transaction(generation) as tx:
                if run_lineage is not None:
                    lineage_payload = dict(run_lineage)
                    if dq_report is not None:
                        lineage_payload["dq_passed"] = dq_report.passed
                    tx.record_runs_many([lineage_payload])
                    if write_local:
                        tx.execute_many(
                            "DELETE FROM factor_materialize_checkpoint WHERE factor_id = ?",
                            [(factor_id,)],
                        )
                # (4) commit watermark LAST — after every required step succeeded.
                tx.update_watermarks_many([(factor_id, start_date, end_date, total_rows)])
            # R39 PERF-051: this transaction knows exactly what it just wrote —
            # return the pending watermark directly instead of a redundant
            # post-write get_watermark query.
            watermark = self._watermark_result_from_pending(
                factor_id, pending_watermark
            )

        rows_written = (
            len(df)
            if (preserve_invalid_rows or null_overwrite or force_tombstones)
            else int(df["value"].notna().sum())
        )

        logger.info(
            "因子 '%s' 落盘完成：%d 行，target=%s，分区 %s，跳过 %s，水位线 [%s → %s]",
            factor_id,
            rows_written,
            target,
            partitions_written,
            partitions_skipped,
            start_date,
            end_date,
        )

        summary = {
            "factor_id": factor_id,
            "rows_written": rows_written,
            "partitions": partitions_written,
            "partition_keys": partition_keys_written,
            "partitions_failed": partitions_failed,
            "partition_keys_failed": partition_keys_failed,
            "partitions_skipped": partitions_skipped,
            "partition_keys_skipped": partition_keys_skipped,
            "watermark": watermark,
            "dq_report": dq_report.to_dict() if dq_report is not None else None,
            "run_id": run_lineage.get("run_id") if run_lineage else None,
            "checkpoint_run_id": checkpoint_run_id,
            "write_target": write_target,
            "preserve_invalid_rows": preserve_invalid_rows,
            "null_overwrite": null_overwrite,
            "tombstoned_rows": tombstoned_rows,
            "watermark_deferred": watermark_deferred,
            "storage_format": policy.storage_format,
            "partition_columns": list(policy.columns),
            "force_tombstones": force_tombstones,
            "identity_digest": identity_digest,
            "run_generation": generation,
            "partition_metrics": partition_metrics,
            "delta_mode": delta_mode_eff,
            "write_amplification": self.write_amplification.snapshot().to_dict(),
        }
        if pending_watermark is not None and watermark_deferred:
            summary["pending_watermark"] = pending_watermark
        if pending_lineage is not None:
            summary["pending_lineage"] = pending_lineage
        if staging_result is not None:
            summary["staging"] = staging_result
        return summary

    def commit_deferred_materialization(self, summary: dict) -> dict:
        """双写成功：提交延迟的水位线与 lineage。
        
        参数:
            summary: 见函数签名
        
        返回:
            dict
        """
        if not summary.get("watermark_deferred"):
            return summary
        factor_id = summary["factor_id"]
        pending = summary.get("pending_watermark") or {}
        # R9-P0-026: record lineage + clear checkpoints first, commit watermark
        # LAST — the watermark must only advance after every required step
        # (including the deferred dependency write) has succeeded.
        pending_lineage = summary.get("pending_lineage")
        if pending_lineage is not None:
            self._catalog.record_run(pending_lineage)
        partitions = summary.get("partitions") or []
        if partitions:
            self._catalog.clear_partition_checkpoints(factor_id)
        self._catalog.update_watermark(
            factor_id=factor_id,
            start_date=pending["start_date"],
            end_date=pending["end_date"],
            row_count=pending.get("row_count"),
        )
        merged = dict(summary)
        # R39 PERF-051: this method just committed the pending watermark — return
        # it directly instead of a redundant post-write get_watermark query.
        merged["watermark"] = self._watermark_result_from_pending(
            factor_id, pending
        )
        merged["watermark_deferred"] = False
        merged.pop("pending_watermark", None)
        merged.pop("pending_lineage", None)
        merged["dual_write_committed"] = True
        logger.info("因子 '%s' 延迟水位线已提交", factor_id)
        return merged

    def abort_deferred_materialization(
        self,
        summary: dict,
        *,
        error: str,
        downstream: str = "clickhouse",
    ) -> dict:
        """双写失败：记录失败 run，不更新水位线。
        
        参数:
            summary: 见函数签名
            error: 见函数签名（可选）
            downstream: 见函数签名（可选）
        
        返回:
            dict
        """
        if not summary.get("watermark_deferred"):
            return summary
        factor_id = summary["factor_id"]
        pending_lineage = dict(summary.get("pending_lineage") or {})
        extra = dict(pending_lineage.get("extra") or {})
        extra.update(
            {
                "dual_write_failed": True,
                "dual_write_downstream": downstream,
                "dual_write_error": error,
                "staging_written": summary.get("staging") is not None,
            }
        )
        pending_lineage["extra"] = extra
        pending_lineage["dq_passed"] = False
        if pending_lineage.get("run_id"):
            self._catalog.record_run(pending_lineage)
        merged = dict(summary)
        merged["dual_write_committed"] = False
        merged["dual_write_aborted"] = True
        logger.error(
            "因子 '%s' 双写中止（%s 失败），水位线未更新: %s",
            factor_id,
            downstream,
            error,
        )
        return merged

    # ------------------------------------------------------------------
    # 内部：数据规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_long_table(
        result: pd.Series,
        *,
        metadata: MaterializeMetadata | None = None,
        value_dtype: str = "float32",
    ) -> pd.DataFrame:
        """MultiIndex Series → 长表 DataFrame[datetime, asset, value(float32)]。
        
        参数:
            result: 因子计算结果 Series
            metadata: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
        
        返回:
            pd.DataFrame
        
        
        Raises
                ------
                ValueError
                    若 ``result`` 不是 MultiIndex Series 或索引层级不足。
        """
        if not isinstance(result, pd.Series):
            raise ValueError(
                f"期望 pd.Series，实际得到 {type(result).__name__}。"
            )
        if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels < 2:
            raise ValueError(
                "期望 MultiIndex(timestamp, instrument) Series，"
                f"实际索引层级数为 {getattr(result.index, 'nlevels', 1)}。"
            )
        # R32-P0-025: OutputGrainContract —— 超过两层（datetime, instrument）的
        # 额外 grain 禁止静默丢弃。minute/session/event/relation 语义有专门 schema，
        # 统一长表物化必须精确保持两层。
        nlevels = int(result.index.nlevels)
        if nlevels > OUTPUT_GRAIN_MAX_NLEVELS:
            extra = [
                str(getattr(result.index, "names", None) or [])[i] or f"<level{i}>"
                for i in range(2, nlevels)
            ]
            raise ValueError(
                f"OutputGrainContract violation: MultiIndex has {nlevels} levels "
                f"(expected exactly 2: timestamp × instrument). Extra grain levels "
                f"{extra} would be silently dropped by the factor-lake long "
                f"materializer — minute/session/event/relation outputs must use "
                f"their dedicated schema, not be down-sampled. Got index names: "
                f"{list(result.index.names)}."
            )

        df = result.reset_index()
        # 统一列名：前两级索引 → datetime, asset；值列 → value
        cols = list(df.columns)
        df.columns = ["datetime", "asset"] + cols[2:]
        # 只保留最后一个值列并重命名
        value_col = df.columns[-1]
        df = df[["datetime", "asset", value_col]].rename(columns={value_col: "value"})

        # 强制类型——剥夺 Pandas 的自动推断权
        df["asset"] = df["asset"].astype("string")
        df["value"] = df["value"].astype(str(value_dtype or "float32"))
        if metadata is not None:
            df["calc_time"] = metadata.calc_time
            df["factor_version"] = metadata.factor_version
            df["data_snapshot_id"] = metadata.data_snapshot_id or ""
            df["resolved_snapshot_id"] = metadata.resolved_snapshot_id or ""
            df["storage_precision_policy"] = metadata.storage_precision_policy or ""
            df["is_valid"] = int(metadata.is_valid)
            df["invalid_reason"] = metadata.invalid_reason or ""
        return df

    @staticmethod
    def _clean(
        df: pd.DataFrame,
        *,
        preserve_invalid_rows: bool = False,
        null_overwrite: bool = False,
    ) -> pd.DataFrame:
        """清洗：inf → NaN；NaN/Inf 行作为显式 tombstone 保留（is_valid=0）。

        R9-P0-027: 默认不再 ``dropna``。Tombstone 标记 = ``value=NaN`` +
        ``is_valid=0`` + ``invalid_reason="inf_or_nan"``：该行保留在分区 Parquet
        里，下游 ``_upsert_partition`` 的 ``[datetime, asset]`` dedup keep="last"
        会用 NaN 覆盖旧有限值，读取端把该格映射为 NaN/invalid，从而支持
        valid→null / valid→deleted / 退出 universe / 源修订删行。Parquet 与
        staging 均能保存 NaN，因此直接以 NaN 为标记（若某存储层无法保存 NaN，
        可改用一个固定浮点 sentinel 并在读取端映射回 NaN）。

        ``preserve_invalid_rows`` / ``null_overwrite`` 保留仅为 API 兼容：
        无论取值如何，NaN/Inf 行都会被保留并标注（历史默认 dropna 的
        valid→NaN 修订会让旧有限值永久残留）。

        参数:
            df: 输入 DataFrame
            preserve_invalid_rows: 见函数签名（可选）
            null_overwrite: 见函数签名（可选）

        返回:
            pd.DataFrame
        """
        df = df.copy()
        df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
        if "is_valid" not in df.columns:
            df["is_valid"] = 1
        if "invalid_reason" not in df.columns:
            df["invalid_reason"] = ""
        invalid = df["value"].isna()
        df.loc[invalid, "is_valid"] = 0
        df.loc[invalid, "invalid_reason"] = "inf_or_nan"
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # R10 #16/#17/#47/#48：production 判定 + checkpoint 身份指纹
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_production(production: bool | None) -> bool:
        """解析 production 标志：显式值优先，否则从运行模式推断。"""
        if production is not None:
            return bool(production)
        from runtime.production_policy import is_production_mode

        return is_production_mode()

    @staticmethod
    def _resolve_force_tombstones(
        force_tombstones: bool | None,
        production: bool,
        run_lineage: dict | None,
    ) -> bool:
        """解析是否对全 NaN 结果写 tombstone（R10 #48）。

        显式 ``force_tombstones`` 优先；否则 production incremental
        （``run_lineage.extra.mode == "incremental"``）自动写 tombstone。
        """
        if force_tombstones is not None:
            return bool(force_tombstones)
        lineage_extra = (run_lineage or {}).get("extra") or {}
        return bool(production) and lineage_extra.get("mode") == "incremental"

    @staticmethod
    def _checkpoint_skippable(
        checkpoint: dict | None,
        fingerprint: dict[str, str],
        *,
        production: bool,
        partition_dir: Path,
    ) -> bool:
        """R10 #47: resume 时判定是否可跳过该分区。

        - ``status != success`` → 不可跳过（重算）；
        - 有指纹文件且全部组件匹配 → 可跳过；
        - 无指纹文件（旧 checkpoint）→ production 必须重算；research 保留旧行为
          （``status == success`` 即跳过）。

        R20-201..206: storage precision policy 是 resume 判定的一部分——旧分区
        用 float32 写的指纹在新 precision policy 下不匹配 → 必须重算。
        """
        if not checkpoint or checkpoint.get("status") != "success":
            return False
        stored = ParquetMaterializer._read_checkpoint_fingerprint_file(partition_dir)
        if stored is None:
            return not production
        if not checkpoint_fingerprint_matches(stored, fingerprint):
            return False
        # 补充精度一致性：身份 digest 变化前先显式比对 precision policy。
        stored_policy = str(stored.get("storage_precision_policy") or "")
        current_policy = str(fingerprint.get("storage_precision_policy") or "")
        if stored_policy and current_policy and stored_policy != current_policy:
            return False
        return True

    @staticmethod
    def _checkpoint_fingerprint_path(partition_dir: Path) -> Path:
        return Path(partition_dir) / ".identity.json"

    @staticmethod
    def _write_checkpoint_fingerprint_file(
        partition_dir: Path,
        fingerprint: dict[str, str],
        *,
        production: bool = False,
        run_id: str | None = None,
    ) -> None:
        """原子写入分区身份指纹 sidecar（``.identity.json``）。

        R32-P0-033: tmp 文件名使用 uuid/run_id（不再只用 PID —— 线程内会碰撞，
        两个并发写同一分区会互相覆盖 tmp）。R32-P0-032: production 下 sidecar
        写失败必须硬失败（抛错），绝不把「数据写成功但身份 sidecar 失败」标成
        完整成功 —— checkpoint success 必须在 sidecar durable 之后提交。
        """
        import uuid

        path = ParquetMaterializer._checkpoint_fingerprint_path(partition_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f".{path.name}.{run_id or uuid.uuid4().hex[:12]}.tmp"
        )
        try:
            with open(str(tmp), "w", encoding="utf-8") as fh:
                json.dump(fingerprint, fh, sort_keys=True, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(path))
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
                finally:
                    os.close(dir_fd)
        except OSError as exc:
            if production:
                raise
            logger.debug(
                "写入 checkpoint 指纹失败 partition_dir=%s", partition_dir, exc_info=True
            )

    @staticmethod
    def _read_checkpoint_fingerprint_file(partition_dir: Path) -> dict | None:
        """读取分区身份指纹；缺失/损坏返回 None（视为旧 checkpoint）。"""
        path = ParquetMaterializer._checkpoint_fingerprint_path(partition_dir)
        try:
            if not path.exists():
                return None
            with open(str(path), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _append_tombstones(
        df: pd.DataFrame,
        deleted_keys: list[tuple],
        *,
        metadata: "MaterializeMetadata | None" = None,
    ) -> pd.DataFrame:
        """把显式删除的 (datetime, asset) 键转成 value=NaN 行（Review-8 #446）。

        这些行随后走同一 upsert 路径：dedup keep="last" 用 NaN 覆盖该键的旧
        有限值，读取时返回 NaN（= 该历史值已被删除/退出 universe）。

        R32-P0-034: deleted_keys-only 输入（df 可能为空）时，metadata 列必须
        从当前 ``MaterializeMetadata`` 显式生成 —— 不能用 ``df.iloc[0]``（空 df
        下返回 None，导致删除 tombstone 缺 run metadata / factor_version /
        snapshot identity）。
        """
        rows = []
        for key in deleted_keys:
            if not isinstance(key, (tuple, list)) or len(key) < 2:
                raise ValueError(
                    f"deleted_keys entries must be (datetime, asset) pairs, got {key!r}"
                )
            dt = pd.Timestamp(key[0])
            asset = str(key[1])
            rows.append({"datetime": dt, "asset": asset, "value": np.nan})
        tomb = pd.DataFrame(rows)
        for col in df.columns:
            if col not in tomb.columns:
                tomb[col] = df[col].iloc[0] if len(df) else None
        if metadata is not None:
            tomb["calc_time"] = metadata.calc_time
            tomb["factor_version"] = metadata.factor_version
            tomb["data_snapshot_id"] = metadata.data_snapshot_id or ""
            tomb["resolved_snapshot_id"] = metadata.resolved_snapshot_id or ""
            tomb["storage_precision_policy"] = metadata.storage_precision_policy or ""
            tomb["is_valid"] = int(metadata.is_valid)
            tomb["invalid_reason"] = metadata.invalid_reason or ""
        # match the numeric dtype so the concat keeps df's column dtype
        tomb["value"] = tomb["value"].astype(df["value"].dtype)
        return pd.concat([df, tomb], ignore_index=True)

    def _upsert_to_data_access_staging(
        self,
        factor_id: str,
        df: pd.DataFrame,
        *,
        policy: PartitionPolicy | None = None,
    ) -> dict:
        """经 data_access 幂等 upsert 到 staging 数据集（生产发布前暂存区）。
        
        参数:
            factor_id: 因子唯一标识
            df: 输入 DataFrame
            policy: 分区策略（可选）
        
        返回:
            dict
        """
        from storage.write_targets import resolve_write_target

        policy = policy or PartitionPolicy.from_config()
        out = attach_partition_columns(df, policy)
        target = resolve_write_target("staging", staging_dataset=self._staging_dataset)
        result = target.write_factor_frame(
            factor_id,
            out,
            upsert_on=["datetime", "asset"],
            partition_by=list(policy.columns),
        )
        logger.info(
            "因子 '%s' 已 upsert 到 %s: %s",
            factor_id,
            self._staging_dataset,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # 内部：分区 Upsert + 原子写入
    # ------------------------------------------------------------------

    @staticmethod
    @contextlib.contextmanager
    def _partition_lock(partition_dir: Path):
        """按分区目录加进程级 advisory ``flock``（Review-8 #443）。

        两个 worker 同时 upsert 同一分区时，先到者持锁完成 read→merge→replace，
        后到者阻塞在锁上，保证不会 ``A 读旧、B 读旧、A 写、B 写`` 丢失更新。
        Linux/macOS 可用 ``fcntl``；其他平台退化为 no-op。
        """
        lock_path = Path(partition_dir) / ".lock"
        fh = open(str(lock_path), "a+")  # noqa: SIM115 - lifecycle tied to context
        try:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            fh.close()

    def _write_partition_atomic(
        self,
        partition_dir: Path,
        *,
        filename: str,
        combined_df: pd.DataFrame | None = None,
        panel: pd.DataFrame | None = None,
    ) -> Path:
        """原子写入分区 parquet（Review-8 #442/#443）。

        * temp 文件名带唯一 run 后缀，多进程并发写同分区也不会互相覆盖 tmp；
        * 文件 fsync 后再 ``os.replace``，再 fsync 目录 —— 崩溃后文件要么是
          旧的完整版本、要么是新的完整版本，绝不半写；
        * 返回正式 parquet 路径。
        """
        import uuid

        if (combined_df is None) == (panel is None):
            raise ValueError("exactly one of combined_df/panel must be provided")
        parquet_path = partition_dir / filename
        tmp_path = partition_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        if panel is not None:
            panel.to_parquet(tmp_path, index=True, engine="pyarrow")
        else:
            combined_df.to_parquet(tmp_path, index=False, engine="pyarrow")
        # fsync 数据文件，确保 os.replace 后内容已落盘
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp_path), str(parquet_path))
        # fsync 目录，持久化 rename 本身
        try:
            dir_fd = os.open(str(partition_dir), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
        return parquet_path

    @staticmethod
    def _cleanup_orphan_tmp_files(factor_dir: Path) -> None:
        """删除目标因子目录下遗留的孤儿 ``.tmp`` 文件（Review-8 #442）。

        崩溃发生在 ``os.replace`` 之前时，唯一的 ``.uuid.tmp`` 文件会残留；
        reconcile 时清理，避免磁盘膨胀。只在目标 factor_dir 内扫描（便宜）。
        """
        if not factor_dir.exists():
            return
        for tmp in factor_dir.rglob(".*.tmp"):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _upsert_partition(
        self,
        factor_dir: Path,
        part_values: dict[str, Any],
        new_df: pd.DataFrame,
        *,
        policy: PartitionPolicy,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
        delta_mode: bool | None = None,
    ) -> PartitionCommitStats | None:
        """对指定 hive 分区做幂等 Upsert（long 或 wide）。

        ``write_mode``（R20-230）：
          - ``upsert``：默认，``[datetime, asset]`` dedup keep="last"（新值覆盖旧值）；
          - ``append``：不覆盖既有 key——冲突时保留旧行（dedup keep="first"）；
          - ``replace_window``：先删除分区内 ``[start, end]`` 窗口的旧行，再写入新行；
          - ``recompute_window``：与 upsert 相同（all-NaN 覆盖旧 finite 由上层
            强制 tombstone 保证）。

        R39 delta-mode（``delta_mode`` 开启）：调用
        :meth:`_upsert_partition_delta` 写 immutable delta fragment（PERF-043/
        052/083），不重写历史分区。

        参数:
            factor_dir: 见函数签名
            part_values: 见函数签名
            new_df: 见函数签名
            policy: 分区策略（可选）
            write_mode: 写入模式（upsert/append/replace_window/recompute_window）
            replace_window: 替换窗口 (start, end)（ISO 字符串）
            delta_mode: 是否启用 delta 存储（可选，缺省从构造/环境解析）

        返回:
            PartitionCommitStats | None —— 每次 commit 返回提交统计
            （R39 PERF-049），供上层增量聚合，无需重扫全因子历史。
        """
        if policy.is_wide:
            return self._upsert_partition_wide(
                factor_dir, part_values, new_df, policy=policy,
                write_mode=write_mode, replace_window=replace_window,
            )

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        if self._resolve_delta_mode(delta_mode):
            return self._upsert_partition_delta(
                partition_dir,
                part_values,
                new_df,
                policy=policy,
                write_mode=write_mode,
            )

        parquet_path = partition_dir / policy.data_filename
        existing_rows = 0
        existing_orig: pd.DataFrame | None = None

        with self._partition_lock(partition_dir):
            # Review-8 #442: reconcile 本分区的孤儿 .tmp 文件。必须在锁内执行 —
            # 若在锁外扫整个 factor_dir，会误删其他进程正持有的 partition temp。
            self._cleanup_orphan_tmp_files(partition_dir)

            # 读取已有数据
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                existing_df["asset"] = existing_df["asset"].astype("string")
                # R32-P0-024: 禁止把已有 float64 历史静默 downcast 成 float32。
                # 已有分区若为 float64，新数据必须上转换到 float64 合并 ——
                # ``np.promote_types`` 只升不降，绝不损失历史精度。
                existing_value_dtype = existing_df["value"].dtype
                combined_dtype = np.promote_types(
                    existing_value_dtype, new_df["value"].dtype
                )
                existing_df["value"] = existing_df["value"].astype(combined_dtype)
                if new_df["value"].dtype != combined_dtype:
                    new_df = new_df.copy()
                    new_df["value"] = new_df["value"].astype(combined_dtype)
                for col in METADATA_COLUMNS:
                    if col not in existing_df.columns:
                        if col == "is_valid":
                            existing_df[col] = 1
                        elif col == "invalid_reason":
                            existing_df[col] = ""
                        else:
                            existing_df[col] = None
                existing_rows = len(existing_df)
                existing_orig = existing_df
                if write_mode == "replace_window" and replace_window is not None:
                    # 删除窗口 [start, end] 内的旧行，再与 new 拼接。
                    w_start = pd.Timestamp(replace_window[0])
                    w_end = pd.Timestamp(replace_window[1])
                    keep = ~(
                        (existing_df["datetime"] >= w_start)
                        & (existing_df["datetime"] <= w_end)
                    )
                    existing_df = existing_df.loc[keep]
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df

            # 去重：按 [datetime, asset]。upsert/recompute -> 保留最新（新值覆盖）；
            # append -> 保留最先（既有 key 不被新数据覆盖）。
            keep_rule = "last" if write_mode != "append" else "first"
            combined = combined.drop_duplicates(
                subset=["datetime", "asset"], keep=keep_rule
            )

            # 排序：保证 Polars join_asof 的物理预排序要求
            combined = combined.sort_values(
                ["asset", "datetime"]
            ).reset_index(drop=True)

            # 原子写入：唯一 temp + fsync + rename + 目录 fsync
            self._write_partition_atomic(
                partition_dir,
                filename=policy.data_filename,
                combined_df=combined,
            )

        logger.info(
            "分区写入完成: factor_dir=%s, partition=%s, mode=%s, existing_rows=%d, "
            "incoming_rows=%d, final_rows=%d",
            factor_dir,
            partition_key(part_values),
            write_mode,
            existing_rows,
            len(new_df),
            len(combined),
        )

        # R39 PERF-059/049: single-pass DQ stats + commit stats from in-memory
        # frames (never a full-history disk re-read for accounting).
        if len(new_df):
            dq_stats = compute_write_pass_dq(
                new_df["value"].to_numpy(),
                index=pd.MultiIndex.from_arrays(
                    [new_df["datetime"], new_df["asset"]]
                ),
            )
        else:
            dq_stats = None
        file_bytes = int(parquet_path.stat().st_size)
        return compute_partition_commit_stats(
            partition_key(part_values),
            existing_df=existing_orig,
            new_df=new_df,
            combined_df=combined,
            file_bytes=file_bytes,
            dq_stats=dq_stats,
        )

    def _upsert_partition_delta(
        self,
        partition_dir: Path,
        part_values: dict[str, Any],
        new_df: pd.DataFrame,
        *,
        policy: PartitionPolicy,
        write_mode: str = "upsert",
        run_generation: str = "0",
    ) -> PartitionCommitStats:
        """R39 PERF-043/052/053/083: append-only delta upsert.

        The changed rows are written as a sorted immutable ``delta/gen_<seq>``
        fragment (never a full-history read→concat→dedup→sort→rewrite).  The
        generation manifest flips once per batch for durability.  Legacy
        monolithic partitions (no manifest) are migrated on first delta write by
        keeping ``data.parquet`` as the base file.
        """
        from storage.delta_store import (
            DeltaManifest,
            read_delta_partition,
            write_delta_fragment,
        )

        writer_id = getattr(self, "_delta_writer_id", "w0")
        run_gen = str(run_generation or "0")
        lock_mgr = self._lock_manager
        lock_ctx = (
            lock_mgr.partition_lock(partition_dir)
            if lock_mgr is not None
            else self._partition_lock(partition_dir)
        )
        with lock_ctx:
            manifest = DeltaManifest.load(partition_dir)
            if manifest is None:
                # Legacy monolithic → keep data.parquet as base (both layouts
                # remain readable), then append deltas on top.
                manifest = DeltaManifest(layout="delta", seq=0, base="data.parquet")
                legacy = partition_dir / "data.parquet"
                if legacy.exists():
                    base = pd.read_parquet(legacy)
                    manifest.base_rows = int(len(base))
                    manifest.merged_rows = int(len(base))
                    manifest.merged_valid_rows = int(
                        (base["is_valid"] == 1).sum()
                        if "is_valid" in base.columns
                        else len(base)
                    )
                manifest.save_atomic(partition_dir)

            rows_before = manifest.merged_rows
            valid_before = manifest.merged_valid_rows

            # Only changed rows are sorted + written (PERF-083).
            delta_entry = write_delta_fragment(
                partition_dir,
                new_df,
                manifest=manifest,
                run_generation=run_gen,
                writer_id=writer_id,
            )
            # PERF-084: record compaction debt = delta bytes awaiting compaction.
            self.write_amplification.record_compaction_debt(
                debt_bytes=manifest.delta_bytes()
            )

        file_bytes = int(delta_entry["bytes"])
        rows_added = int(delta_entry["rows"])
        # In delta mode we never read the whole history per write; running
        # unique-row counts are reconciled exactly at compaction.
        rows_after = rows_before + rows_added
        valid_after = valid_before + int(delta_entry["valid_rows"])
        min_date = max_date = None
        if len(new_df):
            dt = pd.to_datetime(new_df["datetime"])
            min_date = str(dt.min().isoformat())
            max_date = str(dt.max().isoformat())
        dq_stats = compute_write_pass_dq(
            new_df["value"].to_numpy(),
            index=pd.MultiIndex.from_arrays([new_df["datetime"], new_df["asset"]]),
        ) if len(new_df) else None
        return PartitionCommitStats(
            partition_key=partition_key(part_values),
            rows_before=rows_before,
            rows_after=rows_after,
            rows_added=rows_added,
            rows_replaced=0,
            valid_before=valid_before,
            valid_after=valid_after,
            min_date=min_date,
            max_date=max_date,
            file_bytes=file_bytes,
            dq_stats=dq_stats,
        )

    def recover_orphan_delta_tmp_files(self, factor_id: str | None = None) -> int:
        """R39 PERF-053: generation-recovery sweep of orphan ``.tmp`` delta/base
        fragments, run at startup / recovery instead of per-partition-write.

        Sweeps the whole lake (or a single factor) for ``delta/`` and ``base/``
        ``.tmp`` leftovers.  Returns the number of files removed.  In delta mode
        the per-partition-write path does NOT call the monolithic
        ``_cleanup_orphan_tmp_files`` — this is the single startup scan.
        """
        from storage.delta_store import recover_orphan_delta_tmp_files

        if factor_id is not None:
            from security.factor_id import factor_dir_for

            roots = [factor_dir_for(self._lake_root, factor_id)]
        else:
            factors_root = self._lake_root / "factors"
            roots = (
                [factors_root]
                if factors_root.is_dir()
                else [self._lake_root]
            )
        removed = 0
        for root in roots:
            for partition_dir in root.rglob("year=*"):
                if partition_dir.is_dir():
                    removed += recover_orphan_delta_tmp_files(partition_dir)
            if root.is_dir():
                removed += recover_orphan_delta_tmp_files(root)
        return removed

    def compact_partition(self, partition_dir: Path, **kwargs) -> dict:
        """R39 PERF-084: explicit compaction of a delta partition (resource
        controlled — caller decides when; not run on the production hot path).

        Produces a single sorted new base equal to the merged base+deltas.
        """
        from storage.delta_store import compact_partition as _compact

        result = _compact(partition_dir, **kwargs)
        # After compaction the manifest holds no deltas → debt drops to 0.
        from storage.delta_store import DeltaManifest

        man = DeltaManifest.load(partition_dir)
        try:
            debt = man.delta_bytes() if man is not None else 0
        except Exception:
            debt = 0
        self.write_amplification.record_compaction_debt(debt_bytes=debt)
        return result

    def _upsert_partition_wide(
        self,
        factor_dir: Path,
        part_values: dict[str, Any],
        new_df: pd.DataFrame,
        *,
        policy: PartitionPolicy,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
    ) -> PartitionCommitStats | None:
        """宽表 panel 分区 upsert（经 long 去重后再 pivot）。

        R32-P0-028: wide 路径与 long 语义对齐 —— 支持 append / replace_window /
        recompute_window，不再固定 keep-last upsert。R32-P0-024: 读取已有
        panel 时保留其精度，绝不把 float64 历史降为 float32。

        参数:
            factor_dir: 见函数签名
            part_values: 见函数签名
            new_df: 见函数签名
            policy: 分区策略（可选）
            write_mode: 写入模式（upsert/append/replace_window/recompute_window）
            replace_window: 替换窗口 (start, end)（ISO 字符串）

        返回:
            PartitionCommitStats | None（R39 PERF-049）
        """
        from storage.factor_format import pivot_long_to_wide, unpivot_wide_to_long

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / policy.data_filename

        value_cols = ["datetime", "asset", "value"]
        new_long = new_df[value_cols].copy()
        existing_orig: pd.DataFrame | None = None

        with self._partition_lock(partition_dir):
            self._cleanup_orphan_tmp_files(partition_dir)
            if parquet_path.exists():
                existing_panel = pd.read_parquet(parquet_path)
                if "datetime" in existing_panel.columns:
                    existing_panel = existing_panel.set_index("datetime")
                existing_long = unpivot_wide_to_long(existing_panel)
                existing_orig = existing_long
                # R32-P0-024: 宽表 unpivot 保留原精度，不与 new 拼接前降为 float32。
                if existing_long["value"].dtype != new_long["value"].dtype:
                    promoted = np.promote_types(
                        existing_long["value"].dtype, new_long["value"].dtype
                    )
                    existing_long["value"] = existing_long["value"].astype(promoted)
                    new_long = new_long.copy()
                    new_long["value"] = new_long["value"].astype(promoted)
                if write_mode == "replace_window" and replace_window is not None:
                    w_start = pd.Timestamp(replace_window[0])
                    w_end = pd.Timestamp(replace_window[1])
                    keep = ~(
                        (existing_long["datetime"] >= w_start)
                        & (existing_long["datetime"] <= w_end)
                    )
                    existing_long = existing_long.loc[keep]
                combined_long = pd.concat([existing_long, new_long], ignore_index=True)
            else:
                combined_long = new_long

            # R32-P0-028: keep 规则与 long 路径一致 —— append 保留旧 key，其余
            # 模式新值覆盖旧值（recompute_window 的 all-NaN 覆盖由上层 tombstone
            # 保证，与 long 语义完全一致）。
            keep_rule = "last" if write_mode != "append" else "first"
            combined_long = combined_long.drop_duplicates(
                subset=["datetime", "asset"], keep=keep_rule
            )
            panel = pivot_long_to_wide(combined_long)
            self._write_partition_atomic(
                partition_dir,
                filename=policy.data_filename,
                panel=panel,
            )

        logger.info(
            "宽表分区写入完成: factor_dir=%s, partition=%s, mode=%s, shape=%s",
            factor_dir,
            partition_key(part_values),
            write_mode,
            panel.shape,
        )

        # R39 PERF-049: wide commit stats from the long merge frames in memory.
        file_bytes = int(parquet_path.stat().st_size)
        return compute_partition_commit_stats(
            partition_key(part_values),
            existing_df=existing_orig,
            new_df=new_long,
            combined_df=combined_long,
            file_bytes=file_bytes,
            dq_stats=None,
        )

    @staticmethod
    def _count_partition_metrics(
        factor_dir: Path,
    ) -> dict[str, int]:
        """R32-P0-029: 统计因子目录的分解行数指标。

        long 与 wide 的语义不可混用（wide 一行 = 一个日期，long 一行 = 一个
        cell）。统一记录：
          - ``physical_row_count``：物理行数（long=cell 数，wide=日期数）；
          - ``date_count``：去重日期数；
          - ``asset_count``：去重资产数（long）/ 面板列数（wide）；
          - ``cell_count``：date × asset 理论格数（= 长表行数）；
          - ``non_null_cell_count``：有限值格数。
        """
        dates: set[Any] = set()
        assets: set[Any] = set()
        cells = 0
        non_null = 0
        physical_rows = 0
        if not factor_dir.exists():
            return {
                "physical_row_count": 0, "date_count": 0, "asset_count": 0,
                "cell_count": 0, "non_null_cell_count": 0,
            }
        for pq_file in factor_dir.rglob("*.parquet"):
            if pq_file.name.startswith("."):
                continue
            try:
                df = pd.read_parquet(pq_file)
            except Exception:
                continue
            if df.empty:
                continue
            physical_rows += len(df)
            if {"datetime", "asset"}.issubset(df.columns):
                dates |= set(pd.to_datetime(df["datetime"]).dt.normalize())
                assets |= set(df["asset"].astype(str))
                cells += len(df)
                non_null += int(df["value"].notna().sum())
            else:
                # wide panel：index=datetime，columns=asset。
                dates |= set(pd.to_datetime(df.index).normalize())
                assets |= {str(c) for c in df.columns}
                cells += int(df.shape[0] * df.shape[1])
                non_null += int(df.notna().to_numpy().sum())
        return {
            "physical_row_count": physical_rows,
            "date_count": len(dates),
            "asset_count": len(assets),
            "cell_count": cells,
            "non_null_cell_count": non_null,
        }

    @staticmethod
    def _count_total_rows(factor_dir: Path) -> int:
        """统计因子目录下所有分区 Parquet 的总行数（long 语义物理 cell 行数）。

        R32-P0-029: 具体行数/日期/资产/格数分解见 :meth:`_count_partition_metrics`；
        本方法保留历史 ``row_count`` 语义（long=cell 行数）供 watermark 兼容。
        """
        return ParquetMaterializer._count_partition_metrics(factor_dir)[
            "physical_row_count"
        ]

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------

    def delete_factor(self, factor_id: str, *, delete_files: bool = False) -> None:
        """从 Catalog 删除因子。可选同时删除物理文件。

        R32-P0-036/060: delete_files 同样走 FactorId domain gate + resolve-under-root
        —— 删除路径越界是灾难，绝不直接拼 ``lake_root / "factors" / factor_id``。

        参数:
            factor_id: 因子唯一标识
            delete_files: 见函数签名（可选）

        返回:
            无
        """
        from security.factor_id import factor_dir_for, validate_factor_id

        factor_id = validate_factor_id(factor_id)
        self._catalog.delete_factor(factor_id)
        if delete_files:
            import shutil

            factor_dir = factor_dir_for(self._lake_root, factor_id)
            if factor_dir.exists():
                shutil.rmtree(factor_dir)
                logger.info("已删除因子 '%s' 的物理文件。", factor_id)

    def list_factors(self) -> list[dict]:
        """列出所有已注册因子信息。

        参数:
            无

        返回:
            list[dict]
        """
        return self._catalog.list_factors()

    def close(self) -> None:
        """Release per-generation lock handles held by the delta-mode
        :class:`PartitionLockManager` (R39 PERF-054).  Does not close the
        catalog connection (caller-managed)."""
        lm = self._lock_manager
        if lm is not None:
            lm.close()
            self._lock_manager = None


def _sync_module_wa_counters(mat: "ParquetMaterializer") -> None:
    """Mirror the materializer's write-amplification snapshot into the module
    level Gate-04 counter (``historical_rewrite_bytes``) read by
    ``scripts/r39_hard_gates_audit.py``."""
    global historical_rewrite_bytes

    try:
        snap = mat.write_amplification.snapshot()
    except Exception:
        return
    historical_rewrite_bytes = int(snap.historical_rewrite_bytes)


def compare_live_vs_materialized(
    live: pd.Series,
    materialized: pd.Series,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """R20-466: live → materialize → reload 精度差分报告。

    比较原始运行结果与物化后 reload 结果，报告：
    - ``max_abs_error`` / ``max_rel_error``：逐格最大绝对/相对误差；
    - ``rank_changes``：cross-sectional 每行 rank 变化比例（Spearman 一致率）；
    - ``sign_flips``：符号翻转格数；
    - ``threshold_flips``：跨阈值（0 附近 +- atol）翻转格数；
    - ``equal``：完全一致（含 NaN 位置一致）。
    """
    import numpy as np

    if materialized is None:
        return {"equal": False, "reason": "materialized reload is None"}
    # Align on the MultiIndex (materialize sorts [asset, datetime]; live may be
    # datetime-major).  Position-wise comparison would be wrong.
    if isinstance(live, pd.Series) and isinstance(materialized, pd.Series):
        idx = live.index.union(materialized.index)
        live = live.reindex(idx).sort_index()
        materialized = materialized.reindex(idx).sort_index()
    lv = np.asarray(live, dtype=float)
    mt = np.asarray(materialized, dtype=float)
    if lv.shape != mt.shape:
        return {
            "equal": False,
            "reason": f"shape mismatch live={lv.shape} materialized={mt.shape}",
            "max_abs_error": None,
            "max_rel_error": None,
            "rank_changes": None,
            "sign_flips": None,
            "threshold_flips": None,
        }
    lv_nan = np.isnan(lv)
    mt_nan = np.isnan(mt)
    nan_mismatch = int((lv_nan != mt_nan).sum())
    # R32-P1-057: ±Inf mask 分开比较 —— 不能只靠 np.array_equal(equal_nan=True)。
    lv_pos_inf = np.isposinf(lv)
    mt_pos_inf = np.isposinf(mt)
    lv_neg_inf = np.isneginf(lv)
    mt_neg_inf = np.isneginf(mt)
    pos_inf_mask_mismatch = int((lv_pos_inf != mt_pos_inf).sum())
    neg_inf_mask_mismatch = int((lv_neg_inf != mt_neg_inf).sum())
    inf_mask_mismatch = pos_inf_mask_mismatch + neg_inf_mask_mismatch
    finite = ~lv_nan & ~mt_nan & ~lv_pos_inf & ~lv_neg_inf & ~mt_pos_inf & ~mt_neg_inf
    denom = np.where(np.abs(lv) > 0, np.abs(lv), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(finite, np.abs(lv - mt) / denom, 0.0)
        abs_e = np.where(finite, np.abs(lv - mt), 0.0)
    max_abs = float(np.max(abs_e)) if abs_e.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    sign_flips = 0
    threshold_flips = 0
    if finite.any():
        lv_s = np.sign(lv[finite])
        mt_s = np.sign(mt[finite])
        sign_flips = int((lv_s != mt_s).sum())
        threshold_flips = int(
            ((np.abs(lv[finite]) <= atol) != (np.abs(mt[finite]) <= atol)).sum()
        )
    rank_corr = None
    rank_changes = None
    # R32-P0-030: rank 变化必须逐 timestamp 横截面（不 flatten）。Materialize 后
    # 排序为 [asset, datetime]；这里按 (datetime, asset) MultiIndex unstack 成
    # date × asset 面板，再对每行（每个 timestamp）做横截面 rank 比较。
    try:
        if isinstance(live, pd.Series) and isinstance(materialized, pd.Series):
            l_panel = live.unstack()
            m_panel = materialized.unstack()
            common = l_panel.index.intersection(m_panel.index)
            if len(common) and l_panel.shape == m_panel.shape:
                lr = l_panel.loc[common].rank(axis=1, method="average")
                mr = m_panel.loc[common].rank(axis=1, method="average")
                both_finite = np.isfinite(lr.to_numpy()) & np.isfinite(mr.to_numpy())
                if both_finite.sum() > 0:
                    rank_changes = float((lr.to_numpy()[both_finite] != mr.to_numpy()[both_finite]).mean())
        if rank_changes is None and finite.sum() >= 2:
            # 兜底：非 Series/无法 unstack 时退化为扁平比较（尽量保持信息）。
            from scipy.stats import spearmanr

            corr = spearmanr(lv[finite], mt[finite]).statistic
            rank_corr = None if corr is None else float(corr)
            if len(lv[finite]) > 2:
                lr = _rank_1d(lv[finite])
                mr = _rank_1d(mt[finite])
                rank_changes = float((lr != mr).mean())
    except Exception:  # noqa: BLE001 - rank stats are best-effort
        rank_corr = None
    # R32-P0-031: equal 必须同时满足 —— exact key/index、NaN mask 零差异、
    # ±Inf mask 零差异、有限值 tolerance 通过。任一 mask mismatch 即 not equal。
    finite_tolerance_ok = bool(max_abs <= atol and max_rel <= rtol)
    equal = bool(
        nan_mismatch == 0
        and inf_mask_mismatch == 0
        and (np.array_equal(lv, mt, equal_nan=True) or finite_tolerance_ok)
    )
    return {
        "equal": equal,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "rank_correlation": rank_corr,
        "rank_changes": rank_changes,
        "sign_flips": sign_flips,
        "threshold_flips": threshold_flips,
        "nan_mismatch": nan_mismatch,
        "pos_inf_mask_mismatch": pos_inf_mask_mismatch,
        "neg_inf_mask_mismatch": neg_inf_mask_mismatch,
        "inf_mask_mismatch": inf_mask_mismatch,
        "n_finite": int(finite.sum()),
        "n_total": int(lv.size),
    }


def _rank_1d(values: Any) -> Any:
    """Average-rank a 1-D array (stable, tie-aware)."""
    import pandas as pd

    return pd.Series(values).rank(method="average").to_numpy()
