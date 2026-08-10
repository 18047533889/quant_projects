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
    ) -> None:
        """初始化实例。
        
        参数:
            lake_root: 因子湖根目录（可选）
            catalog: FactorCatalog 实例（可选）
            staging_dataset: 见函数签名（可选）
        
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
        if author is None:
            author = _get_default_author()

        # R32-P0-036/043: factor_id 统一 domain gate（长度超限 reject、禁路径
        # 穿越/控制字符、Unicode NFC、保留名）。HTTP 之外直接 Python API /
        # materializer / catalog / delete 走同一个 validator。
        from security.factor_id import validate_factor_id

        factor_id = validate_factor_id(factor_id)

        # R32-P0-027: write_mode 严格枚举门 —— 拼错字符串必须拒绝，绝不静默
        # 回落为 keep-last upsert。replace_window 必须在写盘路径前校验窗口。
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

        # R10 #17: production 物化必须提供真实因子身份（AST Hash / IR）。
        # 缺省从运行模式推断；显式传入覆盖推断。
        production = self._resolve_production(production)
        if production and ast_hash == NO_FACTOR_IDENTITY:
            raise FactorIdentityMismatch(
                f"production materialize of factor '{factor_id}' requires a "
                f"factor identity; got sentinel {NO_FACTOR_IDENTITY!r}. "
                f"Pass ast_hash or ir_node."
            )

        # #收官轮 P0（Integration）：write_target 在任何 side effect（注册因子 /
        # 写文件 / 更新水位线 / 返回 rows_written>0）之前严格枚举。``"locla"``
        # 这类拼写错误必须直接拒绝，不能再静默变成 write_local=False ∧
        # write_staging=False 却照样提交（metadata 说已提交、物理数据不存在）。
        from storage.write_targets import normalize_write_target

        try:
            target_flags = normalize_write_target(write_target)
        except ValueError:
            raise
        write_target_flags = target_flags
        # #收官轮 P0（Integration，incomplete-fix bypass）：production 禁止
        # direct-local 因子湖直写——之前 guard 只放在 LocalParquetWriteTarget，
        # materialize() 主路径直接 ``_upsert_partition`` 绕过它。这里把 guard
        # 提到统一 orchestrator（materialize 本体）：production 下任何含 local
        # 落盘的目标都拒绝，必须走 staging→publish。
        if production and write_target_flags["local"]:
            raise ValueError(
                f"factor_id={factor_id!r}: production 禁止 direct-local factor-lake "
                "write（绕过 DataAccess staging→publish 原子发布/journal/snapshot "
                "manifest）。请用 write_target='staging' + publish_factor_lake。"
            )

        # R10 #47: 断点续写必须绑定身份 —— 在分区循环前算好身份级指纹组件，
        # 分区级输入指纹在循环内逐分区计算。
        # R20-210: 身份计算失败必须区分「合法不可得」（IdentityUnavailable：
        # 无 IR 也无有效 ast_hash -> checkpoint_identity=None，production 重算）
        # 与「代码抛异常」（IdentityComputationFailed：production hard fail，
        # 绝不降级为无指纹，否则两个失败的 computation 都退化成 None 并误判
        # resume 可跳过）。
        checkpoint_identity = semantic_identity
        if checkpoint_identity is None:
            has_any_identity = ir_node is not None and ast_hash != NO_FACTOR_IDENTITY
            if not has_any_identity and ast_hash in (None, NO_FACTOR_IDENTITY):
                # 合法不可得：没有可用的身份输入。
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

        # R20-201..206: storage precision policy —— value_dtype 显式单位化进入
        # lineage / row metadata / checkpoint 指纹，production 默认 float64。
        effective_value_dtype, precision_policy = storage_precision_policy_for(
            value_dtype,
            production=production,
            lineage_extra=lineage_extra,
        )
        if write_metadata:
            # 把解析后的 precision policy 写回 lineage extra，下游 (catalog /
            # 事件增量 rebuild / dual-write) 消费同一份存储精度契约。
            lineage_extra = dict(lineage_extra)
            lineage_extra["storage_precision_policy"] = precision_policy
            lineage_extra["storage_value_dtype"] = effective_value_dtype
            if run_lineage is not None:
                run_lineage = dict(run_lineage)
                run_lineage["extra"] = lineage_extra

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
        # Research / plain materialize keeps the historical "all-NaN -> skip"
        # short-circuit for backward compatibility.
        has_valid_value = bool(df["value"].notna().any())
        force_tombstones = self._resolve_force_tombstones(
            force_tombstones, production, run_lineage
        )
        # R20-230: recompute_window 的 all-NaN 结果必须覆盖旧 finite —— 绝不因
        # 「无有效值」而跳过写盘（跳过会让旧有限值残留）。recompute_window 下
        # 只要 frame 非空（哪怕全 NaN），强制写 tombstone。
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

        target = str(write_target or "local").lower()
        write_local = bool(write_target_flags["local"])
        write_staging = bool(write_target_flags["staging"])
        write_clickhouse = bool(write_target_flags["clickhouse"])
        if write_target_flags.get("staging_dataset"):
            self._staging_dataset = str(write_target_flags["staging_dataset"])

        # R9-P0-025/026: watermark_deferred must be defined before the partition
        # loop — the failure-adjudication block below reads it, and it must never
        # advance the watermark on a failed run.
        watermark_deferred = bool(defer_watermark)
        # #收官轮 P0（Integration）：staging-only（不写本地 lake 也不写 CH）不能
        # 推进**权威水位线**——否则增量调度器误以为「正式 factor lake 已提交到
        # 这里」而 skip。staging-only 强制 defer，只有 publish 成功后 commit 才
        # 推进（权威水位线 = CALCULATED → STAGED → PUBLISHED 语义）。
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
            # staging 优先：失败则不写本地分区、不更新水位线
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

        if write_local:
            # Phase 5 R17：直接迭代 generator，禁止 ``list(iter_partition_groups(...))``
            # ——那会把所有分区组同时引用在内存里，10 年因子等于多一份全量 DataFrame。
            progress = ProgressLogger(
                logger,
                desc=f"落盘因子 {factor_id}",
                total=None,
                unit="partition",
            )
            for part_values, partition_df in iter_partition_groups(work_df, policy):
                pkey = partition_key(part_values)
                ck_year = checkpoint_year(part_values)
                # R10 #47: 分区级身份指纹 —— 在 resume 判定与成功写入后都要用。
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
                # R20-201..206: storage precision policy 进入指纹 sidecar（
                # checkpoint_fingerprint_matches 比较已知 key，precision 变化会
                # 通过 identity_digest / 这里补充的 key 同时失效 resume）。
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
                    self._upsert_partition(
                        factor_dir,
                        part_values,
                        partition_df,
                        policy=policy,
                        write_mode=write_mode,
                        replace_window=replace_window,
                    )
                    # R32-P0-032: checkpoint success 必须在身份 sidecar durable 之后
                    # 提交。production 下 sidecar 写失败抛错 → 走 failed 分支，
                    # 分区不会被标成 success（数据已写但身份缺失 → 下次 resume 重算）。
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

            # Review-8 #440/#441: failure adjudication comes FIRST.  Any
            # required partition failure must never advance the committed
            # watermark, and an all-partitions-failed run must raise instead of
            # returning a ``rows_written=0`` success summary.
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

        # --- 5. 更新水位线（R9-P0-026: commit watermark LAST）---
        # Success-path ordering: (1) all required partitions succeeded (checked
        # above), (2) dependency writes (staging) succeeded (before the loop),
        # (3) lineage recorded + checkpoints cleared, (4) then and only then
        # advance the committed watermark.  If any earlier step fails, the
        # watermark must remain untouched so the next incremental run cannot
        # skip data.
        value_columns = [c for c in work_df.columns if c not in policy.columns]
        if write_local and (partitions_written or partitions_skipped):
            active_keys = set(partition_keys_written) | set(partition_keys_skipped)

            def _row_partition_key(row: pd.Series) -> str:
                """_row_partition_key。

                参数:
                    row: 见函数签名

                返回:
                    str
                """
                return partition_key({col: row[col] for col in policy.columns})

            active_df = work_df.loc[
                work_df.apply(_row_partition_key, axis=1).isin(active_keys),
                value_columns,
            ]
        else:
            active_df = work_df[value_columns]

        start_date = active_df["datetime"].min().isoformat()
        end_date = active_df["datetime"].max().isoformat()
        existing_wm = self._catalog.get_watermark(factor_id)
        if existing_wm is not None:
            if existing_wm["start_date"] < start_date:
                start_date = existing_wm["start_date"]
            if existing_wm["end_date"] > end_date:
                end_date = existing_wm["end_date"]

        if write_local:
            total_rows = self._count_total_rows(factor_dir)
            # R32-P0-029: 统一记录 long/wide 各语义的行数分解（watermark 的
            # row_count 保持历史兼容语义，分解指标进 summary）。
            partition_metrics = self._count_partition_metrics(factor_dir)
        else:
            total_rows = len(active_df)
            partition_metrics = {
                "physical_row_count": total_rows,
                "date_count": int(active_df["datetime"].nunique()),
                "asset_count": int(active_df["asset"].nunique()),
                "cell_count": total_rows,
                "non_null_cell_count": int(active_df["value"].notna().sum()),
            }

        pending_watermark = {
            "start_date": start_date,
            "end_date": end_date,
            "row_count": total_rows,
        }
        pending_lineage = None

        if watermark_deferred:
            # 双写模式：本地分区已落盘，水位线留到 commit_deferred_materialization
            # 确认 staging/clickhouse 依赖成功后再统一提交。
            watermark = self._catalog.get_watermark(factor_id)
            if run_lineage is not None:
                pending_lineage = dict(run_lineage)
                if dq_report is not None:
                    pending_lineage["dq_passed"] = dq_report.passed
        else:
            # (3) lineage + checkpoint cleanup first ...
            if run_lineage is not None:
                lineage_payload = dict(run_lineage)
                if dq_report is not None:
                    lineage_payload["dq_passed"] = dq_report.passed
                self._catalog.record_run(lineage_payload)
                if write_local:
                    self._catalog.clear_partition_checkpoints(factor_id)
            # (4) commit watermark LAST — only after every required step succeeded.
            self._catalog.update_watermark(
                factor_id=factor_id,
                start_date=start_date,
                end_date=end_date,
                row_count=total_rows,
            )
            watermark = self._catalog.get_watermark(factor_id)

        # R9-P0-027: NaN rows are now retained as tombstones, so the physical
        # rows in each partition include is_valid=0 tombstones.  `rows_written`
        # keeps the historical meaning: valid rows for the default path (matching
        # the old dropna behavior), total rows when the caller asked to keep
        # invalid rows explicitly, or when a production-incremental all-NaN run
        # wrote tombstones (R10 #48).
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
        merged["watermark"] = self._catalog.get_watermark(factor_id)
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
    ) -> None:
        """对指定 hive 分区做幂等 Upsert（long 或 wide）。

        ``write_mode``（R20-230）：
          - ``upsert``：默认，``[datetime, asset]`` dedup keep="last"（新值覆盖旧值）；
          - ``append``：不覆盖既有 key——冲突时保留旧行（dedup keep="first"）；
          - ``replace_window``：先删除分区内 ``[start, end]`` 窗口的旧行，再写入新行；
          - ``recompute_window``：与 upsert 相同（all-NaN 覆盖旧 finite 由上层
            强制 tombstone 保证）。

        参数:
            factor_dir: 见函数签名
            part_values: 见函数签名
            new_df: 见函数签名
            policy: 分区策略（可选）
            write_mode: 写入模式（upsert/append/replace_window/recompute_window）
            replace_window: 替换窗口 (start, end)（ISO 字符串）

        返回:
            无
        """
        if policy.is_wide:
            self._upsert_partition_wide(
                factor_dir, part_values, new_df, policy=policy,
                write_mode=write_mode, replace_window=replace_window,
            )
            return

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / policy.data_filename
        existing_rows = 0

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

    def _upsert_partition_wide(
        self,
        factor_dir: Path,
        part_values: dict[str, Any],
        new_df: pd.DataFrame,
        *,
        policy: PartitionPolicy,
        write_mode: str = "upsert",
        replace_window: tuple[str, str] | None = None,
    ) -> None:
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
            无
        """
        from storage.factor_format import pivot_long_to_wide, unpivot_wide_to_long

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / policy.data_filename

        value_cols = ["datetime", "asset", "value"]
        new_long = new_df[value_cols].copy()

        with self._partition_lock(partition_dir):
            self._cleanup_orphan_tmp_files(partition_dir)
            if parquet_path.exists():
                existing_panel = pd.read_parquet(parquet_path)
                if "datetime" in existing_panel.columns:
                    existing_panel = existing_panel.set_index("datetime")
                existing_long = unpivot_wide_to_long(existing_panel)
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
