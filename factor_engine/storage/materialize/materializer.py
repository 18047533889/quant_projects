"""因子落盘核心：pd.Series → 分区 Parquet，支持幂等增量更新与原子写入。

设计要点
--------
1. **Schema 强转**：因子值默认 ``float32``（可配置 ``value_dtype``），资产列强转为 ``string``。
2. **数据清洗**：``±inf → NaN``；默认 ``dropna``；``preserve_invalid_rows=True`` 时保留行并写 ``is_valid=0``、``invalid_reason``。
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

import getpass
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from workspace_paths import default_factor_lake_root

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
    """因子湖长表可选元数据列。"""

    calc_time: str
    factor_version: str
    data_snapshot_id: str | None = None
    is_valid: int = 1
    invalid_reason: str = ""


# ---------------------------------------------------------------------------
# 默认作者
# ---------------------------------------------------------------------------

def _get_default_author() -> str:
    """优先环境变量 ``QUANTSOCIETY_USER``，其次系统登录名，最后 ``anonymous``。"""
    return os.getenv("QUANTSOCIETY_USER", getpass.getuser() or "anonymous")


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------

class ParquetMaterializer:
    """因子落盘器：将引擎计算结果（``pd.Series``）写入分区 Parquet + SQLite Catalog。

    Parameters
    ----------
    lake_root : str | Path | None
        落盘根目录。``None`` 时优先读取 ``FACTOR_LAKE_ROOT``，否则落到
        统一的 ``workspace_data/factors/lake``。
    catalog : FactorCatalog | None
        元数据目录实例。``None`` 时自动在 ``lake_root/_catalog.sqlite`` 创建。
    """

    def __init__(
        self,
        lake_root: str | Path | None = None,
        catalog: FactorCatalog | None = None,
        *,
        staging_dataset: str = "factor_lake_staging",
    ) -> None:
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
        return self._lake_root

    @property
    def catalog(self) -> FactorCatalog:
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
        value_dtype: str = "float32",
        write_target: str = "local",
        defer_watermark: bool = False,
        partition_columns: list[str] | None = None,
        storage_format: str = "long",
    ) -> dict:
        """将因子计算结果落盘为分区 Parquet。

        Parameters
        ----------
        factor_id : str
            因子唯一标识（含版本号），如 ``'momentum_5m_v1'``。
        result : pd.Series
            引擎输出的 MultiIndex(timestamp, instrument) Series。
        author : str | None
            作者名。``None`` 时使用系统登录名。
        frequency : str
            频率标记（``'5m'``, ``'1d'`` 等）。
        ast_hash : str | None
            预先计算好的 AST Hash。如提供则直接使用。
        ir_node : IRNode | None
            IR 根节点。如 ``ast_hash`` 未提供，则从此节点计算。
            两者都未提供时使用占位 Hash。
        description : str | None
            因子描述（人类可读）。
        expression : str | None
            原始 DSL 表达式字符串。

        Returns
        -------
        dict
            落盘摘要：``{factor_id, rows_written, partitions, watermark}``

        Raises
        ------
        FactorHashMismatchError
            公式变更但 factor_id 未升级。
        ValueError
            输入数据格式不符合预期。
        """
        if author is None:
            author = _get_default_author()

        # --- AST Hash ---
        if ast_hash is None:
            if ir_node is not None:
                ast_hash = compute_ir_hash(ir_node)
            else:
                ast_hash = "__no_hash_provided__"

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
            meta = MaterializeMetadata(
                calc_time=datetime.now(timezone.utc).isoformat(),
                factor_version=ast_hash[:16],
                data_snapshot_id=data_snapshot_id,
            )
        df = self._normalize_to_long_table(result, metadata=meta, value_dtype=value_dtype)

        # --- 2. 数据清洗 ---
        df = self._clean(df, preserve_invalid_rows=preserve_invalid_rows)

        if df.empty:
            logger.warning("因子 '%s' 清洗后无有效数据，跳过落盘。", factor_id)
            return {
                "factor_id": factor_id,
                "rows_written": 0,
                "partitions": [],
                "watermark": None,
                "write_target": write_target,
            }

        target = str(write_target or "local").lower()
        write_local = target in ("local", "both")
        write_staging = target in ("staging", "both", "staging_clickhouse")
        # clickhouse / staging_clickhouse：写 catalog + watermark，不落本地分区
        if target in ("clickhouse", "staging_clickhouse"):
            write_local = False

        # --- 3. 注册 / Hash 校验（可能抛 FactorHashMismatchError）---
        self._catalog.register(
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            description=description,
            expression=expression,
            data_source_config=data_source_config,
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

        factor_dir = self._lake_root / "factors" / factor_id
        work_df = attach_partition_columns(df, policy)

        if write_local:
            partition_items = list(iter_partition_groups(work_df, policy))
            progress = ProgressLogger(
                logger,
                desc=f"落盘因子 {factor_id}",
                total=len(partition_items),
                unit="partition",
            )
            for part_values, partition_df in partition_items:
                pkey = partition_key(part_values)
                ck_year = checkpoint_year(part_values)
                if resume:
                    checkpoint = self._catalog.get_partition_checkpoint_by_key(
                        factor_id, pkey
                    )
                    if checkpoint is None and policy.columns == ("year",):
                        checkpoint = self._catalog.get_partition_checkpoint(
                            factor_id, ck_year
                        )
                    if checkpoint and checkpoint["status"] == "success":
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

        # --- 5. 更新水位线 ---
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
        existing_wm = self._catalog.get_watermark(factor_id)
        if existing_wm is not None:
            if existing_wm["start_date"] < start_date:
                start_date = existing_wm["start_date"]
            if existing_wm["end_date"] > end_date:
                end_date = existing_wm["end_date"]

        if write_local:
            total_rows = self._count_total_rows(factor_dir)
        else:
            total_rows = len(active_df)

        pending_watermark = {
            "start_date": start_date,
            "end_date": end_date,
            "row_count": total_rows,
        }
        watermark_deferred = bool(defer_watermark)
        pending_lineage = None

        if watermark_deferred:
            watermark = self._catalog.get_watermark(factor_id)
            if run_lineage is not None:
                pending_lineage = dict(run_lineage)
                if dq_report is not None:
                    pending_lineage["dq_passed"] = dq_report.passed
        else:
            self._catalog.update_watermark(
                factor_id=factor_id,
                start_date=start_date,
                end_date=end_date,
                row_count=total_rows,
            )
            watermark = self._catalog.get_watermark(factor_id)

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

        if run_lineage is not None and not watermark_deferred:
            lineage_payload = dict(run_lineage)
            if dq_report is not None:
                lineage_payload["dq_passed"] = dq_report.passed
            self._catalog.record_run(lineage_payload)
            if write_local:
                self._catalog.clear_partition_checkpoints(factor_id)

        logger.info(
            "因子 '%s' 落盘完成：%d 行，target=%s，分区 %s，跳过 %s，水位线 [%s → %s]",
            factor_id,
            len(df),
            target,
            partitions_written,
            partitions_skipped,
            start_date,
            end_date,
        )

        summary = {
            "factor_id": factor_id,
            "rows_written": len(df),
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
            "watermark_deferred": watermark_deferred,
            "storage_format": policy.storage_format,
            "partition_columns": list(policy.columns),
        }
        if pending_watermark is not None and watermark_deferred:
            summary["pending_watermark"] = pending_watermark
        if pending_lineage is not None:
            summary["pending_lineage"] = pending_lineage
        if staging_result is not None:
            summary["staging"] = staging_result
        return summary

    def commit_deferred_materialization(self, summary: dict) -> dict:
        """双写成功：提交延迟的水位线与 lineage。"""
        if not summary.get("watermark_deferred"):
            return summary
        factor_id = summary["factor_id"]
        pending = summary.get("pending_watermark") or {}
        self._catalog.update_watermark(
            factor_id=factor_id,
            start_date=pending["start_date"],
            end_date=pending["end_date"],
            row_count=pending.get("row_count"),
        )
        pending_lineage = summary.get("pending_lineage")
        if pending_lineage is not None:
            self._catalog.record_run(pending_lineage)
        partitions = summary.get("partitions") or []
        if partitions:
            self._catalog.clear_partition_checkpoints(factor_id)
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
        """双写失败：记录失败 run，不更新水位线。"""
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
            df["is_valid"] = int(metadata.is_valid)
            df["invalid_reason"] = metadata.invalid_reason or ""
        return df

    @staticmethod
    def _clean(df: pd.DataFrame, *, preserve_invalid_rows: bool = False) -> pd.DataFrame:
        """清洗：inf → NaN；生产模式可保留无效行并标注 invalid_reason。"""
        df = df.copy()
        df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
        if preserve_invalid_rows:
            if "is_valid" not in df.columns:
                df["is_valid"] = 1
            if "invalid_reason" not in df.columns:
                df["invalid_reason"] = ""
            invalid = df["value"].isna()
            df.loc[invalid, "is_valid"] = 0
            df.loc[invalid, "invalid_reason"] = "inf_or_nan"
            return df.reset_index(drop=True)
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        return df

    def _upsert_to_data_access_staging(
        self,
        factor_id: str,
        df: pd.DataFrame,
        *,
        policy: PartitionPolicy | None = None,
    ) -> dict:
        """经 data_access 幂等 upsert 到 staging 数据集（生产发布前暂存区）。"""
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

    def _upsert_partition(
        self,
        factor_dir: Path,
        part_values: dict[str, Any],
        new_df: pd.DataFrame,
        *,
        policy: PartitionPolicy,
    ) -> None:
        """对指定 hive 分区做幂等 Upsert（long 或 wide）。"""
        if policy.is_wide:
            self._upsert_partition_wide(factor_dir, part_values, new_df, policy=policy)
            return

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / policy.data_filename
        existing_rows = 0

        # 读取已有数据
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            existing_df["asset"] = existing_df["asset"].astype("string")
            existing_df["value"] = existing_df["value"].astype("float32")
            for col in METADATA_COLUMNS:
                if col not in existing_df.columns:
                    if col == "is_valid":
                        existing_df[col] = 1
                    elif col == "invalid_reason":
                        existing_df[col] = ""
                    else:
                        existing_df[col] = None
            existing_rows = len(existing_df)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        # 去重：按 [datetime, asset] 保留最后出现的（即最新值）
        combined = combined.drop_duplicates(
            subset=["datetime", "asset"], keep="last"
        )

        # 排序：保证 Polars join_asof 的物理预排序要求
        combined = combined.sort_values(
            ["asset", "datetime"]
        ).reset_index(drop=True)

        # 原子写入：先写 tmp，再 rename
        tmp_path = partition_dir / f".{policy.data_filename}.tmp"
        combined.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(str(tmp_path), str(parquet_path))
        logger.info(
            "分区写入完成: factor_dir=%s, partition=%s, existing_rows=%d, "
            "incoming_rows=%d, final_rows=%d",
            factor_dir,
            partition_key(part_values),
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
    ) -> None:
        """宽表 panel 分区 upsert（经 long 去重后再 pivot）。"""
        from storage.factor_format import pivot_long_to_wide, unpivot_wide_to_long

        partition_dir = factor_dir / partition_path_segments(
            part_values, column_order=policy.columns
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / policy.data_filename
        tmp_path = partition_dir / f".{policy.data_filename}.tmp"

        value_cols = ["datetime", "asset", "value"]
        new_long = new_df[value_cols].copy()

        if parquet_path.exists():
            existing_panel = pd.read_parquet(parquet_path)
            if "datetime" in existing_panel.columns:
                existing_panel = existing_panel.set_index("datetime")
            existing_long = unpivot_wide_to_long(existing_panel)
            combined_long = pd.concat([existing_long, new_long], ignore_index=True)
        else:
            combined_long = new_long

        combined_long = combined_long.drop_duplicates(
            subset=["datetime", "asset"], keep="last"
        )
        panel = pivot_long_to_wide(combined_long)
        panel.to_parquet(tmp_path, index=True, engine="pyarrow")
        os.replace(str(tmp_path), str(parquet_path))
        logger.info(
            "宽表分区写入完成: factor_dir=%s, partition=%s, shape=%s",
            factor_dir,
            partition_key(part_values),
            panel.shape,
        )

    @staticmethod
    def _count_total_rows(factor_dir: Path) -> int:
        """统计因子目录下所有分区 Parquet 的总行数。"""
        total = 0
        if not factor_dir.exists():
            return 0
        for pq_file in factor_dir.rglob("*.parquet"):
            if pq_file.name.startswith("."):
                continue
            try:
                import pyarrow.parquet as pq

                meta = pq.read_metadata(pq_file)
                total += meta.num_rows
            except Exception:
                df = pd.read_parquet(pq_file)
                total += len(df)
        return total

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------

    def delete_factor(self, factor_id: str, *, delete_files: bool = False) -> None:
        """从 Catalog 删除因子。可选同时删除物理文件。

        Parameters
        ----------
        factor_id : str
            要删除的因子 ID。
        delete_files : bool
            ``True`` 时同时删除对应 Parquet 目录。
        """
        self._catalog.delete_factor(factor_id)
        if delete_files:
            import shutil

            factor_dir = self._lake_root / "factors" / factor_id
            if factor_dir.exists():
                shutil.rmtree(factor_dir)
                logger.info("已删除因子 '%s' 的物理文件。", factor_id)

    def list_factors(self) -> list[dict]:
        """列出所有已注册因子信息。"""
        return self._catalog.list_factors()
