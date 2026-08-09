"""多因子宽矩阵物化：训练/回测零 pivot 读取路径。

输出布局::

    {matrix_root}/
      universe={universe}/
        freq={freq}/
          year=2025/
            month=01/
              data.parquet   # datetime, asset, factor_a, factor_b, ...
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from logging_utils import get_logger
from storage.factor_format import series_to_long_table
from storage.partition_policy import (
    PartitionPolicy,
    attach_partition_columns,
    iter_partition_groups,
    partition_path_segments,
)

logger = get_logger("storage.factor_matrix_materializer")


@contextlib.contextmanager
def _partition_write_lock(lock_path: Path):
    """分区级写互斥（flock）：并发 matrix 写同一分区时 serial 化 read-merge-write。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _merge_matrix_frames(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    *,
    value_dtype: str,
) -> pd.DataFrame:
    """外连接合并已有分区宽表与本次增量（R11 #7）。

    之前直接 ``os.replace`` 覆盖整月 ``data.parquet``——增量写子集（部分因子/
    部分日期）会丢掉原分区里其它因子与其它日期的值。改为 read-merge-write：
    旧因子列与旧日期保留，同 ``(datetime, asset)`` key 上新值覆盖旧值。
    """
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing
    key_cols = [
        c
        for c in ("datetime", "asset")
        if c in existing.columns and c in new.columns
    ]
    if not key_cols:
        return new
    merged = existing.merge(new, on=key_cols, how="outer", suffixes=("_old", ""))
    new_cols = set(new.columns) - set(key_cols)
    drop = [
        c for c in merged.columns if c.endswith("_old") and c[:-4] in new_cols
    ]
    if drop:
        merged = merged.drop(columns=drop)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    for c in sorted(new_cols):
        if c in merged.columns:
            merged[c] = merged[c].astype(str(value_dtype or "float32"))
    cols = [c for c in key_cols if c in merged.columns] + sorted(
        c for c in merged.columns if c not in key_cols
    )
    return merged[cols].sort_values(key_cols).reset_index(drop=True)


@dataclass(frozen=True)
class FactorMatrixLayout:
    """factor_matrix 宽表布局路径规则。
    
    参数:
        无
    """
    universe: str
    frequency: str = "1d"
    partition_columns: tuple[str, ...] = ("year", "month")

    def base_dir(self, matrix_root: Path) -> Path:
        """base_dir。
        
        参数:
            matrix_root: factor_matrix 根目录
        
        返回:
            Path
        """
        return (
            matrix_root
            / f"universe={self.universe}"
            / f"freq={self.frequency}"
        )


class FactorMatrixMaterializer:
    """多因子矩阵宽表物化器。
    
    参数:
        matrix_root: factor_matrix 根目录（可选）
    """

    def __init__(self, matrix_root: str | Path | None = None) -> None:
        """初始化实例。
        
        参数:
            matrix_root: factor_matrix 根目录（可选）
        
        返回:
            无
        """
        if matrix_root is None:
            env_root = os.environ.get("FACTOR_MATRIX_ROOT")
            matrix_root = env_root or "factor_matrix"
        self._matrix_root = Path(matrix_root)
        self._matrix_root.mkdir(parents=True, exist_ok=True)

    @property
    def matrix_root(self) -> Path:
        """matrix_root。
        
        参数:
            无
        
        返回:
            Path
        """
        return self._matrix_root

    def materialize(
        self,
        results: dict[str, pd.Series],
        *,
        universe: str,
        frequency: str = "1d",
        partition_columns: Iterable[str] | None = None,
        value_dtype: str = "float32",
    ) -> dict[str, Any]:
        """``factor_id -> Series`` 合并为宽表并按 hive 分区落盘。
        
        参数:
            results: 见函数签名
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            partition_columns: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
        
        返回:
            dict[str, Any]
        """
        if not results:
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": [],
                "matrix_root": str(self._matrix_root),
            }

        factor_ids = sorted(results.keys())
        # R11 #7: factor_matrix 目前仍是直写本地的独立 IO（未走 DataAccess
        # staging→publish / manifest / snapshot / factor-version gate）。production
        # 下至少告警，避免运维无感知绕过 DataAccess 治理。
        try:
            from runtime.production_policy import is_production_mode

            if is_production_mode():
                logger.warning(
                    "factor_matrix 直写本地 matrix_root=%s；尚未接入 DataAccess "
                    "staging→publish/manifest/snapshot 治理。生产发布前应改走 "
                    "DataAccess factor_matrix staging/upsert（partial rows / "
                    "partial factors / partial dates 安全 merge）。",
                    self._matrix_root,
                )
        except Exception:  # pragma: no cover - 运行模式解析失败不阻塞写
            pass
        # Phase 5 P1-12：按因子列分块 merge，限制中间 merge 工作集
        # （5000×2500×2000 列的训练矩阵不可能一次成形）。
        try:
            block = max(16, int(os.environ.get("FACTOR_ENGINE_MATRIX_COLUMNS_PER_BLOCK", "256")))
        except ValueError:
            block = 256
        merged: pd.DataFrame | None = None
        block_ids: list[str] = []
        block_frame: pd.DataFrame | None = None

        def _flush_block() -> None:
            nonlocal block_frame, merged
            if block_frame is None:
                return
            if merged is None:
                merged = block_frame
            else:
                merged = merged.merge(block_frame, on=["datetime", "asset"], how="outer")
            block_frame = None

        for fid in factor_ids:
            series = results[fid]
            long_df = series_to_long_table(series)
            long_df = long_df.rename(columns={"value": fid})
            block_ids.append(fid)
            if block_frame is None:
                block_frame = long_df
            else:
                block_frame = block_frame.merge(
                    long_df, on=["datetime", "asset"], how="outer"
                )
            long_df = None  # 释放该因子长表引用
            if len(block_ids) >= block:
                _flush_block()
                block_ids = []
        _flush_block()

        if merged is None or merged.empty:
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": factor_ids,
                "matrix_root": str(self._matrix_root),
            }

        for fid in factor_ids:
            merged[fid] = merged[fid].astype(str(value_dtype or "float32"))

        layout = FactorMatrixLayout(
            universe=universe,
            frequency=frequency,
            partition_columns=tuple(partition_columns or ("year", "month")),
        )
        policy = PartitionPolicy.from_config(
            partition_columns=layout.partition_columns,
            storage_format="long",
        )
        work = attach_partition_columns(merged, policy)
        base = layout.base_dir(self._matrix_root)
        partitions_written: list[str] = []

        for part_values, partition_df in iter_partition_groups(work, policy):
            drop_cols = [c for c in policy.columns if c in partition_df.columns]
            out_df = partition_df.drop(columns=drop_cols).reset_index(drop=True)
            part_dir = base / partition_path_segments(
                part_values, column_order=policy.columns
            )
            part_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = part_dir / "data.parquet"
            lock_path = part_dir / ".data.parquet.lock"
            # R11 #7: read-merge-write 在 flock 互斥内完成——并发 matrix 写同一分区
            # 不再互相覆盖；tmp 文件名带 pid+uuid，崩溃残留不再命中固定 .tmp。
            with _partition_write_lock(lock_path):
                if parquet_path.exists():
                    try:
                        existing = pd.read_parquet(parquet_path)
                    except Exception:  # pragma: no cover - 分区文件损坏按空处理
                        logger.warning(
                            "factor_matrix 分区读取失败 %s，按空分区合并",
                            parquet_path,
                            exc_info=True,
                        )
                        existing = None
                    merged_out = _merge_matrix_frames(
                        existing, out_df, value_dtype=value_dtype
                    )
                else:
                    merged_out = out_df
                tmp_path = (
                    part_dir
                    / f".data.parquet.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                merged_out.to_parquet(tmp_path, index=False, engine="pyarrow")
                os.replace(str(tmp_path), str(parquet_path))
                rows_written_this = len(merged_out)
            pkey = "|".join(f"{k}={part_values[k]}" for k in sorted(part_values))
            partitions_written.append(pkey)
            logger.info(
                "factor_matrix 分区 upsert universe=%s partition=%s rows=%d cols=%d",
                universe,
                pkey,
                rows_written_this,
                len(merged_out.columns),
            )

        return {
            "universe": universe,
            "frequency": frequency,
            "rows_written": len(merged),
            "partitions": partitions_written,
            "factor_ids": factor_ids,
            "matrix_root": str(self._matrix_root),
            "columns": ["datetime", "asset", *factor_ids],
        }

    @staticmethod
    def load_matrix(
        matrix_root: str | Path,
        *,
        universe: str,
        frequency: str = "1d",
        factor_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读取 universe 下全部或指定因子列宽表。
        
        参数:
            matrix_root: factor_matrix 根目录
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            factor_ids: 因子 ID 列表（可选）
        
        返回:
            pd.DataFrame
        """
        base = (
            Path(matrix_root)
            / f"universe={universe}"
            / f"freq={frequency}"
        )
        if not base.exists():
            raise FileNotFoundError(f"factor_matrix 不存在: {base}")

        frames: list[pd.DataFrame] = []
        for pq in sorted(base.rglob("data.parquet")):
            if pq.name.startswith("."):
                continue
            df = pd.read_parquet(pq)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"factor_matrix 无 parquet: {base}")

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["datetime", "asset"], keep="last")
        if factor_ids is not None:
            keep = ["datetime", "asset", *sorted(set(str(f) for f in factor_ids))]
            missing = [c for c in keep if c not in merged.columns]
            if missing:
                raise ValueError(f"factor_matrix 缺少列: {missing}")
            merged = merged[keep]
        return merged.sort_values(["datetime", "asset"]).reset_index(drop=True)
