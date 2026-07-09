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

import os
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


@dataclass(frozen=True)
class FactorMatrixLayout:
    universe: str
    frequency: str = "1d"
    partition_columns: tuple[str, ...] = ("year", "month")

    def base_dir(self, matrix_root: Path) -> Path:
        return (
            matrix_root
            / f"universe={self.universe}"
            / f"freq={self.frequency}"
        )


class FactorMatrixMaterializer:
    """将多因子 MultiIndex Series 合并写入宽矩阵 Parquet。"""

    def __init__(self, matrix_root: str | Path | None = None) -> None:
        if matrix_root is None:
            env_root = os.environ.get("FACTOR_MATRIX_ROOT")
            matrix_root = env_root or "factor_matrix"
        self._matrix_root = Path(matrix_root)
        self._matrix_root.mkdir(parents=True, exist_ok=True)

    @property
    def matrix_root(self) -> Path:
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
        """``factor_id -> Series`` 合并为宽表并按 hive 分区落盘。"""
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
        merged: pd.DataFrame | None = None
        for fid in factor_ids:
            series = results[fid]
            long_df = series_to_long_table(series)
            long_df = long_df.rename(columns={"value": fid})
            if merged is None:
                merged = long_df
            else:
                merged = merged.merge(
                    long_df,
                    on=["datetime", "asset"],
                    how="outer",
                )

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
            tmp_path = part_dir / ".data.parquet.tmp"
            out_df.to_parquet(tmp_path, index=False, engine="pyarrow")
            os.replace(str(tmp_path), str(parquet_path))
            pkey = "|".join(f"{k}={part_values[k]}" for k in sorted(part_values))
            partitions_written.append(pkey)
            logger.info(
                "factor_matrix 分区写入 universe=%s partition=%s rows=%d cols=%d",
                universe,
                pkey,
                len(out_df),
                len(out_df.columns),
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
        """读取 universe 下全部或指定因子列宽表。"""
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
