"""因子落盘核心：pd.Series → 分区 Parquet，支持幂等增量更新与原子写入。

设计要点
--------
1. **Schema 强转**：所有因子值一律降级为 ``Float32``，资产列强转为 ``string``。
2. **数据清洗**：``±inf → NaN``，``dropna(subset=["value"])``。
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

from .catalog import FactorCatalog, compute_ir_hash
from .exceptions import FactorNotFoundError
from logging_utils import ProgressLogger, get_logger

logger = get_logger("storage.materializer")

METADATA_COLUMNS = ("calc_time", "factor_version", "data_snapshot_id", "is_valid")


@dataclass(frozen=True)
class MaterializeMetadata:
    """因子湖长表可选元数据列。"""

    calc_time: str
    factor_version: str
    data_snapshot_id: str | None = None
    is_valid: int = 1


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
    ) -> None:
        if lake_root is None:
            lake_root = default_factor_lake_root()
        self._lake_root = Path(lake_root)
        self._lake_root.mkdir(parents=True, exist_ok=True)

        if catalog is None:
            catalog = FactorCatalog(self._lake_root / "_catalog.sqlite")
        self._catalog = catalog

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
                result, thresholds=dq_thresholds, raise_on_fail=dq_strict
            )

        # --- 1. 转长表 + 强制 Schema ---
        meta = None
        if write_metadata:
            meta = MaterializeMetadata(
                calc_time=datetime.now(timezone.utc).isoformat(),
                factor_version=ast_hash[:16],
                data_snapshot_id=data_snapshot_id,
            )
        df = self._normalize_to_long_table(result, metadata=meta)

        # --- 2. 数据清洗 ---
        df = self._clean(df)

        if df.empty:
            logger.warning("因子 '%s' 清洗后无有效数据，跳过落盘。", factor_id)
            return {
                "factor_id": factor_id,
                "rows_written": 0,
                "partitions": [],
                "watermark": None,
            }

        # --- 3. 注册 / Hash 校验（可能抛 FactorHashMismatchError）---
        self._catalog.register(
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            ast_hash=ast_hash,
            description=description,
            expression=expression,
        )

        # --- 4. 按年分区 Upsert ---
        factor_dir = self._lake_root / "factors" / factor_id
        df["_year"] = df["datetime"].dt.year
        partitions_written: list[int] = []
        partition_years = sorted(int(year) for year in df["_year"].unique())
        progress = ProgressLogger(
            logger,
            desc=f"落盘因子 {factor_id}",
            total=len(partition_years),
            unit="partition",
        )

        for year in partition_years:
            partition_df = df.loc[df["_year"] == year].drop(columns=["_year"])
            self._upsert_partition(factor_dir, int(year), partition_df)
            partitions_written.append(int(year))
            progress.advance(detail=f"year={year}, rows={len(partition_df)}")

        # --- 5. 更新水位线 ---
        start_date = df["datetime"].min().isoformat()
        end_date = df["datetime"].max().isoformat()

        # 水位线需合并旧区间
        existing_wm = self._catalog.get_watermark(factor_id)
        if existing_wm is not None:
            if existing_wm["start_date"] < start_date:
                start_date = existing_wm["start_date"]
            if existing_wm["end_date"] > end_date:
                end_date = existing_wm["end_date"]

        total_rows = self._count_total_rows(factor_dir)
        self._catalog.update_watermark(
            factor_id=factor_id,
            start_date=start_date,
            end_date=end_date,
            row_count=total_rows,
        )

        watermark = self._catalog.get_watermark(factor_id)

        if run_lineage is not None:
            lineage_payload = dict(run_lineage)
            if dq_report is not None:
                lineage_payload["dq_passed"] = dq_report.passed
            self._catalog.record_run(lineage_payload)

        logger.info(
            "因子 '%s' 落盘完成：%d 行，分区 %s，水位线 [%s → %s]",
            factor_id,
            len(df),
            partitions_written,
            start_date,
            end_date,
        )

        return {
            "factor_id": factor_id,
            "rows_written": len(df),
            "partitions": partitions_written,
            "watermark": watermark,
            "dq_report": dq_report.to_dict() if dq_report is not None else None,
            "run_id": run_lineage.get("run_id") if run_lineage else None,
        }

    # ------------------------------------------------------------------
    # 内部：数据规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_long_table(
        result: pd.Series,
        *,
        metadata: MaterializeMetadata | None = None,
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
        df["value"] = df["value"].astype("float32")
        if metadata is not None:
            df["calc_time"] = metadata.calc_time
            df["factor_version"] = metadata.factor_version
            df["data_snapshot_id"] = metadata.data_snapshot_id or ""
            df["is_valid"] = int(metadata.is_valid)
        return df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """清洗：inf → NaN，dropna(value)。"""
        df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 内部：分区 Upsert + 原子写入
    # ------------------------------------------------------------------

    def _upsert_partition(
        self, factor_dir: Path, year: int, new_df: pd.DataFrame
    ) -> None:
        """对指定年份分区做幂等 Upsert。

        步骤：读旧 → concat → 去重(保留最新) → 排序 → 原子覆盖写。
        """
        partition_dir = factor_dir / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / "data.parquet"
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
        tmp_path = partition_dir / ".data.parquet.tmp"
        combined.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(str(tmp_path), str(parquet_path))
        logger.info(
            "分区写入完成: factor_dir=%s, year=%s, existing_rows=%d, incoming_rows=%d, final_rows=%d",
            factor_dir,
            year,
            existing_rows,
            len(new_df),
            len(combined),
        )

    @staticmethod
    def _count_total_rows(factor_dir: Path) -> int:
        """统计因子目录下所有分区 Parquet 的总行数。"""
        total = 0
        if not factor_dir.exists():
            return 0
        for pq_file in factor_dir.rglob("data.parquet"):
            # 用 pyarrow 的 metadata 快速获取行数，避免加载全部数据
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(pq_file)
                total += meta.num_rows
            except Exception:
                # 退化为 pandas 读取
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
