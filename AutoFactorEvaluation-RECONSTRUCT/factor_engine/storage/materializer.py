"""因子落盘核心：pd.Series → 按日分区 Parquet，支持幂等增量更新与原子写入。

设计要点
--------
1. **Schema 强转**：所有因子值一律降级为 ``Float32``，资产列强转为 ``string``。
2. **数据清洗**：``±inf → NaN``，``dropna(subset=["value"])``。
3. **幂等 Upsert**：按日分区，旧数据与新数据 Concat 后按 ``[datetime, asset]``
   去重（保留最新），排序后整体覆盖。
4. **COS 兼容**：路径支持 ``cos://`` 前缀，自动使用 ``admin-cos`` 命令读写。
5. **原子写入**：本地路径 ``os.replace()``；COS 路径先写本地临时文件再上传。

目录拓扑
--------
::

    {output_path}/{factor_id}/data/
    ├── 2024-01-01.parquet
    ├── 2024-01-02.parquet
    └── ...

``output_path`` 可为本地绝对路径或 ``cos://bucket/prefix`` 格式的 COS 路径。
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from .catalog import FactorCatalog, compute_ir_hash
from .exceptions import FactorNotFoundError
from .cos_io import _is_cos, cos_read_parquet, cos_write_parquet, cos_exists, cos_ensure_dir
from logging_utils import ProgressLogger, get_logger

logger = get_logger("storage.materializer")

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
    """因子落盘器：将引擎计算结果（``pd.Series``）写入按日分区 Parquet。

    Parameters
    ----------
    lake_root : str | Path
        落盘根目录。必须显式传入，不提供默认值。
    catalog : FactorCatalog | None
        元数据目录实例。``None`` 时自动在 ``lake_root/_catalog.sqlite`` 创建。
    """

    def __init__(
        self,
        lake_root: str | Path,
        catalog: FactorCatalog | None = None,
    ) -> None:
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
        output_path: str | Path | None = None,
    ) -> dict:
        """将因子计算结果落盘为按日分区 Parquet。

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
        description : str | None
            因子描述（人类可读）。
        expression : str | None
            原始 DSL 表达式字符串。
        output_path : str | Path | None
            输出根路径。若提供，因子值落盘到 ``{output_path}/{factor_id}/data/``
            下的按日分区目录。若为 ``None``，则使用 ``{lake_root}/factors/{factor_id}/``。
            评估库调用时可直接指定每个因子的落盘位置。

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

        # --- 1. 转长表 + 强制 Schema ---
        df = self._normalize_to_long_table(result)

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

        # --- 4. 确定输出目录 + 按日分区 Upsert ---
        if output_path is not None:
            factor_dir = Path(output_path) / factor_id / "data"
        else:
            factor_dir = self._lake_root / "factors" / factor_id

        # 按日分区：每天一个独立的 .parquet 文件
        df["_date"] = df["datetime"].dt.date
        partitions_written: list[str] = []
        date_values = sorted(df["_date"].unique())
        progress = ProgressLogger(
            logger,
            desc=f"落盘因子 {factor_id}",
            total=len(date_values),
            unit="partition",
        )

        for date_val in date_values:
            date_str = date_val.isoformat()  # 如 "2026-07-03"
            partition_df = df.loc[df["_date"] == date_val].drop(columns=["_date"])
            self._upsert_daily_partition(factor_dir, date_str, partition_df)
            partitions_written.append(date_str)
            progress.advance(detail=date_str)

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
        }

    # ------------------------------------------------------------------
    # 内部：数据规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_long_table(result: pd.Series) -> pd.DataFrame:
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

    def _upsert_daily_partition(
        self, factor_dir: Path, date_str: str, new_df: pd.DataFrame
    ) -> None:
        """对指定日分区做幂等 Upsert。

        支持 COS 路径（``cos://``）和本地路径。
        COS 路径：本地临时 parquet → ``admin-cos cp`` 上传。
        本地路径：直接写 parquet + ``os.replace()`` 原子覆盖。

        路径: {factor_dir}/{date_str}.parquet
        """
        parquet_path = str(factor_dir / f"{date_str}.parquet")
        tmp_path = str(factor_dir / f".{date_str}.parquet.tmp")
        is_cos = _is_cos(parquet_path)
        existing_rows = 0

        # ── 读取已有数据 ──
        if is_cos:
            cos_ensure_dir(str(factor_dir))
            if cos_exists(parquet_path):
                existing_df = cos_read_parquet(parquet_path)
                existing_df["asset"] = existing_df["asset"].astype("string")
                existing_df["value"] = existing_df["value"].astype("float32")
                existing_rows = len(existing_df)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df
        else:
            factor_dir.mkdir(parents=True, exist_ok=True)
            local_parquet = Path(parquet_path)
            if local_parquet.exists():
                existing_df = pd.read_parquet(local_parquet)
                existing_df["asset"] = existing_df["asset"].astype("string")
                existing_df["value"] = existing_df["value"].astype("float32")
                existing_rows = len(existing_df)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df

        # ── 去重 + 排序 ──
        combined = combined.drop_duplicates(
            subset=["datetime", "asset"], keep="last"
        )
        combined = combined.sort_values(
            ["asset", "datetime"]
        ).reset_index(drop=True)

        # ── 写入 ──
        if is_cos:
            # COS：写本地临时文件 → 上传 → 清理
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                local_tmp = f.name
            try:
                combined.to_parquet(local_tmp, index=False, engine="pyarrow")
                import subprocess
                cp = subprocess.run(
                    ["admin-cos", "cp", local_tmp, parquet_path],
                    capture_output=True, text=True, timeout=120,
                )
                if cp.returncode != 0:
                    raise RuntimeError(
                        f"COS 上传失败: {parquet_path}\n{cp.stderr}"
                    )
            finally:
                Path(local_tmp).unlink(missing_ok=True)
        else:
            # 本地：原子写入
            combined.to_parquet(tmp_path, index=False, engine="pyarrow")
            os.replace(str(tmp_path), str(parquet_path))

        logger.debug(
            "日分区写入完成: %s, existing=%d, incoming=%d, final=%d",
            parquet_path, existing_rows, len(new_df), len(combined),
        )

    @staticmethod
    def _count_total_rows(factor_dir: Path) -> int:
        """统计因子目录下所有按日分区 Parquet 的总行数。"""
        is_cos = _is_cos(str(factor_dir))
        if is_cos:
            from .cos_io import cos_list_dir, cos_read_parquet
            total = 0
            for name in cos_list_dir(str(factor_dir)):
                if name.endswith(".parquet") and not name.startswith("."):
                    try:
                        df = cos_read_parquet(
                            f"{str(factor_dir).rstrip('/')}/{name}"
                        )
                        total += len(df)
                    except Exception:
                        pass
            return total

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
