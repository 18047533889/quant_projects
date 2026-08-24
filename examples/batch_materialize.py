"""收盘后批量增量落盘脚本。

功能
----
扫描预定义的因子配置列表，逐个执行引擎计算并落盘。
支持查阅 Catalog 水位线做增量裁剪，以及落盘失败后跳过并继续。

用法
----
::

    # 直接运行（使用脚本内置因子列表）
    python examples/batch_materialize.py

    # 指定 lake 根目录
    FACTOR_LAKE_ROOT=/nas/shared_factor_lake python examples/batch_materialize.py

可配合 crontab / Windows 任务计划程序设为每日收盘后自动执行。

自定义
------
修改下方 ``FACTOR_CONFIGS`` 列表即可添加/删除因子。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import time
from dataclasses import dataclass

import pandas as pd

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean, ts_std_dev, ts_delta
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.logging_utils import ProgressLogger, configure_logging, get_logger
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage import ParquetMaterializer
from factor_engine.storage.datasource import DataSource

configure_logging("INFO")
logger = get_logger("examples.batch_materialize")

# ============================================================================
# 因子注册表：在此添加需要每日落盘的因子
# ============================================================================

FACTOR_CONFIGS: list[dict] = [
    {
        "factor_id": "momentum_3d_rank_v1",
        "factor": Factor(
            name="momentum_3d_rank_v1",
            expr=rank(ts_mean(col("close"), 3)),
            freq="1d",
            description="3日均价截面排名",
        ),
        "frequency": "1d",
        "description": "3日均价截面排名",
    },
    {
        "factor_id": "volatility_5d_v1",
        "factor": Factor(
            name="volatility_5d_v1",
            expr=ts_std_dev(col("close"), 5),
            freq="1d",
            description="5日收盘价标准差",
        ),
        "frequency": "1d",
        "description": "5日收盘价标准差",
    },
    {
        "factor_id": "delta_close_1d_v1",
        "factor": Factor(
            name="delta_close_1d_v1",
            expr=ts_delta(col("close"), 1),
            freq="1d",
            description="1日收盘价变动",
        ),
        "frequency": "1d",
        "description": "1日收盘价变动",
    },
]


# ============================================================================
# 数据源（请根据实际环境替换）
# ============================================================================


class InMemorySeriesSource(DataSource):
    """演示用内存数据源。生产中请替换为 KlineParquetSource。"""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self.data = data

    def load_column(self, name: str):
        return self.data[name]


def build_demo_data_source() -> DataSource:
    """构造演示数据。生产中应替换为真实数据源构建。"""
    dates = pd.to_datetime([
        "2024-01-15", "2024-01-16", "2024-01-17",
        "2024-01-18", "2024-01-19", "2024-01-22",
    ])
    assets = ["000001.SZ", "000002.SZ", "600000.SH"]
    idx = pd.MultiIndex.from_product(
        [dates, assets], names=["timestamp", "instrument"]
    )
    close = pd.Series(
        [10.0, 20.0, 30.0,
         11.0, 19.0, 31.0,
         12.0, 21.0, 29.0,
         10.5, 20.5, 32.0,
         13.0, 18.0, 28.0,
         12.5, 22.0, 27.0],
        index=idx,
    )
    return InMemorySeriesSource({"close": close})


# ============================================================================
# 批量落盘主流程
# ============================================================================


def batch_materialize(
    factor_configs: list[dict],
    data_source: DataSource,
    lake_root: str | None = None,
) -> dict:
    """批量执行因子计算与落盘。

    Parameters
    ----------
    factor_configs : list[dict]
        因子配置列表，每项含 ``factor_id``、``factor``、``frequency`` 等。
    data_source : DataSource
        数据源实例。
    lake_root : str | None
        落盘根目录。

    Returns
    -------
    dict
        ``{succeeded: [...], failed: [...], skipped: [...]}``
    """
    engine = FactorEngine(backend=PandasBackend(), data_source=data_source)
    mat = ParquetMaterializer(lake_root=lake_root)

    results = {"succeeded": [], "failed": [], "skipped": []}

    logger.info("=" * 60)
    logger.info("开始批量落盘：共 %d 个因子", len(factor_configs))
    logger.info("Lake 根目录：%s", mat.lake_root)
    logger.info("=" * 60)
    progress = ProgressLogger(
        logger,
        desc="批量落盘",
        total=len(factor_configs),
        unit="factor",
    )

    for i, config in enumerate(factor_configs, 1):
        factor_id = config["factor_id"]
        factor = config["factor"]
        frequency = config.get("frequency", "1d")
        description = config.get("description")

        logger.info("[%d/%d] 处理因子: %s", i, len(factor_configs), factor_id)
        t0 = time.time()

        try:
            # 1. 引擎计算
            result_dict = engine.run(factor)
            series = result_dict["result"]
            ir_node = result_dict["analysis"].ir

            # 2. 落盘（catalog 内部会做 Hash 校验 + 幂等 Upsert）
            summary = mat.materialize(
                factor_id=factor_id,
                result=series,
                ir_node=ir_node,
                frequency=frequency,
                description=description,
            )

            elapsed = time.time() - t0
            logger.info(
                "  ✅ 成功: %d 行, 分区 %s, 耗时 %.2fs",
                summary["rows_written"],
                summary["partitions"],
                elapsed,
            )
            results["succeeded"].append(factor_id)
            progress.advance(detail=f"{factor_id} -> ok")

        except Exception as e:
            elapsed = time.time() - t0
            logger.error("  ❌ 失败: %s — %s (%.2fs)", type(e).__name__, e, elapsed)
            results["failed"].append({"factor_id": factor_id, "error": str(e)})
            progress.advance(detail=f"{factor_id} -> failed")

    # 汇总
    logger.info("=" * 60)
    logger.info("批量落盘完成")
    logger.info("  成功: %d", len(results["succeeded"]))
    logger.info("  失败: %d", len(results["failed"]))
    logger.info("=" * 60)

    # 打印因子目录
    logger.info("当前因子目录:")
    for f in mat.list_factors():
        logger.info(
            "  %-30s  freq=%-4s  %s → %s  (%s rows)",
            f["factor_id"],
            f["frequency"],
            f.get("start_date", "N/A"),
            f.get("end_date", "N/A"),
            f.get("row_count", "?"),
        )

    return results


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    data_source = build_demo_data_source()
    batch_materialize(FACTOR_CONFIGS, data_source)
