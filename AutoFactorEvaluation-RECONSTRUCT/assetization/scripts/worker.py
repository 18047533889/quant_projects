"""
Assetization 工作进程

处理 Gateway 通过后的因子:
    1. 读取 afv.json 获取公式和元信息
    2. 加载市场数据
    3. 生成 factor_id
    4. 计算公式值 → data.parquet
    5. 写入 afv.json(追加 Assetization 段)
    6. 转移到 tier0/assetization_temp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .registry import generate_factor_id, extract_coordinates
from .compute import FormulaEvaluator

logger = logging.getLogger("assetization.worker")


def process_factor_dir(
    factor_dir: str | Path,
    *,
    market_data_path: str | Path,
    output_base: str | Path | None = None,
    temp_base: str | Path | None = None,
) -> tuple[str, str]:
    """处理单个已通过 Gateway 审查的因子。

    Args:
        factor_dir: tier1/gateway_pass_base 中的因子目录路径。
        market_data_path: 市场数据根路径（StockDailyBar 等子目录的父目录）。
        output_base: assetization_raw_factor_base 根目录。默认自动推导。
        temp_base: tier0/assetization_temp 路径。默认自动推导。

    Returns:
        (factor_id, output_dir): 生成的因子 ID 和输出目录路径。
    """
    import sys as _sys
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))

    factor_dir = Path(factor_dir)
    afv_path = factor_dir / "afv.json"

    if not afv_path.exists():
        raise FileNotFoundError(f"因子目录缺少 afv.json: {factor_dir}")

    afv = json.loads(afv_path.read_text(encoding="utf-8"))
    formula = afv.get("formula", "")
    gateway_info = afv.get("Gateway", {})
    campaign_id = afv.get("campaign_id", "")
    candidate_id = afv.get("candidate_id", factor_dir.name)

    if gateway_info.get("Label") != "Pass":
        raise ValueError(f"因子未通过 Gateway 审查: {candidate_id}, Label={gateway_info.get('Label')}")

    if not formula:
        raise ValueError(f"因子 {candidate_id} 公式为空")

    logger.info("Assetization 开始: candidate_id=%s formula=%s", candidate_id, formula[:80])

    # 解析坐标
    # 从候选池 manifest 中读取 domain 信息（暂用默认值）
    manifest_path = factor_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    config = {
        "signal_structure": "cross_sectional",
        "asset_class": "equity",
        "frequency_bucket": manifest.get("frequency_bucket", "1d"),
        "domain_root": manifest.get("domain_root", "us_stock"),
        "domain": "price_volume",
    }

    # 生成 factor_id（基于公式+配置的稳定哈希）
    factor_id, coordinates = _generate_factor_id(config, formula, candidate_id)

    # 加载市场数据
    market_df = _load_market_data(market_data_path, config)

    # 计算行情数据时间区间
    date_col_mkt = "TradeDate" if "TradeDate" in market_df.columns else "datetime"
    market_start = market_df[date_col_mkt].min()
    market_end = market_df[date_col_mkt].max()
    market_start_str = str(market_start.date()) if hasattr(market_start, "date") else str(market_start)[:10]
    market_end_str = str(market_end.date()) if hasattr(market_end, "date") else str(market_end)[:10]
    logger.info("  行情数据时间区间: %s ~ %s", market_start_str, market_end_str)

    # 计算公式值
    evaluator = FormulaEvaluator()
    factor_values = evaluator.evaluate(formula, market_df)
    logger.info("  计算完成: shape=%s", factor_values.shape)

    # 构建输出 DataFrame（与市场数据对齐的行序）
    output_df = pd.DataFrame({
        "TradeDate": market_df.get("TradeDate", market_df.get("datetime", pd.NaT)),
        "Symbol": market_df.get("Symbol", market_df.get("asset", "")),
        "factor_id": factor_id,
        "factor_value": factor_values,
    })
    # 剔除 NaN 行
    output_df = output_df.dropna(subset=["factor_value"]).reset_index(drop=True)

    # 确定输出路径
    if not output_base or not temp_base:
        raise ValueError("assetization: output_base 和 temp_base 必须从外部配置传入")
    output_base = Path(output_base)
    temp_base = Path(temp_base)
    if not temp_base.exists():
        raise FileNotFoundError(
            f"Assetization 临时目录不存在: {temp_base}\n"
            "  → 请确保该路径已在配置文件中声明且已被创建"
        )

    # 使用 factor_id 作为目录名
    temp_dir = temp_base / factor_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 写入 data/ 目录，按 TradeDate 分片为每日一个 parquet 文件
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    date_col = "TradeDate" if "TradeDate" in output_df.columns else "datetime"
    for date_val, group in output_df.groupby(output_df[date_col].dt.date if hasattr(output_df[date_col], "dt") else output_df[date_col]):
        date_str = str(date_val)
        file_path = data_dir / f"{date_str}.parquet"
        group.to_parquet(file_path, index=False)
    logger.info("  写入 data/: %s → %d 个日频文件 (%d 行)", data_dir, len(output_df.groupby(date_col)), len(output_df))

    # 写入 afv.json（保留上游信息 + 追加 Assetization 段）
    _write_afv(temp_dir, afv, factor_id, coordinates, formula, candidate_id, len(output_df),
               market_start=market_start_str, market_end=market_end_str)

    # 复制 manifest.json（保留上游元信息）
    if manifest_path.exists():
        (temp_dir / "manifest.json").write_text(
            json.dumps({**manifest, "factor_id": factor_id}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info("  ✅ 完成: factor_id=%s temp=%s", factor_id, temp_dir)
    return factor_id, str(temp_dir)


def _generate_factor_id(config: dict, formula: str, candidate_id: str) -> tuple[str, dict]:
    """基于公式和配置生成稳定的 factor_id。"""
    coordinates = extract_coordinates(config)
    coordinates["domain"] = config.get("domain", "price_volume")

    seed = {
        "candidate_id": candidate_id,
        "formula": formula,
        "Config": config,
    }
    factor_id = generate_factor_id(coordinates, seed=seed)
    return factor_id, coordinates


def _load_market_data(market_data_path: str | Path, config: dict) -> pd.DataFrame:
    """从 MARKET_DATA_PATH 加载行情数据。

    数据格式: {market_data_path}/StockDailyBar/{YYYY-MM-DD}.parquet
    每文件含: TradeDate, Symbol, Open, High, Low, Close, Volume, ...
    """
    bar_dir = Path(market_data_path)
    if not bar_dir.exists():
        # 尝试回退：上级目录 + daily
        bar_dir = bar_dir.parent / "daily"
        if not bar_dir.exists():
            raise FileNotFoundError(
                f"行情数据目录不存在: {market_data_path}\n"
                "  → 请确保 market_data.path/market_data.standard 配置正确"
            )

    # 使用全局缓存加载行情数据
    from cache_utils import load_market_data_cached
    market_df = load_market_data_cached(
        market_data_path=bar_dir,
    )
    # 确保必要列存在
    if "TradeDate" in market_df.columns:
        market_df["TradeDate"] = pd.to_datetime(market_df["TradeDate"])
        market_df = market_df.sort_values(["Symbol", "TradeDate"]).reset_index(drop=True)
    elif "datetime" in market_df.columns:
        market_df["datetime"] = pd.to_datetime(market_df["datetime"])
        market_df = market_df.sort_values(["asset", "datetime"]).reset_index(drop=True)

    logger.info("  市场数据加载完成: %s → %d 行 %d 列",
                mdp, len(market_df), len(market_df.columns))
    return market_df


def _write_afv(
    temp_dir: Path,
    source_afv: dict,
    factor_id: str,
    coordinates: dict,
    formula: str,
    candidate_id: str,
    row_count: int,
    market_start: str = "",
    market_end: str = "",
):
    """写入 afv.json（追加 Assetization 段）。"""
    now = datetime.now(timezone.utc)
    run_id = f"assetization_{now.strftime('%Y%m%d_%H%M%S')}_{factor_id[-8:]}"

    afv = dict(source_afv)
    afv["factor_id"] = factor_id
    afv["Assetization"] = {
        "Label": "Materialized",
        "status": "passed",
        "factor_id": factor_id,
        "run_id": run_id,
        "materialized_at": now.isoformat().replace("+00:00", "Z"),
        "coordinates": coordinates,
        "market_period": {"start": market_start, "end": market_end},
        "row_count": row_count,
        "formula": formula,
        "candidate_id": candidate_id,
    }

    (temp_dir / "afv.json").write_text(
        json.dumps(afv, indent=2, ensure_ascii=False), encoding="utf-8"
    )



