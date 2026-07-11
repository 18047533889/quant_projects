"""Assetization 计算引擎（纯计算，无 IO）。

职责:
  1. 从 manifest dict 提取公式（优先 code，其次 expr）
  2. 调用 factor_engine 全量计算
  3. 返回按日分区的因子值结果 + afv.json 所需的 Assetization 段

返回结构（与 Gateway 对齐）:
    {
        "Status": "PASS",
        "RunId": "as_...",
        "CheckedAt": "...",
        "Steps": {"ExtractFormula": {...}, "BuildConfig": {...}, ...},
        "factor_id": "...",
        "candidate_id": "...",
        "expression_type": "...",
        "daily_values": {"2024-01-01": pd.Series, ...},
    }
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def run_assetization(
    manifest: dict,
    *,
    market_data_root: str | None = None,
    cache_root: str | None = None,
) -> dict[str, Any]:
    """运行 Assetization 计算。

    Args:
        manifest: 已解析的 manifest dict（pipeline 传入）。
        market_data_root: 行情数据根路径。None 时使用环境变量。
        cache_root: 持久化缓存根路径。None 则不使用缓存。

    Returns:
        包含 Assetization 段信息和 daily_values 的 dict。
    """
    run_id = f"as_{datetime.now():%Y%m%d_%H%M%S_%f}"
    checked_at = datetime.now(timezone.utc).isoformat()
    steps: dict[str, dict] = {}

    # Step 1: 提取 formula
    formula, calc_mode = _extract_formula(manifest)
    candidate_id = manifest.get("candidate_id", "unknown")
    steps["ExtractFormula"] = {
        "status": "PASS",
        "expression_type": calc_mode,
        "formula_preview": formula[:80] + ("..." if len(formula) > 80 else ""),
    }

    # Step 2: 调用 factor_engine
    import yaml as _yaml
    market = manifest.get("market", "")
    data_root = market_data_root or _resolve_data_root(market)
    fields = _get_fields(market)
    ts_col, inst_col = ("TradeDate", "Symbol") if market == "ashare" else ("window_start", "ticker")

    yaml_config = {
        "factor": {"name": candidate_id, "expr": formula, "calc_mode": calc_mode, "freq": "1d"},
        "data_source": {"type": "parquet", "root": data_root,
                        "timestamp_col": ts_col, "instrument_col": inst_col, "fields": fields},
        "backend": {"type": "pandas"},
        "engine": {"enable_cache": True, "tiny_run": False, "cache_root": cache_root},
    }

    _ensure_fe_path()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        _yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
        tmp_path = f.name

    from factor_engine.runtime.engine import FactorEngine
    engine, factor = FactorEngine.from_config(tmp_path, cache_root=cache_root)
    result = engine.run(factor)
    Path(tmp_path).unlink(missing_ok=True)

    series: pd.Series = result.get("result")
    if series is None or not isinstance(series, pd.Series):
        raise ValueError(f"因子引擎返回结果为空或类型错误: {type(series)}")

    steps["RunEngine"] = {"status": "PASS", "total_rows": len(series)}

    # Step 3: 按日拆分
    daily = _split_by_day(series)
    steps["SplitDaily"] = {"status": "PASS", "total_days": len(daily), "date_range": f"{min(daily)} ~ {max(daily)}" if daily else "N/A"}

    # 生成 factor_id
    from assetization.scripts.registry import generate_factor_id, extract_coordinates
    coords = extract_coordinates(manifest)
    factor_id = generate_factor_id(coords, seed={"formula": formula, "candidate_id": candidate_id})

    return {
        "Status": "PASS",
        "RunId": run_id,
        "CheckedAt": checked_at,
        "Steps": steps,
        "factor_id": factor_id,
        "candidate_id": candidate_id,
        "expression_type": calc_mode,
        "daily_values": daily,
    }


def _extract_formula(manifest: dict) -> tuple[str, str]:
    """从 manifest 提取公式。优先级: code > formula。"""
    expr_type = manifest.get("expression_type", "dsl")
    if expr_type in ("python", "code"):
        code = manifest.get("code") or manifest.get("formula", "")
        if not code:
            raise ValueError("expression_type=python 但缺少 code 和 formula")
        return code, "code"
    formula = manifest.get("formula", "")
    if not formula:
        raise ValueError("manifest 缺少 formula 字段")
    return formula, "expr"


def _resolve_data_root(market: str) -> str:
    import os
    return os.environ.get("ASHARE_DATA_ROOT" if market == "ashare" else "US_STOCK_DATA_ROOT", "")


def _get_fields(market: str) -> dict[str, str]:
    return {"close": "Close", "open": "Open", "high": "High", "low": "Low",
            "volume": "Volume", "vwap": "Vwap"}


def _ensure_fe_path():
    p = Path(__file__).resolve().parents[1]
    fe = str(p / "factor_engine")
    if fe not in sys.path:
        sys.path.insert(0, fe)


def _split_by_day(series: pd.Series) -> dict[str, pd.Series]:
    """将 MultiIndex (timestamp, instrument) Series 按日拆分。"""
    if not isinstance(series.index, pd.MultiIndex):
        return {"full": series}
    idx = series.index
    timestamps = idx.get_level_values(0)
    daily: dict[str, pd.Series] = {}
    for dt in sorted(timestamps.normalize().unique()):
        date_str = dt.strftime("%Y-%m-%d")
        mask = timestamps.normalize() == dt
        daily[date_str] = series.loc[mask]
    return daily
