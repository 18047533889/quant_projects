"""Purification 纯化处理入口（纯计算，无 IO）。

接收原始因子值，依次执行:
  1. 缺失值填补 (dynamic imputation)
  2. 去极值 (robust winsorization)
  3. 风险正交化 (risk orthogonalization) — 行业 + 已有因子

返回结构（与 Gateway 对齐）:
    {
        "Status": "PASS",
        "RunId": "pu_...",
        "CheckedAt": "...",
        "Steps": {"Imputation": {...}, "Winsorization": {...}, "Orthogonalization": {...}},
        "factor_values": pd.Series,       # 处理后的因子值
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .low_freq import dynamic_imputation, robust_winsorization, risk_orthogonalization


def run_purification(
    factor_values: pd.Series,
    *,
    imputation_method: str = "forward_fill_decay",
    max_delay: int = 5,
    decay_rate: float = 0.5,
    winsorization_mad_multiplier: float = 3.148,
    orthogonalize_industry: bool = True,
    orthogonalize_existing_factors: bool = True,
    industry_data: pd.DataFrame | None = None,
    industry_asset_col: str = "asset",
    industry_code_col: str = "IndustryCode",
    existing_factor_library: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """运行 Purification 纯化处理。

    Args:
        factor_values: 原始因子值，MultiIndex (timestamp, instrument) Series。
        imputation_method: 缺失值填补方法。``"forward_fill_decay"`` 或 ``"industry_weighted"``。
        max_delay: 前向填充最大延迟期数。
        decay_rate: 前向填充衰减因子。
        winsorization_mad_multiplier: MAD 乘数（默认 3.148 = 正态 3-Sigma）。
        orthogonalize_industry: 是否对行业做正交化。
        orthogonalize_existing_factors: 是否对已有因子库做正交化。
        industry_data: 行业分类数据 DataFrame，至少包含 asset 和 IndustryCode 列。
            若不提供则跳过行业正交化。
        industry_asset_col: 行业数据中资产标识列名。
        industry_code_col: 行业数据中行业代码列名。
        existing_factor_library: 已有因子库 {因子名: Series}。若不提供则跳过因子正交化。

    Returns:
        {
            "factor_values": pd.Series,        # 处理后的因子值
            "purification_segment": {           # 供 afv.json 的 Purification 段
                "Imputation": {"performed": bool, "method": str, ...},
                "Winsorization": {"performed": bool, "mad_multiplier": float},
                "Orthogonalization": {
                    "industry": bool,
                    "existing_factors": bool,
                    "factor_count": int,
                },
                ...
            },
        }
    """
    # 将 MultiIndex Series 转为宽表 panel (index=timestamp, columns=instrument)
    panel = _series_to_panel(factor_values)
    steps = {}
    steps_performed = {"imputation": False, "winsorization": False, "orthogonalization": False}

    # ── Step 1: 缺失值填补 ──
    if imputation_method:
        panel = dynamic_imputation(
            panel,
            method=imputation_method,
            max_delay=max_delay,
        )
        steps_performed["imputation"] = True
        steps["Imputation"] = {"status": "PASS", "method": imputation_method, "max_delay": max_delay}

    # ── Step 2: 去极值 ──
    panel = robust_winsorization(panel)
    steps_performed["winsorization"] = True
    steps["Winsorization"] = {"status": "PASS", "mad_multiplier": winsorization_mad_multiplier}

    # ── Step 3: 正交化 ──
    ortho_detail: dict[str, Any] = {"industry": False, "existing_factors": False, "factor_count": 0}

    if orthogonalize_industry and industry_data is not None:
        # 构建行业哑变量矩阵
        industry_dummies = _build_industry_dummies(
            panel.columns, industry_data, industry_asset_col, industry_code_col
        )
        if industry_dummies is not None:
            weights = pd.Series(1.0, index=panel.columns)
            panel = risk_orthogonalization(panel, industry_dummies, weights)
            ortho_detail["industry"] = True

    if orthogonalize_existing_factors and existing_factor_library:
        risk_factors = _build_risk_factor_matrix(existing_factor_library, panel.index, panel.columns)
        if risk_factors is not None:
            weights = pd.Series(1.0, index=panel.columns)
            panel = risk_orthogonalization(panel, risk_factors, weights)
            ortho_detail["existing_factors"] = True
            ortho_detail["factor_count"] = len(existing_factor_library)

    if ortho_detail["industry"] or ortho_detail["existing_factors"]:
        steps_performed["orthogonalization"] = True
        steps["Orthogonalization"] = {"status": "PASS", **ortho_detail}

    # 宽表转回 MultiIndex Series
    result_series = _panel_to_series(panel)

    run_id = f"pu_{datetime.now():%Y%m%d_%H%M%S_%f}"
    checked_at = datetime.now(timezone.utc).isoformat()

    return {
        "Status": "PASS",
        "RunId": run_id,
        "CheckedAt": checked_at,
        "Steps": steps,
        "factor_values": result_series,
    }


# ============================================================
# 内部工具
# ============================================================


def _series_to_panel(s: pd.Series) -> pd.DataFrame:
    """MultiIndex (timestamp, instrument) Series → 宽表 (index=time, columns=asset)。"""
    return s.unstack(level=1).sort_index()


def _panel_to_series(df: pd.DataFrame) -> pd.Series:
    """宽表 → MultiIndex Series。"""
    return df.stack().sort_index()


def _build_industry_dummies(
    assets: pd.Index, industry_data: pd.DataFrame,
    asset_col: str, code_col: str,
) -> pd.DataFrame | None:
    """根据行业分类数据构建行业哑变量矩阵 (index=asset, columns=industry)。"""
    subset = industry_data[industry_data[asset_col].isin(assets)]
    if subset.empty:
        return None
    dummies = pd.get_dummies(subset.set_index(asset_col)[code_col], prefix="ind")
    # 对齐到所有 assets
    dummies = dummies.reindex(assets, fill_value=0)
    return dummies.astype(float)


def _build_risk_factor_matrix(
    factor_library: dict[str, pd.Series],
    time_index: pd.Index,
    asset_index: pd.Index,
) -> pd.DataFrame | None:
    """将已有因子库拼接为风险因子矩阵 (index=asset, columns=factor_names)。"""
    frames: list[pd.DataFrame] = []
    for name, series in factor_library.items():
        panel = series.unstack(level=1)
        # 取最近一期
        last_row = panel.iloc[-1] if len(panel) > 0 else None
        if last_row is not None:
            frames.append(last_row.to_frame(name))
    if not frames:
        return None
    risk_mat = pd.concat(frames, axis=1).reindex(asset_index)
    return risk_mat
