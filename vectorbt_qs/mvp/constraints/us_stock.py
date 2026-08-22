"""
美股交易约束层

约束清单:
- SSR 限制（跌超 10% 禁做空）
- 退市标注（退市日权重归零）
- SEC/FINRA 费率
- ADR 去重
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


# ============================================================
# SSR 限制（Short Sale Restriction）
# ============================================================

def apply_ssr_filter(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    SSR Rule 201: 前一日跌幅 ≥ 10% → 当日 + 次日禁止新增做空

    不影响平空仓（target_weight 从负变 0 是平空，保持不变）
    只限制 target_weight < 0 且比前一日更负的情况（新增做空）
    """
    daily_ret = close.pct_change(fill_method=None)
    ssr_triggered = daily_ret <= -0.10
    # 触发日 + 次日
    ssr_active = ssr_triggered | ssr_triggered.shift(1, fill_value=False)

    constrained = target_weights.copy()
    previous_target = pd.Series(0.0, index=target_weights.columns)
    for row_index in range(len(constrained.index)):
        current_target = constrained.iloc[row_index]
        # SSR only blocks entering or increasing a short position.  A lower
        # positive target is a long sale and must remain allowed.  When a
        # target crosses from long to short, allow the long to close but stop
        # the target at zero.
        previous_short = previous_target.clip(upper=0.0)
        increasing_short = current_target < previous_short
        current_target = current_target.mask(
            ssr_active.iloc[row_index] & increasing_short,
            previous_short,
        )
        constrained.iloc[row_index] = current_target
        previous_target = current_target.where(current_target.notna(), previous_target)

    return constrained


# ============================================================
# 退市处理（暂时用占位逻辑）
# ============================================================

def apply_delisting(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    delist_info: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    退市日 weight 归零

    delist_info: DataFrame with columns [ticker, delist_date, delist_return]
                 暂无数据源时返回原始矩阵
    """
    if delist_info is None or delist_info.empty:
        # 无退市数据，跳过
        return target_weights

    weights = target_weights.copy()
    for _, row in delist_info.iterrows():
        ticker = row["ticker"]
        delist_date = pd.Timestamp(row["delist_date"])
        if ticker not in weights.columns or pd.isna(delist_date):
            continue

        # Delisting dates can fall on weekends/holidays or between sparse
        # rebalance rows.  Liquidate from the first available row on or after
        # the effective date rather than requiring an exact index match.
        if weights.index.tz is None and delist_date.tzinfo is not None:
            delist_date = delist_date.tz_localize(None)
        elif weights.index.tz is not None and delist_date.tzinfo is None:
            delist_date = delist_date.tz_localize(weights.index.tz)
        effective_rows = weights.index[weights.index >= delist_date]
        if len(effective_rows) > 0:
            weights.loc[effective_rows[0]:, ticker] = 0.0

    return weights


# ============================================================
# ADR 去重
# ============================================================

def filter_adr_duplicates(
    target_weights: pd.DataFrame,
    is_adr: pd.Series,
    adr_mapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    按 ``is_adr`` 标识排除 ADR 标的。

    is_adr: Series index=ticker, values=bool (True=ADR)

    ``adr_mapping`` 预留给未来的“仅在底层证券同时存在时去重”逻辑；
    当前数据契约没有稳定的底层证券映射，因此这里实现的是显式 ADR 排除策略。
    """
    adr_tickers = is_adr.fillna(False).astype(bool)
    adr_tickers = adr_tickers[adr_tickers].index
    weights = target_weights.copy()

    for adr in adr_tickers:
        if adr in weights.columns:
            # 当前策略：显式排除 ADR。
            weights[adr] = 0.0

    return weights


# ============================================================
# 手续费
# ============================================================

US_FEES = {
    "commission_per_share": 0.0035,  # IBKR 约 $0.0035/股
    "sec_fee_per_million": 8.00,     # SEC Section 31: ~$8/$1M 成交额
    "finra_taf_per_share": 0.000119, # FINRA TAF: $0.000119/股 (上限 $5.95)
}


def build_us_fees(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    prev_position: pd.DataFrame,
    init_cash: float = 1e6,
) -> pd.DataFrame:
    """
    构造美股费率矩阵

    简化：合并为单一百分比费率（按成交额计）

    返回 fees: (n_days, n_assets) 百分比费率
    """
    # 简化：合并所有费用为约 0.01%
    # SEC $8/$1M ≈ 0.0008%，FINRA + 佣金按中盘股折合约 0.02%
    # 总计约 0.03% 单边
    fees = pd.DataFrame(0.0003, index=close.index, columns=close.columns)

    return fees


# ============================================================
# 一键约束
# ============================================================

def apply_us_constraints(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    is_adr: Optional[pd.Series] = None,
    delist_info: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    一键应用所有美股约束

    返回
    ----
    {
        "target_weights": 约束后的目标权重,
        "price":          close（美股不约束成交价）,
        "fees":           费率矩阵,
    }
    """
    weights = target_weights.copy()

    # 1. SSR
    weights = apply_ssr_filter(close, weights)

    # 2. 退市
    weights = apply_delisting(close, weights, delist_info)

    # 3. ADR 去重
    if is_adr is not None:
        weights = filter_adr_duplicates(weights, is_adr)

    # 4. 手续费
    fees = build_us_fees(close, weights, pd.DataFrame(0, index=close.index, columns=close.columns))

    return {
        "target_weights": weights,
        "price": close,        # 美股不约束成交价
        "fees": fees,
    }
