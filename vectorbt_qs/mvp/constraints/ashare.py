"""
A 股交易约束层

所有约束都以第 ① 层（预处理矩阵）方式实现，不侵入 vectorbt 源码。

约束清单:
- 信号执行滞后：由 runner.execution_lag 负责，不等同于精确 T+1
- 涨跌停：涨停买入和跌停卖出目标拒绝执行
- 停牌过滤：停牌日不调仓
- 印花税：卖出 0.05%
- 最小交易单位：100 股（1 手）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# 停牌过滤
# ============================================================

def filter_suspend(
    target_weights: pd.DataFrame,
    is_suspend: pd.DataFrame,
) -> pd.DataFrame:
    """
    停牌日不调仓：对应 cell 的 target_weight 置 NaN

    target_weights: (n_days, n_assets) 目标权重
    is_suspend:     (n_days, n_assets) 停牌标记（bool / 0/1 / float 均可）
    """
    # 统一转为 bool，兼容 int/float 类型
    suspended = is_suspend.fillna(0).astype(bool)
    return target_weights.where(~suspended)


# ============================================================
# 涨跌停约束
# ============================================================

def constrain_price(
    close: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    target_weights: pd.DataFrame,
    prev_position: pd.DataFrame,
) -> pd.DataFrame:
    """
    成交价不能超出涨跌停范围

    策略：
      买入方向 → 成交价 ≤ min(close, high_limit)
      卖出方向 → 成交价 ≥ max(close, low_limit)

    prev_position: 近似前一交易日持仓（外部迭代）
    """
    delta = target_weights.fillna(0) - prev_position.fillna(0)

    price = close.copy()
    # 买入（delta > 0）：不能高于涨停价
    buy_mask = delta > 0
    price[buy_mask] = price[buy_mask].clip(upper=high_limit[buy_mask])
    # 卖出（delta < 0）：不能低于跌停价
    sell_mask = delta < 0
    price[sell_mask] = price[sell_mask].clip(lower=low_limit[sell_mask])

    return price


def apply_tradeability_constraints(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    is_suspend: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    initial_position: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reject suspended, limit-up buy, and limit-down sell targets.

    Returns constrained targets and their deltas against the last target that
    was allowed to reach the simulator.

    This is deliberately a target-level approximation.  It does not observe
    fills produced later by vectorbt, so partial fills, cash limits, lot
    rounding, and rejected simulator orders can make ``previous_target``
    differ from the actual portfolio position.
    """
    weights = target_weights.copy()
    deltas = pd.DataFrame(np.nan, index=weights.index, columns=weights.columns)
    suspended = is_suspend.fillna(0).astype(bool)
    previous_target = (
        initial_position.reindex(weights.columns).fillna(0.0)
        if initial_position is not None
        else pd.Series(0.0, index=weights.columns)
    )

    for row_index in range(len(weights.index)):
        desired = weights.iloc[row_index].mask(suspended.iloc[row_index])
        delta = desired - previous_target
        order_price = close.iloc[row_index]
        upper_limit = high_limit.iloc[row_index]
        lower_limit = low_limit.iloc[row_index]
        at_upper_limit = order_price.ge(upper_limit) | np.isclose(order_price, upper_limit, equal_nan=False)
        at_lower_limit = order_price.le(lower_limit) | np.isclose(order_price, lower_limit, equal_nan=False)
        blocked = (delta.gt(0.0) & at_upper_limit) | (delta.lt(0.0) & at_lower_limit)
        allowed = desired.mask(blocked)

        weights.iloc[row_index] = allowed
        deltas.iloc[row_index] = (allowed - previous_target).where(allowed.notna())
        previous_target = allowed.where(allowed.notna(), previous_target)

    return weights, deltas


def constrain_price_simple(
    close: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
) -> pd.DataFrame:
    """
    涨跌停简化版：双向夹在涨跌停之间

    不区分买卖方向，保守处理。
    """
    return close.clip(lower=low_limit, upper=high_limit)


# ============================================================
# T+1 约束（折中实现）
# ============================================================

def apply_t1_constraint(
    target_weights: pd.DataFrame,
    prev_target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    兼容辅助函数：把整张目标权重延后一行。

    逻辑：
      前一日目标权重 > 当前目标权重 → 需要卖出
      如果前一日是"从 0 到正"（新开仓），卖出一方被锁定

    注意：这只是信号延迟，不跟踪可卖股数，不能作为精确 T+1 约束。
    runner 当前也不会调用本函数，而是统一使用 execution_lag。
    """
    delayed = target_weights.shift(1)
    if prev_target_weights is not None and not prev_target_weights.empty:
        previous = prev_target_weights.ffill().iloc[-1].reindex(target_weights.columns)
        delayed.iloc[0] = previous
    return delayed


# ============================================================
# 手续费
# ============================================================

A_SHARE_FEES = {
    "commission": 0.00025,      # 佣金 0.025%
    "stamp_tax": 0.0005,        # 印花税 0.05%（仅卖出）
    "transfer_fee": 0.00001,    # 过户费约 0.001%（近似）
}


def resolve_limit_check_prices(
    order_price: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    *,
    mode: str = "execution",
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return effective limits for the selected tradeability convention.

    In ``strict`` mode, touching the upper/lower limit anywhere during the day
    makes a buy/sell untradeable even when the selected reference price is
    VWAP or another price away from the limit.  Effective limits are set to the
    reference price for touched cells so downstream directional checks retain
    one consistent rule.
    """
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"execution", "strict"}:
        raise ValueError("limit_check_mode 仅支持 execution 或 strict")
    for name, value in (("limit_price_rtol", rtol), ("limit_price_atol", atol)):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} 必须是非负有限数")
    if normalized_mode == "execution":
        return high_limit, low_limit
    if high is None or low is None:
        raise ValueError("limit_check_mode=strict 要求 High 和 Low 行情")

    index, columns = order_price.index, order_price.columns
    aligned_high = high.reindex(index=index, columns=columns)
    aligned_low = low.reindex(index=index, columns=columns)
    aligned_upper = high_limit.reindex(index=index, columns=columns)
    aligned_lower = low_limit.reindex(index=index, columns=columns)

    upper_close = pd.DataFrame(
        np.isclose(
            aligned_high.to_numpy(dtype=float),
            aligned_upper.to_numpy(dtype=float),
            rtol=float(rtol),
            atol=float(atol),
            equal_nan=False,
        ),
        index=index,
        columns=columns,
    )
    lower_close = pd.DataFrame(
        np.isclose(
            aligned_low.to_numpy(dtype=float),
            aligned_lower.to_numpy(dtype=float),
            rtol=float(rtol),
            atol=float(atol),
            equal_nan=False,
        ),
        index=index,
        columns=columns,
    )
    upper_touched = aligned_high.ge(aligned_upper) | upper_close
    lower_touched = aligned_low.le(aligned_lower) | lower_close

    effective_upper = aligned_upper.mask(upper_touched, order_price)
    effective_lower = aligned_lower.mask(lower_touched, order_price)
    # Preserve missing intraday extremes so Accurate can raise instead of
    # silently treating an unverifiable cell as tradable.
    effective_upper = effective_upper.mask(aligned_high.isna())
    effective_lower = effective_lower.mask(aligned_low.isna())
    return effective_upper, effective_lower


def build_ashare_fees(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    prev_position: pd.DataFrame,
    commission: float = A_SHARE_FEES["commission"],
    stamp_tax: float = A_SHARE_FEES["stamp_tax"],
    transfer_fee: float = A_SHARE_FEES["transfer_fee"],
    deltas: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构造 A 股分方向费率矩阵

    返回
    ----
    fees : (n_days, n_assets) 百分比费率
    fixed_fees : (n_days, n_assets) 固定费用
    """
    if deltas is None:
        previous_target = target_weights.ffill().shift(1).fillna(0.0)
        deltas = (target_weights - previous_target).where(target_weights.notna())

    # 基础佣金和过户费（买卖都收）
    fees = pd.DataFrame(commission + transfer_fee, index=close.index, columns=close.columns)
    # 卖出加印花税
    fees[deltas < 0] += stamp_tax

    fixed_fees = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    return fees, fixed_fees


# ============================================================
# 一键约束
# ============================================================

def apply_ashare_constraints(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    is_suspend: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    prev_position: Optional[pd.DataFrame] = None,
    order_price: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    一键应用所有 A 股约束

    输入
    ----
    close : 复权收盘价 (n_days, n_assets)
    target_weights : 目标权重矩阵
    is_suspend : 停牌标记
    high_limit, low_limit : 涨跌停价
    prev_position : 近似前日持仓（None = 从 0 开始）

    返回
    ----
    {
        "target_weights": 约束后的目标权重,
        "price":          约束后的成交价,
        "fees":           费率矩阵,
        "fixed_fees":     固定费用矩阵,
    }
    """
    if prev_position is None:
        prev_position = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if order_price is None:
        order_price = close

    initial_position = prev_position.iloc[0] if not prev_position.empty else None
    weights, deltas = apply_tradeability_constraints(
        order_price,
        target_weights,
        is_suspend,
        high_limit,
        low_limit,
        initial_position=initial_position,
    )

    fees, fixed_fees = build_ashare_fees(close, weights, prev_position, deltas=deltas)

    return {
        "target_weights": weights,
        "price": order_price,
        "fees": fees,
        "fixed_fees": fixed_fees,
    }
