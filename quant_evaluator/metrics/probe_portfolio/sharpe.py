"""Sharpe / Sortino 等基于真实 daily PnL 的绩效指标（不依赖重叠 label）。

AlphaPROBE 重构规范 §7/§13：20 日 vwap→vwap label 下，禁止对 overlapping
forward return 直接年化 Sharpe。本模块只接受**日频** PnL/收益序列
（由 probe_portfolio.compute_cohort_pnl 产生），按日频 std 年化，
不掺入持仓期长度 —— 持仓重叠的影响已经体现在 daily PnL 的真实自相关里。

复用库内既有实现：
- portfolio_stats.compute_sharpe_ratio / compute_sortino_ratio /
  compute_calmar_ratio / compute_maximum_drawdown / compute_win_rate。
- risk/drawdown_analysis.compute_drawdown_statistics /
  compute_drawdown_duration。
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from quant_evaluator.metrics.portfolio_stats import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_maximum_drawdown,
    compute_win_rate,
)
from quant_evaluator.metrics.risk.drawdown_analysis import (
    compute_drawdown_statistics,
    compute_drawdown_duration,
)

EPS = 1e-12


def _as_1d(returns: np.ndarray, name: str = "returns") -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维日频收益序列，got ndim={arr.ndim}")
    return arr


def compute_drawdown_persistence(returns: np.ndarray) -> float:
    """DrawdownPersistence = 0.5*normalized(MaxDDDuration) + 0.5*normalized(TimeUnderWater)。

    归一化口径（全程 [0,1]，越大越差）：
    - MaxDDDuration 按持有期上限归一：duration_days / (10 * holding_days)，
      用默认 holding=20 即上限 200 个交易日；调用方可用 holding_days 参数覆盖。
    - TimeUnderWater 按百分比 / 100 归一。

    Args:
        returns : (T,) 日频 PnL/收益序列。
        holding_days : 用于 MaxDDDuration 归一的上限基准（默认 20，即
            DDP 满分为回撤持续 200 个交易日；配合 20 日 cohort 的口径）。

    Returns:
        float in [0, 1]；数据不足或无回撤时返回 0.0（无回撤 = 最不持久）。
    """
    returns = _as_1d(returns)
    valid = np.isfinite(returns)
    if np.sum(valid) < 10:
        return 0.0
    ret = returns[valid]

    dd_stats = compute_drawdown_statistics(ret, min_periods=10)
    dd_duration = compute_drawdown_duration(ret, min_periods=10)

    max_dd_duration = float(dd_duration["max_drawdown_duration"])
    time_underwater = float(dd_stats["time_underwater_pct"])  # 百分数

    norm_duration = min(max_dd_duration / (10.0 * 20.0), 1.0)
    norm_tuw = min(time_underwater / 100.0, 1.0)
    return float(0.5 * norm_duration + 0.5 * norm_tuw)


def compute_rolling_sharpe_quantile(
    returns: np.ndarray,
    window: int = 60,
    quantile: float = 0.20,
    periods_per_year: int = 252,
    min_periods: int = 30,
) -> float:
    """Rolling Sharpe 的 Q20 分位（衡量绩效在坏时期的稳定下限）。

    Args:
        returns : (T,) 日频收益序列。
        window  : 滚动窗口（默认 60 个交易日）。
        quantile: 分位（默认 0.20 → Q20）。
        periods_per_year / min_periods: 传递给 compute_sharpe_ratio。

    Returns:
        float；数据不足时 NaN。
    """
    returns = _as_1d(returns)
    T = returns.shape[0]
    if T < window:
        return np.nan
    roll = np.full(T, np.nan)
    for t in range(window - 1, T):
        seg = returns[t - window + 1 : t + 1]
        if np.sum(np.isfinite(seg)) >= min_periods:
            roll[t] = float(compute_sharpe_ratio(
                seg, risk_free_rate=0.0,
                periods_per_year=periods_per_year, min_periods=min_periods,
            ))
    valid = roll[np.isfinite(roll)]
    if valid.size == 0:
        return np.nan
    return float(np.quantile(valid, quantile))


def compute_positive_month_ratio(
    returns: np.ndarray,
    periods_per_month: int = 21,
) -> float:
    """Positive Month Ratio：自然月度（21 交易日）内合计收益 > 0 的比例。

    Returns:
        float in [0, 1]；数据不足一个月时 NaN。
    """
    returns = _as_1d(returns)
    valid = np.isfinite(returns)
    if np.sum(valid) < periods_per_month:
        return np.nan
    ret = returns[valid]
    n_months = ret.shape[0] // periods_per_month
    if n_months < 1:
        return np.nan
    months = ret[: n_months * periods_per_month].reshape(n_months, periods_per_month)
    month_rets = np.prod(1.0 + months, axis=1) - 1.0
    return float(np.mean(month_rets > 0.0))


def compute_annualized_return(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """年化复利收益（基于真实日频 PnL 序列）。"""
    returns = _as_1d(returns)
    valid = np.isfinite(returns)
    if np.sum(valid) < 2:
        return np.nan
    ret = returns[valid]
    n = ret.shape[0]
    total = float(np.prod(1.0 + ret))
    if total <= 0:
        return np.nan
    return float(total ** (periods_per_year / n) - 1.0)


def compute_annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """年化波动率（基于真实日频 PnL 序列）。"""
    returns = _as_1d(returns)
    valid = np.isfinite(returns)
    if np.sum(valid) < 2:
        return np.nan
    ret = returns[valid]
    std = float(np.std(ret, ddof=1))
    if std <= EPS:
        return 0.0
    return std * np.sqrt(periods_per_year)


def compute_portfolio_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int = 20,
    rolling_window: int = 60,
    rolling_quantile: float = 0.20,
) -> dict:
    """真实 daily PnL 指标全家桶（Sharpe / Sortino / Calmar / MaxDD 等）。

    全部基于日频序列，复用 QE 库 portfolio_stats 既有实现，与单指标函数
    同源（满足「主表与详情卡同源」的单一真相口径要求）。

    Args:
        returns : (T,) 日频 PnL/收益序列（真实 cohort 组合产生）。
        periods_per_year / risk_free_rate / min_periods: 传递 Sharpe/Sortino/Calmar。
        rolling_window / rolling_quantile: Rolling Sharpe 参数。

    Returns:
        dict（标量）：
            sharpe / sortino / calmar / max_drawdown / win_rate /
            annualized_return / annualized_volatility /
            max_drawdown_duration / time_underwater_pct /
            drawdown_persistence / rolling_sharpe_q20 / positive_month_ratio
    """
    returns = _as_1d(returns)
    metrics = {
        "sharpe": float(compute_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year, min_periods=min_periods,
        )),
        "sortino": float(compute_sortino_ratio(
            returns, risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year, min_periods=min_periods,
        )),
        "calmar": float(compute_calmar_ratio(
            returns, periods_per_year=periods_per_year, min_periods=min_periods,
        )),
        "max_drawdown": float(compute_maximum_drawdown(
            returns, missing_return_policy="zero_fill",
        )[0]),
        "win_rate": float(compute_win_rate(returns)),
        "annualized_return": compute_annualized_return(returns, periods_per_year),
        "annualized_volatility": compute_annualized_volatility(returns, periods_per_year),
        "rolling_sharpe_q20": compute_rolling_sharpe_quantile(
            returns, window=rolling_window, quantile=rolling_quantile,
            periods_per_year=periods_per_year, min_periods=min(30, rolling_window),
        ),
        "positive_month_ratio": compute_positive_month_ratio(returns),
        "drawdown_persistence": compute_drawdown_persistence(returns),
    }
    return metrics
