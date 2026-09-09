"""固定名义权重、独立 cohort 的研究 probe PnL（非实盘执行认证）。

背景（AlphaPROBE 重构规范 §7/§13）
----------------------------------
20 日 vwap→vwap label 下，绝对禁止用 overlapping forward return 直接算
Sharpe：每天取一个 D10-D1 forward return（持仓期 20 日、日度重叠），
std → annualize 会把高度自相关的重叠收益当作独立样本，严重虚高 Sharpe。
必须构造真实 cohort portfolio：每天新建一个 1/20 资本的 cohort，
持有 20 个交易日后平仓，由每个交易日的真实 PnL 计算风险收益指标。

设计要点
--------
- 每日 t：Factor(t) 分位 → 下一交易日 VWAP 入场 → 建立 H=20 日 cohort。
- 每个 cohort：Long D10 每票 +0.5/N_D10，Short D1 每票 -0.5/N_D1。
- 每个 cohort 只分配 1/H 总资本，当天总仓位 W_t = Σ active_cohort_weight / H。
- 由真实 daily PnL_t 计算 Sharpe / Sortino / Calmar / MaxDD / DD Duration /
  TimeUnderWater / Rolling Sharpe(Q20) / Positive Month Ratio。
- 同时产出 long_short（D10-D1，Signal Quality Diagnostic）与 D10 long-only
  active return（相对基准）两套；A 股 D1 short 不一定可执行，
  防「Long-Short 只靠 D1 极差驱动」。
- 分位语义复用 QE 库内 metrics/quantile.py（assign_quantiles，QE-Q-P0-001
  平局策略契约），不重写分位逻辑。
- 回撤统计复用 portfolio_stats.compute_maximum_drawdown 与
  risk/drawdown_analysis.py 既有实现，不形成第三套回测代码。

输入矩阵（均以同一天为「行」）
--------------------------------
- factor_values : (T, N) 因子值（或 FactorBatch.values 去掉最后一维）
- next_ret      : (T, N) 当日持有收益 r_t = P_t / P_{t-1} - 1（vwap→vwap）
- next_vwap     : (T, N) 当日 VWAP 成交价（入场/出场）
- next_open     : (T, N) 当日开盘价（仅用于「停牌=无法在 VWAP 成交」的
                  可行性检查；缺省时退回全可交易）
- weight_matrix : (T, N) 市值权重（基准组合，用于 long-only active return）

约定
----
- 分位 0  = 因子值最低组（bottom, D1），分位 NQ-1 = 因子值最高组（top, D10）。
  （与 QE 库 quantile.py 的 QE2-P0-001 编码一致）
- 信号日 t，入场 t+1，持有 H 个入场后价格区间，计划退出 t+1+H。
- 首个收益是 P_(t+2)/P_(t+1)-1；样本末尾不生成强制平仓。
- 固定名义权重不等同买入持有股数账本；逐 cohort 毛成本不等同净额成交成本。
- 出场也用当日 VWAP 成交（免滑点、gross 口径），成本以 per-side cost 扣除。
- 每票每笔单边成本按成交名义额的比例扣减（默认 0 = gross 语义），并留存
  1x/2x/3x 成本情景接口。
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np

from quant_evaluator.metrics.quantile import assign_quantiles
from quant_evaluator.contracts.portfolio_schedule import cohort_schedule

EPS = 1e-12


def _validate_2d(name: str, arr: Optional[np.ndarray], T: int, N: int) -> None:
    if arr is None:
        return
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape != (T, N):
        raise ValueError(
            f"{name} 必须为 (T, N) 二维数组且与 factor 形状一致，"
            f"got shape={arr.shape}, expected={(T, N)}"
        )


def build_cohort_panels(
    factor_values: np.ndarray,
    n_quantiles: int = 10,
    min_bucket_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """每日 factor 分位 → D1/D10 桶成员矩阵。

    复用 QE 库 metrics/quantile.assign_quantiles（QE-Q-P0-001 平局策略契约，
    默认 method="max"：分位边界上并列值归入更高分位）。

    Returns:
        (long_mask, short_mask)，均为 (T, N) bool；
        long_mask[t, i] = 第 t 日因子值属于最高分位（D10）桶。
    """
    factor_values = np.asarray(factor_values, dtype=np.float64)
    if factor_values.ndim != 2:
        raise ValueError(f"factor_values 必须为 (T, N) 二维数组，got ndim={factor_values.ndim}")

    quantile_ids = assign_quantiles(
        factor_values, n_quantiles=n_quantiles, method="max"
    )  # (T, N) int32；NaN → -1

    top_bucket = n_quantiles - 1
    long_mask = quantile_ids == top_bucket
    short_mask = quantile_ids == 0

    T, N = factor_values.shape
    if min_bucket_size > 1:
        long_counts = np.sum(long_mask, axis=1)
        short_counts = np.sum(short_mask, axis=1)
        degenerate = (long_counts < min_bucket_size) | (short_counts < min_bucket_size)
        long_mask = long_mask & (~degenerate[:, np.newaxis])
        short_mask = short_mask & (~degenerate[:, np.newaxis])

    return long_mask, short_mask


def build_quantile_masks(
    factor_values: np.ndarray,
    n_quantiles: int = 10,
    min_bucket_size: int = 1,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray:
    """每日 factor 分位 → 全分位成员矩阵（便于长期限诊断/对拍）。

    Returns:
        masks : (n_quantiles, T, N) bool；masks[q, t, i] 表示第 t 日 i 票
        属于分位 q（0=最低组 D1 ... n_quantiles-1=最高组 D10）。
    """
    if isinstance(min_bucket_size, (bool, np.bool_)) or not isinstance(min_bucket_size, (int, np.integer)) or min_bucket_size < 1:
        raise ValueError("min_bucket_size must be a positive integer")
    factor_values = np.asarray(factor_values, dtype=np.float64)
    if factor_values.ndim != 2:
        raise ValueError(f"factor_values 必须为 (T, N) 二维数组，got ndim={factor_values.ndim}")
    quantile_ids = assign_quantiles(
        factor_values, n_quantiles=n_quantiles, method="max"
    )
    T, N = factor_values.shape
    masks = np.zeros((n_quantiles, T, N), dtype=bool)
    for q in range(n_quantiles):
        masks[q] = quantile_ids == q
    counts = np.sum(masks, axis=2)  # (Q,T): actual members, never total universe N.
    valid = counts >= min_bucket_size
    masks &= valid[:,:,None]
    if return_diagnostics:
        distinct = np.asarray([len(np.unique(row[np.isfinite(row)])) for row in factor_values])
        return masks, {"bucket_counts": counts, "bucket_valid": valid,
                       "distinct_levels": distinct, "actual_bucket_count": np.sum(counts > 0, axis=0),
                       "degradation_reason": np.where(valid, "", "INSUFFICIENT_BUCKET_MEMBERS"),
                       "policy_version": "quantile_members.v5.1"}
    return masks


def _is_tradable(next_vwap: np.ndarray, next_open: Optional[np.ndarray]) -> np.ndarray:
    """可交易性检查：有 VWAP 成交价，且（若给 open）开盘非 NaN。

    停牌日 VWAP 缺失或 open 缺失 → 该日该票视为不可交易（无法在 VWAP 成交）。
    保守口径：NaN 一律不可交易（不静默按 0 收益处理）。
    """
    tradable = np.isfinite(next_vwap)
    if next_open is not None:
        tradable = tradable & np.isfinite(next_open)
    return tradable


def _bucket_mean(ret_row: np.ndarray, mask: np.ndarray) -> float:
    """桶内市值等权收益（NaN 成员剔除）。空桶 → NaN。"""
    values = ret_row[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def compute_cohort_pnl(
    factor_values: np.ndarray,
    next_ret: np.ndarray,
    next_vwap: np.ndarray,
    next_open: Optional[np.ndarray] = None,
    weight_matrix: Optional[np.ndarray] = None,
    n_quantiles: int = 10,
    holding: int = 20,
    long_weight: float = 0.5,
    short_weight: float = -0.5,
    per_side_cost: float = 0.0,
    min_bucket_size: int = 1,
    require_tradable: bool = True,
    trade_eligibility=None,
    terminal_position_policy: str = "ongoing",
) -> Dict[str, np.ndarray]:
    """构造研究用途的固定名义权重 cohort；不认证可执行市场中性或容量。

    Args:
        factor_values : (T, N) 因子值面板（行=交易日）。
        next_ret      : (T, N) 当日持有收益 r_t = P_t / P_{t-1} - 1（vwap→vwap）。
        next_vwap     : (T, N) 当日 VWAP 价（入场/出场价）。
        next_open     : (T, N) 可选，当日开盘价（停牌检查）。
        weight_matrix : (T, N) 可选，市值权重（基准组合，用于 active return）；
                        缺省 None 时用横截面等权作基准。
        n_quantiles   : 分位数（默认 10：D1..D10）。
        holding       : 持有期 H（默认 20 个交易日）。
        long_weight   : D10 每票目标权重 = long_weight / N_D10（默认 +0.5）。
        short_weight  : D1 每票目标权重 = short_weight / N_D1（默认 -0.5）。
        per_side_cost : 每笔单边比例成本（默认 0 = gross）；扣在入场与出场各一次。
        min_bucket_size: 桶内最少成员数（< 该数的交易日整体跳过）。
        require_tradable: 若 True，入场日不可交易的票剔除；全不可交易则该 cohort 跳过。
        terminal_position_policy: ``ongoing`` 保留未成熟尾部；
                                  ``liquidate_at_end`` 在最后一日收取平仓成本。

    Returns:
        dict with keys:
            pnl_net        : (T,) 每日真实 PnL（净值变化，含 1/H 缩放与成本）。
            active_ret     : (T,) D10 long-only active return 每日 PnL
                             （相对基准，含 1/H 缩放）。
            gross_exposure : (T,) 当日总仓位 Σ|w|/H（多空合计）。
            cohort_weight  : (T,) 当日 active cohort 数 / H。
            long_short_ret : (T,) D10-D1 每日 spread（Signal Quality Diagnostic，
                             信号日对齐）。
            long_ret       : (T,) D10 桶每日收益（市值等权，信号日对齐）。
            short_ret      : (T,) D1 桶每日收益（市值等权，信号日对齐）。
            entry_cost     : (T,) 当日入场成本（净值口径）。
            exit_cost      : (T,) 当日出场成本（净值口径）。
            n_long         : (T,) 当日信号 cohort 的 D10 票数。
            n_short        : (T,) 当日信号 cohort 的 D1 票数。
    """
    factor_values = np.asarray(factor_values, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    next_vwap = np.asarray(next_vwap, dtype=np.float64)
    if factor_values.ndim != 2:
        raise ValueError(f"factor_values 必须为 (T, N)，got ndim={factor_values.ndim}")
    T, N = factor_values.shape
    _validate_2d("next_ret", next_ret, T, N)
    _validate_2d("next_vwap", next_vwap, T, N)
    _validate_2d("next_open", next_open, T, N)
    _validate_2d("weight_matrix", weight_matrix, T, N)
    if trade_eligibility is not None:
        from quant_evaluator.contracts.portfolio_inputs import TradeEligibilityPanel
        if not isinstance(trade_eligibility, TradeEligibilityPanel) or trade_eligibility.can_buy.shape != (T, N):
            raise ValueError("trade_eligibility must be a matching TradeEligibilityPanel")

    entries, exits, matured = cohort_schedule(T, holding)
    if terminal_position_policy not in {"ongoing", "liquidate_at_end"}:
        raise ValueError("terminal_position_policy must be ongoing or liquidate_at_end")
    if not (0.0 <= per_side_cost < 1.0):
        raise ValueError(f"per_side_cost 必须在 [0, 1)，got {per_side_cost}")

    long_mask, short_mask = build_cohort_panels(
        factor_values, n_quantiles=n_quantiles, min_bucket_size=min_bucket_size
    )

    tradable = _is_tradable(next_vwap, next_open)

    # 基准组合：市值加权（weight_matrix）或横截面等权。
    if weight_matrix is not None:
        wm = np.asarray(weight_matrix, dtype=np.float64)
        wm_valid = np.where(np.isfinite(wm), wm, 0.0)
        denom = np.sum(wm_valid, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bench = np.where(
                denom > EPS,
                np.nansum(np.where(np.isfinite(next_ret), next_ret, 0.0) * wm_valid, axis=1)
                / np.maximum(denom, EPS),
                np.nan,
            )
    else:
        finite_ret = np.isfinite(next_ret)
        with np.errstate(invalid="ignore", divide="ignore"):
            bench = np.where(
                np.sum(finite_ret, axis=1) > 0,
                np.nansum(np.where(finite_ret, next_ret, 0.0), axis=1)
                / np.sum(finite_ret, axis=1),
                np.nan,
            )

    pnl_net = np.zeros(T)
    active_ret = np.zeros(T)
    gross = np.zeros(T)
    cohort_weight = np.zeros(T)
    entry_cost_series = np.zeros(T)
    exit_cost_series = np.zeros(T)
    n_long_series = np.zeros(T, dtype=np.int64)
    n_short_series = np.zeros(T, dtype=np.int64)
    long_ret = np.full(T, np.nan)
    short_ret = np.full(T, np.nan)
    long_short_ret = np.full(T, np.nan)

    for start in range(T):
        # cohort 入场需下一交易日；没有下一日则无法入场。
        if start + 1 >= T:
            continue
        entry_t = int(entries[start])

        long_t = long_mask[start]
        short_t = short_mask[start]
        if trade_eligibility is not None:
            long_t = long_t & trade_eligibility.can_buy[entry_t]
            short_t = short_t & trade_eligibility.can_sell[entry_t] & trade_eligibility.borrowable[entry_t]
        n_long = int(np.sum(long_t))
        n_short = int(np.sum(short_t))
        if (long_weight != 0 and n_long < min_bucket_size) or (short_weight != 0 and n_short < min_bucket_size):
            continue

        if require_tradable:
            long_t = long_t & tradable[entry_t]
            short_t = short_t & tradable[entry_t]
            n_long = int(np.sum(long_t))
            n_short = int(np.sum(short_t))
            if (long_weight != 0 and n_long < min_bucket_size) or (short_weight != 0 and n_short < min_bucket_size):
                continue

        # 票级目标权重：每侧合计 = weight（正/负），均分到桶内每票。
        w_long = np.zeros(N)
        w_short = np.zeros(N)
        if n_long > 0:
            w_long[long_t] = long_weight / n_long
        if n_short > 0:
            w_short[short_t] = short_weight / n_short
        side_notional = float(np.sum(np.abs(w_long)) + np.sum(np.abs(w_short)))
        # 理论上 side_notional == |long_weight| + |short_weight|；数值保护。
        if side_notional <= EPS:
            continue

        # H 是入场后的价格区间数；未成熟尾部保留估值但不自动退出。
        exit_t = int(exits[start])
        if trade_eligibility is not None and exit_t < T:
            if np.any(long_t & ~trade_eligibility.can_sell[exit_t]) or np.any(short_t & ~trade_eligibility.coverable[exit_t]):
                raise ValueError("planned exit cannot execute; a fill-ledger simulator is required, probe cannot fabricate liquidation")
        terminal_liquidation = terminal_position_policy == "liquidate_at_end" and exit_t >= T and entry_t < T - 1
        if trade_eligibility is not None and terminal_liquidation:
            if np.any(long_t & ~trade_eligibility.can_sell[T - 1]) or np.any(short_t & ~trade_eligibility.coverable[T - 1]):
                raise ValueError("terminal liquidation cannot execute; a fill-ledger simulator is required")
        entry_fee = side_notional * per_side_cost / holding
        pnl_net[entry_t] -= entry_fee
        entry_cost_series[entry_t] += entry_fee
        if terminal_liquidation:
            exit_cost_series[T - 1] += entry_fee
            pnl_net[T - 1] -= entry_fee
        # Entry quote belongs to the old position; first earned interval ends
        # at entry+1. A truncated observation window never invents an exit.
        for t in range(entry_t + 1, min(exit_t + 1, T)):
            ret_t = np.where(np.isfinite(next_ret[t]), next_ret[t], 0.0)
            # long 侧：+w*r；short 侧（w_short 为负）：做空收益 = -|w|*r。
            day_contrib = w_long * ret_t + w_short * ret_t
            # 成本：入场日扣入场成本，出场日扣出场成本（各自 per-side 一次）。
            cost = np.zeros(N)
            notional = np.abs(w_long) + np.abs(w_short)
            if t == exit_t:
                cost = cost + notional * per_side_cost
            pnl_net[t] += np.sum(day_contrib - cost) / holding
            if np.any((notional > EPS) & ~np.isfinite(next_ret[t])):
                pnl_net[t] = np.nan
                active_ret[t] = np.nan
            if t == exit_t:
                exit_cost_series[t] += np.sum(cost) / holding
            gross[t] += side_notional / holding

            # D10 long-only active return：D10 桶均值 - 基准，按 1/H 缩放。
            if n_long > 0:
                bucket = _bucket_mean(next_ret[t], long_t)
                if np.isfinite(bucket) and np.isfinite(bench[t]):
                    active_ret[t] += (bucket - bench[t]) / holding

        cohort_weight[start] += 1.0 / holding
        # 信号日（start）的 cohort 权重：该 cohort 名义 1.0 均摊到持有期
        # 20 日（1/H 缩放）。注意 gross_exposure 与 pnl 均按 1/H 逐日摊销，
        # 因此 cohort_weight 是「该信号日新建名义的摊余权重」，非总仓位。
        n_long_series[start] = n_long
        n_short_series[start] = n_short

        # 分位桶收益诊断（信号日对齐）。
        lr = _bucket_mean(next_ret[entry_t + 1], long_t) if n_long > 0 and entry_t + 1 < T else np.nan
        sr = _bucket_mean(next_ret[entry_t + 1], short_t) if n_short > 0 and entry_t + 1 < T else np.nan
        long_ret[start] = lr
        short_ret[start] = sr
        if n_long > 0 and n_short > 0 and np.isfinite(lr) and np.isfinite(sr):
            long_short_ret[start] = lr - sr

    return {
        "pnl_net": pnl_net,
        "active_ret": active_ret,
        "gross_exposure": gross,
        "cohort_weight": cohort_weight,
        "long_short_ret": long_short_ret,
        "long_ret": long_ret,
        "short_ret": short_ret,
        "entry_cost": entry_cost_series,
        "exit_cost": exit_cost_series,
        "n_long": n_long_series,
        "n_short": n_short_series,
        "scheduled_exit": exits,
        "matured": matured & (cohort_weight > 0),
    }


def compute_overlapping_forward_returns(
    next_ret: np.ndarray,
    long_mask: Optional[np.ndarray] = None,
    short_mask: Optional[np.ndarray] = None,
    holding: int = 20,
) -> np.ndarray:
    """构造 overlapping H 日 forward return 序列（用于对照「错误算法」）。

    每天算一个 [t+1, t+H] 的复利 forward return，返回 (T,) 数组，
    最后 holding-1 个位置为 NaN（无足够未来数据）。

    若提供 long_mask/short_mask，则构建 D10-D1 的 overlapping spread
    （分位桶逐日收益差复利），否则对全市场等权组合构造。

    注意：该序列日度重叠、高度自相关。本函数只用于在单测中演示
    「直接对该序列做 std → annualize 会虚高 Sharpe」的对照，禁止
    在生产指标链路中使用。
    """
    next_ret = np.asarray(next_ret, dtype=np.float64)
    T, N = next_ret.shape
    fwd = np.full(T, np.nan)
    for t in range(T - holding):
        window = next_ret[t + 1 : t + holding + 1]
        if long_mask is not None:
            lw = window * long_mask[t + 1 : t + holding + 1]
            sw = window * short_mask[t + 1 : t + holding + 1]
            with np.errstate(invalid="ignore"):
                day_spread = np.where(
                    np.isfinite(lw) & np.isfinite(sw),
                    np.where(np.isfinite(lw), lw, 0.0) - np.where(np.isfinite(sw), sw, 0.0),
                    np.nan,
                )
            with np.errstate(invalid="ignore"):
                fwd[t] = np.nanmean(np.log1p(day_spread))
        else:
            # 全市场等权（与旧实现一致）。
            with np.errstate(invalid="ignore"):
                prod = np.nansum(np.log1p(np.where(np.isfinite(window), window, 0.0)), axis=0)
                valid_cols = np.all(np.isfinite(window), axis=0)
            fwd[t] = np.mean(np.expm1(prod[valid_cols])) if np.any(valid_cols) else np.nan
    return fwd


def annualize_overlapping_label_sharpe(
    overlapping_returns: np.ndarray,
    holding: int = 20,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """错误对照算法：把重叠 forward return 序列直接年化 Sharpe。

    该算法把每个重叠的 H 日收益当作独立样本，用 sqrt(periods_per_year/H)
    年化，是用户规范 §7/§13 明令禁止的虚高做法。本函数只用于单测对照，
    禁止在生产链路中使用。
    """
    valid = np.isfinite(overlapping_returns)
    if np.sum(valid) < 2:
        return np.nan
    rets = overlapping_returns[valid]
    rf_per_period = risk_free_rate / periods_per_year * holding
    excess = rets - rf_per_period
    std = np.std(excess, ddof=1)
    if not np.isfinite(std) or std <= EPS:
        return np.nan
    return float(np.mean(excess) / std * np.sqrt(periods_per_year / holding))
