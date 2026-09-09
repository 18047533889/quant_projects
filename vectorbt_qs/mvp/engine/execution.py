"""Calendar scheduling and stateful A-share order planning.

This module keeps the execution model intentionally small: daily, long-only,
and one rebalance event per session. It plans in raw exchange prices and real
shares. Cash dividends and share distributions are passed to vectorbt as
explicit corporate-action flows, so accurate valuation stays in real units.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from decimal import Decimal
from typing import Any, Dict, List
from warnings import warn

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionCosts:
    """A-share transaction-cost parameters."""

    commission: float = 0.00025
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    minimum_commission: float = 5.0
    fee_schedule: Any = None

    def __post_init__(self) -> None:
        for name in ("commission", "stamp_tax", "transfer_fee", "minimum_commission"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExecutionCosts":
        schedule_ref = config.get("fee_schedule_ref")
        if schedule_ref is not None:
            if schedule_ref != "CN_SH_SZ_A_ORDINARY_RESEARCH_V5":
                raise ValueError(f"unknown fee_schedule_ref: {schedule_ref}")
            from vectorbt_qs.contracts.costs import ordinary_ashare_research_schedule
            return cls(fee_schedule=ordinary_ashare_research_schedule())
        costs = config.get("costs")
        if costs is not None:
            if not isinstance(costs, dict):
                raise TypeError("costs 必须是字典")
            legacy_keys = {"fees", "fixed_fees"}.intersection(config)
            if legacy_keys:
                raise ValueError("costs 不能与 fees/fixed_fees 同时配置")
            allowed_keys = {
                "commission", "stamp_tax", "transfer_fee", "minimum_commission"
            }
            unknown_keys = set(costs).difference(allowed_keys)
            if unknown_keys:
                raise ValueError(f"未知 costs 参数: {', '.join(sorted(unknown_keys))}")
            return cls(
                commission=float(costs.get("commission", cls.commission)),
                stamp_tax=float(costs.get("stamp_tax", cls.stamp_tax)),
                transfer_fee=float(costs.get("transfer_fee", cls.transfer_fee)),
                minimum_commission=float(costs.get("minimum_commission", cls.minimum_commission)),
            )

        # Backward compatibility for old benchmark configs.  A scalar ``fees``
        # means a flat all-in rate and deliberately disables directional tax.
        if "fees" in config:
            if float(config.get("fixed_fees", 0.0)) != 0.0:
                raise ValueError("accurate 模式不支持非零 fixed_fees；请改用 costs.minimum_commission")
            flat_rate = float(config["fees"])
            warn(
                "accurate 模式中的 fees 标量按双向统一总费率处理；"
                "若要使用印花税，请改用 costs 配置",
                stacklevel=2,
            )
            return cls(
                commission=flat_rate,
                stamp_tax=0.0,
                transfer_fee=0.0,
                minimum_commission=0.0,
            )
        if float(config.get("fixed_fees", 0.0)) != 0.0:
            raise ValueError("accurate 模式不支持非零 fixed_fees；请改用 costs.minimum_commission")
        return cls()

    def _scheduled_entry(self, side: str, trade_date: Any):
        if self.fee_schedule is None:
            return None
        if trade_date is None:
            raise ValueError("effective-dated fee schedule requires trade_date")
        from vectorbt_qs.contracts.costs import FillForBilling
        fill = FillForBilling(
            trade_date=trade_date.date() if hasattr(trade_date, "date") else Date.fromisoformat(str(trade_date)[:10]),
            market="CN_SH_SZ", instrument="ORDINARY_A", account="RESEARCH",
            side=side, notional_cny=Decimal("0"), billing_group="rate_lookup",
        )
        return self.fee_schedule.resolve(fill)

    def rate(self, side: str, trade_date: Any = None) -> float:
        entry = self._scheduled_entry(side, trade_date)
        if entry is not None:
            rate = float(entry.commission_bps) / 10000.0
            if "transfer_fee" not in entry.included_components:
                rate += float(entry.transfer_fee_bps) / 10000.0
            if "stamp_tax" not in entry.included_components:
                rate += float(entry.stamp_tax_bps) / 10000.0
            return rate
        rate = self.commission + self.transfer_fee
        if side == "sell":
            rate += self.stamp_tax
        return rate

    def fixed_fee(self, notional: float, side: str = "buy", trade_date: Any = None) -> float:
        entry = self._scheduled_entry(side, trade_date)
        minimum = self.minimum_commission if entry is None else float(entry.minimum_commission_cny)
        commission = self.commission if entry is None else float(entry.commission_bps) / 10000.0
        return max(0.0, minimum - notional * commission)

    def total_fee(self, notional: float, side: str, trade_date: Any = None) -> float:
        return notional * self.rate(side, trade_date) + self.fixed_fee(notional, side, trade_date)


@dataclass
class ExecutionPlan:
    """Explicit orders and their execution parameters."""

    order_size: pd.DataFrame
    fees: pd.DataFrame
    fixed_fees: pd.DataFrame
    call_seq: np.ndarray
    log: pd.DataFrame
    final_cash: float
    final_shares: pd.Series
    real_holdings: pd.DataFrame
    cash_deposits: pd.DataFrame
    asset_deposits: pd.DataFrame


def sample_target_signals(target_weights: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Sample target signals without changing their actual dates."""
    normalized = str(freq).upper()
    if normalized == "1D":
        return target_weights
    if normalized == "1W":
        periods = target_weights.index.to_period("W-FRI")
        return target_weights.groupby(periods, sort=True).tail(1)
    raise ValueError("基础准确模式仅支持 freq=1D 或 freq=1W")


def schedule_target_signals(
    target_weights: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    freq: str,
    delay_sessions: int,
    signal_time: str = "close",
) -> pd.DataFrame:
    """Map signal rows to exchange sessions.

    ``delay_sessions=1`` maps a close signal to the next trading session.
    Same-session execution is only allowed for signals declared ``preopen``.
    """
    if not isinstance(delay_sessions, (int, np.integer)) or delay_sessions < 0:
        raise ValueError("execution_lag 必须是非负整数")
    normalized_signal_time = str(signal_time).lower()
    if normalized_signal_time not in {"close", "preopen"}:
        raise ValueError("signal_time 仅支持 close 或 preopen")
    if normalized_signal_time == "close" and delay_sessions == 0:
        raise ValueError("收盘后信号不能在同一交易日成交；请设置 execution_lag>=1")

    signals = sample_target_signals(target_weights, freq)
    sessions = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    signal_dates = signals.index
    if signal_dates.tz is not None:
        signal_dates = signal_dates.tz_localize(None)

    execution_dates: List[pd.Timestamp] = []
    for signal_date in signal_dates:
        session_pos = sessions.searchsorted(signal_date)
        if session_pos >= len(sessions) or sessions[session_pos] != signal_date:
            raise ValueError(f"目标权重日期不是交易日: {signal_date.date()}")
        execution_pos = session_pos + delay_sessions
        if execution_pos >= len(sessions):
            raise ValueError(f"{signal_date.date()} 之后没有足够交易日执行信号")
        execution_dates.append(sessions[execution_pos])

    if len(execution_dates) != len(set(execution_dates)):
        raise ValueError("多个信号映射到同一执行日期")
    scheduled = signals.copy()
    scheduled.index = pd.DatetimeIndex(execution_dates, name="ExecutionDate")
    return scheduled


def _assert_execution_cell(
    date: pd.Timestamp,
    symbol: str,
    price: float,
    suspended: Any,
    high_limit: float,
    low_limit: float,
) -> None:
    values = {
        "price": price,
        "IsSuspend": suspended,
        "HighLimit": high_limit,
        "LowLimit": low_limit,
    }
    missing = [name for name, value in values.items() if pd.isna(value)]
    if missing:
        raise ValueError(f"{date.date()} {symbol} 执行数据缺失: {', '.join(missing)}")
    if not np.isfinite(float(price)) or float(price) <= 0:
        raise ValueError(f"{date.date()} {symbol} 成交价无效: {price}")


def plan_ashare_orders_python(
    close: pd.DataFrame,
    order_price: pd.DataFrame,
    scheduled_targets: pd.DataFrame,
    is_suspend: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    *,
    init_cash: float,
    costs: ExecutionCosts,
    slippage: float = 0.0,
    lot_size: int = 100,
    adjustment_factor: pd.DataFrame | None = None,
    cash_dividend: pd.DataFrame | None = None,
    share_multiplier: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    max_participation_rate: float | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> ExecutionPlan:
    """Plan explicit long-only A-share orders from actual cash and shares.

    Cash and share distributions are applied on their ex-dates before the next
    execution decision. Orders, holdings, fees, and logs remain in real shares.
    ``adjustment_factor`` is validated for the runner's adjusted-unit bridge.
    ``high`` and ``low`` bound the final price after slippage.
    """
    if init_cash <= 0:
        raise ValueError("init_cash 必须大于 0")
    if lot_size <= 0:
        raise ValueError("lot_size 必须为正整数")
    if slippage < 0:
        raise ValueError("slippage 不能为负")
    if slippage >= 1:
        raise ValueError("slippage 必须小于 1")
    if max_participation_rate is not None and (
        not np.isfinite(float(max_participation_rate))
        or float(max_participation_rate) <= 0.0
        or float(max_participation_rate) > 1.0
    ):
        raise ValueError("max_participation_rate 必须在 (0, 1] 内")

    index, columns = close.index, close.columns
    if not scheduled_targets.index.isin(index).all():
        missing_dates = scheduled_targets.index.difference(index)
        raise ValueError(f"执行日期缺少行情: {list(missing_dates[:5])}")

    order_size = pd.DataFrame(np.nan, index=index, columns=columns)
    real_holdings = pd.DataFrame(np.nan, index=index, columns=columns)
    cash_deposits_out = pd.DataFrame(0.0, index=index, columns=columns)
    asset_deposits_out = pd.DataFrame(0.0, index=index, columns=columns)
    fee_rates = pd.DataFrame(0.0, index=index, columns=columns)
    fixed_fees = pd.DataFrame(0.0, index=index, columns=columns)
    call_seq = np.tile(np.arange(len(columns), dtype=np.int64), (len(index), 1))
    shares = pd.Series(0.0, index=columns)
    cash = float(init_cash)
    log_rows: List[Dict[str, Any]] = []
    factors = (
        pd.DataFrame(1.0, index=index, columns=columns)
        if adjustment_factor is None
        else adjustment_factor.reindex(index=index, columns=columns)
    )
    cash_dividends = (
        pd.DataFrame(0.0, index=index, columns=columns)
        if cash_dividend is None
        else cash_dividend.reindex(index=index, columns=columns).fillna(0.0)
    )
    share_multipliers = (
        pd.DataFrame(1.0, index=index, columns=columns)
        if share_multiplier is None
        else share_multiplier.reindex(index=index, columns=columns).fillna(1.0)
    )
    volumes = (
        pd.DataFrame(np.inf, index=index, columns=columns)
        if max_participation_rate is None
        else volume.reindex(index=index, columns=columns)
        if volume is not None
        else pd.DataFrame(np.nan, index=index, columns=columns)
    )
    highs = (
        pd.DataFrame(np.inf, index=index, columns=columns)
        if high is None
        else high.reindex(index=index, columns=columns)
    )
    lows = (
        pd.DataFrame(-np.inf, index=index, columns=columns)
        if low is None
        else low.reindex(index=index, columns=columns)
    )

    last_action_row = -1
    if not scheduled_targets.index.isin(index).all():
        raise ValueError("scheduled targets contain dates outside the execution calendar")
    # Corporate actions and holdings exist on non-rebalance dates too. NaN
    # targets preserve shares, so visiting the complete calendar adds no trades.
    for date, target_row in scheduled_targets.reindex(index=index).iterrows():
        row_pos = index.get_loc(date)
        px = order_price.loc[date]
        suspended = is_suspend.loc[date]
        upper = high_limit.loc[date]
        lower = low_limit.loc[date]
        factor_row = factors.loc[date]
        day_high = highs.loc[date]
        day_low = lows.loc[date]
        volume_row = volumes.loc[date]

        def capacity_for(symbol: Any, requested: float) -> float:
            if max_participation_rate is None:
                return requested
            daily_volume = volume_row[symbol]
            if (
                pd.isna(daily_volume)
                or not np.isfinite(float(daily_volume))
                or float(daily_volume) < 0.0
            ):
                raise ValueError(
                    f"{date.date()} {symbol} 成交量无效: {daily_volume}"
                )
            raw_capacity = float(daily_volume) * float(max_participation_rate)
            if requested <= raw_capacity + 1e-12:
                return requested
            return float(np.floor(raw_capacity / lot_size) * lot_size)

        # NaN means hold the actual position.  Non-NaN values are complete
        # target weights for those assets.
        effective_targets = target_row.reindex(columns)
        if (effective_targets.dropna() < 0).any():
            raise ValueError(f"{date.date()} A 股目标权重不能为负")

        # Apply every corporate action since the previous execution. Cash
        # dividends increase spendable cash; stock distributions change real
        # shares. Factor changes are not treated as stock splits because the
        # vendor factor also contains cash dividends.
        for action_row in range(last_action_row + 1, row_pos + 1):
            action_date = index[action_row]
            for col_pos, symbol in enumerate(columns):
                dividend = float(cash_dividends.iat[action_row, col_pos])
                multiplier = float(share_multipliers.iat[action_row, col_pos])
                if not np.isfinite(dividend) or dividend < 0.0:
                    raise ValueError(
                        f"{action_date.date()} {symbol} 现金分红无效: {dividend}"
                    )
                if not np.isfinite(multiplier) or multiplier <= 0.0:
                    raise ValueError(
                        f"{action_date.date()} {symbol} 送转股比例无效: {multiplier}"
                    )
                if shares[symbol] != 0.0:
                    cash_deposit = float(shares[symbol]) * dividend
                    asset_deposit = float(shares[symbol]) * (multiplier - 1.0)
                    cash += cash_deposit
                    shares[symbol] += asset_deposit
                    cash_deposits_out.iat[action_row, col_pos] += cash_deposit
                    asset_deposits_out.iat[action_row, col_pos] += asset_deposit
        last_action_row = row_pos

        for symbol in columns:
            factor = factor_row[symbol]
            needs_factor = shares[symbol] != 0.0 or (
                not pd.isna(effective_targets[symbol])
                and effective_targets[symbol] != 0.0
            )
            if needs_factor and (
                pd.isna(factor)
                or not np.isfinite(float(factor))
                or float(factor) <= 0.0
            ):
                raise ValueError(
                    f"{date.date()} {symbol} 复权因子无效: {factor}"
                )

        valuation_prices = px.where(shares.ne(0.0), 0.0)
        held_missing = shares.ne(0.0) & px.isna()
        if held_missing.any():
            symbols = ", ".join(map(str, columns[held_missing][:5]))
            raise ValueError(f"{date.date()} 持仓缺少开盘估值价: {symbols}")
        equity = cash + float((shares * valuation_prices.fillna(0.0)).sum())

        desired_shares = shares.copy()
        for symbol in columns:
            weight = effective_targets[symbol]
            if pd.isna(weight):
                continue
            # Full-universe matrices commonly contain zero weights for stocks
            # that have not listed yet.  With no position and no intended
            # order, missing execution data is harmless.
            if weight == 0 and shares[symbol] == 0:
                desired_shares[symbol] = 0.0
                continue
            _assert_execution_cell(
                date,
                str(symbol),
                px[symbol],
                suspended[symbol],
                upper[symbol],
                lower[symbol],
            )
            if weight == 0:
                desired_shares[symbol] = 0.0
            else:
                desired_shares[symbol] = np.floor(
                    equity * float(weight) / float(px[symbol]) / lot_size
                ) * lot_size

        desired_value = float((desired_shares * px.fillna(0.0)).sum())
        if desired_value > equity * (1.0 + 1e-8):
            raise ValueError(
                f"{date.date()} 目标总市值超过组合净值: {desired_value:.2f} > {equity:.2f}"
            )

        sell_symbols: List[str] = []
        buy_candidates: List[tuple[float, str, float]] = []

        # Sells are planned first and can release cash for buys.  Full exits
        # may sell odd lots; partial reductions remain board-lot sized.
        for symbol in columns:
            reduction = float(shares[symbol] - desired_shares[symbol])
            if reduction <= 1e-12:
                continue
            requested = float(shares[symbol]) if desired_shares[symbol] == 0 else np.floor(reduction / lot_size) * lot_size
            if requested <= 0:
                continue
            _assert_execution_cell(date, str(symbol), px[symbol], suspended[symbol], upper[symbol], lower[symbol])
            if bool(suspended[symbol]):
                log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "suspend"})
                continue
            if float(px[symbol]) <= float(lower[symbol]) or np.isclose(float(px[symbol]), float(lower[symbol])):
                log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "limit_down"})
                continue

            executable = capacity_for(symbol, requested)
            if executable <= 0.0:
                log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "volume"})
                continue

            execution_price = float(px[symbol]) * (1.0 - slippage)
            if execution_price <= float(lower[symbol]) or np.isclose(
                execution_price,
                float(lower[symbol]),
            ):
                log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "limit_down"})
                continue
            if (
                np.isfinite(float(day_low[symbol]))
                and execution_price < float(day_low[symbol])
                and not np.isclose(execution_price, float(day_low[symbol]))
            ):
                log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "price_bound"})
                continue
            notional = executable * execution_price
            fixed = costs.fixed_fee(notional, "sell", date)
            cash += notional - costs.total_fee(notional, "sell", date)
            shares[symbol] -= executable
            order_size.at[date, symbol] = -executable
            fee_rates.at[date, symbol] = costs.rate("sell", date)
            fixed_fees.at[date, symbol] = fixed
            sell_symbols.append(symbol)
            status = "filled" if executable == requested else "partial_volume"
            log_rows.append({"date": date, "symbol": symbol, "side": "sell", "requested": requested, "filled": executable, "status": status, "reason": "" if status == "filled" else "volume"})

        for symbol in columns:
            requested = float(desired_shares[symbol] - shares[symbol])
            requested = np.floor(requested / lot_size) * lot_size
            if requested > 0:
                buy_candidates.append((requested * float(px[symbol]), str(symbol), requested))

        # Check tradeability before allocating cash.  When cash is insufficient,
        # scale every buy proportionally rather than filling the first column
        # and starving later assets.
        buy_candidates.sort(key=lambda item: item[1])
        tradable_buys: List[tuple[str, float, float]] = []
        original_buy_requests: Dict[str, float] = {}
        for _, symbol, requested in buy_candidates:
            _assert_execution_cell(date, symbol, px[symbol], suspended[symbol], upper[symbol], lower[symbol])
            if bool(suspended[symbol]):
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "suspend"})
                continue
            if float(px[symbol]) >= float(upper[symbol]) or np.isclose(float(px[symbol]), float(upper[symbol])):
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "limit_up"})
                continue
            execution_price = float(px[symbol]) * (1.0 + slippage)
            if execution_price >= float(upper[symbol]) or np.isclose(
                execution_price,
                float(upper[symbol]),
            ):
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "limit_up"})
                continue
            if (
                np.isfinite(float(day_high[symbol]))
                and execution_price > float(day_high[symbol])
                and not np.isclose(execution_price, float(day_high[symbol]))
            ):
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "price_bound"})
                continue
            executable = capacity_for(symbol, requested)
            if executable <= 0.0:
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": requested, "filled": 0.0, "status": "blocked", "reason": "volume"})
                continue
            original_buy_requests[symbol] = requested
            tradable_buys.append((symbol, executable, execution_price))

        def buy_cost(symbol: str, quantity: float, execution_price: float) -> float:
            if quantity <= 0:
                return 0.0
            notional = quantity * execution_price
            return notional + costs.total_fee(notional, "buy", date)

        requested_cost = sum(buy_cost(*candidate) for candidate in tradable_buys)
        scale = min(1.0, cash / requested_cost) if requested_cost > 0 else 0.0
        planned_buys = {
            symbol: np.floor(requested * scale / lot_size) * lot_size
            for symbol, requested, _ in tradable_buys
        }
        candidate_by_symbol = {symbol: (requested, execution_price) for symbol, requested, execution_price in tradable_buys}

        def total_planned_cost() -> float:
            return sum(
                buy_cost(symbol, quantity, candidate_by_symbol[symbol][1])
                for symbol, quantity in planned_buys.items()
            )

        # Fixed minimum commissions can make proportional scaling slightly
        # exceed cash.  Remove one lot at a time from the smallest remaining
        # deficit until the plan is affordable.
        while total_planned_cost() > cash + 1e-8:
            reducible = [symbol for symbol, quantity in planned_buys.items() if quantity >= lot_size]
            if not reducible:
                break
            symbol = min(
                reducible,
                key=lambda item: (
                    candidate_by_symbol[item][0] - planned_buys[item],
                    item,
                ),
            )
            planned_buys[symbol] -= lot_size

        # Spend residual cash on at most one additional lot per asset, ordered
        # by the largest remaining target-value deficit.
        residual_order = sorted(
            planned_buys,
            key=lambda symbol: (
                -(candidate_by_symbol[symbol][0] - planned_buys[symbol]) * candidate_by_symbol[symbol][1],
                symbol,
            ),
        )
        for symbol in residual_order:
            requested, execution_price = candidate_by_symbol[symbol]
            if planned_buys[symbol] + lot_size > requested:
                continue
            old_cost = buy_cost(symbol, planned_buys[symbol], execution_price)
            new_cost = buy_cost(symbol, planned_buys[symbol] + lot_size, execution_price)
            if total_planned_cost() + new_cost - old_cost <= cash + 1e-8:
                planned_buys[symbol] += lot_size

        buy_symbols: List[str] = []
        for symbol in sorted(planned_buys):
            requested, execution_price = candidate_by_symbol[symbol]
            original_requested = original_buy_requests[symbol]
            affordable = float(planned_buys[symbol])
            if affordable <= 0:
                log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": original_requested, "filled": 0.0, "status": "blocked", "reason": "cash"})
                continue

            notional = affordable * execution_price
            fixed = costs.fixed_fee(notional, "buy", date)
            cash -= notional + costs.total_fee(notional, "buy", date)
            shares[symbol] += affordable
            order_size.at[date, symbol] = affordable
            fee_rates.at[date, symbol] = costs.rate("buy", date)
            fixed_fees.at[date, symbol] = fixed
            buy_symbols.append(symbol)
            if affordable < requested:
                status, reason = "partial_cash", "cash"
            elif affordable < original_requested:
                status, reason = "partial_volume", "volume"
            else:
                status, reason = "filled", ""
            log_rows.append({"date": date, "symbol": symbol, "side": "buy", "requested": original_requested, "filled": affordable, "status": status, "reason": reason})

        ordered_symbols = sell_symbols + buy_symbols
        ordered_symbols += [str(symbol) for symbol in columns if str(symbol) not in ordered_symbols]
        col_pos = {str(symbol): i for i, symbol in enumerate(columns)}
        call_seq[row_pos] = np.array([col_pos[symbol] for symbol in ordered_symbols], dtype=np.int64)
        real_holdings.loc[date] = shares

    return ExecutionPlan(
        order_size=order_size,
        fees=fee_rates,
        fixed_fees=fixed_fees,
        call_seq=call_seq,
        log=pd.DataFrame(log_rows),
        final_cash=cash,
        final_shares=shares,
        real_holdings=real_holdings,
        cash_deposits=cash_deposits_out,
        asset_deposits=asset_deposits_out,
    )


def plan_ashare_orders(
    close: pd.DataFrame,
    order_price: pd.DataFrame,
    scheduled_targets: pd.DataFrame,
    is_suspend: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    *,
    init_cash: float,
    costs: ExecutionCosts,
    slippage: float = 0.0,
    lot_size: int = 100,
    planner_engine: str = "numba",
    adjustment_factor: pd.DataFrame | None = None,
    cash_dividend: pd.DataFrame | None = None,
    share_multiplier: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    max_participation_rate: float | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> ExecutionPlan:
    """Plan explicit A-share orders with a selectable equivalent engine.

    ``numba`` is the production path.  ``python`` keeps the original state
    machine available as a readable oracle for regression and diagnostics.
    """
    normalized_engine = str(planner_engine).lower()
    if not scheduled_targets.index.isin(order_price.index).all():
        raise ValueError("scheduled targets contain dates outside the execution calendar")
    # Keep compiled/reference planners on the same full economic-event clock.
    scheduled_targets = scheduled_targets.reindex(index=order_price.index)
    if costs.fee_schedule is not None and normalized_engine == "numba":
        # The compiled planner accepts only one scalar fee vector.  Falling
        # back preserves the effective-date contract instead of backfilling
        # today's statutory rate through history.
        normalized_engine = "python"
    if normalized_engine == "python":
        return plan_ashare_orders_python(
            close,
            order_price,
            scheduled_targets,
            is_suspend,
            high_limit,
            low_limit,
            init_cash=init_cash,
            costs=costs,
            slippage=slippage,
            lot_size=lot_size,
            adjustment_factor=adjustment_factor,
            cash_dividend=cash_dividend,
            share_multiplier=share_multiplier,
            volume=volume,
            max_participation_rate=max_participation_rate,
            high=high,
            low=low,
        )
    if normalized_engine != "numba":
        raise ValueError("planner_engine 仅支持 numba 或 python")

    # Keep public validation identical and fail before invoking compiled code.
    if init_cash <= 0:
        raise ValueError("init_cash 必须大于 0")
    if lot_size <= 0:
        raise ValueError("lot_size 必须为正整数")
    if slippage < 0:
        raise ValueError("slippage 不能为负")
    if slippage >= 1:
        raise ValueError("slippage 必须小于 1")
    if max_participation_rate is not None and (
        not np.isfinite(float(max_participation_rate))
        or float(max_participation_rate) <= 0.0
        or float(max_participation_rate) > 1.0
    ):
        raise ValueError("max_participation_rate 必须在 (0, 1] 内")

    from .execution_nb import plan_ashare_orders_numba

    return plan_ashare_orders_numba(
        close,
        order_price,
        scheduled_targets,
        is_suspend,
        high_limit,
        low_limit,
        init_cash=init_cash,
        costs=costs,
        slippage=slippage,
        lot_size=lot_size,
        adjustment_factor=adjustment_factor,
        cash_dividend=cash_dividend,
        share_multiplier=share_multiplier,
        volume=volume,
        max_participation_rate=max_participation_rate,
        high=high,
        low=low,
        python_fallback=plan_ashare_orders_python,
    )
