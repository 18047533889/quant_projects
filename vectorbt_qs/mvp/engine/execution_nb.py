"""Numba implementation of the stateful A-share order planner.

The Python planner in :mod:`vectorbt_qs.mvp.engine.execution` remains the reference
implementation.  This module only owns the numeric state transition; pandas
alignment, validation messages, and public result construction stay in the
Python layer.
"""

# Import this module only through the canonical ``vectorbt_qs.mvp`` package
# path. Numba disk caches persist the defining module name.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import List

if TYPE_CHECKING:
    from .execution import ExecutionCosts, ExecutionPlan


_LOG_ENTRY_TYPE = types.Tuple(
    (
        types.int64,    # row position
        types.int64,    # column position
        types.int64,    # side: -1 sell, 1 buy
        types.float64,  # requested
        types.float64,  # filled
        types.int64,    # status: 1 filled, 2 blocked, 3 partial_cash
        types.int64,    # reason: 0 none, 1 suspend, 2 limit_down, 3 limit_up, 4 cash, 5 price_bound
    )
)


@njit(cache=True)
def _isclose_nb(left: float, right: float) -> bool:
    """Match ``numpy.isclose`` defaults for finite execution prices."""
    return abs(left - right) <= 1e-8 + 1e-5 * abs(right)


@njit(cache=True)
def _fixed_fee_nb(
    notional: float,
    commission: float,
    minimum_commission: float,
) -> float:
    return max(0.0, minimum_commission - notional * commission)


@njit(cache=True)
def _buy_cost_nb(
    quantity: float,
    execution_price: float,
    buy_rate: float,
    commission: float,
    minimum_commission: float,
) -> float:
    if quantity <= 0.0:
        return 0.0
    notional = quantity * execution_price
    fixed = _fixed_fee_nb(notional, commission, minimum_commission)
    total_fee = notional * buy_rate + fixed
    return notional + total_fee


@njit(cache=True)
def _total_planned_cost_nb(
    sorted_columns: np.ndarray,
    tradable: np.ndarray,
    planned: np.ndarray,
    execution_prices: np.ndarray,
    buy_rate: float,
    commission: float,
    minimum_commission: float,
) -> float:
    total = 0.0
    for sorted_pos in range(sorted_columns.shape[0]):
        col = sorted_columns[sorted_pos]
        if tradable[col]:
            total += _buy_cost_nb(
                planned[col],
                execution_prices[col],
                buy_rate,
                commission,
                minimum_commission,
            )
    return total


@njit(cache=True)
def _plan_orders_nb(
    prices: np.ndarray,
    factors: np.ndarray,
    cash_dividends: np.ndarray,
    share_multipliers: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    targets: np.ndarray,
    execution_rows: np.ndarray,
    suspended: np.ndarray,
    upper_limits: np.ndarray,
    lower_limits: np.ndarray,
    sorted_columns: np.ndarray,
    init_cash: float,
    commission: float,
    stamp_tax: float,
    transfer_fee: float,
    minimum_commission: float,
    slippage: float,
    lot_size: int,
    max_participation_rate: float,
):
    """Run the numeric state machine.

    ``error_code != 0`` asks the wrapper to execute the Python oracle.  Invalid
    input is deliberately rare, and falling back preserves its exact exception
    type and Chinese diagnostic text without duplicating that policy in Numba.
    """
    n_rows, n_cols = prices.shape
    order_size = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    real_holdings = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    cash_deposits_out = np.zeros((n_rows, n_cols), dtype=np.float64)
    asset_deposits_out = np.zeros((n_rows, n_cols), dtype=np.float64)
    fee_rates = np.zeros((n_rows, n_cols), dtype=np.float64)
    fixed_fees = np.zeros((n_rows, n_cols), dtype=np.float64)
    call_seq = np.empty((n_rows, n_cols), dtype=np.int64)
    for row in range(n_rows):
        for col in range(n_cols):
            call_seq[row, col] = col

    shares = np.zeros(n_cols, dtype=np.float64)
    desired = np.zeros(n_cols, dtype=np.float64)
    requested_buys = np.zeros(n_cols, dtype=np.float64)
    original_requested_buys = np.zeros(n_cols, dtype=np.float64)
    buy_prices = np.zeros(n_cols, dtype=np.float64)
    tradable_buys = np.zeros(n_cols, dtype=np.bool_)
    planned_buys = np.zeros(n_cols, dtype=np.float64)
    selected = np.zeros(n_cols, dtype=np.bool_)
    residual_scores = np.zeros(n_cols, dtype=np.float64)
    logs = List.empty_list(_LOG_ENTRY_TYPE)

    cash = init_cash
    buy_rate = commission + transfer_fee
    sell_rate = buy_rate + stamp_tax
    error_code = 0
    last_action_row = -1

    for exec_pos in range(execution_rows.shape[0]):
        row = execution_rows[exec_pos]

        # The Python oracle checks negative non-NaN weights before calculating
        # state.  Any invalid dynamic cell delegates to that oracle below.
        for col in range(n_cols):
            weight = targets[exec_pos, col]
            if not np.isnan(weight) and weight < 0.0:
                error_code = 1
                break
        if error_code != 0:
            break

        # Apply ex-date cash and share distributions since the last execution.
        for action_row in range(last_action_row + 1, row + 1):
            for col in range(n_cols):
                dividend = cash_dividends[action_row, col]
                multiplier = share_multipliers[action_row, col]
                if (
                    np.isnan(dividend)
                    or not np.isfinite(dividend)
                    or dividend < 0.0
                    or np.isnan(multiplier)
                    or not np.isfinite(multiplier)
                    or multiplier <= 0.0
                ):
                    error_code = 6
                    break
                if shares[col] != 0.0:
                    cash_deposit = shares[col] * dividend
                    asset_deposit = shares[col] * (multiplier - 1.0)
                    cash += cash_deposit
                    shares[col] += asset_deposit
                    cash_deposits_out[action_row, col] += cash_deposit
                    asset_deposits_out[action_row, col] += asset_deposit
            if error_code != 0:
                break
        last_action_row = row
        if error_code != 0:
            break

        # Factor is only the bridge from real holdings to adjusted analytics.
        # It also reflects cash dividends and must not mutate real shares.
        for col in range(n_cols):
            factor = factors[row, col]
            weight = targets[exec_pos, col]
            needs_factor = shares[col] != 0.0 or (
                not np.isnan(weight) and weight != 0.0
            )
            if needs_factor and (
                np.isnan(factor)
                or not np.isfinite(factor)
                or factor <= 0.0
            ):
                error_code = 5
                break
        if error_code != 0:
            break

        equity = cash
        for col in range(n_cols):
            if shares[col] != 0.0:
                price = prices[row, col]
                if np.isnan(price):
                    error_code = 2
                    break
                equity += shares[col] * price
        if error_code != 0:
            break

        for col in range(n_cols):
            desired[col] = shares[col]
            weight = targets[exec_pos, col]
            if np.isnan(weight):
                continue
            if weight == 0.0 and shares[col] == 0.0:
                desired[col] = 0.0
                continue
            price = prices[row, col]
            if (
                np.isnan(price)
                or np.isnan(suspended[row, col])
                or np.isnan(upper_limits[row, col])
                or np.isnan(lower_limits[row, col])
                or not np.isfinite(price)
                or price <= 0.0
            ):
                error_code = 3
                break
            if weight == 0.0:
                desired[col] = 0.0
            else:
                desired[col] = (
                    np.floor(equity * weight / price / lot_size) * lot_size
                )
        if error_code != 0:
            break

        desired_value = 0.0
        for col in range(n_cols):
            price = prices[row, col]
            if not np.isnan(price):
                desired_value += desired[col] * price
        if desired_value > equity * (1.0 + 1e-8):
            error_code = 4
            break

        for col in range(n_cols):
            selected[col] = False
            requested_buys[col] = 0.0
            original_requested_buys[col] = 0.0
            buy_prices[col] = 0.0
            tradable_buys[col] = False
            planned_buys[col] = 0.0
            residual_scores[col] = 0.0

        sequence_pos = 0

        # Sell in input-column order, exactly like the reference planner.
        for col in range(n_cols):
            reduction = shares[col] - desired[col]
            if reduction <= 1e-12:
                continue
            if desired[col] == 0.0:
                requested = shares[col]
            else:
                requested = np.floor(reduction / lot_size) * lot_size
            if requested <= 0.0:
                continue

            price = prices[row, col]
            if (
                np.isnan(price)
                or np.isnan(suspended[row, col])
                or np.isnan(upper_limits[row, col])
                or np.isnan(lower_limits[row, col])
                or not np.isfinite(price)
                or price <= 0.0
            ):
                error_code = 3
                break
            if bool(suspended[row, col]):
                logs.append((row, col, -1, requested, 0.0, 2, 1))
                continue
            lower = lower_limits[row, col]
            if price <= lower or _isclose_nb(price, lower):
                logs.append((row, col, -1, requested, 0.0, 2, 2))
                continue

            executable = requested
            if max_participation_rate > 0.0:
                daily_volume = volumes[row, col]
                if (
                    np.isnan(daily_volume)
                    or not np.isfinite(daily_volume)
                    or daily_volume < 0.0
                ):
                    error_code = 7
                    break
                raw_capacity = daily_volume * max_participation_rate
                if requested > raw_capacity + 1e-12:
                    executable = (
                        np.floor(raw_capacity / lot_size) * lot_size
                    )
            if executable <= 0.0:
                logs.append((row, col, -1, requested, 0.0, 2, 6))
                continue

            execution_price = price * (1.0 - slippage)
            if execution_price <= lower or _isclose_nb(execution_price, lower):
                logs.append((row, col, -1, requested, 0.0, 2, 2))
                continue
            day_low = lows[row, col]
            if (
                np.isfinite(day_low)
                and execution_price < day_low
                and not _isclose_nb(execution_price, day_low)
            ):
                logs.append((row, col, -1, requested, 0.0, 2, 5))
                continue
            notional = executable * execution_price
            fixed = _fixed_fee_nb(notional, commission, minimum_commission)
            total_fee = notional * sell_rate + fixed
            cash += notional - total_fee
            shares[col] -= executable
            order_size[row, col] = -executable
            fee_rates[row, col] = sell_rate
            fixed_fees[row, col] = fixed
            call_seq[row, sequence_pos] = col
            sequence_pos += 1
            selected[col] = True
            if executable == requested:
                logs.append((row, col, -1, requested, executable, 1, 0))
            else:
                logs.append((row, col, -1, requested, executable, 4, 6))
        if error_code != 0:
            break

        for col in range(n_cols):
            requested = np.floor(
                (desired[col] - shares[col]) / lot_size
            ) * lot_size
            if requested > 0.0:
                requested_buys[col] = requested
                original_requested_buys[col] = requested

        # Buy candidates are always evaluated in lexical symbol order.
        requested_cost = 0.0
        for sorted_pos in range(n_cols):
            col = sorted_columns[sorted_pos]
            requested = requested_buys[col]
            if requested <= 0.0:
                continue
            price = prices[row, col]
            if (
                np.isnan(price)
                or np.isnan(suspended[row, col])
                or np.isnan(upper_limits[row, col])
                or np.isnan(lower_limits[row, col])
                or not np.isfinite(price)
                or price <= 0.0
            ):
                error_code = 3
                break
            if bool(suspended[row, col]):
                logs.append((row, col, 1, requested, 0.0, 2, 1))
                continue
            upper = upper_limits[row, col]
            if price >= upper or _isclose_nb(price, upper):
                logs.append((row, col, 1, requested, 0.0, 2, 3))
                continue
            execution_price = price * (1.0 + slippage)
            if execution_price >= upper or _isclose_nb(execution_price, upper):
                logs.append((row, col, 1, requested, 0.0, 2, 3))
                continue
            day_high = highs[row, col]
            if (
                np.isfinite(day_high)
                and execution_price > day_high
                and not _isclose_nb(execution_price, day_high)
            ):
                logs.append((row, col, 1, requested, 0.0, 2, 5))
                continue
            executable = requested
            if max_participation_rate > 0.0:
                daily_volume = volumes[row, col]
                if (
                    np.isnan(daily_volume)
                    or not np.isfinite(daily_volume)
                    or daily_volume < 0.0
                ):
                    error_code = 7
                    break
                raw_capacity = daily_volume * max_participation_rate
                if requested > raw_capacity + 1e-12:
                    executable = (
                        np.floor(raw_capacity / lot_size) * lot_size
                    )
            if executable <= 0.0:
                logs.append((row, col, 1, requested, 0.0, 2, 6))
                continue
            requested_buys[col] = executable
            tradable_buys[col] = True
            buy_prices[col] = execution_price
            requested_cost += _buy_cost_nb(
                executable,
                execution_price,
                buy_rate,
                commission,
                minimum_commission,
            )
        if error_code != 0:
            break

        scale = 0.0
        if requested_cost > 0.0:
            scale = min(1.0, cash / requested_cost)
        for sorted_pos in range(n_cols):
            col = sorted_columns[sorted_pos]
            if tradable_buys[col]:
                planned_buys[col] = (
                    np.floor(requested_buys[col] * scale / lot_size) * lot_size
                )

        # Account for minimum commissions.  The tie-break is lexical symbol
        # order because ``sorted_columns`` is traversed from left to right.
        while (
            _total_planned_cost_nb(
                sorted_columns,
                tradable_buys,
                planned_buys,
                buy_prices,
                buy_rate,
                commission,
                minimum_commission,
            )
            > cash + 1e-8
        ):
            reduce_col = -1
            smallest_deficit = np.inf
            for sorted_pos in range(n_cols):
                col = sorted_columns[sorted_pos]
                if tradable_buys[col] and planned_buys[col] >= lot_size:
                    deficit = requested_buys[col] - planned_buys[col]
                    if deficit < smallest_deficit:
                        smallest_deficit = deficit
                        reduce_col = col
            if reduce_col < 0:
                break
            planned_buys[reduce_col] -= lot_size

        # Stable mergesort preserves lexical order for equal deficits.
        candidate_count = 0
        candidate_columns = np.empty(n_cols, dtype=np.int64)
        candidate_scores = np.empty(n_cols, dtype=np.float64)
        for sorted_pos in range(n_cols):
            col = sorted_columns[sorted_pos]
            if tradable_buys[col]:
                candidate_columns[candidate_count] = col
                candidate_scores[candidate_count] = (
                    requested_buys[col] - planned_buys[col]
                ) * buy_prices[col]
                candidate_count += 1
        residual_order = np.argsort(
            -candidate_scores[:candidate_count],
            kind="mergesort",
        )
        for residual_pos in residual_order:
            col = candidate_columns[residual_pos]
            if planned_buys[col] + lot_size > requested_buys[col]:
                continue
            old_cost = _buy_cost_nb(
                planned_buys[col],
                buy_prices[col],
                buy_rate,
                commission,
                minimum_commission,
            )
            new_cost = _buy_cost_nb(
                planned_buys[col] + lot_size,
                buy_prices[col],
                buy_rate,
                commission,
                minimum_commission,
            )
            current_cost = _total_planned_cost_nb(
                sorted_columns,
                tradable_buys,
                planned_buys,
                buy_prices,
                buy_rate,
                commission,
                minimum_commission,
            )
            if current_cost + new_cost - old_cost <= cash + 1e-8:
                planned_buys[col] += lot_size

        for sorted_pos in range(n_cols):
            col = sorted_columns[sorted_pos]
            if not tradable_buys[col]:
                continue
            requested = requested_buys[col]
            original_requested = original_requested_buys[col]
            affordable = planned_buys[col]
            if affordable <= 0.0:
                logs.append((row, col, 1, original_requested, 0.0, 2, 4))
                continue

            execution_price = buy_prices[col]
            notional = affordable * execution_price
            fixed = _fixed_fee_nb(notional, commission, minimum_commission)
            total_fee = notional * buy_rate + fixed
            cash -= notional + total_fee
            shares[col] += affordable
            order_size[row, col] = affordable
            fee_rates[row, col] = buy_rate
            fixed_fees[row, col] = fixed
            call_seq[row, sequence_pos] = col
            sequence_pos += 1
            selected[col] = True
            if affordable < requested:
                logs.append((row, col, 1, original_requested, affordable, 3, 4))
            elif affordable < original_requested:
                logs.append((row, col, 1, original_requested, affordable, 4, 6))
            else:
                logs.append((row, col, 1, original_requested, affordable, 1, 0))

        for col in range(n_cols):
            if not selected[col]:
                call_seq[row, sequence_pos] = col
                sequence_pos += 1
            real_holdings[row, col] = shares[col]

    return (
        order_size,
        fee_rates,
        fixed_fees,
        call_seq,
        cash,
        shares,
        real_holdings,
        cash_deposits_out,
        asset_deposits_out,
        logs,
        error_code,
    )


def plan_ashare_orders_numba(
    close: pd.DataFrame,
    order_price: pd.DataFrame,
    scheduled_targets: pd.DataFrame,
    is_suspend: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    *,
    init_cash: float,
    costs: "ExecutionCosts",
    slippage: float = 0.0,
    lot_size: int = 100,
    adjustment_factor: pd.DataFrame | None = None,
    cash_dividend: pd.DataFrame | None = None,
    share_multiplier: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    max_participation_rate: float | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    python_fallback: Callable[..., "ExecutionPlan"],
) -> "ExecutionPlan":
    """Build an :class:`ExecutionPlan` using the compiled state machine."""
    # Import lazily to avoid a module cycle during execution.py initialization.
    from .execution import ExecutionPlan

    index, columns = close.index, close.columns
    if not scheduled_targets.index.isin(index).all():
        missing_dates = scheduled_targets.index.difference(index)
        raise ValueError(f"执行日期缺少行情: {list(missing_dates[:5])}")

    execution_rows = index.get_indexer(scheduled_targets.index).astype(np.int64)
    effective_targets = scheduled_targets.reindex(columns=columns)
    sorted_columns = np.argsort(
        np.asarray(columns, dtype=str),
        kind="stable",
    ).astype(np.int64)

    # The reference implementation is label-based (``.loc[date][symbol]``).
    # Reindex every numeric input so a caller cannot change semantics merely by
    # supplying the same labels in a different physical order.
    aligned_order_price = order_price.reindex(index=index, columns=columns)
    aligned_factor = (
        pd.DataFrame(1.0, index=index, columns=columns)
        if adjustment_factor is None
        else adjustment_factor.reindex(index=index, columns=columns)
    )
    aligned_cash_dividend = (
        pd.DataFrame(0.0, index=index, columns=columns)
        if cash_dividend is None
        else cash_dividend.reindex(index=index, columns=columns).fillna(0.0)
    )
    aligned_share_multiplier = (
        pd.DataFrame(1.0, index=index, columns=columns)
        if share_multiplier is None
        else share_multiplier.reindex(index=index, columns=columns).fillna(1.0)
    )
    aligned_volume = (
        pd.DataFrame(np.inf, index=index, columns=columns)
        if max_participation_rate is None
        else volume.reindex(index=index, columns=columns)
        if volume is not None
        else pd.DataFrame(np.nan, index=index, columns=columns)
    )
    aligned_high = (
        pd.DataFrame(np.inf, index=index, columns=columns)
        if high is None
        else high.reindex(index=index, columns=columns)
    )
    aligned_low = (
        pd.DataFrame(-np.inf, index=index, columns=columns)
        if low is None
        else low.reindex(index=index, columns=columns)
    )
    aligned_suspend = is_suspend.reindex(index=index, columns=columns)
    aligned_upper = high_limit.reindex(index=index, columns=columns)
    aligned_lower = low_limit.reindex(index=index, columns=columns)

    result = _plan_orders_nb(
        np.ascontiguousarray(aligned_order_price.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_factor.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_cash_dividend.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_share_multiplier.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_volume.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_high.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_low.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(effective_targets.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(execution_rows),
        np.ascontiguousarray(aligned_suspend.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_upper.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(aligned_lower.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(sorted_columns),
        float(init_cash),
        float(costs.commission),
        float(costs.stamp_tax),
        float(costs.transfer_fee),
        float(costs.minimum_commission),
        float(slippage),
        int(lot_size),
        -1.0 if max_participation_rate is None else float(max_participation_rate),
    )
    (
        order_size,
        fee_rates,
        fixed_fees,
        call_seq,
        final_cash,
        final_shares,
        real_holdings,
        cash_deposits_out,
        asset_deposits_out,
        encoded_logs,
        error_code,
    ) = result

    if error_code:
        return python_fallback(
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

    side_names = {-1: "sell", 1: "buy"}
    status_names = {
        1: "filled",
        2: "blocked",
        3: "partial_cash",
        4: "partial_volume",
    }
    reason_names = {
        0: "",
        1: "suspend",
        2: "limit_down",
        3: "limit_up",
        4: "cash",
        5: "price_bound",
        6: "volume",
    }
    log_rows = [
        {
            "date": index[row],
            "symbol": columns[col],
            "side": side_names[side],
            "requested": requested,
            "filled": filled,
            "status": status_names[status],
            "reason": reason_names[reason],
        }
        for row, col, side, requested, filled, status, reason in encoded_logs
    ]

    return ExecutionPlan(
        order_size=pd.DataFrame(order_size, index=index, columns=columns),
        fees=pd.DataFrame(fee_rates, index=index, columns=columns),
        fixed_fees=pd.DataFrame(fixed_fees, index=index, columns=columns),
        call_seq=call_seq,
        log=pd.DataFrame(log_rows),
        final_cash=float(final_cash),
        final_shares=pd.Series(final_shares, index=columns),
        real_holdings=pd.DataFrame(real_holdings, index=index, columns=columns),
        cash_deposits=pd.DataFrame(
            cash_deposits_out,
            index=index,
            columns=columns,
        ),
        asset_deposits=pd.DataFrame(
            asset_deposits_out,
            index=index,
            columns=columns,
        ),
    )
