# -*- coding: utf-8 -*-
"""A-share limit-price / suspension state-machine operators (2026-08 pack, group 4).

Multi-day state transitions complement the single-day masks
(``ashare_limit_up_touch``, ``ashare_limit_one_price``, …): streaks, days-since
events, touch/failed counts, event density/asymmetry and suspension episodes.

Missing-value policy
--------------------
* NaN in any input, an unknown status or a suspended day ``valid_trade == 0``
  **breaks** a streak — it is never treated as a non-event.
* Streak outputs are NaN before the first valid observation and on an
  unknown/suspended current day; a current valid day that is not at the limit
  yields 0 (not NaN).
* Event densities divide by *known* tradeable days, never by a fixed window
  length.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.strict_params import strict_nonnegative_int
from factor_engine.cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str = "count") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="ashare",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:trading_state",
            f"unit:{unit}", "cost:1",
        ],
    )


def _tolerance(value: Any) -> float:
    tolerance = float(value)
    if tolerance < 0.0:
        raise ValueError("tick_tolerance must be non-negative")
    return tolerance


def _tradeable(valid_trade: np.ndarray, r: int, c: int) -> bool:
    # NEW-050: TradableBool is strictly {0, 1, NaN}.  NaN = unknown -> not
    # tradeable (break); 1 = tradeable; 0 = not tradeable.  The old
    # ``finite and value != 0`` accepted 0.2/-1/2 as "tradeable" — a 0.2 float
    # condition is neither a boolean nor a probability the state machine is
    # allowed to interpret.  Any value outside {0, 1, NaN} is a data-quality
    # error (never silently true/false).
    value = valid_trade[r, c]
    if value != value:  # NaN
        return False
    if value == 1.0:
        return True
    if value == 0.0:
        return False
    raise ValueError(
        f"valid_trade must be a TradableBool ({{0, 1, NaN}}); got {value!r} at "
        "row/col ({r},{c}) — a 0.2 / -1 / 2 value is not a tradeable flag"
    )


def _break_mask(row: int, col: int, arrays: list[np.ndarray]) -> bool:
    return any(not np.isfinite(arr[row, col]) for arr in arrays)


def _consecutive_streak(
    condition: np.ndarray,
    valid: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Per-cell consecutive-true streak ending at the current row.

    ``condition`` is the per-day boolean (computed, may be None on break rows);
    ``valid`` marks tradeable/known days.  NaN/unknown breaks the run and the
    current row outputs NaN.
    """
    out = np.full((rows, cols), np.nan, dtype=float)
    run = np.zeros(cols, dtype=np.int64)
    active = np.zeros(cols, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c] or not np.isfinite(condition[r, c]):
                out[r, c] = np.nan
                run[c] = 0
                active[c] = False
                continue
            if not active[c]:
                active[c] = True
                run[c] = 0
            run[c] = run[c] + 1 if bool(condition[r, c]) else 0
            out[r, c] = float(run[c])
    return out


@register_operator(
    name="ashare_limit_up_streak",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_up_streak",
    source="ashare.state_machine",
)
class AshareLimitUpStreak(SeriesOperator):
    """连续收盘涨停天数：缺失/停牌/未知状态打断，不视为非涨停。"""

    metadata = _metadata(
        "ashare_limit_up_streak",
        "连续收盘涨停天数。",
        ["close", "high_limit", "valid_trade", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, close: pd.DataFrame, high_limit: pd.DataFrame, valid_trade: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tol = _tolerance(tick_tolerance)
        rows, cols = close.shape
        cv = close.to_numpy(dtype=float)
        lv = high_limit.to_numpy(dtype=float)
        vv = valid_trade.to_numpy(dtype=float)
        condition = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                if _break_mask(r, c, [cv, lv]) or not _tradeable(vv, r, c):
                    continue
                if np.isfinite(lv[r, c]) and cv[r, c] >= lv[r, c] * (1.0 - tol):
                    condition[r, c] = 1.0
                else:
                    condition[r, c] = 0.0
        tradeable = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                tradeable[r, c] = _tradeable(vv, r, c)
        return frame_like(close, _consecutive_streak(condition, tradeable, rows, cols))


@register_operator(
    name="ashare_limit_down_streak",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_down_streak",
    source="ashare.state_machine",
)
class AshareLimitDownStreak(SeriesOperator):
    """连续收盘跌停天数：缺失/停牌/未知状态打断。"""

    metadata = _metadata(
        "ashare_limit_down_streak",
        "连续收盘跌停天数。",
        ["close", "low_limit", "valid_trade", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, close: pd.DataFrame, low_limit: pd.DataFrame, valid_trade: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tol = _tolerance(tick_tolerance)
        rows, cols = close.shape
        cv = close.to_numpy(dtype=float)
        lv = low_limit.to_numpy(dtype=float)
        vv = valid_trade.to_numpy(dtype=float)
        condition = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                if _break_mask(r, c, [cv, lv]) or not _tradeable(vv, r, c):
                    continue
                if np.isfinite(lv[r, c]) and cv[r, c] <= lv[r, c] * (1.0 + tol):
                    condition[r, c] = 1.0
                else:
                    condition[r, c] = 0.0
        tradeable = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                tradeable[r, c] = _tradeable(vv, r, c)
        return frame_like(close, _consecutive_streak(condition, tradeable, rows, cols))


def _days_since(condition: np.ndarray, max_lookback: int | None) -> np.ndarray:
    rows, cols = condition.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    last_event = np.full(cols, -1, dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            value = condition[r, c]
            if not np.isfinite(value):
                out[r, c] = np.nan
                last_event[c] = -1
                continue
            if bool(value != 0):
                last_event[c] = r
            if last_event[c] >= 0:
                distance = r - last_event[c]
                if max_lookback is None or distance < max_lookback:
                    out[r, c] = float(distance)
    return out


@register_operator(
    name="ashare_days_since_limit_up",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_days_since_limit_up",
    source="ashare.state_machine",
)
class AshareDaysSinceLimitUp(SeriesOperator):
    """距最近一次收盘涨停的天数；首次有效观测前返回 NaN。"""

    metadata = _metadata(
        "ashare_days_since_limit_up",
        "距最近涨停天数（缺失打断）。",
        ["limit_up_event", "max_lookback"],
        unit="count",
    )

    def _calculate_series(self, limit_up_event: pd.DataFrame, max_lookback: Any = None, **_: Any) -> pd.DataFrame:
        # R25-005/167: strict non-negative integer authority — reject fractional /
        # NaN / Inf / bool instead of silently truncating ``int(3.7) -> 3``.
        limit_v = (
            None
            if max_lookback is None
            else strict_nonnegative_int(max_lookback, "max_lookback")
        )
        return frame_like(limit_up_event, _days_since(limit_up_event.to_numpy(dtype=float), limit_v))


@register_operator(
    name="ashare_days_since_limit_down",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_days_since_limit_down",
    source="ashare.state_machine",
)
class AshareDaysSinceLimitDown(SeriesOperator):
    """距最近一次收盘跌停的天数；首次有效观测前返回 NaN。"""

    metadata = _metadata(
        "ashare_days_since_limit_down",
        "距最近跌停天数（缺失打断）。",
        ["limit_down_event", "max_lookback"],
        unit="count",
    )

    def _calculate_series(self, limit_down_event: pd.DataFrame, max_lookback: Any = None, **_: Any) -> pd.DataFrame:
        # R25-005/167: strict non-negative integer authority (no silent int()).
        limit_v = (
            None
            if max_lookback is None
            else strict_nonnegative_int(max_lookback, "max_lookback")
        )
        return frame_like(limit_down_event, _days_since(limit_down_event.to_numpy(dtype=float), limit_v))


def _rolling_count(condition: np.ndarray, known: np.ndarray, window: int) -> np.ndarray:
    rows, cols = condition.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        lo = max(0, r - window + 1)
        for c in range(cols):
            if not known[r, c]:
                continue
            chunk = condition[lo : r + 1, c]
            if int(np.isfinite(chunk).sum()) == 0:
                continue
            out[r, c] = float(np.nansum(chunk))
    return out


@register_operator(
    name="ashare_limit_touch_count",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_touch_count",
    source="ashare.state_machine",
)
class AshareLimitTouchCount(SeriesOperator):
    """窗口内触及涨停/跌停的天数；无有效数据的天返回 NaN。"""

    metadata = _metadata(
        "ashare_limit_touch_count",
        "窗口内触及涨跌停天数（side='up'/'down'）。",
        ["high", "low", "high_limit", "low_limit", "window", "side", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, high_limit: pd.DataFrame, low_limit: pd.DataFrame, window: int = 20, side: str = "up", tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        tol = _tolerance(tick_tolerance)
        side_kind = str(side).lower()
        if side_kind not in {"up", "down"}:
            raise ValueError("side must be 'up' or 'down'")
        rows, cols = high.shape
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        hlv = high_limit.to_numpy(dtype=float)
        llv = low_limit.to_numpy(dtype=float)
        condition = np.zeros((rows, cols), dtype=float)
        known = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if side_kind == "up":
                    if np.isfinite(hv[r, c]) and np.isfinite(hlv[r, c]):
                        known[r, c] = True
                        if hv[r, c] >= hlv[r, c] * (1.0 - tol):
                            condition[r, c] = 1.0
                else:
                    if np.isfinite(lv[r, c]) and np.isfinite(llv[r, c]):
                        known[r, c] = True
                        if lv[r, c] <= llv[r, c] * (1.0 + tol):
                            condition[r, c] = 1.0
        return frame_like(high, _rolling_count(condition, known, w))


@register_operator(
    name="ashare_failed_limit_count",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_failed_limit_count",
    source="ashare.state_machine",
)
class AshareFailedLimitCount(SeriesOperator):
    """窗口内炸板/开板天数：触及涨停但未收于涨停（或相反）。"""

    metadata = _metadata(
        "ashare_failed_limit_count",
        "窗口内触及未封住的天数（side='up'/'down'）。",
        ["high", "low", "close", "high_limit", "low_limit", "window", "side", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, high_limit: pd.DataFrame, low_limit: pd.DataFrame, window: int = 20, side: str = "up", tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        tol = _tolerance(tick_tolerance)
        side_kind = str(side).lower()
        if side_kind not in {"up", "down"}:
            raise ValueError("side must be 'up' or 'down'")
        rows, cols = high.shape
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        hlv = high_limit.to_numpy(dtype=float)
        llv = low_limit.to_numpy(dtype=float)
        condition = np.zeros((rows, cols), dtype=float)
        known = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if side_kind == "up":
                    ok = all(np.isfinite(v) for v in (hv[r, c], cv[r, c], hlv[r, c]))
                    if ok:
                        known[r, c] = True
                        if hv[r, c] >= hlv[r, c] * (1.0 - tol) and cv[r, c] < hlv[r, c] * (1.0 - tol):
                            condition[r, c] = 1.0
                else:
                    ok = all(np.isfinite(v) for v in (lv[r, c], cv[r, c], llv[r, c]))
                    if ok:
                        known[r, c] = True
                        if lv[r, c] <= llv[r, c] * (1.0 + tol) and cv[r, c] > llv[r, c] * (1.0 + tol):
                            condition[r, c] = 1.0
        return frame_like(high, _rolling_count(condition, known, w))


def _one_price_mask(open_v, high_v, low_v, close_v, limit, tol, rows, cols) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if not all(np.isfinite(v) for v in (open_v[r, c], high_v[r, c], low_v[r, c], close_v[r, c], limit[r, c])):
                continue
            prices = (open_v[r, c], high_v[r, c], low_v[r, c], close_v[r, c])
            # 一字板：全部 OHLC 都在 limit 的 tick 容差内（相对容差）。
            if all(abs(price / limit[r, c] - 1.0) <= tol for price in prices):
                mask[r, c] = 1.0
    return mask


@register_operator(
    name="ashare_one_price_limit_streak",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_one_price_limit_streak",
    source="ashare.state_machine",
)
class AshareOnePriceLimitStreak(SeriesOperator):
    """连续一字涨停/跌停天数：缺失/停牌/未知状态打断。"""

    metadata = _metadata(
        "ashare_one_price_limit_streak",
        "连续一字板天数（side='up'/'down'）。",
        ["open", "high", "low", "close", "high_limit", "low_limit", "valid_trade", "side", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, open_p: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, high_limit: pd.DataFrame, low_limit: pd.DataFrame, valid_trade: pd.DataFrame, side: str = "up", tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tol = _tolerance(tick_tolerance)
        side_kind = str(side).lower()
        if side_kind not in {"up", "down"}:
            raise ValueError("side must be 'up' or 'down'")
        rows, cols = close.shape
        ov = open_p.to_numpy(dtype=float)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        limit = high_limit.to_numpy(dtype=float) if side_kind == "up" else low_limit.to_numpy(dtype=float)
        vv = valid_trade.to_numpy(dtype=float)
        condition = _one_price_mask(ov, hv, lv, cv, limit, tol, rows, cols)
        for r in range(rows):
            for c in range(cols):
                if not _tradeable(vv, r, c):
                    condition[r, c] = np.nan
        tradeable = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                tradeable[r, c] = _tradeable(vv, r, c)
        return frame_like(close, _consecutive_streak(condition, tradeable, rows, cols))


@register_operator(
    name="ashare_limit_event_density",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_event_density",
    source="ashare.state_machine",
)
class AshareLimitEventDensity(SeriesOperator):
    """事件密度：窗口内事件数 / 已知可交易状态日数。"""

    metadata = _metadata(
        "ashare_limit_event_density",
        "窗口内事件密度（除以已知状态日数）。",
        ["event", "window", "known_status"],
        unit="ratio",
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 20, known_status: pd.DataFrame = None, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        ev = event.to_numpy(dtype=float)
        if known_status is not None:
            kv = known_status.to_numpy(dtype=float)
        else:
            kv = np.ones_like(ev)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            lo = max(0, r - w + 1)
            for c in range(cols):
                if not np.isfinite(kv[r, c]):
                    continue
                known = kv[lo : r + 1, c] != 0
                known = known & np.isfinite(ev[lo : r + 1, c])
                known_count = int(known.sum())
                if known_count == 0:
                    continue
                count = float(np.nansum(np.where(known, ev[lo : r + 1, c], 0.0)))
                out[r, c] = count / known_count
        return frame_like(event, out)


@register_operator(
    name="ashare_limit_asymmetry",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_asymmetry",
    source="ashare.state_machine",
)
class AshareLimitAsymmetry(SeriesOperator):
    """涨停/跌停密度差：窗口内涨停密度 - 跌停密度。"""

    metadata = _metadata(
        "ashare_limit_asymmetry",
        "涨停密度减跌停密度。",
        ["up_event", "down_event", "window"],
        unit="ratio",
    )

    def _calculate_series(self, up_event: pd.DataFrame, down_event: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        uv = up_event.to_numpy(dtype=float)
        dv = down_event.to_numpy(dtype=float)
        rows, cols = uv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            lo = max(0, r - w + 1)
            for c in range(cols):
                u_chunk = uv[lo : r + 1, c]
                d_chunk = dv[lo : r + 1, c]
                known = np.isfinite(u_chunk) | np.isfinite(d_chunk)
                known_count = int(known.sum())
                if known_count == 0:
                    continue
                up_count = float(np.nansum(np.where(known & (u_chunk != 0), u_chunk, 0.0)))
                down_count = float(np.nansum(np.where(known & (d_chunk != 0), d_chunk, 0.0)))
                out[r, c] = (up_count - down_count) / known_count
        return frame_like(up_event, out)


@register_operator(
    name="ashare_suspension_episode_length",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_suspension_episode_length",
    source="ashare.state_machine",
)
class AshareSuspensionEpisodeLength(SeriesOperator):
    """当前停牌连续长度：停牌中输出当前连续天数，正常交易日输出 0，未知输出 NaN。"""

    metadata = _metadata(
        "ashare_suspension_episode_length",
        "当前停牌连续长度。",
        ["is_suspend"],
        unit="count",
    )

    def _calculate_series(self, is_suspend: pd.DataFrame, **_: Any) -> pd.DataFrame:
        rows, cols = is_suspend.shape
        sv = is_suspend.to_numpy(dtype=float)
        out = np.full((rows, cols), np.nan, dtype=float)
        run = np.zeros(cols, dtype=np.int64)
        for r in range(rows):
            for c in range(cols):
                value = sv[r, c]
                # NEW-051: is_suspend is a strict EventBool {0, 1, NaN}.  NaN =
                # unknown (break); 1 = suspended; 0 = not suspended.  The old
                # ``value != 0`` counted 0.2/-1/2 as "suspended" — a non-boolean
                # flag is a data-quality error, never a suspend state.
                if value != value:  # NaN
                    out[r, c] = np.nan
                    run[c] = 0
                    continue
                if value == 1.0:
                    run[c] += 1
                    out[r, c] = float(run[c])
                elif value == 0.0:
                    run[c] = 0
                    out[r, c] = 0.0
                else:
                    raise ValueError(
                        f"is_suspend must be an EventBool ({{0, 1, NaN}}); got "
                        f"{value!r} at row/col ({r},{c})"
                    )
        return frame_like(is_suspend, out)


def _open_limit_streak(open_v: np.ndarray, limit: np.ndarray, valid: np.ndarray, tol: float) -> np.ndarray:
    rows, cols = open_v.shape
    condition = np.full((rows, cols), np.nan, dtype=float)
    tradeable = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            tradeable[r, c] = _tradeable(valid, r, c)
            if not tradeable[r, c]:
                continue
            if np.isfinite(open_v[r, c]) and np.isfinite(limit[r, c]):
                condition[r, c] = 1.0 if open_v[r, c] >= limit[r, c] * (1.0 - tol) else 0.0
    return _consecutive_streak(condition, tradeable, rows, cols)


@register_operator(
    name="ashare_limit_open_up_streak",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_open_up_streak",
    source="ashare.state_machine",
)
class AshareLimitOpenUpStreak(SeriesOperator):
    """连续开盘涨停天数。"""

    metadata = _metadata(
        "ashare_limit_open_up_streak",
        "连续开盘涨停天数。",
        ["open", "high_limit", "valid_trade", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, open_p: pd.DataFrame, high_limit: pd.DataFrame, valid_trade: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        return frame_like(
            open_p,
            _open_limit_streak(
                open_p.to_numpy(dtype=float),
                high_limit.to_numpy(dtype=float),
                valid_trade.to_numpy(dtype=float),
                _tolerance(tick_tolerance),
            ),
        )


@register_operator(
    name="ashare_limit_open_down_streak",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_open_down_streak",
    source="ashare.state_machine",
)
class AshareLimitOpenDownStreak(SeriesOperator):
    """连续开盘跌停天数。"""

    metadata = _metadata(
        "ashare_limit_open_down_streak",
        "连续开盘跌停天数。",
        ["open", "low_limit", "valid_trade", "tick_tolerance"],
        unit="count",
    )

    def _calculate_series(self, open_p: pd.DataFrame, low_limit: pd.DataFrame, valid_trade: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        ov = open_p.to_numpy(dtype=float)
        limit = low_limit.to_numpy(dtype=float)
        rows, cols = ov.shape
        condition = np.full((rows, cols), np.nan, dtype=float)
        tradeable = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                tradeable[r, c] = _tradeable(valid_trade.to_numpy(dtype=float), r, c)
                if not tradeable[r, c]:
                    continue
                if np.isfinite(ov[r, c]) and np.isfinite(limit[r, c]):
                    condition[r, c] = 1.0 if ov[r, c] <= limit[r, c] * (1.0 + _tolerance(tick_tolerance)) else 0.0
        return frame_like(open_p, _consecutive_streak(condition, tradeable, rows, cols))


def _event_volume_ratio(volume: np.ndarray, event: np.ndarray, window: int) -> np.ndarray:
    rows, cols = volume.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        lo = max(0, r - window + 1)
        for c in range(cols):
            v = volume[lo : r + 1, c]
            e = event[lo : r + 1, c]
            valid_vol = np.isfinite(v)
            event_days = valid_vol & np.isfinite(e) & (e != 0)
            event_count = int(event_days.sum())
            if event_count == 0 or int(valid_vol.sum()) == 0:
                continue
            event_vol = float(np.mean(v[event_days]))
            base_vol = float(np.mean(v[valid_vol]))
            if base_vol <= 0.0:
                continue
            out[r, c] = event_vol / base_vol
    return out


@register_operator(
    name="ashare_limit_up_volume_ratio",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_up_volume_ratio",
    source="ashare.state_machine",
)
class AshareLimitUpVolumeRatio(SeriesOperator):
    """涨停日均量 / 窗口日均量：涨停日放量程度。"""

    metadata = _metadata(
        "ashare_limit_up_volume_ratio",
        "涨停日均量相对窗口日均量。",
        ["volume", "limit_up_event", "window"],
        unit="ratio",
    )

    def _calculate_series(self, volume: pd.DataFrame, limit_up_event: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(volume, _event_volume_ratio(volume.to_numpy(dtype=float), limit_up_event.to_numpy(dtype=float), w))


@register_operator(
    name="ashare_limit_down_volume_ratio",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_down_volume_ratio",
    source="ashare.state_machine",
)
class AshareLimitDownVolumeRatio(SeriesOperator):
    """跌停日均量 / 窗口日均量：跌停日放量程度。"""

    metadata = _metadata(
        "ashare_limit_down_volume_ratio",
        "跌停日均量相对窗口日均量。",
        ["volume", "limit_down_event", "window"],
        unit="ratio",
    )

    def _calculate_series(self, volume: pd.DataFrame, limit_down_event: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(volume, _event_volume_ratio(volume.to_numpy(dtype=float), limit_down_event.to_numpy(dtype=float), w))


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ashare_limit_up_streak", "ashare_limit_down_streak",
            "ashare_days_since_limit_up", "ashare_days_since_limit_down",
            "ashare_limit_touch_count", "ashare_failed_limit_count",
            "ashare_one_price_limit_streak", "ashare_limit_event_density",
            "ashare_limit_asymmetry", "ashare_suspension_episode_length",
            "ashare_limit_open_up_streak", "ashare_limit_open_down_streak",
            "ashare_limit_up_volume_ratio", "ashare_limit_down_volume_ratio",
        })
    for _canon in (
        "ashare_limit_up_streak", "ashare_limit_down_streak",
        "ashare_days_since_limit_up", "ashare_days_since_limit_down",
        "ashare_limit_touch_count", "ashare_failed_limit_count",
        "ashare_one_price_limit_streak", "ashare_limit_event_density",
        "ashare_limit_asymmetry", "ashare_suspension_episode_length",
        "ashare_limit_open_up_streak", "ashare_limit_open_down_streak",
        "ashare_limit_up_volume_ratio", "ashare_limit_down_volume_ratio",
    ):
        register_polars_bridge(_canon)


_register_surface()
