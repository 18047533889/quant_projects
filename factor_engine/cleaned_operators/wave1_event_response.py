# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: event-response decay.

New canonicals across genuinely new thematic ground:
  * event count/age-weighted decay kernels (half-life decay count, recency
    decay, event-rate decay slope)
  * marked-event response family (decay of marks, sign-consistent decay,
    cross-event spacing, pre/post event hazard)
  * event-window return gradient (pre-event baseline vs post-event drift)

All are real pandas_numpy implementations, causal, deterministic and NaN-safe.
Names are prefixed ``erd_`` and are globally unique.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned, _check_int


def _metadata(
    name: str, description: str, params: list[str], *, domain: str, unit: str,
    cost: int = 1, category: str,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "wave1", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
        output_unit=unit if unit.startswith(("same_as:", "unit(")) else None,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# 1. event count/age-weighted decay kernels
# ---------------------------------------------------------------------------
@register_operator(
    name="erd_half_life_decay_count",
    category="event_response",
    business_category="event_response",
    canonical="erd_half_life_decay_count",
    source="wave1_event_response",
    status="experimental")
class ErdHalfLifeDecayCount(SeriesOperator):
    """半衰期衰减事件计数：sum_{s<=t} 1_{event(s)} × 0.5^((t-s)/half_life)。

    事件输入为 EventBool（有限非零 = 事件）。输出为事件密度/压力的因果衰减
    累计，半衰期越长事件越持久。首次有效观测前输出 NaN。
    """

    metadata = _metadata(
        "erd_half_life_decay_count",
        "事件半衰期衰减累计计数（EventBool × 0.5^(age/half_life)）。",
        ["event", "half_life", "min_periods"],
        domain="event",
        unit="dimensionless",
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, half_life: float = 10.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        mp = _check_int(min_periods, "min_periods", 1)
        weight = 0.5 ** (1.0 / hl)
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = 0
            for row in range(rows):
                v = ev[row, col]
                if np.isfinite(v):
                    if v != 0.0:
                        acc = acc * weight + 1.0
                    else:
                        acc = acc * weight
                    seen += 1
                    if seen >= mp:
                        out[row, col] = acc
                else:
                    if seen >= mp:
                        acc = acc * weight
                        out[row, col] = acc
        return _frame_like(event, out)


@register_operator(
    name="erd_recency_decay",
    category="event_response",
    business_category="event_response",
    canonical="erd_recency_decay",
    source="wave1_event_response",
    status="experimental")
class ErdRecencyDecay(SeriesOperator):
    """近期度衰减：距最近事件的指数衰减强度 exp(-age/half_life)。

    输出距最近事件的时间衰减权重（近期事件 = 高权重）。事件越久远越小。
    事件冷启动前输出 0。
    """

    metadata = _metadata(
        "erd_recency_decay",
        "exp(-age/half_life)，距最近事件的时间衰减权重。",
        ["event", "half_life"],
        domain="event",
        unit="ratio",
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, half_life: float = 10.0, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            age: float | None = None
            for row in range(rows):
                v = ev[row, col]
                if np.isfinite(v) and v != 0.0:
                    age = 0.0
                elif age is None:
                    out[row, col] = np.nan
                    continue
                else:
                    age = age + 1.0
                out[row, col] = float(np.exp(-age / hl))
        return _frame_like(event, out)


@register_operator(
    name="erd_event_rate_decay_slope",
    category="event_response",
    business_category="event_response",
    canonical="erd_event_rate_decay_slope",
    source="wave1_event_response",
    status="experimental")
class ErdEventRateDecaySlope(SeriesOperator):
    """事件率衰减斜率：窗口前半段事件密度与后半段事件密度的对数差。

    判断事件频率是在上升还是下降趋势。>0 = 事件率上升（hot），<0 = 衰减
    （cooling）。
    """

    metadata = _metadata(
        "erd_event_rate_decay_slope",
        "窗口前半段 vs 后半段事件率的对数差。",
        ["event", "window", "min_events"],
        domain="event",
        unit="dimensionless",
        cost=2,
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 40, min_events: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 6)
        me = _check_int(min_events, "min_events", 2)
        if w % 2 == 1:
            w = w - 1
        if w < 6:
            w = 6
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < w:
                    continue
                vals = chunk[ok]
                mid = w // 2
                first = vals[:mid]
                second = vals[mid:]
                n1 = float(np.sum(first != 0))
                n2 = float(np.sum(second != 0))
                if n1 + n2 < me:
                    continue
                den = np.log(max(n1, 1e-9))
                num = np.log(max(n2, 1e-9)) - den
                out[row, col] = num
        return _frame_like(event, out)


# ---------------------------------------------------------------------------
# 2. marked-event response family
# ---------------------------------------------------------------------------
@register_operator(
    name="erd_marked_event_decay",
    category="event_response",
    business_category="event_response",
    canonical="erd_marked_event_decay",
    source="wave1_event_response",
    status="experimental")
class ErdMarkedEventDecay(SeriesOperator):
    """标记事件半衰期衰减：sum_{s<=t} mark_s × 0.5^((t-s)/half_life)。

    每次事件贡献其幅度 mark（保留符号）。输出单位继承 mark。对事件"冲击"
    的持续影响建模。
    """

    metadata = _metadata(
        "erd_marked_event_decay",
        "标记事件幅度 × 半衰期指数衰减累计。",
        ["mark", "half_life", "min_periods"],
        domain="event",
        unit="same_as:mark",
        category="event_response",
    )

    def _calculate_series(self, mark: pd.DataFrame, half_life: float = 10.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        mp = _check_int(min_periods, "min_periods", 1)
        weight = 0.5 ** (1.0 / hl)
        mv = mark.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = 0
            for row in range(rows):
                v = mv[row, col]
                if np.isfinite(v):
                    acc = acc * weight + float(v)
                    seen += 1
                    if seen >= mp:
                        out[row, col] = acc
                else:
                    if seen >= mp:
                        acc = acc * weight
                        out[row, col] = acc
        return _frame_like(mark, out)


@register_operator(
    name="erd_sign_consistent_decay",
    category="event_response",
    business_category="event_response",
    canonical="erd_sign_consistent_decay",
    source="wave1_event_response",
    status="experimental")
class ErdSignConsistentDecay(SeriesOperator):
    """符号一致衰减：sign(mark) 仅累积共向事件的半衰期总和。

    sum_{s<=t} (sign(mark_s) == d ? |mark_s| : 0) × decay。d 为方向参数
    (1 正 / -1 负)，只对同向事件敏感。输出单位继承 mark。
    """

    metadata = _metadata(
        "erd_sign_consistent_decay",
        "只累积指定方向标记事件的半衰期衰减。",
        ["mark", "direction", "half_life", "min_periods"],
        domain="event",
        unit="same_as:mark",
        category="event_response",
    )

    def _calculate_series(self, mark: pd.DataFrame, direction: int = 1, half_life: float = 10.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        d = int(direction)
        if d not in (1, -1):
            raise ValueError("direction must be 1 or -1")
        mp = _check_int(min_periods, "min_periods", 1)
        weight = 0.5 ** (1.0 / hl)
        mv = mark.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = 0
            for row in range(rows):
                v = mv[row, col]
                if np.isfinite(v):
                    if (v > 0 and d == 1) or (v < 0 and d == -1):
                        acc = acc * weight + abs(float(v))
                    else:
                        acc = acc * weight
                    seen += 1
                    if seen >= mp:
                        out[row, col] = acc
                else:
                    if seen >= mp:
                        acc = acc * weight
                        out[row, col] = acc
        return _frame_like(mark, out)


@register_operator(
    name="erd_cross_events_spacing",
    category="event_response",
    business_category="event_response",
    canonical="erd_cross_events_spacing",
    source="wave1_event_response",
    status="experimental")
class ErdCrossEventsSpacing(SeriesOperator):
    """跨事件间隔：窗口内相邻事件间距的均值（同 es1 风格，但以 "事件" 为单位）。

    衡量事件簇的密集成度；小间距 = 事件密集爆发。当前窗口至少 2 个事件才
    输出。
    """

    metadata = _metadata(
        "erd_cross_events_spacing",
        "窗口内相邻事件间距均值（事件簇密度）。",
        ["event", "window", "min_events"],
        domain="event",
        unit="count",
        cost=2,
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 60, min_events: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        me = _check_int(min_events, "min_events", 2)
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < w // 2:
                    continue
                vals = chunk[ok]
                pos = np.flatnonzero(vals != 0)
                if pos.size < me:
                    continue
                gaps = np.diff(pos)
                out[row, col] = float(np.mean(gaps))
        return _frame_like(event, out)


@register_operator(
    name="erd_post_event_hazard",
    category="event_response",
    business_category="event_response",
    canonical="erd_post_event_hazard",
    source="wave1_event_response",
    status="experimental")
class ErdPostEventHazard(SeriesOperator):
    """事件后 hazard：窗口内事件后 k 行内再次发生事件的概率。

    衡量事件自驱动性（事件簇）：高 = 事件后容易再次发生（self-exciting）。
    输出 ratio。
    """

    metadata = _metadata(
        "erd_post_event_hazard",
        "事件后 k 行内再次发生事件的占比。",
        ["event", "window", "k", "min_events"],
        domain="event",
        unit="ratio",
        cost=2,
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 60, k: int = 5, min_events: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 4)
        kk = _check_int(k, "k", 1)
        me = _check_int(min_events, "min_events", 2)
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            ev_col = ev[:, col]
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = ev_col[lo:row + 1]
                ok = np.isfinite(chunk)
                if ok.sum() < w // 2:
                    continue
                vals = chunk
                positions = np.flatnonzero(vals != 0)
                if positions.size < me:
                    continue
                followed = 0
                for p in positions:
                    if p + kk < vals.size:
                        if np.any(vals[p + 1:p + kk + 1] != 0):
                            followed += 1
                out[row, col] = followed / positions.size
        return _frame_like(event, out)


# ---------------------------------------------------------------------------
# 3. event-window return gradient
# ---------------------------------------------------------------------------
@register_operator(
    name="erd_event_window_return_gradient",
    category="event_response",
    business_category="event_response",
    canonical="erd_event_window_return_gradient",
    source="wave1_event_response",
    status="experimental")
class ErdEventWindowReturnGradient(SeriesOperator):
    """事件窗口收益梯度：事件后窗口收益均值 vs 事件前基线收益均值。

    post - pre 衡量事件（按 event 标记）驱动的收益差异。正值 = 事件后正漂移。
    输出 return 单位。
    """

    metadata = _metadata(
        "erd_event_window_return_gradient",
        "事件后窗口收益均值 - 事件前基线收益均值。",
        ["ret", "event", "pre_window", "post_window", "min_events"],
        domain="price_volume",
        unit="same_as:ret",
        cost=2,
        category="event_response",
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, pre_window: int = 10, post_window: int = 10, min_events: int = 2, **_: Any) -> pd.DataFrame:
        pre = _check_int(pre_window, "pre_window", 1)
        post = _check_int(post_window, "post_window", 1)
        me = _check_int(min_events, "min_events", 1)
        from factor_engine.cleaned_operators.common.daily_panel import _aligned as _al
        ret, event = _al(ret, event)
        rv = ret.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - post + 1)
                post_chunk = rv[lo:row + 1, col]
                ok_post = np.isfinite(post_chunk)
                if ok_post.sum() < 1:
                    continue
                # 事件前基线：回溯 pre_window 行的收益
                pre_lo = max(0, row - pre - post + 1)
                pre_chunk = rv[pre_lo:row - post + 1, col]
                ok_pre = np.isfinite(pre_chunk)
                event_chunk = ev[max(0, lo - k if False else pre_lo):row + 1 if True else row + 1, col]
                # 使用最近窗口内事件计数满足 min_events
                ev_ok = np.isfinite(ev[max(0, pre_lo):row + 1, col])
                ev_vals = ev[max(0, pre_lo):row + 1, col][ev_ok]
                if np.sum(ev_vals != 0) < me:
                    continue
                if ok_pre.sum() < 1:
                    continue
                pm = float(np.mean(pre_chunk[ok_pre]))
                po = float(np.mean(post_chunk[ok_post]))
                out[row, col] = po - pm
        return _frame_like(ret, out)


@register_operator(
    name="erd_event_response_amplitude",
    category="event_response",
    business_category="event_response",
    canonical="erd_event_response_amplitude",
    source="wave1_event_response",
    status="experimental")
class ErdEventResponseAmplitude(SeriesOperator):
    """事件响应幅度：|mark| 的半衰期累计（冲击总强度）。

    对标记事件幅度做半衰期平滑，捕获事件冲击的残留幅度。输出单位继承 mark。
    """

    metadata = _metadata(
        "erd_event_response_amplitude",
        "|mark| 半衰期指数累计（事件冲击幅度残留）。",
        ["mark", "half_life", "min_periods"],
        domain="event",
        unit="same_as:mark",
        category="event_response",
    )

    def _calculate_series(self, mark: pd.DataFrame, half_life: float = 10.0, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        mp = _check_int(min_periods, "min_periods", 1)
        weight = 0.5 ** (1.0 / hl)
        mv = mark.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = 0
            for row in range(rows):
                v = mv[row, col]
                if np.isfinite(v):
                    acc = acc * weight + abs(float(v))
                    seen += 1
                    if seen >= mp:
                        out[row, col] = acc
                else:
                    if seen >= mp:
                        acc = acc * weight
                        out[row, col] = acc
        return _frame_like(mark, out)


@register_operator(
    name="erd_burst_duration",
    category="event_response",
    business_category="event_response",
    canonical="erd_burst_duration",
    source="wave1_event_response",
    status="experimental")
class ErdBurstDuration(SeriesOperator):
    """事件爆发持续时间：当前连续事件（非零）帧的 count。

    衡量事件爆发的持续性。事件 = 有限非零。连续 NaN 中断计数。输出 count。
    """

    metadata = _metadata(
        "erd_burst_duration",
        "当前连续非零事件帧数（爆发持续时间）。",
        ["event"],
        domain="event",
        unit="count",
        category="event_response",
    )

    def _calculate_series(self, event: pd.DataFrame, **_: Any) -> pd.DataFrame:
        ev = event.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            streak = 0
            for row in range(rows):
                v = ev[row, col]
                if not np.isfinite(v):
                    out[row, col] = np.nan
                    streak = 0
                    continue
                if v != 0.0:
                    streak += 1
                else:
                    streak = 0
                out[row, col] = float(streak)
        return _frame_like(event, out)