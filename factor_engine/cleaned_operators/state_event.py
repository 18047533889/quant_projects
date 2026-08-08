# -*- coding: utf-8 -*-
"""State-transition and event-spacing operators.

These quantify how often a state changes and how regularly events recur.
All operators are causal daily-panel transforms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_event",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_event", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _truth(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != 0)


@register_operator(
    name="ts_transition_count",
    category="time_series_event",
    business_category="time_series_event",
    canonical="ts_transition_count",
    source="state_event",
    status="experimental",
)
class TsTransitionCount(SeriesOperator):
    """窗口内状态 0→1（或 1→0）切换次数。

    ``missing_policy="break"``（默认）：NaN 表示缺失，不代表 False；缺失中断
    前后状态连续，跨越缺失的切换不计入。
    """

    metadata = _metadata(
        "ts_transition_count",
        "窗口内状态真值变化次数。",
        ["condition", "window", "missing_policy"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 20, missing_policy: str = "break", **_: Any) -> pd.DataFrame:
        w = int(window)
        cv = condition.to_numpy()
        valid = np.isfinite(cv)
        truth = valid & (cv != 0)
        rows, cols = truth.shape
        policy = str(missing_policy).lower()
        if policy not in ("break", "carry"):
            raise ValueError("missing_policy must be 'break' or 'carry'")
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                if not valid[row, col]:
                    out[row, col] = np.nan
                    continue
                transitions = 0
                prev: bool | None = None
                for i in range(start, row + 1):
                    if not valid[i, col]:
                        if policy == "carry":
                            continue  # missing row is transparent; carry the last state
                        prev = None  # break: missing resets continuity
                        continue
                    current = bool(truth[i, col])
                    if prev is not None and current != prev:
                        transitions += 1
                    prev = current
                out[row, col] = float(transitions)
        return _frame_like(condition, out)


@register_operator(
    name="ts_time_since_change",
    category="time_series_event",
    business_category="time_series_event",
    canonical="ts_time_since_change",
    source="state_event",
    status="experimental",
)
class TsTimeSinceChange(SeriesOperator):
    """距当前状态上一次发生变化以来经过的交易行数。

    ``missing_policy="break"``（默认）：缺失期间输出 NaN，并重置连续状态起点；
    不把缺失区间当作有效状态持续期。
    """

    metadata = _metadata(
        "ts_time_since_change",
        "距当前状态上一次变化经过的行数。",
        ["condition", "max_lookback", "missing_policy"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, max_lookback: Any = None, missing_policy: str = "break", **_: Any) -> pd.DataFrame:
        limit = None if max_lookback is None else int(max_lookback)
        cv = condition.to_numpy()
        valid = np.isfinite(cv)
        truth = valid & (cv != 0)
        rows, cols = truth.shape
        policy = str(missing_policy).lower()
        if policy not in ("break", "carry"):
            raise ValueError("missing_policy must be 'break' or 'carry'")
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_change = -1
            prev: bool | None = None
            for row in range(rows):
                if not valid[row, col]:
                    if policy == "carry":
                        # A missing row is a continuation of the same state: the
                        # distance from the last change keeps growing.
                        if last_change >= 0:
                            distance = row - last_change
                            if limit is None or distance < limit:
                                out[row, col] = float(distance)
                        continue
                    # break (default): missing emits NaN and resets continuity.
                    out[row, col] = np.nan
                    last_change = -1
                    prev = None
                    continue
                current = bool(truth[row, col])
                if prev is not None and current != prev:
                    last_change = row
                prev = current
                if last_change >= 0:
                    distance = row - last_change
                    if limit is None or distance < limit:
                        out[row, col] = float(distance)
        return _frame_like(condition, out)


def _event_positions(truth: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero(truth[start:end])


@register_operator(
    name="ts_event_spacing_mean",
    category="time_series_event",
    business_category="time_series_event",
    canonical="ts_event_spacing_mean",
    source="state_event",
    status="experimental",
)
class TsEventSpacingMean(SeriesOperator):
    """窗口内相邻事件间隔的平均值。"""

    metadata = _metadata(
        "ts_event_spacing_mean",
        "窗口内相邻事件间隔的平均值。",
        ["condition", "window", "min_events"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, min_events: int = 2, **_: Any) -> pd.DataFrame:
        w = int(window)
        min_e = max(2, int(min_events))
        cv = condition.to_numpy()
        truth = _truth(cv)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                positions = _event_positions(truth[:, col], start, row + 1)
                if positions.size < min_e:
                    continue
                gaps = np.diff(positions)
                out[row, col] = float(np.mean(gaps)) if gaps.size else np.nan
        return _frame_like(condition, out)


@register_operator(
    name="ts_event_spacing_cv",
    category="time_series_event",
    business_category="time_series_event",
    canonical="ts_event_spacing_cv",
    source="state_event",
    status="experimental",
)
class TsEventSpacingCv(SeriesOperator):
    """窗口内相邻事件间隔的变异系数（std/mean），数值越高越不规律。"""

    metadata = _metadata(
        "ts_event_spacing_cv",
        "窗口内相邻事件间隔的变异系数。",
        ["condition", "window", "min_events"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, min_events: int = 3, **_: Any) -> pd.DataFrame:
        w = int(window)
        min_e = max(3, int(min_events))
        cv = condition.to_numpy()
        truth = _truth(cv)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                positions = _event_positions(truth[:, col], start, row + 1)
                if positions.size < min_e:
                    continue
                gaps = np.diff(positions)
                if gaps.size and float(np.mean(gaps)) > 0:
                    out[row, col] = float(np.std(gaps) / np.mean(gaps))
        return _frame_like(condition, out)


@register_operator(
    name="event_decay_asof",
    category="time_series_event",
    business_category="time_series_event",
    canonical="event_decay_asof",
    source="state_event",
    status="experimental",
)
class EventDecayAsOf(SeriesOperator):
    """历史事件按半衰期指数衰减的因果累计（仅用截至当日的信号）。"""

    metadata = _metadata(
        "event_decay_asof",
        "sum_{s<=t} event_s * 0.5^((t-s)/half_life)。首次有效观测前输出 NaN；缺失按 missing_policy。",
        ["event", "half_life", "missing_policy"],
        domain="event",
        unit="level",
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        half_life: float = 20.0,
        missing_policy: str = "carry",
        **_: Any,
    ) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        if missing_policy not in {"carry", "break"}:
            raise ValueError("missing_policy must be 'carry' or 'break'")
        weight = 0.5 ** (1.0 / hl)
        arr = event.to_numpy(dtype=float)
        out = np.full(arr.shape, np.nan)
        for col in range(arr.shape[1]):
            series = arr[:, col]
            if len(series) == 0:
                continue
            acc = 0.0
            seen = False
            row = np.empty(len(series), dtype=float)
            for i, value in enumerate(series):
                finite = bool(np.isfinite(value))
                if finite:
                    seen = True
                    acc = acc * weight + value
                elif not seen:
                    # 首次有效观测之前：NaN，而不是 0。
                    row[i] = np.nan
                    continue
                elif missing_policy == "break":
                    # 缺失打断衰减序列：输出 NaN 并重置，直到下一次有效事件重启。
                    row[i] = np.nan
                    seen = False
                    acc = 0.0
                    continue
                else:  # "carry"（默认）：有历史观测时缺失不贡献、仅按半衰期衰减
                    acc = acc * weight
                row[i] = acc
            out[:, col] = row
        return _frame_like(event, out)


# Classify new event/state operators on the extended surface.
import cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.EXTENDED_ONLY_CANONICALS = frozenset(set(_surface.EXTENDED_ONLY_CANONICALS) | set(['ts_transition_count', 'ts_time_since_change', 'ts_event_spacing_mean', 'ts_event_spacing_cv', 'event_decay_asof']))
