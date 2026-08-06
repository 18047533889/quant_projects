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
    """窗口内状态 0→1（或 1→0）切换次数。"""

    metadata = _metadata(
        "ts_transition_count",
        "窗口内状态真值变化次数。",
        ["condition", "window"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        cv = condition.to_numpy()
        truth = _truth(cv)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev = None
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = truth[start : row + 1, col]
                if np.all(np.isnan(cv[start : row + 1, col])):
                    out[row, col] = np.nan
                    continue
                transitions = int(np.sum(segment[1:] != segment[:-1]))
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
    """距当前状态上一次发生变化以来经过的交易行数。"""

    metadata = _metadata(
        "ts_time_since_change",
        "距当前状态上一次变化经过的行数。",
        ["condition", "max_lookback"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, max_lookback: Any = None, **_: Any) -> pd.DataFrame:
        limit = None if max_lookback is None else int(max_lookback)
        cv = condition.to_numpy()
        truth = _truth(cv)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_change = -1
            prev = None
            for row in range(rows):
                if np.isfinite(cv[row, col]):
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
        "sum_{s<=t} event_s * 0.5^((t-s)/half_life)。",
        ["event", "half_life"],
        domain="event",
        unit="level",
    )

    def _calculate_series(self, event: pd.DataFrame, half_life: float = 20.0, **_: Any) -> pd.DataFrame:
        hl = max(float(half_life), 1.0)
        weight = 0.5 ** (1.0 / hl)
        arr = np.where(_truth(event.to_numpy(dtype=float)), 1.0, 0.0)
        out = np.full(arr.shape, np.nan)
        for col in range(arr.shape[1]):
            series = arr[:, col]
            if len(series) == 0:
                continue
            acc = 0.0
            row = np.empty(len(series), dtype=float)
            for i, value in enumerate(series):
                acc = acc * weight + value
                row[i] = acc
            out[:, col] = row
        return _frame_like(event, out)


# Classify new event/state operators on the extended surface.
import cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.EXTENDED_ONLY_CANONICALS = frozenset(set(_surface.EXTENDED_ONLY_CANONICALS) | set(['ts_transition_count', 'ts_time_since_change', 'ts_event_spacing_mean', 'ts_event_spacing_cv', 'event_decay_asof']))
