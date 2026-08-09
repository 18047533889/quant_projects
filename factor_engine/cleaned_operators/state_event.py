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
from cleaned_operators.stateful._common import assert_condition_bool


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


def _event_positions(truth: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero(truth[start:end])


def _gaps_censored(valid_col: np.ndarray, positions: np.ndarray, start: int) -> np.ndarray | None:
    """Inter-event gaps for events in ``[start, start+positions.size)``.

    ``positions`` are indices relative to ``start`` (as produced by
    ``_event_positions``).  #145: an interval whose open range ``(a, b)`` crosses
    an UNKNOWN (NaN) row is *not* a precise interval — the distance is undefined,
    so the whole window's spacing statistic is censored (``None`` → NaN) rather
    than silently connecting the two events across the unknown region.  Returns
    ``None`` when fewer than two events are present.
    """
    if positions.size < 2:
        return None
    gaps = np.diff(positions).astype(float)
    for idx in range(positions.size - 1):
        a = start + int(positions[idx])
        b = start + int(positions[idx + 1])
        if np.any(~valid_col[a + 1 : b]):
            return None  # interval crosses an unknown row -> censored
    return gaps


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
        assert_condition_bool(condition, name="condition")
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

    ``initial_semantics`` splits two distinct definitions (#146):
      * ``"since_transition"``（默认）— 距最近一次*观测到的*状态切换的行数；在
        首次真切换之前输出 NaN（时钟未启动）。
      * ``"state_age"`` — 当前观测状态自身的年龄：从首个有效状态观测起算时钟
        （首个有效行输出 0），切换时归零。
    ``missing_policy="break"``（默认）：缺失期间输出 NaN，并重置连续状态起点；
    不把缺失区间当作有效状态持续期。
    ``max_lookback`` 为上界（不含边界，#147）：仅当 ``distance < max_lookback``
    才输出；距离恰好等于 ``max_lookback`` 时输出 NaN。
    """

    metadata = _metadata(
        "ts_time_since_change",
        "距当前状态上一次变化经过的行数(since_transition 或 state_age)。",
        ["condition", "max_lookback", "missing_policy", "initial_semantics"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(
        self,
        condition: pd.DataFrame,
        max_lookback: Any = None,
        missing_policy: str = "break",
        initial_semantics: str = "since_transition",
        **_: Any,
    ) -> pd.DataFrame:
        assert_condition_bool(condition, name="condition")
        limit = None if max_lookback is None else int(max_lookback)
        cv = condition.to_numpy()
        valid = np.isfinite(cv)
        truth = valid & (cv != 0)
        rows, cols = truth.shape
        policy = str(missing_policy).lower()
        if policy not in ("break", "carry"):
            raise ValueError("missing_policy must be 'break' or 'carry'")
        sem = str(initial_semantics).lower()
        if sem not in ("since_transition", "state_age"):
            raise ValueError("initial_semantics must be 'since_transition' or 'state_age'")
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_change = -1
            prev: bool | None = None
            started = False
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
                    started = False
                    continue
                current = bool(truth[row, col])
                if prev is not None and current != prev:
                    last_change = row
                prev = current
                if not started:
                    started = True
                    # #146: ``state_age`` starts the clock at the first VALID
                    # state; ``since_transition`` waits for the first transition
                    # (last_change stays -1 -> NaN until one occurs).
                    if sem == "state_age" and last_change < 0:
                        last_change = row
                if last_change >= 0:
                    distance = row - last_change
                    if limit is None or distance < limit:
                        out[row, col] = float(distance)
        return _frame_like(condition, out)


@register_operator(
    name="ts_event_spacing_mean",
    category="time_series_event",
    business_category="time_series_event",
    canonical="ts_event_spacing_mean",
    source="state_event",
    status="experimental",
)
class TsEventSpacingMean(SeriesOperator):
    """窗口内相邻事件间隔的平均值。

    EventBool 语义（#145）：``0`` = 确认无事件，非零 = 事件，NaN = unknown。
    间隔跨越 unknown 行时被 censored —— 该窗口输出 NaN，而不是把两个事件
    跨过 unknown 区段精确相连。
    """

    metadata = _metadata(
        "ts_event_spacing_mean",
        "窗口内相邻事件间隔的平均值(跨越 unknown 的间隔被 censored)。",
        ["condition", "window", "min_events"],
        domain="trading_state",
        unit="count",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, min_events: int = 2, **_: Any) -> pd.DataFrame:
        assert_condition_bool(condition, name="condition")
        w = int(window)
        min_e = max(2, int(min_events))
        cv = condition.to_numpy()
        valid = np.isfinite(cv)
        truth = valid & (cv != 0)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            valid_col = valid[:, col]
            truth_col = truth[:, col]
            for row in range(rows):
                start = max(0, row - w + 1)
                positions = _event_positions(truth_col, start, row + 1)
                if positions.size < min_e:
                    continue
                gaps = _gaps_censored(valid_col, positions, start)
                if gaps is None:
                    continue  # censored: an interval crossed an unknown row
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
    """窗口内相邻事件间隔的变异系数（std/mean），数值越高越不规律。

    EventBool 语义（#145）：跨越 unknown（NaN）行的间隔被 censored —— 该窗口
    输出 NaN，而不是跨过 unknown 区段把两个事件精确相连。
    """

    metadata = _metadata(
        "ts_event_spacing_cv",
        "窗口内相邻事件间隔的变异系数(跨越 unknown 的间隔被 censored)。",
        ["condition", "window", "min_events"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(self, condition: pd.DataFrame, window: int = 60, min_events: int = 3, **_: Any) -> pd.DataFrame:
        assert_condition_bool(condition, name="condition")
        w = int(window)
        min_e = max(3, int(min_events))
        cv = condition.to_numpy()
        valid = np.isfinite(cv)
        truth = valid & (cv != 0)
        rows, cols = truth.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            valid_col = valid[:, col]
            truth_col = truth[:, col]
            for row in range(rows):
                start = max(0, row - w + 1)
                positions = _event_positions(truth_col, start, row + 1)
                if positions.size < min_e:
                    continue
                gaps = _gaps_censored(valid_col, positions, start)
                if gaps is None:
                    continue  # censored: an interval crossed an unknown row
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
    """历史事件按半衰期指数衰减的因果累计（仅用截至当日的信号）。

    ``event_kind`` 区分两类输入（#148）：
      * ``"marked"``（默认）— MarkedEvent：每次事件贡献其*幅度*（含符号），输出
        单位继承 mark（例如收益 bp 或价格水平）。
      * ``"bool"`` — EventBool：每个有限非零观测视为一次事件，贡献恰好为 1，
        输出为衰减事件计数。
    首次有效观测前输出 NaN；缺失按 missing_policy。
    """

    metadata = _metadata(
        "event_decay_asof",
        "sum_{s<=t} event_s * 0.5^((t-s)/half_life)。首次有效观测前输出 NaN；缺失按 missing_policy；event_kind=bool 时每次事件贡献 1。",
        ["event", "half_life", "missing_policy", "event_kind"],
        domain="event",
        unit="level",
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        half_life: float = 20.0,
        missing_policy: str = "carry",
        event_kind: str = "marked",
        **_: Any,
    ) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        if missing_policy not in {"carry", "break"}:
            raise ValueError("missing_policy must be 'carry' or 'break'")
        kind = str(event_kind).lower()
        if kind not in {"bool", "marked"}:
            raise ValueError("event_kind must be 'bool' or 'marked'")
        if kind == "bool":
            # #148: an EventBool input is a ConditionBool — a finite value outside
            # {0, 1} is neither "no event" nor "one event" and must not be read as
            # a truthy count (fail closed).  MarkedEvent keeps its magnitude.
            assert_condition_bool(event, name="event")
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
                    # #148: EventBool counts a finite non-zero occurrence as
                    # exactly 1 and a confirmed ``0`` as 0 (no event);
                    # MarkedEvent keeps the magnitude (output unit inherits mark).
                    if kind == "bool":
                        contribution = 1.0 if value != 0.0 else 0.0
                    else:
                        contribution = float(value)
                    acc = acc * weight + contribution
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
_surface.extend_extended_only(set(['ts_transition_count', 'ts_time_since_change', 'ts_event_spacing_mean', 'ts_event_spacing_cv', 'event_decay_asof']))

# Round-11 #12: the genuinely RECURSIVE operators in this module declare their
# own execution contract (stateful; full-history replay — no segmented checkpoint
# restore exists for these kernels, so required_full_history is the honest
# chunking).
from runtime.execution_contract import declare_stateful  # noqa: E402

declare_stateful("ts_time_since_change", state_model="recursive", chunking="required_full_history")
declare_stateful("event_decay_asof", state_model="recursive", chunking="required_full_history")
