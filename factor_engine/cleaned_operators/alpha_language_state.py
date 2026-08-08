# -*- coding: utf-8 -*-
"""Alpha-language temporal state and signed-pattern operators (2026-08).

This is the CTA / run-state primitive family for automated alpha search:

* run-state: how long / how strong / how concentrated a sustained non-zero
  state (e.g. ``sign(return)``) has been running;
* hysteresis: a two-band threshold state machine whose output only flips when
  the driver crosses the *outer* band and only resets when it crosses back to
  the *inner* band (no threshold chatter);
* state-integral: state duration x state strength accumulation;
* transition intensity: magnitude-weighted state-switch frequency;
* signed pattern: sign persistence (lag-1 sign autocorrelation) and sign
  run-length clustering.

All operators are causal sequential (per-column forward) or trailing-window
transforms: the output at row ``t`` depends only on rows ``<= t``.  NaN is a
missing value (never 0); a missing driver breaks a run / resets a state unless
documented otherwise.  Degenerate windows return NaN rather than an invented
number.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
    valid_values,
)

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_state",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_state", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _state_series(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite/zero classification of a state panel.

    Returns ``(valid, value)`` where ``value`` is the raw finite state value
    (any non-zero value is an *active* state; ``0`` is the neutral state; NaN
    is missing).
    """
    valid = np.isfinite(state)
    value = np.where(valid, state, 0.0)
    return valid, value


def _run_windows(values: np.ndarray, state: np.ndarray, window: int) -> np.ndarray:
    """Align ``values`` and ``state`` to the same 2D shape / dtype."""
    if values.shape != state.shape:
        raise ValueError("x and state must share the same panel shape")
    return values.astype(float), state.astype(float)


@register_operator(
    name="ts_run_strength",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_run_strength",
    source="alpha_language_state",
)
class TsRunStrength(SeriesOperator):
    """当前连续相同非零 state 区间内 x 的累计（run strength）。

    R_t = 以 t 结尾的连续同 state 非零段; RS_t = sum_{tau in R_t} x_tau。
    normalize=True 时再除以 trailing window 内 x 的 MAD(+eps), 使不同字段可比。
    state 为 NaN 或 0 时无有效 run -> 0(未归一) / NaN(归一化缺基线)。
    """

    metadata = _metadata(
        "ts_run_strength",
        "当前连续同状态区间内 x 的累计;可选按窗口 MAD 归一。",
        ["x", "state", "max_run", "normalize", "min_periods"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        state: pd.DataFrame,
        max_run: int = 20,
        normalize: bool = False,
        min_periods: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(int(max_run), name="max_run")
        mp = max(1, int(min_periods))
        x, state = _aligned(x, state)
        xv, sv = _run_windows(x.to_numpy(), state.to_numpy(), w)
        s_valid, s_val = _state_series(sv)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            # ``buf`` keeps only the latest ``max_run`` rows of the current run,
            # so ``max_run`` genuinely caps the accumulation (P0-27); the
            # uncapped ``run_len`` still drives the min_periods gate.
            buf: deque[float] = deque(maxlen=w)
            run_len = 0
            prev_state = None
            for row in range(rows):
                finite_x = bool(np.isfinite(xv[row, col]))
                if not s_valid[row, col]:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                cur = s_val[row, col]
                if cur == 0.0:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    out[row, col] = 0.0
                    continue
                if prev_state is not None and cur == prev_state:
                    if finite_x:
                        buf.append(xv[row, col])
                        run_len = run_len + 1
                    else:
                        buf.clear()
                        run_len = 0
                else:
                    buf = deque([xv[row, col]], maxlen=w) if finite_x else deque(maxlen=w)
                    run_len = 1 if finite_x else 0
                prev_state = cur
                if run_len < mp or run_len == 0:
                    out[row, col] = np.nan
                    continue
                run_sum = float(sum(buf))
                if normalize:
                    lo = max(0, row - w + 1)
                    base = valid_values(xv[lo : row + 1, col])
                    if base.size < mp:
                        out[row, col] = np.nan
                        continue
                    mad = float(np.median(np.abs(base - np.median(base))))
                    out[row, col] = run_sum / (mad + _EPS)
                else:
                    out[row, col] = run_sum
        return frame_like(x, out)


@register_operator(
    name="ts_run_efficiency",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_run_efficiency",
    source="alpha_language_state",
)
class TsRunEfficiency(SeriesOperator):
    """当前 run 的路径效率: |sum x| / (sum |x| + eps), 范围 [0,1]。

    1 = 路径单向(每个 x 同号); 0 = 状态虽持续但内部剧烈折返。
    """

    metadata = _metadata(
        "ts_run_efficiency",
        "当前 run 内 |sum x| / (sum |x| + eps)。",
        ["x", "state", "max_run", "min_periods"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        state: pd.DataFrame,
        max_run: int = 20,
        min_periods: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(int(max_run), name="max_run")
        mp = max(2, int(min_periods))
        x, state = _aligned(x, state)
        xv, sv = _run_windows(x.to_numpy(), state.to_numpy(), w)
        s_valid, s_val = _state_series(sv)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            # Capped to the latest ``max_run`` rows of the run (P0-28): both the
            # signed and absolute sums are computed from the same deque.
            buf: deque[float] = deque(maxlen=w)
            run_len = 0
            prev_state = None
            for row in range(rows):
                if not s_valid[row, col]:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                if not np.isfinite(xv[row, col]):
                    # P0-003: an invalid driver breaks the episode (never a
                    # carry); the NaN row emits NaN and the run restarts.
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                cur = s_val[row, col]
                if cur == 0.0:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    out[row, col] = 0.0
                    continue
                if prev_state is not None and cur == prev_state:
                    buf.append(xv[row, col])
                    run_len = run_len + 1
                else:
                    buf = deque([xv[row, col]], maxlen=w)
                    run_len = 1
                prev_state = cur
                if run_len < mp:
                    out[row, col] = np.nan
                    continue
                run_sum = float(sum(buf))
                run_abs = float(sum(abs(v) for v in buf))
                out[row, col] = abs(run_sum) / (run_abs + _EPS)
        return frame_like(x, out)


@register_operator(
    name="ts_run_concentration",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_run_concentration",
    source="alpha_language_state",
)
class TsRunConcentration(SeriesOperator):
    """当前 run 的集中度: max|x| / (sum |x| + eps), 范围 [0,1]。

    接近 1 = 趋势主要由某一天贡献; 接近 0 = 趋势由很多天稳定累积。
    """

    metadata = _metadata(
        "ts_run_concentration",
        "当前 run 内 max|x| / (sum |x| + eps)。",
        ["x", "state", "max_run", "min_periods"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        state: pd.DataFrame,
        max_run: int = 20,
        min_periods: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(int(max_run), name="max_run")
        mp = max(2, int(min_periods))
        x, state = _aligned(x, state)
        xv, sv = _run_windows(x.to_numpy(), state.to_numpy(), w)
        s_valid, s_val = _state_series(sv)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            # Capped to the latest ``max_run`` rows of the run (P0-28).
            buf: deque[float] = deque(maxlen=w)
            run_len = 0
            prev_state = None
            for row in range(rows):
                if not s_valid[row, col]:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                if not np.isfinite(xv[row, col]):
                    # P0-004: an invalid driver breaks the episode.
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                cur = s_val[row, col]
                if cur == 0.0:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    out[row, col] = 0.0
                    continue
                if prev_state is not None and cur == prev_state:
                    buf.append(xv[row, col])
                    run_len = run_len + 1
                else:
                    buf = deque([xv[row, col]], maxlen=w)
                    run_len = 1
                prev_state = cur
                if run_len < mp:
                    out[row, col] = np.nan
                    continue
                run_abs = float(sum(abs(v) for v in buf))
                run_max = float(max(abs(v) for v in buf))
                out[row, col] = run_max / (run_abs + _EPS)
        return frame_like(x, out)


@register_operator(
    name="ts_hysteresis_state",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_hysteresis_state",
    source="alpha_language_state",
)
class TsHysteresisState(SeriesOperator):
    """双阈值滞后状态机: z > +upper -> +1; z < -upper -> -1; 已 +1 且 z < lower -> 0; 已 -1 且 z > -lower -> 0。

    要求 0 <= lower < upper。输出 {-1,0,+1}。z 缺失时输出 NaN、内部状态保持(carry)。
    """

    metadata = _metadata(
        "ts_hysteresis_state",
        "双阈值滞后状态机输出 {-1,0,+1}。",
        ["z", "upper", "lower"],
        domain="trading_state",
        unit="state",
    )

    def _calculate_series(
        self, z: pd.DataFrame, upper: float = 1.0, lower: float = 0.0, **_: Any
    ) -> pd.DataFrame:
        hi = float(upper)
        lo = float(lower)
        if not (0.0 <= lo < hi):
            raise ValueError("require 0 <= lower < upper")
        zv = z.to_numpy(dtype=float)
        rows, cols = zv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            cur_state = 0
            for row in range(rows):
                val = zv[row, col]
                if not np.isfinite(val):
                    out[row, col] = np.nan
                    continue
                if val > hi:
                    cur_state = 1
                elif val < -hi:
                    cur_state = -1
                elif cur_state == 1 and val < lo:
                    cur_state = 0
                elif cur_state == -1 and val > -lo:
                    cur_state = 0
                out[row, col] = float(cur_state)
        return frame_like(z, out)


def _hysteresis_states(zv: np.ndarray, hi: float, lo: float, cols: int) -> np.ndarray:
    states = np.full(zv.shape, np.nan, dtype=float)
    for col in range(cols):
        cur_state = 0
        for row in range(zv.shape[0]):
            val = zv[row, col]
            if not np.isfinite(val):
                states[row, col] = np.nan
                continue
            if val > hi:
                cur_state = 1
            elif val < -hi:
                cur_state = -1
            elif cur_state == 1 and val < lo:
                cur_state = 0
            elif cur_state == -1 and val > -lo:
                cur_state = 0
            states[row, col] = float(cur_state)
    return states


@register_operator(
    name="ts_hysteresis_age",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_hysteresis_age",
    source="alpha_language_state",
)
class TsHysteresisAge(SeriesOperator):
    """滞后状态机, 按当前状态的持续年龄归一: state_t * min(age_t, cap) / cap, 范围 [-1,1]。

    同时编码方向与持续时间; 中性状态 -> 0。
    """

    metadata = _metadata(
        "ts_hysteresis_age",
        "状态方向 * 状态持续年龄/cap。",
        ["z", "upper", "lower", "cap"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(
        self, z: pd.DataFrame, upper: float = 1.0, lower: float = 0.0, cap: int = 60, **_: Any
    ) -> pd.DataFrame:
        hi = float(upper)
        lo = float(lower)
        if not (0.0 <= lo < hi):
            raise ValueError("require 0 <= lower < upper")
        cap_n = int(cap)
        if cap_n < 1:
            raise ValueError("cap must be >= 1")
        zv = z.to_numpy(dtype=float)
        states = _hysteresis_states(zv, hi, lo, zv.shape[1])
        rows, cols = zv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            age = 0
            for row in range(rows):
                st = states[row, col]
                if not np.isfinite(st):
                    age = 0
                    continue
                if st == 0.0:
                    age = 0
                    out[row, col] = 0.0
                    continue
                prev_st = states[row - 1, col] if row > 0 else np.nan
                if np.isfinite(prev_st) and prev_st == st:
                    age = age + 1
                else:
                    age = 1
                out[row, col] = st * min(float(age), float(cap_n)) / float(cap_n)
        return frame_like(z, out)


@register_operator(
    name="ts_state_integral",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_state_integral",
    source="alpha_language_state",
)
class TsStateIntegral(SeriesOperator):
    """当前非零滞后状态段内的强度累计: state_t * sum max(|z|-lower, 0)。

    half_life 给定则按 e^{-(t-tau)/lambda} 衰减。表达「状态持续时间 x 状态强度」。
    """

    metadata = _metadata(
        "ts_state_integral",
        "当前状态段内 max(|z|-lower,0) 累计(可选指数衰减)。",
        ["z", "upper", "lower", "max_run", "half_life"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(
        self,
        z: pd.DataFrame,
        upper: float = 1.0,
        lower: float = 0.0,
        max_run: int = 120,
        half_life: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        hi = float(upper)
        lo = float(lower)
        if not (0.0 <= lo < hi):
            raise ValueError("require 0 <= lower < upper")
        w_run = check_window(int(max_run), name="max_run")
        lamb: float | None = None
        if half_life is not None and half_life != "":
            lamb = float(half_life)
            if lamb <= 0.0:
                raise ValueError("half_life must be > 0")
        zv = z.to_numpy(dtype=float)
        states = _hysteresis_states(zv, hi, lo, zv.shape[1])
        rows, cols = zv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            # ``buf`` keeps only the latest ``max_run`` contributions of the
            # current state segment, so ``max_run`` genuinely caps the integral
            # (P0-29).  With ``half_life`` each retained contribution is decayed
            # by its own age from the current row.
            buf: deque[float] = deque(maxlen=w_run)
            cur_state = 0.0
            for row in range(rows):
                st = states[row, col]
                if not np.isfinite(st):
                    buf.clear()
                    cur_state = 0.0
                    continue
                if st == 0.0:
                    buf.clear()
                    cur_state = 0.0
                    out[row, col] = 0.0
                    continue
                if row > 0 and states[row - 1, col] == st:
                    buf.append(max(abs(zv[row, col]) - lo, 0.0))
                else:
                    cur_state = st
                    buf = deque([max(abs(zv[row, col]) - lo, 0.0)], maxlen=w_run)
                if lamb is not None:
                    s = 0.0
                    for age, val in enumerate(reversed(buf)):
                        s += float(val) * float(np.exp(-(age + 1) / lamb))
                else:
                    s = float(sum(buf))
                out[row, col] = st * s
        return frame_like(z, out)


@register_operator(
    name="ts_state_entry_strength",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_state_entry_strength",
    source="alpha_language_state",
)
class TsStateEntryStrength(SeriesOperator):
    """进入状态瞬间的强度, 状态持续期间 carry 最近一次 entry strength。

    Entry_t = state_t * max(|z_t| - upper, 0) 于状态建立时刻; 中性状态 -> 0。
    用于区分「勉强突破进入状态」与「强力突破进入状态」。
    """

    metadata = _metadata(
        "ts_state_entry_strength",
        "状态建立瞬间的突破强度(持续期间 carry)。",
        ["z", "upper", "lower"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(
        self, z: pd.DataFrame, upper: float = 1.0, lower: float = 0.0, **_: Any
    ) -> pd.DataFrame:
        hi = float(upper)
        lo = float(lower)
        if not (0.0 <= lo < hi):
            raise ValueError("require 0 <= lower < upper")
        zv = z.to_numpy(dtype=float)
        states = _hysteresis_states(zv, hi, lo, zv.shape[1])
        rows, cols = zv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            carried = 0.0
            cur_state = 0.0
            for row in range(rows):
                st = states[row, col]
                if not np.isfinite(st):
                    carried = 0.0
                    cur_state = 0.0
                    continue
                if st == 0.0:
                    carried = 0.0
                    cur_state = 0.0
                    out[row, col] = 0.0
                    continue
                if row == 0 or not (np.isfinite(states[row - 1, col]) and states[row - 1, col] == st):
                    # 新状态建立: entry strength = state * max(|z|-upper, 0)
                    carried = st * max(abs(zv[row, col]) - hi, 0.0)
                    cur_state = st
                out[row, col] = carried
        return frame_like(z, out)


@register_operator(
    name="ts_transition_intensity",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_transition_intensity",
    source="alpha_language_state",
)
class TsTransitionIntensity(SeriesOperator):
    """状态切换的幅度加权强度: sum(|dx| * I_tau) / (sum I_tau + eps)。

    I_tau = 1(state_tau != state_{tau-1}); normalize=True 时再除以窗口内 dx 的 MAD。
    窗口内无切换 -> NaN。
    """

    metadata = _metadata(
        "ts_transition_intensity",
        "状态切换处 |dx| 的均值, 可选按 dx MAD 归一。",
        ["x", "state", "window", "normalize"],
        domain="trading_state",
        unit="level",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        state: pd.DataFrame,
        window: int = 20,
        normalize: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(window)
        x, state = _aligned(x, state)
        xv = x.to_numpy(dtype=float)
        sv = state.to_numpy(dtype=float)
        s_valid, s_val = _state_series(sv)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(1, row - w + 1)
                n = 0
                wsum = 0.0
                dx_vals = []
                for i in range(start, row + 1):
                    if not (np.isfinite(xv[i, col]) and np.isfinite(xv[i - 1, col])):
                        continue
                    if not (s_valid[i, col] and s_valid[i - 1, col]):
                        continue
                    if s_val[i, col] != s_val[i - 1, col]:
                        d = xv[i, col] - xv[i - 1, col]
                        wsum = wsum + abs(d)
                        dx_vals.append(d)
                        n = n + 1
                if n == 0:
                    continue
                if normalize:
                    dx_arr = np.asarray(dx_vals, dtype=float)
                    mad = float(np.median(np.abs(dx_arr - np.median(dx_arr))))
                    if not np.isfinite(mad) or mad < _EPS:
                        continue  # 常数 dx -> 归一化分母退化, 不产出虚假巨大值
                    out[row, col] = (wsum / float(n)) / mad
                else:
                    out[row, col] = wsum / float(n)
        return frame_like(x, out)


@register_operator(
    name="ts_sign_persistence",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_sign_persistence",
    source="alpha_language_state",
)
class TsSignPersistence(SeriesOperator):
    """符号滞后 1 阶自相关: corr(sign(x_tau), sign(x_{tau-1}))。

    正 = 趋势持续(符号自维持); 负 = 交替。至少 3 对同位置有效样本, 否则 NaN。
    """

    metadata = _metadata(
        "ts_sign_persistence",
        "sign(x) 滞后 1 自相关。",
        ["x", "window", "min_periods"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            sign = np.where(np.isfinite(chunk) & (chunk > 0), 1.0,
                            np.where(np.isfinite(chunk) & (chunk < 0), -1.0, np.nan))
            s0 = sign[:-1]
            s1 = sign[1:]
            finite = np.isfinite(s0) & np.isfinite(s1)
            if int(finite.sum()) < mp:
                return np.nan
            a = s0[finite]
            b = s1[finite]
            if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
                return np.nan
            return float(np.corrcoef(a, b)[0, 1])

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_sign_cluster_index",
    category="time_series_state",
    business_category="time_series_state",
    canonical="ts_sign_cluster_index",
    source="alpha_language_state",
)
class TsSignClusterIndex(SeriesOperator):
    """窗口内符号 run 长度的集中度(归一化 HHI): 高 = 长 run/少切换, 低 = 高频交替。

    MC = (HHI - 1/m) / (1 - 1/m), 需要 >= 2 个 run, 否则 NaN。
    """

    metadata = _metadata(
        "ts_sign_cluster_index",
        "窗口内符号 run 长度归一化 HHI。",
        ["x", "window", "min_periods"],
        domain="trading_state",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            sign = np.where(valid > 0, 1.0, np.where(valid < 0, -1.0, 0.0))
            sign = sign[sign != 0.0]
            if sign.size < mp:
                return np.nan
            runs = []
            cur = sign[0]
            length = 1
            for v in sign[1:]:
                if v == cur:
                    length = length + 1
                else:
                    runs.append(length)
                    cur = v
                    length = 1
            runs.append(length)
            m = len(runs)
            if m < 2:
                return np.nan
            total = float(sum(runs))
            if total <= 0.0:
                return np.nan
            hhi = sum((r / total) ** 2 for r in runs)
            return float((hhi - 1.0 / m) / (1.0 - 1.0 / m))

        return frame_like(x, map_rolling(xv, w, _fn))


_NEW_CANONICALS = (
    "ts_run_strength",
    "ts_run_efficiency",
    "ts_run_concentration",
    "ts_hysteresis_state",
    "ts_hysteresis_age",
    "ts_state_integral",
    "ts_state_entry_strength",
    "ts_transition_intensity",
    "ts_sign_persistence",
    "ts_sign_cluster_index",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
