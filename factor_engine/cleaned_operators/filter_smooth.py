# -*- coding: utf-8 -*-
"""Filter Layer: Robust EMA (稳健指数移动平均，innovation层截断)。

算子: ts_robust_ema
数学契约（参考文档 §5）:
    alpha = 2 / (span + 1)

    # 第一步：计算 innovation
    e_t = x_t - y_{t-1}

    # 第二步：用历史 innovation 的 MAD 估计 scale
    innovations_hist = [e_{t-warmup_window}, ..., e_{t-1}]
    s_t = 1.4826 * median(|innovations_hist - median(innovations_hist)|)
    s_t = max(s_t, scale_floor)

    # 第三步：clip innovation
    e_t* = clip(e_t, -clip_sigma * s_t, clip_sigma * s_t)

    # 第四步：更新递归 state
    y_t = y_{t-1} + alpha * e_t*

初始化: y_0 = x_0

状态契约:
- stateful=True: 递归依赖历史状态 y_{t-1}
- checkpointable=True: 状态可序列化（y_{t-1} + 历史 innovations）
- time_shard_safe=False: 需要 checkpoint 恢复
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.filter_contracts import FilterContract, FilterRole, JumpPreservationPolicy

_CANONICALS: list[str] = []


@register_operator(
    name="ts_robust_ema",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_robust_ema",
    source="filter_smooth",
)
class RobustEMAOperator(SeriesOperator):
    """稳健 EMA: 在 innovation 层做截断以抑制异常值冲击。

    Parameters:
        x: 输入序列
        span: EMA 窗口（alpha = 2/(span+1)）
        clip_sigma: innovation 截断倍数（单位: scale 的倍数）
        warmup_window: 用于估计 innovation scale 的历史窗口
        scale_floor: scale 估计的最小值（防止除零）

    Returns:
        稳健 EMA 序列
    """

    metadata = OperatorMetadata(
        name="ts_robust_ema",
        category="signal_filter",
        description="Robust exponential moving average with innovation clipping",
        param_names=["x", "span", "clip_sigma", "warmup_window", "scale_floor"],
        return_type="series",
        tags=["signal_filter", "causal", "stateful"],
        param_specs={
            "span": ParamSpec(
                dtype=int,
                min=1,
                default=20,
                param_role=ParamRole.HORIZON,
                searchable=True,
                history_semantics="max_rows",
            ),
            "clip_sigma": ParamSpec(
                dtype=float,
                min=0.0,
                default=3.0,
                param_role=ParamRole.STATE_THRESHOLD,
                searchable=True,
            ),
            "warmup_window": ParamSpec(
                dtype=int,
                min=1,
                default=20,
                param_role=ParamRole.SUPPORT_POLICY,
                searchable=False,
            ),
            "scale_floor": ParamSpec(
                dtype=float,
                min=0.0,
                default=1e-10,
                param_role=ParamRole.NUMERICAL,
                searchable=False,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        span: int = 20,
        clip_sigma: float = 3.0,
        warmup_window: int = 20,
        scale_floor: float = 1e-10,
        **_: Any,
    ) -> pd.DataFrame:
        """执行稳健 EMA 滤波。"""
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        if clip_sigma <= 0:
            raise ValueError(f"clip_sigma must be positive, got {clip_sigma}")
        if warmup_window <= 0:
            raise ValueError(f"warmup_window must be positive, got {warmup_window}")
        if scale_floor <= 0:
            raise ValueError(f"scale_floor must be positive, got {scale_floor}")

        alpha = np.where((span + 1) != 0, 2.0 / (span + 1), np.nan)

        # Process each column independently
        result = x.copy()
        for col in x.columns:
            series = x[col]
            col_result = np.full(len(series), np.nan, dtype=np.float64)

            # 状态: 上一期的 EMA 值 + 历史 innovations 缓冲区
            y_prev = np.nan
            innovations_buffer = deque(maxlen=warmup_window)

            for i in range(len(series)):
                x_i = series.iloc[i]

                if not np.isfinite(x_i):
                    col_result[i] = y_prev  # 缺失观测：保持前值
                    continue

                # 初始化: y_0 = x_0
                if not np.isfinite(y_prev):
                    y_prev = x_i
                    col_result[i] = x_i
                    continue

                # 第一步: 计算 innovation
                e_t = x_i - y_prev

                # 第二步: 用历史 innovations 的 MAD 估计 scale
                # FL-P0-002: warmup期使用ordinary EMA（不clip），避免scale_floor冻结
                if len(innovations_buffer) >= warmup_window:
                    # 有足够历史，可以估计scale并clip
                    innovations_array = np.array(innovations_buffer)
                    innovations_median = np.median(innovations_array)
                    mad = np.median(np.abs(innovations_array - innovations_median))
                    s_t = max(1.4826 * mad, scale_floor)
                    # 第三步: clip innovation
                    e_t_clipped = np.clip(e_t, -clip_sigma * s_t, clip_sigma * s_t)
                else:
                    # warmup期间：使用ordinary EMA（不clip innovation）
                    e_t_clipped = e_t

                # 第四步: 更新递归 state
                y_t = y_prev + alpha * e_t_clipped
                col_result[i] = y_t

                # 更新状态
                y_prev = y_t
                innovations_buffer.append(e_t)

            result[col] = col_result

        return result


_CANONICALS.append("ts_robust_ema")


@register_operator(
    name="ts_super_smoother",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_super_smoother",
    source="filter_smooth",
)
class TSSuperSmoother(SeriesOperator):
    """Two-pole IIR low-pass filter with stronger attenuation than EMA.

    Mathematical formulation (Butterworth-like coefficients):
        a = exp(-sqrt(2) * pi / period)
        b = 2 * a * cos(sqrt(2) * pi / period)

        c2 = b
        c3 = -a²
        c1 = 1 - c2 - c3  # Ensures DC gain = 1

        y_t = c1 * x_t + c2 * y_{t-1} + c3 * y_{t-2}

    The filter requires TWO historical states (y_{t-1}, y_{t-2}), initialized as:
        y_0 = x_0 (first finite observation)
        y_1 = x_1 (second finite observation)

    Certified parameters:
        period: [5, 10, 20, 40] (HORIZON role, searchable)

    Execution contract:
        - Stateful: recursive filter depends on full finite prefix
        - Checkpointable: state is (y_{t-1}, y_{t-2}) — serializable
        - Time shard UNSAFE: requires checkpoint restore across shards
        - Min periods: period (warm-up for stable filter response)

    Parameters:
        x: Input signal series
        period: Filter period (larger = more smoothing), default 10

    Returns:
        Smoothed signal with two-pole attenuation
    """

    metadata = OperatorMetadata(
        name="ts_super_smoother",
        category="signal_filter",
        description="Two-pole IIR low-pass filter with stronger attenuation than EMA",
        param_names=["x", "period"],
        return_type="series",
        tags=["signal_filter", "causal", "stateful"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        period: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        if period < 3:
            raise ValueError(
                f"ts_super_smoother requires period >= 3 for stable two-pole filter, got {period}"
            )

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        sqrt2_pi = np.sqrt(2.0) * np.pi
        a = np.exp(-sqrt2_pi / period)
        b = np.where(period) != 0, 2.0 * a * np.cos(sqrt2_pi / period), np.nan)

        c2 = b
        c3 = -(a * a)
        c1 = 1.0 - c2 - c3  # Ensures DC gain = 1

        for col in range(cols):
            finite_indices = np.where(np.isfinite(xv[:, col]))[0]

            if len(finite_indices) == 0:
                continue

            if len(finite_indices) == 1:
                idx = finite_indices[0]
                out[idx, col] = xv[idx, col]
                continue

            idx0 = finite_indices[0]
            idx1 = finite_indices[1]

            y_t_minus_2 = xv[idx0, col]
            y_t_minus_1 = xv[idx1, col]

            out[idx0, col] = y_t_minus_2
            out[idx1, col] = y_t_minus_1

            for row in range(idx1 + 1, rows):
                curr = xv[row, col]

                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    continue

                y_t = c1 * curr + c2 * y_t_minus_1 + c3 * y_t_minus_2
                out[row, col] = y_t

                y_t_minus_2 = y_t_minus_1
                y_t_minus_1 = y_t

        from cleaned_operators.rolling_pack import frame_like
        return frame_like(x, out)


_CANONICALS.append("ts_super_smoother")


def ts_kama(
    x: pd.DataFrame,
    er_window: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Kaufman Adaptive Moving Average.

    An adaptive filter that automatically adjusts smoothing based on market
    efficiency: follows price closely during trending markets (high ER),
    smooths heavily during choppy/sideways markets (low ER).

    Parameters
    ----------
    x : pd.DataFrame
        Input signal (price or any time series).
    er_window : int, default=10
        Window for computing efficiency ratio. Certified: [10, 20, 40].
    fast_period : int, default=2
        Fast EMA period for trending markets. Certified: [2, 3].
    slow_period : int, default=30
        Slow EMA period for sideways markets. Certified: [20, 30, 40, 60].
    min_periods : int, optional
        Minimum observations required. Defaults to er_window.

    Returns
    -------
    pd.DataFrame
        Adaptive moving average. NaN during warmup or when input is NaN.

    Notes
    -----
    Mathematical definition (§6):
        ER_t = |x_t - x_{t-er_window}| / (sum_{j=1}^{er_window} |x_{t-j+1} - x_{t-j}| + epsilon)
        alpha_fast = 2 / (fast_period + 1)
        alpha_slow = 2 / (slow_period + 1)
        SC_t = (ER_t * (alpha_fast - alpha_slow) + alpha_slow)^2
        y_t = y_{t-1} + SC_t * (x_t - y_{t-1})

    Filter properties:
    - Role: adaptive_low_pass
    - Causal: strictly causal (ER computed from historical data only)
    - Stateful: recursive (depends on y_{t-1})
    - Checkpointable: single float per instrument
    - Warmup: er_window + 1 contiguous bars
    - Lag: variable (adapts to market regime)

    Missing data policy: A NaN input invalidates the recursive state. KAMA
    emits NaN until er_window + 1 consecutive finite prices re-accumulate.
    """
    if isinstance(er_window, bool):
        raise ValueError("er_window must be integer")
    if isinstance(er_window, float):
        raise ValueError("er_window must be integer")
    er_window = int(er_window)
    if er_window < 2:
        raise ValueError("er_window must be >= 2")

    if isinstance(fast_period, bool):
        raise ValueError("fast_period must be integer")
    if isinstance(fast_period, float):
        raise ValueError("fast_period must be integer")
    fast_period = int(fast_period)
    if fast_period < 1:
        raise ValueError("fast_period must be >= 1")

    if isinstance(slow_period, bool):
        raise ValueError("slow_period must be integer")
    if isinstance(slow_period, float):
        raise ValueError("slow_period must be integer")
    slow_period = int(slow_period)
    if slow_period < 1:
        raise ValueError("slow_period must be >= 1")

    if fast_period >= slow_period:
        raise ValueError("fast_period must be < slow_period")

    # FL-P0-003: min_periods parameter now has real runtime semantics
    # It controls the minimum contiguous observations required before KAMA starts emitting
    if min_periods is None:
        min_periods = er_window + 1  # Default: ER computation requirement
    else:
        if isinstance(min_periods, bool):
            raise ValueError("min_periods must be integer")
        if isinstance(min_periods, float):
            raise ValueError("min_periods must be integer")
        min_periods = int(min_periods)
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        # FL-P0-003: Ensure min_periods is at least er_window+1 (ER computation floor)
        if min_periods < er_window + 1:
            min_periods = er_window + 1

    # Efficiency Ratio: |change| / volatility
    change = (x - x.shift(er_window)).abs()
    vol = x.diff().abs().rolling(er_window, min_periods=er_window).sum()
    epsilon = 1e-10
    efficiency = change / (vol + epsilon)

    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2

    # Recursive filter
    arr = x.to_numpy(float)
    alpha = sc.to_numpy(float)
    out = np.full_like(arr, np.nan, float)

    for c in range(arr.shape[1]):
        # Break + rewarm: a NaN price invalidates the recursive state.
        last = np.nan
        contiguous = 0

        for t in range(arr.shape[0]):
            if not np.isfinite(arr[t, c]):
                last = np.nan
                contiguous = 0
                continue

            contiguous += 1

            # FL-P0-003: Use min_periods (at least er_window+1)
            if contiguous < min_periods:
                continue

            if not np.isfinite(last):
                # First legal KAMA seed after warmup/gap: the CURRENT close.
                last = arr[t, c]
            elif np.isfinite(alpha[t, c]):
                # Recursive update: y_t = y_{t-1} + SC_t * (x_t - y_{t-1})
                last = last + alpha[t, c] * (arr[t, c] - last)

            out[t, c] = last

    return pd.DataFrame(out, index=x.index, columns=x.columns)


class TsKamaOperator(SeriesOperator):
    """Kaufman Adaptive Moving Average operator."""

    def _calculate_series(
        self,
        x: pd.DataFrame,
        er_window: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        min_periods: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        return ts_kama(x, er_window, fast_period, slow_period, min_periods)

    def __call__(
        self,
        x: pd.DataFrame,
        er_window: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        min_periods: int | None = None,
    ) -> pd.DataFrame:
        return ts_kama(x, er_window, fast_period, slow_period, min_periods)


@register_operator(
    name="ts_kama",
    canonical="ts_kama",
    category="signal_filter",
    business_category="signal_filter",
    source="filter_smooth",
)
class TSKama(TsKamaOperator):
    """Kaufman Adaptive Moving Average: adaptive smoothing driven by efficiency ratio."""

    metadata = OperatorMetadata(
        name="ts_kama",
        category="signal_filter",
        description="Kaufman Adaptive Moving Average: adaptive smoothing driven by efficiency ratio",
        param_names=["x", "er_window", "fast_period", "slow_period", "min_periods"],
        return_type="series",
        tags=["signal_filter", "causal", "stateful", "adaptive"],
    )

    stateful = True
    checkpointable = True
    time_shard_safe = False

    def _calculate_series(
        self,
        x: pd.DataFrame,
        er_window: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        min_periods: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        """Delegate to ts_kama function."""
        return ts_kama(x, er_window, fast_period, slow_period, min_periods)

_CANONICALS.append("ts_kama")


@register_operator(
    name="ts_causal_local_linear_smoother",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_causal_local_linear_smoother",
    source="filter_smooth",
)
class TSCausalLocalLinearSmoother(SeriesOperator):
    """One-sided local linear smoother: fits trailing window with linear regression.

    Mathematical formulation (§5 C3):
        For each t, use trailing window [t-window+1, ..., t]:
            positions = [0, 1, 2, ..., window-1]
            y_window = x[t-window+1:t+1]

            OLS fit: y = a + b * position

            Output: y_t = a + b * (window - 1)  (endpoint fitted value)

    This provides lower lag on trends compared to SMA because the linear fit
    extrapolates the slope to the endpoint rather than averaging.

    Certified parameters:
        window: [10, 20, 40] (HORIZON role, searchable)

    Execution contract:
        - Non-stateful: each period independently fits the trailing window
        - Time shard safe: no recursive state
        - Min periods: window (need full window for stable fit)
        - Strictly one-sided: NOT centered LOESS

    Parameters:
        x: Input signal series
        window: Trailing window length, default 20
        min_periods: Minimum observations for valid fit, default 10

    Returns:
        Smoothed signal with local linear endpoint values
    """

    metadata = OperatorMetadata(
        name="ts_causal_local_linear_smoother",
        category="signal_filter",
        description="One-sided local linear smoother with lower lag on trends",
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=["signal_filter", "causal", "stateless"],
        param_specs={
            "window": ParamSpec(
                dtype=int,
                min=2,
                default=20,
                param_role=ParamRole.HORIZON,
                searchable=True,
                history_semantics="max_rows",
            ),
            "min_periods": ParamSpec(
                dtype=int,
                min=2,
                default=10,
                param_role=ParamRole.SUPPORT_POLICY,
                searchable=False,
            ),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        min_periods: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        if window < 2:
            raise ValueError(
                f"ts_causal_local_linear_smoother requires window >= 2, got {window}"
            )
        if min_periods < 2:
            raise ValueError(
                f"ts_causal_local_linear_smoother requires min_periods >= 2, got {min_periods}"
            )
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})"
            )

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            for row in range(rows):
                # Determine trailing window
                start_idx = max(0, row - window + 1)
                window_data = xv[start_idx:row+1, col]

                # FL-P0-005: Filter finite values AND preserve physical bar offsets
                finite_mask = np.isfinite(window_data)
                finite_values = window_data[finite_mask]

                if len(finite_values) < min_periods:
                    continue

                # FL-P0-005: Build position array preserving physical time offsets
                # WRONG: positions = np.arange(len(finite_values))  # compresses time
                # RIGHT: positions = indices_where_finite  # preserves gaps
                n = len(finite_values)
                window_indices = np.arange(len(window_data), dtype=float)
                positions = window_indices[finite_mask]  # physical bar offsets

                # OLS: y = a + b * position
                # Normal equations:
                #   a * n + b * sum(pos) = sum(y)
                #   a * sum(pos) + b * sum(pos^2) = sum(pos * y)
                sum_pos = positions.sum()
                sum_pos2 = (positions ** 2).sum()
                sum_y = finite_values.sum()
                sum_pos_y = (positions * finite_values).sum()

                # Solve 2x2 system
                denom = n * sum_pos2 - sum_pos * sum_pos
                if abs(denom) < 1e-14:
                    # Degenerate case: perfectly collinear positions (shouldn't happen)
                    # Fall back to mean
                    out[row, col] = finite_values.mean()
                    continue

                a = (sum_pos2 * sum_y - sum_pos * sum_pos_y) / denom
                b = (n * sum_pos_y - sum_pos * sum_y) / denom

                # FL-P0-005: Endpoint fitted value at the PHYSICAL position of current bar
                # The last position in the window is positions[-1] (not n-1 after compression)
                y_t = a + b * positions[-1]
                out[row, col] = y_t

        from cleaned_operators.rolling_pack import frame_like
        return frame_like(x, out)


_CANONICALS.append("ts_causal_local_linear_smoother")


@register_operator(
    name="ts_butterworth_lowpass_causal",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_butterworth_lowpass_causal",
    source="filter_smooth",
)
class ButterworthLowpassCausalOperator(SeriesOperator):
    """Causal Butterworth IIR low-pass filter using second-order sections.

    A maximally flat magnitude filter with strong attenuation beyond the cutoff
    frequency. Uses SOS (second-order sections) representation for numerical
    stability in recursive filtering.

    Mathematical formulation:
        cutoff_freq = 1.0 / cutoff_period
        b, a = scipy.signal.butter(order, cutoff_freq, btype='low', output='ba')
        sos = scipy.signal.tf2sos(b, a)

        Forward-only recursive filtering through cascaded second-order sections.
        Each section maintains 2 historical inputs + 2 historical outputs.

    Certified parameters:
        cutoff_period: [10, 20, 40, 60] (HORIZON role, searchable)
        order: [2, 3, 4] (SCALAR_INT role, searchable)

    Execution contract:
        - Stateful: recursive filter depends on full finite prefix
        - Checkpointable: state is SOS history (zi) — serializable
        - Time shard UNSAFE: requires checkpoint restore across shards
        - Strictly causal: forward-only (no filtfilt)
        - Warmup: minimal (initialized with zero state)

    Parameters:
        x: Input signal series
        cutoff_period: Filter cutoff period (larger = more smoothing), default 20
        order: Filter order (higher = steeper rolloff), default 2

    Returns:
        Low-pass filtered signal with Butterworth magnitude response
    """

    metadata = OperatorMetadata(
        name="ts_butterworth_lowpass_causal",
        category="signal_filter",
        description="Causal Butterworth IIR low-pass filter with maximally flat magnitude response",
        param_names=["x", "cutoff_period", "order"],
        return_type="series",
        tags=["signal_filter", "causal", "stateful"],
        param_specs={
            "cutoff_period": ParamSpec(
                dtype=int,
                min=3,
                default=20,
                param_role=ParamRole.HORIZON,
                searchable=True,
                history_semantics="max_rows",
            ),
            "order": ParamSpec(
                dtype=int,
                min=1,
                max=10,
                default=2,
                param_role=ParamRole.MODEL_ORDER,
                searchable=True,
            ),
        },
    )

    filter_contract = FilterContract(
        role=FilterRole.LOW_PASS,
        causal=True,
        uses_current_observation=True,
        stateful=True,
        checkpointable=True,
        time_shard_safe=False,
        warmup=0,  # cutoff_period is effective warmup
        lag_class="zero",
        jump_policy=JumpPreservationPolicy.PRESERVE_ALL_FINITE_JUMPS,
        turnover_control=False,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        cutoff_period: int = 20,
        order: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        """Execute Butterworth low-pass filtering."""
        if cutoff_period < 3:
            raise ValueError(
                f"ts_butterworth_lowpass_causal requires cutoff_period >= 3, got {cutoff_period}"
            )
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > 10:
            raise ValueError(f"order must be <= 10 for numerical stability, got {order}")

        try:
            from scipy import signal as sp_signal
        except ImportError:
            raise ImportError("scipy is required for Butterworth filter")

        # Design Butterworth digital filter
        cutoff_freq = 1.0 / cutoff_period
        # Nyquist frequency is 0.5 (normalized), ensure cutoff < Nyquist
        if cutoff_freq >= 0.5:
            raise ValueError(
                f"cutoff_period must be > 2 (cutoff_freq={cutoff_freq:.3f} >= Nyquist=0.5), got {cutoff_period}"
            )

        sos = sp_signal.butter(order, cutoff_freq, btype='low', analog=False, output='sos')

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            # Initialize state for cascaded second-order sections
            # sos shape: (n_sections, 6) where each row is [b0, b1, b2, a0, a1, a2]
            n_sections = sos.shape[0]
            zi = None  # Lazy initialization with first valid value

            for row in range(rows):
                x_curr = xv[row, col]

                if not np.isfinite(x_curr):
                    # NaN input: output NaN, state freezes
                    out[row, col] = np.nan
                    continue

                # Initialize state with first valid observation
                if zi is None:
                    zi_template = sp_signal.sosfilt_zi(sos)
                    zi = zi_template * x_curr
                    out[row, col] = x_curr
                    continue

                # Forward filter through cascaded sections
                y = x_curr
                for s in range(n_sections):
                    b0, b1, b2, a0, a1, a2 = sos[s, :]
                    # Normalize by a0 (should be 1.0 but for safety)
                    b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

                    # Direct Form II transposed (standard for IIR SOS)
                    y_out = b0 * y + zi[s, 0]
                    zi[s, 0] = b1 * y - a1 * y_out + zi[s, 1]
                    zi[s, 1] = b2 * y - a2 * y_out
                    y = y_out

                out[row, col] = y

        from cleaned_operators.rolling_pack import frame_like
        return frame_like(x, out)


_CANONICALS.append("ts_butterworth_lowpass_causal")


def filter_smooth_contract(canonical: str) -> FilterContract:
    """返回 filter_smooth 模块算子的 FilterContract。"""
    if canonical == "ts_robust_ema":
        return FilterContract(
            role=FilterRole.LOW_PASS,
            causal=True,
            uses_current_observation=True,
            stateful=True,
            checkpointable=True,
            time_shard_safe=False,
            warmup=20,  # warmup_window 默认值
            lag_class="one",  # EMA 有一期滞后
            jump_policy=JumpPreservationPolicy.ROBUST_CLIP_INNOVATION,
            turnover_control=True,  # 通过 clip_sigma 控制换手
        )
    if canonical == "ts_kama":
        return FilterContract(
            role=FilterRole.ADAPTIVE_LOW_PASS,
            causal=True,
            uses_current_observation=True,
            stateful=True,
            checkpointable=True,
            time_shard_safe=False,
            warmup=11,  # er_window + 1 默认值
            lag_class="variable",  # 自适应滞后
            jump_policy=JumpPreservationPolicy.PRESERVE_ALL_FINITE_JUMPS,
            turnover_control=True,  # ER 自动控制响应速度
        )
    if canonical == "ts_super_smoother":
        return FilterContract(
            role=FilterRole.LOW_PASS,
            causal=True,
            uses_current_observation=True,
            stateful=True,
            checkpointable=True,
            time_shard_safe=False,
            warmup=0,  # period via min_periods
            lag_class="zero",
            jump_policy=JumpPreservationPolicy.PRESERVE_ALL_FINITE_JUMPS,
            turnover_control=False,
        )
    if canonical == "ts_butterworth_lowpass_causal":
        return FilterContract(
            role=FilterRole.LOW_PASS,
            causal=True,
            uses_current_observation=True,
            stateful=True,
            checkpointable=True,
            time_shard_safe=False,
            warmup=0,  # cutoff_period via min_periods
            lag_class="one",  # IIR has some phase lag
            jump_policy=JumpPreservationPolicy.PRESERVE_ALL_FINITE_JUMPS,
            turnover_control=False,
        )
    if canonical == "ts_causal_local_linear_smoother":
        return FilterContract(
            role=FilterRole.LOW_PASS,
            causal=True,
            uses_current_observation=True,
            stateful=False,
            checkpointable=False,
            time_shard_safe=True,
            warmup=10,  # min_periods default
            lag_class="zero",  # Lower lag than SMA on trends
            jump_policy=JumpPreservationPolicy.PRESERVE_ALL_FINITE_JUMPS,
            turnover_control=False,
        )
    raise ValueError(f"Unknown canonical: {canonical}")


def _register_surface() -> None:
    """注册到 extended surface 并添加 Polars 后端支持。"""
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.rolling_pack import register_polars_bridge

    _surface.extend_extended_only(set(_CANONICALS))

    # Polars 后端：委托 pandas reference
    for _canon in _CANONICALS:
        register_polars_bridge(_canon)


_register_surface()

