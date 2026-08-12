# -*- coding: utf-8 -*-
"""严格因果Hampel去毛刺算子（2026-08-12信号滤波层专项）。

Hampel filter用中位数+MAD估计稳健尺度，识别并压制异常幅度的毛刺。严格因果版本
使用过去窗口估计参考（当前点不参与自己的阈值），保留跳变方向但抑制极端幅度。
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.filter_contracts import (
    FilterContract,
    FilterRole,
    JumpPreservationPolicy,
)


@register_operator(
    name="ts_hampel_filter_causal",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_hampel_filter_causal",
    source="filter_despike",
)
class TSHampelFilterCausal(SeriesOperator):
    """严格因果Hampel去毛刺：过去窗口估计中位数+MAD，clip当前观测的异常幅度。

    数学（严格使用过去窗口）：
        m_t = median(x_{t-window:t-1})
        s_t = 1.4826 * median(|x_{t-window:t-1} - m_t|)
        s_t = max(s_t, scale_floor)

        if |x_t - m_t| > n_sigma * s_t:
            if replacement == "clip":
                x_t* = m_t + clip(x_t - m_t, -n_sigma*s_t, n_sigma*s_t)
            elif replacement == "median":
                x_t* = m_t
        else:
            x_t* = x_t

    认证参数：
        n_sigma: 只允许 [2.5, 3.0, 4.0]
        scale_floor: NUMERICAL_POLICY，不可搜索

    Parameters:
        x: 输入信号
        window: 回溯窗口（估计参考的历史长度）
        n_sigma: 标准差倍数阈值（默认3.0）
        replacement: 替换策略 "clip"（保留方向）或 "median"（直接替换为中位数）
        scale_floor: MAD下限（防止除零），默认1e-10

    Returns:
        去毛刺后的信号（保留跳变方向，压制异常幅度）
    """

    metadata = OperatorMetadata(
        name="ts_hampel_filter_causal",
        category="signal_filter",
        description="严格因果Hampel去毛刺：过去窗口估计中位数+MAD，clip异常幅度",
        param_names=["x", "window", "n_sigma", "replacement", "scale_floor"],
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x,window,n_sigma,replacement,scale_floor->series",
            "domain:signal_filter", "unit:dimensionless",
            "filter_contract:despike",
        ],
    )

    filter_contract = FilterContract(
        role=FilterRole.DESPIKE,
        causal=True,
        uses_current_observation=True,
        stateful=False,
        checkpointable=False,
        time_shard_safe=True,
        warmup=0,  # window即预热
        lag_class="zero",
        jump_policy=JumpPreservationPolicy.ROBUST_CLIP_INNOVATION,
        turnover_control=False,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        n_sigma: float = 3.0,
        replacement: Literal["clip", "median"] = "clip",
        scale_floor: float = 1e-10,
        **_: Any,
    ) -> pd.DataFrame:
        # 参数校验
        if window < 2:
            raise ValueError(f"ts_hampel_filter_causal requires window >= 2, got {window}")
        if n_sigma <= 0:
            raise ValueError(f"ts_hampel_filter_causal requires n_sigma > 0, got {n_sigma}")
        if scale_floor <= 0:
            raise ValueError(f"ts_hampel_filter_causal requires scale_floor > 0, got {scale_floor}")
        if replacement not in ("clip", "median"):
            raise ValueError(f"ts_hampel_filter_causal replacement must be 'clip' or 'median', got {replacement!r}")

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # MAD常数（使MAD估计一致于标准差）
        MAD_SCALE = 1.4826

        for col in range(cols):
            for row in range(rows):
                curr = xv[row, col]
                if not np.isfinite(curr):
                    out[row, col] = np.nan
                    continue

                # 预热期：窗口不足，直接输出原值
                if row < window:
                    out[row, col] = curr
                    continue

                # 严格因果：只使用过去窗口 [row-window:row]（不含row）
                past_window = xv[row - window : row, col]
                finite_mask = np.isfinite(past_window)
                finite_vals = past_window[finite_mask]

                # 至少需要2个有限值才能估计MAD
                if len(finite_vals) < 2:
                    out[row, col] = curr
                    continue

                # 估计中位数和MAD
                m_t = float(np.median(finite_vals))
                mad_raw = float(np.median(np.abs(finite_vals - m_t)))
                s_t = max(MAD_SCALE * mad_raw, scale_floor)

                # 判断是否为毛刺
                deviation = abs(curr - m_t)
                threshold = n_sigma * s_t

                if deviation > threshold:
                    # 超出阈值：应用替换策略
                    if replacement == "clip":
                        # clip：保留方向，压制幅度
                        delta = curr - m_t
                        clipped_delta = float(np.clip(delta, -threshold, threshold))
                        out[row, col] = m_t + clipped_delta
                    else:
                        # median：直接替换为中位数
                        out[row, col] = m_t
                else:
                    # 正常范围：保留原值
                    out[row, col] = curr

        return frame_like(x, out)


@register_operator(
    name="ts_median3_causal",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_median3_causal",
    source="filter_despike",
)
class TSMedian3Causal(SeriesOperator):
    """三点中位数去毛刺：对孤立单点 spike 的极快抑制。

    数学：
        y_t = median(x_t, x_{t-1}, x_{t-2})

    机制：
        - 孤立 spike（单点异常）被周围两个正常值的中位数压制
        - 真实跳变（多点持续）保留
        - O(1) 每点，极快
        - 无参数

    认证参数：
        无参数（固定 3 点窗口）

    Parameters:
        x: 输入信号

    Returns:
        去毛刺后的信号（孤立 spike 被抑制）
    """

    metadata = OperatorMetadata(
        name="ts_median3_causal",
        category="signal_filter",
        description="三点中位数去毛刺：median(x_t, x_{t-1}, x_{t-2})，抑制孤立 spike",
        param_names=["x"],
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x->series",
            "domain:signal_filter", "unit:dimensionless",
            "filter_contract:despike",
        ],
    )

    filter_contract = FilterContract(
        role=FilterRole.DESPIKE,
        causal=True,
        uses_current_observation=True,
        stateful=False,
        checkpointable=False,
        time_shard_safe=True,
        warmup=0,
        lag_class="zero",
        jump_policy=JumpPreservationPolicy.ROBUST_CLIP_INNOVATION,
        turnover_control=False,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        **_: Any,
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            for row in range(rows):
                curr = xv[row, col]

                # 前两行：窗口不足，直接输出
                if row < 2:
                    out[row, col] = curr
                    continue

                # 收集最近三个观测
                val_0 = xv[row - 2, col]  # t-2
                val_1 = xv[row - 1, col]  # t-1
                val_2 = curr              # t

                # 如果任何一个是 NaN，输出 NaN
                if not (np.isfinite(val_0) and np.isfinite(val_1) and np.isfinite(val_2)):
                    out[row, col] = np.nan
                    continue

                # 计算三点中位数
                out[row, col] = float(np.median([val_0, val_1, val_2]))

        return frame_like(x, out)


@register_operator(
    name="ts_rolling_median_causal",
    category="signal_filter",
    business_category="signal_filter",
    canonical="ts_rolling_median_causal",
    source="filter_despike",
)
class TSRollingMedianCausal(SeriesOperator):
    """短窗口滚动中位数去毛刺：比 mean 更抗 outlier。

    数学：
        y_t = median(x_{t-window+1}, ..., x_t)

    机制：
        - 中位数天然抗 outlier（50% breakdown point）
        - 短窗口（3/5/7）避免过大滞后
        - 适合中等频率毛刺抑制

    认证参数：
        window: 只允许 [3, 5, 7]（避免大窗口高滞后）
        min_periods: NUMERICAL_POLICY，不可搜索

    Parameters:
        x: 输入信号
        window: 滚动窗口（默认 5）
        min_periods: 最小有效观测数（默认 3）

    Returns:
        去毛刺后的信号（outlier 被窗口中位数压制）
    """

    metadata = OperatorMetadata(
        name="ts_rolling_median_causal",
        category="signal_filter",
        description="短窗口滚动中位数去毛刺：比 mean 更抗 outlier",
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x,window,min_periods->series",
            "domain:signal_filter", "unit:dimensionless",
            "filter_contract:despike",
        ],
    )

    filter_contract = FilterContract(
        role=FilterRole.DESPIKE,
        causal=True,
        uses_current_observation=True,
        stateful=False,
        checkpointable=False,
        time_shard_safe=True,
        warmup=0,
        lag_class="zero",
        jump_policy=JumpPreservationPolicy.ROBUST_CLIP_INNOVATION,
        turnover_control=False,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 5,
        min_periods: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        # 参数校验
        if window < 2:
            raise ValueError(f"ts_rolling_median_causal requires window >= 2, got {window}")
        if min_periods < 1:
            raise ValueError(f"ts_rolling_median_causal requires min_periods >= 1, got {min_periods}")
        if min_periods > window:
            raise ValueError(
                f"ts_rolling_median_causal requires min_periods <= window, got {min_periods} > {window}"
            )

        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            for row in range(rows):
                # 窗口起点（包含当前点）
                start = max(0, row - window + 1)
                window_vals = xv[start : row + 1, col]

                # 只保留有限值
                finite_mask = np.isfinite(window_vals)
                finite_vals = window_vals[finite_mask]

                # 有效观测数不足 min_periods
                if len(finite_vals) < min_periods:
                    out[row, col] = np.nan
                    continue

                # 计算中位数
                out[row, col] = float(np.median(finite_vals))

        return frame_like(x, out)
