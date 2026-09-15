# -*- coding: utf-8 -*-
"""严格因果Hampel去毛刺算子（2026-08-12信号滤波层专项）。

Hampel filter用中位数+MAD估计稳健尺度，识别并压制异常幅度的毛刺。严格因果版本
使用过去窗口估计参考（当前点不参与自己的阈值），保留跳变方向但抑制极端幅度。
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, RelationalParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like
from factor_engine.cleaned_operators.filter_contracts import (
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
        panel_params=("x",), scalar_params=("window","n_sigma","replacement","scale_floor"),
        output_unit="same_as:x",
        param_specs={
            "window": ParamSpec(dtype=int,min=2,default=20,param_role=ParamRole.HORIZON,
                                history_formula="window + 1"),
            "n_sigma": ParamSpec(dtype=float,min=float(np.nextafter(0.,1.)),default=3.,
                                param_role=ParamRole.STATE_THRESHOLD),
            "replacement": ParamSpec(dtype=str,choices=("clip","median"),default="clip",
                                    param_role=ParamRole.POLICY,searchable=False),
            "scale_floor": ParamSpec(dtype=float,min=float(np.nextafter(0.,1.)),default=1e-10,
                                    param_role=ParamRole.NUMERICAL,searchable=False),
        },
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x,window,n_sigma,replacement,scale_floor->series",
            "domain:signal_filter", "unit:same_as:x",
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

    def _calculate_series(self,x,window=20,n_sigma=3.,replacement="clip",scale_floor=1e-10,**_):
        w,sigma,policy,floor=_hampel_parameters(window,n_sigma,replacement,scale_floor)
        return frame_like(x,_hampel(x.to_numpy(dtype=float),w,sigma,policy,floor))


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
        panel_params=("x",), scalar_params=(), output_unit="same_as:x",
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x->series",
            "domain:signal_filter", "unit:same_as:x",
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

    def _calculate_series(self,x,**_):
        return frame_like(x,_median3(x.to_numpy(dtype=float)))


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
        panel_params=("x",), scalar_params=("window","min_periods"), output_unit="same_as:x",
        param_specs={
            "window": ParamSpec(dtype=int,min=2,default=5,param_role=ParamRole.HORIZON,
                                history_semantics="max_rows"),
            "min_periods": ParamSpec(dtype=int,min=1,default=3,param_role=ParamRole.SUPPORT_POLICY,
                                    searchable=False),
        },
        relational_specs=[RelationalParamSpec("min_periods <= window","min_periods must not exceed window")],
        return_type="series",
        tags=[
            "signal_filter", "causal", "typed_v2", "pit_safe",
            "signature:x,window,min_periods->series",
            "domain:signal_filter", "unit:same_as:x",
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

    def _calculate_series(self,x,window=5,min_periods=3,**_):
        w,minimum=_median_parameters(window,min_periods)
        return frame_like(x,_rolling_median(x.to_numpy(dtype=float),w,minimum))


def _hampel_parameters(window,n_sigma,replacement,scale_floor):
    from factor_engine.cleaned_operators.common.strict_params import strict_int,strict_float,strict_enum
    positive=float(np.nextafter(0.,1.))
    return (strict_int(window,"window",minimum=2),
            strict_float(n_sigma,"n_sigma",minimum=positive),
            strict_enum(replacement,"replacement",("clip","median")),
            strict_float(scale_floor,"scale_floor",minimum=positive))


def _median_parameters(window,min_periods):
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    w=strict_int(window,"window",minimum=2)
    minimum=strict_int(min_periods,"min_periods",minimum=1,maximum=w)
    return w,minimum


def _finite_median(values):
    # Wide accumulation avoids overflow of the two central finite float64
    # values and preserves small representable midpoints.
    return np.median(np.asarray(values,dtype=np.longdouble))


def _hampel(values,window,n_sigma,replacement,scale_floor):
    out=np.full(values.shape,np.nan)
    for c in range(values.shape[1]):
        for t in range(values.shape[0]):
            current=values[t,c]
            if not np.isfinite(current):
                continue
            out[t,c]=current
            if t<window:
                continue
            past=values[t-window:t,c]
            past=past[np.isfinite(past)].astype(np.longdouble)
            if past.size<2:
                continue
            center=_finite_median(past)
            mad=_finite_median(np.abs(past-center))
            # Exact degeneracy, not a price-unit-dependent absolute epsilon.
            if mad==0:
                span=past.max()-past.min()
                if span==0:
                    continue  # Existing truly-flat-history bypass policy.
                robust_scale=np.longdouble("1.4826")*span/4
            else:
                robust_scale=np.longdouble("1.4826")*mad
            threshold=np.longdouble(n_sigma)*max(robust_scale,np.longdouble(scale_floor))
            delta=np.longdouble(current)-center
            if abs(delta)>threshold:
                filtered=center if replacement=="median" else center+np.clip(delta,-threshold,threshold)
                out[t,c]=float(filtered) if np.isfinite(filtered) else np.nan
    return out


def _median3(values):
    out=np.full(values.shape,np.nan)
    for c in range(values.shape[1]):
        for t in range(values.shape[0]):
            recent=values[max(0,t-2):t+1,c]
            if t<2:
                out[t,c]=values[t,c] if np.isfinite(values[t,c]) else np.nan
            elif np.isfinite(recent).all():
                out[t,c]=float(_finite_median(recent))
    return out


def _rolling_median(values,window,min_periods):
    out=np.full(values.shape,np.nan)
    for c in range(values.shape[1]):
        for t in range(values.shape[0]):
            recent=values[max(0,t-window+1):t+1,c]
            recent=recent[np.isfinite(recent)]
            if recent.size>=min_periods:
                out[t,c]=float(_finite_median(recent))
    return out


def _register_surface() -> None:
    """注册到 extended surface 并添加 Polars 后端支持。"""
    import factor_engine.cleaned_operators.operator_surface as _surface
    from factor_engine.cleaned_operators.rolling_pack import register_polars_udf

    _CANONICALS = {
        "ts_hampel_filter_causal",
        "ts_median3_causal",
        "ts_rolling_median_causal",
    }

    _surface.extend_extended_only(_CANONICALS)

    # Polars 后端：同一 NumPy 内核，避免 pandas 面板转换
    for _canon in _CANONICALS:
        register_polars_udf(_canon)


_register_surface()
