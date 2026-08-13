# -*- coding: utf-8 -*-
"""Alpha-language path / shape geometry operators (2026-08).

Two stocks can end at the same cumulative return yet follow very different
paths.  This family describes *how* the path got there:

* monotonicity: Kendall-style directional agreement over the window;
* turning rate / intensity: how often the path reverses and how violent the
  reversals are;
* path efficiency / roughness: net displacement vs path length, second-order
  jaggedness;
* trend break: split-window slope difference (acceleration / reversal);
* weighted time centroid / mass concentration: where the action happened in
  time and whether it is concentrated in a few bars;
* endpoint deviation: how far the last point sits off its own trailing OLS fit.

All operators are causal trailing-window transforms (row ``t`` depends only on
rows ``<= t``).  Constant / degenerate windows return NaN, never an invented
value; weights are required non-negative.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    check_window,
    frame_like,
    map_rolling,
    register_polars_bridge,
)

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_shape",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_shape", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:path_geometry",
            f"unit:{unit}", "cost:1",
            *extra_tags,
        ],
    )


def _finite(chunk: np.ndarray) -> np.ndarray:
    return chunk[np.isfinite(chunk)]


def _trailing_contiguous(chunk: np.ndarray) -> np.ndarray:
    """Trailing contiguous run of finite values (no time-axis compression).

    Drops only *leading* NaNs; an interior or trailing NaN (gap) truncates the
    run, so the retained points keep their original relative spacing.  Empty
    when the current (trailing) value is NaN.  Unlike ``_finite`` this never
    reconnects points across a missing-value gap, which would distort path
    geometry (turning rate / efficiency / roughness / trend break / centroid).
    """
    chunk = np.asarray(chunk, dtype=float)
    n = chunk.size
    if n == 0:
        return chunk
    finite = np.isfinite(chunk)
    if not finite[-1]:
        return np.empty(0, dtype=float)
    last_bad = np.flatnonzero(~finite)
    if last_bad.size == 0:
        return chunk
    return chunk[last_bad[-1] + 1 :]


def _ols_slope(y: np.ndarray) -> float:
    n = y.size
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    denom = n * float((x * x).sum()) - float(x.sum()) ** 2
    if denom <= 0.0:
        return np.nan
    return np.where(denom != 0, (n * float((x * y).sum()) - float(x.sum()) * float(y.sum())) / denom, np.nan)


def _ols_fit(y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, residual-std ddof=0) for y ~ a + b*j."""
    n = y.size
    x = np.arange(n, dtype=float)
    sx = float(x.sum())
    sy = float(y.sum())
    denom = n * float((x * x).sum()) - sx * sx
    b = (n * float((x * y).sum()) - sx * sy) / denom if denom != 0 else np.nan
    a = (sy - b * sx) / n if n > 0 else np.nan
    resid = y - (a + b * x)
    return float(b), float(a), float(np.std(resid))


@register_operator(
    name="ts_monotonicity",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_monotonicity",
    source="alpha_language_shape",
)
class TsMonotonicity(SeriesOperator):
    """Kendall 单调性: (C - D) / (C + D), 范围 [-1, 1]。

    对窗口内所有 (i<j) 对, C = 同向对, D = 反向对(按值差与位置差的符号)。
    +1 = 几乎严格单调上升; -1 = 单调下降; 0 = 无明显单调结构。
    """

    metadata = _metadata(
        "ts_monotonicity",
        "Kendall 式单调性 (C-D)/(C+D)。",
        ["x", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _finite(chunk)
            n = v.size
            if n < mp:
                return np.nan
            c = 0
            d = 0
            for i in range(n):
                for j in range(i + 1, n):
                    val_diff = v[j] - v[i]
                    if val_diff == 0.0:
                        continue
                    if (val_diff > 0) == (j > i):
                        c += 1
                    else:
                        d += 1
            total = c + d
            if total == 0:
                return np.nan
            return np.where(total) != 0, float((c - d) / total), np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_turning_rate",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_turning_rate",
    source="alpha_language_shape",
)
class TsTurningRate(SeriesOperator):
    """转向率 (Definition A): 窗口内相邻 delta 符号翻转数 / 全部相邻 delta 对数, 范围 [0,1]。

    Definition A 的分母是**所有**相邻 delta 对, 包括 0 delta (平台段) —— 因此
    (+ ,0,0,0,-) 这类含平台段的序列会被人为拉低转向率 (0 翻转 / 4 对 = 0)。
    sign_eps(d): |d|>epsilon 才计入符号; epsilon 用于滤除噪声。高 = 高频折返。

    R11 round-3 P1-I-127：需要忽略平台段稀释的度量使用 Definition B ——
    ``ts_effective_turning_rate``（分母只计两 delta 均非零的相邻对）。
    """

    metadata = _metadata(
        "ts_turning_rate",
        "delta 符号翻转比例 (Definition A: 分母=全部相邻 delta 对数)。",
        ["x", "window", "epsilon", "min_periods"],
        unit="ratio",
        extra_tags=("definition:a",),
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, epsilon: float = 0.0, min_periods: int = 2, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        eps = float(epsilon)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _sign(d: float) -> int:
            if d > eps:
                return 1
            if d < -eps:
                return -1
            return 0

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            if v.size < mp + 1:
                return np.nan
            d = np.diff(v)
            flips = 0
            pairs = 0
            for i in range(1, d.size):
                s0 = _sign(float(d[i - 1]))
                s1 = _sign(float(d[i]))
                pairs += 1
                if s0 != 0 and s1 != 0 and s0 != s1:
                    flips += 1
            if pairs == 0:
                return np.nan
            return np.where(pairs) != 0, float(flips / pairs), np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_effective_turning_rate",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_effective_turning_rate",
    source="alpha_language_shape",
)
class TsEffectiveTurningRate(SeriesOperator):
    """有效转向率 (Definition B): 相邻 delta 对中, 两 delta 均有效(非零)的对里
    符号翻转的比例, 范围 [0,1]。

    Definition B 的分母只计 **两 delta 均非零** 的相邻对 —— 平台段 (0 delta) 不
    稀释分母, 与 ``ts_turning_rate`` (Definition A, 分母=全部相邻对) 区分。
    (+ ,0,0,0,-) 在 Definition B 下无任何"两 delta 均有效"的相邻对 -> NaN。
    """

    metadata = _metadata(
        "ts_effective_turning_rate",
        "有效转向率 (Definition B: 分母=两 delta 均非零的相邻对数)。",
        ["x", "window", "epsilon", "min_periods"],
        unit="ratio",
        extra_tags=("definition:b",),
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, epsilon: float = 0.0, min_periods: int = 2, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        eps = float(epsilon)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _sign(d: float) -> int:
            if d > eps:
                return 1
            if d < -eps:
                return -1
            return 0

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            if v.size < mp + 1:
                return np.nan
            d = np.diff(v)
            flips = 0
            active_pairs = 0
            for i in range(1, d.size):
                s0 = _sign(float(d[i - 1]))
                s1 = _sign(float(d[i]))
                if s0 == 0 or s1 == 0:
                    continue  # platform segment (0 delta) is not an active pair
                active_pairs += 1
                if s0 != s1:
                    flips += 1
            if active_pairs == 0:
                return np.nan
            return np.where(active_pairs) != 0, float(flips / active_pairs), np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_turning_intensity",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_turning_intensity",
    source="alpha_language_shape",
)
class TsTurningIntensity(SeriesOperator):
    """转向猛烈度: 仅转向点上的 |Δ²x| 均值 / 窗口内 Δx 的 MAD。

    不是数转向次数, 而是衡量每次转向有多猛烈。
    R11 round-3 P1-I-128: MAD(delta)=0 (|Δx| 恒定) 时比率退化 —— 返回 NaN,
    绝不除以 eps 制造 1e12 的爆炸值。
    """

    metadata = _metadata(
        "ts_turning_intensity",
        "转向点 |Δ²x| 均值(按 Δx MAD 归一)。",
        ["x", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            if v.size < mp:
                return np.nan
            d = np.diff(v)
            if d.size < 2:
                return np.nan
            mad_d = float(np.median(np.abs(d - np.median(d))))
            mags = []
            for i in range(1, d.size):
                if d[i - 1] == 0.0 or d[i] == 0.0:
                    continue
                if (d[i] > 0) != (d[i - 1] > 0):
                    mags.append(abs(d[i] - d[i - 1]))
            if not mags:
                return np.nan
            # R11 round-3 P1-I-128: MAD(delta)=0 (constant |delta|) makes the
            # ratio 4e12 via the eps guard — an EPS explosion.  Return NaN, never
            # divide by _EPS.
            if mad_d == 0.0:
                return np.nan
            return np.where(mad_d != 0, float(np.mean(mags)) / mad_d, np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_path_efficiency",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_path_efficiency",
    source="alpha_language_shape",
)
class TsPathEfficiency(SeriesOperator):
    """路径效率: |x_t - x_{窗口首}| / 窗口内 |Δx| 之和, 范围 [0,1]。

    1 = 近乎直线; 0 = 大量折返。与 Kaufman efficiency ratio 同族的日频版本。
    常量路径 (net=0, path=0) 的 0/0 不是 0 —— 本算子显式声明
    ``constant_path_policy=ZERO``: 直接返回 0.0, 而非 0/(0+eps) 的数值事故
    (R11 round-3 P1-I-129)。
    """

    metadata = _metadata(
        "ts_path_efficiency",
        "净位移 / 路径长度 (常量路径 constant_path_policy=ZERO)。",
        ["x", "window", "min_periods"],
        unit="ratio",
        extra_tags=("constant_path_policy:zero",),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            if v.size < mp:
                return np.nan
            net = abs(float(v[-1]) - float(v[0]))
            path = float(np.sum(np.abs(np.diff(v))))
            if path == 0.0:
                # constant_path_policy=ZERO (declared): 0/0 is deliberately 0.0,
                # not an _EPS accident.
                return 0.0
            return np.where(path != 0, net / path, np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_roughness",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_roughness",
    source="alpha_language_shape",
)
class TsRoughness(SeriesOperator):
    """路径粗糙度: Σ(Δ²x)² / (Σ(Δx)² + eps)。衡量局部锯齿程度。"""

    metadata = _metadata(
        "ts_roughness",
        "二阶差分平方和 / 一阶差分平方和。",
        ["x", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            if v.size < mp:
                return np.nan
            d = np.diff(v)
            if d.size < 2:
                return np.nan
            d2 = np.diff(d)
            s2 = float(np.sum(d2 * d2))
            s1 = float(np.sum(d * d))
            return s2 / (s1 + _EPS)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_trend_break_score",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_trend_break_score",
    source="alpha_language_shape",
)
class TsTrendBreakScore(SeriesOperator):
    """趋势断点分: (b_recent - b_old) / (σ_recent_resid + eps)。

    窗口按 split 分为 old/recent 两段, 分别 OLS 斜率; 正 = 趋势加速, 负 = 减速/反转。
    每段需 >= 2 个点。
    """

    metadata = _metadata(
        "ts_trend_break_score",
        "分窗斜率差(按 recent 残差 std 归一)。",
        ["x", "window", "split", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, split: float = 0.5, min_periods: int = 4, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        sp = float(split)
        if not 0.0 < sp < 1.0:
            raise ValueError("split must be in (0, 1)")
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            n = v.size
            if n < mp:
                return np.nan
            old_len = max(2, int(np.floor(sp * n)))
            recent_len = n - old_len
            if recent_len < 2:
                return np.nan
            b_old = _ols_slope(v[:old_len])
            b_recent = _ols_slope(v[old_len:])
            if not (np.isfinite(b_old) and np.isfinite(b_recent)):
                return np.nan
            _, _, sigma = _ols_fit(v[old_len:])
            # R11 round-3 P1-I-130: recent residual std == 0 (perfectly linear
            # recent segment) made the ratio explode via the _EPS guard when the
            # two slopes differ.  Degenerate scale -> NaN, unless the two slopes
            # are also identical (0/0 -> policy 0.0, preserving the linear-window
            # semantics).
            if sigma < _EPS:
                if abs(b_recent - b_old) < _EPS:
                    return 0.0
                return np.nan
            return np.where(sigma != 0, (b_recent - b_old) / sigma, np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_weighted_time_centroid",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_weighted_time_centroid",
    source="alpha_language_shape",
)
class TsWeightedTimeCentroid(SeriesOperator):
    """时间加权质心: TC = 2*sum(j*w_j)/((n-1)*sum(w_j) + eps) - 1, 范围 [-1,1]。

    +1 = 活动集中在窗口近期; -1 = 集中在窗口早期。weight 要求非负。
    """

    metadata = _metadata(
        "ts_weighted_time_centroid",
        "权重时间质心(映射到 [-1,1])。",
        ["weight", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, weight: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        wv = weight.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            n = v.size
            if n < mp:
                return np.nan
            if np.any(v < 0.0):
                return np.nan
            total = float(v.sum())
            if total <= 0.0:
                return np.nan
            pos = np.arange(n, dtype=float)
            num = float(np.sum(pos * v))
            if n <= 1:
                return np.nan
            tc = 2.0 * num / ((n - 1.0) * total + _EPS) - 1.0
            return tc

        return frame_like(weight, map_rolling(wv, w, _fn))


@register_operator(
    name="ts_endpoint_deviation",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_endpoint_deviation",
    source="alpha_language_shape",
)
class TsEndpointDeviation(SeriesOperator):
    """端点偏离: (x_t - OLS 预测_x_t) / 残差 std。

    历史窗口形状描述(非预测), 因此允许包含 t; 完美线性窗口 -> 0。

    R11 round-3 P1-I-131: 这是路径/趋势几何 —— 使用 trailing-contiguous 尾部连续
    段, **不做** drop-finite 压缩。旧实现 ``chunk[np.isfinite(chunk)]`` 把
    (day1, day2, NaN, day10) 压成连续 3 点, OLS 的 x 轴重置为 0,1,2, 把跨缺口的
    点当作相邻。缺口后只保留真正相邻的尾部连续段。
    """

    metadata = _metadata(
        "ts_endpoint_deviation",
        "末端点相对自身 OLS 拟合的偏离(按残差 std 归一; 尾部连续段, 不压缩缺口)。",
        ["x", "window", "min_periods"],
        unit="ratio",
        extra_tags=("gap_policy:trailing_contiguous",),
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = _trailing_contiguous(chunk)
            n = v.size
            if n < mp:
                return np.nan
            b, a, sigma = _ols_fit(v)
            x_hat_last = a + b * float(n - 1)
            num = float(v[-1]) - x_hat_last
            if sigma < _EPS:
                return 0.0 if abs(num) < _EPS else np.nan
            return np.where(sigma != 0, num / sigma, np.nan)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_mass_concentration",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_mass_concentration",
    source="alpha_language_shape",
)
class TsMassConcentration(SeriesOperator):
    """质量集中度: (HHI - 1/n) / (1 - 1/n), 范围 [0,1]。

    HHI = sum(w/Σw)^2。高 = 变化/成交集中少数几天。等于 ts_abs_concentration 的
    归一化形式(对非负 weight 等价)。weight 要求非负。
    """

    metadata = _metadata(
        "ts_mass_concentration",
        "窗口权重归一化 HHI。",
        ["weight", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, weight: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        wv = weight.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            finite = np.isfinite(chunk)
            v = chunk[finite]
            n = v.size
            if n < mp:
                return np.nan
            if np.any(v < 0.0):
                return np.nan
            total = float(v.sum())
            if total <= 0.0:
                return np.nan
            shares = v / total if total != 0 else np.nan
            hhi = float(np.sum(shares * shares))
            if n <= 1:
                return np.nan
            return np.where(n) / (1.0 - 1.0 / n)) != 0, float((hhi - 1.0 / n) / (1.0 - 1.0 / n)), np.nan)

        return frame_like(weight, map_rolling(wv, w, _fn))


@register_operator(
    name="ts_chord_excursion_area",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_chord_excursion_area",
    source="alpha_language_shape",
)
class TsChordExcursionArea(SeriesOperator):
    """弦弓形面积: Σ_j(x_j - L_j) / (Σ_j|x_j - L_j| + eps), 范围 [-1,1]。

    L_j = x_0 + (j/(W-1))·(x_{W-1} - x_0) 为首尾连线。+1 = 路径主要拱在首尾
    连线上方（先跌后反弹）；-1 = 主要在下方（先涨后回落）。同样首尾收益的两只
    股票因此可区分。要求窗口内全 finite。
    """

    metadata = _metadata(
        "ts_chord_excursion_area",
        "路径相对首尾连线的净弓形面积（归一化 [-1,1]）。",
        ["x", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            if chunk.size < w or not np.all(np.isfinite(chunk)):
                return np.nan
            v = chunk.astype(float)
            if v.size < mp:
                return np.nan
            j = np.arange(v.size, dtype=float)
            den = max(v.size - 1, 1)
            L = np.where(den) * (v[-1] - v[0]) != 0, v[0] + (j / den) * (v[-1] - v[0]), np.nan)
            dev = v - L
            s_dev = float(np.sum(dev))
            s_abs = float(np.sum(np.abs(dev)))
            return s_dev / (s_abs + _EPS)

        return frame_like(x, map_rolling(xv, w, _fn))


@register_operator(
    name="ts_max_chord_excursion",
    category="time_series_shape",
    business_category="time_series_shape",
    canonical="ts_max_chord_excursion",
    source="alpha_language_shape",
)
class TsMaxChordExcursion(SeriesOperator):
    """最大弦偏移: max_j|x_j - L_j| / (Σ|Δx| + eps)。

    衡量中间曾偏离首尾最终路径多远（相对路径总长度）。对假突破、深回撤后修复、
    V 型路径敏感。要求窗口内全 finite。
    """

    metadata = _metadata(
        "ts_max_chord_excursion",
        "相对首尾连线的最大偏移（按路径长度归一）。",
        ["x", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        xv = x.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            if chunk.size < w or not np.all(np.isfinite(chunk)):
                return np.nan
            v = chunk.astype(float)
            if v.size < mp:
                return np.nan
            j = np.arange(v.size, dtype=float)
            den = max(v.size - 1, 1)
            L = np.where(den) * (v[-1] - v[0]) != 0, v[0] + (j / den) * (v[-1] - v[0]), np.nan)
            mce = float(np.max(np.abs(v - L)))
            path = float(np.sum(np.abs(np.diff(v))))
            return mce / (path + _EPS)

        return frame_like(x, map_rolling(xv, w, _fn))


_NEW_CANONICALS = (
    "ts_monotonicity",
    "ts_turning_rate",
    "ts_effective_turning_rate",
    "ts_turning_intensity",
    "ts_path_efficiency",
    "ts_roughness",
    "ts_trend_break_score",
    "ts_weighted_time_centroid",
    "ts_endpoint_deviation",
    "ts_mass_concentration",
    "ts_chord_excursion_area",
    "ts_max_chord_excursion",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
