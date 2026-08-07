# -*- coding: utf-8 -*-
"""Alpha-language volatility-structure operators (2026-08).

Beyond the plain volatility level these describe *how* volatility is built:

* vol of vol: volatility of (log) volatility;
* vol acceleration / term structure: volatility regime change and term slope;
* semivariance balance / realized quarticity: upside-vs-downside variance and
  tail-driven concentration of realized variance;
* vol clustering: lag-1 autocorrelation of |return|;
* leverage effect: return vs lagged-volatility correlation;
* jump / bipower proxy: daily realized bipower jump ratio.

All operators are causal trailing transforms (only rows ``<= t``).  Windows with
too few finite observations or zero variance return NaN, never Inf.
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
    valid_values,
)

_EPS = 1e-12
_MAD_CONST = 1.4826


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_volatility",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_volatility", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:volatility",
            f"unit:{unit}", "cost:1",
        ],
    )


def _std(values: np.ndarray, min_periods: int) -> float:
    v = valid_values(values)
    if v.size < min_periods:
        return np.nan
    return float(np.std(v))


def _corr(a: np.ndarray, b: np.ndarray, min_periods: int) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < min_periods:
        return np.nan
    x = a[finite]
    y = b[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


@register_operator(
    name="ts_vol_of_vol",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_of_vol",
    source="alpha_language_volatility",
)
class TsVolOfVol(SeriesOperator):
    """波动率的波动率: std(log(inner_vol + eps)) over outer_window。

    用 log vol 降低水平效应。inner_vol = trailing inner_window 内 ret 的 std。
    """

    metadata = _metadata(
        "ts_vol_of_vol",
        "log 波动率的外层 std。",
        ["ret", "inner_window", "outer_window", "min_periods"],
        unit="log",
    )

    def _calculate_series(
        self, ret: pd.DataFrame, inner_window: int = 5, outer_window: int = 40, min_periods: int = 2, **_: Any
    ) -> pd.DataFrame:
        wi = check_window(inner_window, name="inner_window")
        wo = check_window(outer_window, name="outer_window")
        mp = max(2, int(min_periods))
        rv = ret.to_numpy(dtype=float)
        inner = map_rolling(rv, wi, lambda c: _std(c, mp))
        logv = np.log(inner + _EPS)
        outer = map_rolling(logv, wo, lambda c: _std(c, mp))
        return frame_like(ret, outer)


@register_operator(
    name="ts_vol_acceleration",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_acceleration",
    source="alpha_language_volatility",
)
class TsVolAcceleration(SeriesOperator):
    """波动加速度: log((v_t + eps) / (v_{t-lag} + eps))。正 = 波动升温, 负 = 降温。"""

    metadata = _metadata(
        "ts_vol_acceleration",
        "当前波动 vs lag 前波动的对数比。",
        ["ret", "inner_window", "lag", "min_periods"],
        unit="log",
    )

    def _calculate_series(
        self, ret: pd.DataFrame, inner_window: int = 5, lag: int = 5, min_periods: int = 2, **_: Any
    ) -> pd.DataFrame:
        wi = check_window(inner_window, name="inner_window")
        la = int(lag)
        if la < 1:
            raise ValueError("lag must be >= 1")
        mp = max(2, int(min_periods))
        rv = ret.to_numpy(dtype=float)
        v = map_rolling(rv, wi, lambda c: _std(c, mp))
        rows, cols = v.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(la, rows):
                if not np.isfinite(v[row, col]) or not np.isfinite(v[row - la, col]):
                    continue
                out[row, col] = float(np.log((v[row, col] + _EPS) / (v[row - la, col] + _EPS)))
        return frame_like(ret, out)


@register_operator(
    name="ts_vol_term_structure",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_term_structure",
    source="alpha_language_volatility",
)
class TsVolTermStructure(SeriesOperator):
    """波动率期限结构: log((vol_short + eps) / (vol_long + eps))。要求 short < long。"""

    metadata = _metadata(
        "ts_vol_term_structure",
        "短窗波动 vs 长窗波动对数比。",
        ["ret", "short_window", "long_window", "min_periods"],
        unit="log",
    )

    def _calculate_series(
        self, ret: pd.DataFrame, short_window: int = 5, long_window: int = 40, min_periods: int = 2, **_: Any
    ) -> pd.DataFrame:
        ws = check_window(short_window, name="short_window")
        wl = check_window(long_window, name="long_window")
        if ws >= wl:
            raise ValueError("short_window must be < long_window")
        mp = max(2, int(min_periods))
        rv = ret.to_numpy(dtype=float)
        short = map_rolling(rv, ws, lambda c: _std(c, mp))
        longv = map_rolling(rv, wl, lambda c: _std(c, mp))
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(short[row, col]) or not np.isfinite(longv[row, col]):
                    continue
                out[row, col] = float(np.log((short[row, col] + _EPS) / (longv[row, col] + _EPS)))
        return frame_like(ret, out)


@register_operator(
    name="ts_semivariance_balance",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_semivariance_balance",
    source="alpha_language_volatility",
)
class TsSemivarianceBalance(SeriesOperator):
    """半方差平衡: (SV+ - SV-) / (SV+ + SV- + eps), 范围 [-1,1]。

    SV+ = sum r^2 I(r>0), SV- = sum r^2 I(r<0)。正 = 上行波动主导。
    """

    metadata = _metadata(
        "ts_semivariance_balance",
        "上行半方差 vs 下行半方差失衡。",
        ["ret", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(2, int(min_periods))
        rv = ret.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            if v.size < mp:
                return np.nan
            pos = float(np.sum(v[v > 0.0] ** 2))
            neg = float(np.sum(v[v < 0.0] ** 2))
            denom = pos + neg
            if denom < _EPS:
                return np.nan
            return float((pos - neg) / denom)

        return frame_like(ret, map_rolling(rv, w, _fn))


@register_operator(
    name="ts_realized_quarticity",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_realized_quarticity",
    source="alpha_language_volatility",
)
class TsRealizedQuarticity(SeriesOperator):
    """已实现四次幂比: W*sum r^4 / (3*RV^2 + eps)。

    高 = 波动由极端日集中贡献(W 取窗口内有效观测数)。
    """

    metadata = _metadata(
        "ts_realized_quarticity",
        "窗口内 r^4 相对 RV^2 之比。",
        ["ret", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        rv = ret.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            n = v.size
            if n < mp:
                return np.nan
            rv2 = float(np.sum(v * v))
            if rv2 < _EPS:
                return np.nan
            rv4 = float(np.sum(v ** 4))
            return float(n * rv4 / (3.0 * rv2 * rv2 + _EPS))

        return frame_like(ret, map_rolling(rv, w, _fn))


@register_operator(
    name="ts_vol_clustering",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_clustering",
    source="alpha_language_volatility",
)
class TsVolClustering(SeriesOperator):
    """波动聚集: |ret| 的滞后 1 阶自相关。正 = 大波动倾向连续出现。"""

    metadata = _metadata(
        "ts_vol_clustering",
        "|ret| 滞后 1 自相关。",
        ["ret", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        rv = np.abs(ret.to_numpy(dtype=float))

        def _fn(chunk: np.ndarray) -> float:
            a = chunk[:-1]
            b = chunk[1:]
            return _corr(a, b, mp)

        return frame_like(ret, map_rolling(rv, w, _fn))


@register_operator(
    name="ts_leverage_effect",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_leverage_effect",
    source="alpha_language_volatility",
)
class TsLeverageEffect(SeriesOperator):
    """杠杆效应: corr(r_tau, v_{tau-1}) over window, v = trailing 波动。

    负 = 收益下跌伴随波动上行(经典杠杆效应)。
    """

    metadata = _metadata(
        "ts_leverage_effect",
        "收益与滞后波动自相关。",
        ["ret", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 60, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        rv = ret.to_numpy(dtype=float)
        vol = map_rolling(rv, w, lambda c: _std(c, 2))
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(1, row - w + 1)
                a = rv[start : row + 1, col]
                b = vol[start - 1 : row, col]  # v_{tau-1} aligned to r_tau
                out[row, col] = _corr(a, b, mp)
        return frame_like(ret, out)


@register_operator(
    name="ts_jump_bipower_proxy",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_jump_bipower_proxy",
    source="alpha_language_volatility",
)
class TsJumpBipowerProxy(SeriesOperator):
    """日频跳跃代理: max(RV - BV, 0) / (RV + eps), 范围 [0,1]。

    BV = (pi/2) * sum |r_tau||r_{tau-1}|。高 = 波动主要来自离散跳跃而非连续路径。
    """

    metadata = _metadata(
        "ts_jump_bipower_proxy",
        "日频 bipower 跳跃占比。",
        ["ret", "window", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = check_window(window)
        mp = max(3, int(min_periods))
        rv = ret.to_numpy(dtype=float)

        def _fn(chunk: np.ndarray) -> float:
            v = valid_values(chunk)
            n = v.size
            if n < mp:
                return np.nan
            rv2 = float(np.sum(v * v))
            if rv2 < _EPS:
                return np.nan
            bv = float(np.sum(np.abs(v[1:]) * np.abs(v[:-1])))
            bv = (np.pi / 2.0) * bv
            return float(max(rv2 - bv, 0.0) / (rv2 + _EPS))

        return frame_like(ret, map_rolling(rv, w, _fn))


_NEW_CANONICALS = (
    "ts_vol_of_vol",
    "ts_vol_acceleration",
    "ts_vol_term_structure",
    "ts_semivariance_balance",
    "ts_realized_quarticity",
    "ts_vol_clustering",
    "ts_leverage_effect",
    "ts_jump_bipower_proxy",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
