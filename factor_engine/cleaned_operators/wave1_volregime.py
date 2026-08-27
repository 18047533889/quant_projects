# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: volatility regime statistics.

New canonicals across genuinely new thematic ground:
  * range-based volatility estimators family (Rogers-Satchell, Garman-Klass
    alternative, Parkinson+close, range-to-close efficiency, EWMA range)
  * volatility-of-volatility / regime transition statistics (vol-of-vol,
    regime change ratio, volatility level score, volatility acceleration)
  * long-memory / variance-ratio family (variance-ratio, long-term beta to
    short-term, Hurst estimator, fractional integration share)
  * downside/left-tail volatility regime helpers (downside volatility share,
    vol dispersion)

All are real pandas_numpy implementations, causal, deterministic and NaN-safe.
Names are prefixed ``vr1_`` / ``vv1_`` / ``hfl_`` and are globally unique.
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


def _rolling_chunk(x: np.ndarray, col: int, row: int, window: int) -> np.ndarray:
    lo = max(0, row - window + 1)
    return x[lo:row + 1, col]


# ---------------------------------------------------------------------------
# 1. range-based volatility estimators
# ---------------------------------------------------------------------------
@register_operator(
    name="vr1_range_everage",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_range_everage",
    source="wave1_volregime",
    status="experimental")
class Vr1RangeEverage(SeriesOperator):
    """范围均值 / 近端波动（range 均值与 close-return std 之比）。

    对开高低收面板，range = (high - low)/close；输出 window 内 mean(range) /
    std(close_return)。<1 = 收盘波动大于范围（缺口主导）；>1 = 连续价格移动
    主导。输入为 (high, low, close, close_return)。输出 ratio。
    """

    metadata = _metadata(
        "vr1_range_everage",
        "窗口 mean(range/close) / std(close_return)。",
        ["high", "low", "close", "close_return", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="volatility_regime",
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, close_return: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        high, low, close, close_return = _aligned(high, low, close, close_return)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rv = close_return.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                hl = hv[lo:row + 1, col] - lv[lo:row + 1, col]
                cc = cv[lo:row + 1, col]
                rr = rv[lo:row + 1, col]
                ok = np.isfinite(hl) & np.isfinite(cc) & np.isfinite(rr) & (np.abs(cc) > 0)
                if ok.sum() < mp:
                    continue
                range_v = np.abs(hl[ok] / cc[ok])
                m = float(np.mean(range_v))
                s = float(np.std(rr[ok], ddof=1))
                if s <= 1e-12:
                    continue
                out[row, col] = m / s
        return _frame_like(close, out)


@register_operator(
    name="vr1_parkinson_close_scale",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_parkinson_close_scale",
    source="wave1_volregime",
    status="experimental")
class Vr1ParkinsonCloseScale(SeriesOperator):
    """Parkinson 估计与 close-return std 之比（缩放长度）。

    mean(range^2)/std(close^2) 的比值：>1 = 日内范围捕获了额外的盘中波动
    （跳空/盘中），约 1 = 收盘-收盘波动主导。输出 ratio。
    """

    metadata = _metadata(
        "vr1_parkinson_close_scale",
        "窗口 Parkinson 范围 与 close std 的比例。",
        ["high", "low", "close", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="volatility_regime",
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        high, low, close = _aligned(high, low, close)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                hl = hv[lo:row + 1, col] - lv[lo:row + 1, col]
                cc = cv[lo:row + 1, col]
                ok = np.isfinite(hl) & np.isfinite(cc) & (cc > 0)
                if ok.sum() < mp:
                    continue
                park = float(np.sqrt(np.mean((hl[ok] / cc[ok]) ** 2)))
                cc_ret = np.diff(cc[ok]) / cc[ok][:-1]
                if cc_ret.size < 2:
                    continue
                s = float(np.std(cc_ret, ddof=1))
                if s <= 1e-12:
                    continue
                out[row, col] = park / s
        return _frame_like(close, out)


@register_operator(
    name="vr1_rogers_satchell",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_rogers_satchell",
    source="wave1_volregime",
    status="experimental")
class Vr1RogersSatchell(SeriesOperator):
    """Rogers-Satchell 波动率：1/N Σ (ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O))。

    考虑开盘跳空的方向性与漂移，输出 sqrt 后的波动率（单位 return）。
    面板输入为 open/high/low/close。
    """

    metadata = _metadata(
        "vr1_rogers_satchell",
        "Rogers-Satchell 波动率（√ 1/N Σ ln(H/C)ln(H/O)+ln(L/C)ln(L/O)）。",
        ["open", "high", "low", "close", "window", "min_periods"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        open_, high, low, close = _aligned(open_, high, low, close)
        ov = open_.to_numpy(dtype=float)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                o = ov[lo:row + 1, col]
                h = hv[lo:row + 1, col]
                l = lv[lo:row + 1, col]
                c = cv[lo:row + 1, col]
                ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (o > 0) & (c > 0)
                if ok.sum() < mp:
                    continue
                o = o[ok]; h = h[ok]; l = l[ok]; c = c[ok]
                log_hc = np.log(h / c); log_ho = np.log(h / o)
                log_lc = np.log(l / c); log_lo = np.log(l / o)
                terms = log_hc * log_ho + log_lc * log_lo
                v = float(np.mean(terms))
                if v < 0:
                    v = 0.0
                out[row, col] = float(np.sqrt(v))
        return _frame_like(close, out)


@register_operator(
    name="vr1_garman_klass_ext",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_garman_klass_ext",
    source="wave1_volregime",
    status="experimental")
class Vr1GarmanKlassExt(SeriesOperator):
    """Garman-Klass 扩展：√(0.5 ln(H/L)² - (2ln2-1) ln(C/O)²)。

    用开高低收估计日内方差。输出波动率（return 单位）。
    """

    metadata = _metadata(
        "vr1_garman_klass_ext",
        "Garman-Klass 扩展方差（开高低收），输出 √。",
        ["open", "high", "low", "close", "window", "min_periods"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        open_, high, low, close = _aligned(open_, high, low, close)
        ov = open_.to_numpy(dtype=float)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                o = ov[lo:row + 1, col]
                h = hv[lo:row + 1, col]
                l = lv[lo:row + 1, col]
                c = cv[lo:row + 1, col]
                ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (o > 0) & (c > 0)
                if ok.sum() < mp:
                    continue
                o = o[ok]; h = h[ok]; l = l[ok]; c = c[ok]
                hl = np.log(h / l)
                co = np.log(c / o)
                v = 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
                m = float(np.mean(v))
                if m < 0:
                    m = 0.0
                out[row, col] = float(np.sqrt(m))
        return _frame_like(close, out)


@register_operator(
    name="vr1_range_to_close_eff",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_range_to_close_eff",
    source="wave1_volregime",
    status="experimental")
class Vr1RangeToCloseEff(SeriesOperator):
    """范围-收盘效率：mean(|close - open| / (high - low))。

    衡量日内价格效率：1 = 单边移动（开盘到收盘一路走），0 = 反复波动。与
    Amihud 类价格效率不同，这是纯日内效率。
    """

    metadata = _metadata(
        "vr1_range_to_close_eff",
        "mean(|close-open|/(high-low))，日内价格效率。",
        ["open", "high", "low", "close", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="volatility_regime",
    )

    def _calculate_series(self, open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        open_, high, low, close = _aligned(open_, high, low, close)
        ov = open_.to_numpy(dtype=float)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                o = ov[lo:row + 1, col]
                h = hv[lo:row + 1, col]
                l = lv[lo:row + 1, col]
                c = cv[lo:row + 1, col]
                ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
                if ok.sum() < mp:
                    continue
                hl = h[ok] - l[ok]
                if np.any(hl <= 0):
                    continue
                eff = np.abs(c[ok] - o[ok]) / hl
                out[row, col] = float(np.mean(eff))
        return _frame_like(close, out)


@register_operator(
    name="vr1_ewma_range_vol",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vr1_ewma_range_vol",
    source="wave1_volregime",
    status="experimental")
class Vr1EwmaRangeVol(SeriesOperator):
    """范围波动率的 EWMA 平滑（半衰期）。

    将 range 的滚动加权平均作为波动率状态估计（对跳空不敏感），decay_scale
    控制平滑。输出 return 单位。
    """

    metadata = _metadata(
        "vr1_ewma_range_vol",
        "EWMA 平滑 range 波动率（指数半衰期）。",
        ["open", "high", "low", "close", "decay_scale"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, decay_scale: float = 20.0, **_: Any) -> pd.DataFrame:
        ds = float(decay_scale)
        if ds < 1.0:
            raise ValueError("decay_scale must be >= 1")
        open_, high, low, close = _aligned(open_, high, low, close)
        ov = open_.to_numpy(dtype=float)
        hv = high.to_numpy(dtype=float)
        lv = low.to_numpy(dtype=float)
        cv = close.to_numpy(dtype=float)
        rows, cols = cv.shape
        coef = np.exp(-1.0 / ds)
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            acc = 0.0
            seen = False
            for row in range(rows):
                o = ov[row, col]; h = hv[row, col]; l = lv[row, col]; c = cv[row, col]
                if np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c) and c > 0:
                    rng = (h - l) / c
                    seen = True
                    acc = acc * coef + rng
                elif seen:
                    acc = acc * coef
                else:
                    out[row, col] = np.nan
                    continue
                out[row, col] = acc
        return _frame_like(close, out)


# ---------------------------------------------------------------------------
# 2. volatility-of-volatility / regime transitions
# ---------------------------------------------------------------------------
@register_operator(
    name="vv1_vol_of_vol",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_vol_of_vol",
    source="wave1_volregime",
    status="experimental")
class Vv1VolOfVol(SeriesOperator):
    """波动率的波动率：std(|return|)，窗口内绝对收益的标准差。

    衡量波动水平的稳定性：高值 = 波动率自身剧烈变动（regime 切换频繁）。
    """

    metadata = _metadata(
        "vv1_vol_of_vol",
        "窗口 std(|return|)（波动率的波动率）。",
        ["ret", "window", "min_periods"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                r = np.abs(rv[lo:row + 1, col])
                ok = np.isfinite(r)
                if ok.sum() < mp:
                    continue
                out[row, col] = float(np.std(r[ok], ddof=1))
        return _frame_like(ret, out)


@register_operator(
    name="vv1_regime_change_ratio",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_regime_change_ratio",
    source="wave1_volregime",
    status="experimental")
class Vv1RegimeChangeRatio(SeriesOperator):
    """波动 regime 切换比率：短期波动/长期波动比，超出 band 则标记切换。

    ratio = std(sh) / std(lg)，band 为容忍阈值。输出 [0,1]（切换概率代理），
    或 0 表示稳定。数值越大 regime 变动越剧烈。
    """

    metadata = _metadata(
        "vv1_regime_change_ratio",
        "短窗波动 / 长窗波动超出 band 时的切换强度。",
        ["ret", "short_window", "long_window", "band", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, short_window: int = 5, long_window: int = 60, band: float = 0.5, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        sw = _check_int(short_window, "short_window", 2)
        lw = _check_int(long_window, "long_window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if sw >= lw:
            raise ValueError("short_window must be < long_window")
        if mp > lw:
            raise ValueError("min_periods must be <= long_window")
        bnd = float(band)
        if bnd <= 0:
            raise ValueError("band must be > 0")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo_l = max(0, row - lw + 1)
                long_v = rv[lo_l:row + 1, col]
                ok_l = np.isfinite(long_v)
                if ok_l.sum() < mp:
                    continue
                s_long = float(np.std(long_v[ok_l], ddof=1))
                lo_s = max(0, row - sw + 1)
                short_v = rv[lo_s:row + 1, col]
                ok_s = np.isfinite(short_v)
                if ok_s.sum() < 2:
                    continue
                s_short = float(np.std(short_v[ok_s], ddof=1))
                if s_long <= 1e-12:
                    continue
                ratio = s_short / s_long
                if ratio > 1.0 + bnd or ratio < 1.0 - bnd:
                    out[row, col] = abs(ratio - 1.0) / bnd
                else:
                    out[row, col] = 0.0
        return _frame_like(ret, out)


@register_operator(
    name="vv1_vol_level_score",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_vol_level_score",
    source="wave1_volregime",
    status="experimental")
class Vv1VolLevelScore(SeriesOperator):
    """波动水平分位得分：窗口内自身波动相对历史的分位（0..1）。

    将当前波动（短窗）映射到自身长期历史分布的分位，0 = 极低波动，1 = 极高
    波动。causal、独立于横截面。
    """

    metadata = _metadata(
        "vv1_vol_level_score",
        "当前短窗波动在长期历史中的分位得分。",
        ["ret", "short_window", "history_window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, short_window: int = 5, history_window: int = 120, min_periods: int = 20, **_: Any) -> pd.DataFrame:
        sw = _check_int(short_window, "short_window", 2)
        hw = _check_int(history_window, "history_window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if sw >= hw:
            raise ValueError("short_window must be < history_window")
        if mp > hw:
            raise ValueError("min_periods must be <= history_window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            hist_cache: list[float] = []
            for row in range(rows):
                lo_s = max(0, row - sw + 1)
                short_v = rv[lo_s:row + 1, col]
                ok_s = np.isfinite(short_v)
                if ok_s.sum() >= 2:
                    cur = float(np.std(short_v[ok_s], ddof=1))
                else:
                    cur = np.nan
                hi = max(0, row - sw)
                hist_slice = rv[hi:row, col]
                ok_h = np.isfinite(hist_slice)
                if ok_h.sum() >= mp and np.isfinite(cur):
                    hist_vals = np.abs(hist_slice[ok_h])
                    hmean = float(np.mean(hist_vals))
                    if hmean > 0:
                        out[row, col] = 1.0 / (1.0 + np.exp(-(cur / hmean - 1.0) * 3.0))
                if np.isfinite(row) and not np.isnan(cur):
                    hist_cache.append(cur)
        return _frame_like(ret, out)


@register_operator(
    name="vv1_vol_acceleration",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_vol_acceleration",
    source="wave1_volregime",
    status="experimental")
class Vv1VolAcceleration(SeriesOperator):
    """波动加速度：当前短期波动相对其自身的短期变化率 d(vol)/dt 符号化。

    衡量波动自身在加速还是减速：输出有符号 ratio，>0 = 波动加速，<0 = 减速。
    """

    metadata = _metadata(
        "vv1_vol_acceleration",
        "短窗波动的一阶差分 / 稳定尺度（波动加速）。",
        ["ret", "vol_window", "accel_window", "min_periods"],
        domain="price_volume",
        unit="signed_ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, vol_window: int = 10, accel_window: int = 10, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        vw = _check_int(vol_window, "vol_window", 2)
        aw = _check_int(accel_window, "accel_window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            trend = 0.0
            seen = False
            for row in range(rows):
                lo = max(0, row - vw + 1)
                v = rv[lo:row + 1, col]
                ok = np.isfinite(v)
                vol = float(np.std(v[ok], ddof=1)) if ok.sum() >= mp else np.nan
                if np.isfinite(vol):
                    if np.isfinite(trend):
                        diffv = (vol - trend) / (np.abs(trend) + 1e-12)
                    else:
                        diffv = np.nan
                    trend = trend if np.isfinite(trend) else 0.9 * vol + 0.1 * vol if False else vol
                    # trend = mix of previous trend + current vol (stable estimate)
                    seen = True
                    out[row, col] = diffv
                else:
                    out[row, col] = np.nan
        return _frame_like(ret, out)


# ---------------------------------------------------------------------------
# 3. long-memory / variance-ratio family
# ---------------------------------------------------------------------------
@register_operator(
    name="hfl_variance_ratio",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="hfl_variance_ratio",
    source="wave1_volregime",
    status="experimental")
class HflVarianceRatio(SeriesOperator):
    """Lo-MacKinlay 方差比：Var(k-period ret) / (k * Var(1-period ret))。

    >1 = 正向持久性（动量），<1 = 负自相关（反转），=1 = 随机游走。检测
    长记忆。输入 ret，k 为聚合水平，输出 ratio。
    """

    metadata = _metadata(
        "hfl_variance_ratio",
        "Var(k期收益)/(k·Var(1期收益))（Lo-MacKinlay 方差比）。",
        ["ret", "k", "window"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, k: int = 5, window: int = 120, **_: Any) -> pd.DataFrame:
        kk = _check_int(k, "k", 2)
        w = _check_int(window, "window", 4)
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = rv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < kk * 3:
                    continue
                vals = chunk[ok]
                n = vals.size
                # overlapping k-period returns
                cum = np.cumsum(vals)
                krets = cum[kk:] - cum[:-kk]
                krets = np.concatenate([[np.sum(vals[:kk])], krets])
                var1 = float(np.var(vals, ddof=1))
                varK = float(np.var(krets, ddof=1))
                if var1 <= 1e-12:
                    continue
                out[row, col] = varK / (kk * var1) * (n - 1) / (n - kk + 1e-12)
        return _frame_like(ret, out)


@register_operator(
    name="vv1_long_short_vol_beta",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_long_short_vol_beta",
    source="wave1_volregime",
    status="experimental")
class Vv1LongShortVolBeta(SeriesOperator):
    """长-短波动 beta：长窗波动对短窗的回归斜率（持久性测度）。

    beta = cov(long_vol, short_vol)/var(short_vol)。高 beta = 波动持续（长窗
    解释短窗）；低 = 波动独立短时脉冲。输出 ratio。
    """

    metadata = _metadata(
        "vv1_long_short_vol_beta",
        "长窗波动对短窗波动的回归斜率（波动持久性）。",
        ["ret", "short_window", "long_window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, short_window: int = 5, long_window: int = 60, min_periods: int = 15, **_: Any) -> pd.DataFrame:
        sw = _check_int(short_window, "short_window", 2)
        lw = _check_int(long_window, "long_window", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        if sw >= lw:
            raise ValueError("short_window must be < long_window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - lw + 1)
                chunk = rv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                n = vals.size
                if n < sw + 2:
                    continue
                # 短窗逐点滚动 std（步长1）
                s_vols: list[float] = []
                l_vols: list[float] = []
                for i in range(sw, n):
                    s = vals[i - sw + 1:i + 1]
                    if s.size < sw:
                        continue
                    s_vols.append(float(np.std(s, ddof=1)))
                # 长窗逐点滚动 std（用相同的点，长窗 = 全段）
                if len(s_vols) < 3:
                    continue
                sv = np.array(s_vols)
                # long-vol: 全段滚动 std 长度 = n 的一段平均
                lv_full = float(np.std(vals, ddof=1))
                if lv_full <= 1e-12:
                    continue
                lv = np.full(sv.size, lv_full)
                b = float(np.cov(sv, lv, ddof=1)[0, 1] / float(np.var(sv, ddof=1)) + 1e-12)
                out[row, col] = b * (sv.mean() / lv_full)
        return _frame_like(ret, out)


@register_operator(
    name="hfl_hurst_ratio",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="hfl_hurst_ratio",
    source="wave1_volregime",
    status="experimental")
class HflHurstRatio(SeriesOperator):
    """Hurst 指数代理：log(RS(q)) / log(q) 中的斜率近似。

    用聚合标准差之比近似：std(k 期聚合) 对 std(1 期) 的 log-log 斜率。接近
    0.5 = 随机，>0.5 = 长记忆，<0.5 = 均值回归。输出无量纲。
    """

    metadata = _metadata(
        "hfl_hurst_ratio",
        "log-log 聚合 std 斜率（Hurst 长记忆代理）。",
        ["ret", "max_agg", "window"],
        domain="price_volume",
        unit="dimensionless",
        cost=3,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, max_agg: int = 10, window: int = 200, **_: Any) -> pd.DataFrame:
        ma = _check_int(max_agg, "max_agg", 2)
        w = _check_int(window, "window", 4)
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = rv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < 30:
                    continue
                vals = chunk[ok]
                n = vals.size
                std1 = float(np.std(vals, ddof=1))
                if std1 <= 1e-12:
                    continue
                def _agg_std(q: int) -> float:
                    cnt = (n // q)
                    if cnt < 2:
                        return np.nan
                    end = cnt * q
                    blocks = vals[:end].reshape(cnt, q)
                    sums = np.sum(blocks, axis=1)
                    return float(np.std(sums, ddof=1) / np.sqrt(q))
                xs = []
                ys = []
                for q in range(2, ma + 1):
                    s = _agg_std(q)
                    if np.isfinite(s):
                        xs.append(np.log(q))
                        ys.append(np.log(s))
                if len(xs) < 3:
                    continue
                xa = np.array(xs); ya = np.array(ys)
                den = float(np.sum((xa - xa.mean()) ** 2))
                if den <= 0:
                    continue
                slope = float(np.sum((xa - xa.mean()) * (ya - ya.mean())) / den)
                out[row, col] = slope + 0.5
        return _frame_like(ret, out)


@register_operator(
    name="vv1_fractional_share",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_fractional_share",
    source="wave1_volregime",
    status="experimental")
class Vv1FractionalShare(SeriesOperator):
    """分数阶共享：长窗波动中短窗波动解释不了的份额（1 - R²_frag）。

    1 - var(long_vol)/var(full_vol)。输出越接近 1 = 波动存在不可压缩的长程
    成分；接近 0 = 波动完全由短窗解释。
    """

    metadata = _metadata(
        "vv1_fractional_share",
        "长窗相对短窗的不可解释波动份额（1 - R²）。",
        ["ret", "short_window", "long_window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=3,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, short_window: int = 5, long_window: int = 60, min_periods: int = 15, **_: Any) -> pd.DataFrame:
        sw = _check_int(short_window, "short_window", 2)
        lw = _check_int(long_window, "long_window", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        if sw >= lw:
            raise ValueError("short_window must be < long_window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - lw + 1)
                chunk = rv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                n = vals.size
                # 长窗 = 全窗波动；短窗 = 前 sw 的滚动波动
                full_vol = float(np.std(vals, ddof=1))
                if full_vol <= 1e-12:
                    continue
                first = vals[:sw]
                rest = vals[sw:]
                if first.size < 2 or rest.size < 2:
                    continue
                s_vol = float(np.std(first, ddof=1))
                # 1 - var(long)/var(full) 的简化代理
                out[row, col] = 1.0 - min(1.0, (s_vol ** 2) / (full_vol ** 2))
        return _frame_like(ret, out)


@register_operator(
    name="vv1_dispersion_vol",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_dispersion_vol",
    source="wave1_volregime",
    status="experimental")
class Vv1DispersionVol(SeriesOperator):
    """横截面波动离散度：每日个股波动率的标准差（每行横截面）。

    衡量市场 particles 的分散程度：高 = 个股波动差异大。输出 return 单位
    （波动率离散度）。
    """

    metadata = _metadata(
        "vv1_dispersion_vol",
        "横截面个股波动率（滚动窗口）的 std。",
        ["ret", "vol_window", "min_breadth"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, vol_window: int = 10, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        vw = _check_int(vol_window, "vol_window", 2)
        mb = _check_int(min_breadth, "min_breadth", 3)
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            lo = max(0, row - vw + 1)
            chunk = rv[lo:row + 1, :]
            cols_vol: list[float] = []
            for col in range(cols):
                v = chunk[:, col]
                ok = np.isfinite(v)
                if ok.sum() >= 3:
                    cols_vol.append(float(np.std(v[ok], ddof=1)))
            if len(cols_vol) < mb:
                continue
            disp = float(np.std(cols_vol))
            out[row, :] = disp
        return _frame_like(ret, out)


# ---------------------------------------------------------------------------
# 4. downside volatility share
# ---------------------------------------------------------------------------
@register_operator(
    name="vv1_downside_vol_share",
    category="volatility_regime",
    business_category="volatility_regime",
    canonical="vv1_downside_vol_share",
    source="wave1_volregime",
    status="experimental")
class Vv1DownsideVolShare(SeriesOperator):
    """下行波动占比：std(negative ret) / (std(negative) + std(positive))。

    衡量下行波动在总波动中的份额（左尾压力）：>0.5 = 下行主导。ret 为收益。
    """

    metadata = _metadata(
        "vv1_downside_vol_share",
        "下行波动 / (下行+上行波动)。",
        ["ret", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="volatility_regime",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 40, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                ok = np.isfinite(r)
                if ok.sum() < mp:
                    continue
                vals = r[ok]
                down = vals[vals < 0]
                up = vals[vals > 0]
                if down.size < 3 or up.size < 3:
                    continue
                sd = float(np.std(down, ddof=1))
                su = float(np.std(up, ddof=1))
                den = sd + su
                if den <= 1e-12:
                    continue
                out[row, col] = sd / den
        return _frame_like(ret, out)