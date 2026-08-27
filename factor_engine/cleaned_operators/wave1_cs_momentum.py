# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: cross-sectional momentum / liquidity-adjusted.

New canonicals across genuinely new thematic ground:
  * rank / volume-adjusted momentum family
  * cross-sectional momentum strength / stability
  * cross-correlation momentum (own return vs market return lead-lag)
  * liquidity-adjusted return decomposition

All are real pandas_numpy implementations, causal, deterministic and NaN-safe.
Names are prefixed ``m1_`` / ``vax_`` and are globally unique.
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


# ---------------------------------------------------------------------------
# 1. rank / volume-adjusted momentum family
# ---------------------------------------------------------------------------
@register_operator(
    name="m1_ranked_momentum",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_ranked_momentum",
    source="wave1_cs_momentum",
    status="experimental")
class M1RankedMomentum(SeriesOperator):
    """排名动量：window 总收益的横截面百分位排名（0..1）。

    与原始收益动量（ts_ret）不同，排名把跨截面的量纲消除，输出稳健的
    相对动量强度。每日一行横截面。
    """

    metadata = _metadata(
        "m1_ranked_momentum",
        "窗口累计收益的横截面百分位排名（相对动量）。",
        ["ret", "window", "min_periods", "min_breadth"],
        domain="price_volume",
        unit="dimensionless",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 10, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        mb = _check_int(min_breadth, "min_breadth", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            lo = max(0, row - w + 1)
            chunk = rv[lo:row + 1, :]
            mom = np.full(cols, np.nan)
            for col in range(cols):
                v = chunk[:, col]
                ok = np.isfinite(v)
                if ok.sum() < mp:
                    continue
                mom[col] = float(np.prod(1.0 + v[ok]) - 1.0)
            fin = np.isfinite(mom)
            if fin.sum() < mb:
                continue
            rnk = pd.Series(mom[fin]).rank(pct=True).to_numpy()
            out[row, fin] = rnk
        return _frame_like(ret, out)


@register_operator(
    name="m1_volume_adjusted_momentum",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_volume_adjusted_momentum",
    source="wave1_cs_momentum",
    status="experimental")
class M1VolumeAdjustedMomentum(SeriesOperator):
    """量调整动量：Σ(ret_t × turn_t) / Σ(turn_t)（成交额加权收益）。

    收益经换手率加权，高换手时段的收益对动量贡献更大（volume clock 动量）。
    输出窗口加权收益（return 单位）。
    """

    metadata = _metadata(
        "m1_volume_adjusted_momentum",
        "Σ(ret×turn)/Σ(turn)，成交额加权动量。",
        ["ret", "turnover", "window", "min_periods"],
        domain="price_volume",
        unit="same_as:ret",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, turnover: pd.DataFrame, window: int = 20, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ret, turnover = _aligned(ret, turnover)
        rv = ret.to_numpy(dtype=float)
        tv = turnover.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                t = tv[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(t) & (t > 0)
                if ok.sum() < mp:
                    continue
                den = float(np.sum(t[ok]))
                if den <= 0:
                    continue
                out[row, col] = float(np.sum(r[ok] * t[ok]) / den)
        return _frame_like(ret, out)


@register_operator(
    name="m1_momentum_strength",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_momentum_strength",
    source="wave1_cs_momentum",
    status="experimental")
class M1MomentumStrength(SeriesOperator):
    """动量强度：|window 累计收益| / 窗口每日收益 std（动量信噪比）。

    高 = 趋势强劲（收益一致），低 = 噪声主导。输出 ratio。
    """

    metadata = _metadata(
        "m1_momentum_strength",
        "|总收益| / std(日收益)（动量信噪比）。",
        ["ret", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 10, **_: Any) -> pd.DataFrame:
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
                total = float(np.prod(1.0 + vals) - 1.0)
                s = float(np.std(vals, ddof=1))
                if s <= 1e-12:
                    out[row, col] = 0.0 if total == 0 else np.nan
                    continue
                out[row, col] = abs(total) / s
        return _frame_like(ret, out)


@register_operator(
    name="m1_momentum_stability",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_momentum_stability",
    source="wave1_cs_momentum",
    status="experimental")
class M1MomentumStability(SeriesOperator):
    """动量稳定性：窗口内正收益日占比（趋势一致性）。

    连续同向（一致）动量 = 稳定趋势；交替 = 不可靠。输出 ratio。
    """

    metadata = _metadata(
        "m1_momentum_stability",
        "窗口内正收益日占比（动量一致性）。",
        ["ret", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 8, **_: Any) -> pd.DataFrame:
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
                out[row, col] = float(np.mean(vals > 0))
        return _frame_like(ret, out)


# ---------------------------------------------------------------------------
# 2. cross-sectional momentum speed / curvature
# ---------------------------------------------------------------------------
@register_operator(
    name="m1_momentum_speed_change",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_momentum_speed_change",
    source="wave1_cs_momentum",
    status="experimental")
class M1MomentumSpeedChange(SeriesOperator):
    """动量速度变化：两个嵌套窗口累计收益之差（加速度）。

    mom(fast) - mom(slow) 度量动量本身在加速（>0）还是衰减（<0）。输出
    return 单位。
    """

    metadata = _metadata(
        "m1_momentum_speed_change",
        "快窗累计收益 - 慢窗累计收益（动量加速度）。",
        ["ret", "fast_window", "slow_window", "min_periods"],
        domain="price_volume",
        unit="same_as:ret",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, fast_window: int = 5, slow_window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        fw = _check_int(fast_window, "fast_window", 2)
        sw = _check_int(slow_window, "slow_window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if fw >= sw:
            raise ValueError("fast_window must be < slow_window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(rv[row, col]):
                    continue
                lo_s = max(0, row - sw + 1)
                slow_r = rv[lo_s:row + 1, col]
                ok_s = np.isfinite(slow_r)
                if ok_s.sum() < mp:
                    continue
                mom_slow = float(np.prod(1.0 + slow_r[ok_s]) - 1.0)
                lo_f = max(0, row - fw + 1)
                fast_r = rv[lo_f:row + 1, col]
                ok_f = np.isfinite(fast_r)
                if ok_f.sum() < 2:
                    continue
                mom_fast = float(np.prod(1.0 + fast_r[ok_f]) - 1.0)
                out[row, col] = mom_fast - mom_slow
        return _frame_like(ret, out)


@register_operator(
    name="m1_cs_momentum_dispersion",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_cs_momentum_dispersion",
    source="wave1_cs_momentum",
    status="experimental")
class M1CsMomentumDispersion(SeriesOperator):
    """横截面动量离散度：每日窗口动量的横截面 std（市场动量宽度）。

    高 = 个股动量分化大；低 = 市场同涨同跌。每日一行。
    """

    metadata = _metadata(
        "m1_cs_momentum_dispersion",
        "横截面窗口动量的 std（动量宽度）。",
        ["ret", "window", "min_periods", "min_breadth"],
        domain="price_volume",
        unit="return",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, min_periods: int = 10, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        mb = _check_int(min_breadth, "min_breadth", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            lo = max(0, row - w + 1)
            chunk = rv[lo:row + 1, :]
            mom: list[float] = []
            for col in range(cols):
                v = chunk[:, col]
                ok = np.isfinite(v)
                if ok.sum() < mp:
                    continue
                mom.append(float(np.prod(1.0 + v[ok]) - 1.0))
            if len(mom) < mb:
                continue
            out[row, :] = float(np.std(mom))
        return _frame_like(ret, out)


@register_operator(
    name="m1_rank_momentum_gap",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_rank_momentum_gap",
    source="wave1_cs_momentum",
    status="experimental")
class M1RankMomentumGap(SeriesOperator):
    """排名动量落差：快窗动量排名 - 慢窗动量排名（横截面加速度）。

    正 = 相对动量加速上升（被市场重新定价），负 = 减速。输出 dimensionless
    （排名差）。
    """

    metadata = _metadata(
        "m1_rank_momentum_gap",
        "快窗动量横截面排名 - 慢窗动量横截面排名。",
        ["ret", "fast_window", "slow_window", "min_periods", "min_breadth"],
        domain="price_volume",
        unit="dimensionless",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, fast_window: int = 5, slow_window: int = 20, min_periods: int = 3, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        fw = _check_int(fast_window, "fast_window", 2)
        sw = _check_int(slow_window, "slow_window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        mb = _check_int(min_breadth, "min_breadth", 3)
        if fw >= sw:
            raise ValueError("fast_window must be < slow_window")
        rv = ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            lo_s = max(0, row - sw + 1)
            slow_chunk = rv[lo_s:row + 1, :]
            lo_f = max(0, row - fw + 1)
            fast_chunk = rv[lo_f:row + 1, :]
            feat_f = np.full(cols, np.nan)
            feat_s = np.full(cols, np.nan)
            for col in range(cols):
                vf = fast_chunk[:, col]
                of_ = np.isfinite(vf)
                if of_.sum() >= 2:
                    feat_f[col] = float(np.prod(1.0 + vf[of_]) - 1.0)
                vs = slow_chunk[:, col]
                os_ = np.isfinite(vs)
                if os_.sum() >= mp:
                    feat_s[col] = float(np.prod(1.0 + vs[os_]) - 1.0)
            fin = np.isfinite(feat_f) & np.isfinite(feat_s)
            if fin.sum() < mb:
                continue
            rf = pd.Series(feat_f[fin]).rank(pct=True).to_numpy()
            rs = pd.Series(feat_s[fin]).rank(pct=True).to_numpy()
            out[row, fin] = rf - rs
        return _frame_like(ret, out)


# ---------------------------------------------------------------------------
# 3. cross-correlation momentum (lead-lag) family
# ---------------------------------------------------------------------------
@register_operator(
    name="m1_corr_momentum",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_corr_momentum",
    source="wave1_cs_momentum",
    status="experimental")
class M1CorrMomentum(SeriesOperator):
    """自身收益与市场收益的相关动量：滚动 corr(ret_i, market_ret) 的斜率。

    个股与市场的相关趋势：>0 = 协同增强；<0 = 与市场脱钩（独立 alpha）。
    输出 dimensionless。
    """

    metadata = _metadata(
        "m1_corr_momentum",
        "corr(ret_i, market_ret) 的滚动斜率（相关动量）。",
        ["ret", "market_ret", "window", "corr_window", "min_periods"],
        domain="price_volume",
        unit="dimensionless",
        cost=3,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, market_ret: pd.DataFrame, window: int = 60, corr_window: int = 20, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        cw = _check_int(corr_window, "corr_window", 3)
        mp = _check_int(min_periods, "min_periods", 5)
        ret, market_ret = _aligned(ret, market_ret)
        rv = ret.to_numpy(dtype=float)
        mv = market_ret.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            corr_cache: list[float] = []
            for row in range(rows):
                lo = max(0, row - cw + 1)
                r = rv[lo:row + 1, col]
                m = mv[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(m)
                c: float | None = None
                if ok.sum() >= mp:
                    x = r[ok]
                    y = m[ok]
                    if float(np.std(x, ddof=1)) > 1e-12 and float(np.std(y, ddof=1)) > 1e-12:
                        c = float(np.corrcoef(x, y)[0, 1])
                if c is not None:
                    corr_cache.append(c)
                if len(corr_cache) >= 3:
                    arr = np.array(corr_cache[-w:])
                    if arr.size >= 3:
                        t = np.arange(arr.size, dtype=float)
                        den = float(np.sum((t - t.mean()) ** 2))
                        if den > 1e-12:
                            out[row, col] = float(np.sum((t - t.mean()) * (arr - arr.mean())) / den)
        return _frame_like(ret, out)


@register_operator(
    name="m1_momentum_regime",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="m1_momentum_regime",
    source="wave1_cs_momentum",
    status="experimental")
class M1MomentumRegime(SeriesOperator):
    """动量 regime：窗口累计收益符号化强度分档（>0 动量，<0 反转）。

    输出三态：1（正动量）、-1（负动量）、0（中性/低信号）。作为 condition/
    state 输入使用。
    """

    metadata = _metadata(
        "m1_momentum_regime",
        "窗口动量三态 regime（1 正 / -1 负 / 0 中性）。",
        ["ret", "window", "threshold", "min_periods"],
        domain="price_volume",
        unit="signed_ratio",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, window: int = 20, threshold: float = 0.02, min_periods: int = 8, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        th = float(threshold)
        if th < 0:
            raise ValueError("threshold must be >= 0")
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
                mom = float(np.prod(1.0 + r[ok]) - 1.0)
                if mom > th:
                    out[row, col] = 1.0
                elif mom < -th:
                    out[row, col] = -1.0
                else:
                    out[row, col] = 0.0
        return _frame_like(ret, out)


# ---------------------------------------------------------------------------
# 4. liquidity-adjusted return decomposition
# ---------------------------------------------------------------------------
@register_operator(
    name="vax_liquidity_adjusted_return",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="vax_liquidity_adjusted_return",
    source="wave1_cs_momentum",
    status="experimental")
class VaxLiquidityAdjustedReturn(SeriesOperator):
    """流动性调整收益：ret - sqrt(amihud) × 流动性惩罚系数。

    amihud 为单位成交额的绝对收益。若 amihud 高（不流动）则惩罚收益，突出
    在流动环境下实现的收益。输出 same_as 单位。
    """

    metadata = _metadata(
        "vax_liquidity_adjusted_return",
        "ret - penalty × sqrt(amihud)（流动性调整收益）。",
        ["ret", "amihud", "penalty"],
        domain="price_volume",
        unit="same_as:ret",
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, amihud: pd.DataFrame, penalty: float = 1.0, **_: Any) -> pd.DataFrame:
        p = float(penalty)
        if p < 0:
            raise ValueError("penalty must be >= 0")
        ret, amihud = _aligned(ret, amihud)
        rv = ret.to_numpy(dtype=float)
        av = amihud.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                r = rv[row, col]
                a = av[row, col]
                if np.isfinite(r) and np.isfinite(a) and a >= 0:
                    out[row, col] = r - p * float(np.sqrt(a))
        return _frame_like(ret, out)


@register_operator(
    name="vax_liquidity_penalty_exposure",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="vax_liquidity_penalty_exposure",
    source="wave1_cs_momentum",
    status="experimental")
class VaxLiquidityPenaltyExposure(SeriesOperator):
    """流动性惩罚暴露：amihud 的滚动均值（流动性成本暴露）。

    作为风格中性化器：高值 = 持续不流动（高交易成本）。输出 ratio。
    """

    metadata = _metadata(
        "vax_liquidity_penalty_exposure",
        "窗口 amihud 均值（流动性成本暴露）。",
        ["amihud", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, amihud: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        av = amihud.to_numpy(dtype=float)
        rows, cols = av.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                a = av[lo:row + 1, col]
                ok = np.isfinite(a) & (a >= 0)
                if ok.sum() < mp:
                    continue
                out[row, col] = float(np.mean(a[ok]))
        return _frame_like(amihud, out)


@register_operator(
    name="vax_ret_per_liquidity_unit",
    category="cross_sectional_momentum",
    business_category="cross_sectional_momentum",
    canonical="vax_ret_per_liquidity_unit",
    source="wave1_cs_momentum",
    status="experimental")
class VaxRetPerLiquidityUnit(SeriesOperator):
    """单位流动性收益：窗口累计收益 / 窗口平均成交额（收益效率）。

    高 = 低成交额实现同等收益（流动性效率高）。输出 dimensionless。
    """

    metadata = _metadata(
        "vax_ret_per_liquidity_unit",
        "窗口累计收益 / 窗口平均成交额（单位流动性收益）。",
        ["ret", "amount", "window", "min_periods"],
        domain="price_volume",
        unit="dimensionless",
        cost=2,
        category="cross_sectional_momentum",
    )

    def _calculate_series(self, ret: pd.DataFrame, amount: pd.DataFrame, window: int = 20, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ret, amount = _aligned(ret, amount)
        rv = ret.to_numpy(dtype=float)
        av = amount.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                a = av[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(a) & (a > 0)
                if ok.sum() < mp:
                    continue
                mom = float(np.prod(1.0 + r[ok]) - 1.0)
                am = float(np.mean(a[ok]))
                if am <= 0:
                    continue
                out[row, col] = mom / am
        return _frame_like(ret, out)