# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: earnings stability / accrual quality.

New canonicals across genuinely new thematic ground:
  * earnings stability family (coefficient of variation, negative-earnings
    streak, revenue-earnings decoupling, earnings smoothness proxy)
  * accrual-quality family (working-capital accrual rate, accrual stability,
    abnormal accrual proxy, cash-conversion strength, cash-flow volatility,
    accrual ratio dispersion)
  * earnings-persistence family (earnings autocorrelation, ROA stability
    complement, earnings-consistency score, earnings-surprise decay)

All are real pandas_numpy implementations (causal daily-panel transforms on a
PIT financial panel), deterministic, NaN per family convention (window outputs
need the current row finite; ratios inherit sign).  Names are prefixed
``es1_`` / ``aq1_`` / ``ep1_`` and are globally unique.
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
# 1. earnings stability family
# ---------------------------------------------------------------------------
@register_operator(
    name="es1_earnings_cv",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="es1_earnings_cv",
    source="wave1_earnings",
    status="experimental")
class Es1EarningsCv(SeriesOperator):
    """盈利变异系数：std(earnings)/|mean(earnings)| 的稳定度。

    数值小 = 盈利稳定（质量高），大 = 波动剧烈。对含负值的盈利序列用
    |mean| 处理分母。输出 ratio。
    """

    metadata = _metadata(
        "es1_earnings_cv",
        "std(earnings)/|mean(earnings)|（盈利稳定度）。",
        ["earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(ev[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1))
                den = abs(m)
                if den <= 1e-12:
                    out[row, col] = 0.0
                    continue
                out[row, col] = s / den
        return _frame_like(earnings, out)


@register_operator(
    name="es1_negative_earnings_streak",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="es1_negative_earnings_streak",
    source="wave1_earnings",
    status="experimental")
class Es1NegativeEarningsStreak(SeriesOperator):
    """连续负盈利期数（自最近一次非负盈利以来的 count）。

    衡量盈利困境持续性。当前行为负时计数；为正时输出 0。fail-closed：当前
    行 NaN 输出 NaN。
    """

    metadata = _metadata(
        "es1_negative_earnings_streak",
        "当前连续负盈利期数（最近非负后归零）。",
        ["earnings"],
        domain="fundamental",
        unit="count",
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, **_: Any) -> pd.DataFrame:
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            streak = 0
            for row in range(rows):
                v = ev[row, col]
                if not np.isfinite(v):
                    out[row, col] = np.nan
                    streak = 0
                    continue
                if v < 0:
                    streak += 1
                else:
                    streak = 0
                out[row, col] = float(streak)
        return _frame_like(earnings, out)


@register_operator(
    name="es1_revenue_earnings_divergence",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="es1_revenue_earnings_divergence",
    source="wave1_earnings",
    status="experimental")
class Es1RevenueEarningsDivergence(SeriesOperator):
    """营收-盈利背离：Δlog(revenue) 与 Δlog(earnings) 之差的窗口均值。

    收入增长但盈利下降（或反之）预示质量恶化。输出 dimensionless（对数差）。
    """

    metadata = _metadata(
        "es1_revenue_earnings_divergence",
        "窗口 mean(Δlog(revenue) - Δlog(earnings))（营收盈利背离）。",
        ["revenue", "earnings", "window", "min_periods"],
        domain="fundamental",
        unit="dimensionless",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, revenue: pd.DataFrame, earnings: pd.DataFrame, window: int = 8, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        revenue, earnings = _aligned(revenue, earnings)
        rv = revenue.to_numpy(dtype=float)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(rv[row, col]) or not np.isfinite(ev[row, col]):
                    continue
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                e = ev[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(e) & (r > 0) & (e > 0)
                if ok.sum() < mp:
                    continue
                rv_ = r[ok]; ev_ = e[ok]
                if rv_.size < 2:
                    continue
                dlog_r = np.diff(np.log(rv_))
                dlog_e = np.diff(np.log(ev_))
                if dlog_r.size < 1:
                    continue
                out[row, col] = float(np.mean(dlog_r - dlog_e))
        return _frame_like(revenue, out)


@register_operator(
    name="es1_earnings_smoothness",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="es1_earnings_smoothness",
    source="wave1_earnings",
    status="experimental")
class Es1EarningsSmoothness(SeriesOperator):
    """盈利平滑度代理：std(earnings) / std(operating_cash_flow)。

    低比值 = 盈利比现金流更平滑（可能通过应计平滑利润，质量存疑）；接近 1 =
    盈利贴近现金流。输出 ratio。
    """

    metadata = _metadata(
        "es1_earnings_smoothness",
        "std(earnings)/std(operating_cash_flow)（盈利平滑度）。",
        ["earnings", "operating_cash_flow", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, operating_cash_flow: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        earnings, operating_cash_flow = _aligned(earnings, operating_cash_flow)
        ev = earnings.to_numpy(dtype=float)
        cv = operating_cash_flow.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                e = ev[lo:row + 1, col]
                c = cv[lo:row + 1, col]
                ok = np.isfinite(e) & np.isfinite(c)
                if ok.sum() < mp:
                    continue
                se = float(np.std(e[ok], ddof=1))
                sc = float(np.std(c[ok], ddof=1))
                if sc <= 1e-12:
                    continue
                out[row, col] = se / sc
        return _frame_like(earnings, out)


@register_operator(
    name="es1_earnings_mean_reversion_speed",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="es1_earnings_mean_reversion_speed",
    source="wave1_earnings",
    status="experimental")
class Es1EarningsMeanReversionSpeed(SeriesOperator):
    """盈利均值回归速度：AR(1) 系数（earnings 对 lag1 earnings 回归）。

    高正系数 = 盈利持续性（平稳）；接近 0 = 快速回归；负 = 盈利波动反相。
    输出 dimensionless（AR 系数）。
    """

    metadata = _metadata(
        "es1_earnings_mean_reversion_speed",
        "earnings AR(1) 系数（盈利持续性）。",
        ["earnings", "window", "min_periods"],
        domain="fundamental",
        unit="dimensionless",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(ev[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                x = vals[:-1]
                y = vals[1:]
                if x.size < 2:
                    continue
                den = float(np.sum((x - x.mean()) ** 2))
                if den <= 1e-12:
                    continue
                out[row, col] = float(np.sum((x - x.mean()) * (y - y.mean())) / den)
        return _frame_like(earnings, out)


# ---------------------------------------------------------------------------
# 2. accrual-quality family
# ---------------------------------------------------------------------------
@register_operator(
    name="aq1_working_capital_accrual",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="aq1_working_capital_accrual",
    source="wave1_earnings",
    status="experimental")
class Aq1WorkingCapitalAccrual(SeriesOperator):
    """营运资本应计率：Δ(working_capital)/|earnings|。

    应计越高（相对盈利），盈利现金含量越可疑。输出 signed_ratio（保留符号：
    working capital 增加为正）。
    """

    metadata = _metadata(
        "aq1_working_capital_accrual",
        "Δ(working_capital)/|earnings|（营运资本应计率）。",
        ["working_capital", "earnings", "min_periods"],
        domain="fundamental",
        unit="signed_ratio",
        category="fundamental_quality",
    )

    def _calculate_series(self, working_capital: pd.DataFrame, earnings: pd.DataFrame, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        mp = _check_int(min_periods, "min_periods", 2)
        working_capital, earnings = _aligned(working_capital, earnings)
        wv = working_capital.to_numpy(dtype=float)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = wv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev_wc: float | None = None
            prev_e: float | None = None
            for row in range(rows):
                w = wv[row, col]
                e = ev[row, col]
                if not np.isfinite(w) or not np.isfinite(e):
                    prev_wc = None
                    prev_e = None
                    out[row, col] = np.nan
                    continue
                if prev_wc is not None and prev_e is not None:
                    dw = w - prev_wc
                    den = abs(e)
                    if den > 1e-12:
                        out[row, col] = dw / den
                    else:
                        out[row, col] = np.nan
                prev_wc = w
                prev_e = e
        return _frame_like(working_capital, out)


@register_operator(
    name="aq1_accrual_stability",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="aq1_accrual_stability",
    source="wave1_earnings",
    status="experimental")
class Aq1AccrualStability(SeriesOperator):
    """应计稳定性：应计率（accrual/earnings）的负 CV（越大越稳定）。

    取负使"稳定性"方向一致：应计率波动大（质量不稳）输出低值。输出 ratio。
    """

    metadata = _metadata(
        "aq1_accrual_stability",
        "窗口应计率 |ΔWC|/|earnings| 的负 CV。",
        ["working_capital", "earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, working_capital: pd.DataFrame, earnings: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        working_capital, earnings = _aligned(working_capital, earnings)
        wv = working_capital.to_numpy(dtype=float)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = wv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(wv[row, col]) or not np.isfinite(ev[row, col]):
                    continue
                lo = max(0, row - w + 1)
                w_win = wv[lo:row + 1, col]
                e_win = ev[lo:row + 1, col]
                ok = np.isfinite(w_win) & np.isfinite(e_win) & (np.abs(e_win) > 1e-12)
                if ok.sum() < mp:
                    continue
                dw = np.diff(w_win[ok])
                e_ = np.abs(e_win[ok][1:])
                if dw.size < 2:
                    continue
                ratio = dw / (e_ + 1e-12)
                m = float(np.mean(ratio))
                s = float(np.std(ratio, ddof=1))
                if m <= 1e-12:
                    out[row, col] = 0.0
                    continue
                out[row, col] = -s / abs(m)
        return _frame_like(working_capital, out)


@register_operator(
    name="aq1_cash_conversion_strength",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="aq1_cash_conversion_strength",
    source="wave1_earnings",
    status="experimental")
class Aq1CashConversionStrength(SeriesOperator):
    """现金转化强度：operating_cash_flow / earnings（窗口均值）。

    越高（>1）盈利越由现金流支持（高质量）；越低越依赖应计。输出 ratio。
    """

    metadata = _metadata(
        "aq1_cash_conversion_strength",
        "窗口 mean(ocf/earnings)（现金转化强度）。",
        ["operating_cash_flow", "earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, operating_cash_flow: pd.DataFrame, earnings: pd.DataFrame, window: int = 8, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        operating_cash_flow, earnings = _aligned(operating_cash_flow, earnings)
        cv = operating_cash_flow.to_numpy(dtype=float)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                c = cv[lo:row + 1, col]
                e = ev[lo:row + 1, col]
                ok = np.isfinite(c) & np.isfinite(e) & (np.abs(e) > 1e-12)
                if ok.sum() < mp:
                    continue
                out[row, col] = float(np.mean(c[ok] / e[ok]))
        return _frame_like(operating_cash_flow, out)


@register_operator(
    name="aq1_cash_flow_volatility",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="aq1_cash_flow_volatility",
    source="wave1_earnings",
    status="experimental")
class Aq1CashFlowVolatility(SeriesOperator):
    """现金流波动率：std(operating_cash_flow)/|mean|（负向质量信号）。

    高 = 现金流不稳定（质量存疑）。输出 ratio。
    """

    metadata = _metadata(
        "aq1_cash_flow_volatility",
        "std(ocf)/|mean(ocf)|（现金流波动）。",
        ["operating_cash_flow", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, operating_cash_flow: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        cv = operating_cash_flow.to_numpy(dtype=float)
        rows, cols = cv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                c = cv[lo:row + 1, col]
                ok = np.isfinite(c)
                if ok.sum() < mp:
                    continue
                vals = c[ok]
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1))
                den = abs(m)
                if den <= 1e-12:
                    out[row, col] = 0.0
                    continue
                out[row, col] = s / den
        return _frame_like(operating_cash_flow, out)


@register_operator(
    name="aq1_accrual_ratio_dispersion",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="aq1_accrual_ratio_dispersion",
    source="wave1_earnings",
    status="experimental")
class Aq1AccrualRatioDispersion(SeriesOperator):
    """应计率离散度：|accrual_ratio| 的滚动 std（质量异动）。

    应计率 = (ΔWC)/|earnings|；其滚动 std 高 = 应计行为不稳定（异动）。输出
    ratio。
    """

    metadata = _metadata(
        "aq1_accrual_ratio_dispersion",
        "滚动 |ΔWC/earnings| 的 std（应计行为异动）。",
        ["working_capital", "earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, working_capital: pd.DataFrame, earnings: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        working_capital, earnings = _aligned(working_capital, earnings)
        wv = working_capital.to_numpy(dtype=float)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = wv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(wv[row, col]) or not np.isfinite(ev[row, col]):
                    continue
                lo = max(0, row - w + 1)
                w_win = wv[lo:row + 1, col]
                e_win = ev[lo:row + 1, col]
                ok = np.isfinite(w_win) & np.isfinite(e_win) & (np.abs(e_win) > 1e-12)
                if ok.sum() < mp:
                    continue
                dw = np.diff(w_win[ok])
                e_ = np.abs(e_win[ok][1:])
                if dw.size < 2:
                    continue
                ratio = dw / (e_ + 1e-12)
                out[row, col] = float(np.std(np.abs(ratio), ddof=1))
        return _frame_like(working_capital, out)


# ---------------------------------------------------------------------------
# 3. earnings-persistence family
# ---------------------------------------------------------------------------
@register_operator(
    name="ep1_earnings_autocorr",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="ep1_earnings_autocorr",
    source="wave1_earnings",
    status="experimental")
class Ep1EarningsAutocorr(SeriesOperator):
    """盈利自相关（lag-1 Pearson）：盈利持续性测度。

    高正 = 盈利序列平稳持续；接近 0 = 随机。输出 ratio。
    """

    metadata = _metadata(
        "ep1_earnings_autocorr",
        "earnings lag-1 自相关（盈利持续性）。",
        ["earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                x = vals[:-1]
                y = vals[1:]
                if x.size < 2:
                    continue
                sx = float(np.std(x, ddof=1))
                sy = float(np.std(y, ddof=1))
                if sx <= 1e-12 or sy <= 1e-12:
                    continue
                out[row, col] = float(np.corrcoef(x, y)[0, 1])
        return _frame_like(earnings, out)


@register_operator(
    name="ep1_roa_stability",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="ep1_roa_stability",
    source="wave1_earnings",
    status="experimental")
class Ep1RoaStability(SeriesOperator):
    """ROA 稳定性：1 - std(roa)/mean|roa| 的截断（质量正向）。

    高值 = ROA 稳定（质量好）。输出 ratio（截断在 [0,1] 外保持单调）。
    """

    metadata = _metadata(
        "ep1_roa_stability",
        "1 - std(roa)/|mean(roa)|（ROA 稳定性）。",
        ["roa", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, roa: pd.DataFrame, window: int = 8, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        rv = roa.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = rv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1))
                den = abs(m)
                if den <= 1e-12:
                    out[row, col] = 0.0
                    continue
                out[row, col] = 1.0 - s / den
        return _frame_like(roa, out)


@register_operator(
    name="ep1_earnings_consistency",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="ep1_earnings_consistency",
    source="wave1_earnings",
    status="experimental")
class Ep1EarningsConsistency(SeriesOperator):
    """盈利一致性得分：窗口内盈利同号占比（质量信号）。

    比例接近 1 = 长期稳定盈利；低 = 盈亏交替。输出 ratio。
    """

    metadata = _metadata(
        "ep1_earnings_consistency",
        "窗口内盈利同号（非负）占比。",
        ["earnings", "window", "min_periods"],
        domain="fundamental",
        unit="ratio",
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, window: int = 8, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = ev[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                out[row, col] = float(np.mean(vals >= 0))
        return _frame_like(earnings, out)


@register_operator(
    name="ep1_earnings_surprise_decay",
    category="fundamental_quality",
    business_category="fundamental_quality",
    canonical="ep1_earnings_surprise_decay",
    source="wave1_earnings",
    status="experimental")
class Ep1EarningsSurpriseDecay(SeriesOperator):
    """盈利惊喜衰减：历史 surprise 的半衰期指数平滑（事件响应衰减）。

    surprise = (earnings - mean_prev)/std_prev（标准化）。用 half_life 指数
    衰减累计，捕获盈利惊喜的持续性/衰减速度。输出 dimensionless。
    """

    metadata = _metadata(
        "ep1_earnings_surprise_decay",
        "标准化盈利惊喜的半衰期指数平滑。",
        ["earnings", "half_life", "min_periods"],
        domain="fundamental",
        unit="dimensionless",
        cost=2,
        category="fundamental_quality",
    )

    def _calculate_series(self, earnings: pd.DataFrame, half_life: float = 6.0, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        hl = float(half_life)
        if hl < 1.0:
            raise ValueError("half_life must be >= 1")
        mp = _check_int(min_periods, "min_periods", 3)
        weight = 0.5 ** (1.0 / hl)
        ev = earnings.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            hist: list[float] = []
            acc = 0.0
            seen = False
            for row in range(rows):
                v = ev[row, col]
                if not np.isfinite(v):
                    if seen:
                        acc = acc * weight
                        out[row, col] = acc
                    continue
                if len(hist) >= mp:
                    h = np.array(hist)
                    m = float(np.mean(h))
                    s = float(np.std(h, ddof=1))
                    if s > 1e-12:
                        surprise = (v - m) / s
                    else:
                        surprise = 0.0
                    acc = acc * weight + surprise
                    out[row, col] = acc
                    seen = True
                hist.append(v)
        return _frame_like(earnings, out)