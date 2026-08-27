# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: valuation-relative measures.

New canonicals across genuinely new thematic ground:
  * earnings-yield trend / FCF yield growth (valuation-relative momentum)
  * relative-valuation z / percentile / spread (vs trailing history or a
    reference series)
  * Q-style quality-minus-junk composite helpers
  * valuation dispersion and discount-regime statistics

All are real pandas_numpy implementations, causal, deterministic and NaN-safe.
Names are prefixed ``val1_`` and are globally unique.
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
# 1. earnings-yield trend family
# ---------------------------------------------------------------------------
@register_operator(
    name="val1_earnings_yield_slope",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_earnings_yield_slope",
    source="wave1_valuation",
    status="experimental")
class Val1EarningsYieldSlope(SeriesOperator):
    """盈利收益率（1/PE）的滚动斜率 × (n-1)（估值趋势）。

    上行收益率趋势 = 估值改善/盈利增长。输出 dimensionless（斜率×尺度）。
    """

    metadata = _metadata(
        "val1_earnings_yield_slope",
        "earnings_yield 对时间回归的斜率 × sqrt(n)（估值趋势）。",
        ["earnings_yield", "window", "min_periods"],
        domain="valuation",
        unit="dimensionless",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, earnings_yield: pd.DataFrame, window: int = 12, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        yv = earnings_yield.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(yv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = yv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                n = vals.size
                t = np.arange(n, dtype=float)
                den = float(np.sum((t - t.mean()) ** 2))
                if den <= 1e-12:
                    continue
                slope = float(np.sum((t - t.mean()) * (vals - vals.mean())) / den)
                out[row, col] = slope * float(np.sqrt(n))
        return _frame_like(earnings_yield, out)


@register_operator(
    name="val1_earnings_yield_ma_diff",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_earnings_yield_ma_diff",
    source="wave1_valuation",
    status="experimental")
class Val1EarningsYieldMaDiff(SeriesOperator):
    """盈利收益率相对其移动均值的偏离（相对估值反转）。

    mean(yield_{t-s..t-1}) 为基准，输出当前 yield - 基准（同单位），捕获
    估值相对自身历史的便宜/昂贵。
    """

    metadata = _metadata(
        "val1_earnings_yield_ma_diff",
        "yield 相对过去均值的差（相对自身历史估值）。",
        ["earnings_yield", "baseline_window", "min_periods"],
        domain="valuation",
        unit="same_as:earnings_yield",
        category="valuation_relative",
    )

    def _calculate_series(self, earnings_yield: pd.DataFrame, baseline_window: int = 12, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(baseline_window, "baseline_window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        yv = earnings_yield.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(yv[row, col]):
                    continue
                lo = max(0, row - w)
                base = yv[lo:row, col]
                ok = np.isfinite(base)
                if ok.sum() < mp:
                    continue
                bm = float(np.mean(base[ok]))
                out[row, col] = yv[row, col] - bm
        return _frame_like(earnings_yield, out)


@register_operator(
    name="val1_fcf_yield_growth",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_fcf_yield_growth",
    source="wave1_valuation",
    status="experimental")
class Val1FcfYieldGrowth(SeriesOperator):
    """自由现金流收益率增长：(yield_t - yield_{t-1})/|yield_{t-1}|。

    现金流收益率加速增长 = 价值改善或价格落后的现金创造力（相对估值动量）。
    输出 signed_ratio。
    """

    metadata = _metadata(
        "val1_fcf_yield_growth",
        "FCF 收益率的一阶相对变化。",
        ["fcf_yield", "min_periods"],
        domain="valuation",
        unit="signed_ratio",
        category="valuation_relative",
    )

    def _calculate_series(self, fcf_yield: pd.DataFrame, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        mp = _check_int(min_periods, "min_periods", 2)
        fv = fcf_yield.to_numpy(dtype=float)
        rows, cols = fv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            prev: float | None = None
            count = 0
            for row in range(rows):
                v = fv[row, col]
                if not np.isfinite(v):
                    prev = None
                    count = 0
                    out[row, col] = np.nan
                    continue
                if prev is not None and count >= mp - 1 and abs(prev) > 1e-12:
                    out[row, col] = (v - prev) / abs(prev)
                else:
                    out[row, col] = np.nan
                prev = v
                count += 1
        return _frame_like(fcf_yield, out)


@register_operator(
    name="val1_earnings_yield_persistence",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_earnings_yield_persistence",
    source="wave1_valuation",
    status="experimental")
class Val1EarningsYieldPersistence(SeriesOperator):
    """盈利收益率趋势持续性：收益率为正的窗口占比（估值改善持续性）。

    比率接近 1 = 估值持续改善（收益率持续上升），接近 0 = 持续恶化。输出
    ratio。
    """

    metadata = _metadata(
        "val1_earnings_yield_persistence",
        "窗口内 yield 一阶差分为正的占比（估值趋势持续）。",
        ["earnings_yield", "window", "min_periods"],
        domain="valuation",
        unit="ratio",
        category="valuation_relative",
    )

    def _calculate_series(self, earnings_yield: pd.DataFrame, window: int = 12, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        yv = earnings_yield.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = yv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                if vals.size < 2:
                    continue
                d = np.diff(vals)
                out[row, col] = float(np.mean(d > 0))
        return _frame_like(earnings_yield, out)


# ---------------------------------------------------------------------------
# 2. relative-valuation z / percentile / spread family
# ---------------------------------------------------------------------------
def _as_z(vals: np.ndarray) -> float:
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    if s <= 1e-12:
        return 0.0
    return (vals[-1] - m) / s


@register_operator(
    name="val1_valuation_z_own",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_valuation_z_own",
    source="wave1_valuation",
    status="experimental")
class Val1ValuationZOwn(SeriesOperator):
    """自身估值 z-score：当前 value_metric 相对其过去历史分布。

    正值 = 相对自身历史更贵/因子值更高；负值 = 更便宜。causal。
    """

    metadata = _metadata(
        "val1_valuation_z_own",
        "(当前 - 历史均值)/历史 std（相对自身历史估值）。",
        ["value_metric", "history_window", "min_periods"],
        domain="valuation",
        unit="dimensionless",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, value_metric: pd.DataFrame, history_window: int = 60, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        hw = _check_int(history_window, "history_window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > hw:
            raise ValueError("min_periods must be <= history_window")
        mv = value_metric.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(mv[row, col]):
                    continue
                lo = max(0, row - hw + 1)
                hist = mv[lo:row, col]
                ok = np.isfinite(hist)
                if ok.sum() < mp:
                    continue
                out[row, col] = _as_z(np.concatenate([hist[ok], [mv[row, col]]]))
        return _frame_like(value_metric, out)


@register_operator(
    name="val1_valuation_percentile_own",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_valuation_percentile_own",
    source="wave1_valuation",
    status="experimental")
class Val1ValuationPercentileOwn(SeriesOperator):
    """自身估值分位：(rank of current among history)/(count) in [0,1]。

    与 z 版本互补的稳健缺失风格分布。输出 ratio。
    """

    metadata = _metadata(
        "val1_valuation_percentile_own",
        "当前值在自身历史中的分位 [0,1]。",
        ["value_metric", "history_window", "min_periods"],
        domain="valuation",
        unit="ratio",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, value_metric: pd.DataFrame, history_window: int = 60, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        hw = _check_int(history_window, "history_window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        if mp > hw:
            raise ValueError("min_periods must be <= history_window")
        mv = value_metric.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(mv[row, col]):
                    continue
                lo = max(0, row - hw + 1)
                hist = mv[lo:row, col]
                ok = np.isfinite(hist)
                if ok.sum() < mp:
                    continue
                h = hist[ok]
                cur = mv[row, col]
                rank = float(np.sum(h < cur)) + 0.5 * float(np.sum(h == cur))
                out[row, col] = rank / h.size
        return _frame_like(value_metric, out)


@register_operator(
    name="val1_relative_valuation_gap",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_relative_valuation_gap",
    source="wave1_valuation",
    status="experimental")
class Val1RelativeValuationGap(SeriesOperator):
    """相对估值落差：own_metric - reference_metric（同单位）。

    用 reference 序列作为估值锚（如行业基准、market 中位数），捕捉相对
    便宜/昂贵。
    """

    metadata = _metadata(
        "val1_relative_valuation_gap",
        "own_metric - reference_metric（相对估值落差）。",
        ["own_metric", "reference_metric"],
        domain="valuation",
        unit="same_as:own_metric",
        category="valuation_relative",
    )

    def _calculate_series(self, own_metric: pd.DataFrame, reference_metric: pd.DataFrame, **_: Any) -> pd.DataFrame:
        own_metric, reference_metric = _aligned(own_metric, reference_metric)
        ov = own_metric.to_numpy(dtype=float)
        rv = reference_metric.to_numpy(dtype=float)
        rows, cols = ov.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if np.isfinite(ov[row, col]) and np.isfinite(rv[row, col]):
                    out[row, col] = ov[row, col] - rv[row, col]
        return _frame_like(own_metric, out)


@register_operator(
    name="val1_valuation_stat_spread",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_valuation_stat_spread",
    source="wave1_valuation",
    status="experimental")
class Val1ValuationStatSpread(SeriesOperator):
    """估值横截面离散度：当前行价值度量的 std（market-wide cheap/expensive 梯度）。

    每日一行的遍历：跨股票 valuation 的标准差。高 = 市场内估值分化大。输出
    same_as 单位。
    """

    metadata = _metadata(
        "val1_valuation_stat_spread",
        "横截面 value_metric 的 std（估值分化度）。",
        ["value_metric", "min_breadth"],
        domain="valuation",
        unit="same_as:value_metric",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, value_metric: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 3)
        mv = value_metric.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            vals = mv[row]
            ok = np.isfinite(vals)
            if ok.sum() < mb:
                continue
            out[row, :] = float(np.std(vals[ok], ddof=1))
        return _frame_like(value_metric, out)


@register_operator(
    name="val1_discount_regime_share",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_discount_regime_share",
    source="wave1_valuation",
    status="experimental")
class Val1DiscountRegimeShare(SeriesOperator):
    """折扣 regime 份额：当前行中因子值低于自身历史中位数的股票占比。

    衡量市场整体处于"便宜"状态的比例。需逐股历史中位数（因果滚动中位）。
    输出 ratio（0..1，全市场折扣强度）。
    """

    metadata = _metadata(
        "val1_discount_regime_share",
        "当下 value_metric 低于自身历史中位数的横截面占比。",
        ["value_metric", "history_window", "min_periods", "min_breadth"],
        domain="valuation",
        unit="ratio",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, value_metric: pd.DataFrame, history_window: int = 40, min_periods: int = 10, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        hw = _check_int(history_window, "history_window", 3)
        mp = _check_int(min_periods, "min_periods", 3)
        mb = _check_int(min_breadth, "min_breadth", 3)
        if mp > hw:
            raise ValueError("min_periods must be <= history_window")
        mv = value_metric.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        # 每行维护每只股票的历史缓冲
        hist: list[list[float]] = [[] for _ in range(cols)]
        for row in range(rows):
            below = 0
            total = 0
            for c in range(cols):
                cur = mv[row, c]
                h = hist[c]
                if np.isfinite(cur) and len(h) >= mp:
                    med = float(np.median(h))
                    if med > 1e-12:
                        if cur < med:
                            below += 1
                        total += 1
                if np.isfinite(cur):
                    h.append(cur)
            if total >= mb:
                out[row, :] = below / total
        return _frame_like(value_metric, out)


# ---------------------------------------------------------------------------
# 3. Q-style composite helpers + valuation dispersion
# ---------------------------------------------------------------------------
@register_operator(
    name="val1_qmj_quality_rank",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_qmj_quality_rank",
    source="wave1_valuation",
    status="experimental")
class Val1QmjQualityRank(SeriesOperator):
    """QMJ 风格质量-排名：earnings_stability × profitability 的横截面合成。

    将 stability（1/std 代理）与 profitability（roa）在横截面内相乘后 rank。
    输出无量纲排名分（0..1）。
    """

    metadata = _metadata(
        "val1_qmj_quality_rank",
        "quality ~ profitability × 盈利稳定性的横截面估值。",
        ["roa", "earnings_stability", "min_breadth"],
        domain="valuation",
        unit="dimensionless",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, roa: pd.DataFrame, earnings_stability: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 3)
        roa, earnings_stability = _aligned(roa, earnings_stability)
        rv = roa.to_numpy(dtype=float)
        sv = earnings_stability.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            r = rv[row]
            s = sv[row]
            ok = np.isfinite(r) & np.isfinite(s)
            if ok.sum() < mb:
                continue
            quality = r[ok] * (1.0 / (1.0 + np.abs(s[ok])))
            rnk = pd.Series(quality).rank(pct=True).to_numpy()
            out[row, ok] = rnk
        return _frame_like(roa, out)


@register_operator(
    name="val1_pe_beta_to_market",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_pe_beta_to_market",
    source="wave1_valuation",
    status="experimental")
class Val1PeBetaToMarket(SeriesOperator):
    """个股 PE 对市场 PE 的 beta（估值杠杆/同步性）。

    >1 = 个股估值放大/同步于市场估值变动；<0 = 反向。输入 own_pe 与
    market_pe（宽基中位数）。输出 ratio。
    """

    metadata = _metadata(
        "val1_pe_beta_to_market",
        "own_pe 对 market_pe 的滚动 beta（估值同步性）。",
        ["own_pe", "market_pe", "window", "min_periods"],
        domain="valuation",
        unit="ratio",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, own_pe: pd.DataFrame, market_pe: pd.DataFrame, window: int = 30, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 5)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        own_pe, market_pe = _aligned(own_pe, market_pe)
        ov = own_pe.to_numpy(dtype=float)
        mv = market_pe.to_numpy(dtype=float)
        rows, cols = ov.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                x = mv[lo:row + 1, col]
                y = ov[lo:row + 1, col]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < mp:
                    continue
                xv = x[ok]
                yv = y[ok]
                vx = float(np.var(xv, ddof=1))
                if vx <= 1e-12:
                    continue
                out[row, col] = float(np.cov(xv, yv, ddof=1)[0, 1] / vx)
        return _frame_like(own_pe, out)


@register_operator(
    name="val1_valuations_lag_component",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_valuations_lag_component",
    source="wave1_valuation",
    status="experimental")
class Val1ValuationsLagComponent(SeriesOperator):
    """估值滞后成分：当前值相对过去 n 日均值的符号化偏离强度。

    对自身估值的「滞后」滤波器：平滑当前估值与其滞后均值的差异，突出最近
    变动。输出 dimensionless。
    """

    metadata = _metadata(
        "val1_valuations_lag_component",
        "短期均线相对长期均值的标准化差（估值滞后成分）。",
        ["value_metric", "fast_window", "slow_window", "min_periods"],
        domain="valuation",
        unit="dimensionless",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, value_metric: pd.DataFrame, fast_window: int = 5, slow_window: int = 20, min_periods: int = 4, **_: Any) -> pd.DataFrame:
        fw = _check_int(fast_window, "fast_window", 2)
        sw = _check_int(slow_window, "slow_window", 3)
        mp = _check_int(min_periods, "min_periods", 2)
        if fw >= sw:
            raise ValueError("fast_window must be < slow_window")
        mv = value_metric.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            cache: list[float] = []
            for row in range(rows):
                v = mv[row, col]
                if np.isfinite(v):
                    cache.append(v)
                if len(cache) < sw:
                    continue
                hist = np.array(cache[-sw:])
                fast_m = float(np.mean(hist[-fw:]))
                slow_m = float(np.mean(hist))
                s = float(np.std(hist, ddof=1))
                if s <= 1e-12:
                    out[row, col] = 0.0
                else:
                    out[row, col] = (fast_m - slow_m) / s
        return _frame_like(value_metric, out)


@register_operator(
    name="val1_equity_yield_dispersion",
    category="valuation_relative",
    business_category="valuation_relative",
    canonical="val1_equity_yield_dispersion",
    source="wave1_valuation",
    status="experimental")
class Val1EquityYieldDispersion(SeriesOperator):
    """股权收益率（earnings_yield）横截面离散度：IQR（分位差）。

    跨股票的 yield 差异大 = 估值分散（市场对个体差异定价强）。输出 same_as
    单位。每行横截面。
    """

    metadata = _metadata(
        "val1_equity_yield_dispersion",
        "横截面 earnings_yield 的 IQR（估值离散度）。",
        ["earnings_yield", "min_breadth"],
        domain="valuation",
        unit="same_as:earnings_yield",
        cost=2,
        category="valuation_relative",
    )

    def _calculate_series(self, earnings_yield: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 4)
        yv = earnings_yield.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            vals = yv[row]
            ok = np.isfinite(vals)
            if ok.sum() < mb:
                continue
            q = np.quantile(vals[ok], [0.25, 0.75])
            out[row, :] = q[1] - q[0]
        return _frame_like(earnings_yield, out)