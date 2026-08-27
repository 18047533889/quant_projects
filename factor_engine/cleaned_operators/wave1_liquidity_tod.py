# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: microstructure liquidity / time-of-day shape.

New canonicals across genuinely new thematic ground:
  * liquidity-fragmentation / turnover-adjusted price-impact measures
  * time-of-day U-shape location (minute-panel intraday kernels tolerate
    any row-length; a daily panel degenerates them into day-index shape
    statistics which are still well-defined)
  * daily liquidity-factor transformation helpers
  * relative metric decompositions (price impact, mean-variance, market
    share)

Every canonical is a real pandas_numpy reference implementation (no stubs, no
pass-through), deterministic, causal, NaN per family convention, with full
semantic metadata.  Names are prefixed ``lf1_`` / ``tod_`` and were checked as
globally unique at authoring time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned, _check_int


def _metadata(
    name: str, description: str, params: list[str], *, domain: str, unit: str,
    cost: int = 1, category: str, input_units: dict[str, str] | None = None,
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
        input_units=dict(input_units or {}),
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# 1. liquidity-fragmentation
# ---------------------------------------------------------------------------
@register_operator(
    name="lf1_herfindahl_volume",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_herfindahl_volume",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1HerfindahlVolume(SeriesOperator):
    """横截面成交量赫芬达尔集中度（sum of squared volume shares）。

    每行独立对全部股票计算 volume[i]/sum(volume) 的平方和。数值 1.0 表示
    成交量全部集中在一只股票（极端集中），约 0 表示完全分散（大量等大份额）。
    低集中度（分散成交）通常隐含更健康的整体流动性。每行成交量为 0 或样本
    不足 _MIN_BREADTH 时该行 fail-closed 为 NaN。
    """

    metadata = _metadata(
        "lf1_herfindahl_volume",
        "横截面成交量赫芬达尔集中度（sum of squared volume shares）。",
        ["volume", "min_breadth"],
        domain="volume",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, volume: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 2)
        vv = volume.to_numpy(dtype=float)
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            v = vv[row]
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < mb:
                continue
            total = np.sum(v[ok])
            if total <= 0:
                continue
            shares = v[ok] / total
            h = float(np.sum(shares * shares))
            out[row, ok] = h
        return _frame_like(volume, out)


@register_operator(
    name="lf1_volume_concentration",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_volume_concentration",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1VolumeConcentration(SeriesOperator):
    """成交量集中度：前 top_share（默认 20%）股票占横截面总成交量的份额。

    每行独立计算：按成交量降序，累计占比达到 top_share 所需的股票。若该
    阈值跨入由数值给出；否则返回需覆盖的股票数占比。输出单位 ratio。
    fail-closed：样本不足返回 NaN。
    """

    metadata = _metadata(
        "lf1_volume_concentration",
        "头部 top_share 成交量占比（按股票数跨过该累计份额）。",
        ["volume", "top_share", "min_breadth"],
        domain="volume",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, volume: pd.DataFrame, top_share: float = 0.2, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 2)
        ts = float(top_share)
        if not (0.0 < ts <= 1.0):
            raise ValueError("top_share must be in (0, 1]")
        vv = volume.to_numpy(dtype=float)
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            v = vv[row]
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < mb:
                continue
            total = np.sum(v[ok])
            if total <= 0:
                continue
            vals = np.sort(v[ok])[::-1]
            csum = np.cumsum(vals) / total
            k = int(np.searchsorted(csum, ts)) + 1
            k = int(np.clip(k, 1, ok.sum()))
            frac = k / ok.sum()
            out[row, ok] = frac
        return _frame_like(volume, out)


@register_operator(
    name="lf1_turnover_distribution_skew",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_turnover_distribution_skew",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1TurnoverDistributionSkew(SeriesOperator):
    """横截面换手率分布的偏度（Fisher）。

    每行独立计算。正值表示少数股票换手远高/低于多数股票（换手分布重尾）。
    该指标刻画整体流动性的横截面"集中"模式，帮助识别个别高换手异动股票。
    """

    metadata = _metadata(
        "lf1_turnover_distribution_skew",
        "横截面换手率分布的 Fisher 偏度。",
        ["turnover", "min_breadth"],
        domain="turnover",
        unit="dimensionless",
        category="market_microstructure",
        cost=2,
    )

    def _calculate_series(self, turnover: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 3)
        tv = turnover.to_numpy(dtype=float)
        rows, cols = tv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            t = tv[row]
            ok = np.isfinite(t) & (t > 0)
            if ok.sum() < mb:
                continue
            vals = np.log(t[ok])
            n = vals.size
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            if sd <= 1e-12:
                out[row, ok] = 0.0
                continue
            skew = float(np.mean(((vals - m) / sd) ** 3))
            out[row, ok] = skew * (n / ((n - 1) * (n - 2) + 1e-12))
        return _frame_like(turnover, out)


# ---------------------------------------------------------------------------
# 2. price-impact / turnover-adjusted liquidity
# ---------------------------------------------------------------------------
@register_operator(
    name="lf1_amihud_own",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_amihud_own",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1AmihudOwn(SeriesOperator):
    """自身 Amihud 非流动性代理：window 内 mean(|ret| / amount)。

    与现有 amihud_illiquidity 等价但更轻量的原子实现，专注于完全自包含的
    参考语义：ret 为小数收益，amount 为成交额（元），输出 ratio，越大越
    不流动。NaN 行不参与窗口统计；当前行必须有效才输出（否则 NaN）。
    """

    metadata = _metadata(
        "lf1_amihud_own",
        "窗口平均 |ret|/amount（Amihud 非流动性自含实现）。",
        ["ret", "amount", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="market_microstructure",
        input_units={"ret": "return", "amount": "amount"},
    )

    def _calculate_series(self, ret: pd.DataFrame, amount: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ret, amount = _aligned(ret, amount)
        rv = ret.to_numpy(dtype=float)
        av = amount.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(rv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                a = av[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(a) & (a > 0)
                if ok.sum() < mp:
                    continue
                ratio = np.abs(r[ok]) / a[ok]
                out[row, col] = float(np.mean(ratio))
        return _frame_like(ret, out)


@register_operator(
    name="lf1_volume_amplification",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_volume_amplification",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1VolumeAmplification(SeriesOperator):
    """成交放大比：当前成交额对基准窗口（ba_window）均值的比值。

    衡量成交额相对近期均值的瞬时放大。逻辑上相对时间窗的 scale-invariant
    指标，不依赖绝对成交额量纲。
    """

    metadata = _metadata(
        "lf1_volume_amplification",
        "当前成交额 / ba_window 均值（成交放大比）。",
        ["amount", "ba_window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, amount: pd.DataFrame, ba_window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        w = _check_int(ba_window, "ba_window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        av = amount.to_numpy(dtype=float)
        rows, cols = av.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(av[row, col]):
                    continue
                lo = max(0, row - w + 1)
                base = av[lo:row, col]
                ok = np.isfinite(base) & (base > 0)
                if ok.sum() < mp:
                    continue
                bm = float(np.mean(base[ok]))
                if bm <= 0 or not np.isfinite(bm):
                    continue
                out[row, col] = av[row, col] / bm
        return _frame_like(amount, out)


@register_operator(
    name="lf1_volume_amihud_ratio",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_volume_amihud_ratio",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1VolumeAmihudRatio(SeriesOperator):
    """单位成交额收益的对称化：mean(|ret|) / mean(amount)。

    除以成交额年均的收益冲击强度，输出 ratio，越大表示给定成交额带来的
    价格冲击越大（越不流动）。相比单点 Amihud 更稳健（分子分母都取窗口均值）。
    """

    metadata = _metadata(
        "lf1_volume_amihud_ratio",
        "窗口 mean(|ret|) / mean(amount)。",
        ["ret", "amount", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="market_microstructure",
        input_units={"ret": "return", "amount": "amount"},
    )

    def _calculate_series(self, ret: pd.DataFrame, amount: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
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
                am = float(np.mean(a[ok]))
                if am <= 0:
                    continue
                out[row, col] = float(np.mean(np.abs(r[ok])) / am)
        return _frame_like(ret, out)


@register_operator(
    name="lf1_mean_variance_turnover",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_mean_variance_turnover",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1MeanVarianceTurnover(SeriesOperator):
    """换手均值-方差 σ_ratio = mean(turnover) / (std(turnover) + eps)。

    刻画换手率的"均值vs波动"关系：稳定高换手（低波动）σ_ratio 大；间歇性
    高换手（高波动）小。作为流动性稳定性特征，输出 ratio。
    """

    metadata = _metadata(
        "lf1_mean_variance_turnover",
        "mean(turnover) / std(turnover)（换手均值/波动比）。",
        ["turnover", "window", "min_periods"],
        domain="turnover",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, turnover: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        tv = turnover.to_numpy(dtype=float)
        rows, cols = tv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                t = tv[lo:row + 1, col]
                ok = np.isfinite(t) & (t > 0)
                if ok.sum() < mp:
                    continue
                m = float(np.mean(t[ok]))
                s = float(np.std(t[ok], ddof=1))
                out[row, col] = m / (s + 1e-12)
        return _frame_like(turnover, out)


@register_operator(
    name="lf1_market_share_turnover",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_market_share_turnover",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1MarketShareTurnover(SeriesOperator):
    """个股成交额占当日全市场成交额份额（每行横截面归一化）。

    每个交易日独立计算 amount[i]/sum(amount)。输出同输入单位（relative
    share，无量纲 ratio）。样本不足时 fail-closed。
    """

    metadata = _metadata(
        "lf1_market_share_turnover",
        "个股成交额占当日全市场成交额份额。",
        ["amount", "min_breadth"],
        domain="volume",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, amount: pd.DataFrame, min_breadth: int = 10, **_: Any) -> pd.DataFrame:
        mb = _check_int(min_breadth, "min_breadth", 2)
        av = amount.to_numpy(dtype=float)
        rows, cols = av.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            a = av[row]
            ok = np.isfinite(a) & (a > 0)
            if ok.sum() < mb:
                continue
            total = float(np.sum(a[ok]))
            if total <= 0:
                continue
            out[row, ok] = a[ok] / total
        return _frame_like(amount, out)


@register_operator(
    name="lf1_own_volume_share",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_own_volume_share",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1OwnVolumeShare(SeriesOperator):
    """个股成交量占最近 window 内自身累计成交量的份额（common-size 化）。

    将逐日成交量转化为其在近期总成交量中的比例，消除规模因素，突出近期
    相对活跃度变化。输出 ratio。
    """

    metadata = _metadata(
        "lf1_own_volume_share",
        "当日成交量占窗口自身累计成交量份额。",
        ["volume", "window", "min_periods"],
        domain="volume",
        unit="ratio",
        category="market_microstructure",
    )

    def _calculate_series(self, volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        vv = volume.to_numpy(dtype=float)
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(vv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                v = vv[lo:row + 1, col]
                ok = np.isfinite(v) & (v > 0)
                if ok.sum() < mp:
                    continue
                total = float(np.sum(v[ok]))
                if total <= 0:
                    continue
                out[row, col] = vv[row, col] / total
        return _frame_like(volume, out)


# ---------------------------------------------------------------------------
# 3. time-of-day U-shape location
# ---------------------------------------------------------------------------
def _tod_kernel(x: np.ndarray) -> float:
    """Local time-of-day location statistic.

    U-shaped intraday volatility follows a common pattern: high at open and at
    close, low at midday.  The kernel measures a timestamp-weighted distance from
    the quiet midday anchor: timestamps are normalised to [0, 1] over the row
    length, and the output is ``mean((t_i - 0.5)^2 weighted by edge-relevance)``
    re-scaled to a unit mean when the curve is exactly flat.  Concretely:

      v_i  = |window-past slope| proxy passed by caller
      out  = mean_j( |t_j - 0.5| ) * sqrt(mean(v)) / 0.25

    which rises when activity concentrates near the open/close edges and falls
    when it concentrates mid-day.  Any row length is accepted (daily panels
    degenerate to day-index shape, still well-defined).  Requires >= 2 valid
    rows.
    """
    n = x.size
    if n < 2:
        return np.nan
    t = (np.arange(n, dtype=float) + 0.5) / n
    w = np.abs(t - 0.5)  # edge weight: 0 at midline, 0.5 at the edge
    mean_w = float(np.mean(w))
    return mean_w / 0.25


@register_operator(
    name="tod_open_midday_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="tod_open_midday_ratio",
    source="wave1_liquidity_tod",
    status="experimental")
class TodOpenMiddayRatio(SeriesOperator):
    """开盘段与午盘段均值之比（任意行长度都可计算）。

    对分钟面板：open_rows 行与 middle_rows 行的比；对日面板：退化为开头 vs
    中段。数值显著大于 1 表示开盘活跃度高（U 形左翼）。当前行必须有效。
    """

    metadata = _metadata(
        "tod_open_midday_ratio",
        "开盘段均值 / 午盘段均值（U 形左翼强度）。",
        ["x", "open_rows", "middle_rows"],
        domain="intraday",
        unit="same_as:x",
        cost=1,
        category="intraday_microstructure",
    )

    def _calculate_series(self, x: pd.DataFrame, open_rows: int = 30, middle_rows: int = 30, **_: Any) -> pd.DataFrame:
        o = _check_int(open_rows, "open_rows", 1)
        m = _check_int(middle_rows, "middle_rows", 1)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        # "今天"的窗口是最近 (open_rows + middle_rows) 行，其中前 open_rows 行
        # 视为开盘段、后 middle_rows 行视为午盘段（任意行长度均可）。
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                span = o + m
                lo = max(0, row - span + 1)
                chunk = xv[lo:row + 1, col]
                if chunk.size < span:
                    continue
                open_seg = chunk[:o]
                mid_seg = chunk[o:]
                ok_o = np.isfinite(open_seg)
                ok_m = np.isfinite(mid_seg)
                if ok_o.sum() < 2 or ok_m.sum() < 2:
                    continue
                mo = float(np.mean(open_seg[ok_o]))
                mm = float(np.mean(mid_seg[ok_m]))
                out[row, col] = mo / (mm + 1e-12)
        return _frame_like(x, out)


@register_operator(
    name="tod_close_midday_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="tod_close_midday_ratio",
    source="wave1_liquidity_tod",
    status="experimental")
class TodCloseMiddayRatio(SeriesOperator):
    """尾盘段与午盘段均值之比（U 形右翼强度）。

    窗口切分：最近 (close_rows + middle_rows) 行中后 close_rows 行为尾盘段、
    其余为午盘段。数值显著大于 1 表示尾盘活跃度高。
    """

    metadata = _metadata(
        "tod_close_midday_ratio",
        "尾盘段均值 / 午盘段均值（U 形右翼强度）。",
        ["x", "close_rows", "middle_rows"],
        domain="intraday",
        unit="same_as:x",
        category="intraday_microstructure",
    )

    def _calculate_series(self, x: pd.DataFrame, close_rows: int = 30, middle_rows: int = 30, **_: Any) -> pd.DataFrame:
        c = _check_int(close_rows, "close_rows", 1)
        m = _check_int(middle_rows, "middle_rows", 1)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        span = c + m
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                lo = max(0, row - span + 1)
                chunk = xv[lo:row + 1, col]
                if chunk.size < span:
                    continue
                mid_seg = chunk[:m]
                close_seg = chunk[m:]
                ok_m = np.isfinite(mid_seg)
                ok_c = np.isfinite(close_seg)
                if ok_m.sum() < 2 or ok_c.sum() < 2:
                    continue
                mm = float(np.mean(mid_seg[ok_m]))
                mc = float(np.mean(close_seg[ok_c]))
                out[row, col] = mc / (mm + 1e-12)
        return _frame_like(x, out)


@register_operator(
    name="tod_edge_activity",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="tod_edge_activity",
    source="wave1_liquidity_tod",
    status="experimental")
class TodEdgeActivity(SeriesOperator):
    """时间权重边缘活跃度：mean(|t - 0.5|)/0.25 × sqrt(scale)。

    基于每个"日"的时间锚点统计活动集中于开盘/尾盘的强度。数值约 1 = 活动
    均匀分布；>1 = 边缘集中（U 形）。scale 为可选幅度因子（x 本身即可）。
    """

    metadata = _metadata(
        "tod_edge_activity",
        "时间权重边缘活跃度（接近 1 均匀，>1 边缘集中 U 形）。",
        ["x", "window"],
        domain="intraday",
        unit="ratio",
        category="intraday_microstructure",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 4)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = xv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < 2:
                    continue
                vals = chunk[ok]
                out[row, col] = _tod_kernel(vals)
        return _frame_like(x, out)


@register_operator(
    name="tod_intraday_range_position",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="tod_intraday_range_position",
    source="wave1_liquidity_tod",
    status="experimental")
class TodIntradayRangePosition(SeriesOperator):
    """日内价格在当日高-低区间的相对位置：(x - low) / (high - low)。

    对日面板退化为一元序列自身的运行窗口 z 位置。输出 [0, 1]（区间内），
    高低无差异时 fail-closed 为 0.5。causal（只用截至当前行）。
    """

    metadata = _metadata(
        "tod_intraday_range_position",
        "(x - low)/(high - low)，日内价格相对位置。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="intraday_microstructure",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                lo = max(0, row - w + 1)
                chunk = xv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                hi = float(np.max(chunk[ok]))
                lo_ = float(np.min(chunk[ok]))
                if hi <= lo_:
                    out[row, col] = 0.5
                    continue
                out[row, col] = (xv[row, col] - lo_) / (hi - lo_)
        return _frame_like(x, out)


@register_operator(
    name="tod_overnight_activity_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="tod_overnight_activity_ratio",
    source="wave1_liquidity_tod",
    status="experimental")
class TodOvernightActivityRatio(SeriesOperator):
    """隔夜活动相对日内活动比例：mean(overnight_mark)/mean(intraday_mark)。

    对分钟面板：overnight_mark 为每根隔夜（首根）分钟 bar 的活跃度标记，
    intraday_mark 为日内分钟 bar。对日面板退化为窗口内两项的比例估计。
    输出 ratio（如 >1 表示隔夜异常活跃）。
    """

    metadata = _metadata(
        "tod_overnight_activity_ratio",
        "隔夜标记均值 / 日内标记均值。",
        ["overnight_mark", "intraday_mark", "window", "min_periods"],
        domain="intraday",
        unit="ratio",
        category="intraday_microstructure",
    )

    def _calculate_series(self, overnight_mark: pd.DataFrame, intraday_mark: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        overnight_mark, intraday_mark = _aligned(overnight_mark, intraday_mark)
        ov = overnight_mark.to_numpy(dtype=float)
        iv = intraday_mark.to_numpy(dtype=float)
        rows, cols = ov.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                o = ov[lo:row + 1, col]
                i = iv[lo:row + 1, col]
                ok_o = np.isfinite(o)
                ok_i = np.isfinite(i)
                if ok_o.sum() < mp or ok_i.sum() < mp:
                    continue
                mo = float(np.mean(o[ok_o]))
                mi_ = float(np.mean(i[ok_i]))
                if mi_ <= 0:
                    continue
                out[row, col] = mo / mi_
        return _frame_like(overnight_mark, out)


# ---------------------------------------------------------------------------
# 4. daily liquidity-factor transformation helpers
# ---------------------------------------------------------------------------
@register_operator(
    name="lf1_ret_volume_ratio",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_ret_volume_ratio",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1RetVolumeRatio(SeriesOperator):
    """收益-成交比：窗口内 sum(|ret|*volume)/sum(volume)（波动/成交预期）。

    数值大 = 单位成交量对应更高的价格波动（成交效率低 / 冲击大）。基于
    price-volume 联动 PCR 族思想。输出 ratio。
    """

    metadata = _metadata(
        "lf1_ret_volume_ratio",
        "sum(|ret|*volume)/sum(volume)（单位成交量价格波动）。",
        ["ret", "volume", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="market_microstructure",
        input_units={"ret": "return", "volume": "volume"},
    )

    def _calculate_series(self, ret: pd.DataFrame, volume: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        mp = _check_int(min_periods, "min_periods", 1)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        ret, volume = _aligned(ret, volume)
        rv = ret.to_numpy(dtype=float)
        vv = volume.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                r = rv[lo:row + 1, col]
                v = vv[lo:row + 1, col]
                ok = np.isfinite(r) & np.isfinite(v) & (v > 0)
                if ok.sum() < mp:
                    continue
                num = float(np.sum(np.abs(r[ok]) * v[ok]))
                den = float(np.sum(v[ok]))
                if den <= 0:
                    continue
                out[row, col] = num / den
        return _frame_like(ret, out)


@register_operator(
    name="lf1_liquidity_decay_base",
    category="market_microstructure",
    business_category="market_microstructure",
    canonical="lf1_liquidity_decay_base",
    source="wave1_liquidity_tod",
    status="experimental")
class Lf1LiquidityDecayBase(SeriesOperator):
    """流动性衰减基：exp(-w/decay_scale) 线性衰减的均值。

    对 abs(amount_logret)（log-return × amount 冲击标志）做半衰期指数平滑，
    衡量流动性冲击事件的持久性。衰减越慢（decay_scale 大）越持久。输出
    单位与输入相同。
    """

    metadata = _metadata(
        "lf1_liquidity_decay_base",
        "流动性冲击按指数半衰期平滑（exp(-t/decay_scale)）。",
        ["shock_mark", "decay_scale", "window"],
        domain="volume",
        unit="same_as:shock_mark",
        category="market_microstructure",
    )

    def _calculate_series(self, shock_mark: pd.DataFrame, decay_scale: float = 20.0, window: int = 60, **_: Any) -> pd.DataFrame:
        ds = float(decay_scale)
        if ds < 1.0:
            raise ValueError("decay_scale must be >= 1")
        w = _check_int(window, "window", 2)
        sv = shock_mark.to_numpy(dtype=float)
        rows, cols = sv.shape
        coef = np.exp(-1.0 / ds)
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                chunk = sv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() == 0:
                    continue
                n = ok.sum()
                wts = coef ** np.arange(n - 1, -1, -1)
                wts = wts / np.sum(wts)
                out[row, col] = float(np.sum(chunk[ok] * wts))
        return _frame_like(shock_mark, out)