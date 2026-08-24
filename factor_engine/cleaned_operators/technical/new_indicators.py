# -*- coding: utf-8 -*-
"""New daily technical indicators (2026-08 R47).

Daily wide panels (columns = instruments, index = dates) in, daily wide panels
out.  All seven operators are causal trailing-only, deterministic and NaN-until-
warmup (never filled with 0).

  1. HMA                  -- Hull moving average (WMA of 2*WMA(n/2) - WMA(n)).
  2. QQE                  -- RSI-based adaptive band / trend state machine.
  3. RSX                  -- Jurik-inspired low-lag RSI (multi-stage cascade).
  4. ALMA                 -- Arnaud Legoux moving average (Gaussian weights).
  5. CoppockCurve         -- WMA of (ROC(roc1) + ROC(roc2)).
  6. ElderRay             -- high/low minus EMA(close) bull/bear power.
  7. FisherTransform      -- Fisher transform of a smoothed raw position.

Registration follows the ``technical_signal`` family convention (see
``cleaned_operators/technical/signal.py``): ``@register_operator(...)`` on a
``SeriesOperator`` subclass, canonical = operator name, source =
``technical.new_indicators``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12
_CANONICALS: list[str] = []


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "technical_signal", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:technical_signal",
            f"unit:{unit}", "cost:1",
        ],
        param_specs=param_specs or {},
    )


def _wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average, weights 1..w newest-largest, normalised by sum."""
    w = max(1, int(window))
    weights = np.arange(1, w + 1, dtype=float)
    return series.rolling(window=w, min_periods=w).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder-smoothed RSI for a single series."""
    w = max(2, int(window))
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _round_rule(x: float, rounding: str) -> int:
    if rounding == "floor":
        return int(np.floor(x))
    return int(round(x))


def _two_pole_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """Single two-pole critically-damped low-pass stage.

    ``y_t = alpha^2*x_t + 2*beta*y_{t-1} - beta^2*y_{t-2}``, ``beta = 1-alpha``,
    with zero initial conditions.  DC gain is exactly 1.
    """
    n = len(x)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    a = alpha * alpha
    b = 1.0 - alpha
    out[0] = a * x[0]
    if n > 1:
        out[1] = a * x[1] + 2.0 * b * out[0]
    for t in range(2, n):
        out[t] = a * x[t] + 2.0 * b * out[t - 1] - b * b * out[t - 2]
    return out


def _two_pole_cascade(x: np.ndarray, alpha: float, stages: int = 3) -> np.ndarray:
    out = np.asarray(x, dtype=float)
    for _ in range(int(stages)):
        out = _two_pole_filter(out, alpha)
    return out


# ---------------------------------------------------------------------------
# 1. HMA -- Hull moving average
# ---------------------------------------------------------------------------
def _hma_series(x: pd.Series, window: int, rounding: str) -> pd.Series:
    w = max(2, int(window))
    n2 = _round_rule(w / 2.0, rounding)
    ns = _round_rule(np.sqrt(w), rounding)
    n2 = max(1, n2)
    ns = max(1, ns)
    wma_w = _wma(x, w)
    wma_n2 = _wma(x, n2)
    inner = 2.0 * wma_n2 - wma_w
    return _wma(inner, ns)


@register_operator(
    name="HMA",
    category="technical_signal",
    business_category="technical_signal",
    canonical="HMA",
    source="technical.new_indicators",
)
class HMA(SeriesOperator):
    """Hull moving average: WMA(2*WMA(x, round_rule(w/2)) - WMA(x, w), round_rule(sqrt(w))).

    Weights of every WMA are 1..w newest-largest, normalised by their sum.  The
    result is NaN until every component WMA has a full valid window.
    """

    metadata = _meta(
        "HMA",
        "赫尔移动平均：2*WMA(w/2) - WMA(w) 再 WMA(sqrt(w))。",
        ["x", "window", "rounding"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "rounding": ParamSpec(dtype=str, choices=("floor", "round")),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 16, rounding: str = "floor", **_: Any
    ) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _hma_series(x[col], int(window), str(rounding))
        return out


# ---------------------------------------------------------------------------
# 2. QQE -- RSI adaptive bands / trend state machine
# ---------------------------------------------------------------------------
def _qqe_series(x: pd.Series, length: int, smooth: int, factor: float) -> dict[str, pd.Series]:
    n = len(x)
    rsi = _wilder_rsi(x, length)
    rsi_ma = rsi.rolling(window=smooth, min_periods=smooth).mean()  # "line"
    span = 2 * length - 1
    abs_drift = rsi_ma.diff().abs()
    rng = abs_drift.ewm(span=span, adjust=False).mean().ewm(span=span, adjust=False).mean()
    da = rng * factor
    base_long = rsi_ma - da
    base_short = rsi_ma + da
    basis = rsi_ma.rolling(window=span, min_periods=span).mean()

    long_band = np.full(n, np.nan)
    short_band = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    carry_long = np.nan
    carry_short = np.nan
    carry_trend = 0.0
    for i in range(n):
        s = rsi_ma.iloc[i]
        if not np.isfinite(s):
            carry_long = np.nan
            carry_short = np.nan
            carry_trend = 0.0
            continue
        bl = base_long.iloc[i]
        bs = base_short.iloc[i]
        if not np.isfinite(bl) or not np.isfinite(bs):
            carry_long = np.nan
            carry_short = np.nan
            carry_trend = 0.0
            continue
        if not np.isfinite(carry_long):
            # re-seed the band state machine on the first finite value / after gap
            long_band[i] = bl
            short_band[i] = bs
            trend[i] = 0.0
            carry_long = bl
            carry_short = bs
            carry_trend = 0.0
            continue
        # trailing band switch
        if s > carry_long:
            long_band[i] = max(bl, carry_long)
        else:
            long_band[i] = bl
        if s < carry_short:
            short_band[i] = min(bs, carry_short)
        else:
            short_band[i] = bs
        # trend flips when the smoothed RSI crosses the PRIOR opposite band
        if s > carry_short:
            carry_trend = 1.0
        elif s < carry_long:
            carry_trend = -1.0
        trend[i] = carry_trend
        carry_long = long_band[i]
        carry_short = short_band[i]

    return {
        "line": rsi_ma,
        "basis": basis,
        "long": pd.Series(long_band, index=x.index),
        "short": pd.Series(short_band, index=x.index),
        "trend": pd.Series(trend, index=x.index),
    }


@register_operator(
    name="QQE",
    category="technical_signal",
    business_category="technical_signal",
    canonical="QQE",
    source="technical.new_indicators",
)
class QQE(SeriesOperator):
    """RSI-based adaptive-band QQE (Quantitative Qualitative Estimator).

    ``line`` is the smoothed RSI (MA(smooth) of Wilder RSI(length)); the band
    width is ``factor`` times the double-EMA(span = 2*length-1) of the absolute
    drift of ``line``.  ``long``/``short`` carry the trailing adaptive bands
    (flipped when the smoothed RSI crosses the prior opposite band); ``trend``
    is +1 long / -1 short / 0 flat.  ``basis`` is the span mean of ``line``.
    Stateful recursion resets after a hard gap.  NaN until warmup.
    """

    metadata = _meta(
        "QQE",
        "QQE 自适应带/趋势：RSI 平滑 + 双 EMA 波动带。",
        ["x", "length", "smooth", "factor", "output"],
        unit="level",
        param_specs={
            "length": ParamSpec(dtype=int, min=2),
            "smooth": ParamSpec(dtype=int, min=1),
            "factor": ParamSpec(dtype=float, min=1e-6),
            "output": ParamSpec(dtype=str, choices=("line", "basis", "long", "short", "trend")),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        length: int = 14,
        smooth: int = 5,
        factor: float = 4.236,
        output: str = "line",
        **_: Any,
    ) -> pd.DataFrame:
        length = max(2, int(length))
        smooth = max(1, int(smooth))
        factor = float(factor)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            series = _qqe_series(x[col], length, smooth, factor)
            out[col] = series[str(output)]
        return out


# ---------------------------------------------------------------------------
# 3. RSX -- Jurik-inspired low-lag RSI
# ---------------------------------------------------------------------------
def _rsx_segment(seg: np.ndarray, length: int) -> np.ndarray:
    alpha = 3.0 / (length + 2.0)
    f88 = 100.0 / (length + 1.0)  # price scale (cancels in the ratio)
    scaled = seg * f88
    m = len(seg)
    change = np.zeros(m, dtype=float)
    if m > 1:
        change[1:] = np.diff(scaled)
    abs_change = np.abs(change)
    signed = _two_pole_cascade(change, alpha, stages=3)
    abss = _two_pole_cascade(abs_change, alpha, stages=3)
    ratio = np.zeros(m, dtype=float)
    mask = abss > _EPS
    ratio[mask] = signed[mask] / abss[mask]
    rsx = 50.0 * (ratio + 1.0)
    return np.clip(rsx, 0.0, 100.0)


def _rsx_series(x: pd.Series, length: int) -> pd.Series:
    vals = x.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n < 1:
        return pd.Series(out, index=x.index)
    # process contiguous finite segments; reset the cascade after a hard gap
    seg_start = None
    for t in range(n):
        if not np.isfinite(vals[t]):
            if seg_start is not None:
                seg_end = t
                rsx = _rsx_segment(vals[seg_start:seg_end], length)
                start_out = seg_start + (length - 1)
                if start_out < seg_end:
                    out[start_out:seg_end] = rsx[start_out - seg_start:]
                seg_start = None
            continue
        if seg_start is None:
            seg_start = t
    if seg_start is not None:
        seg_end = n
        rsx = _rsx_segment(vals[seg_start:seg_end], length)
        start_out = seg_start + (length - 1)
        if start_out < seg_end:
            out[start_out:seg_end] = rsx[start_out - seg_start:]
    return pd.Series(out, index=x.index)


@register_operator(
    name="RSX",
    category="technical_signal",
    business_category="technical_signal",
    canonical="RSX",
    source="technical.new_indicators",
)
class RSX(SeriesOperator):
    """Jurik-inspired low-lag RSI.

    Price is scaled (``f88``), then the signed and absolute one-step changes are
    smoothed through three cascaded two-pole low-pass stages (alpha =
    3/(length+2), beta = 1-alpha); output = 50*(signed/abs + 1), clamped to
    [0, 100].  Emits the neutral 50 during cascade initialization.  Output is
    NaN for the first ``length-1`` observations of each finite segment; the
    cascade resets after a hard gap.
    """

    metadata = _meta(
        "RSX",
        "Jurik 低滞后 RSI：三阶双极点级联平滑的有符号/绝对变化比。",
        ["x", "length"],
        unit="level",
        param_specs={"length": ParamSpec(dtype=int, min=2)},
    )

    def _calculate_series(self, x: pd.DataFrame, length: int = 14, **_: Any) -> pd.DataFrame:
        length = max(2, int(length))
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _rsx_series(x[col], length)
        return out


# ---------------------------------------------------------------------------
# 4. ALMA -- Arnaud Legoux moving average
# ---------------------------------------------------------------------------
def _alma_series(x: pd.Series, window: int, offset: float, sigma: float) -> pd.Series:
    w = max(2, int(window))
    m = float(offset) * (w - 1)
    s = w / max(float(sigma), 1e-6)
    weights = np.array([np.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(w)], dtype=float)
    total = weights.sum()
    if total <= _EPS:
        weights = np.ones(w, dtype=float) / w
    else:
        weights = weights / total
    return x.rolling(window=w, min_periods=w).apply(
        lambda vals: float(np.dot(vals, weights)), raw=True
    )


@register_operator(
    name="ALMA",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ALMA",
    source="technical.new_indicators",
)
class ALMA(SeriesOperator):
    """Arnaud Legoux moving average with Gaussian weights.

    Weights ``w_i = exp(-(i - m)^2 / (2 s^2))`` for i = 0 (oldest) .. window-1
    (newest), with ``m = offset*(window-1)`` and ``s = window/sigma``, normalised
    to sum 1.  NaN until a full valid window.
    """

    metadata = _meta(
        "ALMA",
        "Arnaud Legoux 移动平均：offset 偏移高斯加权。",
        ["x", "window", "offset", "sigma"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "offset": ParamSpec(dtype=float, min=0.0, max=1.0),
            "sigma": ParamSpec(dtype=float, min=1e-6),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 10, offset: float = 0.85, sigma: float = 6.0, **_: Any
    ) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _alma_series(x[col], int(window), float(offset), float(sigma))
        return out


# ---------------------------------------------------------------------------
# 5. CoppockCurve
# ---------------------------------------------------------------------------
def _coppock_series(close: pd.Series, roc1: int, roc2: int, wma_window: int, roc_mode: str) -> pd.Series:
    r1 = close / close.shift(roc1) - 1.0 if roc_mode == "pct" else np.log(close / close.shift(roc1))
    r2 = close / close.shift(roc2) - 1.0 if roc_mode == "pct" else np.log(close / close.shift(roc2))
    total = r1 + r2
    return _wma(total, wma_window)


@register_operator(
    name="CoppockCurve",
    category="technical_signal",
    business_category="technical_signal",
    canonical="CoppockCurve",
    source="technical.new_indicators",
)
class CoppockCurve(SeriesOperator):
    """Coppock curve: WMA of (ROC(roc1) + ROC(roc2)).

    ROC(k) = close[t]/close[t-k] - 1 for ``roc_mode="pct"`` and ln(...) for
    ``roc_mode="log"``.  WMA weights are 1..w newest-largest.  NaN until both
    ROCs and the WMA are warm.
    """

    metadata = _meta(
        "CoppockCurve",
        "Coppock 曲线：ROC(roc1)+ROC(roc2) 的 WMA。",
        ["close", "roc1", "roc2", "wma_window", "roc_mode"],
        unit="level",
        param_specs={
            "roc1": ParamSpec(dtype=int, min=1),
            "roc2": ParamSpec(dtype=int, min=1),
            "wma_window": ParamSpec(dtype=int, min=1),
            "roc_mode": ParamSpec(dtype=str, choices=("pct", "log")),
        },
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        roc1: int = 14,
        roc2: int = 11,
        wma_window: int = 10,
        roc_mode: str = "pct",
        **_: Any,
    ) -> pd.DataFrame:
        roc1 = max(1, int(roc1))
        roc2 = max(1, int(roc2))
        wma_window = max(1, int(wma_window))
        out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
        for col in close.columns:
            out[col] = _coppock_series(close[col], roc1, roc2, wma_window, str(roc_mode))
        return out


# ---------------------------------------------------------------------------
# 6. ElderRay
# ---------------------------------------------------------------------------
@register_operator(
    name="ElderRay",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ElderRay",
    source="technical.new_indicators",
)
class ElderRay(SeriesOperator):
    """Elder Ray: bull = high - EMA(close); bear = low - EMA(close); spread = high-low.

    EMA alpha = 2/(ema+1), ``adjust=False``, ``min_periods=ema``.  NaN until the
    EMA is warm.
    """

    metadata = _meta(
        "ElderRay",
        "Elder Ray：多头/空头力量（high/low 减 EMA(close)）。",
        ["high", "low", "close", "ema", "output"],
        unit="price",
        param_specs={
            "ema": ParamSpec(dtype=int, min=1),
            "output": ParamSpec(dtype=str, choices=("bull", "bear", "spread")),
        },
    )

    def _calculate_series(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        ema: int = 13,
        output: str = "bull",
        **_: Any,
    ) -> pd.DataFrame:
        ema_p = max(1, int(ema))
        e = close.ewm(alpha=2.0 / (ema_p + 1.0), adjust=False, min_periods=ema_p).mean()
        if output == "bull":
            return high - e
        if output == "bear":
            return low - e
        if output == "spread":
            return high - low
        raise ValueError(f"ElderRay: unknown output {output!r}")


# ---------------------------------------------------------------------------
# 7. FisherTransform
# ---------------------------------------------------------------------------
def _fisher_series(
    high: pd.Series,
    low: pd.Series,
    window: int,
    smooth: float,
    signal_smooth: float,
) -> dict[str, pd.Series]:
    n = len(high)
    source = (high + low) / 2.0
    roll_min = source.rolling(window=window, min_periods=window).min()
    roll_max = source.rolling(window=window, min_periods=window).max()
    rng = roll_max - roll_min
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 2.0 * ((source - roll_min) / rng - 0.5)
    # NaN until the rolling range is valid; exactly 0 on a degenerate zero-range
    # window (max == min), never a fabricated value.
    valid_range = rng.notna()
    raw = raw.where(valid_range)
    raw = raw.mask(valid_range & (rng <= _EPS), 0.0)

    raw_v = raw.to_numpy(dtype=float)
    z = np.full(n, np.nan)
    prev = 0.0
    started = False
    for t in range(n):
        if not np.isfinite(raw_v[t]):
            prev = 0.0
            started = False
            continue
        if not started:
            z[t] = smooth * raw_v[t]
            started = True
        else:
            z[t] = smooth * raw_v[t] + (1.0 - smooth) * prev
        prev = z[t]
    z = np.clip(z, -0.999, 0.999)
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = 0.5 * np.log((1.0 + z) / (1.0 - z))

    # signal = lagged exponential smoothing of fisher (alpha = signal_smooth)
    sig = np.full(n, np.nan)
    prev_sig = 0.0
    sig_started = False
    for t in range(n):
        f = fisher[t]
        if not np.isfinite(f):
            prev_sig = 0.0
            sig_started = False
            continue
        if not sig_started:
            sig[t] = signal_smooth * f
            sig_started = True
        else:
            sig[t] = signal_smooth * f + (1.0 - signal_smooth) * prev_sig
        prev_sig = sig[t]
    signal = pd.Series(sig, index=high.index).shift(1)  # lagged
    fisher_s = pd.Series(fisher, index=high.index)
    return {"value": fisher_s, "signal": signal, "trigger": fisher_s - signal}


@register_operator(
    name="FisherTransform",
    category="technical_signal",
    business_category="technical_signal",
    canonical="FisherTransform",
    source="technical.new_indicators",
)
class FisherTransform(SeriesOperator):
    """Fisher transform of a smoothed rolling position.

    ``source = (high+low)/2``; rolling min/max over ``window``; raw position
    ``2*((source-min)/(max-min) - 0.5)`` (0 on zero range); recursively smoothed
    ``z_t = smooth*raw + (1-smooth)*z_{t-1}`` and clipped to [-0.999, 0.999];
    ``fisher = 0.5*ln((1+z)/(1-z))``.  ``signal`` = lag of the exponentially
    smoothed fisher (alpha = signal_smooth); ``trigger`` = fisher - signal.  NaN
    until the rolling range is valid.
    """

    metadata = _meta(
        "FisherTransform",
        "Fisher 变换：平滑后的 (high+low)/2 位置取 ln((1+z)/(1-z))/2。",
        ["high", "low", "window", "smooth", "signal_smooth", "output"],
        unit="level",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "smooth": ParamSpec(dtype=float, min=0.0, max=1.0),
            "signal_smooth": ParamSpec(dtype=float, min=0.0, max=1.0),
            "output": ParamSpec(dtype=str, choices=("value", "signal", "trigger")),
        },
    )

    def _calculate_series(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 9,
        smooth: float = 0.33,
        signal_smooth: float = 0.5,
        output: str = "value",
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        smooth = float(smooth)
        signal_smooth = float(signal_smooth)
        out = pd.DataFrame(np.nan, index=high.index, columns=high.columns, dtype=float)
        for col in high.columns:
            series = _fisher_series(high[col], low[col], window, smooth, signal_smooth)
            out[col] = series[str(output)]
        return out


_CANONICALS.extend(["HMA", "QQE", "RSX", "ALMA", "CoppockCurve", "ElderRay", "FisherTransform"])


def _register_surface() -> None:
    """注册到 extended surface 并添加 Polars 后端支持。"""
    import factor_engine.cleaned_operators.operator_surface as _surface
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    _surface.extend_extended_only(set(_CANONICALS))

    # Polars 后端：委托 pandas reference
    for _canon in _CANONICALS:
        register_polars_bridge(_canon)


_register_surface()

