# -*- coding: utf-8 -*-
"""Production-oriented price/volume structure and technical-analysis extensions.

Design rules
------------
* Every operator is causal: only current/past observations are used.
* Breakout/support baselines exclude the current bar where appropriate.
* Confirmed pivots emit on the *confirmation* timestamp, never retroactively on
  the pivot timestamp.
* Public pattern outputs are deterministic numeric panels suitable for models.
* Simple formulas may later be lowered to existing primitives; these Pandas/
  NumPy implementations remain the semantic reference backend.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12


def _w(value: int, *, minimum: int = 1) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"window must be >= {minimum}")
    return value


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / den.replace(0, np.nan)


def _prev_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _w(window)
    return x.shift(1).rolling(w, min_periods=w).max()


def _prev_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _w(window)
    return x.shift(1).rolling(w, min_periods=w).min()


def _true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=0,
        keys=["hl", "hc", "lc"],
    ).groupby(level=1).max()


def _rolling_mad(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _w(window, minimum=2)
    return frame.rolling(w, min_periods=w).apply(
        lambda a: float(np.mean(np.abs(a - np.mean(a)))), raw=True
    )


def _register(
    name: str,
    param_names: Iterable[str],
    fn: Callable[..., pd.DataFrame],
    *,
    category: str,
    description: str,
    tags: Iterable[str] = (),
) -> None:
    metadata = OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=list(param_names),
        return_type="series",
        tags=[*tags, "pit_safe", "causal", "production_extension"],
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"Production_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category=category,
        business_category="technical_extension",
        canonical=name,
        source="production_technical_extensions",
        backend="pandas_numpy",
        status="production",
    )(cls)


# ---------------------------------------------------------------------------
# 1. Price structure / breakout primitives
# ---------------------------------------------------------------------------


def _ts_prev_high(x, window):
    return _prev_max(x, window)


def _ts_prev_low(x, window):
    return _prev_min(x, window)


def _ts_distance_to_high(x, window):
    return _safe_div(x, _prev_max(x, window)) - 1.0


def _ts_distance_to_low(x, window):
    return _safe_div(x, _prev_min(x, window)) - 1.0


def _ts_breakout_high(x, window):
    return (_safe_div(x, _prev_max(x, window)) - 1.0).clip(lower=0.0)


def _ts_breakdown_low(x, window):
    return (_safe_div(_prev_min(x, window), x) - 1.0).clip(lower=0.0)


def _ts_new_high(x, window):
    return x.gt(_prev_max(x, window)).astype(float)


def _ts_new_low(x, window):
    return x.lt(_prev_min(x, window)).astype(float)


def _ts_channel_position(x, window):
    hi, lo = _prev_max(x, window), _prev_min(x, window)
    return _safe_div(x - lo, hi - lo)


def _ts_days_since_extreme(x: pd.DataFrame, window: int, *, high: bool) -> pd.DataFrame:
    w = _w(window)
    shifted = x.shift(1)
    fn = (
        (lambda a: float(len(a) - 1 - int(np.nanargmax(a))))
        if high
        else (lambda a: float(len(a) - 1 - int(np.nanargmin(a))))
    )
    return shifted.rolling(w, min_periods=w).apply(fn, raw=True)


def _ts_days_since_high(x, window):
    return _ts_days_since_extreme(x, window, high=True)


def _ts_days_since_low(x, window):
    return _ts_days_since_extreme(x, window, high=False)


def _ts_range_expansion(high, low, window):
    current = high - low
    baseline = current.shift(1).rolling(_w(window), min_periods=_w(window)).mean()
    return _safe_div(current, baseline) - 1.0


for _name, _params, _fn, _desc in [
    ("ts_prev_high", ["x", "window"], _ts_prev_high, "Previous-window high excluding current bar."),
    ("ts_prev_low", ["x", "window"], _ts_prev_low, "Previous-window low excluding current bar."),
    ("ts_distance_to_high", ["x", "window"], _ts_distance_to_high, "Signed distance to prior rolling high."),
    ("ts_distance_to_low", ["x", "window"], _ts_distance_to_low, "Signed distance to prior rolling low."),
    ("ts_breakout_high", ["x", "window"], _ts_breakout_high, "Positive breakout magnitude above prior high."),
    ("ts_breakdown_low", ["x", "window"], _ts_breakdown_low, "Positive breakdown magnitude below prior low."),
    ("ts_new_high", ["x", "window"], _ts_new_high, "1 when current value exceeds prior rolling high."),
    ("ts_new_low", ["x", "window"], _ts_new_low, "1 when current value falls below prior rolling low."),
    ("ts_channel_position", ["x", "window"], _ts_channel_position, "Current position inside prior high-low channel."),
    ("ts_days_since_high", ["x", "window"], _ts_days_since_high, "Bars since the most recent high inside the prior window."),
    ("ts_days_since_low", ["x", "window"], _ts_days_since_low, "Bars since the most recent low inside the prior window."),
    ("ts_range_expansion", ["high", "low", "window"], _ts_range_expansion, "Current high-low range versus prior rolling average range."),
]:
    _register(_name, _params, _fn, category="price_structure", description=_desc)


# ---------------------------------------------------------------------------
# 2. Confirmed pivots / support / resistance
# ---------------------------------------------------------------------------


def _confirmed_pivot_events(
    frame: pd.DataFrame,
    left_window: int,
    right_window: int,
    *,
    high: bool,
) -> tuple[np.ndarray, np.ndarray]:
    left = _w(left_window)
    right = _w(right_window)
    values = frame.to_numpy(dtype=float)
    rows, cols = values.shape
    price_at_confirm = np.full((rows, cols), np.nan, dtype=float)
    pivot_pos_at_confirm = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        s = values[:, col]
        for pivot in range(left, rows - right):
            segment = s[pivot - left : pivot + right + 1]
            if not np.isfinite(segment).all():
                continue
            center = s[pivot]
            others = np.delete(segment, left)
            is_pivot = bool(center > np.max(others)) if high else bool(center < np.min(others))
            if not is_pivot:
                continue
            confirm = pivot + right
            price_at_confirm[confirm, col] = center
            pivot_pos_at_confirm[confirm, col] = float(pivot)
    return price_at_confirm, pivot_pos_at_confirm


def _confirmed_pivot_frame(frame, left_window, right_window, *, high):
    price, _ = _confirmed_pivot_events(frame, left_window, right_window, high=high)
    return pd.DataFrame(price, index=frame.index, columns=frame.columns)


def _pivot_line(
    frame: pd.DataFrame,
    left_window: int,
    right_window: int,
    points: int,
    *,
    high: bool,
    output: str,
) -> pd.DataFrame:
    k = _w(points, minimum=2)
    prices, positions = _confirmed_pivot_events(frame, left_window, right_window, high=high)
    rows, cols = prices.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        history: deque[tuple[float, float]] = deque(maxlen=k)
        for t in range(rows):
            if np.isfinite(prices[t, col]) and np.isfinite(positions[t, col]):
                history.append((positions[t, col], prices[t, col]))
            if len(history) < 2:
                continue
            xs = np.asarray([p[0] for p in history], dtype=float)
            ys = np.asarray([p[1] for p in history], dtype=float)
            xbar, ybar = xs.mean(), ys.mean()
            denom = float(np.dot(xs - xbar, xs - xbar))
            if denom <= 0:
                continue
            slope = float(np.dot(xs - xbar, ys - ybar) / denom)
            out[t, col] = slope if output == "slope" else ybar + slope * (float(t) - xbar)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _last_pivot(frame, left_window, right_window, *, high):
    confirmed = _confirmed_pivot_frame(frame, left_window, right_window, high=high)
    return confirmed.ffill()


def _pivot_age(frame, left_window, right_window, *, high):
    confirmed = _confirmed_pivot_frame(frame, left_window, right_window, high=high)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for col in frame.columns:
        age = np.nan
        vals = confirmed[col].to_numpy(dtype=float)
        dst = np.full(len(vals), np.nan)
        for i, value in enumerate(vals):
            if np.isfinite(value):
                age = 0.0
            elif np.isfinite(age):
                age += 1.0
            dst[i] = age
        out[col] = dst
    return out


def _resistance_level(high, left_window, right_window, points):
    return _pivot_line(high, left_window, right_window, points, high=True, output="level")


def _support_level(low, left_window, right_window, points):
    return _pivot_line(low, left_window, right_window, points, high=False, output="level")


def _resistance_slope(high, left_window, right_window, points):
    return _pivot_line(high, left_window, right_window, points, high=True, output="slope")


def _support_slope(low, left_window, right_window, points):
    return _pivot_line(low, left_window, right_window, points, high=False, output="slope")


def _distance_to_resistance(close, high, left_window, right_window, points):
    return _safe_div(close, _resistance_level(high, left_window, right_window, points)) - 1.0


def _distance_to_support(close, low, left_window, right_window, points):
    return _safe_div(close, _support_level(low, left_window, right_window, points)) - 1.0


def _resistance_break(close, high, left_window, right_window, points):
    return _distance_to_resistance(close, high, left_window, right_window, points).clip(lower=0.0)


def _support_break(close, low, left_window, right_window, points):
    return (-_distance_to_support(close, low, left_window, right_window, points)).clip(lower=0.0)


_register("ts_confirmed_pivot_high", ["high", "left_window", "right_window"], lambda high, left_window, right_window: _confirmed_pivot_frame(high, left_window, right_window, high=True), category="price_structure", description="Confirmed pivot high emitted at the confirmation timestamp.")
_register("ts_confirmed_pivot_low", ["low", "left_window", "right_window"], lambda low, left_window, right_window: _confirmed_pivot_frame(low, left_window, right_window, high=False), category="price_structure", description="Confirmed pivot low emitted at the confirmation timestamp.")
_register("ts_last_pivot_high", ["high", "left_window", "right_window"], lambda high, left_window, right_window: _last_pivot(high, left_window, right_window, high=True), category="price_structure", description="Most recent confirmed pivot-high price.")
_register("ts_last_pivot_low", ["low", "left_window", "right_window"], lambda low, left_window, right_window: _last_pivot(low, left_window, right_window, high=False), category="price_structure", description="Most recent confirmed pivot-low price.")
_register("ts_pivot_high_age", ["high", "left_window", "right_window"], lambda high, left_window, right_window: _pivot_age(high, left_window, right_window, high=True), category="price_structure", description="Bars since the latest confirmed pivot high.")
_register("ts_pivot_low_age", ["low", "left_window", "right_window"], lambda low, left_window, right_window: _pivot_age(low, left_window, right_window, high=False), category="price_structure", description="Bars since the latest confirmed pivot low.")
_register("ts_resistance_level", ["high", "left_window", "right_window", "points"], _resistance_level, category="price_structure", description="Projected resistance line from recent confirmed pivot highs.")
_register("ts_support_level", ["low", "left_window", "right_window", "points"], _support_level, category="price_structure", description="Projected support line from recent confirmed pivot lows.")
_register("ts_resistance_slope", ["high", "left_window", "right_window", "points"], _resistance_slope, category="price_structure", description="Slope of resistance line fitted to confirmed pivot highs.")
_register("ts_support_slope", ["low", "left_window", "right_window", "points"], _support_slope, category="price_structure", description="Slope of support line fitted to confirmed pivot lows.")
_register("ts_distance_to_resistance", ["close", "high", "left_window", "right_window", "points"], _distance_to_resistance, category="price_structure", description="Signed close-to-resistance distance.")
_register("ts_distance_to_support", ["close", "low", "left_window", "right_window", "points"], _distance_to_support, category="price_structure", description="Signed close-to-support distance.")
_register("ts_resistance_break", ["close", "high", "left_window", "right_window", "points"], _resistance_break, category="price_structure", description="Positive magnitude of a confirmed-resistance breakout.")
_register("ts_support_break", ["close", "low", "left_window", "right_window", "points"], _support_break, category="price_structure", description="Positive magnitude of a confirmed-support breakdown.")


# ---------------------------------------------------------------------------
# 3. Volume / price-volume confirmation
# ---------------------------------------------------------------------------


def _rolling_vwap(price, volume, window):
    w = _w(window)
    num = (price * volume).rolling(w, min_periods=w).sum()
    den = volume.rolling(w, min_periods=w).sum()
    return _safe_div(num, den)


def _vwap_deviation(price, volume, window):
    return _safe_div(price, _rolling_vwap(price, volume, window)) - 1.0


def _relative_volume(volume, window):
    baseline = volume.shift(1).rolling(_w(window), min_periods=_w(window)).mean()
    return _safe_div(volume, baseline)


def _prior_zscore(x, window):
    w = _w(window, minimum=2)
    mean = x.shift(1).rolling(w, min_periods=w).mean()
    std = x.shift(1).rolling(w, min_periods=w).std(ddof=1)
    return _safe_div(x - mean, std)


def _dollar_volume(close, volume):
    return close * volume


def _dollar_volume_zscore(close, volume, window):
    return _prior_zscore(close * volume, window)


def _volume_momentum(volume, window):
    return _safe_div(volume, volume.shift(_w(window))) - 1.0


def _turnover_momentum(turnover, window):
    return _safe_div(turnover, turnover.shift(_w(window))) - 1.0


def _turnover_zscore(turnover, window):
    return _prior_zscore(turnover, window)


def _return_volume_corr(ret, volume, window):
    return ret.rolling(_w(window), min_periods=_w(window)).corr(volume)


def _abs_return_volume_corr(ret, volume, window):
    return ret.abs().rolling(_w(window), min_periods=_w(window)).corr(volume)


def _signed_volume(ret, volume):
    return np.sign(ret) * volume


def _signed_dollar_volume(ret, close, volume):
    return np.sign(ret) * close * volume


def _rolling_obv(close, volume, window):
    flow = np.sign(close.diff()) * volume
    return flow.rolling(_w(window), min_periods=_w(window)).sum()


def _rolling_pvt(close, volume, window):
    flow = close.pct_change(fill_method=None) * volume
    return flow.rolling(_w(window), min_periods=_w(window)).sum()


def _cmf(high, low, close, volume, window):
    spread = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / spread
    mfv = mfm * volume
    w = _w(window)
    return _safe_div(mfv.rolling(w, min_periods=w).sum(), volume.rolling(w, min_periods=w).sum())


def _mfi(high, low, close, volume, window):
    tp = (high + low + close) / 3.0
    raw = tp * volume
    delta = tp.diff()
    pos = raw.where(delta > 0, 0.0)
    neg = raw.where(delta < 0, 0.0)
    w = _w(window)
    pos_sum = pos.rolling(w, min_periods=w).sum()
    neg_sum = neg.rolling(w, min_periods=w).sum()
    ratio = _safe_div(pos_sum, neg_sum)
    out = 100.0 - 100.0 / (1.0 + ratio)
    out = out.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    out = out.mask((neg_sum == 0) & (pos_sum == 0), 50.0)
    return out


for _name, _params, _fn, _desc in [
    ("rolling_vwap", ["price", "volume", "window"], _rolling_vwap, "Rolling volume-weighted average price."),
    ("vwap_deviation", ["price", "volume", "window"], _vwap_deviation, "Price deviation from rolling VWAP."),
    ("relative_volume", ["volume", "window"], _relative_volume, "Current volume divided by prior-window average volume."),
    ("volume_zscore", ["volume", "window"], _prior_zscore, "Volume z-score relative to the prior window."),
    ("dollar_volume", ["close", "volume"], _dollar_volume, "Close multiplied by volume."),
    ("dollar_volume_zscore", ["close", "volume", "window"], _dollar_volume_zscore, "Dollar-volume z-score relative to prior history."),
    ("volume_momentum", ["volume", "window"], _volume_momentum, "Volume change versus the value window bars ago."),
    ("turnover_momentum", ["turnover", "window"], _turnover_momentum, "Turnover change versus the value window bars ago."),
    ("turnover_zscore", ["turnover", "window"], _turnover_zscore, "Turnover z-score relative to the prior window."),
    ("return_volume_corr", ["ret", "volume", "window"], _return_volume_corr, "Rolling return-volume correlation."),
    ("abs_return_volume_corr", ["ret", "volume", "window"], _abs_return_volume_corr, "Rolling absolute-return/volume correlation."),
    ("signed_volume", ["ret", "volume"], _signed_volume, "Volume signed by return direction."),
    ("signed_dollar_volume", ["ret", "close", "volume"], _signed_dollar_volume, "Dollar volume signed by return direction."),
    ("rolling_obv", ["close", "volume", "window"], _rolling_obv, "Bounded-window OBV flow; avoids start-date dependence of cumulative OBV."),
    ("rolling_pvt", ["close", "volume", "window"], _rolling_pvt, "Bounded-window price-volume-trend flow."),
    ("CMF", ["high", "low", "close", "volume", "window"], _cmf, "Chaikin Money Flow."),
    ("MFI", ["high", "low", "close", "volume", "window"], _mfi, "Money Flow Index."),
]:
    _register(_name, _params, _fn, category="price_volume_extension", description=_desc)


# ---------------------------------------------------------------------------
# 4. Channels / classic bounded technical indicators
# ---------------------------------------------------------------------------


def _donchian_upper(high, window):
    return _prev_max(high, window)


def _donchian_lower(low, window):
    return _prev_min(low, window)


def _donchian_mid(high, low, window):
    return (_donchian_upper(high, window) + _donchian_lower(low, window)) / 2.0


def _donchian_position(close, high, low, window):
    upper, lower = _donchian_upper(high, window), _donchian_lower(low, window)
    return _safe_div(close - lower, upper - lower)


def _bollinger_pct_b(x, window, std_dev):
    w = _w(window, minimum=2)
    k = float(std_dev)
    mean = x.rolling(w, min_periods=w).mean()
    std = x.rolling(w, min_periods=w).std(ddof=1)
    lower, upper = mean - k * std, mean + k * std
    return _safe_div(x - lower, upper - lower)


def _bollinger_width(x, window, std_dev):
    w = _w(window, minimum=2)
    k = float(std_dev)
    mean = x.rolling(w, min_periods=w).mean()
    std = x.rolling(w, min_periods=w).std(ddof=1)
    return _safe_div(2.0 * k * std, mean.abs())


def _aroon_component(x, window, *, high):
    w = _w(window, minimum=2)
    fn = (lambda a: 100.0 * (int(np.nanargmax(a)) + 1) / len(a)) if high else (lambda a: 100.0 * (int(np.nanargmin(a)) + 1) / len(a))
    return x.rolling(w, min_periods=w).apply(fn, raw=True)


def _aroon_up(high, window):
    return _aroon_component(high, window, high=True)


def _aroon_down(low, window):
    return _aroon_component(low, window, high=False)


def _aroon(high, low, window):
    return _aroon_up(high, window) - _aroon_down(low, window)


def _cci(high, low, close, window):
    tp = (high + low + close) / 3.0
    w = _w(window, minimum=2)
    mean = tp.rolling(w, min_periods=w).mean()
    mad = _rolling_mad(tp, w)
    return _safe_div(tp - mean, 0.015 * mad)


def _stochastic_k(high, low, close, window):
    w = _w(window)
    hh, ll = high.rolling(w, min_periods=w).max(), low.rolling(w, min_periods=w).min()
    return 100.0 * _safe_div(close - ll, hh - ll)


def _stochastic_d(high, low, close, window):
    return _stochastic_k(high, low, close, window).rolling(3, min_periods=3).mean()


def _williams_r(high, low, close, window):
    w = _w(window)
    hh, ll = high.rolling(w, min_periods=w).max(), low.rolling(w, min_periods=w).min()
    return -100.0 * _safe_div(hh - close, hh - ll)


def _efficiency_ratio(close, window):
    w = _w(window)
    change = (close - close.shift(w)).abs()
    path = close.diff().abs().rolling(w, min_periods=w).sum()
    return _safe_div(change, path)


def _choppiness_index(high, low, close, window):
    w = _w(window, minimum=2)
    tr = _true_range(high, low, close)
    num = tr.rolling(w, min_periods=w).sum()
    den = high.rolling(w, min_periods=w).max() - low.rolling(w, min_periods=w).min()
    ratio = _safe_div(num, den).clip(lower=_EPS)
    return 100.0 * np.log10(ratio) / np.log10(float(w))


for _name, _params, _fn, _desc in [
    ("donchian_upper", ["high", "window"], _donchian_upper, "Prior-window Donchian upper channel."),
    ("donchian_lower", ["low", "window"], _donchian_lower, "Prior-window Donchian lower channel."),
    ("donchian_mid", ["high", "low", "window"], _donchian_mid, "Donchian channel midpoint."),
    ("donchian_position", ["close", "high", "low", "window"], _donchian_position, "Close position inside prior Donchian channel."),
    ("bollinger_pct_b", ["x", "window", "std_dev"], _bollinger_pct_b, "Bollinger percent-B."),
    ("bollinger_width", ["x", "window", "std_dev"], _bollinger_width, "Normalized Bollinger bandwidth."),
    ("AROON_up", ["high", "window"], _aroon_up, "Aroon Up from rolling highs."),
    ("AROON_down", ["low", "window"], _aroon_down, "Aroon Down from rolling lows."),
    ("AROON", ["high", "low", "window"], _aroon, "Aroon oscillator: Aroon Up minus Aroon Down."),
    ("CCI", ["high", "low", "close", "window"], _cci, "Commodity Channel Index."),
    ("StochasticK", ["high", "low", "close", "window"], _stochastic_k, "Fast stochastic percent-K."),
    ("StochasticD", ["high", "low", "close", "window"], _stochastic_d, "Three-period average of stochastic percent-K."),
    ("WilliamsR", ["high", "low", "close", "window"], _williams_r, "Williams percent-R."),
    ("efficiency_ratio", ["close", "window"], _efficiency_ratio, "Kaufman efficiency ratio: directional change divided by path length."),
    ("choppiness_index", ["high", "low", "close", "window"], _choppiness_index, "Choppiness Index based on true-range concentration."),
]:
    _register(_name, _params, _fn, category="technical_signal", description=_desc)


# ---------------------------------------------------------------------------
# 5. OHLC volatility estimators
# ---------------------------------------------------------------------------


def _parkinson_vol(high, low, window):
    w = _w(window, minimum=2)
    rs = np.log(_safe_div(high, low)).pow(2)
    var = rs.rolling(w, min_periods=w).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var.clip(lower=0.0) * 252.0)


def _garman_klass_vol(open_, high, low, close, window):
    w = _w(window, minimum=2)
    hl = np.log(_safe_div(high, low))
    co = np.log(_safe_div(close, open_))
    daily = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
    return np.sqrt(daily.rolling(w, min_periods=w).mean().clip(lower=0.0) * 252.0)


def _rogers_satchell_daily(open_, high, low, close):
    ho = np.log(_safe_div(high, open_))
    hc = np.log(_safe_div(high, close))
    lo = np.log(_safe_div(low, open_))
    lc = np.log(_safe_div(low, close))
    return ho * hc + lo * lc


def _rogers_satchell_vol(open_, high, low, close, window):
    w = _w(window, minimum=2)
    daily = _rogers_satchell_daily(open_, high, low, close)
    return np.sqrt(daily.rolling(w, min_periods=w).mean().clip(lower=0.0) * 252.0)


def _yang_zhang_vol(open_, high, low, close, window):
    w = _w(window, minimum=3)
    prev_close = close.shift(1)
    overnight = np.log(_safe_div(open_, prev_close))
    intraday = np.log(_safe_div(close, open_))
    rs = _rogers_satchell_daily(open_, high, low, close)
    k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
    var = (
        overnight.rolling(w, min_periods=w).var(ddof=1)
        + k * intraday.rolling(w, min_periods=w).var(ddof=1)
        + (1.0 - k) * rs.rolling(w, min_periods=w).mean()
    )
    return np.sqrt(var.clip(lower=0.0) * 252.0)


def _overnight_volatility(open_, close, window):
    ret = np.log(_safe_div(open_, close.shift(1)))
    return ret.rolling(_w(window, minimum=2), min_periods=_w(window, minimum=2)).std(ddof=1) * np.sqrt(252.0)


def _intraday_volatility(open_, close, window):
    ret = np.log(_safe_div(close, open_))
    return ret.rolling(_w(window, minimum=2), min_periods=_w(window, minimum=2)).std(ddof=1) * np.sqrt(252.0)


def _range_volatility(high, low, close, window):
    x = _safe_div(high - low, close.abs())
    w = _w(window, minimum=2)
    return x.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(252.0)


def _ulcer_index(close, window):
    w = _w(window, minimum=2)
    rolling_high = close.rolling(w, min_periods=w).max()
    drawdown_pct = 100.0 * (_safe_div(close, rolling_high) - 1.0)
    return np.sqrt(drawdown_pct.pow(2).rolling(w, min_periods=w).mean())


for _name, _params, _fn, _desc in [
    ("parkinson_vol", ["high", "low", "window"], _parkinson_vol, "Annualized Parkinson high-low volatility estimator."),
    ("garman_klass_vol", ["open", "high", "low", "close", "window"], _garman_klass_vol, "Annualized Garman-Klass OHLC volatility estimator."),
    ("rogers_satchell_vol", ["open", "high", "low", "close", "window"], _rogers_satchell_vol, "Annualized Rogers-Satchell OHLC volatility estimator."),
    ("yang_zhang_vol", ["open", "high", "low", "close", "window"], _yang_zhang_vol, "Annualized Yang-Zhang OHLC volatility estimator."),
    ("overnight_volatility", ["open", "close", "window"], _overnight_volatility, "Annualized volatility of open versus previous close."),
    ("intraday_volatility", ["open", "close", "window"], _intraday_volatility, "Annualized close-to-open intraday volatility."),
    ("range_volatility", ["high", "low", "close", "window"], _range_volatility, "Annualized volatility of normalized high-low range."),
    ("ulcer_index", ["close", "window"], _ulcer_index, "Rolling Ulcer Index from percentage drawdowns."),
]:
    _register(_name, _params, _fn, category="ohlc_volatility", description=_desc)


# ---------------------------------------------------------------------------
# 6. Candle geometry and a compact, deterministic pattern core
# ---------------------------------------------------------------------------


def _candle_parts(open_, high, low, close):
    body = close - open_
    abs_body = body.abs()
    rng = high - low
    upper = high - pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    lower = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns) - low
    return body, abs_body, rng, upper, lower


def _candle_body(open_, close): return close - open_
def _candle_abs_body(open_, close): return (close - open_).abs()
def _candle_range(high, low): return high - low

def _candle_body_ratio(open_, high, low, close):
    _, body, rng, _, _ = _candle_parts(open_, high, low, close)
    return _safe_div(body, rng)


def _candle_upper_shadow(open_, high, close):
    return high - pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)


def _candle_lower_shadow(open_, low, close):
    return pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns) - low


def _candle_upper_shadow_ratio(open_, high, low, close):
    _, _, rng, upper, _ = _candle_parts(open_, high, low, close)
    return _safe_div(upper, rng)


def _candle_lower_shadow_ratio(open_, high, low, close):
    _, _, rng, _, lower = _candle_parts(open_, high, low, close)
    return _safe_div(lower, rng)


def _candle_close_location(high, low, close):
    return _safe_div(close - low, high - low)


def _candle_gap(open_, close):
    return open_ - close.shift(1)


def _candle_gap_pct(open_, close):
    return _safe_div(open_, close.shift(1)) - 1.0


def _candle_direction(open_, close):
    return np.sign(close - open_)


def _candle_range_atr(open_, high, low, close, window):
    tr = _true_range(high, low, close)
    atr = tr.rolling(_w(window), min_periods=_w(window)).mean()
    return _safe_div(high - low, atr)


for _name, _params, _fn, _desc in [
    ("candle_body", ["open", "close"], _candle_body, "Signed candle body: close-open."),
    ("candle_abs_body", ["open", "close"], _candle_abs_body, "Absolute candle body."),
    ("candle_range", ["high", "low"], _candle_range, "High-low candle range."),
    ("candle_body_ratio", ["open", "high", "low", "close"], _candle_body_ratio, "Absolute body divided by full candle range."),
    ("candle_upper_shadow", ["open", "high", "close"], _candle_upper_shadow, "Upper candle shadow length."),
    ("candle_lower_shadow", ["open", "low", "close"], _candle_lower_shadow, "Lower candle shadow length."),
    ("candle_upper_shadow_ratio", ["open", "high", "low", "close"], _candle_upper_shadow_ratio, "Upper shadow divided by full candle range."),
    ("candle_lower_shadow_ratio", ["open", "high", "low", "close"], _candle_lower_shadow_ratio, "Lower shadow divided by full candle range."),
    ("candle_close_location", ["high", "low", "close"], _candle_close_location, "Close location in the candle range, 0=low and 1=high."),
    ("candle_gap", ["open", "close"], _candle_gap, "Open minus previous close."),
    ("candle_gap_pct", ["open", "close"], _candle_gap_pct, "Open gap versus previous close."),
    ("candle_direction", ["open", "close"], _candle_direction, "Candle direction: -1, 0, +1."),
    ("candle_range_atr", ["open", "high", "low", "close", "window"], _candle_range_atr, "Candle range normalized by rolling true range."),
]:
    _register(_name, _params, _fn, category="candle_geometry", description=_desc)


def _cdl_doji(open_, high, low, close):
    _, body, rng, _, _ = _candle_parts(open_, high, low, close)
    return (body <= 0.10 * rng).astype(float)


def _cdl_hammer(open_, high, low, close):
    _, body, rng, upper, lower = _candle_parts(open_, high, low, close)
    flag = (body <= 0.35 * rng) & (lower >= 2.0 * body) & (upper <= 0.35 * np.maximum(body, _EPS))
    return flag.astype(float)


def _cdl_inverted_hammer(open_, high, low, close):
    _, body, rng, upper, lower = _candle_parts(open_, high, low, close)
    flag = (body <= 0.35 * rng) & (upper >= 2.0 * body) & (lower <= 0.35 * np.maximum(body, _EPS))
    return flag.astype(float)


def _cdl_shooting_star(open_, high, low, close):
    return -_cdl_inverted_hammer(open_, high, low, close)


def _cdl_marubozu(open_, high, low, close):
    body, abs_body, rng, upper, lower = _candle_parts(open_, high, low, close)
    flag = (abs_body >= 0.90 * rng) & (upper <= 0.05 * rng) & (lower <= 0.05 * rng)
    return np.sign(body) * flag.astype(float)


def _cdl_spinning_top(open_, high, low, close):
    body, abs_body, rng, upper, lower = _candle_parts(open_, high, low, close)
    flag = (abs_body <= 0.35 * rng) & (upper >= abs_body) & (lower >= abs_body)
    return np.sign(body).replace(0, 1) * flag.astype(float)


def _cdl_engulfing(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    bull = (close > open_) & (prev_close < prev_open) & (open_ <= prev_close) & (close >= prev_open)
    bear = (close < open_) & (prev_close > prev_open) & (open_ >= prev_close) & (close <= prev_open)
    return bull.astype(float) - bear.astype(float)


def _cdl_inside_bar(open_, high, low, close):
    return ((high < high.shift(1)) & (low > low.shift(1))).astype(float)


def _cdl_outside_bar(open_, high, low, close):
    flag = (high > high.shift(1)) & (low < low.shift(1))
    return np.sign(close - open_) * flag.astype(float)


for _name, _fn, _desc in [
    ("cdl_doji", _cdl_doji, "Deterministic doji geometry pattern."),
    ("cdl_hammer", _cdl_hammer, "Raw hammer geometry; trend context is intentionally separate."),
    ("cdl_inverted_hammer", _cdl_inverted_hammer, "Raw inverted-hammer geometry."),
    ("cdl_shooting_star", _cdl_shooting_star, "Raw shooting-star geometry, encoded -1 when present."),
    ("cdl_marubozu", _cdl_marubozu, "Marubozu geometry, signed by candle direction."),
    ("cdl_spinning_top", _cdl_spinning_top, "Spinning-top geometry, signed by candle direction."),
    ("cdl_engulfing", _cdl_engulfing, "Bullish +1 / bearish -1 engulfing body pattern."),
    ("cdl_inside_bar", _cdl_inside_bar, "Inside-bar pattern."),
    ("cdl_outside_bar", _cdl_outside_bar, "Outside-bar pattern signed by candle direction."),
]:
    _register(_name, ["open", "high", "low", "close"], _fn, category="candle_pattern", description=_desc)
