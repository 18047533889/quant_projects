# -*- coding: utf-8 -*-
"""Native Polars backends for OHLC volatility estimators, Ichimoku and ts_*
structure primitives from technical_extensions / indicators_v2.

Every ``_safe_div`` zero-denominator maps to null (pandas ``.replace(0, nan)``);
``clip(lower_bound=0)`` preserves null; rolling var/std use ddof=1 to match the
pandas reference.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den != 0).then(num / den).otherwise(None)


def _max_pair(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(a.is_null() | b.is_null()).then(None).otherwise(pl.max_horizontal(a, b))


def _min_pair(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(a.is_null() | b.is_null()).then(None).otherwise(pl.min_horizontal(a, b))


def _ohlc(open_, high, low, close, column):
    return pl.DataFrame(
        {"open": open_[column], "high": high[column], "low": low[column], "close": close[column]}
    )


# ---------------------------------------------------------------------------
# OHLC volatility estimators (technical_extensions)
# ---------------------------------------------------------------------------


def _parkinson(high, low, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(high, low):
        frame = pl.DataFrame({"high": high[c], "low": low[c]})
        rs = _safe_div(pl.col("high"), pl.col("low")).log().pow(2)
        var = pl.when((4.0 * np.log(2.0) != 0).then((rs.rolling_mean(w, min_samples=w)) / ((4.0 * np.log(2.0)))).otherwise(None)
        values[c] = _one(frame, c, (var.clip(lower_bound=0.0) * 252.0).sqrt())
    return _result(high, values)


def parkinson_vol(high, low, window):
    return _parkinson(high, low, window)


def _garman_klass(open_, high, low, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _ohlc(open_, high, low, close, c)
        hl = _safe_div(pl.col("high"), pl.col("low")).log()
        co = _safe_div(pl.col("close"), pl.col("open")).log()
        daily = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
        values[c] = _one(frame, c, (daily.rolling_mean(w, min_samples=w).clip(lower_bound=0.0) * 252.0).sqrt())
    return _result(close, values)


def garman_klass_vol(open_, high, low, close, window):
    return _garman_klass(open_, high, low, close, window)


def _rogers_satchell_daily():
    ho = _safe_div(pl.col("high"), pl.col("open")).log()
    hc = _safe_div(pl.col("high"), pl.col("close")).log()
    lo = _safe_div(pl.col("low"), pl.col("open")).log()
    lc = _safe_div(pl.col("low"), pl.col("close")).log()
    return ho * hc + lo * lc


def rogers_satchell_vol(open_, high, low, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _ohlc(open_, high, low, close, c)
        daily = _rogers_satchell_daily()
        values[c] = _one(frame, c, (daily.rolling_mean(w, min_samples=w).clip(lower_bound=0.0) * 252.0).sqrt())
    return _result(close, values)


def yang_zhang_vol(open_, high, low, close, window):
    w = _pi(window, "window", 3)
    k = np.where((1.34 + (w + 1.0) / (w - 1.0) != 0, 0.34 / (1.34 + (w + 1.0) / (w - 1.0)), np.nan)
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _ohlc(open_, high, low, close, c)
        overnight = _safe_div(pl.col("open"), pl.col("close").shift(1)).log()
        intraday = _safe_div(pl.col("close"), pl.col("open")).log()
        rs = _rogers_satchell_daily()
        var = (
            overnight.rolling_var(w, min_samples=w)
            + k * intraday.rolling_var(w, min_samples=w)
            + (1.0 - k) * rs.rolling_mean(w, min_samples=w)
        )
        values[c] = _one(frame, c, (var.clip(lower_bound=0.0) * 252.0).sqrt())
    return _result(close, values)


def overnight_volatility(open_, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(open_, close):
        frame = pl.DataFrame({"open": open_[c], "close": close[c]})
        ret = _safe_div(pl.col("open"), pl.col("close").shift(1)).log()
        values[c] = _one(frame, c, ret.rolling_std(w, min_samples=w) * np.sqrt(252.0))
    return _result(close, values)


def intraday_volatility(open_, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(open_, close):
        frame = pl.DataFrame({"open": open_[c], "close": close[c]})
        ret = _safe_div(pl.col("close"), pl.col("open")).log()
        values[c] = _one(frame, c, ret.rolling_std(w, min_samples=w) * np.sqrt(252.0))
    return _result(close, values)


def range_volatility(high, low, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(high, low, close):
        frame = pl.DataFrame({"high": high[c], "low": low[c], "close": close[c]})
        x = _safe_div(pl.col("high") - pl.col("low"), pl.col("close").abs())
        values[c] = _one(frame, c, x.rolling_std(w, min_samples=w) * np.sqrt(252.0))
    return _result(close, values)


def ulcer_index(close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(close):
        rolling_high = pl.col(c).rolling_max(w, min_samples=w)
        drawdown_pct = 100.0 * (_safe_div(pl.col(c), rolling_high) - 1.0)
        values[c] = _one(close, c, (drawdown_pct.pow(2).rolling_mean(w, min_samples=w)).sqrt())
    return _result(close, values)


# ---------------------------------------------------------------------------
# Ichimoku (indicators_v2)
# ---------------------------------------------------------------------------


def _ichimoku_midpoint(high, low, window, name):
    w = _pi(window, name, 2)
    values = {}
    for c in _cols(high, low):
        frame = pl.DataFrame({"high": high[c], "low": low[c]})
        values[c] = _one(frame, c, (pl.col("high").rolling_max(w, min_samples=w) + pl.col("low").rolling_min(w, min_samples=w)) / 2.0)
    return _result(high, values)


def ichimoku_tenkan(high, low, tenkan_window):
    return _ichimoku_midpoint(high, low, tenkan_window, "tenkan_window")


def ichimoku_kijun(high, low, kijun_window):
    return _ichimoku_midpoint(high, low, kijun_window, "kijun_window")


def ichimoku_senkou_a(high, low, tenkan_window, kijun_window):
    tenkan = ichimoku_tenkan(high, low, tenkan_window)
    kijun = ichimoku_kijun(high, low, kijun_window)
    cols = _cols(tenkan, kijun)
    return _result(high, {c: (tenkan[c] + kijun[c]) / 2.0 for c in cols})


def ichimoku_senkou_b(high, low, senkou_b_window):
    return _ichimoku_midpoint(high, low, senkou_b_window, "senkou_b_window")


def ichimoku_cloud_width(high, low, tenkan_window, kijun_window, senkou_b_window):
    a = ichimoku_senkou_a(high, low, tenkan_window, kijun_window)
    b = ichimoku_senkou_b(high, low, senkou_b_window)
    cols = _cols(a, b)
    return _result(high, {c: (a[c] - b[c]).abs() for c in cols})


def ichimoku_cloud_position(high, low, close, tenkan_window, kijun_window, senkou_b_window):
    a = ichimoku_senkou_a(high, low, tenkan_window, kijun_window)
    b = ichimoku_senkou_b(high, low, senkou_b_window)
    cols = _cols(high, low, close, a, b)
    values = {}
    for c in cols:
        frame = pl.DataFrame({"close": close[c], "a": a[c], "b": b[c]})
        lo = _min_pair(pl.col("a"), pl.col("b"))
        hi = _max_pair(pl.col("a"), pl.col("b"))
        values[c] = _one(frame, c, _safe_div(pl.col("close") - lo, hi - lo))
    return _result(close, values)


# ---------------------------------------------------------------------------
# ts_* structure primitives (technical_extensions)
# ---------------------------------------------------------------------------


def ts_prev_high(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        values[c] = _one(x, c, pl.col(c).shift(1).rolling_max(w, min_samples=w))
    return _result(x, values)


def ts_prev_low(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        values[c] = _one(x, c, pl.col(c).shift(1).rolling_min(w, min_samples=w))
    return _result(x, values)


def _dist_to(x, window, high):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        if high:
            base = pl.col(c).shift(1).rolling_max(w, min_samples=w)
        else:
            base = pl.col(c).shift(1).rolling_min(w, min_samples=w)
        values[c] = _one(x, c, _safe_div(pl.col(c), base) - 1.0)
    return _result(x, values)


def ts_distance_to_high(x, window):
    return _dist_to(x, window, True)


def ts_distance_to_low(x, window):
    return _dist_to(x, window, False)


def ts_breakout_high(x, window):
    dist = ts_distance_to_high(x, window)
    cols = _cols(dist)
    values = {}
    for c in cols:
        values[c] = dist[c].clip(lower_bound=0.0)
    return _result(dist, values)


def ts_breakdown_low(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        base = pl.col(c).shift(1).rolling_min(w, min_samples=w)
        values[c] = _one(x, c, (_safe_div(base, pl.col(c)) - 1.0).clip(lower_bound=0.0))
    return _result(x, values)


def ts_new_high(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        base = pl.col(c).shift(1).rolling_max(w, min_samples=w)
        cur = pl.col(c).fill_nan(None)
        # Missing current value OR missing baseline -> null ("cannot judge"),
        # not 0 ("confirmed not a new high") — matches the pandas reference
        # (review P0-07).
        values[c] = _one(
            x, c,
            pl.when(base.is_not_null() & cur.is_not_null())
            .then(pl.when(cur > base).then(1.0).otherwise(0.0))
            .otherwise(None),
        )
    return _result(x, values)


def ts_new_low(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        base = pl.col(c).shift(1).rolling_min(w, min_samples=w)
        cur = pl.col(c).fill_nan(None)
        values[c] = _one(
            x, c,
            pl.when(base.is_not_null() & cur.is_not_null())
            .then(pl.when(cur < base).then(1.0).otherwise(0.0))
            .otherwise(None),
        )
    return _result(x, values)


def ts_channel_position(x, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(x):
        hi = pl.col(c).shift(1).rolling_max(w, min_samples=w)
        lo = pl.col(c).shift(1).rolling_min(w, min_samples=w)
        values[c] = _one(x, c, _safe_div(pl.col(c) - lo, hi - lo))
    return _result(x, values)


def _days_since(x, window, high):
    w = _pi(window, "window")
    # Ties must reference the LATEST occurrence (audit 12).  ``np.nanargmax``
    # picks the FIRST of a tie; the first max/min in the reversed window is the
    # last occurrence in chronological order, and equals the "bars since" count.
    if high:
        def fn(window_arr) -> float:
            a = np.asarray(window_arr, dtype=float)
            if not np.isfinite(a).any():
                return np.nan
            return float(int(np.nanargmax(a[::-1])))
    else:
        def fn(window_arr) -> float:
            a = np.asarray(window_arr, dtype=float)
            if not np.isfinite(a).any():
                return np.nan
            return float(int(np.nanargmin(a[::-1])))
    values = {}
    for c in _cols(x):
        # normalize NaN -> null so rolling_map min_samples counts match pandas
        # rolling min_periods (non-NaN observations)
        shifted = pl.col(c).shift(1).fill_nan(None)
        values[c] = _one(x, c, shifted.rolling_map(fn, window_size=w, min_samples=w))
    return _result(x, values)


def ts_days_since_high(x, window):
    return _days_since(x, window, True)


def ts_days_since_low(x, window):
    return _days_since(x, window, False)


def ts_range_expansion(high, low, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(high, low):
        frame = pl.DataFrame({"high": high[c], "low": low[c]})
        current = pl.col("high") - pl.col("low")
        baseline = current.shift(1).rolling_mean(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(current, baseline) - 1.0)
    return _result(high, values)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("parkinson_vol", ("high", "low", "window"), parkinson_vol, "Annualized Parkinson high-low volatility estimator."),
    ("garman_klass_vol", ("open", "high", "low", "close", "window"), garman_klass_vol, "Annualized Garman-Klass OHLC volatility estimator."),
    ("rogers_satchell_vol", ("open", "high", "low", "close", "window"), rogers_satchell_vol, "Annualized Rogers-Satchell OHLC volatility estimator."),
    ("yang_zhang_vol", ("open", "high", "low", "close", "window"), yang_zhang_vol, "Annualized Yang-Zhang OHLC volatility estimator."),
    ("overnight_volatility", ("open", "close", "window"), overnight_volatility, "Annualized volatility of open versus previous close."),
    ("intraday_volatility", ("open", "close", "window"), intraday_volatility, "Annualized close-to-open intraday volatility."),
    ("range_volatility", ("high", "low", "close", "window"), range_volatility, "Annualized volatility of normalized high-low range."),
    ("ulcer_index", ("close", "window"), ulcer_index, "Rolling Ulcer Index from percentage drawdowns."),
    ("ichimoku_tenkan", ("high", "low", "tenkan_window"), ichimoku_tenkan, "Raw Tenkan value."),
    ("ichimoku_kijun", ("high", "low", "kijun_window"), ichimoku_kijun, "Raw Kijun value."),
    ("ichimoku_senkou_a", ("high", "low", "tenkan_window", "kijun_window"), ichimoku_senkou_a, "Raw Senkou A without chart-forward shift."),
    ("ichimoku_senkou_b", ("high", "low", "senkou_b_window"), ichimoku_senkou_b, "Raw Senkou B without chart-forward shift."),
    ("ichimoku_cloud_width", ("high", "low", "tenkan_window", "kijun_window", "senkou_b_window"), ichimoku_cloud_width, "Raw Ichimoku cloud width."),
    ("ichimoku_cloud_position", ("high", "low", "close", "tenkan_window", "kijun_window", "senkou_b_window"), ichimoku_cloud_position, "Close position inside raw Ichimoku cloud."),
    ("ts_prev_high", ("x", "window"), ts_prev_high, "Prior-window high."),
    ("ts_prev_low", ("x", "window"), ts_prev_low, "Prior-window low."),
    ("ts_distance_to_high", ("x", "window"), ts_distance_to_high, "Signed distance to prior-window high."),
    ("ts_distance_to_low", ("x", "window"), ts_distance_to_low, "Signed distance to prior-window low."),
    ("ts_breakout_high", ("x", "window"), ts_breakout_high, "Positive breakout above prior-window high."),
    ("ts_breakdown_low", ("x", "window"), ts_breakdown_low, "Positive breakdown below prior-window low."),
    ("ts_new_high", ("x", "window"), ts_new_high, "1 when above prior-window high."),
    ("ts_new_low", ("x", "window"), ts_new_low, "1 when below prior-window low."),
    ("ts_channel_position", ("x", "window"), ts_channel_position, "Position inside prior-window channel."),
    ("ts_days_since_high", ("x", "window"), ts_days_since_high, "Bars since the prior-window high."),
    ("ts_days_since_low", ("x", "window"), ts_days_since_low, "Bars since the prior-window low."),
    ("ts_range_expansion", ("high", "low", "window"), ts_range_expansion, "Range expansion versus prior baseline."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsMiscV2_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="polars_misc_v2",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
