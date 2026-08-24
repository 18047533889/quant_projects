# -*- coding: utf-8 -*-
"""Native Polars implementations for candle geometry and candlestick patterns.

Mirrors the pandas references in ``technical_extensions`` (candle_*/cdl_*),
``candle_geometry_v2`` and ``candle_patterns_extended``.

NaN/null discipline: pandas boolean comparisons treat NaN as False, while
polars treats float NaN as +inf (``NaN > x`` -> True).  Inputs are therefore
normalized with ``fill_nan(None)`` so comparisons become null (-> propagated),
and every boolean-flagged output goes through ``_flag`` which maps null to 0.0,
matching pandas ``.astype(float)`` on a NaN comparison.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})
_EPS = 1e-12


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
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


def _flag(cond: pl.Expr) -> pl.Expr:
    """pandas bool cond with NaN -> False -> 0.0; null -> 0.0."""
    return pl.when(cond.is_not_null() & cond).then(1.0).otherwise(0.0)


def _signed_flag(sign: pl.Expr, cond: pl.Expr) -> pl.Expr:
    # pandas: sign * flag.astype(float)；sign 为 NaN(输入缺失) 时 NaN*0 = NaN 保留。
    return sign * _flag(cond)


def _max_pair(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    # pandas np.maximum 传播 NaN；polars max_horizontal 可能忽略 NaN 或按值比较。
    # 显式把 NaN 一并传播为 null，保证 ``upper <= 0.35*max(body, EPS)`` 的 NaN
    # 语义与 pandas 一致（NaN 比较为 False → flag 0）。
    return pl.when(a.is_null() | b.is_null() | a.is_nan() | b.is_nan()).then(None).otherwise(pl.max_horizontal(a, b))


def _min_pair(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(a.is_null() | b.is_null() | a.is_nan() | b.is_nan()).then(None).otherwise(pl.min_horizontal(a, b))


def _frame(
    open_: pl.DataFrame | None,
    high: pl.DataFrame | None,
    low: pl.DataFrame | None,
    close: pl.DataFrame | None,
    column: str,
) -> pl.DataFrame:
    data: dict[str, pl.Series] = {}
    if open_ is not None:
        data["open"] = open_[column].fill_nan(None)
    if high is not None:
        data["high"] = high[column].fill_nan(None)
    if low is not None:
        data["low"] = low[column].fill_nan(None)
    if close is not None:
        data["close"] = close[column].fill_nan(None)
    return pl.DataFrame(data)


def _parts(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr):
    body = c - o
    abs_body = (c - o).abs()
    rng = h - l
    max_oc = _max_pair(o, c)
    min_oc = _min_pair(o, c)
    upper = h - max_oc
    lower = min_oc - l
    return body, abs_body, rng, upper, lower


def _z_expr(expr: pl.Expr, frame: pl.DataFrame, column: str, window: int, valid: pl.Expr | None = None) -> pl.Series:
    """(x - shift(1).rolling_mean(w)) / shift(1).rolling_std(w), 0-std -> null.

    ``valid`` (optional): a boolean OHLC-validity expr; rows where it is False
    emit null (round-3 audit item 15 — invalid bars never produce a shadow
    z-score built from a negative shadow length).
    """
    w = _pi(window, "window", 2)
    shifted = expr.shift(1)
    mean = shifted.rolling_mean(w, min_samples=w)
    std = shifted.rolling_std(w, min_samples=w)
    z = _safe_div(expr - mean, std)
    if valid is not None:
        z = pl.when(valid).then(z).otherwise(None)
    return _one(frame, column, z)


def _pct_rank_expr(expr: pl.Expr, frame: pl.DataFrame, column: str, window: int, valid: pl.Expr | None = None) -> pl.Series:
    w = _pi(window, "window", 2)

    def _rank_fn(window_arr) -> float:
        arr = np.asarray(window_arr, dtype=float)
        n = len(arr)
        if n < 2:
            return np.nan
        cur = arr[-1]
        prev = arr[:-1]
        # numpy 中 NaN<x 与 NaN==x 均 False，与 pandas 布尔语义一致
        return float((np.sum(prev < cur) + 0.5 * np.sum(prev == cur)) / max(1, n - 1))

    rank = expr.rolling_map(_rank_fn, window_size=w, min_samples=w)
    if valid is not None:
        rank = pl.when(valid).then(rank).otherwise(None)
    return _one(frame, column, rank)


def _true_range(h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    # pandas _true_range 用 concat(...).groupby(level=1).max()：逐行 max 忽略 NaN，
    # 首行 prev=NaN 时取 high-low。polars max_horizontal 忽略 null，故先归一 NaN。
    previous = c.shift(1)
    raw = pl.max_horizontal(h - l, (h - previous).abs(), (l - previous).abs()).fill_nan(None)
    return raw


def _tr_propagate(h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    # candle_geometry 的 _tr 用 np.maximum.reduce：任一 NaN 传播为 NaN。
    # 算术 max_pair 传播 null，等价 pandas 的 NaN 传播语义。
    previous = c.shift(1)
    hl = h - l
    hc = (h - previous).abs()
    lc = (l - previous).abs()
    return _max_pair(_max_pair(hl, hc), lc)


def _validate_ohlc(o=None, h=None, l=None, c=None) -> pl.Expr:
    """Unified OHLC structural-validity mask (round-3 audit item 15).

    Mirrors the pandas ``_validate_ohlc`` in ``candle_geometry_v2``: returns a
    non-null boolean expr, True where every provided OHLC field is strictly
    positive and the bar satisfies
    ``High >= max(Open, Close) >= min(Open, Close) >= Low``.  Omitted fields
    are not checked (subset operators keep their own axis contract); any
    null/NaN field fails closed to False.  Callers wrap the output with
    ``pl.when(valid).then(...).otherwise(None)`` so an invalid bar emits null
    instead of a negative shadow / out-of-range strength / fabricated pattern.
    """
    conds: list[pl.Expr] = []
    if o is not None:
        conds.append((o > 0).fill_null(False))
    if h is not None:
        conds.append((h > 0).fill_null(False))
    if l is not None:
        conds.append((l > 0).fill_null(False))
    if c is not None:
        conds.append((c > 0).fill_null(False))
    if h is not None and l is not None:
        conds.append((h >= l).fill_null(False))
    if o is not None and c is not None and h is not None:
        conds.append((h >= _max_pair(o, c)).fill_null(False))
    if o is not None and c is not None and l is not None:
        conds.append((l <= _min_pair(o, c)).fill_null(False))
    # subset operators without ``open`` (e.g. candle_close_strength) must not
    # emit close-strength outside [-1,1]: close must sit inside the range.
    if o is None and h is not None and l is not None and c is not None:
        conds.append(((c >= l) & (c <= h)).fill_null(False))
    valid = conds[0]
    for cond in conds[1:]:
        valid = valid & cond
    return valid


# ---------------------------------------------------------------------------
# candle geometry (technical_extensions)
# ---------------------------------------------------------------------------


def candle_body(open_, close):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        values[c] = _one(frame, c, pl.col("close") - pl.col("open"))
    return _result(close, values)


def candle_abs_body(open_, close):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        values[c] = _one(frame, c, (pl.col("close") - pl.col("open")).abs())
    return _result(close, values)


def candle_range(high, low):
    values = {}
    for c in _cols(high, low):
        frame = _frame(None, high, low, None, c)
        values[c] = _one(frame, c, pl.col("high") - pl.col("low"))
    return _result(high, values)


def candle_body_ratio(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, body, rng, _, _ = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, _safe_div(body, rng))
    return _result(close, values)


def candle_upper_shadow(open_, high, close):
    values = {}
    for c in _cols(open_, high, close):
        frame = _frame(open_, high, None, close, c)
        _, _, _, upper, _ = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, upper)
    return _result(close, values)


def candle_lower_shadow(open_, low, close):
    values = {}
    for c in _cols(open_, low, close):
        frame = _frame(open_, None, low, close, c)
        _, _, _, _, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, lower)
    return _result(close, values)


def candle_upper_shadow_ratio(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, _, rng, upper, _ = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, _safe_div(upper, rng))
    return _result(close, values)


def candle_lower_shadow_ratio(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, _, rng, _, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, _safe_div(lower, rng))
    return _result(close, values)


def candle_close_location(high, low, close):
    values = {}
    for c in _cols(high, low, close):
        frame = _frame(None, high, low, close, c)
        values[c] = _one(frame, c, _safe_div(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")))
    return _result(close, values)


def candle_gap(open_, close):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        values[c] = _one(frame, c, pl.col("open") - pl.col("close").shift(1))
    return _result(close, values)


def candle_gap_pct(open_, close):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        values[c] = _one(frame, c, _safe_div(pl.col("open"), pl.col("close").shift(1)) - 1.0)
    return _result(close, values)


def candle_direction(open_, close):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        values[c] = _one(frame, c, (pl.col("close") - pl.col("open")).sign())
    return _result(close, values)


def candle_range_atr(open_, high, low, close, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        tr = _true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        atr = tr.rolling_mean(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(pl.col("high") - pl.col("low"), atr))
    return _result(close, values)


# ---------------------------------------------------------------------------
# classic cdl patterns (technical_extensions)
# ---------------------------------------------------------------------------


def cdl_doji(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, body, rng, _, _ = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, _flag(body <= 0.10 * rng))
    return _result(close, values)


def _cdl_hammer_like(open_, high, low, close, *, inverted: bool) -> pl.DataFrame:
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, body, rng, upper, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        if inverted:
            flag = (body <= 0.35 * rng) & (upper >= 2.0 * body) & (lower <= 0.35 * _max_pair(body, pl.lit(_EPS)))
        else:
            flag = (body <= 0.35 * rng) & (lower >= 2.0 * body) & (upper <= 0.35 * _max_pair(body, pl.lit(_EPS)))
        values[c] = _one(frame, c, _flag(flag))
    return _result(close, values)


def _prior_trend_expr(close: pl.Expr, *, up: bool, window: int = 5) -> pl.Expr:
    """因果 prior-trend 上下文（audit item 5）：当前 bar 之前的 close 是否高于/
    低于前 window 根 close 的均值；基线不可得时为 null（"cannot judge"）。
    与 pandas ``_prior_trend`` 一致。"""
    prev = close.shift(1)
    base = close.shift(1).rolling_mean(window_size=window, min_samples=window)
    cmp = (prev > base) if up else (prev < base)
    return pl.when(base.is_not_null() & base.is_not_nan()).then(cmp).otherwise(None)


def _hammer_flag(body: pl.Expr, rng: pl.Expr, upper: pl.Expr, lower: pl.Expr, close: pl.Expr) -> pl.Expr:
    """hammer 几何 + min_tick 门（rng > 1e-6*|close|，audit item 4/5）。"""
    return (
        (body <= 0.35 * rng)
        & (lower >= 2.0 * body)
        & (upper <= 0.35 * body.clip(lower_bound=_EPS))
        & (rng > 1e-6 * close.abs())
    )


def cdl_hammer(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        _, body, rng, upper, lower = _parts(o, h, l, cl)
        flag = _hammer_flag(body, rng, upper, lower, cl)
        downtrend = _prior_trend_expr(cl, up=False)
        pattern = flag & downtrend.fill_null(False)
        out = _flag(pattern)
        # 与 pandas ``out.where(downtrend.notna())``：基线不可得 → NaN。
        # round-3 audit item 15: 结构非法 / 非正价格 bar 也不判定为 pattern。
        valid = _validate_ohlc(o, h, l, cl)
        values[c] = _one(frame, c, pl.when(downtrend.is_not_null() & valid).then(out).otherwise(None))
    return _result(close, values)


def cdl_inverted_hammer(open_, high, low, close):
    return _cdl_hammer_like(open_, high, low, close, inverted=True)


def cdl_shooting_star(open_, high, low, close):
    star = _cdl_hammer_like(open_, high, low, close, inverted=True)
    return _result(star, {c: -star[c] for c in _cols(star)})


def cdl_marubozu(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, abs_body, rng, upper, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        sign = (pl.col("close") - pl.col("open")).sign()
        flag = (abs_body >= 0.90 * rng) & (upper <= 0.05 * rng) & (lower <= 0.05 * rng)
        values[c] = _one(frame, c, _signed_flag(sign, flag))
    return _result(close, values)


def cdl_spinning_top(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, abs_body, rng, upper, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        sign = (pl.col("close") - pl.col("open")).sign()
        # pandas: np.sign(body).replace(0, 1) —— 0 变 1，NaN 保留
        sign0 = pl.when(sign == 0).then(1.0).otherwise(sign)
        flag = (abs_body <= 0.35 * rng) & (upper >= abs_body) & (lower >= abs_body)
        values[c] = _one(frame, c, _signed_flag(sign0, flag))
    return _result(close, values)


def cdl_engulfing(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        prev_o, prev_c = o.shift(1), cl.shift(1)
        bull = (cl > o) & (prev_c < prev_o) & (o <= prev_c) & (cl >= prev_o)
        bear = (cl < o) & (prev_c > prev_o) & (o >= prev_c) & (cl <= prev_o)
        values[c] = _one(frame, c, _flag(bull) - _flag(bear))
    return _result(close, values)


def cdl_inside_bar(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        h, l = pl.col("high"), pl.col("low")
        flag = (h < h.shift(1)) & (l > l.shift(1))
        values[c] = _one(frame, c, _flag(flag))
    return _result(close, values)


def cdl_outside_bar(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        h, l, o, cl = pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close")
        flag = (h > h.shift(1)) & (l < l.shift(1))
        values[c] = _one(frame, c, _signed_flag((cl - o).sign(), flag))
    return _result(close, values)


# ---------------------------------------------------------------------------
# candle geometry v2
# ---------------------------------------------------------------------------


def candle_body_zscore(open_, close, window):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        valid = _validate_ohlc(pl.col("open"), None, None, pl.col("close"))
        values[c] = _z_expr((pl.col("close") - pl.col("open")).abs(), frame, c, window, valid)
    return _result(close, values)


def candle_range_zscore(high, low, window):
    values = {}
    for c in _cols(high, low):
        frame = _frame(None, high, low, None, c)
        valid = _validate_ohlc(None, pl.col("high"), pl.col("low"), None)
        values[c] = _z_expr(pl.col("high") - pl.col("low"), frame, c, window, valid)
    return _result(high, values)


def candle_upper_shadow_zscore(open_, high, close, window):
    values = {}
    for c in _cols(open_, high, close):
        frame = _frame(open_, high, None, close, c)
        upper = pl.col("high") - _max_pair(pl.col("open"), pl.col("close"))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), None, pl.col("close"))
        values[c] = _z_expr(upper, frame, c, window, valid)
    return _result(close, values)


def candle_lower_shadow_zscore(open_, low, close, window):
    values = {}
    for c in _cols(open_, low, close):
        frame = _frame(open_, None, low, close, c)
        lower = _min_pair(pl.col("open"), pl.col("close")) - pl.col("low")
        valid = _validate_ohlc(pl.col("open"), None, pl.col("low"), pl.col("close"))
        values[c] = _z_expr(lower, frame, c, window, valid)
    return _result(close, values)


def candle_body_percentile(open_, close, window):
    values = {}
    for c in _cols(open_, close):
        frame = _frame(open_, None, None, close, c)
        valid = _validate_ohlc(pl.col("open"), None, None, pl.col("close"))
        values[c] = _pct_rank_expr((pl.col("close") - pl.col("open")).abs(), frame, c, window, valid)
    return _result(close, values)


def candle_range_percentile(high, low, window):
    values = {}
    for c in _cols(high, low):
        frame = _frame(None, high, low, None, c)
        valid = _validate_ohlc(None, pl.col("high"), pl.col("low"), None)
        values[c] = _pct_rank_expr(pl.col("high") - pl.col("low"), frame, c, window, valid)
    return _result(high, values)


def candle_gap_atr(open_, high, low, close, atr_window):
    w = _pi(atr_window, "atr_window", 2)
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        tr = _tr_propagate(pl.col("high"), pl.col("low"), pl.col("close"))
        # R4: unify ATR with the engine's Wilder definition (pandas
        # ``ewm(alpha=1/w, adjust=False, min_periods=w).mean()``), NOT a plain
        # rolling mean — one ATR object across the engine.
        atr = tr.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w)
        # round-3 audit item 16: the denominator must be ATR AS OF t-1 (excludes
        # today's High/Low/Close), so today's late-session range does not
        # retroactively weaken the morning gap.  pandas ``ewm`` carries the
        # recursive state forward over NaN TR bars (the ATR value stays put);
        # polars ``ewm_mean`` (ignore_nulls=False) emits null at those bars, so
        # forward-fill the shifted ATR to reproduce the pandas reference.
        atr_prev = atr.shift(1).fill_null(strategy="forward")
        gap = pl.col("open") - pl.col("close").shift(1)
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        valid = valid & valid.shift(1)
        values[c] = _one(frame, c, pl.when(valid).then(_safe_div(gap, atr_prev)).otherwise(None))
    return _result(close, values)


def candle_body_position(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        midpoint = (pl.col("open") + pl.col("close")) / 2.0
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        out = _safe_div(midpoint - pl.col("low"), pl.col("high") - pl.col("low"))
        values[c] = _one(frame, c, pl.when(valid).then(out).otherwise(None))
    return _result(close, values)


def candle_overlap_ratio(high, low):
    values = {}
    for c in _cols(high, low):
        frame = _frame(None, high, low, None, c)
        h, l = pl.col("high"), pl.col("low")
        overlap = (_min_pair(h, h.shift(1)) - _max_pair(l, l.shift(1))).clip(lower_bound=0.0)
        union = _max_pair(h, h.shift(1)) - _min_pair(l, l.shift(1))
        valid = _validate_ohlc(None, h, l, None) & _validate_ohlc(None, h, l, None).shift(1)
        values[c] = _one(frame, c, pl.when(valid).then(_safe_div(overlap, union)).otherwise(None))
    return _result(high, values)


def candle_inside_ratio(high, low):
    values = {}
    for c in _cols(high, low):
        frame = _frame(None, high, low, None, c)
        h, l = pl.col("high"), pl.col("low")
        prev = pl.when((h.shift(1) - l.shift(1)) != 0).then(h.shift(1) - l.shift(1)).otherwise(None)
        current = h - l
        ratio = current / prev
        inside = (h <= h.shift(1)) & (l >= l.shift(1))
        outside = (h >= h.shift(1)) & (l <= l.shift(1))
        # R4/P1-12: a shifted-up / shifted-down / gap bar is NEITHER inside NOR
        # outside containment — only true containment states get a value, and
        # ambiguous bars are NaN.  Mirrors the pandas reference exactly.
        in_v = pl.when(inside).then(ratio).otherwise(None)
        out_v = pl.when(outside & ~inside).then(1.0 + prev / current).otherwise(None)
        result = in_v.fill_null(out_v)
        keep = prev.is_not_null() & current.is_not_null() & (inside | outside)
        valid = _validate_ohlc(None, h, l, None) & _validate_ohlc(None, h, l, None).shift(1)
        values[c] = _one(frame, c, pl.when(keep & valid).then(result).otherwise(None))
    return _result(high, values)


def candle_close_strength(high, low, close):
    values = {}
    for c in _cols(high, low, close):
        frame = _frame(None, high, low, close, c)
        valid = _validate_ohlc(None, pl.col("high"), pl.col("low"), pl.col("close"))
        out = 2.0 * _safe_div(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")) - 1.0
        values[c] = _one(frame, c, pl.when(valid).then(out).otherwise(None))
    return _result(close, values)


def candle_rejection_upper(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, _, rng, upper, _ = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, pl.when(valid).then(_safe_div(upper, rng)).otherwise(None))
    return _result(close, values)


def candle_rejection_lower(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, _, rng, _, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, pl.when(valid).then(_safe_div(lower, rng)).otherwise(None))
    return _result(close, values)


# ---------------------------------------------------------------------------
# extended cdl patterns (candle_patterns_extended)
# ---------------------------------------------------------------------------


def cdl_dragonfly_doji(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, body, rng, upper, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        flag = _flag((body <= 0.10 * rng) & (lower >= 0.60 * rng) & (upper <= 0.10 * rng))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def cdl_gravestone_doji(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        _, body, rng, upper, lower = _parts(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        flag = _flag((body <= 0.10 * rng) & (upper >= 0.60 * rng) & (lower <= 0.10 * rng))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def cdl_hanging_man(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        _, body, rng, upper, lower = _parts(o, h, l, cl)
        flag = _hammer_flag(body, rng, upper, lower, cl)
        uptrend = _prior_trend_expr(cl, up=True)
        pattern = flag & uptrend.fill_null(False)
        out = -_flag(pattern)
        valid = _validate_ohlc(o, h, l, cl)
        values[c] = _one(frame, c, pl.when(uptrend.is_not_null() & valid).then(out).otherwise(None))
    return _result(close, values)


def _cdl_harami(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        prev_o, prev_c = o.shift(1), cl.shift(1)
        prev_hi = _max_pair(prev_o, prev_c)
        prev_lo = _min_pair(prev_o, prev_c)
        cur_hi = _max_pair(o, cl)
        cur_lo = _min_pair(o, cl)
        inside = (cur_hi < prev_hi) & (cur_lo > prev_lo)
        bullish = inside & (prev_c < prev_o) & (cl > o)
        bearish = inside & (prev_c > prev_o) & (cl < o)
        valid = _validate_ohlc(o, h, l, cl) & _validate_ohlc(o, h, l, cl).shift(1)
        out = _flag(bullish) - _flag(bearish)
        values[c] = _one(frame, c, pl.when(valid).then(out).otherwise(None))
    return _result(close, values)


def cdl_harami(open_, high, low, close):
    return _cdl_harami(open_, high, low, close)


def cdl_harami_cross(open_, high, low, close):
    base = _cdl_harami(open_, high, low, close)
    doji = cdl_doji(open_, high, low, close)
    return _result(base, {c: base[c] * doji[c] for c in _cols(base)})


def cdl_piercing(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, cl = pl.col("open"), pl.col("close")
        prev_o, prev_c = o.shift(1), cl.shift(1)
        midpoint = (prev_o + prev_c) / 2.0
        flag = _flag((prev_c < prev_o) & (cl > o) & (o <= prev_c) & (cl > midpoint) & (cl < prev_o))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        valid = valid & valid.shift(1)
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def cdl_dark_cloud_cover(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, cl = pl.col("open"), pl.col("close")
        prev_o, prev_c = o.shift(1), cl.shift(1)
        midpoint = (prev_o + prev_c) / 2.0
        flag = -_flag((prev_c > prev_o) & (cl < o) & (o >= prev_c) & (cl < midpoint) & (cl > prev_o))
        valid = _validate_ohlc(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        valid = valid & valid.shift(1)
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def cdl_morning_star(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        o2, c2 = o.shift(2), cl.shift(2)
        o1, c1 = o.shift(1), cl.shift(1)
        body2 = (c2 - o2).abs()
        body1 = (c1 - o1).abs()
        range2 = pl.when((h.shift(2) - l.shift(2)) != 0).then(h.shift(2) - l.shift(2)).otherwise(None)
        range1 = pl.when((h.shift(1) - l.shift(1)) != 0).then(h.shift(1) - l.shift(1)).otherwise(None)
        midpoint2 = (o2 + c2) / 2.0
        flag = _flag((c2 < o2) & (body2 >= 0.50 * range2) & (body1 <= 0.35 * range1) & (cl > o) & (cl > midpoint2))
        valid = _validate_ohlc(o, h, l, cl)
        valid = valid & valid.shift(1) & valid.shift(2)
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def cdl_evening_star(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        o2, c2 = o.shift(2), cl.shift(2)
        o1, c1 = o.shift(1), cl.shift(1)
        body2 = (c2 - o2).abs()
        body1 = (c1 - o1).abs()
        range2 = pl.when((h.shift(2) - l.shift(2)) != 0).then(h.shift(2) - l.shift(2)).otherwise(None)
        range1 = pl.when((h.shift(1) - l.shift(1)) != 0).then(h.shift(1) - l.shift(1)).otherwise(None)
        midpoint2 = (o2 + c2) / 2.0
        flag = -_flag((c2 > o2) & (body2 >= 0.50 * range2) & (body1 <= 0.35 * range1) & (cl < o) & (cl < midpoint2))
        valid = _validate_ohlc(o, h, l, cl)
        valid = valid & valid.shift(1) & valid.shift(2)
        values[c] = _one(frame, c, pl.when(valid).then(flag).otherwise(None))
    return _result(close, values)


def _long_bodies_expr(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr, min_body_frac: float = 0.5) -> pl.Expr:
    """pandas ``_long_bodies``：abs_body>=frac*rng 且 rng>1e-6*|close|。"""
    _, abs_body, rng, _, _ = _parts(o, h, l, c)
    return (abs_body >= min_body_frac * rng) & (rng > 1e-6 * c.abs())


def _short_shadows_expr(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr, max_body_frac: float = 0.4) -> pl.Expr:
    """pandas ``_short_shadows``：upper/lower <= max_body_frac*abs_body。"""
    _, abs_body, _, upper, lower = _parts(o, h, l, c)
    return (upper <= max_body_frac * abs_body) & (lower <= max_body_frac * abs_body)


def cdl_three_white_soldiers(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, cl = pl.col("open"), pl.col("close")
        bull0, bull1, bull2 = cl > o, cl.shift(1) > o.shift(1), cl.shift(2) > o.shift(2)
        rising = (cl > cl.shift(1)) & (cl.shift(1) > cl.shift(2))
        opens_inside = (
            (o >= o.shift(1)) & (o <= cl.shift(1))
            & (o.shift(1) >= o.shift(2)) & (o.shift(1) <= cl.shift(2))
        )
        # audit item 7：长实体 + 短影线门（与 pandas 一致）。
        long_b = (
            _long_bodies_expr(o, pl.col("high"), pl.col("low"), cl)
            & _long_bodies_expr(o.shift(1), pl.col("high").shift(1), pl.col("low").shift(1), cl.shift(1))
            & _long_bodies_expr(o.shift(2), pl.col("high").shift(2), pl.col("low").shift(2), cl.shift(2))
        )
        short_sh = (
            _short_shadows_expr(o, pl.col("high"), pl.col("low"), cl)
            & _short_shadows_expr(o.shift(1), pl.col("high").shift(1), pl.col("low").shift(1), cl.shift(1))
            & _short_shadows_expr(o.shift(2), pl.col("high").shift(2), pl.col("low").shift(2), cl.shift(2))
        )
        valid = _validate_ohlc(o, pl.col("high"), pl.col("low"), cl)
        valid = valid & valid.shift(1) & valid.shift(2)
        out = _flag(bull0 & bull1 & bull2 & rising & opens_inside & long_b & short_sh)
        values[c] = _one(frame, c, pl.when(valid).then(out).otherwise(None))
    return _result(close, values)


def cdl_three_black_crows(open_, high, low, close):
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, cl = pl.col("open"), pl.col("close")
        bear0, bear1, bear2 = cl < o, cl.shift(1) < o.shift(1), cl.shift(2) < o.shift(2)
        falling = (cl < cl.shift(1)) & (cl.shift(1) < cl.shift(2))
        opens_inside = (
            (o <= o.shift(1)) & (o >= cl.shift(1))
            & (o.shift(1) <= o.shift(2)) & (o.shift(1) >= cl.shift(2))
        )
        # audit item 7：长实体 + 短影线门（与 pandas 一致）。
        long_b = (
            _long_bodies_expr(o, pl.col("high"), pl.col("low"), cl)
            & _long_bodies_expr(o.shift(1), pl.col("high").shift(1), pl.col("low").shift(1), cl.shift(1))
            & _long_bodies_expr(o.shift(2), pl.col("high").shift(2), pl.col("low").shift(2), cl.shift(2))
        )
        short_sh = (
            _short_shadows_expr(o, pl.col("high"), pl.col("low"), cl)
            & _short_shadows_expr(o.shift(1), pl.col("high").shift(1), pl.col("low").shift(1), cl.shift(1))
            & _short_shadows_expr(o.shift(2), pl.col("high").shift(2), pl.col("low").shift(2), cl.shift(2))
        )
        valid = _validate_ohlc(o, pl.col("high"), pl.col("low"), cl)
        valid = valid & valid.shift(1) & valid.shift(2)
        out = -_flag(bear0 & bear1 & bear2 & falling & opens_inside & long_b & short_sh)
        values[c] = _one(frame, c, pl.when(valid).then(out).otherwise(None))
    return _result(close, values)


def _tick_size_expr(price: pl.Expr) -> pl.Expr:
    """A-share tick：|p|<10 → 0.01，<100 → 0.05，否则 0.1（与 pandas 一致）。"""
    return (
        pl.when(price.abs() < 10.0)
        .then(0.01)
        .when(price.abs() < 100.0)
        .then(0.05)
        .otherwise(0.1)
    )


def _cdl_tweezer(open_, high, low, close, *, top: bool) -> pl.DataFrame:
    values = {}
    for c in _cols(open_, high, low, close):
        frame = _frame(open_, high, low, close, c)
        o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        valid = _validate_ohlc(o, h, l, cl) & _validate_ohlc(o, h, l, cl).shift(1)
        # audit item 6: tick-aware 容差（_TICK_K=2），非旧的 1e-4*price 相对带。
        if top:
            same = (h - h.shift(1)).abs() <= _tick_size_expr(h) * 2.0
            reversal = (cl.shift(1) > o.shift(1)) & (cl < o)
            values[c] = _one(frame, c, pl.when(valid).then(-_flag(same & reversal)).otherwise(None))
        else:
            same = (l - l.shift(1)).abs() <= _tick_size_expr(l) * 2.0
            reversal = (cl.shift(1) < o.shift(1)) & (cl > o)
            values[c] = _one(frame, c, pl.when(valid).then(_flag(same & reversal)).otherwise(None))
    return _result(close, values)


def cdl_tweezer_top(open_, high, low, close):
    return _cdl_tweezer(open_, high, low, close, top=True)


def cdl_tweezer_bottom(open_, high, low, close):
    return _cdl_tweezer(open_, high, low, close, top=False)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("candle_body", ("open", "close"), candle_body, "Signed candle body: close-open."),
    ("candle_abs_body", ("open", "close"), candle_abs_body, "Absolute candle body."),
    ("candle_range", ("high", "low"), candle_range, "High-low candle range."),
    ("candle_body_ratio", ("open", "high", "low", "close"), candle_body_ratio, "Absolute body divided by full candle range."),
    ("candle_upper_shadow", ("open", "high", "close"), candle_upper_shadow, "Upper candle shadow length."),
    ("candle_lower_shadow", ("open", "low", "close"), candle_lower_shadow, "Lower candle shadow length."),
    ("candle_upper_shadow_ratio", ("open", "high", "low", "close"), candle_upper_shadow_ratio, "Upper shadow divided by full candle range."),
    ("candle_lower_shadow_ratio", ("open", "high", "low", "close"), candle_lower_shadow_ratio, "Lower shadow divided by full candle range."),
    ("candle_close_location", ("high", "low", "close"), candle_close_location, "Close location in the candle range."),
    ("candle_gap", ("open", "close"), candle_gap, "Open minus previous close."),
    ("candle_gap_pct", ("open", "close"), candle_gap_pct, "Open gap versus previous close."),
    ("candle_direction", ("open", "close"), candle_direction, "Candle direction: -1, 0, +1."),
    ("candle_range_atr", ("open", "high", "low", "close", "window"), candle_range_atr, "Candle range normalized by rolling true range."),
    ("cdl_doji", ("open", "high", "low", "close"), cdl_doji, "Deterministic doji geometry pattern."),
    ("cdl_hammer", ("open", "high", "low", "close"), cdl_hammer, "Raw hammer geometry."),
    ("cdl_inverted_hammer", ("open", "high", "low", "close"), cdl_inverted_hammer, "Raw inverted-hammer geometry."),
    ("cdl_shooting_star", ("open", "high", "low", "close"), cdl_shooting_star, "Raw shooting-star geometry, -1 when present."),
    ("cdl_marubozu", ("open", "high", "low", "close"), cdl_marubozu, "Marubozu geometry, signed by candle direction."),
    ("cdl_spinning_top", ("open", "high", "low", "close"), cdl_spinning_top, "Spinning-top geometry, signed by direction."),
    ("cdl_engulfing", ("open", "high", "low", "close"), cdl_engulfing, "Bullish +1 / bearish -1 engulfing body pattern."),
    ("cdl_inside_bar", ("open", "high", "low", "close"), cdl_inside_bar, "Inside-bar pattern."),
    ("cdl_outside_bar", ("open", "high", "low", "close"), cdl_outside_bar, "Outside-bar pattern signed by candle direction."),
    ("candle_body_zscore", ("open", "close", "window"), candle_body_zscore, "Body size z-score versus prior candles."),
    ("candle_range_zscore", ("high", "low", "window"), candle_range_zscore, "Range z-score versus prior candles."),
    ("candle_upper_shadow_zscore", ("open", "high", "close", "window"), candle_upper_shadow_zscore, "Upper-shadow z-score versus prior candles."),
    ("candle_lower_shadow_zscore", ("open", "low", "close", "window"), candle_lower_shadow_zscore, "Lower-shadow z-score versus prior candles."),
    ("candle_body_percentile", ("open", "close", "window"), candle_body_percentile, "Body-size percentile within a window."),
    ("candle_range_percentile", ("high", "low", "window"), candle_range_percentile, "Range percentile within a window."),
    ("candle_gap_atr", ("open", "high", "low", "close", "atr_window"), candle_gap_atr, "Opening gap normalized by rolling ATR."),
    ("candle_body_position", ("open", "high", "low", "close"), candle_body_position, "Body midpoint location inside range."),
    ("candle_overlap_ratio", ("high", "low"), candle_overlap_ratio, "Current/previous candle range-overlap ratio."),
    ("candle_inside_ratio", ("high", "low"), candle_inside_ratio, "Continuous inside/outside range-size encoding."),
    ("candle_close_strength", ("high", "low", "close"), candle_close_strength, "Close location mapped to [-1,1]."),
    ("candle_rejection_upper", ("open", "high", "low", "close"), candle_rejection_upper, "Upper-wick rejection fraction."),
    ("candle_rejection_lower", ("open", "high", "low", "close"), candle_rejection_lower, "Lower-wick rejection fraction."),
    ("cdl_dragonfly_doji", ("open", "high", "low", "close"), cdl_dragonfly_doji, "Dragonfly-doji raw geometry."),
    ("cdl_gravestone_doji", ("open", "high", "low", "close"), cdl_gravestone_doji, "Gravestone-doji raw geometry."),
    ("cdl_hanging_man", ("open", "high", "low", "close"), cdl_hanging_man, "Raw hanging-man geometry."),
    ("cdl_harami", ("open", "high", "low", "close"), cdl_harami, "Bullish +1 / bearish -1 harami body pattern."),
    ("cdl_harami_cross", ("open", "high", "low", "close"), cdl_harami_cross, "Harami whose current candle is a doji."),
    ("cdl_piercing", ("open", "high", "low", "close"), cdl_piercing, "Two-bar bullish piercing geometry."),
    ("cdl_dark_cloud_cover", ("open", "high", "low", "close"), cdl_dark_cloud_cover, "Two-bar bearish dark-cloud-cover geometry."),
    ("cdl_morning_star", ("open", "high", "low", "close"), cdl_morning_star, "Three-bar bullish morning-star geometry."),
    ("cdl_evening_star", ("open", "high", "low", "close"), cdl_evening_star, "Three-bar bearish evening-star geometry."),
    ("cdl_three_white_soldiers", ("open", "high", "low", "close"), cdl_three_white_soldiers, "Three consecutive rising bullish candles."),
    ("cdl_three_black_crows", ("open", "high", "low", "close"), cdl_three_black_crows, "Three consecutive falling bearish candles."),
    ("cdl_tweezer_top", ("open", "high", "low", "close"), cdl_tweezer_top, "Two-bar equal-high bearish reversal geometry."),
    ("cdl_tweezer_bottom", ("open", "high", "low", "close"), cdl_tweezer_bottom, "Two-bar equal-low bullish reversal geometry."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="candle_pattern",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "candle", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsCandle_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="candle_pattern",
        business_category="technical_extension",
        canonical=name,
        source="polars_candle",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
