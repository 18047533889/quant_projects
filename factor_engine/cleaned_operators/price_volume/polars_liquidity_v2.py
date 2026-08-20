# -*- coding: utf-8 -*-
"""Native Polars implementations for liquidity / price-volume interaction operators.

Mirrors the pandas reference backends in ``liquidity_v2`` and
``technical_extensions`` (volume family).  NaN/null discipline:
* pandas ``rolling(...).agg`` skips NaN; polars rolling_* ignores null but
  propagates float NaN, so inputs are normalized via ``fill_nan(None)``.
* pandas ``ewm`` outputs the previous state at a NaN row; polars ``ewm_mean``
  outputs null, so we forward-fill null after ewm.
* pandas ``pct_change`` defaults to ``fill_method='pad'`` (forward fill before
  computing); we emulate with ``fill_null(strategy='forward')``.
* pairwise rolling corr/cov do not exist on this polars Expr; a NumPy window
  loop reproduces pandas ``rolling(w).corr/cov`` with min_periods=w semantics.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamSpec, ParamRole, RelationalParamSpec
from cleaned_operators.registry import OperatorRegistry

_SKIP = frozenset({"date", "stock_code"})


def _check_nonneg(frame: "pl.DataFrame", name: str) -> None:
    """Fail loudly when a finite volume/turnover input is negative (#19)."""
    for c in _cols(frame):
        v = frame[c].to_numpy()
        if np.any((v < 0.0) & np.isfinite(v)):
            raise ValueError(
                f"{name} must be non-negative (input_units='non_negative_volume'); "
                f"found a negative value in {c!r}"
            )


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
    """num / den with den == 0 -> null (mirrors pandas .replace(0, nan))."""
    return pl.when(den != 0).then(num / den).otherwise(None)


def _pct_change(col: pl.Expr) -> pl.Expr:
    """pandas Series.pct_change with fill_method=None (no forward fill).

    pandas >= 2.x defaults ``pct_change(fill_method=None)``; mirror that exactly
    instead of the removed forward-fill default.
    """
    return col.fill_nan(None).pct_change()


def _ewm_span(col: pl.Expr, span: int) -> pl.Expr:
    span = _pi(span, "span")
    return col.fill_nan(None).ewm_mean(span=span, adjust=False, min_samples=span).fill_null(strategy="forward")


def _pair_rolling(x: np.ndarray, y: np.ndarray, w: int, kind: str) -> np.ndarray:
    """Pairwise rolling corr/cov matching pandas rolling(w, min_periods=w)."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - w + 1)
        a, b = x[lo:i + 1], y[lo:i + 1]
        ok = np.isfinite(a) & np.isfinite(b)
        k = int(ok.sum())
        if k < w:
            continue
        a2, b2 = a[ok], b[ok]
        da, db = a2 - a2.mean(), b2 - b2.mean()
        if kind == "cov":
            out[i] = float(np.sum(da * db) / (k - 1))
        else:
            vx, vy = np.sum(da * da), np.sum(db * db)
            out[i] = float(np.sum(da * db) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# liquidity_v2 family
# ---------------------------------------------------------------------------


def average_volume(volume, window):
    w = _pi(window, "window")
    return _result(volume, {
        c: _one(volume, c, pl.col(c).rolling_mean(w, min_samples=w)) for c in _cols(volume)
    })


def adv(close, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        values[c] = _one(frame, c, (pl.col("close").abs() * pl.col("volume")).rolling_mean(w, min_samples=w))
    return _result(close, values)


def abnormal_volume(volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(volume):
        base = pl.col(c).shift(1).rolling_mean(w, min_samples=w)
        values[c] = _one(volume, c, _safe_div(pl.col(c), base) - 1.0)
    return _result(volume, values)


def volume_volatility(volume, window):
    w = _pi(window, "window", 2)
    # R11 round-3 #18: guarded growth (0 base -> null) mirrors the pandas twin /
    # SQL ``NULLIF(v,0)``; a raw pct_change would blow up to inf at a 0 base.
    return _result(volume, {
        c: _one(volume, c, (_safe_div(pl.col(c), pl.col(c).shift(1)) - 1.0).rolling_std(w, min_samples=w)) for c in _cols(volume)
    })


def volume_autocorr(volume, window=20, lag=1):
    # R11 round-3 #17: ``window`` counts ALIGNED PAIRS; the raw lookback is
    # ``window + lag`` prior bars (first output at raw index ``window + lag - 1``)
    # so the lag term has genuine history.  ``lag < window`` is a declared
    # RelationalParamSpec (mirror of the pandas twin).
    w = _pi(window, "window", 3)
    lag = _pi(lag, "lag")
    values = {}
    for c in _cols(volume):
        x = volume[c].to_numpy()
        y = np.full_like(x, np.nan)
        y[lag:] = x[:-lag]
        values[c] = pl.Series(name=c, values=_pair_rolling(x, y, w, "corr"))
    return _result(volume, values)


def amihud_illiquidity(ret, close, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(ret, close, volume):
        frame = pl.DataFrame({"ret": ret[c], "close": close[c], "volume": volume[c]})
        dv = pl.col("close").abs() * pl.col("volume")
        raw = _safe_div(pl.col("ret").abs(), dv)
        values[c] = _one(frame, c, raw.rolling_mean(w, min_samples=w))
    return _result(ret, values)


def price_impact(ret, dollar_volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(ret, dollar_volume):
        frame = pl.DataFrame({"ret": ret[c], "dv": dollar_volume[c]})
        raw = _safe_div(pl.col("ret").abs(), pl.col("dv"))
        values[c] = _one(frame, c, raw.rolling_mean(w, min_samples=w))
    return _result(ret, values)


def return_per_turnover(ret, turnover):
    values = {}
    for c in _cols(ret, turnover):
        frame = pl.DataFrame({"ret": ret[c], "turnover": turnover[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("ret"), pl.col("turnover")))
    return _result(ret, values)


def _prior_zscore(x, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(x):
        shifted = pl.col(c).shift(1)
        mean = shifted.rolling_mean(w, min_samples=w)
        std = shifted.rolling_std(w, min_samples=w)
        values[c] = _one(x, c, _safe_div(pl.col(c) - mean, std))
    return _result(x, values)


def volume_acceleration(volume, short_window, long_window):
    s = _pi(short_window, "short_window")
    l = _pi(long_window, "long_window")
    if s >= l:
        raise ValueError("short_window must be < long_window")
    values = {}
    for c in _cols(volume):
        short = pl.col(c).rolling_mean(s, min_samples=s)
        long = pl.col(c).rolling_mean(l, min_samples=l)
        values[c] = _one(volume, c, _safe_div(short, long) - 1.0)
    return _result(volume, values)


def _up_ratio(ret, volume, window, *, positive):
    w = _pi(window, "window")
    _check_nonneg(volume, "volume")
    values = {}
    for c in _cols(ret, volume):
        frame = pl.DataFrame({"ret": ret[c], "volume": volume[c]})
        # R4-29: a missing/unknown return must NOT count as "not up/down"
        # volume (unknown != zero).  Mirror the pandas reference exactly:
        # invalid rows are masked to null so a window with any missing input
        # emits NaN; only a fully-known window with zero up/down volume -> 0.
        retf = pl.col("ret").fill_nan(None)
        voln = pl.col("volume").fill_nan(None)
        valid = retf.is_not_null() & voln.is_not_null()
        cond = retf > 0 if positive else retf < 0
        contrib = pl.when(valid & cond).then(pl.col("volume")).otherwise(0.0)
        up = pl.when(valid).then(contrib).otherwise(None).rolling_sum(w, min_samples=w)
        # #19: volume is non-negative by contract — no abs() on the denominator.
        total = voln.rolling_sum(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(up, total))
    return _result(ret, values)


def up_volume_ratio(ret, volume, window):
    return _up_ratio(ret, volume, window, positive=True)


def down_volume_ratio(ret, volume, window):
    return _up_ratio(ret, volume, window, positive=False)


def volume_weighted_return(ret, volume, window):
    w = _pi(window, "window")
    _check_nonneg(volume, "volume")
    values = {}
    for c in _cols(ret, volume):
        frame = pl.DataFrame({"ret": ret[c], "volume": volume[c]})
        num = (pl.col("ret") * pl.col("volume")).rolling_sum(w, min_samples=w)
        # #19: volume is non-negative by contract — no abs() on the denominator.
        den = pl.col("volume").rolling_sum(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(num, den))
    return _result(ret, values)


def price_turnover_divergence(close, turnover, price_window, turnover_window):
    return price_volume_divergence(close, turnover, price_window, turnover_window)


def up_down_volume_ratio(ret, volume, window):
    up = up_volume_ratio(ret, volume, window)
    down = down_volume_ratio(ret, volume, window)
    values = {}
    for c in _cols(up, down):
        frame = pl.DataFrame({"up": up[c], "down": down[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("up"), pl.col("down")))
    return _result(up, values)


def volume_weighted_momentum(close, volume, window):
    w = _pi(window, "window")
    _check_nonneg(volume, "volume")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        ret = _pct_change(pl.col("close"))
        num = (ret * pl.col("volume")).rolling_sum(w, min_samples=w)
        # #19: volume is non-negative by contract — no abs() on the denominator.
        den = pl.col("volume").rolling_sum(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(num, den))
    return _result(close, values)


def price_volume_divergence(close, volume, price_window, volume_window):
    pw = _pi(price_window, "price_window")
    vw = _pi(volume_window, "volume_window")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        pr = pl.col("close") / pl.col("close").shift(pw) - 1.0
        vr = _safe_div(pl.col("volume"), pl.col("volume").shift(vw)) - 1.0
        values[c] = _one(frame, c, pr - vr)
    return _result(close, values)


def return_volume_beta(ret, volume, window):
    w = _pi(window, "window", 3)
    _check_nonneg(volume, "volume")
    values = {}
    for c in _cols(ret, volume):
        x = ret[c].to_numpy()
        # R11 round-3 #18: guarded growth (0 base -> null) mirrors the pandas twin.
        vc = _safe_div(pl.col(c), pl.col(c).shift(1)) - 1.0
        y = volume.select(vc.alias("vc"))["vc"].to_numpy()
        cov = _pair_rolling(x, y, w, "cov")
        # rolling var (ddof=1) via numpy to keep paired-sample semantics
        var_out = np.full(len(y), np.nan)
        for i in range(len(y)):
            lo = max(0, i - w + 1)
            seg = y[lo:i + 1]
            ok = np.isfinite(seg)
            if int(ok.sum()) < w:
                continue
            seg2 = seg[ok]
            var_out[i] = float(np.sum((seg2 - seg2.mean()) ** 2) / (len(seg2) - 1))
        values[c] = pl.Series(name=c, values=np.where(var_out != 0, cov / var_out, np.nan))
    return _result(ret, values)


def _mf_multiplier(high, low, close):
    spread = pl.col("high") - pl.col("low")
    return pl.when(spread != 0).then(
        ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / spread
    ).otherwise(None)


def rolling_adl_flow(high, low, close, volume, window):
    # R11 round-3 #20: BOUNDED ROLLING money-flow sum over an explicit window —
    # NOT the classic cumulative Accumulation/Distribution Line (which integrates
    # the full history with a running total).  No truly cumulative/stateful ADL
    # exists in the catalog; the name says what this computes.
    w = _pi(window, "window")
    values = {}
    for c in _cols(high, low, close, volume):
        frame = pl.DataFrame({
            "high": high[c], "low": low[c], "close": close[c], "volume": volume[c],
        })
        flow = _mf_multiplier(pl.col("high"), pl.col("low"), pl.col("close")) * pl.col("volume")
        values[c] = _one(frame, c, flow.rolling_sum(w, min_samples=w))
    return _result(close, values)


# Legacy in-module alias for the renamed rolling-flow operator.
ADL = rolling_adl_flow


def ChaikinOscillator(high, low, close, volume, fast_window, slow_window, adl_window):
    f = _pi(fast_window, "fast_window")
    s = _pi(slow_window, "slow_window")
    if f >= s:
        raise ValueError("fast_window must be < slow_window")
    values = {}
    for c in _cols(high, low, close, volume):
        frame = pl.DataFrame({
            "high": high[c], "low": low[c], "close": close[c], "volume": volume[c],
        })
        flow = _mf_multiplier(pl.col("high"), pl.col("low"), pl.col("close")) * pl.col("volume")
        # #20: built on the BOUNDED ROLLING ADL flow, NOT the cumulative ADL.
        adl = flow.rolling_sum(_pi(adl_window, "adl_window"), min_samples=_pi(adl_window, "adl_window"))
        fast = _ewm_span(adl, f)
        slow = _ewm_span(adl, s)
        values[c] = _one(frame, c, fast - slow)
    return _result(close, values)


def ForceIndex(close, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        raw = pl.col("close").diff() * pl.col("volume")
        values[c] = _one(frame, c, _ewm_span(raw, w))
    return _result(close, values)


def EaseOfMovement(high, low, volume, window, volume_scale=1.0):
    w = _pi(window, "window", 2)
    scale = _pf(volume_scale, "volume_scale", 0)
    values = {}
    for c in _cols(high, low, volume):
        frame = pl.DataFrame({"high": high[c], "low": low[c], "volume": volume[c]})
        midpoint = (pl.col("high") + pl.col("low")) / 2.0
        distance = midpoint.diff()
        box = _safe_div(pl.col("high") - pl.col("low"), pl.col("volume") / float(scale))
        raw = distance * box
        values[c] = _one(frame, c, raw.rolling_mean(w, min_samples=w))
    return _result(volume, values)


def _bounded_nvi_pvi(close, volume, window, *, volume_up):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        # R4-29/R4-96: a missing volume or missing volume-growth bar must NOT
        # become a silent "no-volume day".  ``when(null)`` falls through to
        # ``otherwise(0.0)``, which injected fabricated zero-return bars into the
        # index (the pandas reference NaN-outs the invalid rows via a ``valid``
        # mask).  Only a KNOWN volume comparison with a known return contributes.
        vol = pl.col("volume").fill_nan(None)
        ret = _pct_change(pl.col("close"))
        cond = (vol > vol.shift(1)) if volume_up else (vol < vol.shift(1))
        valid = cond.is_not_null() & ret.is_not_null()
        r = pl.when(valid).then(pl.when(cond).then(ret).otherwise(0.0)).otherwise(None)
        log_sum = r.clip(lower_bound=-0.999999).log1p().rolling_sum(w, min_samples=w)
        values[c] = _one(frame, c, log_sum.exp() - 1.0)
    return _result(close, values)


def bounded_nvi(close, volume, window):
    return _bounded_nvi_pvi(close, volume, window, volume_up=False)


def bounded_pvi(close, volume, window):
    return _bounded_nvi_pvi(close, volume, window, volume_up=True)


def zero_return_ratio(ret, window, epsilon=1e-12):
    w = _pi(window, "window")
    eps = _pf(epsilon, "epsilon", 0)
    values = {}
    for c in _cols(ret):
        val = pl.col(c).abs()
        # R4-29/R4-96: a missing return is NaN (cannot judge), never a
        # "non-zero" bar — ``NaN <= eps`` is False and would inflate the ratio
        # with a 0 (the pandas reference NaN-outs invalid rows first).
        valid = pl.col(c).is_not_null() & pl.col(c).is_not_nan()
        flag = (
            pl.when(valid).then(pl.when(val <= eps).then(1.0).otherwise(0.0))
            .otherwise(None)
        )
        values[c] = _one(ret, c, flag.rolling_mean(w, min_samples=w))
    return _result(ret, values)


def roll_spread_proxy(ret, window):
    w = _pi(window, "window", 3)
    values = {}
    for c in _cols(ret):
        x = ret[c].to_numpy()
        y = np.empty_like(x)
        y[0] = np.nan
        y[1:] = x[:-1]
        cov = _pair_rolling(x, y, w, "cov")
        values[c] = pl.Series(name=c, values=2.0 * np.sqrt(np.maximum(-cov, 0.0)))
    return _result(ret, values)


def corwin_schultz_spread(high, low, window):
    w = _pi(window, "window", 2)
    den = 3.0 - 2.0 * np.sqrt(2.0)
    values = {}
    for c in _cols(high, low):
        frame = pl.DataFrame({"high": high[c], "low": low[c]})
        h0 = pl.when(pl.col("high") != 0).then(pl.col("high")).otherwise(None)
        l0 = pl.when(pl.col("low") != 0).then(pl.col("low")).otherwise(None)
        loghl = (h0 / l0).log()
        beta = loghl.pow(2) + loghl.shift(1).pow(2)
        high2 = ((h0 + h0.shift(1)) + (h0 - h0.shift(1)).abs()) / 2.0
        low2 = ((l0 + l0.shift(1)) - (l0 - l0.shift(1)).abs()) / 2.0
        gamma = (high2 / low2).log().pow(2)
        alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / den - (gamma / den).sqrt()
        alpha = alpha.clip(lower_bound=0.0)
        spread = 2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp())
        values[c] = _one(frame, c, spread.rolling_mean(w, min_samples=w))
    return _result(high, values)


def high_low_spread_proxy(high, low, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(high, low):
        frame = pl.DataFrame({"high": high[c], "low": low[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("high"), pl.col("low")).log().rolling_mean(w, min_samples=w))
    return _result(high, values)


def turnover_adjusted_volatility(ret, turnover, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(ret, turnover):
        frame = pl.DataFrame({"ret": ret[c], "turnover": turnover[c]})
        vol = pl.col("ret").rolling_std(w, min_samples=w)
        act = pl.col("turnover").rolling_mean(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(vol, act))
    return _result(ret, values)


def volume_price_range_density(volume, high, low, window):
    # R11 round-3 #21: Volume/(High-Low) is a unit-dependent "volume per unit
    # price range" density.  The canonical name states the semantic (output unit:
    # ``volume_per_price_range``).  A non-positive range (== 0 or High<Low) ->
    # null so the density cannot explode as range -> 0.
    w = _pi(window, "window")
    values = {}
    for c in _cols(volume, high, low):
        frame = pl.DataFrame({"volume": volume[c], "high": high[c], "low": low[c]})
        rng = pl.col("high") - pl.col("low")
        raw = pl.when(rng > 0).then(pl.col("volume") / rng).otherwise(None)
        values[c] = _one(frame, c, raw.rolling_mean(w, min_samples=w))
    return _result(volume, values)


# Legacy in-module alias for the renamed density op.
volume_to_range = volume_price_range_density


# ---------------------------------------------------------------------------
# technical_extensions volume family
# ---------------------------------------------------------------------------


def rolling_vwap(price, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(price, volume):
        frame = pl.DataFrame({"price": price[c], "volume": volume[c]})
        num = (pl.col("price") * pl.col("volume")).rolling_sum(w, min_samples=w)
        den = pl.col("volume").rolling_sum(w, min_samples=w)
        values[c] = _one(frame, c, _safe_div(num, den))
    return _result(price, values)


def vwap_deviation(price, volume, window):
    vwap = rolling_vwap(price, volume, window)
    values = {}
    for c in _cols(price, volume):
        frame = pl.DataFrame({"price": price[c], "vwap": vwap[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("price"), pl.col("vwap")) - 1.0)
    return _result(price, values)


def relative_volume(volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(volume):
        baseline = pl.col(c).shift(1).rolling_mean(w, min_samples=w)
        values[c] = _one(volume, c, _safe_div(pl.col(c), baseline))
    return _result(volume, values)


def dollar_volume(close, volume):
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        values[c] = _one(frame, c, pl.col("close") * pl.col("volume"))
    return _result(close, values)


def volume_momentum(volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(volume):
        values[c] = _one(volume, c, _safe_div(pl.col(c), pl.col(c).shift(w)) - 1.0)
    return _result(volume, values)


def return_volume_corr(ret, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(ret, volume):
        values[c] = pl.Series(name=c, values=_pair_rolling(ret[c].to_numpy(), volume[c].to_numpy(), w, "corr"))
    return _result(ret, values)


def abs_return_volume_corr(ret, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(ret, volume):
        values[c] = pl.Series(name=c, values=_pair_rolling(np.abs(ret[c].to_numpy()), volume[c].to_numpy(), w, "corr"))
    return _result(ret, values)


def signed_volume(ret, volume):
    values = {}
    for c in _cols(ret, volume):
        frame = pl.DataFrame({"ret": ret[c], "volume": volume[c]})
        values[c] = _one(frame, c, pl.col("ret").sign() * pl.col("volume"))
    return _result(volume, values)


def signed_dollar_volume(ret, close, volume):
    values = {}
    for c in _cols(ret, close, volume):
        frame = pl.DataFrame({"ret": ret[c], "close": close[c], "volume": volume[c]})
        values[c] = _one(frame, c, pl.col("ret").sign() * pl.col("close") * pl.col("volume"))
    return _result(volume, values)


def rolling_obv(close, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        flow = pl.col("close").diff().sign() * pl.col("volume")
        values[c] = _one(frame, c, flow.rolling_sum(w, min_samples=w))
    return _result(close, values)


def rolling_pvt(close, volume, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close, volume):
        frame = pl.DataFrame({"close": close[c], "volume": volume[c]})
        flow = _pct_change(pl.col("close")) * pl.col("volume")
        values[c] = _one(frame, c, flow.rolling_sum(w, min_samples=w))
    return _result(close, values)


# ---------------------------------------------------------------------------
# scatter: naming / structure / ashare / microstructure volume operators
# ---------------------------------------------------------------------------


def ts_average_volume(volume, window):
    return average_volume(volume, window)


def average_turnover(turnover, window):
    w = _pi(window, "window")
    return _result(turnover, {
        c: _one(turnover, c, pl.col(c).rolling_mean(w, min_samples=w)) for c in _cols(turnover)
    })


def ts_impulse_return(close, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close):
        values[c] = _one(close, c, _safe_div(pl.col(c), pl.col(c).shift(w)) - 1.0)
    return _result(close, values)


def ts_impulse_strength(close, window, vol_window):
    # R11 round-3 #26: the scale must be volatility_{t-1} — estimated on PRIOR
    # data, excluding the current impulse.  ``rolling_std(...).shift(1)`` makes
    # the denominator use rows [t-vol_window, t-1] so the impulse return cannot
    # inflate its own volatility denominator.
    w = _pi(window, "window")
    vw = _pi(vol_window, "vol_window", 2)
    values = {}
    for c in _cols(close):
        ret = _safe_div(pl.col(c), pl.col(c).shift(w)) - 1.0
        rv = _pct_change(pl.col(c)).rolling_std(vw, min_samples=vw).shift(1)
        values[c] = _one(close, c, _safe_div(ret, rv * np.sqrt(float(w))))
    return _result(close, values)


def ts_impulse_volume(volume, window, baseline_window):
    w = _pi(window, "window")
    b = _pi(baseline_window, "baseline_window")
    values = {}
    for c in _cols(volume):
        recent = pl.col(c).rolling_mean(w, min_samples=w)
        base = pl.col(c).shift(w).rolling_mean(b, min_samples=b)
        values[c] = _one(volume, c, _safe_div(recent, base) - 1.0)
    return _result(volume, values)


def _slope_1d(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return np.nan
    if not np.isfinite(values).all():
        return np.nan
    x = np.arange(n, dtype=float)
    xb = x.mean()
    den = np.sum((x - xb) ** 2)
    if den == 0:
        return np.nan
    return float(np.sum((x - xb) * (values - values.mean())) / den)


def ts_consolidation_slope(close, window):
    w = _pi(window, "window", 2)
    n = close.height
    values = {}
    for c in _cols(close):
        arr = close[c].to_numpy()
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            lo = max(0, i - w + 1)
            seg = arr[lo:i + 1]
            if int(np.isfinite(seg).sum()) < w:
                continue
            out[i] = _slope_1d(seg)
        values[c] = pl.Series(name=c, values=out)
    return _result(close, values)


def ts_consolidation_volume_decay(volume, window):
    slope = ts_consolidation_slope(volume, window)
    return _result(volume, {c: -slope[c] for c in _cols(slope)})


def true_turnover_rate(volume, free_float_shares):
    values = {}
    for c in _cols(volume, free_float_shares):
        frame = pl.DataFrame({"volume": volume[c], "shares": free_float_shares[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("volume"), pl.col("shares")))
    return _result(volume, values)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("adv", ("close", "volume", "window"), adv, "Average dollar volume."),
    ("abnormal_volume", ("volume", "window"), abnormal_volume, "Current volume versus prior rolling mean."),
    ("abnormal_turnover", ("turnover", "window"), abnormal_volume, "Current turnover versus prior rolling mean."),
    ("volume_volatility", ("volume", "window"), volume_volatility, "Volatility of volume growth."),
    ("turnover_volatility", ("turnover", "window"), volume_volatility, "Volatility of turnover growth."),
    ("volume_autocorr", ("volume", "window", "lag"), volume_autocorr, "Rolling volume autocorrelation."),
    ("turnover_autocorr", ("turnover", "window", "lag"), volume_autocorr, "Rolling turnover autocorrelation."),
    ("amihud_illiquidity", ("ret", "close", "volume", "window"), amihud_illiquidity, "Rolling Amihud illiquidity proxy."),
    ("price_impact", ("ret", "dollar_volume", "window"), price_impact, "Absolute return per dollar-volume price-impact proxy."),
    ("return_per_turnover", ("ret", "turnover"), return_per_turnover, "Return per unit turnover."),
    ("volume_shock", ("volume", "window"), _prior_zscore, "Prior-window volume z-score."),
    ("turnover_shock", ("turnover", "window"), _prior_zscore, "Prior-window turnover z-score."),
    ("volume_acceleration", ("volume", "short_window", "long_window"), volume_acceleration, "Short/long average volume acceleration."),
    ("turnover_acceleration", ("turnover", "short_window", "long_window"), volume_acceleration, "Short/long turnover acceleration."),
    ("up_volume_ratio", ("ret", "volume", "window"), up_volume_ratio, "Share of recent volume on positive-return bars."),
    ("down_volume_ratio", ("ret", "volume", "window"), down_volume_ratio, "Share of recent volume on negative-return bars."),
    ("signed_volume_imbalance", ("ret", "volume", "window"), lambda ret, volume, window: up_volume_ratio(ret, volume, window) - down_volume_ratio(ret, volume, window), "Up-volume share minus down-volume share."),
    ("up_down_volume_ratio", ("ret", "volume", "window"), up_down_volume_ratio, "Up-volume to down-volume ratio."),
    ("volume_weighted_return", ("ret", "volume", "window"), volume_weighted_return, "Rolling volume-weighted return."),
    ("volume_weighted_momentum", ("close", "volume", "window"), volume_weighted_momentum, "Rolling volume-weighted close return."),
    ("price_volume_divergence", ("close", "volume", "price_window", "volume_window"), price_volume_divergence, "Price momentum minus volume momentum."),
    ("price_turnover_divergence", ("close", "turnover", "price_window", "turnover_window"), price_turnover_divergence, "Price momentum minus turnover momentum."),
    ("return_volume_beta", ("ret", "volume", "window"), return_volume_beta, "Rolling beta of return to volume growth."),
    ("return_turnover_beta", ("ret", "turnover", "window"), return_volume_beta, "Rolling beta of return to turnover growth."),
    ("rolling_adl_flow", ("high", "low", "close", "volume", "window"), rolling_adl_flow, "Bounded rolling accumulation/distribution money-flow over an explicit window (NOT the cumulative ADL)."),
    ("ChaikinOscillator", ("high", "low", "close", "volume", "fast_window", "slow_window", "adl_window"), ChaikinOscillator, "Chaikin oscillator over the bounded rolling ADL flow (NOT cumulative ADL)."),
    ("ForceIndex", ("close", "volume", "window"), ForceIndex, "EMA-smoothed price-change times volume."),
    ("EaseOfMovement", ("high", "low", "volume", "window", "volume_scale"), EaseOfMovement, "Ease-of-Movement with explicit smoothing and volume scale."),
    ("bounded_nvi", ("close", "volume", "window"), bounded_nvi, "Bounded Negative Volume Index return."),
    ("bounded_pvi", ("close", "volume", "window"), bounded_pvi, "Bounded Positive Volume Index return."),
    ("zero_return_ratio", ("ret", "window", "epsilon"), zero_return_ratio, "Fraction of near-zero returns in recent window."),
    ("roll_spread_proxy", ("ret", "window"), roll_spread_proxy, "Roll implied-spread proxy from negative first-order return covariance."),
    ("corwin_schultz_spread", ("high", "low", "window"), corwin_schultz_spread, "Corwin-Schultz high-low spread proxy."),
    ("high_low_spread_proxy", ("high", "low", "window"), high_low_spread_proxy, "Rolling log high-low spread proxy."),
    ("turnover_adjusted_volatility", ("ret", "turnover", "window"), turnover_adjusted_volatility, "Return volatility scaled by trading activity."),
    ("volume_price_range_density", ("volume", "high", "low", "window"), volume_price_range_density, "Rolling volume per unit intraday range (Volume/(High-Low); unit: volume per price range)."),
    ("rolling_vwap", ("price", "volume", "window"), rolling_vwap, "Rolling VWAP."),
    ("vwap_deviation", ("price", "volume", "window"), vwap_deviation, "Price relative to rolling VWAP."),
    ("relative_volume", ("volume", "window"), relative_volume, "Current volume relative to prior baseline."),
    ("dollar_volume", ("close", "volume"), dollar_volume, "Price times volume."),
    ("dollar_volume_zscore", ("close", "volume", "window"), lambda close, volume, window: _prior_zscore(dollar_volume(close, volume), window), "Prior-window dollar-volume z-score."),
    ("volume_momentum", ("volume", "window"), volume_momentum, "Volume ratio to its level `window` bars ago."),
    ("turnover_momentum", ("turnover", "window"), volume_momentum, "Turnover ratio to its level `window` bars ago."),
    ("volume_zscore", ("volume", "window"), _prior_zscore, "Prior-window volume z-score (shifted baseline)."),
    ("turnover_zscore", ("turnover", "window"), _prior_zscore, "Prior-window turnover z-score (shifted baseline)."),
    ("return_volume_corr", ("ret", "volume", "window"), return_volume_corr, "Rolling return-volume correlation."),
    ("abs_return_volume_corr", ("ret", "volume", "window"), abs_return_volume_corr, "Rolling |return|-volume correlation."),
    ("signed_volume", ("ret", "volume"), signed_volume, "Sign-scaled volume."),
    ("signed_dollar_volume", ("ret", "close", "volume"), signed_dollar_volume, "Sign-scaled dollar volume."),
    ("rolling_obv", ("close", "volume", "window"), rolling_obv, "Bounded-window OBV flow."),
    ("rolling_pvt", ("close", "volume", "window"), rolling_pvt, "Bounded-window price-volume-trend flow."),
    ("average_turnover", ("turnover", "window"), average_turnover, "Rolling average turnover/activity."),
    ("ts_average_volume", ("volume", "window"), ts_average_volume, "Rolling average volume (renamed field-safe alias)."),
    ("ts_impulse_return", ("close", "window"), ts_impulse_return, "Return over an explicit impulse window."),
    ("ts_impulse_strength", ("close", "window", "vol_window"), ts_impulse_strength, "Impulse return scaled by realized volatility."),
    ("ts_impulse_volume", ("volume", "window", "baseline_window"), ts_impulse_volume, "Impulse-period volume versus a prior baseline."),
    ("ts_consolidation_volume_decay", ("volume", "window"), ts_consolidation_volume_decay, "Positive score when volume trends down through consolidation."),
    ("true_turnover_rate", ("volume", "free_float_shares"), true_turnover_rate, "True turnover rate: volume / free-float shares."),
)


# R11 round-3: per-operator metadata contracts (mirror of the pandas module) —
# relational feasibility, per-parameter history semantics and typed input units.
_AUTOCORR_REL = [
    RelationalParamSpec(
        "lag < window",
        message="lag must be < window (window counts aligned pairs; raw history = window + lag)",
    ),
]
_AUTOCORR_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=3, history_formula="window + lag"),
    "lag": ParamSpec(dtype=int, min=1, history_semantics="exact_rows"),
}
_EXTRA: dict[str, dict] = {
    "volume_autocorr": {"param_specs": dict(_AUTOCORR_PARAM_SPECS), "relational_specs": list(_AUTOCORR_REL)},
    "turnover_autocorr": {"param_specs": dict(_AUTOCORR_PARAM_SPECS), "relational_specs": list(_AUTOCORR_REL)},
    "volume_volatility": {"input_units": {"volume": "non_negative_volume"}},
    "turnover_volatility": {"input_units": {"turnover": "non_negative_volume"}},
    "price_volume_divergence": {"input_units": {"volume": "non_negative_volume"}},
    "price_turnover_divergence": {"input_units": {"turnover": "non_negative_volume"}},
    "return_volume_beta": {"input_units": {"volume": "non_negative_volume"}},
    "return_turnover_beta": {"input_units": {"turnover": "non_negative_volume"}},
    "up_volume_ratio": {"input_units": {"volume": "non_negative_volume"}},
    "down_volume_ratio": {"input_units": {"volume": "non_negative_volume"}},
    "volume_weighted_return": {"input_units": {"volume": "non_negative_volume"}},
    "volume_weighted_momentum": {"input_units": {"volume": "non_negative_volume"}},
    "rolling_adl_flow": {"input_units": {"volume": "non_negative_volume"}},
    "volume_price_range_density": {
        "input_units": {"volume": "non_negative_volume", "high": "price", "low": "price"},
        "output_unit": "volume_per_price_range",
    },
    "zero_return_ratio": {
        "param_specs": {"epsilon": ParamSpec(dtype=float, min=0.0, searchable=False, param_role=ParamRole.NUMERICAL)},
    },
}


def _register(name: str, params: tuple[str, ...], function: Callable, description: str, *, extra: dict | None = None) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="price_volume_extension",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "bounded_history", "polars", "native"],
    )
    if name == "EaseOfMovement":
        # Parity with the pandas reference (P1-86 / R5-34): ``volume_scale`` is
        # a pure unit-conversion constant — it multiplies the whole output
        # uniformly and leaves cross-sectional ordering invariant, so it must
        # not be an alpha-search dimension.
        metadata.param_specs = {"volume_scale": ParamSpec(dtype=float, searchable=False)}
        metadata.tags = [*(metadata.tags or ()), "unit_conversion_only"]
    if extra:
        if extra.get("param_specs"):
            _specs = dict(getattr(metadata, "param_specs", None) or {})
            _specs.update(extra["param_specs"])
            metadata.param_specs = _specs
        # base_polars.OperatorMetadata has no declared relational_specs/input_units
        # fields; setattr is the duck-typed mirror so validate_operator_call and
        # the semantic audit read the SAME contracts as the pandas twin.
        if extra.get("relational_specs"):
            metadata.relational_specs = list(extra["relational_specs"])
        if extra.get("input_units"):
            metadata.input_units = dict(extra["input_units"])
        if extra.get("output_unit"):
            metadata.output_unit = extra["output_unit"]

    def _calculate_series(self, *args, _fn=function, **kwargs):
        return _fn(*args, **kwargs)

    cls = type(
        f"PolarsLiquidityV2_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="price_volume_extension",
        business_category="price_volume",
        canonical=name,
        source="polars_liquidity_v2",
        backend="polars",
        status="production",
        replace=True,
        expected_old_source="factor_dsl_np",
        replacement_reason="Consolidating polars native operators into polars_liquidity_v2",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description, extra=_EXTRA.get(_name))

# R11 round-3 #20 / #21: honest renames (mirror of the pandas module).  The old
# names stay as resolving aliases so existing recipes keep loading.
_RENAMES = {"ADL": "rolling_adl_flow", "volume_to_range": "volume_price_range_density"}
for _old, _new in _RENAMES.items():
    try:
        OperatorRegistry.register_alias(_old, _new)
    except (KeyError, ValueError):
        pass
import cleaned_operators.operator_surface as _surface
# Keep the live extended surface in sync with the rename (mirror of the pandas
# module): old names leave the static partition, new canonicals enter it.
_surface.extend_extended_only(set(_RENAMES.values()))
_surface.retract_extended_only(set(_RENAMES.keys()))
