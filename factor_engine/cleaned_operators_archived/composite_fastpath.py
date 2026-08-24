# -*- coding: utf-8 -*-
"""Composable and fused fast paths for common factor operators.

Design rule
-----------
* Reuse registered primitives when an indicator is a straightforward DAG.
* Fuse shared intermediates when composing operators would repeat expensive
  rolling/EWM work (KAMA, VP-MACD and Wilder families).
* Every Polars backend stays in Polars/NumPy space and never converts to pandas.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    pl,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry

PANDAS_SOURCE = "composite_fastpath_primitives"
POLARS_SOURCE = "composite_fastpath_native_polars"

FASTPATH_CANONICALS = frozenset(
    {
        "ADXR",
        "ATR",
        "ATR_WILDER",
        "BollingerBands",
        "BollingerLower",
        "BollingerUpper",
        "DPO",
        "KAMA",
        "MOM",
        "OBV",
        "ROC",
        "RSI_WILDER",
        "StochasticD",
        "StochasticK",
        "TRIX",
        "WilliamsR",
        "WMA",
        "close_gap",
        "log_returns",
        "open_gap",
        "sharpe_ratio",
        "volatility",
        "vp_weighted_price",
        "vpmacd",
        "vpmacd_signal",
        "vwap",
    }
)


def _call(name: str, backend: str, *args: Any, **kwargs: Any):
    operator = OperatorRegistry.get(name, backend=backend)
    if operator is None:
        raise RuntimeError(f"required primitive missing: {name}/{backend}")
    return operator.calculate(*args, **kwargs)


def _register(
    name: str,
    category: str,
    params: list[str],
    description: str,
    pandas_fn: Callable[..., pd.DataFrame] | None,
    polars_fn: Callable[..., "pl.DataFrame"] | None,
) -> None:
    if pandas_fn is not None:
        OperatorRegistry.register(
            PandasFunctionOperator(name, category, params, description, pandas_fn),
            canonical=name,
            backend="pandas_numpy",
            source=PANDAS_SOURCE,
            status="production",
            backend_explicit=True,
        )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(name, category, params, description, polars_fn),
            canonical=name,
            backend="polars",
            source=POLARS_SOURCE,
            status="production",
            backend_explicit=True,
        )


def _pd_delay(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return _call("ts_delay", "pandas_numpy", x, positive_int(periods, "periods"))


def _pd_mean(x: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    return _call("ts_mean", "pandas_numpy", x, positive_int(window, "window"), min_periods=min_periods)


def _pd_std(x: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    return _call("ts_std", "pandas_numpy", x, positive_int(window, "window"), min_periods=min_periods)


def _pd_min(x: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    return _call("ts_min", "pandas_numpy", x, positive_int(window, "window"), min_periods=min_periods)


def _pd_max(x: pd.DataFrame, window: int, min_periods: int = 1) -> pd.DataFrame:
    return _call("ts_max", "pandas_numpy", x, positive_int(window, "window"), min_periods=min_periods)


def _pd_ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return _call("ts_ema", "pandas_numpy", x, positive_int(span, "span"))


def _pd_true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    high, low, close = aligned_pd(high, low, close)
    previous_close = _pd_delay(close, 1)
    return pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(dtype=float),
                (high - previous_close).abs().to_numpy(dtype=float),
                (low - previous_close).abs().to_numpy(dtype=float),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )


def _pd_wilder(x: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    w = positive_int(window, "window")
    mp = w if min_periods is None else positive_int(min_periods, "min_periods")
    return x.ewm(alpha=1.0 / w, adjust=False, min_periods=mp).mean()


def _pl_align(*frames: "pl.DataFrame") -> list[str]:
    """Strict multi-panel alignment (WS-B #242): never take the column intersection.

    Every frame must share the same PanelIdentity (same time axis, same ordered
    instrument columns).  A stock missing from one frame (or permuted) raises
    ``ValueError`` instead of silently dropping it.  The reserved bridge
    ``__fe_time__`` metadata column is excluded from the instrument set even if
    a caller's local skip set does not list it.
    """
    from cleaned_operators.common._polars_bridge import (
        FE_TIME_COL,
        verify_frames_share_identity,
    )

    verify_frames_share_identity("composite_fastpath._pl_align", *frames)

    def _instruments(frame: "pl.DataFrame") -> list[str]:
        return [c for c in pl_cols(frame) if c != FE_TIME_COL]

    cols = _instruments(frames[0])
    for i, frame in enumerate(frames[1:], start=1):
        other_cols = _instruments(frame)
        if other_cols != cols:
            raise ValueError(
                f"composite_fastpath._pl_align: panel input {i} instrument columns "
                f"{other_cols} != base {cols} (a silent intersection would drop stocks)"
            )
    return cols


def _pl_wilder_expr(expr: "pl.Expr", window: int, min_samples: int | None = None) -> "pl.Expr":
    w = positive_int(window, "window")
    mp = w if min_samples is None else positive_int(min_samples, "min_samples")
    return expr.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=mp)


def _pl_true_range_expr(high: "pl.Expr", low: "pl.Expr", close: "pl.Expr") -> "pl.Expr":
    previous_close = close.shift(1)
    return pl.max_horizontal(
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    )


def _pl_ewm_frame(x: "pl.DataFrame", span: int) -> "pl.DataFrame":
    s = positive_int(span, "span")
    alpha = 2.0 / (s + 1.0)
    return x.with_columns(
        [pl.col(c).ewm_mean(alpha=alpha, adjust=False).alias(c) for c in pl_cols(x)]
    )


def pd_atr(high, low, close, window=14, **_):
    return _pd_mean(_pd_true_range(high, low, close), window, 1)


def pl_atr(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    cols = _pl_align(high, low, close)
    return close.with_columns(
        [
            _pl_true_range_expr(high[c], low[c], close[c])
            .rolling_mean(window_size=w, min_samples=1)
            .alias(c)
            for c in cols
        ]
    )


def pd_atr_wilder(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    return _pd_wilder(_pd_true_range(high, low, close), w, w)


def pl_atr_wilder(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    cols = _pl_align(high, low, close)
    return close.with_columns(
        [
            _pl_wilder_expr(_pl_true_range_expr(high[c], low[c], close[c]), w, w).alias(c)
            for c in cols
        ]
    )


def pd_rsi_wilder(x, window=14, **_):
    w = positive_int(window, "window")
    delta = x - _pd_delay(x, 1)
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = _pd_wilder(gain, w, w)
    avg_loss = _pd_wilder(loss, w, w)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def pl_rsi_wilder(x, window=14, **_):
    w = positive_int(window, "window")
    exprs = []
    for c in pl_cols(x):
        delta = pl.col(c).diff()
        gain = (
            pl.when(delta.is_null())
            .then(None)
            .when(delta > 0)
            .then(delta)
            .otherwise(0.0)
        )
        loss = (
            pl.when(delta.is_null())
            .then(None)
            .when(delta < 0)
            .then(-delta)
            .otherwise(0.0)
        )
        avg_gain = _pl_wilder_expr(gain, w, w)
        avg_loss = _pl_wilder_expr(loss, w, w)
        ratio = avg_gain / pl.when(avg_loss.abs() > EPS).then(avg_loss).otherwise(None)
        raw = 100.0 - 100.0 / (1.0 + ratio)
        exprs.append(
            pl.when((avg_loss == 0) & (avg_gain > 0))
            .then(100.0)
            .when((avg_gain == 0) & (avg_loss > 0))
            .then(0.0)
            .when((avg_gain == 0) & (avg_loss == 0))
            .then(50.0)
            .otherwise(raw)
            .alias(c)
        )
    return x.with_columns(exprs)


def pd_adxr(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    adx = _call("ADX", "pandas_numpy", high, low, close, w)
    return (adx + _pd_delay(adx, w)) * 0.5


def pl_adxr(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    adx = _call("ADX", "polars", high, low, close, w)
    return adx.with_columns(
        [((pl.col(c) + pl.col(c).shift(w)) * 0.5).alias(c) for c in pl_cols(adx)]
    )


def pd_bollinger_mid(x, window=20, std_dev=2.0, **_):
    return _pd_mean(x, window, 1)


def pd_bollinger_lower(x, window=20, std_dev=2.0, **_):
    mean = _pd_mean(x, window, 1)
    std = _pd_std(x, window, 1)
    return mean - float(std_dev) * std


def pd_bollinger_upper(x, window=20, std_dev=2.0, **_):
    mean = _pd_mean(x, window, 1)
    std = _pd_std(x, window, 1)
    return mean + float(std_dev) * std


def _pl_bollinger(x, window, std_dev, side):
    w = positive_int(window, "window")
    sd = float(std_dev)
    exprs = []
    for c in pl_cols(x):
        mean = pl.col(c).rolling_mean(window_size=w, min_samples=1)
        if side == "mid":
            value = mean
        else:
            std = pl.col(c).rolling_std(window_size=w, min_samples=1, ddof=1)
            value = mean + sd * std if side == "upper" else mean - sd * std
        exprs.append(value.alias(c))
    return x.with_columns(exprs)


def pd_mom(price, window=10, **_):
    return price - _pd_delay(price, positive_int(window, "window"))


def pl_mom(price, window=10, **_):
    w = positive_int(window, "window")
    return price.with_columns([(pl.col(c) - pl.col(c).shift(w)).alias(c) for c in pl_cols(price)])


def pd_roc(price, window=10, **_):
    previous = _pd_delay(price, positive_int(window, "window"))
    return ((price / previous.replace(0, np.nan)) - 1.0) * 100.0


def pl_roc(price, window=10, **_):
    w = positive_int(window, "window")
    exprs = []
    for c in pl_cols(price):
        previous = pl.col(c).shift(w)
        exprs.append(
            pl.when(previous.abs() > EPS)
            .then((pl.col(c) / previous - 1.0) * 100.0)
            .otherwise(None)
            .alias(c)
        )
    return price.with_columns(exprs)


def pd_dpo(close, window=20, **_):
    w = positive_int(window, "window")
    delayed_mean = _pd_delay(_pd_mean(close, w, 1), (w // 2) + 1)
    return close - delayed_mean


def pl_dpo(close, window=20, **_):
    w = positive_int(window, "window")
    shift = (w // 2) + 1
    return close.with_columns(
        [
            (
                pl.col(c)
                - pl.col(c).rolling_mean(window_size=w, min_samples=1).shift(shift)
            ).alias(c)
            for c in pl_cols(close)
        ]
    )


def _pd_stochastic_k(high, low, close, window):
    w = positive_int(window, "window")
    lowest = _pd_min(low, w, 1)
    highest = _pd_max(high, w, 1)
    denominator = highest - lowest
    return 100.0 * (close - lowest) / denominator.replace(0, np.nan)


def pd_stochastic_k(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    return _pd_stochastic_k(high, low, close, window)


def pd_stochastic_d(high, low, close, window=14, **_):
    k = _call("StochasticK", "pandas_numpy", high, low, close, positive_int(window, "window"))
    return _pd_mean(k, 3, 1)


def pd_williams_r(high, low, close, window=14, **_):
    # Williams %R is exactly Stochastic %K - 100.
    k = _call("StochasticK", "pandas_numpy", high, low, close, positive_int(window, "window"))
    return k - 100.0


def _pl_stochastic_frame(high, low, close, window):
    w = positive_int(window, "window")
    cols = _pl_align(high, low, close)
    exprs = []
    for c in cols:
        lowest = low[c].rolling_min(window_size=w, min_samples=1)
        highest = high[c].rolling_max(window_size=w, min_samples=1)
        denominator = highest - lowest
        exprs.append(
            pl.when(denominator.abs() > EPS)
            .then(100.0 * (close[c] - lowest) / denominator)
            .otherwise(None)
            .alias(c)
        )
    return close.with_columns(exprs)


def pl_stochastic_k(high, low, close, window=14, **_):
    return _pl_stochastic_frame(high, low, close, window)


def pl_stochastic_d(high, low, close, window=14, **_):
    k = _pl_stochastic_frame(high, low, close, window)
    return k.with_columns(
        [pl.col(c).rolling_mean(window_size=3, min_samples=1).alias(c) for c in pl_cols(k)]
    )


def pl_williams_r(high, low, close, window=14, **_):
    k = _pl_stochastic_frame(high, low, close, window)
    return k.with_columns([(pl.col(c) - 100.0).alias(c) for c in pl_cols(k)])


def pd_trix(close, window=12, **_):
    w = positive_int(window, "window")
    ema3 = _pd_ema(_pd_ema(_pd_ema(close, w), w), w)
    return _call("ROC", "pandas_numpy", ema3, 1)


def pl_trix(close, window=12, **_):
    w = positive_int(window, "window")
    ema3 = _pl_ewm_frame(_pl_ewm_frame(_pl_ewm_frame(close, w), w), w)
    return pl_roc(ema3, 1)


def pd_obv(price, volume, **_):
    price, volume = aligned_pd(price, volume)
    direction = np.sign(price - _pd_delay(price, 1)).fillna(0.0)
    return (direction * volume).cumsum()


def pl_obv(price, volume, **_):
    cols = _pl_align(price, volume)
    return price.with_columns(
        [
            (pl.col(c).diff().sign().fill_null(0.0) * volume[c]).cum_sum().alias(c)
            for c in cols
        ]
    )


def _kama_numpy(values: np.ndarray, smoothing: np.ndarray) -> np.ndarray:
    out = values.astype(float, copy=True)
    if out.shape[0] == 0:
        return out
    for row in range(1, out.shape[0]):
        current = values[row]
        previous = out[row - 1]
        if not np.isfinite(current):
            # Round-2 review §technical/stateful: the SINGLE production missing
            # policy is ``break + rewarm`` — a missing price breaks the recursion
            # (NaN); the smoothing series (ER) is NaN until ``er_window``
            # consecutive finite bars re-accumulate, so ``candidate`` below stays
            # NaN through the re-warm and KAMA never emits a frozen first-price.
            out[row] = np.nan
            continue
        fallback = np.where(np.isfinite(previous), previous, current)
        candidate = fallback + smoothing[row] * (current - fallback)
        out[row] = candidate
    return out


def pd_kama(close, er_window=10, fast_window=2, slow_window=30, **_):
    # R4-100: aligned to the canonical ``(close, er_window, fast_window,
    # slow_window)`` contract (technical.indicators_v2.KAMA is the pandas
    # reference).  The old fast-path signature ``(close, window)`` accepted a
    # different simplified formula under the same canonical.
    w = positive_int(er_window, "er_window")
    fast = positive_int(fast_window, "fast_window")
    slow = positive_int(slow_window, "slow_window")
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    delayed = _pd_delay(close, w)
    direction = (close - delayed).abs()
    volatility = _call("ts_sum", "pandas_numpy", (close - _pd_delay(close, 1)).abs(), w, min_periods=w)
    efficiency = direction / volatility.replace(0, np.nan)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    smoothing = (efficiency * (fast_sc - slow_sc) + slow_sc).pow(2)
    return pd.DataFrame(
        _kama_numpy(close.to_numpy(dtype=float), smoothing.to_numpy(dtype=float)),
        index=close.index,
        columns=close.columns,
    )


def pl_kama(close, er_window=10, fast_window=2, slow_window=30, **_):
    w = positive_int(er_window, "er_window")
    fast = positive_int(fast_window, "fast_window")
    slow = positive_int(slow_window, "slow_window")
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    cols = pl_cols(close)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    smoothing = close.select(
        [
            (
                (
                    (pl.col(c) - pl.col(c).shift(w)).abs()
                    / pl.col(c).diff().abs().rolling_sum(window_size=w, min_samples=w)
                )
                * (fast_sc - slow_sc)
                + slow_sc
            )
            .pow(2)
            .alias(c)
            for c in cols
        ]
    )
    values = close.select(cols).to_numpy()
    out = _kama_numpy(values, smoothing.to_numpy())
    return close.with_columns([pl.Series(c, out[:, idx]) for idx, c in enumerate(cols)])


def pl_wma(x, window=10, **_):
    w = positive_int(window, "window")
    weights = np.arange(1.0, w + 1.0)
    weights /= weights.sum()
    return x.with_columns(
        [
            pl.col(c)
            .rolling_mean(window_size=w, weights=weights.tolist(), min_samples=1)
            .alias(c)
            for c in pl_cols(x)
        ]
    )


def _pd_vp_weighted_price(close, volume, open_=None, high=None, low=None, window=20, min_periods=5):
    close, volume = aligned_pd(close, volume)
    if high is not None and low is not None:
        _, high, low = aligned_pd(close, high, low)
        amplitude = high - low
    else:
        amplitude = (close - _pd_delay(close, 1)).abs().fillna(0.0)
    sigma = (amplitude / close.replace(0, np.nan)).fillna(0.0)
    if open_ is not None:
        _, open_ = aligned_pd(close, open_)
        relative = ((open_ - close).abs() / amplitude.replace(0, np.nan)).fillna(0.5)
    else:
        previous = _pd_delay(close, 1)
        ret = close / previous.replace(0, np.nan) - 1.0
        relative = pd.DataFrame(0.5, index=close.index, columns=close.columns)
        relative = relative.mask(ret > 0, 0.7).mask(ret < 0, 0.3)
    weights = volume * sigma * relative
    numerator = (close * weights).rolling(window, min_periods=min_periods).sum()
    denominator = weights.rolling(window, min_periods=min_periods).sum().replace(0, np.nan)
    return (numerator / denominator).fillna(close)


def _pd_macd_parts(price, fast=12, slow=26, signal=9):
    fast_i, slow_i, signal_i = (
        positive_int(fast, "fast"),
        positive_int(slow, "slow"),
        positive_int(signal, "signal"),
    )
    if fast_i >= slow_i:
        raise ValueError("fast must be smaller than slow")
    line = _pd_ema(price, fast_i) - _pd_ema(price, slow_i)
    signal_line = _pd_ema(line, signal_i)
    return line, signal_line


def pd_vp_weighted_price(close, volume, open_=None, high=None, low=None, window=20, min_periods=5, **_):
    return _pd_vp_weighted_price(close, volume, open_, high, low, window, min_periods)


def pd_vpmacd(close, volume, open_=None, high=None, low=None, lambda_param=0.9, **kwargs):
    weighted = _pd_vp_weighted_price(
        close,
        volume,
        open_,
        high,
        low,
        int(kwargs.get("window", 20)),
        int(kwargs.get("min_periods", 5)),
    )
    line, signal_line = _pd_macd_parts(weighted, 12, 26, 9)
    return line - float(kwargs.get("lambda", lambda_param)) * signal_line


def pd_vpmacd_signal(close, volume, open_=None, high=None, low=None, lambda_param=0.9, **kwargs):
    weighted = _pd_vp_weighted_price(
        close,
        volume,
        open_,
        high,
        low,
        int(kwargs.get("window", 20)),
        int(kwargs.get("min_periods", 5)),
    )
    line, signal_line = _pd_macd_parts(weighted, 12, 26, 9)
    adjusted = float(kwargs.get("lambda", lambda_param)) * signal_line
    golden = (line > adjusted) & (_pd_delay(line, 1) <= _pd_delay(adjusted, 1))
    death = (line < adjusted) & (_pd_delay(line, 1) >= _pd_delay(adjusted, 1))
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    return out.mask(golden, 1.0).mask(death, -1.0)


def _pl_vp_weighted_price(close, volume, open_=None, high=None, low=None, window=20, min_periods=5):
    w = positive_int(window, "window")
    mp = positive_int(min_periods, "min_periods")
    cols = _pl_align(close, volume)
    exprs = []
    for c in cols:
        if high is not None and low is not None and c in high.columns and c in low.columns:
            amplitude = high[c] - low[c]
        else:
            amplitude = close[c].diff().abs().fill_null(0.0)
        sigma = (amplitude / close[c]).fill_nan(0.0).fill_null(0.0)
        if open_ is not None and c in open_.columns:
            relative = ((open_[c] - close[c]).abs() / amplitude).fill_nan(0.5).fill_null(0.5)
        else:
            ret = close[c] / close[c].shift(1) - 1.0
            relative = pl.when(ret > 0).then(0.7).when(ret < 0).then(0.3).otherwise(0.5)
        weights = volume[c] * sigma * relative
        numerator = (close[c] * weights).rolling_sum(window_size=w, min_samples=mp)
        denominator = weights.rolling_sum(window_size=w, min_samples=mp)
        exprs.append(pl.coalesce([numerator / denominator, close[c]]).alias(c))
    return close.with_columns(exprs)


def _pl_macd_parts(price, fast=12, slow=26, signal=9):
    fast_i, slow_i, signal_i = (
        positive_int(fast, "fast"),
        positive_int(slow, "slow"),
        positive_int(signal, "signal"),
    )
    fast_frame = _pl_ewm_frame(price, fast_i)
    slow_frame = _pl_ewm_frame(price, slow_i)
    line = price.with_columns(
        [(fast_frame[c] - slow_frame[c]).alias(c) for c in pl_cols(price)]
    )
    return line, _pl_ewm_frame(line, signal_i)


def pl_vp_weighted_price(close, volume, open_=None, high=None, low=None, window=20, min_periods=5, **_):
    return _pl_vp_weighted_price(close, volume, open_, high, low, window, min_periods)


def pl_vpmacd(close, volume, open_=None, high=None, low=None, lambda_param=0.9, **kwargs):
    weighted = _pl_vp_weighted_price(
        close,
        volume,
        open_,
        high,
        low,
        int(kwargs.get("window", 20)),
        int(kwargs.get("min_periods", 5)),
    )
    line, signal_line = _pl_macd_parts(weighted, 12, 26, 9)
    lam = float(kwargs.get("lambda", lambda_param))
    return line.with_columns(
        [(pl.col(c) - lam * signal_line[c]).alias(c) for c in pl_cols(line)]
    )


def pl_vpmacd_signal(close, volume, open_=None, high=None, low=None, lambda_param=0.9, **kwargs):
    weighted = _pl_vp_weighted_price(
        close,
        volume,
        open_,
        high,
        low,
        int(kwargs.get("window", 20)),
        int(kwargs.get("min_periods", 5)),
    )
    line, signal_line = _pl_macd_parts(weighted, 12, 26, 9)
    lam = float(kwargs.get("lambda", lambda_param))
    exprs = []
    for c in pl_cols(line):
        adjusted = signal_line[c] * lam
        golden = (line[c] > adjusted) & (line[c].shift(1) <= adjusted.shift(1))
        death = (line[c] < adjusted) & (line[c].shift(1) >= adjusted.shift(1))
        exprs.append(pl.when(golden).then(1.0).when(death).then(-1.0).otherwise(0.0).alias(c))
    return close.with_columns(exprs)


def pd_log_returns(x, **_):
    previous = _pd_delay(x, 1)
    ratio = x / previous.replace(0, np.nan)
    return np.log(ratio.where(ratio > 0))


def pl_log_returns(x, **_):
    exprs = []
    for c in pl_cols(x):
        previous = pl.col(c).shift(1)
        ratio = pl.when(previous != 0).then((pl.col(c)) / (previous)).otherwise(None)
        exprs.append(pl.when((previous.abs() > EPS) & (ratio > 0)).then(ratio.log()).otherwise(None).alias(c))
    return x.with_columns(exprs)


def pd_open_gap(open_, close, **_):
    previous = _pd_delay(close, 1)
    return open_ / previous.replace(0, np.nan) - 1.0


def pl_open_gap(open_, close, **_):
    cols = _pl_align(open_, close)
    return open_.with_columns(
        [
            pl.when(close[c].shift(1).abs() > EPS)
            .then(open_[c] / close[c].shift(1) - 1.0)
            .otherwise(None)
            .alias(c)
            for c in cols
        ]
    )


def pd_close_gap(close, open_, **_):
    return close / open_.replace(0, np.nan) - 1.0


def pl_close_gap(close, open_, **_):
    cols = _pl_align(close, open_)
    return close.with_columns(
        [
            pl.when(open_[c].abs() > EPS)
            .then(close[c] / open_[c] - 1.0)
            .otherwise(None)
            .alias(c)
            for c in cols
        ]
    )


def pd_volatility(x, window=20, min_periods=None, **_):
    w = positive_int(window, "window")
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 2)
    return _pd_std(x, w, mp) * np.sqrt(252.0)


def pl_volatility(x, window=20, min_periods=None, **_):
    w = positive_int(window, "window")
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 2)
    scale = float(np.sqrt(252.0))
    return x.with_columns(
        [pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1).mul(scale).alias(c) for c in pl_cols(x)]
    )


def pd_sharpe(returns, window=60, **_):
    w = positive_int(window, "window")
    mean = _pd_mean(returns, w, 2)
    std = _pd_std(returns, w, 2)
    return mean / std.replace(0, np.nan) * np.sqrt(252.0)


def pl_sharpe(returns, window=60, **_):
    w = positive_int(window, "window")
    scale = float(np.sqrt(252.0))
    return returns.with_columns(
        [
            (
                pl.col(c).rolling_mean(window_size=w, min_samples=2)
                / pl.col(c).rolling_std(window_size=w, min_samples=2, ddof=1)
                * scale
            ).alias(c)
            for c in pl_cols(returns)
        ]
    )


def pd_vwap(price, volume, window=20, min_periods=1, **_):
    price, volume = aligned_pd(price, volume)
    w = positive_int(window, "window")
    mp = positive_int(min_periods, "min_periods")
    valid = price.notna() & volume.notna()
    numerator = ((price * volume).where(valid)).rolling(w, min_periods=mp).sum()
    denominator = volume.where(valid).rolling(w, min_periods=mp).sum().replace(0, np.nan)
    return numerator / denominator


def pl_vwap(price, volume, window=20, min_periods=1, **_):
    w = positive_int(window, "window")
    mp = positive_int(min_periods, "min_periods")
    cols = _pl_align(price, volume)
    exprs = []
    for c in cols:
        valid = price[c].is_not_null() & volume[c].is_not_null()
        numerator = pl.when(valid).then(price[c] * volume[c]).otherwise(None).rolling_sum(w, min_samples=mp)
        denominator = pl.when(valid).then(volume[c]).otherwise(None).rolling_sum(w, min_samples=mp)
        exprs.append(pl.when(denominator.abs() > EPS).then(numerator / denominator).otherwise(None).alias(c))
    return price.with_columns(exprs)


def register_composite_fastpaths() -> None:
    specs = (
        ("ATR", "technical_signal", ["high", "low", "close", "window"], "真实波幅后复用滚动均值", pd_atr, pl_atr),
        ("ATR_WILDER", "technical_signal", ["high", "low", "close", "window"], "共享真实波幅内核的 Wilder ATR", pd_atr_wilder, pl_atr_wilder),
        ("RSI_WILDER", "technical_signal", ["x", "window"], "共享 Wilder 平滑内核的 RSI", pd_rsi_wilder, pl_rsi_wilder),
        ("ADXR", "technical_signal", ["high", "low", "close", "window"], "复用 ADX 与时序滞后", pd_adxr, pl_adxr),
        ("BollingerBands", "technical_signal", ["x", "window", "std_dev"], "复用 ts_mean 的布林中轨", pd_bollinger_mid, lambda x, window=20, std_dev=2.0, **kw: _pl_bollinger(x, window, std_dev, "mid")),
        ("BollingerLower", "technical_signal", ["x", "window", "std_dev"], "复用 ts_mean/ts_std 的布林下轨", pd_bollinger_lower, lambda x, window=20, std_dev=2.0, **kw: _pl_bollinger(x, window, std_dev, "lower")),
        ("BollingerUpper", "technical_signal", ["x", "window", "std_dev"], "复用 ts_mean/ts_std 的布林上轨", pd_bollinger_upper, lambda x, window=20, std_dev=2.0, **kw: _pl_bollinger(x, window, std_dev, "upper")),
        ("MOM", "technical_signal", ["price", "window"], "复用 ts_delay 的动量", pd_mom, pl_mom),
        ("ROC", "technical_signal", ["price", "window"], "复用 ts_delay 的变化率", pd_roc, pl_roc),
        ("DPO", "technical_signal", ["close", "window"], "复用 ts_mean 与 ts_delay 的 DPO", pd_dpo, pl_dpo),
        ("StochasticK", "technical_signal", ["high", "low", "close", "window"], "复用滚动高低值的随机指标 K", pd_stochastic_k, pl_stochastic_k),
        ("StochasticD", "technical_signal", ["high", "low", "close", "window"], "复用 StochasticK 与 ts_mean", pd_stochastic_d, pl_stochastic_d),
        ("WilliamsR", "technical_signal", ["high", "low", "close", "window"], "复用 StochasticK 的 Williams R", pd_williams_r, pl_williams_r),
        ("TRIX", "technical_signal", ["close", "window"], "复用三层 ts_ema 与 ROC", pd_trix, pl_trix),
        ("OBV", "technical_signal", ["price", "volume"], "复用滞后方向并累计成交量", pd_obv, pl_obv),
        ("KAMA", "technical_signal", ["close", "er_window", "fast_window", "slow_window"], "共享效率比并按行向量化递归", pd_kama, pl_kama),
        ("vp_weighted_price", "technical_signal", ["close", "volume", "open", "high", "low", "window"], "VP-MACD 共享量价加权价格内核", pd_vp_weighted_price, pl_vp_weighted_price),
        ("vpmacd", "technical_signal", ["close", "volume", "open", "high", "low", "lambda_param"], "一次计算 weighted price、MACD line 和 signal", pd_vpmacd, pl_vpmacd),
        ("vpmacd_signal", "technical_signal", ["close", "volume", "open", "high", "low", "lambda_param"], "复用同一次 VP-MACD 中间量的离散交叉信号", pd_vpmacd_signal, pl_vpmacd_signal),
        ("log_returns", "price_volume", ["x"], "复用时序滞后的对数收益", pd_log_returns, pl_log_returns),
        ("open_gap", "price_volume", ["open", "close"], "复用时序滞后的隔夜缺口", pd_open_gap, pl_open_gap),
        ("close_gap", "price_volume", ["close", "open"], "安全除法的日内缺口", pd_close_gap, pl_close_gap),
        ("volatility", "price_volume", ["x", "window", "min_periods"], "复用 ts_std 的年化波动率", pd_volatility, pl_volatility),
        ("sharpe_ratio", "price_volume", ["returns", "window"], "复用 ts_mean/ts_std 的年化夏普", pd_sharpe, pl_sharpe),
        ("vwap", "price_volume", ["price", "volume", "window", "min_periods"], "共享有效样本掩码的滚动 VWAP", pd_vwap, pl_vwap),
    )
    for spec in specs:
        _register(*spec)
    # Pandas WMA already uses the optimized rolling kernel; only replace the
    # Polars Python rolling_map implementation with the native weighted kernel.
    _register("WMA", "time_series", ["x", "window"], "Polars 原生加权 rolling_mean", None, pl_wma)


register_composite_fastpaths()
