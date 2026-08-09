# -*- coding: utf-8
"""技术指标 Polars 实现（EMA / WMA / RSI / MACD 簇）。"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _ewm_mean(x: pl.DataFrame, span: int) -> pl.DataFrame:
    cols = _numeric_cols(x)
    alpha = 2.0 / (float(span) + 1.0)
    return x.with_columns([
        pl.col(c).ewm_mean(alpha=alpha, adjust=False).alias(c) for c in cols
    ])


def _wma(x: pl.DataFrame, window: int) -> pl.DataFrame:
    cols = _numeric_cols(x)
    w = np.arange(1, int(window) + 1, dtype=float)
    w /= w.sum()

    def _apply(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        weights = w[-len(arr):]
        return float(np.dot(arr, weights) / weights.sum())

    return x.with_columns([
        pl.col(c).rolling_map(_apply, window_size=int(window), min_samples=1).alias(c)
        for c in cols
    ])


@register_operator(name="EMA", category="time_series", business_category="time_series", canonical="ts_ema", source="factor_dsl_polars")
class EMAPolars(SeriesOperator):
    """Polars 指数移动平均"""
    metadata = OperatorMetadata(
        name="EMA", category="time_series", description="指数移动平均",
        param_names=["x", "span"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 12, **kwargs) -> pl.DataFrame:
        window = int(kwargs.get("window", kwargs.get("d", span)))
        return _ewm_mean(x, window)


@register_operator(name="WMA", category="time_series", business_category="time_series", canonical="WMA", source="factor_dsl_polars")
class WMAPolars(SeriesOperator):
    """Polars 加权移动平均"""
    metadata = OperatorMetadata(
        name="WMA", category="time_series", description="加权移动平均",
        param_names=["x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _wma(x, w)


@register_operator(name="RSI", category="financial", business_category="technical_signal", canonical="RSI", source="factor_dsl_polars")
class RSIPolars(SeriesOperator):
    """Polars 相对强弱指数"""
    metadata = OperatorMetadata(
        name="RSI", category="financial", description="相对强弱指数",
        param_names=["x", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 14, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            delta = pl.col(c).diff()
            gain = pl.when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
            avg_gain = gain.rolling_mean(window_size=w, min_periods=1)
            avg_loss = loss.rolling_mean(window_size=w, min_periods=1)
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            exprs.append(rsi.alias(c))
        return x.with_columns(exprs)


@register_operator(
    name="RSI_WILDER",
    category="financial",
    business_category="technical_signal",
    canonical="RSI_WILDER",
    source="factor_dsl_polars",
    backend="polars",
)
class RSIWilderPolars(SeriesOperator):
    """Polars Wilder 平滑 RSI"""
    metadata = OperatorMetadata(
        name="RSI_WILDER",
        category="financial",
        description="Wilder 平滑 RSI",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "wilder"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 14, **kwargs) -> pl.DataFrame:
        from cleaned_operators.technical.signal import _compute_rsi_wilder

        w = max(2, int(kwargs.get("d", window)))
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        out = _compute_rsi_wilder(pdf, w)
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="MACD", category="financial", business_category="technical_signal", canonical="MACD", source="factor_dsl_polars")
class MACDPolars(SeriesOperator):
    """Polars MACD 线"""
    metadata = OperatorMetadata(
        name="MACD", category="financial", description="MACD 线",
        param_names=["x", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        fast_e = _ewm_mean(x, int(fast))
        slow_e = _ewm_mean(x, int(slow))
        cols = _numeric_cols(x)
        return fast_e.with_columns([
            (fast_e[c] - slow_e[c]).alias(c) for c in cols
        ])


@register_operator(name="MACD_line", category="financial", business_category="technical_signal", canonical="MACD_line", source="factor_dsl_polars")
class MACDLinePolars(MACDPolars):
    """Polars MACD 线"""
    metadata = OperatorMetadata(
        name="MACD_line", category="financial", description="MACD 线",
        param_names=["price", "fast", "slow"], return_type="series", tags=["financial", "polars"],
    )


@register_operator(name="MACD_signal", category="financial", business_category="technical_signal", canonical="MACD_signal", source="factor_dsl_polars")
class MACDSignalPolars(SeriesOperator):
    """Polars MACD 信号线"""
    metadata = OperatorMetadata(
        name="MACD_signal", category="financial", description="MACD 信号线",
        param_names=["price", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        line = MACDPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        cols = _numeric_cols(line)
        return _ewm_mean(line, int(signal))


@register_operator(name="MACD_hist", category="financial", business_category="technical_signal", canonical="MACD_hist", source="factor_dsl_polars")
class MACDHistPolars(SeriesOperator):
    """Polars MACD 柱"""
    metadata = OperatorMetadata(
        name="MACD_hist", category="financial", description="MACD 柱",
        param_names=["price", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        line = MACDPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        sig = MACDSignalPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        cols = _numeric_cols(line)
        return line.with_columns([(line[c] - sig[c]).alias(c) for c in cols])


@register_operator(name="ATR", category="financial", business_category="technical_signal", canonical="ATR", source="factor_dsl_polars")
class ATRPolars(SeriesOperator):
    """Polars 平均真实波幅"""
    metadata = OperatorMetadata(
        name="ATR", category="financial", description="平均真实波幅",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        window: int = 14,
        **kwargs,
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = [c for c in _numeric_cols(close) if c in high.columns and c in low.columns]
        exprs = []
        for c in cols:
            prev_close = close[c].shift(1)
            tr = pl.max_horizontal(
                high[c] - low[c],
                (high[c] - prev_close).abs(),
                (low[c] - prev_close).abs(),
            )
            exprs.append(tr.rolling_mean(window_size=w, min_samples=1).alias(c))
        return close.with_columns(exprs)


@register_operator(
    name="ATR_WILDER",
    category="financial",
    business_category="technical_signal",
    canonical="ATR_WILDER",
    source="factor_dsl_polars",
    backend="polars",
)
class ATRWilderPolars(SeriesOperator):
    """Polars Wilder 平滑 ATR"""
    metadata = OperatorMetadata(
        name="ATR_WILDER",
        category="financial",
        description="Wilder 平滑 ATR",
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "polars", "wilder"],
    )

    def _calculate_series(
        self,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        window: int = 14,
        **kwargs,
    ) -> pl.DataFrame:
        from cleaned_operators.technical.signal import _compute_atr_wilder

        w = max(2, int(kwargs.get("d", window)))
        cols = [c for c in _numeric_cols(close) if c in high.columns and c in low.columns]
        h = high.select(cols).to_pandas()
        lo = low.select(cols).to_pandas()
        cl = close.select(cols).to_pandas()
        out = _compute_atr_wilder(h, lo, cl, w)
        return close.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


def _rolling_mean(x: pl.DataFrame, window: int) -> pl.DataFrame:
    w = max(int(window), 1)
    cols = _numeric_cols(x)
    return x.with_columns([
        pl.col(c).rolling_mean(window_size=w, min_samples=1).alias(c) for c in cols
    ])


def _rolling_std(x: pl.DataFrame, window: int) -> pl.DataFrame:
    w = max(int(window), 1)
    cols = _numeric_cols(x)
    return x.with_columns([
        pl.col(c).rolling_std(window_size=w, min_samples=1).alias(c) for c in cols
    ])


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    base = _numeric_cols(dfs[0])
    out = base
    for df in dfs[1:]:
        out = [c for c in out if c in df.columns]
    return out


@register_operator(name="BollingerBands", category="financial", business_category="technical_signal", canonical="BollingerBands", source="factor_dsl_polars")
class BollingerBandsPolars(SeriesOperator):
    """Polars 布林带中轨"""
    metadata = OperatorMetadata(
        name="BollingerBands", category="financial", description="布林带中轨",
        param_names=["price", "window", "std_dev"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _rolling_mean(price, w)


@register_operator(name="BollingerLower", category="financial", business_category="technical_signal", canonical="BollingerLower", source="factor_dsl_polars")
class BollingerLowerPolars(SeriesOperator):
    """Polars 布林带下轨"""
    metadata = OperatorMetadata(
        name="BollingerLower", category="financial", description="布林带下轨",
        param_names=["x", "window", "std_dev"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        sd = float(kwargs.get("std_dev", std_dev))
        mean = _rolling_mean(x, w)
        std = _rolling_std(x, w)
        cols = _numeric_cols(x)
        return mean.with_columns([(mean[c] - sd * std[c]).alias(c) for c in cols])


@register_operator(name="BollingerUpper", category="financial", business_category="technical_signal", canonical="BollingerUpper", source="factor_dsl_polars")
class BollingerUpperPolars(SeriesOperator):
    """Polars 布林带上轨"""
    metadata = OperatorMetadata(
        name="BollingerUpper", category="financial", description="布林带上轨",
        param_names=["x", "window", "std_dev"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        sd = float(kwargs.get("std_dev", std_dev))
        mean = _rolling_mean(x, w)
        std = _rolling_std(x, w)
        cols = _numeric_cols(x)
        return mean.with_columns([(mean[c] + sd * std[c]).alias(c) for c in cols])


@register_operator(name="StochasticK", category="financial", business_category="technical_signal", canonical="StochasticK", source="factor_dsl_polars")
class StochasticKPolars(SeriesOperator):
    """Polars 随机指标 %K"""
    metadata = OperatorMetadata(
        name="StochasticK", category="financial", description="随机指标 %K",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(high, low, close)
        exprs = []
        for c in cols:
            lo = low[c].rolling_min(window_size=w, min_samples=1)
            hi = high[c].rolling_max(window_size=w, min_samples=1)
            exprs.append((100.0 * (close[c] - lo) / (hi - lo)).alias(c))
        return close.with_columns(exprs)


@register_operator(name="StochasticD", category="financial", business_category="technical_signal", canonical="StochasticD", source="factor_dsl_polars")
class StochasticDPolars(SeriesOperator):
    """Polars 随机指标 %D"""
    metadata = OperatorMetadata(
        name="StochasticD", category="financial", description="随机指标 %D",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        k = StochasticKPolars()._calculate_series(high, low, close, window=window, **kwargs)
        cols = _numeric_cols(k)
        return k.with_columns([
            pl.col(c).rolling_mean(window_size=3, min_samples=1).alias(c) for c in cols
        ])


@register_operator(name="MOM", category="financial", business_category="technical_signal", canonical="MOM", source="factor_dsl_polars")
class MOMPolars(SeriesOperator):
    """Polars 动量"""
    metadata = OperatorMetadata(
        name="MOM", category="financial", description="动量",
        param_names=["price", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(price)
        return price.with_columns([(pl.col(c) - pl.col(c).shift(w)).alias(c) for c in cols])


@register_operator(name="ROC", category="financial", business_category="technical_signal", canonical="ROC", source="factor_dsl_polars")
class ROCPolars(SeriesOperator):
    """Polars 变化率"""
    metadata = OperatorMetadata(
        name="ROC", category="financial", description="变化率",
        param_names=["price", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(price)
        return price.with_columns([
            ((pl.col(c) / pl.col(c).shift(w) - 1.0) * 100.0).alias(c) for c in cols
        ])


@register_operator(name="WilliamsR", category="financial", business_category="technical_signal", canonical="WilliamsR", source="factor_dsl_polars")
class WilliamsRPolars(SeriesOperator):
    """Polars 威廉 %R"""
    metadata = OperatorMetadata(
        name="WilliamsR", category="financial", description="威廉 %R",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(high, low, close)
        exprs = []
        for c in cols:
            lo = low[c].rolling_min(window_size=w, min_samples=1)
            hi = high[c].rolling_max(window_size=w, min_samples=1)
            exprs.append((-100.0 * (hi - close[c]) / (hi - lo)).alias(c))
        return close.with_columns(exprs)


@register_operator(name="TRIX", category="financial", business_category="technical_signal", canonical="TRIX", source="factor_dsl_polars")
class TRIXPolars(SeriesOperator):
    """Polars 三重 EMA 变化率"""
    metadata = OperatorMetadata(
        name="TRIX", category="financial", description="三重 EMA 变化率",
        param_names=["close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 12, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        ema3 = _ewm_mean(_ewm_mean(_ewm_mean(close, w), w), w)
        cols = _numeric_cols(ema3)
        return ema3.with_columns([
            ((pl.col(c) / pl.col(c).shift(1) - 1.0) * 100.0).alias(c) for c in cols
        ])


@register_operator(name="OBV", category="financial", business_category="technical_signal", canonical="OBV", source="factor_dsl_polars")
class OBVPolars(SeriesOperator):
    """Polars 能量潮"""
    metadata = OperatorMetadata(
        name="OBV", category="financial", description="能量潮",
        param_names=["price", "volume"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _align_cols(price, volume)
        exprs = []
        for c in cols:
            direction = price[c].diff().sign()
            exprs.append((direction * volume[c]).cum_sum().alias(c))
        return price.with_columns(exprs)


@register_operator(name="CCI", category="financial", business_category="technical_signal", canonical="CCI", source="factor_dsl_polars")
class CCIPolars(SeriesOperator):
    """Polars 商品通道指数"""
    metadata = OperatorMetadata(
        name="CCI", category="financial", description="商品通道指数",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(high, low, close)

        def _mad(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            m = np.nanmean(arr)
            return float(np.nanmean(np.abs(arr - m)))

        exprs = []
        for c in cols:
            tp = (high[c] + low[c] + close[c]) / 3.0
            sma = tp.rolling_mean(window_size=w, min_samples=1)
            mad = tp.rolling_map(_mad, window_size=w, min_samples=1)
            exprs.append(((tp - sma) / (0.015 * mad)).alias(c))
        return close.with_columns(exprs)


@register_operator(name="DPO", category="financial", business_category="technical_signal", canonical="DPO", source="factor_dsl_polars")
class DPOPolars(SeriesOperator):
    """Polars 去趋势价格振荡器"""
    metadata = OperatorMetadata(
        name="DPO", category="financial", description="去趋势价格振荡器",
        param_names=["close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        shift_period = (w // 2) + 1
        mean = _rolling_mean(close, w)
        cols = _numeric_cols(close)
        return close.with_columns([(close[c] - mean[c].shift(shift_period)).alias(c) for c in cols])


def _compute_dmi_adx_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """NumPy Wilder DMI→ADX（与 ``signal._compute_dmi_adx`` 对齐）。"""
    w = max(int(window), 1)
    alpha = 1.0 / w
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
            continue
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        p = high[i] - high[i - 1]
        m = low[i - 1] - low[i]
        if p > m and p > 0:
            plus_dm[i] = p
        if m > p and m > 0:
            minus_dm[i] = m

    def _wilder(arr: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return out
        out[0] = arr[0]
        for i in range(1, n):
            prev = out[i - 1] if np.isfinite(out[i - 1]) else arr[i]
            out[i] = prev + alpha * (arr[i] - prev)
        return out

    atr = _wilder(tr)
    plus_di = np.where(atr != 0, 100.0 * _wilder(plus_dm) / atr, np.nan)
    minus_di = np.where(atr != 0, 100.0 * _wilder(minus_dm) / atr, np.nan)
    denom = plus_di + minus_di
    dx = np.where(denom != 0, 100.0 * np.abs(plus_di - minus_di) / denom, np.nan)
    return _wilder(dx)


def _wilder_ewm(col: pl.Expr, window: int) -> pl.Expr:
    alpha = 1.0 / max(int(window), 1)
    return col.ewm_mean(alpha=alpha, adjust=False)


def _dmi_adx_exprs(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    w = max(int(window), 1)
    prev_close = close.shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    plus_dm = pl.when((plus_dm > minus_dm) & (plus_dm > 0)).then(plus_dm).otherwise(0.0)
    minus_dm = pl.when((minus_dm > plus_dm) & (minus_dm > 0)).then(minus_dm).otherwise(0.0)
    atr = _wilder_ewm(tr, w)
    plus_di = 100.0 * (_wilder_ewm(plus_dm, w) / atr)
    minus_di = 100.0 * (_wilder_ewm(minus_dm, w) / atr)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return _wilder_ewm(dx, w)


def _kama_1d(values: np.ndarray, sc: np.ndarray, er_window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    last = np.nan
    contiguous = 0
    for t in range(n):
        if not np.isfinite(values[t]):
            # break + rewarm (round-11 P0): a NaN price INVALIDATES the state;
            # KAMA emits NaN until ``er_window`` consecutive finite prices
            # re-accumulate and the ER is re-derived over that contiguous
            # history — no frozen flat line during the re-warmup.
            last = np.nan
            contiguous = 0
            continue
        contiguous += 1
        if contiguous < er_window:
            continue
        if not np.isfinite(last):
            last = values[t]
        elif np.isfinite(sc[t]):
            last = last + sc[t] * (values[t] - last)
        out[t] = last
    return out


@register_operator(name="ADX", category="financial", business_category="technical_signal", canonical="ADX", source="factor_dsl_polars")
class ADXPolars(SeriesOperator):
    """Polars 平均趋向指数"""
    metadata = OperatorMetadata(
        name="ADX", category="financial", description="平均趋向指数",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _align_cols(high, low, close)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _compute_dmi_adx_np(
                high[c].to_numpy(),
                low[c].to_numpy(),
                close[c].to_numpy(),
                w,
            )
        result = pl.DataFrame(out)
        if "date" in close.columns:
            result = result.with_columns(close["date"])
        return result


@register_operator(name="ADXR", category="financial", business_category="technical_signal", canonical="ADXR", source="factor_dsl_polars")
class ADXRPolars(SeriesOperator):
    """Polars 平滑 ADX"""
    metadata = OperatorMetadata(
        name="ADXR", category="financial", description="平滑 ADX",
        param_names=["high", "low", "close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int = 14, **kwargs
    ) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        adx = ADXPolars()._calculate_series(high, low, close, window=w, **kwargs)
        cols = _numeric_cols(adx)
        return adx.with_columns([
            ((pl.col(c) + pl.col(c).shift(w)) / 2.0).alias(c) for c in cols
        ])


def _aroon_component(close: pl.Expr, window: int, *, up: bool) -> pl.Expr:
    w = max(int(window), 1)

    def _pos(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        # Audit P1-I: periods since the MOST RECENT extremum — a tie must use
        # the last occurrence, not the first that np.argmax/argmin returns.
        return float(w - (np.argmax(arr[::-1]) if up else np.argmin(arr[::-1])))

    pos = close.rolling_map(_pos, window_size=w + 1, min_samples=w + 1)
    return 100.0 * pos / float(w)


@register_operator(name="AROON_up", category="financial", business_category="technical_signal", canonical="AROON_up", source="factor_dsl_polars")
class AroonUpPolars(SeriesOperator):
    """Polars Aroon Up"""
    metadata = OperatorMetadata(
        name="AROON_up", category="financial", description="Aroon Up",
        param_names=["close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 25, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(close)
        return close.with_columns([_aroon_component(pl.col(c), w, up=True).alias(c) for c in cols])


@register_operator(name="AROON_down", category="financial", business_category="technical_signal", canonical="AROON_down", source="factor_dsl_polars")
class AroonDownPolars(SeriesOperator):
    """Polars Aroon Down"""
    metadata = OperatorMetadata(
        name="AROON_down", category="financial", description="Aroon Down",
        param_names=["close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 25, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(close)
        return close.with_columns([_aroon_component(pl.col(c), w, up=False).alias(c) for c in cols])


@register_operator(name="AROON", category="financial", business_category="technical_signal", canonical="AROON", source="factor_dsl_polars")
class AroonPolars(SeriesOperator):
    """Polars Aroon Up - Down"""
    metadata = OperatorMetadata(
        name="AROON", category="financial", description="Aroon Up - Down",
        param_names=["close", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 25, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        up = AroonUpPolars()._calculate_series(close, window=w, **kwargs)
        down = AroonDownPolars()._calculate_series(close, window=w, **kwargs)
        cols = _numeric_cols(close)
        return close.with_columns([(up[c] - down[c]).alias(c) for c in cols])


@register_operator(name="KAMA", category="financial", business_category="technical_signal", canonical="KAMA", source="factor_dsl_polars")
class KAMAPolars(SeriesOperator):
    """Polars 考夫曼自适应移动平均.

    R4-100: aligned to the canonical ``(close, er_window, fast_window, slow_window)``
    contract (pandas reference: technical.indicators_v2.KAMA).  The previous
    ``(close, window)`` signature silently accepted a DIFFERENT simplified
    formula under the same canonical — positional drift across backends.
    """
    metadata = OperatorMetadata(
        name="KAMA", category="financial", description="考夫曼自适应移动平均",
        param_names=["close", "er_window", "fast_window", "slow_window"],
        return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        er_window: int = 10,
        fast_window: int = 2,
        slow_window: int = 30,
        **kwargs,
    ) -> pl.DataFrame:
        er = max(int(kwargs.get("d", er_window)), 2)  # pandas reference _pi(..., 2)
        fast = max(int(kwargs.get("p", fast_window)), 1)
        slow = max(int(kwargs.get("q", slow_window)), 1)
        if fast >= slow:
            raise ValueError("fast_window must be < slow_window")
        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        cols = _numeric_cols(close)
        out_data: dict[str, np.ndarray] = {}
        for c in cols:
            col = pl.col(c)
            direction = (col - col.shift(er)).abs()
            volatility = col.diff().abs().rolling_sum(window_size=er, min_samples=er)
            efficiency = direction / volatility
            sc = (efficiency * (fast_sc - slow_sc) + slow_sc).pow(2)
            sc_arr = close.select(sc.alias("_sc")).to_series().to_numpy()
            vals = close[c].to_numpy()
            out_data[c] = _kama_1d(vals, sc_arr, er)
        result = pl.DataFrame(out_data)
        if "date" in close.columns:
            result = result.with_columns(close["date"])
        return result


@register_operator(name="is_finite", category="signal", business_category="technical_signal", canonical="is_finite", source="factor_dsl_polars")
class IsFinitePolars(SeriesOperator):
    """Polars 是否有限"""
    metadata = OperatorMetadata(
        name="is_finite", category="signal", description="是否有限",
        param_names=["x"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from backend.elementwise_semantics import is_finite_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_finite_polars_expr(c).alias(c) for c in cols])


@register_operator(name="saturate", category="signal", business_category="technical_signal", canonical="saturate", source="factor_dsl_polars")
class SaturatePolars(SeriesOperator):
    """Polars 限制到 [0,1]"""
    metadata = OperatorMetadata(
        name="saturate", category="signal", description="限制到 [0,1]",
        param_names=["x"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).clip(0.0, 1.0).alias(c) for c in cols])


@register_operator(name="signed_log", category="signal", business_category="technical_signal", canonical="signed_log", source="factor_dsl_polars")
class SignedLogPolars(SeriesOperator):
    """Polars 符号对数"""
    metadata = OperatorMetadata(
        name="signed_log", category="signal", description="符号对数",
        param_names=["x"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c).sign() * (pl.col(c).abs() + 1e-10).log()).alias(c) for c in cols
        ])


@register_operator(name="signed_power", category="signal", business_category="technical_signal", canonical="signed_power", source="factor_dsl_polars")
class SignedPowerPolars(SeriesOperator):
    """Polars 符号幂"""
    metadata = OperatorMetadata(
        name="signed_power", category="signal", description="符号幂",
        param_names=["x", "c"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, c: float = 2.0, **kwargs) -> pl.DataFrame:
        exp = float(kwargs.get("power", c))
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(col).sign() * pl.col(col).abs().pow(exp)).alias(col) for col in cols
        ])


@register_operator(name="trade_when", category="signal", business_category="technical_signal", canonical="trade_when", source="factor_dsl_polars")
class TradeWhenPolars(SeriesOperator):
    """Polars 条件信号"""
    metadata = OperatorMetadata(
        name="trade_when", category="signal", description="条件信号",
        param_names=["condition", "signal", "fallback"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        condition: pl.DataFrame,
        signal: pl.DataFrame,
        fallback: pl.DataFrame | float = 0,
        **kwargs,
    ) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(condition) if c in signal.columns]
        if isinstance(fallback, pl.DataFrame):
            return condition.select([
                pl.when(pl.col(c).cast(pl.Boolean, strict=False))
                .then(signal[c])
                .otherwise(fallback[c])
                .alias(c)
                for c in cols
                if c in fallback.columns
            ] + ([condition["date"]] if "date" in condition.columns else []))
        fb = float(fallback)
        return condition.select([
            pl.when(pl.col(c).cast(pl.Boolean, strict=False))
            .then(signal[c])
            .otherwise(fb)
            .alias(c)
            for c in cols
        ] + ([condition["date"]] if "date" in condition.columns else []))


def _hump_decay_1d(values: np.ndarray, hump: float) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if len(out) == 0:
        return out
    prev = out[0]
    for i in range(1, len(out)):
        curr = out[i]
        if np.isnan(curr):
            out[i] = prev
        elif np.isnan(prev):
            prev = curr
        elif abs(curr - prev) > hump:
            prev = curr
        else:
            out[i] = prev
    return out


def _vp_weighted_price(
    close: pl.DataFrame,
    volume: pl.DataFrame,
    *,
    open_: pl.DataFrame | None = None,
    high: pl.DataFrame | None = None,
    low: pl.DataFrame | None = None,
    window: int = 20,
) -> pl.DataFrame:
    cols = _align_cols(close, volume)
    exprs = []
    for c in cols:
        if high is not None and low is not None and c in high.columns and c in low.columns:
            amplitude = high[c] - low[c]
        else:
            amplitude = close[c].diff().abs().fill_null(0.0)
        sigma_i = (amplitude / close[c]).fill_nan(0.0).fill_null(0.0)
        if open_ is not None and c in open_.columns:
            ri = ((open_[c] - close[c]).abs() / amplitude.replace(0, None)).fill_nan(0.5).fill_null(0.5)
        else:
            ret = close[c].pct_change().fill_null(0.0)
            ri = pl.when(ret > 0).then(0.7).when(ret < 0).then(0.3).otherwise(0.5)
        weights = volume[c] * sigma_i * ri
        num = (close[c] * weights).rolling_sum(window_size=window, min_samples=1)
        den = weights.rolling_sum(window_size=window, min_samples=1)
        exprs.append(pl.coalesce([num / den, close[c]]).alias(c))
    return close.with_columns(exprs)


@register_operator(name="hump_decay", category="signal", business_category="technical_signal", canonical="hump_decay", source="factor_dsl_polars")
class HumpDecayPolars(SeriesOperator):
    """Polars 阈值衰减"""
    metadata = OperatorMetadata(
        name="hump_decay", category="signal", description="阈值衰减",
        param_names=["x", "hump"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, hump: float = 0.05, **kwargs) -> pl.DataFrame:
        h = float(kwargs.get("threshold", hump))
        cols = _numeric_cols(x)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = _hump_decay_1d(x[c].to_numpy(), h)
        result = pl.DataFrame(out)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="vp_weighted_price", category="signal", business_category="technical_signal", canonical="vp_weighted_price", source="factor_dsl_polars")
class VPWeightedPricePolars(SeriesOperator):
    """Polars 量价加权价格"""
    metadata = OperatorMetadata(
        name="vp_weighted_price", category="signal", description="量价加权价格",
        param_names=["close", "volume", "open", "high", "low"], return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        volume: pl.DataFrame,
        open_: pl.DataFrame | None = None,
        high: pl.DataFrame | None = None,
        low: pl.DataFrame | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        w = int(kwargs.get("window", 20))
        return _vp_weighted_price(close, volume, open_=open_, high=high, low=low, window=w)


@register_operator(name="vpmacd", category="signal", business_category="technical_signal", canonical="vpmacd", source="factor_dsl_polars")
class VPMACDPolars(SeriesOperator):
    """Polars VP-MACD"""
    metadata = OperatorMetadata(
        name="vpmacd", category="signal", description="VP-MACD",
        param_names=["close", "volume", "open", "high", "low", "lambda_param"],
        return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        volume: pl.DataFrame,
        open_: pl.DataFrame | None = None,
        high: pl.DataFrame | None = None,
        low: pl.DataFrame | None = None,
        lambda_param: float = 0.9,
        **kwargs,
    ) -> pl.DataFrame:
        lam = float(kwargs.get("lambda", lambda_param))
        wp = _vp_weighted_price(close, volume, open_=open_, high=high, low=low, window=20)
        line = MACDPolars()._calculate_series(wp, fast=12, slow=26, signal=9)
        sig = MACDSignalPolars()._calculate_series(wp, fast=12, slow=26, signal=9)
        cols = _numeric_cols(line)
        return line.with_columns([(line[c] - lam * sig[c]).alias(c) for c in cols])


@register_operator(name="vpmacd_signal", category="signal", business_category="technical_signal", canonical="vpmacd_signal", source="factor_dsl_polars")
class VPMACDSignalPolars(SeriesOperator):
    """Polars VP-MACD 信号线"""
    metadata = OperatorMetadata(
        name="vpmacd_signal", category="signal", description="VP-MACD 信号线",
        param_names=["close", "volume", "open", "high", "low", "lambda_param"],
        return_type="series", tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        volume: pl.DataFrame,
        open_: pl.DataFrame | None = None,
        high: pl.DataFrame | None = None,
        low: pl.DataFrame | None = None,
        lambda_param: float = 0.9,
        **kwargs,
    ) -> pl.DataFrame:
        lam = float(kwargs.get("lambda", lambda_param))
        wp = _vp_weighted_price(close, volume, open_=open_, high=high, low=low, window=20)
        sig = MACDSignalPolars()._calculate_series(wp, fast=12, slow=26, signal=9)
        cols = _numeric_cols(sig)
        return sig.with_columns([(lam * sig[c]).alias(c) for c in cols])
