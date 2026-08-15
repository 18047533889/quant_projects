# -*- coding: utf-8 -*-
"""LQTP 平台「算子实现参考」补充包（不是因子库）。

本文件每一段 ``def`` 都是一个 **可注册 DSL 算子** 的参考实现：
输入/输出为单票 ``pandas.Series``（DatetimeIndex），因果 / PIT-safe，除零→NaN。

对接方请：
1. 按函数名注册到算子表（``max_``/``min_``/``pow_`` 注册为 ``max``/``min``/``pow``）
2. 用本文件语义做 C++/Rust/Numba 移植；浮点允许 1e-10 级误差
3. **不要**把本文件当成因子提交；这里没有因子公式

对照 ``lqtp_dsl_compat._LQTP_KNOWN_CALLS``：``NEW_OPERATORS`` 与平台已知名无交集；
``ALIAS_OPERATORS`` 仅是同名兼容（平台已有 ``safe_div``/``cap``/``delay``）。
含 ``tanh``（不要用 sigmoid 偷换）。
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

Number = Union[int, float]
ArrayLike = Union[pd.Series, float, int]


# =============================================================================
# internal helpers (not DSL surface)
# =============================================================================

def _as_f64(x: ArrayLike) -> pd.Series:
    if np.isscalar(x):
        return pd.Series([float(x)], dtype="float64")
    s = pd.Series(x, dtype="float64").copy()
    return s.replace([np.inf, -np.inf], np.nan)


def _align2(a: ArrayLike, b: ArrayLike) -> tuple[pd.Series, pd.Series]:
    a = _as_f64(a)
    b = _as_f64(b)
    if len(a) == 1 and len(b) > 1:
        a = pd.Series(float(a.iloc[0]), index=b.index, dtype="float64")
    elif len(b) == 1 and len(a) > 1:
        b = pd.Series(float(b.iloc[0]), index=a.index, dtype="float64")
    else:
        b = b.reindex(a.index)
    return a, b


def _safe_div(a: ArrayLike, b: ArrayLike) -> pd.Series:
    a, b = _align2(a, b)
    out = a / b.where(b != 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def _wilder_smooth(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    w = max(2, int(window))
    mp = w if min_periods is None else int(min_periods)
    return series.ewm(alpha=1.0 / w, adjust=False, min_periods=mp).mean()


# =============================================================================
# NEW — 技术指标（HTML / catalog 交不上的主因）
# =============================================================================

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR = max(H-L, |H-Cprev|, |L-Cprev|). catalog / ATR 依赖。"""
    high, low, close = map(_as_f64, (high, low, close))
    prev = close.shift(1)
    tr = np.maximum(np.maximum(high - low, (high - prev).abs()), (low - prev).abs())
    return pd.Series(tr, index=close.index, dtype="float64", name="true_range")


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均真实波幅（SMA 平滑）。FE_ONLY；HTML 默认 ATR。"""
    w = max(1, int(window))
    out = true_range(high, low, close).rolling(window=w, min_periods=1).mean()
    out.name = "ATR"
    return out


def ATR_WILDER(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder ATR。catalog 未知调用命中最高（~35）。推荐生产口径。"""
    w = max(2, int(window))
    out = _wilder_smooth(true_range(high, low, close), w, min_periods=w)
    out.name = "ATR_WILDER"
    return out


def RSI(x: pd.Series, window: int = 14) -> pd.Series:
    """RSI（SMA 平滑）。FE_ONLY。"""
    x = _as_f64(x)
    w = max(1, int(window))
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(window=w, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=w, min_periods=1).mean()
    rs = _safe_div(gain, loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    rsi.name = "RSI"
    return rsi


def RSI_WILDER(x: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI。"""
    x = _as_f64(x)
    w = max(2, int(window))
    delta = x.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_smooth(gain, w, min_periods=w)
    avg_loss = _wilder_smooth(loss, w, min_periods=w)
    rs = _safe_div(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi.name = "RSI_WILDER"
    return rsi


def ROC(x: pd.Series, window: int = 10) -> pd.Series:
    """变动率 %：100 * (x / delay(x,n) - 1)。FE_ONLY。"""
    x = _as_f64(x)
    w = max(1, int(window))
    prev = x.shift(w)
    out = 100.0 * (_safe_div(x, prev) - 1.0)
    out = out.where(prev.notna() & (prev != 0))
    out.name = "ROC"
    return out


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均趋向指数（Wilder DMI/ADX）。FE_ONLY。"""
    high, low, close = map(_as_f64, (high, low, close))
    w = max(2, int(window))
    tr = true_range(high, low, close)
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = _wilder_smooth(tr, w)
    plus_di = 100.0 * _safe_div(_wilder_smooth(plus_dm, w), atr)
    minus_di = 100.0 * _safe_div(_wilder_smooth(minus_dm, w), atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    out = _wilder_smooth(dx, w)
    out.name = "ADX"
    return out


def SMA(x: pd.Series, window: int) -> pd.Series:
    """简单均线。FE_ONLY 名；语义 ≈ ts_mean / ma。建议仍注册同名以免拒单。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x.rolling(window=w, min_periods=1).mean()
    out.name = "SMA"
    return out


# =============================================================================
# NEW — 数学 / 元素二元（手册写明平台拒 scalar min/max/minimum/maximum）
# =============================================================================

def tanh(x: pd.Series) -> pd.Series:
    """双曲正切。is_lqtp_native 黑名单；HTML 命中高。不可用 sigmoid 偷换语义。"""
    x = _as_f64(x)
    out = np.tanh(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="tanh").where(x.notna())


def exp(x: pd.Series) -> pd.Series:
    """自然指数 e^x。catalog 未知。"""
    x = _as_f64(x)
    out = np.exp(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="exp").where(x.notna())


def maximum(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """逐元素 max(a,b)。手册禁止 scalar max；HTML/DSL 大量用 maximum(open,close)。"""
    a, b = _align2(a, b)
    out = np.maximum(a.to_numpy(dtype="float64", copy=False), b.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=a.index, dtype="float64", name="maximum")


def minimum(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """逐元素 min(a,b)。"""
    a, b = _align2(a, b)
    out = np.minimum(a.to_numpy(dtype="float64", copy=False), b.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=a.index, dtype="float64", name="minimum")


def max_(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """DSL 名 ``max`` 的实现（Python 保留字，平台侧注册为 max）。"""
    return maximum(a, b)


def min_(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """DSL 名 ``min`` 的实现。"""
    return minimum(a, b)


def pow_(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """幂运算。平台有 power；HTML/FE 写 pow。注册 ``pow`` 或改写均可。"""
    a, b = _align2(a, b)
    out = np.power(a.to_numpy(dtype="float64", copy=False), b.to_numpy(dtype="float64", copy=False))
    out = pd.Series(out, index=a.index, dtype="float64", name="pow")
    return out.replace([np.inf, -np.inf], np.nan)


def divide(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """安全除法别名（catalog 出现 divide）。建议 = safe_div。"""
    out = _safe_div(a, b)
    out.name = "divide"
    return out


# =============================================================================
# NEW — 时序扩展
# =============================================================================

def ts_median(x: pd.Series, window: int) -> pd.Series:
    """滚动中位数。手册写可用 ts_quantile(x,w,0.5)；HTML/DSL 仍大量写 ts_median。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x.rolling(window=w, min_periods=w).median()
    out.name = "ts_median"
    return out


def ts_zscore(x: pd.Series, window: int) -> pd.Series:
    """滚动 zscore：(x - ts_mean) / ts_std。catalog 未知。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mu = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = _safe_div(x - mu, sd)
    out.name = "ts_zscore"
    return out


def ts_ema(x: pd.Series, span: int) -> pd.Series:
    """EMA。平台有 ema；DSL map 大量写 ts_ema，且 is_lqtp_native 把 ts_ema 当禁名。
    建议平台同时接受 ts_ema 作为 ema 同义词。"""
    x = _as_f64(x)
    s = max(1, int(span))
    out = x.ewm(span=s, adjust=False, min_periods=s).mean()
    out.name = "ts_ema"
    return out


def ewm_mean(x: pd.Series, span: int) -> pd.Series:
    """同 ts_ema / ema。禁名兼容。"""
    out = ts_ema(x, span)
    out.name = "ewm_mean"
    return out


def rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """滚动 VWAP = sum(price*volume)/sum(volume)。catalog 未知。"""
    price, volume = _align2(price, volume)
    w = max(1, int(window))
    num = (price * volume).rolling(window=w, min_periods=w).sum()
    den = volume.rolling(window=w, min_periods=w).sum()
    out = _safe_div(num, den)
    out.name = "rolling_vwap"
    return out


# =============================================================================
# NEW — 工具 / regime（HTML 命名函数，pandas 配方里直接调用）
# =============================================================================

def classify_volume_regime(
    volume: pd.Series,
    window: int = 20,
    high_threshold: float = 1.5,
    low_threshold: float = 0.6,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """成交量高低 regime。HTML 命中 ~27。

    返回 (is_high_vol, is_low_vol, vol_ratio)。
    若平台不支持多返回，可拆：
      vol_ratio = volume / ema(volume, window)
      is_high = where(vol_ratio > high_threshold, 1, 0)
    """
    volume = _as_f64(volume)
    w = max(2, int(window))
    vol_ema = volume.ewm(span=w, adjust=False, min_periods=w).mean()
    vol_ratio = _safe_div(volume, vol_ema)
    is_high = (vol_ratio > float(high_threshold)).astype(float)
    is_low = (vol_ratio < float(low_threshold)).astype(float)
    is_high = is_high.mask(vol_ratio.isna())
    is_low = is_low.mask(vol_ratio.isna())
    is_high.name = "is_high_vol"
    is_low.name = "is_low_vol"
    vol_ratio.name = "vol_ratio"
    return is_high, is_low, vol_ratio


def decompose_overnight_intraday(
    close: pd.Series,
    open_price: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """隔夜/日内收益。HTML 命中 ~3。

    overnight = open/delay(close,1)-1 ; intraday = close/open-1
    """
    close = _as_f64(close)
    open_price = _as_f64(open_price).reindex(close.index)
    overnight = _safe_div(open_price, close.shift(1)) - 1.0
    intraday = _safe_div(close, open_price) - 1.0
    overnight.name = "overnight_ret"
    intraday.name = "intraday_ret"
    return overnight, intraday


def imbalance(buy_vol: pd.Series, sell_vol: pd.Series) -> pd.Series:
    """(buy-sell)/(buy+sell)。HTML 偶发。"""
    out = _safe_div(_as_f64(buy_vol) - _as_f64(sell_vol), _as_f64(buy_vol) + _as_f64(sell_vol))
    out.name = "imbalance"
    return out


# =============================================================================
# NEW — 逻辑包装（is_lqtp_native 黑名单 and_/or_/not_）
# =============================================================================

def and_(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """逻辑与，数值 0/1。a!=0 and b!=0。"""
    a, b = _align2(a, b)
    out = ((a.fillna(0) != 0) & (b.fillna(0) != 0)).astype(float)
    out = out.mask(a.isna() | b.isna())
    out.name = "and_"
    return out


def or_(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """逻辑或。"""
    a, b = _align2(a, b)
    out = ((a.fillna(0) != 0) | (b.fillna(0) != 0)).astype(float)
    out = out.mask(a.isna() | b.isna())
    out.name = "or_"
    return out


def not_(a: ArrayLike) -> pd.Series:
    """逻辑非。"""
    a = _as_f64(a)
    out = (a.fillna(0) == 0).astype(float)
    out = out.mask(a.isna())
    out.name = "not_"
    return out


# =============================================================================
# ALIAS — 平台已有等价；仅为 HTML/FE 同名兼容（可选）
# =============================================================================

def protected_div(a: ArrayLike, b: ArrayLike) -> pd.Series:
    """≡ safe_div。"""
    out = _safe_div(a, b)
    out.name = "protected_div"
    return out


def clip(x: pd.Series, lower: float, upper: float) -> pd.Series:
    """≡ cap(x, lo, hi)。is_lqtp_native 黑名单名。"""
    x = _as_f64(x)
    out = x.clip(lower=float(lower), upper=float(upper))
    out.name = "clip"
    return out


def ts_delay(x: pd.Series, n: int) -> pd.Series:
    """≡ delay(x,n)。"""
    x = _as_f64(x)
    out = x.shift(int(n))
    out.name = "ts_delay"
    return out


# =============================================================================
# EXTRA — 未来常用、非模型算子（无 beta/CAPM/回归残差/中性化）
# =============================================================================

# --- 数学变换 ---

def log1p(x: pd.Series) -> pd.Series:
    """log(1+x)；x<=-1 → NaN。"""
    x = _as_f64(x)
    out = np.log1p(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="log1p").where(x > -1.0)


def expm1(x: pd.Series) -> pd.Series:
    """exp(x)-1。"""
    x = _as_f64(x)
    out = np.expm1(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="expm1").where(x.notna())


def cbrt(x: pd.Series) -> pd.Series:
    """立方根（保留符号）。"""
    x = _as_f64(x)
    out = np.cbrt(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="cbrt").where(x.notna())


def softsign(x: pd.Series) -> pd.Series:
    """x / (1+|x|)，比 tanh 更轻量的饱和。"""
    x = _as_f64(x)
    out = _safe_div(x, 1.0 + x.abs())
    out.name = "softsign"
    return out


def softplus(x: pd.Series) -> pd.Series:
    """log(1+exp(x))，数值稳定写法。"""
    x = _as_f64(x)
    # softplus(x) = log1p(exp(-|x|)) + max(x,0)
    ax = x.abs()
    out = np.log1p(np.exp((-ax).to_numpy(dtype="float64", copy=False))) + np.maximum(x.to_numpy(dtype="float64", copy=False), 0.0)
    return pd.Series(out, index=x.index, dtype="float64", name="softplus").where(x.notna())


def relu(x: pd.Series) -> pd.Series:
    """max(x, 0)。"""
    x = _as_f64(x)
    out = x.clip(lower=0.0)
    out.name = "relu"
    return out


def signed_power(x: pd.Series, p: float) -> pd.Series:
    """sign(x) * |x|^p。"""
    x = _as_f64(x)
    out = np.sign(x.to_numpy(dtype="float64", copy=False)) * np.power(
        np.abs(x.to_numpy(dtype="float64", copy=False)), float(p)
    )
    return pd.Series(out, index=x.index, dtype="float64", name="signed_power").replace([np.inf, -np.inf], np.nan)


def saturate(x: pd.Series, scale: float = 1.0) -> pd.Series:
    """x / sqrt(scale^2 + x^2)，光滑饱和到 (-1,1)。"""
    x = _as_f64(x)
    s = float(scale)
    out = _safe_div(x, np.sqrt(s * s + x * x))
    out.name = "saturate"
    return out


def sqrt_abs(x: pd.Series) -> pd.Series:
    """sign(x)*sqrt(|x|)。"""
    x = _as_f64(x)
    out = np.sign(x.to_numpy(dtype="float64", copy=False)) * np.sqrt(np.abs(x.to_numpy(dtype="float64", copy=False)))
    return pd.Series(out, index=x.index, dtype="float64", name="sqrt_abs").where(x.notna())


def unitize(x: pd.Series) -> pd.Series:
    """x / (|x|+eps) → 近似 sign，零附近平滑。"""
    x = _as_f64(x)
    out = _safe_div(x, x.abs() + 1e-12)
    out.name = "unitize"
    return out


# --- 价格结构（OHLC 几何，极常用） ---

def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
    """(H+L)/2。"""
    high, low = _align2(high, low)
    out = 0.5 * (high + low)
    out.name = "hl2"
    return out


def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """典型价 (H+L+C)/3。"""
    high, low, close = map(_as_f64, (high, low, close))
    out = (high + low + close) / 3.0
    out.name = "hlc3"
    return out


def ohlc4(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(O+H+L+C)/4。"""
    open_, high, low, close = map(_as_f64, (open_, high, low, close))
    out = (open_ + high + low + close) / 4.0
    out.name = "ohlc4"
    return out


def candle_body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """实体：close - open。"""
    open_, close = _align2(open_, close)
    out = close - open_
    out.name = "candle_body"
    return out


def upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    """上影线：high - max(open, close)。"""
    open_, high, close = map(_as_f64, (open_, high, close))
    out = high - maximum(open_, close)
    out.name = "upper_shadow"
    return out


def lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """下影线：min(open, close) - low。"""
    open_, low, close = map(_as_f64, (open_, low, close))
    out = minimum(open_, close) - low
    out.name = "lower_shadow"
    return out


def dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """成交额代理：price * volume。"""
    price, volume = _align2(price, volume)
    out = price * volume
    out.name = "dollar_volume"
    return out


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """volume / ts_mean(volume, w)。"""
    volume = _as_f64(volume)
    w = max(1, int(window))
    base = volume.rolling(window=w, min_periods=w).mean()
    out = _safe_div(volume, base)
    out.name = "relative_volume"
    return out


def vwap_deviation(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """price / rolling_vwap - 1。"""
    price = _as_f64(price)
    v = rolling_vwap(price, volume, window)
    out = _safe_div(price, v) - 1.0
    out.name = "vwap_deviation"
    return out


def amihud(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """非流动性：ts_mean(|r| / dollar_volume, w)。volume 作份额时需外部已是金额也可。"""
    close, volume = _align2(close, volume)
    r = close.pct_change()
    illiq = _safe_div(r.abs(), (close * volume).where(volume != 0))
    w = max(1, int(window))
    out = illiq.rolling(window=w, min_periods=w).mean()
    out.name = "amihud"
    return out


# --- 时序统计扩展 ---

def WMA(x: pd.Series, window: int) -> pd.Series:
    """加权移动平均（线性权重 1..w）。"""
    x = _as_f64(x)
    w = max(1, int(window))
    weights = np.arange(1, w + 1, dtype="float64")

    def _wma(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(np.dot(arr, weights) / weights.sum())

    out = x.rolling(window=w, min_periods=w).apply(_wma, raw=True)
    out.name = "WMA"
    return out


def ts_var(x: pd.Series, window: int) -> pd.Series:
    """滚动方差（总体 ddof=0）。"""
    x = _as_f64(x)
    w = max(2, int(window))
    out = x.rolling(window=w, min_periods=w).var(ddof=0)
    out.name = "ts_var"
    return out


def ts_mad(x: pd.Series, window: int) -> pd.Series:
    """滚动中位绝对偏差 median(|x - median(x)|)（同窗）。"""
    x = _as_f64(x)
    w = max(1, int(window))

    def _mad(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        med = np.median(arr)
        return float(np.median(np.abs(arr - med)))

    out = x.rolling(window=w, min_periods=w).apply(_mad, raw=True)
    out.name = "ts_mad"
    return out


def ts_iqr(x: pd.Series, window: int) -> pd.Series:
    """滚动四分位距 q75-q25。"""
    x = _as_f64(x)
    w = max(2, int(window))
    q75 = x.rolling(window=w, min_periods=w).quantile(0.75)
    q25 = x.rolling(window=w, min_periods=w).quantile(0.25)
    out = q75 - q25
    out.name = "ts_iqr"
    return out


def ts_range(x: pd.Series, window: int) -> pd.Series:
    """滚动极差 ts_max - ts_min。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x.rolling(window=w, min_periods=w).max() - x.rolling(window=w, min_periods=w).min()
    out.name = "ts_range"
    return out


def ts_cv(x: pd.Series, window: int) -> pd.Series:
    """变异系数 ts_std / |ts_mean|。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mu = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = _safe_div(sd, mu.abs())
    out.name = "ts_cv"
    return out


def ts_demean(x: pd.Series, window: int) -> pd.Series:
    """x - ts_mean(x,w)。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x - x.rolling(window=w, min_periods=w).mean()
    out.name = "ts_demean"
    return out


def ts_product(x: pd.Series, window: int) -> pd.Series:
    """滚动乘积（用 logsumexp 风格：对 log1p 不通用，直接 prod）。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x.rolling(window=w, min_periods=w).apply(np.prod, raw=True)
    out.name = "ts_product"
    return out.replace([np.inf, -np.inf], np.nan)


def ts_cumsum(x: pd.Series) -> pd.Series:
    """累计和（因果）。"""
    x = _as_f64(x)
    out = x.cumsum()
    out.name = "ts_cumsum"
    return out


def ts_cummax(x: pd.Series) -> pd.Series:
    """累计最大。"""
    x = _as_f64(x)
    out = x.cummax()
    out.name = "ts_cummax"
    return out


def ts_cummin(x: pd.Series) -> pd.Series:
    """累计最小。"""
    x = _as_f64(x)
    out = x.cummin()
    out.name = "ts_cummin"
    return out


def ts_drawdown(x: pd.Series) -> pd.Series:
    """相对峰值回撤：(x / cummax(x) - 1)。价格或净值序列。"""
    x = _as_f64(x)
    peak = x.cummax()
    out = _safe_div(x, peak) - 1.0
    out.name = "ts_drawdown"
    return out


def ts_upside_deviation(x: pd.Series, window: int, threshold: float = 0.0) -> pd.Series:
    """上行偏差：sqrt(mean(max(x-thr,0)^2))。"""
    x = _as_f64(x)
    w = max(1, int(window))
    up = (x - float(threshold)).clip(lower=0.0) ** 2
    out = np.sqrt(up.rolling(window=w, min_periods=w).mean())
    out.name = "ts_upside_deviation"
    return out


def ts_downside_deviation(x: pd.Series, window: int, threshold: float = 0.0) -> pd.Series:
    """下行偏差：sqrt(mean(min(x-thr,0)^2))。"""
    x = _as_f64(x)
    w = max(1, int(window))
    dn = (x - float(threshold)).clip(upper=0.0) ** 2
    out = np.sqrt(dn.rolling(window=w, min_periods=w).mean())
    out.name = "ts_downside_deviation"
    return out


def ts_positive_ratio(x: pd.Series, window: int) -> pd.Series:
    """窗口内 x>0 的比例。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = (x > 0).astype(float).rolling(window=w, min_periods=w).mean()
    out.name = "ts_positive_ratio"
    return out


def ts_negative_ratio(x: pd.Series, window: int) -> pd.Series:
    """窗口内 x<0 的比例。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = (x < 0).astype(float).rolling(window=w, min_periods=w).mean()
    out.name = "ts_negative_ratio"
    return out


def ts_vol_of_vol(x: pd.Series, vol_window: int = 20, meta_window: int = 20) -> pd.Series:
    """波动的波动：ts_std(ts_std(x, vol_w), meta_w)。"""
    x = _as_f64(x)
    vw = max(2, int(vol_window))
    mw = max(2, int(meta_window))
    vol = x.rolling(window=vw, min_periods=vw).std(ddof=0)
    out = vol.rolling(window=mw, min_periods=mw).std(ddof=0)
    out.name = "ts_vol_of_vol"
    return out


def ts_trimmed_mean(x: pd.Series, window: int, trim_frac: float = 0.1) -> pd.Series:
    """滚动截尾均值（两端各去掉 trim_frac）。"""
    x = _as_f64(x)
    w = max(3, int(window))
    frac = min(max(float(trim_frac), 0.0), 0.4)

    def _trim(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        k = int(len(arr) * frac)
        if 2 * k >= len(arr):
            return float(np.mean(arr))
        s = np.sort(arr)
        return float(np.mean(s[k: len(arr) - k]))

    out = x.rolling(window=w, min_periods=w).apply(_trim, raw=True)
    out.name = "ts_trimmed_mean"
    return out


def ts_decay_exp(x: pd.Series, window: int) -> pd.Series:
    """指数衰减加权和（近端权重大），再按权重归一：≈ 因果 ewma-sum / sum(w)。"""
    x = _as_f64(x)
    w = max(1, int(window))
    # alpha such that half-life ~ window/2 style: weights = (1-a)^i from old to new
    alpha = 2.0 / (w + 1.0)
    weights = np.array([(1.0 - alpha) ** i for i in range(w - 1, -1, -1)], dtype="float64")
    weights /= weights.sum()

    def _f(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(np.dot(arr, weights))

    out = x.rolling(window=w, min_periods=w).apply(_f, raw=True)
    out.name = "ts_decay_exp"
    return out


def ts_true_streak(cond: pd.Series) -> pd.Series:
    """连续满足条件的当前 streak 长度（平台手册曾拒同名；因子侧很常用）。
    cond 非零为 True；不满足则归零。"""
    cond = _as_f64(cond)
    flag = (cond.fillna(0) != 0).astype(int)
    # group by breaks
    groups = (flag == 0).cumsum()
    streak = flag.groupby(groups).cumsum()
    streak = streak.where(flag == 1, 0.0).astype(float)
    streak = streak.mask(cond.isna())
    streak.name = "ts_true_streak"
    return streak


def ts_time_since_change(x: pd.Series) -> pd.Series:
    """距上次取值变化的 bar 数（含当前为 0 当发生变化）。"""
    x = _as_f64(x)
    changed = x.ne(x.shift(1))
    changed.iloc[0] = True
    groups = changed.cumsum()
    out = x.groupby(groups).cumcount().astype(float)
    out = out.mask(x.isna())
    out.name = "ts_time_since_change"
    return out


# --- 常用技术指标（非模型） ---

def NATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """归一化 ATR：100 * ATR / close。"""
    close = _as_f64(close)
    out = 100.0 * _safe_div(ATR(high, low, close, window), close)
    out.name = "NATR"
    return out


def CCI(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Commodity Channel Index。"""
    tp = hlc3(high, low, close)
    w = max(2, int(window))
    ma = tp.rolling(window=w, min_periods=w).mean()
    md = (tp - ma).abs().rolling(window=w, min_periods=w).mean()
    out = _safe_div(tp - ma, 0.015 * md)
    out.name = "CCI"
    return out


def WILLR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R：-100 * (hh - close) / (hh - ll)。"""
    high, low, close = map(_as_f64, (high, low, close))
    w = max(1, int(window))
    hh = high.rolling(window=w, min_periods=w).max()
    ll = low.rolling(window=w, min_periods=w).min()
    out = -100.0 * _safe_div(hh - close, hh - ll)
    out.name = "WILLR"
    return out


def STOCH_K(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Stochastic %K。"""
    high, low, close = map(_as_f64, (high, low, close))
    w = max(1, int(window))
    hh = high.rolling(window=w, min_periods=w).max()
    ll = low.rolling(window=w, min_periods=w).min()
    out = 100.0 * _safe_div(close - ll, hh - ll)
    out.name = "STOCH_K"
    return out


def STOCH_D(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """Stochastic %D = SMA(%K)。"""
    k = STOCH_K(high, low, close, k_window)
    d = k.rolling(window=max(1, int(d_window)), min_periods=1).mean()
    d.name = "STOCH_D"
    return d


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume（累计）。"""
    close, volume = _align2(close, volume)
    direction = np.sign(close.diff().fillna(0.0).to_numpy(dtype="float64", copy=False))
    signed = direction * volume.to_numpy(dtype="float64", copy=False)
    out = pd.Series(signed, index=close.index, dtype="float64").cumsum()
    out.name = "OBV"
    return out


def MACD_line(x: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD 线 = ema(fast) - ema(slow)。"""
    x = _as_f64(x)
    f = max(1, int(fast))
    s = max(f + 1, int(slow))
    out = x.ewm(span=f, adjust=False, min_periods=f).mean() - x.ewm(span=s, adjust=False, min_periods=s).mean()
    out.name = "MACD_line"
    return out


def MACD_signal(x: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD 信号线 = ema(MACD_line, signal)。"""
    line = MACD_line(x, fast, slow)
    sig = max(1, int(signal))
    out = line.ewm(span=sig, adjust=False, min_periods=sig).mean()
    out.name = "MACD_signal"
    return out


def MACD_hist(x: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD 柱 = MACD_line - MACD_signal。"""
    out = MACD_line(x, fast, slow) - MACD_signal(x, fast, slow, signal)
    out.name = "MACD_hist"
    return out


def BB_percent_b(x: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """布林 %B：(x - lower) / (upper - lower)。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mid = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    upper = mid + float(n_std) * sd
    lower = mid - float(n_std) * sd
    out = _safe_div(x - lower, upper - lower)
    out.name = "BB_percent_b"
    return out


def BB_bandwidth(x: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """布林带宽：(upper - lower) / mid。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mid = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = _safe_div(2.0 * float(n_std) * sd, mid)
    out.name = "BB_bandwidth"
    return out


def KAMA(x: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average（因果递推）。"""
    x = _as_f64(x)
    n = max(2, int(window))
    fast_sc = 2.0 / (float(fast) + 1.0)
    slow_sc = 2.0 / (float(slow) + 1.0)
    change = (x - x.shift(n)).abs()
    volatility = x.diff().abs().rolling(window=n, min_periods=n).sum()
    er = _safe_div(change, volatility)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    vals = x.to_numpy(dtype="float64", copy=True)
    sc_a = sc.to_numpy(dtype="float64", copy=False)
    out = np.full_like(vals, np.nan)
    # seed
    start = n - 1
    if start < len(vals) and np.isfinite(vals[start]):
        out[start] = vals[start]
        for i in range(start + 1, len(vals)):
            if not np.isfinite(vals[i]) or not np.isfinite(sc_a[i]) or not np.isfinite(out[i - 1]):
                out[i] = np.nan
                continue
            out[i] = out[i - 1] + sc_a[i] * (vals[i] - out[i - 1])
    return pd.Series(out, index=x.index, dtype="float64", name="KAMA")


# --- EXTRA2：再补一批，凑到约 100+（仍无模型类） ---

def floor(x: pd.Series) -> pd.Series:
    """向下取整。"""
    x = _as_f64(x)
    out = np.floor(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="floor").where(x.notna())


def ceil(x: pd.Series) -> pd.Series:
    """向上取整。"""
    x = _as_f64(x)
    out = np.ceil(x.to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="ceil").where(x.notna())


def frac(x: pd.Series) -> pd.Series:
    """小数部分 x - floor(x)。"""
    x = _as_f64(x)
    out = x - np.floor(x.to_numpy(dtype="float64", copy=False))
    out.name = "frac"
    return out.where(x.notna())


def square(x: pd.Series) -> pd.Series:
    """x^2。"""
    x = _as_f64(x)
    out = x * x
    out.name = "square"
    return out


def cube(x: pd.Series) -> pd.Series:
    """x^3。"""
    x = _as_f64(x)
    out = x * x * x
    out.name = "cube"
    return out


def inv(x: pd.Series) -> pd.Series:
    """1/x，零→NaN。"""
    out = _safe_div(1.0, x)
    out.name = "inv"
    return out


def clip01(x: pd.Series) -> pd.Series:
    """截断到 [0,1]。"""
    x = _as_f64(x)
    out = x.clip(lower=0.0, upper=1.0)
    out.name = "clip01"
    return out


def logistic(x: pd.Series, scale: float = 1.0) -> pd.Series:
    """1/(1+exp(-x/scale))；与 sigmoid 同族，带可调 scale。"""
    x = _as_f64(x)
    s = float(scale) if float(scale) != 0 else 1.0
    z = (-x / s).clip(lower=-60, upper=60)
    out = 1.0 / (1.0 + np.exp(z.to_numpy(dtype="float64", copy=False)))
    return pd.Series(out, index=x.index, dtype="float64", name="logistic").where(x.notna())


def log_return(x: pd.Series, window: int = 1) -> pd.Series:
    """对数收益 log(x / delay(x,n))。"""
    x = _as_f64(x)
    w = max(1, int(window))
    prev = x.shift(w)
    out = np.log(_safe_div(x, prev).to_numpy(dtype="float64", copy=False))
    return pd.Series(out, index=x.index, dtype="float64", name="log_return").where(prev.notna() & (prev > 0) & (x > 0))


def simple_return(x: pd.Series, window: int = 1) -> pd.Series:
    """简单收益 x/delay(x,n)-1。"""
    x = _as_f64(x)
    w = max(1, int(window))
    prev = x.shift(w)
    out = _safe_div(x, prev) - 1.0
    out.name = "simple_return"
    return out


def overnight_ret(close: pd.Series, open_price: pd.Series) -> pd.Series:
    """隔夜收益 open/delay(close,1)-1（单返回版）。"""
    o, _ = decompose_overnight_intraday(close, open_price)
    o.name = "overnight_ret"
    return o


def intraday_ret(close: pd.Series, open_price: pd.Series) -> pd.Series:
    """日内收益 close/open-1（单返回版）。"""
    _, i = decompose_overnight_intraday(close, open_price)
    i.name = "intraday_ret"
    return i


def gap_pct(close: pd.Series, open_price: pd.Series) -> pd.Series:
    """跳空百分比，同 overnight_ret。"""
    return overnight_ret(close, open_price).rename("gap_pct")


def realized_vol(x: pd.Series, window: int = 20, annualize: float = 252.0) -> pd.Series:
    """已实现波动：std(r) * sqrt(annualize)；x 为收益序列。"""
    x = _as_f64(x)
    w = max(2, int(window))
    out = x.rolling(window=w, min_periods=w).std(ddof=0) * np.sqrt(float(annualize))
    out.name = "realized_vol"
    return out


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20, annualize: float = 252.0) -> pd.Series:
    """Parkinson 波动估计（高低价）。"""
    high, low = _align2(high, low)
    w = max(1, int(window))
    rs = (np.log(_safe_div(high, low).to_numpy(dtype="float64", copy=False))) ** 2
    rs = pd.Series(rs, index=high.index, dtype="float64")
    # var = rs / (4*ln2) ; vol = sqrt(mean(var))*sqrt(ann)
    const = 1.0 / (4.0 * np.log(2.0))
    out = np.sqrt(rs.rolling(window=w, min_periods=w).mean() * const) * np.sqrt(float(annualize))
    out.name = "parkinson_vol"
    return out


def ts_ewm_std(x: pd.Series, span: int) -> pd.Series:
    """EWM 标准差。"""
    x = _as_f64(x)
    s = max(2, int(span))
    out = x.ewm(span=s, adjust=False, min_periods=s).std(bias=True)
    out.name = "ts_ewm_std"
    return out


def ts_autocorr(x: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """滚动自相关 corr(x, delay(x,lag))。"""
    x = _as_f64(x)
    w = max(3, int(window))
    L = max(1, int(lag))
    y = x.shift(L)
    out = x.rolling(window=w, min_periods=w).corr(y)
    out.name = "ts_autocorr"
    return out


def ts_zero_ratio(x: pd.Series, window: int) -> pd.Series:
    """窗口内 x==0 的比例。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = (x == 0).astype(float).rolling(window=w, min_periods=w).mean()
    out.name = "ts_zero_ratio"
    return out


def ts_sign_persistence(x: pd.Series, window: int) -> pd.Series:
    """符号持续性：mean(sign(x_t)==sign(x_{t-1}))。"""
    x = _as_f64(x)
    w = max(2, int(window))
    s = np.sign(x.to_numpy(dtype="float64", copy=False))
    same = (s == np.roll(s, 1)).astype(float)
    same[0] = np.nan
    ser = pd.Series(same, index=x.index, dtype="float64").where(x.notna())
    out = ser.rolling(window=w, min_periods=w).mean()
    out.name = "ts_sign_persistence"
    return out


def ts_quantile_range(x: pd.Series, window: int, q_low: float = 0.1, q_high: float = 0.9) -> pd.Series:
    """滚动分位距 q_high - q_low。"""
    x = _as_f64(x)
    w = max(2, int(window))
    hi = x.rolling(window=w, min_periods=w).quantile(float(q_high))
    lo = x.rolling(window=w, min_periods=w).quantile(float(q_low))
    out = hi - lo
    out.name = "ts_quantile_range"
    return out


def ts_winsorize(x: pd.Series, window: int, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """时序滚动截尾：用同窗分位上下限 clip。"""
    x = _as_f64(x)
    w = max(2, int(window))
    lo = x.rolling(window=w, min_periods=w).quantile(float(q_low))
    hi = x.rolling(window=w, min_periods=w).quantile(float(q_high))
    out = x.clip(lower=lo, upper=hi)
    out.name = "ts_winsorize"
    return out


def ts_count(x: pd.Series, window: int) -> pd.Series:
    """窗口内非 NaN 计数。"""
    x = _as_f64(x)
    w = max(1, int(window))
    out = x.rolling(window=w, min_periods=1).count().astype(float)
    out.name = "ts_count"
    return out


def BB_mid(x: pd.Series, window: int = 20) -> pd.Series:
    """布林中轨 = SMA。"""
    out = SMA(x, window)
    out.name = "BB_mid"
    return out


def BB_upper(x: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """布林上轨。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mid = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = mid + float(n_std) * sd
    out.name = "BB_upper"
    return out


def BB_lower(x: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """布林下轨。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mid = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = mid - float(n_std) * sd
    out.name = "BB_lower"
    return out


def AROON_UP(high: pd.Series, window: int = 25) -> pd.Series:
    """Aroon Up：100 * (n - bars_since_hh) / n。"""
    high = _as_f64(high)
    w = max(1, int(window))
    # rolling argmax: position of max in window
    def _aroon(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return 100.0 * (np.argmax(arr) + 1) / float(len(arr))  # argmax from left; last max preferred via reverse

    # use last occurrence of max
    def _aroon_last(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        idx = len(arr) - 1 - int(np.argmax(arr[::-1]))
        return 100.0 * (idx + 1) / float(len(arr))

    out = high.rolling(window=w, min_periods=w).apply(_aroon_last, raw=True)
    out.name = "AROON_UP"
    return out


def AROON_DOWN(low: pd.Series, window: int = 25) -> pd.Series:
    """Aroon Down：100 * (n - bars_since_ll) / n。"""
    low = _as_f64(low)
    w = max(1, int(window))

    def _aroon_last_min(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        idx = len(arr) - 1 - int(np.argmin(arr[::-1]))
        return 100.0 * (idx + 1) / float(len(arr))

    out = low.rolling(window=w, min_periods=w).apply(_aroon_last_min, raw=True)
    out.name = "AROON_DOWN"
    return out


def AD_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line（累计）。"""
    high, low, close, volume = map(_as_f64, (high, low, close, volume))
    # mfm = ((c-l)-(h-c))/(h-l)
    mfm = _safe_div((close - low) - (high - close), high - low)
    mfv = mfm * volume
    out = mfv.fillna(0.0).cumsum()
    out = out.mask(high.isna() | low.isna() | close.isna() | volume.isna())
    out.name = "AD_line"
    return out


def VPT(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend：cumsum(volume * pct_change(close))。"""
    close, volume = _align2(close, volume)
    out = (volume * close.pct_change().fillna(0.0)).cumsum()
    out.name = "VPT"
    return out


def force_index(close: pd.Series, volume: pd.Series, window: int = 13) -> pd.Series:
    """Force Index = ema( (close-delay(close,1))*volume , window)。"""
    close, volume = _align2(close, volume)
    raw = (close - close.shift(1)) * volume
    w = max(1, int(window))
    out = raw.ewm(span=w, adjust=False, min_periods=w).mean()
    out.name = "force_index"
    return out


def PPO(x: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Percentage Price Oscillator：100*(ema_fast-ema_slow)/ema_slow。"""
    x = _as_f64(x)
    f = max(1, int(fast))
    s = max(f + 1, int(slow))
    ef = x.ewm(span=f, adjust=False, min_periods=f).mean()
    es = x.ewm(span=s, adjust=False, min_periods=s).mean()
    out = 100.0 * _safe_div(ef - es, es)
    out.name = "PPO"
    return out


def TRIX(x: pd.Series, window: int = 15) -> pd.Series:
    """三重 EMA 的变动率。"""
    x = _as_f64(x)
    w = max(1, int(window))
    e1 = x.ewm(span=w, adjust=False, min_periods=w).mean()
    e2 = e1.ewm(span=w, adjust=False, min_periods=w).mean()
    e3 = e2.ewm(span=w, adjust=False, min_periods=w).mean()
    out = e3.pct_change()
    out.name = "TRIX"
    return out


def MFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Money Flow Index。"""
    tp = hlc3(high, low, close)
    volume = _as_f64(volume).reindex(tp.index)
    raw_mf = tp * volume
    delta = tp.diff()
    pos = raw_mf.where(delta > 0, 0.0)
    neg = raw_mf.where(delta < 0, 0.0)
    w = max(1, int(window))
    pos_sum = pos.rolling(window=w, min_periods=w).sum()
    neg_sum = neg.rolling(window=w, min_periods=w).sum()
    mr = _safe_div(pos_sum, neg_sum)
    out = 100.0 - (100.0 / (1.0 + mr))
    out = out.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    out = out.mask((pos_sum == 0) & (neg_sum > 0), 0.0)
    out.name = "MFI"
    return out


def rolling_sharpe(x: pd.Series, window: int = 60, annualize: float = 252.0) -> pd.Series:
    """滚动 Sharpe：mean/std * sqrt(ann)；x 为收益。非模型回归。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mu = x.rolling(window=w, min_periods=w).mean()
    sd = x.rolling(window=w, min_periods=w).std(ddof=0)
    out = _safe_div(mu, sd) * np.sqrt(float(annualize))
    out.name = "rolling_sharpe"
    return out


def rolling_sortino(x: pd.Series, window: int = 60, annualize: float = 252.0) -> pd.Series:
    """滚动 Sortino：mean / downside_dev * sqrt(ann)。"""
    x = _as_f64(x)
    w = max(2, int(window))
    mu = x.rolling(window=w, min_periods=w).mean()
    dd = ts_downside_deviation(x, w, 0.0)
    out = _safe_div(mu, dd) * np.sqrt(float(annualize))
    out.name = "rolling_sortino"
    return out


# DSL 注册名映射（Python 保留字用 max_/min_/pow_ 实现）
DSL_REGISTER_NAME = {
    "max_": "max",
    "min_": "min",
    "pow_": "pow",
}


# ---------------------------------------------------------------------------
# 公开算子清单（给对接方数）
# ---------------------------------------------------------------------------
NEW_OPERATORS = [
    # 技术（缺口）
    "true_range", "ATR", "ATR_WILDER", "RSI", "RSI_WILDER", "ROC", "ADX", "SMA",
    # 数学（缺口）— 含 tanh
    "tanh", "exp", "maximum", "minimum", "max", "min", "pow", "divide",
    # 时序（缺口）
    "ts_median", "ts_zscore", "ts_ema", "ewm_mean", "rolling_vwap",
    # 工具（缺口）
    "classify_volume_regime", "decompose_overnight_intraday", "imbalance",
    # 逻辑（缺口）
    "and_", "or_", "not_",
    # ---- EXTRA 未来常用（非模型） ----
    "log1p", "expm1", "cbrt", "softsign", "softplus", "relu",
    "signed_power", "saturate", "sqrt_abs", "unitize",
    "hl2", "hlc3", "ohlc4", "candle_body", "upper_shadow", "lower_shadow",
    "dollar_volume", "relative_volume", "vwap_deviation", "amihud",
    "WMA", "ts_var", "ts_mad", "ts_iqr", "ts_range", "ts_cv", "ts_demean",
    "ts_product", "ts_cumsum", "ts_cummax", "ts_cummin", "ts_drawdown",
    "ts_upside_deviation", "ts_downside_deviation",
    "ts_positive_ratio", "ts_negative_ratio", "ts_vol_of_vol",
    "ts_trimmed_mean", "ts_decay_exp", "ts_true_streak", "ts_time_since_change",
    "NATR", "CCI", "WILLR", "STOCH_K", "STOCH_D", "OBV",
    "MACD_line", "MACD_signal", "MACD_hist", "BB_percent_b", "BB_bandwidth", "KAMA",
    # ---- EXTRA2 凑 100+ ----
    "floor", "ceil", "frac", "square", "cube", "inv", "clip01", "logistic",
    "log_return", "simple_return", "overnight_ret", "intraday_ret", "gap_pct",
    "realized_vol", "parkinson_vol", "ts_ewm_std", "ts_autocorr",
    "ts_zero_ratio", "ts_sign_persistence", "ts_quantile_range", "ts_winsorize", "ts_count",
    "BB_mid", "BB_upper", "BB_lower", "AROON_UP", "AROON_DOWN",
    "AD_line", "VPT", "force_index", "PPO", "TRIX", "MFI",
    "rolling_sharpe", "rolling_sortino",
]

ALIAS_OPERATORS = [
    "protected_div",  # → safe_div（平台已有等价，仅同名兼容）
    "clip",           # → cap
    "ts_delay",       # → delay
]

EXTRA_FUTURE_COMMON = [
    "log1p", "expm1", "cbrt", "softsign", "softplus", "relu",
    "signed_power", "saturate", "sqrt_abs", "unitize",
    "hl2", "hlc3", "ohlc4", "candle_body", "upper_shadow", "lower_shadow",
    "dollar_volume", "relative_volume", "vwap_deviation", "amihud",
    "WMA", "ts_var", "ts_mad", "ts_iqr", "ts_range", "ts_cv", "ts_demean",
    "ts_product", "ts_cumsum", "ts_cummax", "ts_cummin", "ts_drawdown",
    "ts_upside_deviation", "ts_downside_deviation",
    "ts_positive_ratio", "ts_negative_ratio", "ts_vol_of_vol",
    "ts_trimmed_mean", "ts_decay_exp", "ts_true_streak", "ts_time_since_change",
    "NATR", "CCI", "WILLR", "STOCH_K", "STOCH_D", "OBV",
    "MACD_line", "MACD_signal", "MACD_hist", "BB_percent_b", "BB_bandwidth", "KAMA",
    "floor", "ceil", "frac", "square", "cube", "inv", "clip01", "logistic",
    "log_return", "simple_return", "overnight_ret", "intraday_ret", "gap_pct",
    "realized_vol", "parkinson_vol", "ts_ewm_std", "ts_autocorr",
    "ts_zero_ratio", "ts_sign_persistence", "ts_quantile_range", "ts_winsorize", "ts_count",
    "BB_mid", "BB_upper", "BB_lower", "AROON_UP", "AROON_DOWN",
    "AD_line", "VPT", "force_index", "PPO", "TRIX", "MFI",
    "rolling_sharpe", "rolling_sortino",
]

__all__ = NEW_OPERATORS + ALIAS_OPERATORS + [
    "max_", "min_", "pow_", "NEW_OPERATORS", "ALIAS_OPERATORS",
    "EXTRA_FUTURE_COMMON", "DSL_REGISTER_NAME",
]
