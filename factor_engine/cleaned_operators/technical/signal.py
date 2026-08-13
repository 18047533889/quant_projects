# -*- coding: utf-8 -*-
"""
技术指标与信号算子（runtime 实现层）。

文件位置
--------
本模块是 cleaned_operators 里 **唯一** 存放经典 TA 指标与信号条件算子的文件：
``cleaned_operators/technical_signal.py``。
导入 ``cleaned_operators.technical_signal`` 或调用 ``load_all()`` 后，
算子会注册到 ``OperatorRegistry``，可通过::

    from cleaned_operators.registry import OperatorRegistry
    op = OperatorRegistry.get("RSI")          # 或别名 ts_rsi → RSI
    out = op.calculate(close_df, window=14)

命名约定
--------
- **canonical 名** 多为 factor_dsl_np 风格：``MACD`` / ``RSI`` / ``ADX``（大写或 PascalCase）。
- **factor_engine** 侧 DSL 常用 ``ts_macd`` / ``ts_rsi`` / ``ts_adx`` 前缀；
  对应关系见 ``_aliases.py`` 的「技术指标」段，勿与 ``time_series.py`` 里的 ``ts_mean`` 等滚动算子混淆。
- 完整清单与来源见 ``operators_catalog.md`` → ``### technical_signal``。

模块内分区（按源码顺序）
------------------------
1. **经典技术指标**（来源 ``factor_dsl_np/financial``，约 L21–L555）
   ADX, ADXR, AROON(_up/_down), ATR, BollingerBands/BollingerLower/BollingerUpper,
   CCI, DPO, KAMA, MACD(_hist/_line/_signal), MOM, OBV, ROC, RSI,
   StochasticK/StochasticD, TRIX, WilliamsR

2. **信号 / 条件 / 自定义 MACD 变体**（来源 ``factor_dsl_np/signal``，约 L556–文件末）
   clamp, hump_decay, if_else, ifnan, is_finite, saturate,
   signed_log, signed_power, trade_when, where,
   vp_weighted_price, vpmacd, vpmacd_signal

输入约定
--------
- 入参均为 ``pd.DataFrame``（宽表，列通常为字段名或标的）。
- OHLC 类指标需传入 ``high`` / ``low`` / ``close`` 等对应 DataFrame；
  单序列指标（如 RSI）传入 ``close`` 与 ``window`` 即可。
- 尚未按 MultiIndex (date, symbol) 做分组滚动；与 factor_engine backend 的 panel 语义可能不一致，
  接入全市场 panel 前需在调用侧按标的切分或后续统一封装。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)


def _wilder_smooth(series: pd.Series, window: int, *, min_periods: int = 1) -> pd.Series:
    """Wilder 平滑（等价于 alpha=1/window 的 EWM）。"""
    return np.where(window, adjust=False, min_periods=min_periods).mean() != 0, series.ewm(alpha=1.0 / window, adjust=False, min_periods=min_periods).mean(), np.nan)


def _compute_rsi_wilder(close: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(2, int(window))
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = np.where(w, adjust=False, min_periods=w).mean() != 0, gain.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean(), np.nan)
    avg_loss = np.where(w, adjust=False, min_periods=w).mean() != 0, loss.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean(), np.nan)
    rs = np.where(avg_loss.replace(0, np.nan) != 0, avg_gain / avg_loss.replace(0, np.nan), np.nan)
    rsi = np.where((1 + rs)) != 0, 100 - (100 / (1 + rs)), np.nan)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _compute_atr_wilder(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    w = max(2, int(window))
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return np.where(w, adjust=False, min_periods=w).mean() != 0, tr.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean(), np.nan)


def _compute_dmi_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    tr = np.maximum(np.maximum(high - low, (high - close.shift(1)).abs()), (low - close.shift(1)).abs())
    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = _wilder_smooth(tr, window)
    plus_di = np.where(atr.replace(0, np.nan)) != 0, 100 * (_wilder_smooth(plus_dm, window) / atr.replace(0, np.nan)), np.nan)
    minus_di = np.where(atr.replace(0, np.nan)) != 0, 100 * (_wilder_smooth(minus_dm, window) / atr.replace(0, np.nan)), np.nan)
    dx = np.where((plus_di + minus_di).replace(0, np.nan)) != 0, 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)), np.nan)
    return _wilder_smooth(dx, window)

# ---------------------------------------------------------------------------
# §1 经典技术指标（TA-Lib / factor_dsl_np financial 风格）
#    canonical 多为大写；factor_engine 的 ts_* 别名见 _aliases.py
# ---------------------------------------------------------------------------

# canonical=ADX backend=pandas_numpy selected=ADX source=financial/__init__.py
@register_operator(name="ADX", category="financial", business_category="technical_signal", canonical="ADX", source="factor_dsl_np")
class ADX(SeriesOperator):
    """平均趋向指数"""
    metadata = OperatorMetadata(
        name="ADX",
        category="financial",
        description="平均趋向指数",
        examples=["ADX(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "ADX"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(np.nan, index=high.index, columns=high.columns)
        for col in high.columns:
            result[col] = _compute_dmi_adx(high[col], low[col], close[col], window)
        return result



# canonical=ADXR backend=pandas_numpy selected=ADXR source=financial/__init__.py
@register_operator(name="ADXR", category="financial", business_category="technical_signal", canonical="ADXR", source="factor_dsl_np")
class ADXR(SeriesOperator):
    """平滑平均趋向指数"""
    metadata = OperatorMetadata(
        name="ADXR",
        category="financial",
        description="平滑平均趋向指数",
        examples=["ADXR(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "ADX"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        w = max(2, int(window))
        result = pd.DataFrame(np.nan, index=high.index, columns=high.columns)
        for col in high.columns:
            adx = _compute_dmi_adx(high[col], low[col], close[col], w)
            adxr = np.where(2.0 != 0, (adx + adx.shift(w)) / 2.0, np.nan)
            result[col] = adxr.mask(adx.isna() | adx.shift(w).isna())
        return result



# canonical=AROON backend=pandas_numpy selected=AROON source=financial/__init__.py
@register_operator(name="AROON", category="financial", business_category="technical_signal", canonical="AROON", source="factor_dsl_np")
class Aroon(SeriesOperator):
    """Aroon指标（Aroon Up - Aroon Down）"""
    metadata = OperatorMetadata(
        name="AROON",
        category="financial",
        description="Aroon指标（Aroon Up - Aroon Down）",
        examples=["AROON(close, 25)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "Aroon"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 25, **kwargs) -> pd.DataFrame:
        w = max(1, int(window))
        roll = close.rolling(window=w + 1, min_periods=w + 1)
        # Audit P1-I: Aroon counts periods since the HIGHEST high / LOWEST low;
        # on a tie it must use the MOST RECENT extremum (``argmax[::-1]``), not
        # the first occurrence that ``np.argmax`` returns.
        aroon_up = 100 * roll.apply(lambda x: w - np.argmax(x[::-1]), raw=True) / w if w != 0 else np.nan
        aroon_down = 100 * roll.apply(lambda x: w - np.argmin(x[::-1]), raw=True) / w if w != 0 else np.nan
        return aroon_up - aroon_down



# canonical=AROON_down backend=pandas_numpy selected=AROON_down source=financial/__init__.py
@register_operator(name="AROON_down", category="financial", business_category="technical_signal", canonical="AROON_down", source="factor_dsl_np")
class AroonDown(SeriesOperator):
    """Aroon下降指标"""
    metadata = OperatorMetadata(
        name="AROON_down",
        category="financial",
        description="Aroon下降指标",
        examples=["AROON_down(close, 25)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "Aroon"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 25, **kwargs) -> pd.DataFrame:
        w = max(1, int(window))
        # Most recent extremum on ties (audit P1-I).
        return 100 * close.rolling(window=w + 1, min_periods=w + 1).apply(
            lambda x: w - np.argmin(x[::-1]), raw=True
        ) / w



# canonical=AROON_up backend=pandas_numpy selected=AROON_up source=financial/__init__.py
@register_operator(name="AROON_up", category="financial", business_category="technical_signal", canonical="AROON_up", source="factor_dsl_np")
class AroonUp(SeriesOperator):
    """Aroon上升指标"""
    metadata = OperatorMetadata(
        name="AROON_up",
        category="financial",
        description="Aroon上升指标",
        examples=["AROON_up(close, 25)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "Aroon"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 25, **kwargs) -> pd.DataFrame:
        w = max(1, int(window))
        # Most recent extremum on ties (audit P1-I).
        return 100 * close.rolling(window=w + 1, min_periods=w + 1).apply(
            lambda x: w - np.argmax(x[::-1]), raw=True
        ) / w



# canonical=ATR backend=pandas_numpy selected=ATR source=financial/__init__.py
@register_operator(name="ATR", category="financial", business_category="technical_signal", canonical="ATR", source="factor_dsl_np")
class ATR(SeriesOperator):
    """平均真实波幅"""
    metadata = OperatorMetadata(
        name="ATR",
        category="financial",
        description="平均真实波幅（SMA 平滑；Wilder 版见 ATR_WILDER）",
        examples=["ATR(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "ATR"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return tr.rolling(window=window, min_periods=1).mean()


@register_operator(
    name="ATR_WILDER",
    category="financial",
    business_category="technical_signal",
    canonical="ATR_WILDER",
    source="factor_dsl_np",
)
class ATRWilder(SeriesOperator):
    """Wilder 平均真实波幅"""
    metadata = OperatorMetadata(
        name="ATR_WILDER",
        category="financial",
        description="Wilder 平均真实波幅",
        examples=["ATR_WILDER(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "ATR", "pit_safe"],
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        return _compute_atr_wilder(high, low, close, window)



# canonical=BollingerBands backend=pandas_numpy selected=BollingerBands source=financial/__init__.py
@register_operator(name="BollingerBands", category="financial", business_category="technical_signal", canonical="BollingerBands", source="factor_dsl_np")
class BollingerBands(SeriesOperator):
    """布林带中轨（移动平均）"""
    metadata = OperatorMetadata(
        name="BollingerBands",
        category="financial",
        description="布林带中轨（移动平均）",
        examples=["BollingerBands(close, 20, 2)"],
        param_names=["price", "window", "std_dev"],
        return_type="series",
        tags=["financial", "technical", "Bollinger"]
    )

    def _calculate_series(self, price: pd.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pd.DataFrame:
        return price.rolling(window=window, min_periods=1).mean()



# canonical=BollingerLower backend=pandas_numpy selected=BollingerLower source=financial/__init__.py
@register_operator(name="BollingerLower", category="financial", business_category="technical_signal", canonical="BollingerLower", source="factor_dsl_np")
class BollingerLower(SeriesOperator):
    """布林带下轨"""
    metadata = OperatorMetadata(
        name="BollingerLower",
        category="financial",
        description="布林带下轨",
        examples=["BollingerLower(close, 20, 2)"],
        param_names=["x", "window", "std_dev"],
        return_type="series",
        tags=["financial", "technical", "Bollinger"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pd.DataFrame:
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return mean - std_dev * std



# canonical=BollingerUpper backend=pandas_numpy selected=BollingerUpper source=financial/__init__.py
@register_operator(name="BollingerUpper", category="financial", business_category="technical_signal", canonical="BollingerUpper", source="factor_dsl_np")
class BollingerUpper(SeriesOperator):
    """布林带上轨"""
    metadata = OperatorMetadata(
        name="BollingerUpper",
        category="financial",
        description="布林带上轨",
        examples=["BollingerUpper(close, 20, 2)"],
        param_names=["x", "window", "std_dev"],
        return_type="series",
        tags=["financial", "technical", "Bollinger"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, std_dev: float = 2, **kwargs) -> pd.DataFrame:
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return mean + std_dev * std



# canonical=CCI backend=pandas_numpy selected=CCI source=financial/__init__.py
@register_operator(name="CCI", category="financial", business_category="technical_signal", canonical="CCI", source="factor_dsl_np")
class CCI(SeriesOperator):
    """商品通道指数"""
    metadata = OperatorMetadata(
        name="CCI",
        category="financial",
        description="商品通道指数",
        examples=["CCI(high, low, close, 20)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "CCI"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        tp = np.where(3 != 0, (high + low + close) / 3, np.nan)
        sma = tp.rolling(window=window, min_periods=1).mean()
        mad = tp.rolling(window=window, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return np.where((0.015 * mad.replace(0, np.nan)) != 0, (tp - sma) / (0.015 * mad.replace(0, np.nan)), np.nan)



# canonical=DPO backend=pandas_numpy selected=DPO source=financial/__init__.py
@register_operator(name="DPO", category="financial", business_category="technical_signal", canonical="DPO", source="factor_dsl_np")
class DPO(SeriesOperator):
    """去趋势价格振荡器"""
    metadata = OperatorMetadata(
        name="DPO",
        category="financial",
        description="去趋势价格振荡器",
        examples=["DPO(close, 20)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "DPO"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        shift_period = (window // 2) + 1
        return close - close.rolling(window=window, min_periods=1).mean().shift(shift_period)



# canonical=KAMA backend=pandas_numpy selected=KAMA source=financial/__init__.py
@register_operator(name="KAMA", category="financial", business_category="technical_signal", canonical="KAMA", source="factor_dsl_np")
class KAMA(SeriesOperator):
    """考夫曼自适应移动平均"""
    metadata = OperatorMetadata(
        name="KAMA",
        category="financial",
        description="考夫曼自适应移动平均",
        examples=["KAMA(close, 10)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "KAMA"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 10, **kwargs) -> pd.DataFrame:
        missing_policy = kwargs.get("missing_policy", "interrupt")
        max_gap = kwargs.get("max_gap", 0)
        fast_sc = np.where((2 + 1) != 0, 2 / (2 + 1), np.nan)
        slow_sc = np.where((30 + 1) != 0, 2 / (30 + 1), np.nan)
        direction = (close - close.shift(window)).abs()
        volatility = close.diff().abs().rolling(window=window, min_periods=1).sum()
        er = direction / volatility.replace(0, np.nan) if volatility.replace(0, np.nan) > 1e-10 else np.nan
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result = close.copy()
        gap_max = max(int(max_gap), 0)
        for col in close.columns:
            kama_vals = close[col].values.copy()
            sc_vals = sc[col].values if isinstance(sc, pd.DataFrame) else sc.values
            gap = 0
            for i in range(1, len(kama_vals)):
                if not np.isfinite(kama_vals[i]):
                    # Audit 13.6 unified missing-state policy: a missing price
                    # must not silently bridge a suspension / data gap.
                    # ``interrupt`` (default) -> NaN and break the recursion
                    # (next finite bar re-seeds); ``carry`` bridges at most
                    # ``max_gap`` bars with the last finite value, then
                    # interrupts.
                    gap += 1
                    prev = kama_vals[i - 1]
                    if missing_policy == "carry" and gap <= gap_max and np.isfinite(prev):
                        kama_vals[i] = prev
                    else:
                        kama_vals[i] = np.nan
                    continue
                gap = 0
                prev = kama_vals[i - 1] if np.isfinite(kama_vals[i - 1]) else kama_vals[i]
                kama_vals[i] = prev + sc_vals[i] * (kama_vals[i] - prev)
            result[col] = kama_vals
        return result



# canonical=MACD backend=pandas_numpy selected=MACD source=financial/__init__.py
@register_operator(name="MACD", category="financial", business_category="technical_signal", canonical="MACD", source="factor_dsl_np")
class MACD(SeriesOperator):
    """MACD指标"""
    metadata = OperatorMetadata(
        name="MACD",
        category="financial",
        description="MACD指标 (快线-慢线)",
        examples=["MACD(close, 12, 26, 9)"],
        param_names=["x", "fast", "slow", "signal"],
        return_type="series",
        tags=["financial", "technical", "MACD"]
    )

    def _calculate_series(self, x: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
        ema_fast = x.ewm(span=fast, adjust=False).mean()
        ema_slow = x.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line

# aliases: ts_macd



# canonical=MACD_hist backend=pandas_numpy selected=MACD_hist source=financial/__init__.py
@register_operator(name="MACD_hist", category="financial", business_category="technical_signal", canonical="MACD_hist", source="factor_dsl_np")
class MACDHist(SeriesOperator):
    """MACD柱状图（MACD线 - 信号线）"""
    metadata = OperatorMetadata(
        name="MACD_hist",
        category="financial",
        description="MACD柱状图（MACD线 - 信号线）",
        examples=["MACD_hist(close, 12, 26, 9)"],
        param_names=["price", "fast", "slow", "signal"],
        return_type="series",
        tags=["financial", "technical", "MACD"]
    )

    def _calculate_series(self, price: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line



# canonical=MACD_line backend=pandas_numpy selected=MACD_line source=financial/__init__.py
@register_operator(name="MACD_line", category="financial", business_category="technical_signal", canonical="MACD_line", source="factor_dsl_np")
class MACDLine(SeriesOperator):
    """MACD线（快线EMA - 慢线EMA）"""
    metadata = OperatorMetadata(
        name="MACD_line",
        category="financial",
        description="MACD线（快线EMA - 慢线EMA）",
        examples=["MACD_line(close, 12, 26)"],
        param_names=["price", "fast", "slow", "signal"],
        return_type="series",
        tags=["financial", "technical", "MACD"],
        # NEW-040/042: the historical ``MACD(price, fast, slow, signal)`` call
        # form is an ALIAS of this canonical (the kernel computes fast-slow only;
        # ``signal`` is a dead knob of the old composite MACD).  It is declared
        # here as a non-searchable compatibility parameter so historical 4-arg
        # calls bind + validate instead of being silently dropped, and the
        # search grammar never treats it as an alpha dimension.
        param_specs={
            "signal": ParamSpec(
                dtype=int, min=1, searchable=False,
                param_role=ParamRole.POLICY, default=9,
            ),
        },
    )

    def _calculate_series(self, price: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
        # ``signal`` is a declared compatibility no-op (NEW-040): the MACD LINE
        # is fast-EMA minus slow-EMA; the signal line is a separate output
        # (``MACD_signal``).  Historical ``MACD(x,12,26,9)`` factors alias here.
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow



# canonical=MACD_signal backend=pandas_numpy selected=MACD_signal source=financial/__init__.py
@register_operator(name="MACD_signal", category="financial", business_category="technical_signal", canonical="MACD_signal", source="factor_dsl_np")
class MACDSignal(SeriesOperator):
    """MACD信号线"""
    metadata = OperatorMetadata(
        name="MACD_signal",
        category="financial",
        description="MACD信号线",
        examples=["MACD_signal(close, 12, 26, 9)"],
        param_names=["x", "fast", "slow", "signal"],
        return_type="series",
        tags=["financial", "technical", "MACD"]
    )

    def _calculate_series(self, x: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
        ema_fast = x.ewm(span=fast, adjust=False).mean()
        ema_slow = x.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        return macd_line.ewm(span=signal, adjust=False).mean()



# canonical=MOM backend=pandas_numpy selected=MOM source=financial/__init__.py
@register_operator(name="MOM", category="financial", business_category="technical_signal", canonical="MOM", source="factor_dsl_np")
class MOM(SeriesOperator):
    """动量指标"""
    metadata = OperatorMetadata(
        name="MOM",
        category="financial",
        description="动量指标",
        examples=["MOM(close, 10)"],
        param_names=["price", "window"],
        return_type="series",
        tags=["financial", "technical", "momentum"]
    )

    def _calculate_series(self, price: pd.DataFrame, window: int = 10, **kwargs) -> pd.DataFrame:
        from cleaned_operators._causal import causal_lag

        return price - causal_lag(price, int(window))



# canonical=OBV backend=pandas_numpy selected=OBV source=financial/__init__.py
@register_operator(name="OBV", category="financial", business_category="technical_signal", canonical="OBV", source="factor_dsl_np")
class OBV(SeriesOperator):
    """能量潮"""
    metadata = OperatorMetadata(
        name="OBV",
        category="financial",
        description="能量潮",
        examples=["OBV(close, volume)"],
        param_names=["price", "volume"],
        return_type="series",
        tags=["financial", "technical", "OBV"]
    )

    def _calculate_series(self, price: pd.DataFrame, volume: pd.DataFrame, **kwargs) -> pd.DataFrame:
        direction = np.sign(price.diff()).fillna(0)
        return (direction * volume).cumsum()



# canonical=ROC backend=pandas_numpy selected=ROC source=financial/__init__.py
@register_operator(name="ROC", category="financial", business_category="technical_signal", canonical="ROC", source="factor_dsl_np")
class ROC(SeriesOperator):
    """变动率指标"""
    metadata = OperatorMetadata(
        name="ROC",
        category="financial",
        description="变动率指标",
        examples=["ROC(close, 10)"],
        param_names=["price", "window"],
        return_type="series",
        tags=["financial", "technical", "ROC"]
    )

    def _calculate_series(self, price: pd.DataFrame, window: int = 10, **kwargs) -> pd.DataFrame:
        from cleaned_operators._causal import causal_lag

        prev = causal_lag(price, int(window))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(prev - 1.0) * 100.0 != 0, (price / prev - 1.0) * 100.0, np.nan)
        return out.where(prev.notna() & (prev != 0))



# canonical=RSI backend=pandas_numpy selected=RSI source=financial/__init__.py
@register_operator(name="RSI", category="financial", business_category="technical_signal", canonical="RSI", source="factor_dsl_np")
class RSI(SeriesOperator):
    """相对强弱指数"""
    metadata = OperatorMetadata(
        name="RSI",
        category="financial",
        description="相对强弱指数（SMA 平滑；Wilder 版见 RSI_WILDER）",
        examples=["RSI(close, 14)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "technical", "RSI"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        delta = x.diff()
        gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=1).mean()
        rs = np.where(loss.replace(0, np.nan) != 0, gain / loss.replace(0, np.nan), np.nan)
        rsi = np.where((1 + rs)) != 0, 100 - (100 / (1 + rs)), np.nan)
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
        rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
        return rsi


@register_operator(
    name="RSI_WILDER",
    category="financial",
    business_category="technical_signal",
    canonical="RSI_WILDER",
    source="factor_dsl_np",
)
class RSIWilder(SeriesOperator):
    """Wilder 相对强弱指数"""
    metadata = OperatorMetadata(
        name="RSI_WILDER",
        category="financial",
        description="Wilder 相对强弱指数",
        examples=["RSI_WILDER(close, 14)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "technical", "RSI", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        return _compute_rsi_wilder(x, window)

# aliases: ts_rsi



# canonical=StochasticD backend=pandas_numpy selected=StochasticD source=financial/__init__.py
@register_operator(name="StochasticD", category="financial", business_category="technical_signal", canonical="StochasticD", source="factor_dsl_np")
class StochasticD(SeriesOperator):
    """随机指标%D（%K的移动平均）"""
    metadata = OperatorMetadata(
        name="StochasticD",
        category="financial",
        description="随机指标%D（%K的移动平均）",
        examples=["StochasticD(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "stochastic"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        lowest = low.rolling(window=window, min_periods=1).min()
        highest = high.rolling(window=window, min_periods=1).max()
        k = np.where((highest - lowest).replace(0, np.nan) != 0, 100 * (close - lowest) / (highest - lowest).replace(0, np.nan), np.nan)
        return k.rolling(window=3, min_periods=1).mean()



# canonical=StochasticK backend=pandas_numpy selected=StochasticK source=financial/__init__.py
@register_operator(name="StochasticK", category="financial", business_category="technical_signal", canonical="StochasticK", source="factor_dsl_np")
class StochasticK(SeriesOperator):
    """随机指标%K"""
    metadata = OperatorMetadata(
        name="StochasticK",
        category="financial",
        description="随机指标%K",
        examples=["StochasticK(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "stochastic"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        lowest = low.rolling(window=window, min_periods=1).min()
        highest = high.rolling(window=window, min_periods=1).max()
        return np.where((highest - lowest).replace(0, np.nan) != 0, 100 * (close - lowest) / (highest - lowest).replace(0, np.nan), np.nan)



# canonical=TRIX backend=pandas_numpy selected=TRIX source=financial/__init__.py
@register_operator(name="TRIX", category="financial", business_category="technical_signal", canonical="TRIX", source="factor_dsl_np")
class TRIX(SeriesOperator):
    """三重指数平滑移动平均变化率"""
    metadata = OperatorMetadata(
        name="TRIX",
        category="financial",
        description="三重指数平滑移动平均变化率",
        examples=["TRIX(close, 12)"],
        param_names=["close", "window"],
        return_type="series",
        tags=["financial", "technical", "TRIX"]
    )

    def _calculate_series(self, close: pd.DataFrame, window: int = 12, **kwargs) -> pd.DataFrame:
        ema1 = close.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        return np.where(ema3.shift(1).replace(0, np.nan) * 100 != 0, (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan) * 100, np.nan)



# canonical=WilliamsR backend=pandas_numpy selected=WilliamsR source=financial/__init__.py
@register_operator(name="WilliamsR", category="financial", business_category="technical_signal", canonical="WilliamsR", source="factor_dsl_np")
class WilliamsR(SeriesOperator):
    """威廉指标%R"""
    metadata = OperatorMetadata(
        name="WilliamsR",
        category="financial",
        description="威廉指标%R",
        examples=["WilliamsR(high, low, close, 14)"],
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["financial", "technical", "Williams"]
    )

    def _calculate_series(self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
        highest = high.rolling(window=window, min_periods=1).max()
        lowest = low.rolling(window=window, min_periods=1).min()
        return np.where((highest - lowest).replace(0, np.nan) != 0, -100 * (highest - close) / (highest - lowest).replace(0, np.nan), np.nan)



# ---------------------------------------------------------------------------
# §2 信号 / 条件算子（factor_dsl_np signal；含 trade_when、where 等 WQ 风格）
#    注意：factor_engine 的 bucket / hump / ts_step 尚未迁入本模块
# ---------------------------------------------------------------------------

# 重复实现：见 clip（elementwise_math.py）；dedupe 注销
# @register_operator(name="clamp", ...)
class Clamp(SeriesOperator):
    """限制范围"""

    metadata = OperatorMetadata(
        name="clamp",
        category="signal",
        description="限制x在[a, b]范围内",
        examples=["clamp(value, 0, 1)"],
        param_names=["x", "a", "b"],
        return_type="series",
        tags=["signal", "clamp"]
    )

    def _calculate_series(self, x: pd.DataFrame, a: float = 0, b: float = 1, **kwargs) -> pd.DataFrame:
        return x.clip(lower=a, upper=b)



# canonical=hump_decay backend=pandas_numpy selected=hump_decay source=signal/__init__.py
@register_operator(name="hump_decay", category="signal", business_category="technical_signal", canonical="hump_decay", source="factor_dsl_np")
class HumpDecay(SeriesOperator):
    """阈值衰减：仅当变化量绝对值超过hump时才更新值"""
    metadata = OperatorMetadata(
        name="hump_decay",
        category="signal",
        description="阈值衰减：仅当变化量绝对值超过hump时才更新值",
        examples=["hump_decay(close, 0.05)"],
        param_names=["x", "hump"],
        return_type="series",
        tags=["signal", "decay"]
    )

    def _calculate_series(self, x: pd.DataFrame, hump: float = 0.05, **kwargs) -> pd.DataFrame:
        result = x.copy()
        cols = x.columns if isinstance(x, pd.DataFrame) else [x.name] if hasattr(x, 'name') else None
        for col in x.columns:
            prev = x[col].iloc[0]
            for i in range(1, len(x)):
                curr = x[col].iloc[i]
                if pd.isna(curr):
                    result.iloc[i, result.columns.get_loc(col)] = prev
                elif pd.isna(prev):
                    prev = curr
                else:
                    change = abs(curr - prev)
                    if change > hump:
                        prev = curr
                    else:
                        result.iloc[i, result.columns.get_loc(col)] = prev
        return result



# canonical=if_else backend=pandas_numpy selected=if_else source=signal/__init__.py
def _truthy_condition(condition: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """非空且非零为真；NULL/NaN 视为 false（与 Polars/SQL fast path 一致）。"""
    return condition.notna() & (condition != 0)


@register_operator(name="if_else", category="signal", business_category="technical_signal", canonical="if_else", source="factor_dsl_np")
class IfElse(SeriesOperator):
    """条件选择"""

    metadata = OperatorMetadata(
        name="if_else",
        category="signal",
        description="条件选择：condition为真返回v1，否则返回v2",
        examples=["if_else(close > open, 1, -1)"],
        param_names=["condition", "v1", "v2"],
        return_type="series",
        tags=["signal", "conditional"]
    )

    def _calculate_series(self, condition, v1=1, v2=0, **kwargs) -> pd.DataFrame:
        truthy = _truthy_condition(condition)
        if isinstance(condition, pd.DataFrame):
            if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
                return v1.where(truthy, v2)
            result = pd.DataFrame(
                np.where(truthy.values, v1, v2),
                index=condition.index,
                columns=condition.columns,
            )
            return result
        if isinstance(v1, pd.Series) and isinstance(v2, pd.Series):
            return v1.where(truthy, v2)
        return np.where(truthy, v1, v2)



# canonical=ifnan backend=pandas_numpy selected=ifnan source=signal/__init__.py
@register_operator(name="ifnan", category="signal", business_category="technical_signal", canonical="ifnan", source="factor_dsl_np")
class IfNaN(SeriesOperator):
    """NaN替换"""

    metadata = OperatorMetadata(
        name="ifnan",
        category="signal",
        description="如果x为NaN则返回default",
        examples=["ifnan(close, prev(close))"],
        param_names=["x", "default"],
        return_type="series",
        tags=["signal", "nan"]
    )

    def _calculate_series(self, x: pd.DataFrame, default=0, **kwargs) -> pd.DataFrame:
        return x.fillna(default)



# canonical=is_finite backend=pandas_numpy selected=is_finite source=signal/__init__.py
@register_operator(name="is_finite", category="signal", business_category="technical_signal", canonical="is_finite", source="factor_dsl_np")
class IsFinite(SeriesOperator):
    """判断有限"""

    metadata = OperatorMetadata(
        name="is_finite",
        category="signal",
        description="判断是否为有限数",
        examples=["is_finite(value)"],
        param_names=["x"],
        return_type="series",
        tags=["signal", "check"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from backend.elementwise_semantics import is_finite_pandas

        return is_finite_pandas(x)



# canonical=saturate backend=pandas_numpy selected=saturate source=signal/__init__.py
@register_operator(name="saturate", category="signal", business_category="technical_signal", canonical="saturate", source="factor_dsl_np")
class Saturate(SeriesOperator):
    """饱和"""

    metadata = OperatorMetadata(
        name="saturate",
        category="signal",
        description="限制x在[0, 1]范围内",
        examples=["saturate(Return)"],
        param_names=["x"],
        return_type="series",
        tags=["signal", "saturate"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.clip(lower=0, upper=1)



# canonical=signed_log backend=pandas_numpy selected=signed_log source=signal/__init__.py
@register_operator(name="signed_log", category="signal", business_category="technical_signal", canonical="signed_log", source="factor_dsl_np")
class SignedLog(SeriesOperator):
    """符号对数"""

    metadata = OperatorMetadata(
        name="signed_log",
        category="signal",
        description="符号对数：sign(x) * log(|x|)",
        examples=["signed_log(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["signal", "transform", "log"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sign(x) * np.log(x.abs() + 1e-10)



# canonical=signed_power backend=pandas_numpy selected=signed_power source=signal/__init__.py
@register_operator(name="signed_power", category="signal", business_category="technical_signal", canonical="signed_power", source="factor_dsl_np")
class SignedPower(SeriesOperator):
    """符号保持幂"""

    metadata = OperatorMetadata(
        name="signed_power",
        category="signal",
        description="符号保持幂：sign(x) * |x|^c",
        examples=["signed_power(zscore, 2)"],
        param_names=["x", "c"],
        return_type="series",
        tags=["signal", "transform"]
    )

    def _calculate_series(self, x: pd.DataFrame, c: float = 2, **kwargs) -> pd.DataFrame:
        return np.sign(x) * (x.abs() ** c)



# canonical=trade_when backend=pandas_numpy selected=trade_when source=signal/__init__.py
@register_operator(name="trade_when", category="signal", business_category="technical_signal", canonical="trade_when", source="factor_dsl_np")
class TradeWhen(SeriesOperator):
    """条件信号：condition为真时返回signal，否则返回fallback"""
    metadata = OperatorMetadata(
        name="trade_when",
        category="signal",
        description="条件信号：condition为真时返回signal，否则返回fallback",
        examples=["trade_when(volume > 1000, close, open)"],
        param_names=["condition", "signal", "fallback"],
        return_type="series",
        tags=["signal", "conditional"]
    )

    def _calculate_series(self, condition, signal, fallback=0, **kwargs) -> pd.DataFrame:
        if isinstance(condition, pd.DataFrame):
            cond_bool = condition.astype(bool)
            if isinstance(signal, pd.DataFrame) and isinstance(fallback, pd.DataFrame):
                return signal.where(cond_bool, fallback)
            result = pd.DataFrame(
                np.where(cond_bool, signal, fallback),
                index=condition.index, columns=condition.columns
            )
            return result
        return signal * condition + fallback * (1 - condition)



# canonical=vp_weighted_price backend=pandas_numpy selected=vp_weighted_price source=signal/vpmacd_ops.py
@register_operator(name="vp_weighted_price", category="signal", business_category="technical_signal", canonical="vp_weighted_price", source="factor_dsl_np")
class VPWeightedPrice(SeriesOperator):
    """量价加权价格"""

    metadata = OperatorMetadata(
        name="vp_weighted_price",
        category="signal",
        description="计算量价加权价格 P*t = Σ(Pi × Volumei × σi × ri) / ΣVolumei",
        examples=[
            "vp_weighted_price(close, volume)",
            "vp_weighted_price(close, volume, open, high, low)"
        ],
        param_names=["close", "volume", "open", "high", "low"],
        return_type="series",
        tags=["signal", "price", "volume", "weighted"]
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        open_: pd.DataFrame = None,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None,
        **kwargs
    ) -> pd.DataFrame:
        if high is not None and low is not None:
            amplitude = high - low
        else:
            amplitude = close.diff().abs().fillna(0)
        
        sigma_i = np.where(close.replace(0, np.nan) != 0, amplitude / close.replace(0, np.nan), np.nan)
        sigma_i = sigma_i.fillna(0)
        
        if open_ is not None:
            price_range = open_ - close
            ri = np.where(amplitude.replace(0, np.nan) != 0, price_range.abs() / amplitude.replace(0, np.nan), np.nan)
        else:
            ret = close.pct_change(fill_method=None).fillna(0)
            ri = (ret > 0).astype(float) * 0.7 + (ret < 0).astype(float) * 0.3
        
        ri = ri.fillna(0.5)
        
        weights = volume * sigma_i * ri
        
        weighted_price = np.where(\ != 0, (close * weights).rolling(window=20, min_periods=5).sum() / \, np.nan)
                         weights.rolling(window=20, min_periods=5).sum()
        
        return weighted_price.fillna(close)



# canonical=vpmacd backend=pandas_numpy selected=vpmacd source=signal/vpmacd_ops.py
@register_operator(name="vpmacd", category="signal", business_category="technical_signal", canonical="vpmacd", source="factor_dsl_np")
class VPMACD(SeriesOperator):
    """VP-MACD（量价调整MACD）"""

    metadata = OperatorMetadata(
        name="vpmacd",
        category="signal",
        description="VP-MACD因子：将成交量、波动率、日内K线结构嵌入MACD指标，搭配灵敏度参数λ",
        examples=[
            "vpmacd(close, volume)",
            "vpmacd(close, volume, open, high, low, 0.9)"
        ],
        param_names=["close", "volume", "open", "high", "low", "lambda_param"],
        return_type="series",
        tags=["signal", "macd", "volume", "price", "vpmacd"]
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        open_: pd.DataFrame = None,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None,
        lambda_param: float = 0.9,
        **kwargs
    ) -> pd.DataFrame:
        if high is not None and low is not None:
            amplitude = high - low
        else:
            amplitude = close.diff().abs().fillna(0)
        
        sigma_i = np.where(close.replace(0, np.nan) != 0, amplitude / close.replace(0, np.nan), np.nan)
        sigma_i = sigma_i.fillna(0)
        
        if open_ is not None:
            price_range = open_ - close
            ri = np.where(amplitude.replace(0, np.nan) != 0, price_range.abs() / amplitude.replace(0, np.nan), np.nan)
        else:
            ret = close.pct_change(fill_method=None).fillna(0)
            ri = (ret > 0).astype(float) * 0.7 + (ret < 0).astype(float) * 0.3
        
        ri = ri.fillna(0.5)
        
        weights = volume * sigma_i * ri
        
        weighted_price = np.where(\ != 0, (close * weights).rolling(window=20, min_periods=5).sum() / \, np.nan)
                         weights.rolling(window=20, min_periods=5).sum()
        weighted_price = weighted_price.fillna(close)
        
        ema_fast = weighted_price.ewm(span=12, adjust=False).mean()
        ema_slow = weighted_price.ewm(span=26, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        adjusted_signal = lambda_param * signal_line
        
        return macd_line - adjusted_signal



# canonical=vpmacd_signal backend=pandas_numpy selected=vpmacd_signal source=signal/vpmacd_ops.py
@register_operator(name="vpmacd_signal", category="signal", business_category="technical_signal", canonical="vpmacd_signal", source="factor_dsl_np")
class VPMACDSignal(SeriesOperator):
    """VP-MACD交易信号"""

    metadata = OperatorMetadata(
        name="vpmacd_signal",
        category="signal",
        description="VP-MACD交易信号：1=买入信号, -1=卖出信号, 0=持有",
        examples=[
            "vpmacd_signal(close, volume)",
            "vpmacd_signal(close, volume, open, high, low, 0.9)"
        ],
        param_names=["close", "volume", "open", "high", "low", "lambda_param"],
        return_type="series",
        tags=["signal", "macd", "volume", "price", "vpmacd", "trading_signal"]
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        open_: pd.DataFrame = None,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None,
        lambda_param: float = 0.9,
        **kwargs
    ) -> pd.DataFrame:
        if high is not None and low is not None:
            amplitude = high - low
        else:
            amplitude = close.diff().abs().fillna(0)
        
        sigma_i = np.where(close.replace(0, np.nan) != 0, amplitude / close.replace(0, np.nan), np.nan)
        sigma_i = sigma_i.fillna(0)
        
        if open_ is not None:
            price_range = open_ - close
            ri = np.where(amplitude.replace(0, np.nan) != 0, price_range.abs() / amplitude.replace(0, np.nan), np.nan)
        else:
            ret = close.pct_change(fill_method=None).fillna(0)
            ri = (ret > 0).astype(float) * 0.7 + (ret < 0).astype(float) * 0.3
        
        ri = ri.fillna(0.5)
        
        weights = volume * sigma_i * ri
        
        weighted_price = np.where(\ != 0, (close * weights).rolling(window=20, min_periods=5).sum() / \, np.nan)
                         weights.rolling(window=20, min_periods=5).sum()
        weighted_price = weighted_price.fillna(close)
        
        ema_fast = weighted_price.ewm(span=12, adjust=False).mean()
        ema_slow = weighted_price.ewm(span=26, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        adjusted_signal = lambda_param * signal_line
        
        golden_cross = (macd_line > adjusted_signal) & (macd_line.shift(1) <= adjusted_signal.shift(1))
        death_cross = (macd_line < adjusted_signal) & (macd_line.shift(1) >= adjusted_signal.shift(1))
        
        signal = pd.DataFrame(0, index=close.index, columns=close.columns)
        signal[golden_cross] = 1
        signal[death_cross] = -1
        
        return signal



# canonical=where backend=pandas_numpy selected=where source=signal/__init__.py

# helper for where
class IfElse(SeriesOperator):
    """条件选择"""

    metadata = OperatorMetadata(
        name="if_else",
        category="signal",
        description="条件选择：condition为真返回v1，否则返回v2",
        examples=["if_else(close > open, 1, -1)"],
        param_names=["condition", "v1", "v2"],
        return_type="series",
        tags=["signal", "conditional"]
    )

    def _calculate_series(self, condition, v1=1, v2=0, **kwargs) -> pd.DataFrame:
        truthy = _truthy_condition(condition)
        if isinstance(condition, pd.DataFrame):
            if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
                return v1.where(truthy, v2)
            result = pd.DataFrame(
                np.where(truthy.values, v1, v2),
                index=condition.index,
                columns=condition.columns,
            )
            return result
        if isinstance(v1, pd.Series) and isinstance(v2, pd.Series):
            return v1.where(truthy, v2)
        return np.where(truthy, v1, v2)

@register_operator(name="where", category="signal", business_category="technical_signal", canonical="where", source="factor_dsl_np")
class Where(IfElse):
    """条件选择（if_else的别名）"""

    metadata = OperatorMetadata(
        name="where",
        category="signal",
        description="条件选择（与if_else相同）",
        examples=["where(volume > 1000, 1, 0)"],
        param_names=["condition", "v1", "v2"],
        return_type="series",
        tags=["signal", "conditional"]
    )

# aliases: IIF, WHERE, if, iif

