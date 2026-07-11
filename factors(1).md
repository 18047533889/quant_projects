# CogAlpha Factor Catalog

Incremental export of elite/qualified factors (formula, rationale, dependency names, code only).

---

## Factor 1

**Formula:** `-regime * momentum, where regime = 1 if rolling_std(ret,20) > rolling_median(rolling_std(ret,20),60) else -1, momentum = 12-day return`

**Rationale:** Volatility regime indicator based on rolling 20-day return standard deviation compared to its 60-day median. In high-volatility states (above median), the factor takes a contrarian stance against 12-day momentum, anticipating mean reversion. In low-volatility states, it follows the trend. This captures state-dependent market phases as described in AgentVolatilityRegime focus.

**Tools:** library_functions=np.where

```python
def factor_volatility_regime_momentum(df):
    close = df['close']
    result = -np.where(close.pct_change().rolling(20).std() > close.pct_change().rolling(20).std().rolling(60).median(), 1, -1) * close.pct_change(12)
    result = result.fillna(0)
    result.name = 'factor_volatility_regime_momentum'
    return result
```

---

## Factor 11

**Formula:** `tanh((close - SMA20(close,20)) / (volatility(close,20) * SMA20(close,20)))`

**Rationale:** Mutates parent factor_nonlinear_deviation by scaling the close deviation from SMA20 by rolling volatility of returns. This adjusts for market regime changes, potentially improving IC and mutual information while preserving the bounded nonlinearity and rank stability from the parent. The hyperbolic tangent compresses outliers.

**Tools:** library_functions=np.tanh

```python
def factor_volatility_scaled_deviation(df):
    df_copy = df.copy()
    sma20 = df_copy['close'].rolling(window=20, min_periods=20).mean()
    deviation = df_copy['close'] - sma20
    ret = df_copy['close'].pct_change()
    vol20 = ret.rolling(window=20, min_periods=20).std()
    scaled_dev = deviation / (vol20 * sma20 + 1e-8)
    factor = np.tanh(scaled_dev)
    df_copy['factor_volatility_scaled_deviation'] = factor
    return df_copy['factor_volatility_scaled_deviation']
```

---

## Factor 12

**Formula:** `factor = (short_ma_abs_log_ret / long_ma_abs_log_ret - 1) * (1 + drawdown_depth * drawdown_duration) where drawdown_depth = (close - cummax(close))/cummax(close), drawdown_duration = consecutive days below cummax`

**Rationale:** Combines multi-scale roughness and drawdown geometry to capture volatility amplification during drawdown regimes. The interaction term intensifies the roughness signal when drawdowns are deep and prolonged, potentially identifying panic-driven movements that precede reversals or further declines. This preserves the parent strengths of scale-sensitive turbulence and drawdown severity.

**Tools:** library_functions=np.log, np.maximum.accumulate

```python
def factor_roughness_drawdown_interaction(df):
    df_copy = df.copy()
    close = df_copy['close']
    # roughness ratio
    log_ret = np.log(close / close.shift(1))
    short_ma = log_ret.abs().rolling(5, min_periods=5).mean()
    long_ma = log_ret.abs().rolling(20, min_periods=20).mean()
    ratio = short_ma / long_ma - 1
    # drawdown geometry
    rolling_max = np.maximum.accumulate(close)
    depth = (close - rolling_max) / rolling_max
    is_dd = close < rolling_max
    dd_group = (~is_dd).cumsum()
    duration = df_copy.groupby(dd_group).cumcount() + 1
    duration = duration.where(is_dd, 0)
    dd_factor = -depth * duration
    # interaction
    factor = ratio * (1 + dd_factor)
    factor.name = 'factor_roughness_drawdown_interaction'
    return factor
```

---

## Factor 24

**Formula:** `-MA(|d(r)|, 10)`

**Rationale:** This factor measures the temporal consistency of daily returns by averaging the absolute change in consecutive returns over a 10-day rolling window. A smaller absolute change indicates more persistent return direction, which aligns with the Stability & Regime-Gating layer's focus on smoothness and persistence. The negative sign inverts the scale so higher factor values correspond to greater persistence.

```python
def factor_persistence(df):
    df_copy = df.copy()
    returns = df_copy["close"].pct_change()
    abs_return_diff = returns.diff().abs()
    smoothness = -abs_return_diff.rolling(window=10, min_periods=1).mean()
    df_copy["factor_persistence"] = smoothness.fillna(0)
    return df_copy["factor_persistence"]
```

---

## Factor 25

**Formula:** `-EWMA(|d(r)|, span=10)`

**Rationale:** This factor measures temporal consistency of returns using an exponentially weighted moving average of absolute consecutive return changes. The EWMA places more weight on recent observations, adapting faster to regime changes while still capturing persistence. Negative sign inverts so higher values indicate more persistent return direction.

```python
def factor_persistence_ewma(df):
    df_copy = df.copy()
    returns = df_copy["close"].pct_change()
    abs_return_diff = returns.diff().abs()
    smoothness = -abs_return_diff.ewm(span=10, min_periods=1).mean()
    df_copy["factor_persistence_ewma"] = smoothness.fillna(0)
    return df_copy["factor_persistence_ewma"]
```

---

## Factor 26

**Formula:** `factor = (2*I(vol20<med_vol)-1) * mom * (volume/med_volume)`

**Rationale:** Extends volatility regime switching by weighting signals with relative volume to focus on liquid periods where momentum/contrarian effects are more reliable. Preserves the core state-dependent market phase detection while adding a liquidity filter to potentially improve IC and MI without sacrificing rank stability.

```python
def factor_vol_regime_volume(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol20 = ret.rolling(20).std()
    med_vol = vol20.rolling(60).median()
    mom = (df_copy['close'] - df_copy['close'].shift(10)) / df_copy['close'].shift(10)
    volume = df_copy['volume']
    med_volume = volume.rolling(60).median()
    vol_ratio = volume / med_volume
    regime = (vol20 < med_vol).astype(int)
    factor = (2 * regime - 1) * mom * vol_ratio
    df_copy['factor_vol_regime_volume'] = factor
    return df_copy['factor_vol_regime_volume']
```

---

## Factor 27

**Formula:** `rolling(5, median(abs(pct_change(close)))) / rolling(5, mean(log(volume+1)))`

**Rationale:** Mutated from factor_price_impact_5d by replacing mean with median for robustness and using log volume to reduce scale sensitivity. This preserves the core price impact hypothesis while targeting the bottleneck of low mutual information and improving rank stability.

**Tools:** library_functions=np.log

```python
def factor_price_impact_stable_5d(df):
    df_copy = df.copy()
    df_copy['ret'] = df_copy['close'].pct_change()
    df_copy['abs_ret'] = df_copy['ret'].abs()
    median_abs_ret = df_copy['abs_ret'].rolling(window=5, min_periods=3).median()
    log_volume = np.log(df_copy['volume'] + 1)
    avg_log_vol = log_volume.rolling(window=5, min_periods=3).mean()
    df_copy['factor_price_impact_stable_5d'] = median_abs_ret / avg_log_vol
    df_copy['factor_price_impact_stable_5d'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_copy['factor_price_impact_stable_5d']
```

---

## Factor 28

**Formula:** `OI * (vol_z * norm_range), where OI = sum_{i=t-19}^{t}(CLV_i * volume_i), CLV = (2*close - high - low)/(high - low), vol_z = (volume - rolling_mean_20)/rolling_std_20, norm_range = (high - low)/close`

**Rationale:** This factor multiplies cumulative order imbalance (directional pressure) by a volume abnormality signal (z-score times normalized range). Order imbalance alone captures buying/selling pressure, but its predictive power is enhanced when volume is abnormal and price range is wide, indicating informed trading. The product amplifies signals during high conviction periods while reducing noise during normal trading, potentially improving RankIC and MI.

**Tools:** library_functions=np.where

```python
def factor_imbalance_volume_interaction(df):
    """Combine order imbalance with volume abnormality to enhance predictive information."""
    df_copy = df.copy()
    clv = np.where(df_copy['high'] - df_copy['low'] != 0,
                   (2*df_copy['close'] - df_copy['high'] - df_copy['low']) / (df_copy['high'] - df_copy['low']),
                   0)
    money_flow = clv * df_copy['volume']
    order_imbalance = money_flow.rolling(window=20, min_periods=1).sum()
    vol_mean = df_copy['volume'].rolling(window=20, min_periods=10).mean()
    vol_std = df_copy['volume'].rolling(window=20, min_periods=10).std()
    vol_z = (df_copy['volume'] - vol_mean) / (vol_std + 1e-8)
    norm_range = (df_copy['high'] - df_copy['low']) / (df_copy['close'] + 1e-8)
    vol_abnormality = vol_z * norm_range
    df_copy['factor_imbalance_volume_interaction'] = order_imbalance * vol_abnormality
    return df_copy['factor_imbalance_volume_interaction']
```

---

## Factor 30

**Formula:** `OI_10 * (vol_ratio * norm_range), where OI_10 = sum_{i=t-9}^{t}(CLV_i * vol_i), vol_ratio = vol/vol_ma_10, norm_range = (high-low)/close`

**Rationale:** This mutation preserves the core idea of combining directional pressure (order imbalance) with volume abnormality, but uses shorter 10-day windows and a volume ratio instead of z-score to reduce noise and improve responsiveness, targeting improved RankIC and ICIR while maintaining interpretability.

**Tools:** library_functions=np.where

```python
def factor_imbalance_volume_mutation(df):
    df_copy = df.copy()
    clv = np.where(df_copy['high'] - df_copy['low'] != 0,
                   (2*df_copy['close'] - df_copy['high'] - df_copy['low']) / (df_copy['high'] - df_copy['low']),
                   0)
    money_flow = clv * df_copy['volume']
    order_imbalance = money_flow.rolling(window=10, min_periods=1).sum()
    vol_ma = df_copy['volume'].rolling(window=10, min_periods=5).mean()
    vol_ratio = df_copy['volume'] / (vol_ma + 1e-8)
    norm_range = (df_copy['high'] - df_copy['low']) / (df_copy['close'] + 1e-8)
    vol_abnormality = vol_ratio * norm_range
    df_copy['factor_imbalance_volume_mutation'] = order_imbalance * vol_abnormality
    return df_copy['factor_imbalance_volume_mutation']
```

---

## Factor 40

**Formula:** `regime = sign(vol_20 - med_vol_60); factor = regime * (close - EMA(close, 10))`

**Rationale:** This factor preserves the volatility-regime switching intuition of the parent by comparing 20-day rolling volatility to its 60-day median, but replaces the raw 10-day price change with a smoothed exponential moving average (EMA) of close. The EMA reduces noise and improves rank stability, targeting the RankIC bottleneck while maintaining the core regime-dependent momentum/mean-reversion hypothesis.

```python
def factor_vol_regime_smoothed_trend(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_20 = ret.rolling(20).std()
    med_vol_60 = vol_20.rolling(60).median()
    regime = ((vol_20 > med_vol_60).astype(int) * 2) - 1
    trend = df_copy["close"] - df_copy["close"].ewm(span=10, adjust=False).mean()
    df_copy["factor_vol_regime_smoothed_trend"] = regime * trend
    return df_copy["factor_vol_regime_smoothed_trend"]
```

---

## Factor 41

**Formula:** `factor = regime * (close - close_10) * (1 + 0.5 * vol_dev), where regime = sign(vol_20 - med_vol_60), vol_dev = volume / rolling_median(volume, 20) - 1`

**Rationale:** This factor combines volatility regime switching with volume median deviation. The base regime signal (momentum in high vol, contrarian in low vol) is strengthened when volume is above its median (confirming) and weakened when volume is low (suggesting noise). The volume adjustment aims to improve rank consistency and nonlinear information capture.

```python
def factor_vol_volume_regime(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_20 = ret.rolling(20).std()
    med_vol_60 = vol_20.rolling(60).median()
    regime = ((vol_20 > med_vol_60).astype(int) * 2) - 1
    price_change = df_copy["close"] - df_copy["close"].shift(10)
    vol_med_20 = df_copy["volume"].rolling(20).median()
    vol_dev = (df_copy["volume"] / vol_med_20) - 1
    factor = regime * price_change * (1 + 0.5 * vol_dev)
    df_copy["factor_vol_volume_regime"] = factor
    return df_copy["factor_vol_volume_regime"]
```

---

## Factor 43

**Formula:** `factor = regime * ret_10 * (1 + 0.5 * log(volume / med_vol_20)), where regime = sign(vol_20 - med_vol_60), ret_10 = 10-day percentage change`

**Rationale:** This factor mutates the parent by replacing absolute price change with a percentage return over 10 days for better cross-sectional comparability, and using log transformation on volume deviation to reduce outlier influence. The regime-switching component is preserved, targeting improved rank consistency (RankIC) and nonlinear information (MI) while maintaining the core economic intuition.

**Tools:** library_functions=np.log

```python
def factor_smooth_vol_regime(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_20 = ret.rolling(20).std()
    med_vol_60 = vol_20.rolling(60).median()
    regime = ((vol_20 > med_vol_60).astype(int) * 2) - 1
    ret_10 = df_copy['close'].pct_change(10)
    med_vol_20 = df_copy['volume'].rolling(20).median()
    vol_dev = np.log(df_copy['volume'] / med_vol_20)
    factor = regime * ret_10 * (1 + 0.5 * vol_dev)
    df_copy['factor_smooth_vol_regime'] = factor
    return df_copy['factor_smooth_vol_regime']
```

---

## Factor 45

**Formula:** `((EWMA(range,10)/EWMA(range,40)-1) * (EWMA(volume,10)/EWMA(volume,40)-1) * (EWMA(up_range,20)/EWMA(down_range,20))).shift(1)`

**Rationale:** This factor mutates the parent crossover_v1 by replacing simple moving averages with EWMA for smoother, more adaptive estimates of volatility and volume deviations. The asymmetric ratio is redefined using conditional range on up/down days instead of squared returns, reducing noise and improving rank stability. Outlier clipping at 5th and 95th percentiles further enhances robustness. The core economic intuition of combining volatility-volume interaction with asymmetry is preserved, targeting improved RankIC and MI.

```python
def factor_vol_volume_asym_ewma(df):
    df_copy = df.copy()
    range_ = df_copy['high'] - df_copy['low']
    # EWMA smooth range and volume
    range_ewma10 = range_.ewm(span=10, adjust=False).mean()
    range_ewma40 = range_.ewm(span=40, adjust=False).mean()
    vol_ratio = range_ewma10 / range_ewma40 - 1
    vol_ewma10 = df_copy['volume'].ewm(span=10, adjust=False).mean()
    vol_ewma40 = df_copy['volume'].ewm(span=40, adjust=False).mean()
    vol_change = vol_ewma10 / vol_ewma40 - 1
    # Asymmetric range: up range and down range using close direction
    ret = df_copy['close'].pct_change()
    up = ret > 0
    down = ret < 0
    # Compute rolling average range on up/down days
    up_range = range_.where(up, 0)
    down_range = range_.where(down, 0)
    # Use EWMA to smooth these conditional ranges
    up_range_ewma = up_range.ewm(span=20, adjust=False).mean()
    down_range_ewma = down_range.ewm(span=20, adjust=False).mean()
    # Avoid division by zero
    down_range_ewma = down_range_ewma.replace(0, np.nan)
    asym_ratio = up_range_ewma / down_range_ewma
    asym_ratio = asym_ratio.replace([np.inf, -np.inf], np.nan)
    # Combine components with shift to avoid future lookahead
    factor = vol_ratio.shift(1) * vol_change.shift(1) * asym_ratio.shift(1)
    # Winsorize extreme values at 5th and 95th percentile
    lower = factor.quantile(0.05)
    upper = factor.quantile(0.95)
    factor = factor.clip(lower, upper)
    df_copy['factor_vol_volume_asym_ewma'] = factor
    return df_copy['factor_vol_volume_asym_ewma']
```

---

## Factor 57

**Formula:** `factor = (close / lag(close) - 1) * (volume / rolling_mean(volume, 20) - 1) * 10000`

**Rationale:** Mutates the original price-volume coherence factor by using percentage price return instead of absolute price change, making the signal scale-invariant across stocks with different price levels. Volume confirmation mechanism is preserved. The multiplication by 10000 scales the factor to a typical range. This mutation targets improved cross-sectional IC while maintaining the robust RankICIR of the parent.

```python
def factor_volume_confirmed_return(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(1) - 1
    vol_ma = df_copy['volume'].rolling(window=20, min_periods=5).mean()
    vol_dev = df_copy['volume'] / vol_ma - 1
    df_copy['factor_volume_confirmed_return'] = ret * vol_dev * 10000
    return df_copy['factor_volume_confirmed_return']
```

---

## Factor 58

**Formula:** `Z(log(volume,20)) * (high-low) / SMA(high-low,20)`

**Rationale:** This factor combines volume abnormality detection (log-volume z-score relative to 20-day median) with volatility regime (range expansion/compression ratio). Multiplying the volume z-score by the range ratio amplifies signals when volatility is expanding and attenuates when compressing, reflecting that volume abnormality is more informative during high-volatility periods. This fusion preserves parent1's volume abnormality insight and parent2's range regime insight.

**Tools:** library_functions=np.isnan, np.log, np.where

```python
def factor_volume_abnormality_range_interaction(df):
    """Volume abnormality interacting with range expansion/compression regime."""
    df_copy = df.copy()
    log_vol = np.log(df_copy["volume"] + 1)
    roll_med = log_vol.rolling(window=20, min_periods=10).median()
    roll_std = log_vol.rolling(window=20, min_periods=10).std()
    vol_z = np.where(roll_std > 0, (log_vol - roll_med) / roll_std, 0.0)
    range_ = df_copy["high"] - df_copy["low"]
    avg_range = range_.rolling(window=20, min_periods=10).mean()
    range_ratio = (range_ + 1e-8) / (avg_range + 1e-8)
    factor = vol_z * range_ratio
    factor = np.where(np.isnan(factor), 0.0, factor)
    df_copy["factor_volume_abnormality_range_interaction"] = factor
    return df_copy["factor_volume_abnormality_range_interaction"]
```

---

## Factor 70

**Formula:** `factor = log(EWMA_up_vol / EWMA_down_vol) * rank_pct(high - low)`

**Rationale:** Mutates parent lag-response-001 by introducing log transformation of the up/down volume ratio to capture multiplicative asymmetry and nonlinear information, and rank-scaling the range to preserve the parent's strong RankIC/RankICIR. The log ratio enhances MI potential while the rank-normalized range maintains robust cross-sectional ordering. This targets improved MI without sacrificing rank stability.

**Tools:** library_functions=np.log, np.where

```python
def factor_adaptive_vol_volume_asym(df):
    df_copy = df.copy()
    # Log returns for volatility estimation
    log_ret = np.log(df_copy['close'] / df_copy['close'].shift(1))
    # Rolling 20-day volatility (std of log returns)
    vol = log_ret.rolling(window=20, min_periods=10).std()
    # Adaptive span: base 10, increase with volatility, bounded between 5 and 30
    vol_ratio = vol / vol.rolling(window=60, min_periods=20).mean()
    span = (10 * (1 + vol_ratio)).clip(5, 30)
    # Handle NaNs at the start
    span = span.fillna(10)
    # Up and down days
    up_day = df_copy['close'] >= df_copy['open']
    down_day = ~up_day
    # Volume on up/down days (zero otherwise)
    vol_up = df_copy['volume'].where(up_day, 0)
    vol_down = df_copy['volume'].where(down_day, 0)
    # Compute EWMA with adaptive span using ewm (span must be scalar, so we loop over each row - but vectorized alternative: use expanding window approach?)
    # Since span is a series, we need to compute EWMA for each row with different decay. This is not directly vectorizable with ewm.
    # Alternative: Use a fixed span of 10 and add a volatility modulation factor to the ratio.
    # Simpler mutation: use fixed span 10, but take log of ratio and scale by rank of range.
    # Let's simplify: keep span fixed at 10, but use log ratio and rank scale.
    span = 10
    alpha = 2.0 / (span + 1)
    ewma_up = vol_up.ewm(span=span, adjust=False).mean()
    ewma_down = vol_down.ewm(span=span, adjust=False).mean()
    # Avoid division by zero
    ratio = np.where(ewma_down > 0, ewma_up / ewma_down, 1.0)
    # Log ratio for nonlinearity
    log_ratio = np.log(ratio + 1e-10)
    # Range
    range_ = df_copy['high'] - df_copy['low']
    # Rank-normalize range to preserve rank stability
    range_rank = range_.rank(pct=True)
    factor = log_ratio * range_rank
    df_copy['factor_adaptive_vol_volume_asym'] = factor
    return df_copy['factor_adaptive_vol_volume_asym']
```

---

## Factor 71

**Formula:** `R(DD) + R(Dur) + R(Rec) * R(V/MA20_V)`

**Rationale:** Extends the drawdown geometry factor by incorporating volume-confirmed recovery. The parent's recovery slope is multiplied by relative volume rank so that only recoveries accompanied by elevated volume contribute strongly, introducing nonlinear interaction that can improve MI. Depth and duration components remain additive to preserve rank stability.

```python
def factor_drawdown_volume_complexity(df):
    """Mutated drawdown geometry with volume-confirmed recovery."""
    df_copy = df.copy()
    rolling_high = df_copy["close"].rolling(window=252, min_periods=20).max()
    drawdown = (rolling_high - df_copy["close"]) / rolling_high
    new_peak = df_copy["close"] >= rolling_high
    group = new_peak.cumsum()
    duration = new_peak.groupby(group).cumcount()
    recovery = -drawdown.diff(5)
    vol_ma = df_copy["volume"].rolling(window=20).mean()
    rel_volume = df_copy["volume"] / vol_ma
    rank_dd = drawdown.rank(pct=True)
    rank_dur = duration.rank(pct=True)
    rank_rec = recovery.rank(pct=True)
    rank_vol = rel_volume.rank(pct=True)
    confirmed_recovery = rank_rec * rank_vol
    df_copy["factor_drawdown_volume_complexity"] = rank_dd + rank_dur + confirmed_recovery
    return df_copy["factor_drawdown_volume_complexity"]
```

---

## Factor 72

**Formula:** `abs(z_score(log(volume),20)) * (high-low)/close * EWMA_up_vol/EWMA_down_vol`

**Rationale:** Combines volume abnormality (z-score of log volume over 20 days weighted by range ratio) with lagged volume asymmetry (ratio of up-day to down-day volume EWMA over 10 days). The product amplifies the asymmetry signal when volume is abnormal, capturing extreme volatility events with delayed volume adjustment to price direction. This leverages the rank stability of the abnormality component and the lagged response insight of the asymmetry component.

**Tools:** library_functions=np.abs, np.log, np.where

```python
def factor_abnormality_asymmetry(df):
    df_copy = df.copy()
    log_vol = np.log(df_copy['volume'] + 1)
    rolling_mean = log_vol.rolling(window=20).mean()
    rolling_std = log_vol.rolling(window=20).std()
    z_score = (log_vol - rolling_mean) / rolling_std
    abnormality = np.abs(z_score)
    range_ratio = (df_copy['high'] - df_copy['low']) / df_copy['close']
    vol_abnorm = abnormality * range_ratio
    up_day = df_copy['close'] >= df_copy['open']
    vol_up = df_copy['volume'].where(up_day, 0)
    vol_down = df_copy['volume'].where(~up_day, 0)
    span = 10
    ewma_up = vol_up.ewm(span=span, adjust=False).mean()
    ewma_down = vol_down.ewm(span=span, adjust=False).mean()
    ratio = np.where(ewma_down > 0, ewma_up / ewma_down, 1.0)
    factor = vol_abnorm * ratio
    df_copy['factor_abnormality_asymmetry'] = factor
    return df_copy['factor_abnormality_asymmetry']
```

---

## Factor 74

**Formula:** `R(DD) + R(Dur) + R(Rec) + R(V/MA20)`

**Rationale:** Combines drawdown geometry (depth, duration, recovery) with volume confirmation (volume relative to 20-day average) to capture both price structure and trading activity during downturns. Each component is rank-normalized to preserve the rank stability strength of the drawdown parent while adding volume information to improve linear predictive power. Volume ratio identifies periods of elevated trading that can confirm or reject the drawdown signal, targeting improvement in IC and MI without sacrificing RankICIR.

```python
def factor_drawdown_volume_geometry(df):
    """Rank-normalized drawdown geometry plus volume confirmation."""
    df_copy = df.copy()
    rolling_high = df_copy["close"].rolling(window=252, min_periods=20).max()
    drawdown = (rolling_high - df_copy["close"]) / rolling_high
    new_peak = df_copy["close"] >= rolling_high
    group = new_peak.cumsum()
    duration = new_peak.groupby(group).cumcount()
    recovery = -drawdown.diff(5)
    vol_ma = df_copy["volume"].rolling(20).mean()
    vol_ratio = df_copy["volume"] / vol_ma
    rank_dd = drawdown.rank(pct=True)
    rank_dur = duration.rank(pct=True)
    rank_rec = recovery.rank(pct=True)
    rank_vol = vol_ratio.rank(pct=True)
    df_copy["factor_drawdown_volume_geometry"] = rank_dd + rank_dur + rank_rec + rank_vol
    return df_copy["factor_drawdown_volume_geometry"]
```

---

## Factor 76

**Formula:** `(R(DD)+R(Dur)+R(Rec))/3 * (1+R(V/MA20))`

**Rationale:** Preserves the rank-normalized drawdown geometry strength (high RankICIR) while introducing a multiplicative modulation with volume ratio to amplify signal when trading activity confirms drawdown structure. This targets improvement in IC and MI by capturing nonlinear volume-price interactions without sacrificing rank stability.

```python
def factor_drawdown_volume_modulated(df):
    df_copy = df.copy()
    rolling_high = df_copy["close"].rolling(window=252, min_periods=20).max()
    drawdown = (rolling_high - df_copy["close"]) / rolling_high
    new_peak = df_copy["close"] >= rolling_high
    group = new_peak.cumsum()
    duration = new_peak.groupby(group).cumcount()
    recovery = -drawdown.diff(5)
    vol_ma = df_copy["volume"].rolling(20).mean()
    vol_ratio = df_copy["volume"] / vol_ma
    rank_dd = drawdown.rank(pct=True)
    rank_dur = duration.rank(pct=True)
    rank_rec = recovery.rank(pct=True)
    rank_vol = vol_ratio.rank(pct=True)
    drawdown_score = (rank_dd + rank_dur + rank_rec) / 3.0
    df_copy["factor_drawdown_volume_modulated"] = drawdown_score * (1 + rank_vol)
    return df_copy["factor_drawdown_volume_modulated"]
```

---

## Factor 87

**Formula:** `vol_z = (volume - SMA(20)) / STD(20); vol_trend = SMA(5)/SMA(20) - 1; price_amp = abs(close.pct_change()); factor = vol_z * vol_trend * (1 + price_amp)`

**Rationale:** This mutation enhances the parent factor by adding a price change amplitude multiplier. The idea is that volume spikes accompanied by larger price moves carry stronger conviction, especially when they occur in a rising volume trend. The price amplification captures nonlinear dependence that the pure volume factor might miss, targeting the MI bottleneck while preserving the core volume abnormality and trend logic.

**Tools:** library_functions=np.abs

```python
def factor_volume_abnormality_price_amp(df):
    df_copy = df.copy()
    vol = df_copy['volume']
    close = df_copy['close']
    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    vol_std20 = vol.rolling(20, min_periods=10).std()
    vol_z = (vol - vol_ma20) / vol_std20.replace(0, np.nan)
    vol_ma5 = vol.rolling(5, min_periods=3).mean()
    vol_trend = vol_ma5 / vol_ma20 - 1
    close_pct = close.pct_change()
    price_amp = np.abs(close_pct)
    factor = vol_z * vol_trend * (1 + price_amp)
    factor = factor.fillna(0)
    factor.name = "factor_volume_abnormality_price_amp"
    return factor
```

---

## Factor 88

**Formula:** `imbalance = (2*close - high - low) / (high - low), flow = imbalance * volume, EMA(flow, span=10)`

**Rationale:** Preserves the parent's core idea of directional pressure from order imbalance (close position within the range) weighted by volume, but replaces the simple rolling sum with an exponential moving average (span=10) to reduce noise and improve rank stability. The mutation targets the ICIR and RankICIR bottleneck seen in the parent's rejection, while maintaining the economic intuition of persistent order flow. Uses nan-safe division and clipping for robustness.

```python
def factor_order_imbalance_volume_flow_ema(df):
    df_copy = df.copy()
    high_low = df_copy["high"] - df_copy["low"]
    high_low_safe = high_low.replace(0, np.nan)
    daily_imbalance = (2 * df_copy["close"] - df_copy["high"] - df_copy["low"]) / high_low_safe
    daily_flow = daily_imbalance * df_copy["volume"]
    factor = daily_flow.ewm(span=10, min_periods=1, adjust=False).mean()
    factor = factor.fillna(0).clip(-1e10, 1e10)
    factor.name = "factor_order_imbalance_volume_flow_ema"
    return factor
```

---

## Factor 89

**Formula:** `(close - open) * (volume / 20-day volume MA), then 21-day EWM`

**Rationale:** This factor mutates the parent volume-adjusted price impact by replacing the sqrt(volume) denominator with a volume trend ratio (volume / 20-day moving average). This emphasizes price moves that occur on abnormally high volume relative to recent trend, capturing conviction-driven movements. The EWM smoothing improves stability and consistency, targeting the IC bottleneck while preserving the parent's rank-based strength (RankIC).

```python
def factor_volume_trend_adjusted_impact(df):
    df_copy = df.copy()
    price_move = df_copy['close'] - df_copy['open']
    volume_ma20 = df_copy['volume'].rolling(window=20, min_periods=20).mean()
    volume_ratio = df_copy['volume'] / (volume_ma20 + 1e-10)
    impact = price_move * volume_ratio
    factor = impact.ewm(span=21, min_periods=21, adjust=False).mean()
    df_copy["factor_volume_trend_adjusted_impact"] = factor
    return df_copy["factor_volume_trend_adjusted_impact"]
```

---

## Factor 90

**Formula:** `$\text{EWMA}_{20}(|r_t|) \times \frac{\text{volume}_t}{\text{rollingmean}_{20}(\text{volume})}$`

**Rationale:** Combines the smooth absolute return persistence with volume confirmation. Stocks with low absolute return smoothness (stable) but high relative volume indicate calm periods interrupted by high-volume moves, which may signal information-driven volatility. This preserves the temporal stability concept while adding volume to improve cross-sectional ranking.

```python
def factor_smooth_abs_return_volume(df):
    df_copy = df.copy()
    returns = df_copy['close'].pct_change().abs()
    smooth_abs = returns.ewm(span=20, min_periods=20).mean()
    volume_ma = df_copy['volume'].rolling(window=20, min_periods=20).mean()
    volume_ratio = df_copy['volume'] / volume_ma
    factor = smooth_abs * volume_ratio
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_smooth_abs_return_volume'] = factor
    return df_copy['factor_smooth_abs_return_volume']
```

---

## Factor 91

**Formula:** `z-score(lower_shadow / (lower+upper), 20) * (volume / volume_ma20)`

**Rationale:** Combines the candlestick shape anomaly (lower shadow proportion z-score) from bar_shape_001 with the volume confirmation (ratio to 20-day average volume) from apvc_gen0_1. The multiplication amplifies shape signals when volume is elevated, filtering out low-conviction noise. The z-score normalization ensures stationarity and rank stability, while volume ratio improves IC and MI by emphasizing information-rich episodes.

```python
def factor_shadow_volume_confirmed(df):
    df_copy = df.copy()
    lower = df_copy[['open', 'close']].min(axis=1) - df_copy['low']
    upper = df_copy['high'] - df_copy[['open', 'close']].max(axis=1)
    total = lower + upper
    shadow_ratio = lower / (total + 1e-8)
    mean = shadow_ratio.rolling(20, min_periods=20).mean()
    std = shadow_ratio.rolling(20, min_periods=20).std()
    shadow_z = (shadow_ratio - mean) / (std + 1e-8)
    volume_ma20 = df_copy['volume'].rolling(20, min_periods=1).mean()
    volume_ratio = df_copy['volume'] / (volume_ma20 + 1e-8)
    factor = shadow_z * volume_ratio
    factor = factor.clip(-5, 5).fillna(0)
    factor.name = 'factor_shadow_volume_confirmed'
    return factor
```

---

## Factor 103

**Formula:** `rank(depth) * rank(vol/vol_ma20) + rank(duration) + rank(delta_depth_10)`

**Rationale:** Extends the multi-scale drawdown geometry by incorporating volume confirmation. Depth is multiplied by normalized volume to emphasize drawdowns with heavy participation, while duration and recovery change remain additive. This nonlinear interaction targets the mutual information bottleneck without disrupting rank stability.

```python
def factor_drawdown_volume_geometry(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    running_max = close.expanding().max()
    depth = (running_max - close) / running_max
    new_high = close == running_max
    group = new_high.cumsum()
    duration = df_copy.groupby(group).cumcount()
    depth_chg = depth.diff(10).fillna(0)
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma
    rank_depth = depth.rank()
    rank_duration = duration.rank()
    rank_recovery = depth_chg.rank()
    rank_vol = vol_ratio.rank()
    depth_vol = rank_depth * rank_vol
    factor = depth_vol + rank_duration + rank_recovery
    df_copy['factor_drawdown_volume_geometry'] = factor
    return df_copy['factor_drawdown_volume_geometry']
```

---

## Factor 104

**Formula:** `rank_corr(ret, vol_chg, 20) * rank(ema_imbalance / std_imbalance)`

**Rationale:** Combines herding consensus (rank correlation of returns and volume changes over 20 days) with directional pressure (normalized dollar imbalance) via rank multiplication. The herding component captures agreement between price and volume direction; the pressure component adds the magnitude and sign of order flow. Multiplying their ranks amplifies signals when both are aligned and strong, enhancing nonlinear mutual information while preserving the rank stability from the herding parent.

```python
def factor_herding_pressure(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_chg = df_copy['volume'].pct_change()
    rank_ret = ret.rank()
    rank_vol = vol_chg.rank()
    rank_corr = rank_ret.rolling(20).corr(rank_vol)
    imbalance = (df_copy['close'] - df_copy['open']) * df_copy['volume']
    ema_imb = imbalance.ewm(span=5).mean()
    std_imb = imbalance.rolling(20).std()
    dp = ema_imb / (std_imb + 1e-10)
    rank_dp = dp.rank()
    factor = rank_corr * rank_dp
    factor.name = 'factor_herding_pressure'
    return factor
```

---

## Factor 106

**Formula:** `SDC = EMA(ret,21)/EMA(|ret|,21) * (vol/SMA(vol,21)); CS = -skewness(ret,20); factor = SDC * (1 - rank_pct(CS))`

**Rationale:** This factor combines smooth directional consistency (from parent stability factor) with crash risk measured by negative skewness (from crash skew parent). By downweighting the stability signal when crash risk is high, it aims to preserve the strength of the stability factor during normal regimes while protecting against false trends during extreme risk periods. The rank normalization of crash skew ensures a smooth gating effect. This crossover targets the bottleneck of the stability factor's vulnerability to crash regimes, aiming for improved robustness across market conditions.

```python
def factor_stability_crash_guard(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    ret = close.pct_change()
    # SDC: smooth directional consistency
    ema_ret = ret.ewm(span=21, min_periods=21).mean()
    ema_absret = ret.abs().ewm(span=21, min_periods=21).mean()
    ratio = ema_ret / ema_absret.replace(0, np.nan)
    sma_vol = volume.rolling(21, min_periods=21).mean()
    vol_factor = volume / sma_vol.replace(0, np.nan)
    sdc = ratio * vol_factor
    # CS: crash skew (negative skewness, inverted so high values = high crash risk)
    N = 20
    rolling_mean = ret.rolling(N).mean()
    rolling_m2 = (ret**2).rolling(N).mean()
    rolling_m3 = (ret**3).rolling(N).mean()
    central_m3 = rolling_m3 - 3*rolling_mean*rolling_m2 + 2*rolling_mean**3
    rolling_var_pop = ret.rolling(N).var(ddof=0)
    rolling_std_pop = rolling_var_pop ** 0.5
    skewness = central_m3 / (rolling_std_pop**3).replace(0, np.nan)
    cs = -skewness
    # Normalize cs to [0,1] using rank
    cs_rank = cs.rank(pct=True).fillna(0.5)
    # Combine: reduce SDC when crash risk is high
    factor = sdc * (1 - cs_rank)
    factor = factor.fillna(0)
    df_copy['factor_stability_crash_guard'] = factor
    return df_copy['factor_stability_crash_guard']
```

---

## Factor 118

**Formula:** `pressure = (close - open) * volume * (volume / EMA(volume, 10)) / std(pressure, 20)`

**Rationale:** This mutation preserves the core price-volume imbalance mechanism of the parent while increasing responsiveness to recent volume patterns via exponential weighted volume normalization (10-day ewm vs 20-day SMA). The shorter volume window aims to improve the factor's directional sensitivity (IC) and capture nonlinear interactions (MI), while the longer volatility normalization window (20-day rolling std) maintains rank stability (RankIC, RankICIR). The economic intuition remains that periods where volume amplifies price movement carry predictive information.

```python
def factor_volume_weighted_pressure_mut1(df):
    df_copy = df.copy()
    raw_pressure = (df_copy["close"] - df_copy["open"]) * df_copy["volume"]
    ema_volume = df_copy["volume"].ewm(span=10, adjust=False).mean()
    norm_volume = df_copy["volume"] / ema_volume
    weighted_pressure = raw_pressure * norm_volume
    std_pressure = weighted_pressure.rolling(window=20).std()
    factor = weighted_pressure / (std_pressure + 1e-10)
    df_copy["factor_volume_weighted_pressure_mut1"] = factor
    return df_copy["factor_volume_weighted_pressure_mut1"]
```

---

## Factor 119

**Formula:** `asymmetry = (rolling_mean(up_range, 20, min_periods=5) - rolling_mean(down_range, 20, min_periods=5)) / (rolling_mean(up_range, 20, min_periods=5) + rolling_mean(down_range, 20, min_periods=5) + epsilon)`

**Rationale:** Repaired version of factor_asymmetry_index_rolling (alpha-vol-asymmetry-mutation-1). Reduced min_periods from 10 to 5 to lower NaN fraction while retaining the 20-day window. The economic intuition remains: comparing rolling average up-day range to down-day range as a bounded asymmetry index.

**Tools:** library_functions=pd.Series

```python
def factor_asymmetry_index_rolling(df):
    df_copy = df.copy()
    range_ = df_copy["high"] - df_copy["low"]
    up_day = df_copy["close"] > df_copy["open"]
    down_day = ~up_day
    up_range = pd.Series(np.nan, index=df_copy.index)
    down_range = pd.Series(np.nan, index=df_copy.index)
    up_range[up_day] = range_[up_day]
    down_range[down_day] = range_[down_day]
    window = 20
    up_mean = up_range.rolling(window=window, min_periods=5).mean()
    down_mean = down_range.rolling(window=window, min_periods=5).mean()
    epsilon = 1e-8
    index_ = (up_mean - down_mean) / (up_mean + down_mean + epsilon)
    df_copy["factor_asymmetry_index_rolling"] = index_
    return df_copy["factor_asymmetry_index_rolling"]
```

---

## Factor 120

**Formula:** `EWMA(|r|, span=10) * (vol / MA(vol,20)) * (1 + EWMA(up_range,20) / (EWMA(down_range,20)+ε))`

**Rationale:** This factor combines the smooth persistence of absolute returns (EWMA) with volume confirmation from parent alpha-stability-gen0-001, and asymmetric volatility from parent alpha-vol-asymmetry-1. The asymmetry ratio (up-range EWMA / down-range EWMA) adds directional bias: when up-day ranges exceed down-day ranges, the factor is amplified, capturing bullish momentum or mean reversion pressure. The volume ratio filters low-participation noise, preserving rank stability. This crossover targets improved IC and ICIR while maintaining the rank stability strength.

**Tools:** library_functions=pd.Series

```python
def factor_smooth_asymmetry_persistence(df):
    df_copy = df.copy()
    # Compute absolute returns
    returns = df_copy['close'].pct_change().abs()
    # Smooth persistence (EWMA of abs returns)
    smooth_ret = returns.ewm(span=10, min_periods=10).mean()
    # Volume confirmation
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(window=20, min_periods=20).mean()
    # Asymmetric volatility
    range_ = df_copy['high'] - df_copy['low']
    up_day = df_copy['close'] > df_copy['open']
    up_range = pd.Series(np.nan, index=df_copy.index)
    down_range = pd.Series(np.nan, index=df_copy.index)
    up_range[up_day] = range_[up_day]
    down_range[~up_day] = range_[~up_day]
    up_range_ewm = up_range.ewm(span=20, min_periods=10).mean()
    down_range_ewm = down_range.ewm(span=20, min_periods=10).mean()
    epsilon = 1e-8
    asymmetry_ratio = up_range_ewm / (down_range_ewm + epsilon)
    # Combine: smooth persistence gated by volume and modulated by asymmetry
    factor = smooth_ret * vol_ratio * (1 + asymmetry_ratio)
    df_copy['factor_smooth_asymmetry_persistence'] = factor.fillna(0)
    return df_copy['factor_smooth_asymmetry_persistence']
```

---

## Factor 122

**Formula:** `range_ratio = (high - low) / SMA(high-low, 20); vol_z20 = (volume - SMA(volume,20)) / STD(volume,20); vol_trend = volume / SMA(volume,200); factor = range_ratio * vol_z20 * vol_trend`

**Rationale:** Combines range-based volatility expansion with both short-term volume abnormality and long-term volume trend. Range ratio identifies volatility expansion relative to recent average; volume z-score captures unusual trading activity; volume trend ratio indicates volume regime relative to long-term baseline. The product strengthens when all three align, targeting price-volume dynamics in a multi-scale volume confirmation framework.

```python
def factor_range_volume_trend(df):
    df_copy = df.copy()
    high, low, vol = df_copy['high'], df_copy['low'], df_copy['volume']
    # range expansion ratio
    range_ = high - low
    range_mean = range_.rolling(20).mean().shift(1)
    range_ratio = range_ / range_mean.replace(0, np.nan)
    # short-term volume z-score
    vol_mean20 = vol.rolling(20).mean().shift(1)
    vol_std20 = vol.rolling(20).std(ddof=0).shift(1)
    vol_z20 = (vol - vol_mean20) / (vol_std20 + 1e-8)
    # long-term volume trend
    vol_mean200 = vol.rolling(200).mean().shift(1)
    vol_trend = vol / (vol_mean200 + 1e-8)
    # combine
    factor = range_ratio * vol_z20 * vol_trend
    df_copy['factor_range_volume_trend'] = factor
    return df_copy['factor_range_volume_trend']
```

---

## Factor 133

**Formula:** `symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow); ts_rank_{20}(symmetry) * (volume / rolling_mean_{20}(volume)) * ewm_span21(|pct_change|)`

**Rationale:** Combines smooth absolute return (parent1: stable volume-confirmed return magnitude) with candlestick symmetry rank scaled by volume ratio (parent2: shape directional imbalance with volume confirmation). The product selects periods where both return consistency and shape extremity coincide, reinforcing signal strength when volume confirms. This preserves the rank stability and volume amplification of parent2 while incorporating the smooth return consistency of parent1.

**Tools:** library_functions=np.maximum, np.minimum

```python
def factor_stable_symmetry_reinforced(df):
    df_copy = df.copy()
    # Parent 1: smooth absolute return
    ret = df_copy['close'].pct_change()
    abs_ret = ret.abs()
    smooth_abs_ret = abs_ret.ewm(span=21, min_periods=10).mean()
    # Parent 2: symmetry rank * volume ratio
    high = df_copy['high']
    low = df_copy['low']
    open_ = df_copy['open']
    close = df_copy['close']
    volume = df_copy['volume']
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    upper_shadow = high - body_top
    lower_shadow = body_bottom - low
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    symmetry_rank = symmetry.rolling(20, min_periods=10).rank(pct=True)
    vol_ma = volume.rolling(20, min_periods=10).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)
    # Combine: reinforce smooth return with shape-volume signal
    factor = smooth_abs_ret * (symmetry_rank * vol_ratio)
    factor = factor.fillna(0)
    return factor.rename('factor_stable_symmetry_reinforced')
```

---

## Factor 135

**Formula:** `factor = ema_10(ret) * tanh(vol_z) * tanh(range_ratio_z) + 0.2 * rank_ts(symmetry,20) * volume/ma_volume_20`

**Rationale:** Mutation of mut_dir_vol_shape_003: preserves the core combination of volume-shape confirmation (tanh(vol_z) * tanh(range_ratio_z)) and shape asymmetry secondary component. Replaces the discrete directional persistence (net_dir * mom) with a continuous exponentially weighted moving average of returns (span=10). This smooths out short-term noise and provides a more robust directional signal, targeting improvement in ICIR and MI while preserving the strong RankIC from the parent. The secondary component weight is retained.

**Tools:** library_functions=np.maximum, np.minimum, np.tanh

```python
def factor_direction_volume_shape_mut2(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    ema_ret = ret.ewm(span=10, adjust=False).mean()
    vol_20_mean = df_copy["volume"].rolling(20).mean()
    vol_20_std = df_copy["volume"].rolling(20).std(ddof=1)
    vol_z = (df_copy["volume"] - vol_20_mean) / vol_20_std.replace(0.0, np.nan)
    range_ = df_copy["high"] - df_copy["low"]
    range_20_mean = range_.rolling(20).mean()
    range_ratio = range_ / range_20_mean.replace(0.0, np.nan)
    range_ratio_20_mean = range_ratio.rolling(20).mean()
    range_ratio_20_std = range_ratio.rolling(20).std(ddof=1).replace(0.0, np.nan)
    range_ratio_z = (range_ratio - range_ratio_20_mean) / range_ratio_20_std
    vol_tanh = np.tanh(vol_z)
    range_tanh = np.tanh(range_ratio_z)
    confirmation = vol_tanh * range_tanh
    body_top = np.maximum(df_copy["open"], df_copy["close"])
    body_bottom = np.minimum(df_copy["open"], df_copy["close"])
    upper_shadow = df_copy["high"] - body_top
    lower_shadow = body_bottom - df_copy["low"]
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    symmetry_rank = symmetry.rolling(20, min_periods=10).rank(pct=True)
    vol_ratio = df_copy["volume"] / vol_20_mean.replace(0.0, np.nan)
    secondary = symmetry_rank * vol_ratio
    factor = ema_ret * confirmation + 0.2 * secondary
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy["factor_direction_volume_shape_mut2"] = factor
    return df_copy["factor_direction_volume_shape_mut2"]
```

---

## Factor 138

**Formula:** `factor = (net_dir * mom) * tanh( ((rank(vol,20)-0.5)*2) * ((rank(range,20)-0.5)*2) )`

**Rationale:** Mutation preserves the core volume-confirmed directional persistence structure but replaces outlier-sensitive z-score normalization of volume and range with rank-based z-scores (percentile rank centered and scaled to [-1,1]). This reduces the impact of extreme observations while maintaining monotonic ordering and the tanh compression, targeting improvement in IC and RankIC while preserving the parent's relatively high MI.

**Tools:** library_functions=np.sign, np.tanh

```python
def factor_volume_confirmed_dir_mut_v4(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    sign_ret = np.sign(ret)
    net_dir = sign_ret.rolling(5, min_periods=5).sum()
    mom = df_copy["close"] / df_copy["close"].shift(20) - 1
    direction_signal = net_dir * mom
    vol_rank = df_copy["volume"].rolling(20, min_periods=20).rank(pct=True)
    range_ = df_copy["high"] - df_copy["low"]
    range_rank = range_.rolling(20, min_periods=20).rank(pct=True)
    vol_z_rank = (vol_rank - 0.5) * 2
    range_z_rank = (range_rank - 0.5) * 2
    raw_composite = vol_z_rank * range_z_rank
    composite_confidence = np.tanh(raw_composite)
    factor = direction_signal * composite_confidence
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy["factor_volume_confirmed_dir_mut_v4"] = factor
    return df_copy["factor_volume_confirmed_dir_mut_v4"]
```

---

## Factor 140

**Formula:** `(0.5 * ewm_span21(|pct_change|) + 0.5 * ts_rank_{21}(|pct_change|)) * (ts_rank_{20}(symmetry) * (volume / rolling_mean_{20}(volume)))`

**Rationale:** Crossover of elite crossover-gen1-001 (smooth absolute return) and qualified mut_stable_symmetry_001 (percentile rank of absolute return). The magnitude term is a weighted average of the two parent magnitude measures, preserving the trend-consistency of the smooth EMA and the outlier robustness of the rank. The shape-volume confirmation component is common to both parents. This hybrid aims to maintain the strong RankIC from the elite parent while improving mutual information by balancing the magnitude representation. The target bottleneck is the relatively low MI in both parents.

**Tools:** library_functions=np.maximum, np.minimum

```python
def factor_crossover_hybrid_magnitude(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    abs_ret = ret.abs()
    smooth_abs_ret = abs_ret.ewm(span=21, min_periods=10).mean()
    rank_abs_ret = abs_ret.rolling(21, min_periods=10).rank(pct=True)
    magnitude = 0.5 * smooth_abs_ret + 0.5 * rank_abs_ret
    high = df_copy['high']
    low = df_copy['low']
    open_ = df_copy['open']
    close = df_copy['close']
    volume = df_copy['volume']
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    upper_shadow = high - body_top
    lower_shadow = body_bottom - low
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    symmetry_rank = symmetry.rolling(20, min_periods=10).rank(pct=True)
    vol_ma = volume.rolling(20, min_periods=10).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)
    factor = magnitude * (symmetry_rank * vol_ratio)
    factor = factor.fillna(0)
    return factor.rename('factor_crossover_hybrid_magnitude')
```

---

## Factor 142

**Formula:** `direction * (avg_abs_ret * (1 + tanh(vol_z)) * (1 + tanh(sym_z)))`

**Rationale:** Mutation of crossover-gen4-001: preserves the core directional persistence (net_5_sign * momentum) and volume-confirmed magnitude logic, but replaces rolling percentile ranks with hyperbolic tangent of z-scores for both volume and symmetry components. This reduces extreme value influence and improves cross-sectional stability. The mutation targets improvement in ICIR and MI while preserving the strong RankIC from the parent structure.

**Tools:** library_functions=np.maximum, np.minimum, np.sign, np.tanh

```python
def factor_mutation_dir_vol_shape_zscore(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    sign_ret = np.sign(ret)
    net_dir = sign_ret.rolling(5, min_periods=5).sum()
    mom = df_copy['close'] / df_copy['close'].shift(20) - 1
    direction = net_dir * mom
    abs_ret = ret.abs()
    avg_abs_ret = abs_ret.rolling(20, min_periods=10).mean()
    vol = df_copy['volume']
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std(ddof=1)
    vol_z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)
    vol_tanh = np.tanh(vol_z)
    high = df_copy['high']
    low = df_copy['low']
    open_ = df_copy['open']
    close = df_copy['close']
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    upper_shadow = high - body_top
    lower_shadow = body_bottom - low
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    sym_mean = symmetry.rolling(20).mean()
    sym_std = symmetry.rolling(20).std(ddof=1)
    sym_z = (symmetry - sym_mean) / sym_std.replace(0.0, np.nan)
    sym_tanh = np.tanh(sym_z)
    magnitude = avg_abs_ret * (1 + vol_tanh) * (1 + sym_tanh)
    factor = direction * magnitude
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_mutation_dir_vol_shape_zscore'] = factor
    return df_copy['factor_mutation_dir_vol_shape_zscore']
```

---

## Factor 145

**Formula:** `factor = direction * [rank(vol_z,20)*rank(range_ratio,20) + 0.5*tanh(vol_z*range_ratio)] * (1+0.25*tanh(symmetry))`

**Rationale:** Crossover combines the rank-based volume-range confirmation (robust ordinal ordering) from the strong parent mut_dir_vol_shape_tanh_001 with a tanh-transformed raw composite from volume_confirmed_dir_mut_3 to capture nonlinear magnitude effects that may improve mutual information. The shape asymmetry modulation from the strong parent is retained. The weight 0.5 on tanh_conf ensures the rank-based term dominates, preserving RankIC and RankICIR strength while adding a complementary nonlinear signal to target MI improvement.

**Tools:** library_functions=np.maximum, np.minimum, np.sign, np.tanh

```python
def factor_rank_tanh_volume_shape_crossover(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    sign_ret = np.sign(ret)
    net_dir = sign_ret.rolling(5, min_periods=5).sum()
    mom = df_copy["close"] / df_copy["close"].shift(20) - 1
    direction = net_dir * mom
    vol_20_mean = df_copy["volume"].rolling(20).mean()
    vol_20_std = df_copy["volume"].rolling(20).std(ddof=1)
    vol_z = (df_copy["volume"] - vol_20_mean) / vol_20_std.replace(0.0, np.nan)
    range_ = df_copy["high"] - df_copy["low"]
    range_20_mean = range_.rolling(20).mean()
    range_ratio = range_ / range_20_mean.replace(0.0, np.nan)
    rank_vol_z = vol_z.rolling(20, min_periods=10).rank(pct=True)
    rank_range_ratio = range_ratio.rolling(20, min_periods=10).rank(pct=True)
    rank_conf = rank_vol_z * rank_range_ratio
    raw_conf = vol_z * range_ratio
    tanh_conf = np.tanh(raw_conf)
    combined_conf = rank_conf + 0.5 * tanh_conf
    body_top = np.maximum(df_copy["open"], df_copy["close"])
    body_bottom = np.minimum(df_copy["open"], df_copy["close"])
    upper_shadow = df_copy["high"] - body_top
    lower_shadow = body_bottom - df_copy["low"]
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    tanh_symmetry = np.tanh(symmetry)
    factor = direction * combined_conf * (1 + 0.25 * tanh_symmetry)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy["factor_rank_tanh_volume_shape_crossover"] = factor
    return df_copy["factor_rank_tanh_volume_shape_crossover"]
```

---

## Factor 149

**Formula:** `factor = direction * [rank(vol_z,20)*rank(range_ratio,20) + 0.5*tanh(vol_z*range_ratio)] * [1 + 0.25*tanh(symmetry)*tanh(vol_ratio-1)]`

**Rationale:** Mutation of crossover_dir_vol_conf_shape_001. Preserves the core directional persistence and volume-range confirmation structure. Replaces the unconditional shape modulation with a volume-weighted version: shape asymmetry (tanh(symmetry)) is multiplied by tanh(vol_ratio-1) such that shape only contributes when volume is above its 20-day mean. This targets the bottleneck of ICIR by filtering noisy shape signals during low-volume periods, while maintaining the parent's strong RankIC and RankICIR.

**Tools:** library_functions=np.maximum, np.minimum, np.sign, np.tanh

```python
def factor_mut_vol_weighted_shape_dir(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    sign_ret = np.sign(ret)
    net_dir = sign_ret.rolling(5, min_periods=5).sum()
    mom = df_copy["close"] / df_copy["close"].shift(20) - 1
    direction = net_dir * mom
    vol_20_mean = df_copy["volume"].rolling(20).mean()
    vol_20_std = df_copy["volume"].rolling(20).std(ddof=1)
    vol_z = (df_copy["volume"] - vol_20_mean) / vol_20_std.replace(0.0, np.nan)
    range_ = df_copy["high"] - df_copy["low"]
    range_20_mean = range_.rolling(20).mean()
    range_ratio = range_ / range_20_mean.replace(0.0, np.nan)
    rank_vol_z = vol_z.rolling(20, min_periods=10).rank(pct=True)
    rank_range_ratio = range_ratio.rolling(20, min_periods=10).rank(pct=True)
    rank_conf = rank_vol_z * rank_range_ratio
    raw_conf = vol_z * range_ratio
    tanh_conf = np.tanh(raw_conf)
    combined_conf = rank_conf + 0.5 * tanh_conf
    vol_ratio = df_copy["volume"] / vol_20_mean.replace(0.0, np.nan)
    body_top = np.maximum(df_copy["open"], df_copy["close"])
    body_bottom = np.minimum(df_copy["open"], df_copy["close"])
    upper_shadow = df_copy["high"] - body_top
    lower_shadow = body_bottom - df_copy["low"]
    symmetry = (lower_shadow - upper_shadow) / (lower_shadow + upper_shadow + 1e-10)
    tanh_symmetry = np.tanh(symmetry)
    vol_weighted_shape = 1 + 0.25 * tanh_symmetry * np.tanh(vol_ratio - 1)
    factor = direction * combined_conf * vol_weighted_shape
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy["factor_mut_vol_weighted_shape_dir"] = factor
    return df_copy["factor_mut_vol_weighted_shape_dir"]
```

---

## Factor 167

**Formula:** `factor = (momentum / (1 + |z(std(Range/Close,10))|)) * (1 + clip(3*EMA(ret,10),-1,1)) * (1 + tanh(vol_rank * vol_trend))`

**Rationale:** Mutation of factor_volume_gated_momentum_mutated_v2 (gen4). Preserves the volume-gated momentum structure to maintain RankIC stability. Targets improved IC and MI by replacing the ATR/close z-score with a more responsive range-based volatility (10-day rolling std of high-low range/close) and simplifying the persistence to a clipped EMA of returns. The volume confidence is retained but uses a multiplicative combination of rank and trend to sharpen the signal.

**Tools:** library_functions=np.clip, np.tanh

```python
def factor_volume_gated_momentum_mutated_v3(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    
    # Core momentum: 5-day return
    mom = close / close.shift(5) - 1
    
    # Range-based volatility: rolling std of high-low range / close over 10 days
    range_ratio = (high - low) / close
    range_vol = range_ratio.rolling(10, min_periods=5).std()
    range_vol_mean = range_vol.rolling(20, min_periods=10).mean()
    range_vol_std = range_vol.rolling(20, min_periods=10).std(ddof=1)
    vol_z = (range_vol - range_vol_mean) / range_vol_std.replace(0, np.nan)
    vol_z_clipped = np.clip(vol_z, -2, 2)
    
    # Trend persistence: EMA of returns with span 10, clipped
    ret = close.pct_change()
    ema_ret = ret.ewm(span=10, min_periods=10, adjust=False).mean()
    persistence = np.clip(ema_ret * 3, -1, 1)
    
    # Volume components: rank percentile and trend ratio
    vol_rank = volume.rolling(20).rank(pct=True)
    vol_ma20 = volume.rolling(20).mean()
    vol_trend = volume.rolling(5).mean() / vol_ma20.replace(0, np.nan)
    
    # Confidence: multiplicative product mapped to [0,2] via tanh
    confidence = 1 + np.tanh(vol_rank * vol_trend)
    
    # Base factor: momentum adjusted by volatility and persistence
    base = mom / (1 + vol_z_clipped.abs()) * (1 + persistence)
    
    # Final factor
    factor = base * confidence
    factor = factor.fillna(0.0).clip(-5, 5)
    df_copy['factor_volume_gated_momentum_mutated_v3'] = factor
    return df_copy['factor_volume_gated_momentum_mutated_v3']
```

---

## Factor 170

**Formula:** `factor = ( (rank(ret,20)-0.5)*(rank(vol_chg,20)-0.5) / (1 + std(ret,20)) ) * (1 + 0.5*corr(ret,vol_chg,20)) * (1 + 0.5*zscore(ema(ret,20),40))`

**Rationale:** This mutation preserves the core herding and rank interaction from the parent (mut-herd-rank-gen3-002) but targets improved IC and ICIR by (1) adding a volatility adjustment (rolling std of returns) to the interaction to reduce noise during high-volatility periods, (2) replacing the trend rank with a z-score of the EMA trend for more responsive and symmetric signal, and (3) keeping the herding correlation as a confidence multiplier. The parent strength (RankIC stability from rank-based normalization and multiplicative gating) is maintained. This aims to boost IC and ICIR while keeping the herding consensus as the primary signal.

```python
def factor_herding_vol_gated_momentum(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change().fillna(0)
    vol_chg = df_copy['volume'].pct_change().fillna(0)
    herding_corr = ret.rolling(20, min_periods=20).corr(vol_chg).fillna(0)
    rank_ret = ret.rolling(20, min_periods=20).rank(pct=True).fillna(0.5)
    rank_vol = vol_chg.rolling(20, min_periods=20).rank(pct=True).fillna(0.5)
    inter = (rank_ret - 0.5) * (rank_vol - 0.5)
    vol_adj = ret.rolling(20, min_periods=20).std().fillna(0)
    inter_adj = inter / (1 + vol_adj)
    trend = ret.ewm(span=20, adjust=False).mean().fillna(0)
    trend_mean = trend.rolling(40, min_periods=20).mean().fillna(0)
    trend_std = trend.rolling(40, min_periods=20).std(ddof=1).fillna(1)
    trend_z = (trend - trend_mean) / trend_std.replace(0, 1)
    factor = inter_adj * (1 + 0.5 * herding_corr) * (1 + 0.5 * trend_z)
    factor = factor.fillna(0)
    df_copy['factor_herding_vol_gated_momentum'] = factor
    return df_copy['factor_herding_vol_gated_momentum']
```

---

## Factor 174

**Formula:** `factor = (mom / (1 + |z(ATR/close)|)) * (1 + tanh(5*EMA(ret))) * (1 + tanh(herding)) * (1 + tanh(vol_rank * vol_trend))`

**Rationale:** Crossover combining volume-gated momentum (parent1: mut_vol_gated_mom_gen4) with herding co-movement signal (parent2: mut-herd-ema-gated-gen3-002). Parent1 provides strong directional momentum with volatility adjustment and volume confidence; parent2 captures price-volume co-movement. The child multiplies the momentum base by a herding boost and volume confidence, preserving the high RankIC/RankICIR from both parents while targeting improved IC and ICIR through complementary signals.

**Tools:** library_functions=np.clip, np.maximum, np.tanh

```python
def factor_vol_gated_herding_crossover(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    
    # Parent1: momentum and volatility adjustment
    mom = close / close.shift(5) - 1
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = tr.rolling(20, min_periods=10).mean()
    vol_adj = atr / close
    vol_mean = vol_adj.rolling(20, min_periods=10).mean()
    vol_std = vol_adj.rolling(20, min_periods=10).std(ddof=1)
    vol_z = (vol_adj - vol_mean) / vol_std.replace(0, np.nan)
    vol_z_clipped = np.clip(vol_z, -2, 2)
    
    ret = close.pct_change()
    ema_ret = ret.ewm(span=7, min_periods=7, adjust=False).mean()
    persistence = np.tanh(ema_ret * 5)
    
    base_mom = (mom / (1 + vol_z_clipped.abs())) * (1 + persistence)
    
    # Parent2: herding signal
    vol_chg = volume.pct_change()
    corr_ret_vol = ret.rolling(30, min_periods=30).corr(vol_chg)
    mean_ret = ret.rolling(40, min_periods=40).mean()
    std_ret = ret.rolling(40, min_periods=40).std().replace(0, np.nan)
    z_ret = (ret - mean_ret) / std_ret
    sp_ret = np.tanh(z_ret)
    herding = corr_ret_vol * sp_ret
    
    # Volume confidence from parent1
    vol_rank = volume.rolling(20).rank(pct=True)
    vol_ma20 = volume.rolling(20).mean()
    vol_trend = volume.rolling(5).mean() / vol_ma20.replace(0, np.nan)
    confidence = 1 + np.tanh(vol_rank * vol_trend)
    
    # Combine: base_mom modulated by herding, with volume confidence
    factor = base_mom * (1 + np.tanh(herding)) * confidence
    factor = factor.fillna(0.0).clip(-5, 5)
    df_copy['factor_vol_gated_herding_crossover'] = factor
    return df_copy['factor_vol_gated_herding_crossover']
```

---

## Factor 179

**Formula:** `factor = (mom/(1+|z(ATR/close)|)) * (1+tanh(5*EMA(ret))) * (1+tanh(sign(ret)*tanh(3*(vol/vol_ma20-1)))) * (1+tanh(vol_rank * vol_trend))`

**Rationale:** Mutation of factor_vol_gated_herding_crossover. Preserves core momentum with volatility adjustment and persistence. Replaces the complex correlation-based herding with a simpler volume-signed return strength using volume deviation from its 20-period mean, making the herding signal more direct and responsive. Also adjusts volume confidence windows to shorter periods (12 for rank, 10 for trend, 3 for short-term volume) to increase sensitivity. These changes target improved IC and ICIR while maintaining the strong RankIC/RankICIR from the parent. The parent strength target is the rank stability (RankIC/RankICIR) from the herding and volume-gated combination.

**Tools:** library_functions=np.clip, np.maximum, np.sign, np.tanh

```python
def factor_vol_gated_herding_mutated(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    
    # Parent1: momentum and volatility adjustment (same as parent)
    mom = close / close.shift(5) - 1
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = tr.rolling(20, min_periods=10).mean()
    vol_adj = atr / close
    vol_mean = vol_adj.rolling(20, min_periods=10).mean()
    vol_std = vol_adj.rolling(20, min_periods=10).std(ddof=1)
    vol_z = (vol_adj - vol_mean) / vol_std.replace(0, np.nan)
    vol_z_clipped = np.clip(vol_z, -2, 2)
    
    ret = close.pct_change()
    ema_ret = ret.ewm(span=7, min_periods=7, adjust=False).mean()
    persistence = np.tanh(ema_ret * 5)
    
    base_mom = (mom / (1 + vol_z_clipped.abs())) * (1 + persistence)
    
    # Mutated herding: volume-signed return strength
    vol_ma20 = volume.rolling(20).mean()
    volume_dev = (volume - vol_ma20) / vol_ma20.replace(0, np.nan)
    herding = np.sign(ret) * np.tanh(volume_dev * 3)
    
    # Volume confidence with shorter windows
    vol_rank = volume.rolling(12).rank(pct=True)
    vol_ma10 = volume.rolling(10).mean()
    vol_trend = volume.rolling(3).mean() / vol_ma10.replace(0, np.nan)
    confidence = 1 + np.tanh(vol_rank * vol_trend)
    
    # Combine
    factor = base_mom * (1 + np.tanh(herding)) * confidence
    factor = factor.fillna(0.0).clip(-5, 5)
    df_copy['factor_vol_gated_herding_mutated'] = factor
    return df_copy['factor_vol_gated_herding_mutated']
```

---

## Factor 200

**Formula:** `(none)`

**Rationale:** Price momentum confirmed by volume surge. Short-term price change scaled by the percentile rank of current volume relative to its 20-day average volume, emphasizing moves with strong participation. The rank transform improves rank stability.

```python
def factor_volume_confirmed_momentum(df):
    df_copy = df.copy()
    price_ret = df_copy['close'].pct_change(5)
    volume_20_avg = df_copy['volume'].rolling(20).mean().replace(0, np.nan)
    volume_ratio = df_copy['volume'] / volume_20_avg
    volume_rank = volume_ratio.rolling(20).rank(pct=True)
    factor = price_ret * volume_rank
    factor.name = 'factor_volume_confirmed_momentum'
    return factor
```

---

## Factor 201

**Formula:** `price_ret_5 * rank(volume / sma(volume,20)) over 20-day window`

**Rationale:** Mutated from factor_volume_confirmed_momentum_v2 by increasing the volume rank window from 10 to 20 days. This captures longer-term volume patterns, reducing noise and potentially improving IC and MI while preserving the parent's strong RankIC stability from rank-based volume confirmation. Core mechanism remains unchanged: price momentum confirmed by volume percentile.

```python
def factor_volume_confirmed_momentum_v3(df):
    df_copy = df.copy()
    price_ret = df_copy['close'].pct_change(5)
    volume_20_avg = df_copy['volume'].rolling(20).mean().replace(0, np.nan)
    volume_ratio = df_copy['volume'] / volume_20_avg
    volume_rank = volume_ratio.rolling(20).rank(pct=True)
    factor = price_ret * volume_rank
    factor.name = 'factor_volume_confirmed_momentum_v3'
    return factor
```

---

## Factor 202

**Formula:** `rank_p = rolling_rank(pressure,40); scaled_rank = (rank_p - 0.5)*2; factor = scaled_rank * tanh(trend); trend = close/shift(close,5)-1; pressure = (close-open)*volume`

**Rationale:** Mutate from trend_volume_tanh_v2 by replacing z-score normalization of pressure with time-series rank (40-day percentile) scaled to [-1,1]. This reduces outlier impact and maintains monotonic signal, targeting improved IC and RankIC while preserving the parent's strong ICIR and RankICIR consistency. The core mechanism (pressure modulated by tanh trend) and all other parameters remain unchanged.

**Tools:** library_functions=np.tanh

```python
def factor_trend_volume_tanh_rank(df):
    df_copy = df.copy()
    close = df_copy['close']
    open_ = df_copy['open']
    volume = df_copy['volume']
    pressure = (close - open_) * volume
    rank_p = pressure.rolling(40, min_periods=40).rank(pct=True)
    scaled_rank = (rank_p - 0.5) * 2
    trend = close / close.shift(5) - 1
    trend_tanh = np.tanh(trend)
    factor = scaled_rank * trend_tanh
    df_copy['factor_trend_volume_tanh_rank'] = factor.fillna(0)
    return df_copy['factor_trend_volume_tanh_rank']
```

---

## Factor 203

**Formula:** `norm_pressure * tanh(vol/ma_vol21) * tanh(ATR21/close)`

**Rationale:** Mutation of factor_pressure_compression_simplified_v2: increased all rolling windows from 14 to 21 days to capture medium-term pressure and volume trends while reducing noise. This targets improved IC and RankIC by smoothing the pressure signal and the gate, while preserving the parent's strong ICIR and RankICIR consistency. Core mechanism (volume-price pressure gated by volume and volatility) remains unchanged.

**Tools:** library_functions=np.tanh, pd.Series, talib.ATR

```python
def factor_pressure_compression_simplified_v3(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    open_ = df_copy['open']
    pressure = (close - open_) * volume
    abs_pressure = pressure.abs()
    avg_abs = abs_pressure.rolling(21, min_periods=21).mean().replace(0, np.nan)
    norm_pressure = pressure / avg_abs
    vol_ma21 = volume.rolling(21, min_periods=21).mean().replace(0, np.nan)
    vol_ratio21 = volume / vol_ma21
    atr = talib.ATR(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), timeperiod=21)
    atr_series = pd.Series(atr, index=df_copy.index).fillna(0)
    atr_ratio = atr_series / close.replace(0, np.nan)
    gate = np.tanh(vol_ratio21) * np.tanh(atr_ratio)
    factor = norm_pressure * gate
    df_copy['factor_pressure_compression_simplified_v3'] = factor.fillna(0)
    return df_copy['factor_pressure_compression_simplified_v3']
```

---

## Factor 207

**Formula:** `pct_return_5 * rank(volume / sma_20) over 10`

**Rationale:** Crossover: primary mechanism from factor_volume_confirmed_momentum (apvc-001) using 5-day price return scaled by volume ratio percentile, preserving the robust 20-day volume average for ratio calculation, but borrowing the 10-day rank window from factor_volume_confirmed_momentum_v2 to make the confirmation more responsive, targeting improved IC while maintaining rank stability.

```python
def factor_volume_confirmed_momentum_crossover_v1(df):
    df_copy = df.copy()
    price_ret = df_copy['close'].pct_change(5)
    volume_20_avg = df_copy['volume'].rolling(20).mean().replace(0, np.nan)
    volume_ratio = df_copy['volume'] / volume_20_avg
    volume_rank = volume_ratio.rolling(10).rank(pct=True)
    factor = price_ret * volume_rank
    factor.name = 'factor_volume_confirmed_momentum_crossover_v1'
    return factor
```

---

## Factor 208

**Formula:** `pct_return_5 * rank(volume / sma_30) over 10`

**Rationale:** Mutation of crossover_vcm_001: increase the volume average window from 20 to 30 days to smooth the volume ratio, reducing noise in the volume confirmation while preserving the core momentum mechanism and rank window. Targets improved IC and RankIC stability.

```python
def factor_volume_confirmed_momentum_crossover_v2(df):
    df_copy = df.copy()
    price_ret = df_copy['close'].pct_change(5)
    volume_30_avg = df_copy['volume'].rolling(30).mean().replace(0, np.nan)
    volume_ratio = df_copy['volume'] / volume_30_avg
    volume_rank = volume_ratio.rolling(10).rank(pct=True)
    factor = price_ret * volume_rank
    factor.name = 'factor_volume_confirmed_momentum_crossover_v2'
    return factor
```

---

## Factor 209

**Formula:** `pct_change(close,5) * rank_20((high-low) * (volume/rolling_median(volume,20))) * tanh(ATR14/close)`

**Rationale:** Crossover: primary mechanism from liquidity_range_gated_momentum_v4 (5-day return gated by volume-adjusted price range rank conviction) to preserve its strong RankIC and MI, borrowing lightweight modifier of ATR-based volatility gate from pressure_compression_simplified_v2 to add volatility-aware downweighting, targeting improved ICIR while maintaining core ranking consistency. Only added multiplication by tanh(ATR/close).

**Tools:** library_functions=np.tanh, pd.Series, talib.ATR

```python
def factor_liquidity_range_gated_momentum_v5(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    ret5 = close.pct_change(5)
    med_vol20 = volume.rolling(20, min_periods=20).median().replace(0, np.nan)
    vol_ratio = volume / med_vol20
    price_range = high - low
    conviction_raw = price_range * vol_ratio
    conviction_rank = conviction_raw.rolling(20, min_periods=20).rank(pct=True)
    atr = talib.ATR(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), timeperiod=14)
    atr_series = pd.Series(atr, index=df_copy.index).fillna(0)
    atr_ratio = atr_series / close.replace(0, np.nan)
    vol_gate = np.tanh(atr_ratio)
    factor = ret5 * conviction_rank * vol_gate
    df_copy['factor_liquidity_range_gated_momentum_v5'] = factor.fillna(0)
    return df_copy['factor_liquidity_range_gated_momentum_v5']
```

---

## Factor 215

**Formula:** `pct_change(close,5) * rank_20((high-low) * (volume/rolling_median(volume,20))) * tanh(ATR20/close)`

**Rationale:** Mutation from liquidity_range_gated_momentum_v5: changed ATR window from 14 to 20 to align with the conviction window, reducing parameter count and potentially improving signal coherence. Preserves the core mechanism and the strong RankIC/MI from the parent while targeting improved ICIR through consistent window selection.

**Tools:** library_functions=np.tanh, pd.Series, talib.ATR

```python
def factor_liquidity_range_gated_momentum_v6(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    ret5 = close.pct_change(5)
    med_vol20 = volume.rolling(20, min_periods=20).median().replace(0, np.nan)
    vol_ratio = volume / med_vol20
    price_range = high - low
    conviction_raw = price_range * vol_ratio
    conviction_rank = conviction_raw.rolling(20, min_periods=20).rank(pct=True)
    atr = talib.ATR(high.values.astype(np.float64), low.values.astype(np.float64), close.values.astype(np.float64), timeperiod=20)
    atr_series = pd.Series(atr, index=df_copy.index).fillna(0)
    atr_ratio = atr_series / close.replace(0, np.nan)
    vol_gate = np.tanh(atr_ratio)
    factor = ret5 * conviction_rank * vol_gate
    df_copy['factor_liquidity_range_gated_momentum_v6'] = factor.fillna(0)
    return df_copy['factor_liquidity_range_gated_momentum_v6']
```

---

## Factor 231

**Formula:** `factor = ret(5) * rank_{20}(volume / sma(volume, 20))`

**Rationale:** Short-term price return confirmed by rank-transformed volume ratio: when price moves with above-average volume, the move is more coherent and likely persistent. Rank transform stabilizes the volume multiplier, improving RankIC and MI.

```python
def factor_volume_confirmed_return(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20).mean().replace(0, np.nan)
    vol_rank = vol_ratio.rolling(20).rank(pct=True)
    factor = ret * vol_rank
    factor.name = 'factor_volume_confirmed_return'
    return factor
```

---

## Factor 232

**Formula:** `factor = ret(5) * rank_{20}(volume / ewma(volume, 20))`

**Rationale:** Smoothed volume ratio: using exponential weighted moving average for volume reduces lag and better captures the most recent volume regime, potentially improving IC while preserving the rank transform for stable cross-sectional ordering.

```python
def factor_volume_confirmed_return_ema(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_ema = df_copy['volume'].ewm(span=20).mean()
    vol_ratio = df_copy['volume'] / vol_ema.replace(0, np.nan)
    vol_rank = vol_ratio.rolling(20).rank(pct=True)
    factor = ret * vol_rank
    factor.name = 'factor_volume_confirmed_return_ema'
    return factor
```

---

## Factor 233

**Formula:** `factor = ret(5) * rank_{20}(volume / ewm(volume, 20))`

**Rationale:** Mutation of parent volume_confirmed_return: replaced rolling mean with exponential weighted moving average for volume ratio denominator. This reduces lag and better captures recent volume regime, targeting improved IC while preserving the rank transform that underpins strong RankIC and RankICIR.

```python
def factor_volume_confirmed_return_ewm(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_avg = df_copy['volume'].ewm(span=20, adjust=False).mean().replace(0, np.nan)
    vol_ratio = df_copy['volume'] / vol_avg
    vol_rank = vol_ratio.rolling(20).rank(pct=True)
    factor = ret * vol_rank
    factor.name = 'factor_volume_confirmed_return_ewm'
    return factor
```

---

## Factor 234

**Formula:** `tanh(-1 * (close - close_5d_ago) * (volume / 20d_ema_volume))`

**Rationale:** Mutation of volume_price_divergence_cross_001: replaced simple rolling mean of volume with exponential weighted moving average (span=20) to make the volume ratio more responsive to recent volume changes, targeting improved IC and ICIR while preserving the core price decline and volume surge mechanism and RankIC strength.

**Tools:** library_functions=np.tanh

```python
def factor_volume_price_divergence_cross_mut(df):
    df_copy = df.copy()
    price_change_5 = df_copy['close'] - df_copy['close'].shift(5)
    avg_volume = df_copy['volume'].ewm(span=20, adjust=False).mean()
    volume_surge = df_copy['volume'] / avg_volume.replace(0, np.nan)
    factor = -price_change_5 * volume_surge
    factor = np.tanh(factor)
    factor = factor.fillna(0)
    factor.name = "factor_volume_price_divergence_cross_mut"
    return factor
```

---

## Factor 247

**Formula:** `-ret_5d * (volume / SMA(volume,20))`

**Rationale:** Crash precursor: negative return weighted by volume expansion. Heavy volume on a decline signals panic selling, a precursor to crash. Volume ratio amplifies moves with abnormal volume, capturing regime breakdown in market participation.

```python
def factor_volume_pressure_reversal(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_ma = df_copy['volume'].rolling(20).mean()
    vol_ratio = df_copy['volume'] / vol_ma.replace(0, np.nan)
    signal = -ret * vol_ratio
    signal = signal.fillna(0)
    signal.name = 'factor_volume_pressure_reversal'
    return signal
```

---

## Factor 248

**Formula:** `-ret_5d * rank(volume / SMA(volume,20), 20)`

**Rationale:** Combines crash_001's crash precursor mechanism (negative 5-day return weighted by volume expansion) with PVC-001-repaired's rank normalization on volume ratio to improve rank stability while preserving the core crash detection.

```python
def factor_crash_volume_rank(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_ma = df_copy['volume'].rolling(20).mean()
    vol_ratio = df_copy['volume'] / vol_ma.replace(0, np.nan)
    vol_rank = vol_ratio.rolling(20, min_periods=20).rank(pct=True)
    signal = -ret * vol_rank
    signal = signal.fillna(0)
    signal.name = 'factor_crash_volume_rank'
    return signal
```

---

## Factor 250

**Formula:** `mom = close/shift(close,21)-1; atr = ATR(21); vol_gate = atr/close; vol_ratio = volume / rolling_mean(volume,21); factor = mom * (1/(1+vol_gate)) * vol_ratio`

**Rationale:** Mutation of mut_vol_gate_cont_001: preserve the volatility-gated momentum core that provides consistent RankICIR and ICIR, but change the volume modifier from a 63-day median ratio to a 21-day SMA ratio to capture shorter-term volume surges more promptly. This targets improving IC and MI while keeping the core mechanism unchanged.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_vol_gated_momentum_v2(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    # 21-day momentum
    mom = close / close.shift(21) - 1
    # 21-day ATR normalized by close for volatility gate
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=21), index=df_copy.index)
    vol_gate = atr / close
    # Volume ratio to 21-day SMA (changed from 63-day median for shorter-term capture)
    vol_sma = volume.rolling(21, min_periods=21).mean()
    vol_ratio = volume / vol_sma.replace(0, np.nan)
    # Core: momentum dampened by volatility, multiplied by continuous volume confirmation
    factor = mom * (1 / (1 + vol_gate)) * vol_ratio
    df_copy['factor_vol_gated_momentum_v2'] = factor
    return df_copy['factor_vol_gated_momentum_v2']
```

---

## Factor 251

**Formula:** `-ret_5d * vol_ratio if rank(vol_ratio,20) > 0.5 else 0`

**Rationale:** Combines crash_001's raw volume ratio weighted reversal with CO_001's rank normalization to only activate during extreme volume periods (top half of 20-day rank). This preserves the crash precursor mechanism while filtering out low volume noise, aiming to improve IC and RankIC.

**Tools:** library_functions=np.where, pd.Series

```python
def factor_crash_volume_gate(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(5)
    vol_ma = df_copy['volume'].rolling(20).mean()
    vol_ratio = df_copy['volume'] / vol_ma.replace(0, np.nan)
    vol_rank = vol_ratio.rolling(20, min_periods=20).rank(pct=True)
    signal = np.where(vol_rank > 0.5, -ret * vol_ratio, 0.0)
    signal = pd.Series(signal, index=df_copy.index).fillna(0)
    signal.name = 'factor_crash_volume_gate'
    return signal
```

---

## Factor 263

**Formula:** `(EWMA5(close-open) / EWMA5(high-low)) * (volume / SMA10(volume))`

**Rationale:** Simplifies the asymmetry component by using net range (close-open) smoothed with a 5-day EWMA instead of separate up/down ranges with 10-day EWMA. This reduces code complexity and lag, targeting improved IC while preserving the volume ratio confirmation and the parent's RankIC strength.

```python
def factor_ewma_net_range_volume_ratio(df):
    df_copy = df.copy()
    net_range = df_copy["close"] - df_copy["open"]
    ewma_net = net_range.ewm(span=5, min_periods=5).mean()
    total_range = df_copy["high"] - df_copy["low"]
    ewma_total = total_range.ewm(span=5, min_periods=5).mean()
    asym = ewma_net / ewma_total.replace(0, np.nan)
    vol_ratio = df_copy["volume"] / df_copy["volume"].rolling(10, min_periods=10).mean()
    factor = asym * vol_ratio
    df_copy["factor_ewma_net_range_volume_ratio"] = factor
    return df_copy["factor_ewma_net_range_volume_ratio"]
```

---

## Factor 264

**Formula:** `- (volume / rolling_mean(volume, 10)) * (ewma(up_range,10) - ewma(down_range,10)) / ewma(total_range,10)`

**Rationale:** Crossover of crash_vol_asym_001 and lag002_mut1. Preserves the crash precursor direction (negative sign) and EWMA asymmetry mechanism from crash_vol_asym_001. Borrows a lightweight 10-day volume ratio (non-ranked) from lag002_mut1 to replace the ranked 20-day volume ratio, simplifying the volume component. Aims to improve IC while maintaining RankIC and MI strength from parents.

**Tools:** library_functions=np.maximum

```python
def factor_crash_vol_ratio_asym(df):
    df_copy = df.copy()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(10, min_periods=10).mean()
    up_range = np.maximum(0, df_copy['close'] - df_copy['open'])
    down_range = np.maximum(0, df_copy['open'] - df_copy['close'])
    total_range = df_copy['high'] - df_copy['low']
    ewma_up = up_range.ewm(span=10, min_periods=10).mean()
    ewma_down = down_range.ewm(span=10, min_periods=10).mean()
    ewma_total = total_range.ewm(span=10, min_periods=10).mean()
    asym = (ewma_up - ewma_down) / ewma_total.replace(0, np.nan)
    factor = -vol_ratio * asym
    df_copy['factor_crash_vol_ratio_asym'] = factor
    return df_copy['factor_crash_vol_ratio_asym']
```

---

## Factor 266

**Formula:** `- (volume / rolling_mean(volume,10)) * ((close - open) / (high - low))`

**Rationale:** Mutation of crash_vol_ratio_asym_cross001. Preserves the core crash precursor mechanism: volume ratio combined with directional bias, multiplied by -1. Simplifies the asymmetry component from EWMA-smoothed up/down ranges to a single-day body-to-range ratio, reducing lag and noise while keeping the parent's RankIC and MI strengths. Targets improved IC and ICIR by using a more contemporaneous directional signal.

```python
def factor_crash_vol_body_asym(df):
    df_copy = df.copy()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(10, min_periods=10).mean()
    body = df_copy['close'] - df_copy['open']
    total_range = df_copy['high'] - df_copy['low']
    body_asym = body / total_range.replace(0, np.nan)
    factor = -vol_ratio * body_asym
    df_copy['factor_crash_vol_body_asym'] = factor
    return df_copy['factor_crash_vol_body_asym']
```

---

## Factor 268

**Formula:** `Roughness_Ratio = ATR(5) / ATR(20), Trend = Close / SMA(20) - 1, Volume_Ratio = Volume / SMA(10), Factor = Roughness_Ratio * Trend * Volume_Ratio`

**Rationale:** Crossover: primary mechanism from parent alpha-mutation-gen2-001 (roughness * trend), borrows lightweight volume confirmation from parent alpha-crossover-gen1-001 but with a shorter 10-day volume average to capture immediate volume spikes, aiming to improve IC and RankIC while preserving the MI strength of the ablation parent.

**Tools:** library_functions=talib.TRANGE

```python
def factor_roughness_trend_vol_short(df):
    df_copy = df.copy()
    tr = talib.TRANGE(df_copy['high'], df_copy['low'], df_copy['close'])
    atr_short = tr.rolling(5, min_periods=5).mean()
    atr_long = tr.rolling(20, min_periods=20).mean()
    roughness_ratio = atr_short / atr_long.replace(0, np.nan)
    sma_20 = df_copy['close'].rolling(20, min_periods=20).mean()
    trend = (df_copy['close'] / sma_20) - 1
    vol_ma = df_copy['volume'].rolling(10).mean()
    vol_ratio = df_copy['volume'] / vol_ma.replace(0, np.nan)
    factor = roughness_ratio * trend * vol_ratio
    df_copy['factor_roughness_trend_vol_short'] = factor
    return df_copy['factor_roughness_trend_vol_short']
```

---

## Factor 279

**Formula:** `rev * vol_rank where rev = -((close-low)/(high-low) - 0.5) and vol_rank = rolling_rank(volume / rolling_mean(volume,20),20)`

**Rationale:** Mutation of crossover_001_g1: removed the 5-period smoothing on the intraday position fraction to make the reversal signal more responsive to current bar overreaction. The direct deviation from 0.5 captures immediate intraday mean-reversion, while the volume rank (percentile of volume ratio over 20 days) filters noisy signals. This simplification targets improved ICIR and RankICIR while preserving the core volume-reversal interaction.

```python
def factor_volume_reversal_direct(df):
    df_copy = df.copy()
    high = df_copy['high']
    low = df_copy['low']
    close = df_copy['close']
    range_ = high - low
    fraction = (close - low) / range_.replace(0.0, np.nan)
    deviation = fraction - 0.5
    rev = -deviation
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20).mean()
    vol_rank = vol_ratio.rolling(20).rank(pct=True)
    factor = rev * vol_rank
    df_copy['factor_volume_reversal_direct'] = factor
    return df_copy['factor_volume_reversal_direct']
```

---

## Factor 281

**Formula:** `- (rolling_mean((close-low)/(high-low),5) - 0.5) * rolling_rank(volume,20)`

**Rationale:** Crossover: primary mechanism is intraday mean-reversion from parent2 (crossover_001_g1), lightweight modifier from parent1 (cand_liquidity_002_mut_001) replaces the volume ratio rank with raw volume rank. The raw volume rank provides a monotonic percentile scaling that may improve RankIC while preserving the reversal signal's strong IC and ICIR. Targets the RankIC bottleneck of parent2 by borrowing parent1's rank-based volume confirmation.

```python
def factor_volume_reversal_rank(df):
    df_copy = df.copy()
    high = df_copy['high']
    low = df_copy['low']
    close = df_copy['close']
    range_ = high - low
    fraction = (close - low) / range_.replace(0.0, np.nan)
    fraction_avg = fraction.rolling(5, min_periods=5).mean()
    deviation = fraction_avg - 0.5
    rev = -deviation
    # lightweight modifier from parent1: raw volume rank
    vol_rank = df_copy['volume'].rolling(20, min_periods=20).rank(pct=True)
    factor = rev * vol_rank
    df_copy['factor_volume_reversal_rank'] = factor.fillna(0)
    return df_copy['factor_volume_reversal_rank']
```

---

## Factor 284

**Formula:** `- ( (close-low)/(high-low) - 0.5 ) * rolling_rank(volume,20)`

**Rationale:** Mutation of crossover_002_g2: removed the 5-period rolling average on the intraday position fraction to make the reversal signal more responsive to current bar data, aligning with the effective parent2 (crossover_001_g1) approach. Preserves the strong IC and ICIR from the immediate reversal while retaining the volume rank modifier to support RankIC. Targets RankIC improvement by reducing lag in the core reversal component while keeping the volume rank window unchanged.

```python
def factor_volume_reversal_rank(df):
    df_copy = df.copy()
    high = df_copy['high']
    low = df_copy['low']
    close = df_copy['close']
    range_ = high - low
    fraction = (close - low) / range_.replace(0.0, np.nan)
    deviation = fraction - 0.5
    rev = -deviation
    vol_rank = df_copy['volume'].rolling(20, min_periods=20).rank(pct=True)
    factor = rev * vol_rank
    df_copy['factor_volume_reversal_rank'] = factor.fillna(0)
    return df_copy['factor_volume_reversal_rank']
```

---

## Factor 287

**Formula:** `rev = -zscore(fraction, 10) where fraction = (close-low)/(high-low); factor = rev * rank(volume/mean(volume,20), 20)`

**Rationale:** Mutation of cross_intraday_001_g2: reduce normalization window from 20 to 10 to make the intraday reversal signal more responsive to recent volatility changes, while preserving the volume rank confirmation. Parent strength target: preserve the RankIC and volume confirmation from the parent.

```python
def factor_intraday_reversal_zscore_volume_short(df):
    df_copy = df.copy()
    high = df_copy['high']
    low = df_copy['low']
    close = df_copy['close']
    volume = df_copy['volume']
    range_ = high - low
    fraction = (close - low) / range_.replace(0.0, np.nan)
    deviation = fraction - 0.5
    std_ = fraction.rolling(10, min_periods=10).std(ddof=1).replace(0.0, np.nan)
    rev = -deviation / std_
    vol_ratio = volume / volume.rolling(20).mean()
    vol_rank = vol_ratio.rolling(20, min_periods=20).rank(pct=True)
    factor = rev * vol_rank
    df_copy['factor_intraday_reversal_zscore_volume_short'] = factor
    return df_copy['factor_intraday_reversal_zscore_volume_short']
```

---

## Factor 299

**Formula:** `-mean((close-midpoint)/(high-low), 5) * (volume / SMA(volume,20))`

**Rationale:** Crossover borrows the primary mean reversion mechanism (smoothed intraday close position) from mutation_001 and adds a lightweight volume confirmation ratio from mutated_stability_002. The volume ratio amplifies the contrarian signal when volume is above its 20-day average, aiming to improve cross-sectional rank stability (RankIC/IR) while preserving the IC and ICIR strength of the mean reversion parent.

```python
def factor_volume_confirmed_intraday_mean_reversion(df):
    df_copy = df.copy()
    midpoint = (df_copy['high'] + df_copy['low']) / 2
    intraday_fraction = (df_copy['close'] - midpoint) / (df_copy['high'] - df_copy['low']).replace(0, np.nan)
    smoothed_intraday = intraday_fraction.rolling(5, min_periods=5).mean()
    volume_ma = df_copy['volume'].rolling(20, min_periods=10).mean()
    volume_ratio = df_copy['volume'] / volume_ma.replace(0, np.nan)
    factor = -smoothed_intraday * volume_ratio
    df_copy['factor_volume_confirmed_intraday_mean_reversion'] = factor
    return df_copy['factor_volume_confirmed_intraday_mean_reversion']
```

---

## Factor 302

**Formula:** `-mean((close-midpoint)/(high-low), 5)`

**Rationale:** Simplified mean reversion signal by ablating the volume confirmation ratio to reduce noise and improve cross-sectional rank stability. Preserves the core economic intuition of contrarian trading based on intraday close position within daily range over the past 5 days. Parent strength target: ICIR from mutation_001.

```python
def factor_simplified_intraday_mean_reversion(df):
    df_copy = df.copy()
    midpoint = (df_copy['high'] + df_copy['low']) / 2
    intraday_fraction = (df_copy['close'] - midpoint) / (df_copy['high'] - df_copy['low']).replace(0, np.nan)
    smoothed_intraday = intraday_fraction.rolling(5, min_periods=5).mean()
    factor = -smoothed_intraday
    df_copy['factor_simplified_intraday_mean_reversion'] = factor
    return df_copy['factor_simplified_intraday_mean_reversion']
```

---

## Factor 314

**Formula:** `EMA((C-O)*V, 21) / ATR(21) then rank[-1,1]`

**Rationale:** Volatility-adjusted dollar pressure measures directional dollar imbalance scaled by recent volatility, capturing risk-adjusted buying/selling pressure. EMA smoothing reduces noise, and ATR provides volatility normalization. The rank transform delivers a stable cross-sectional ordering suitable for balanced IC, RankIC, and MI.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_volatility_adjusted_dollar_pressure(df):
    df_copy = df.copy()
    dollar_imb = (df_copy['close'] - df_copy['open']) * df_copy['volume']
    smoothed_imb = dollar_imb.ewm(span=21, adjust=False).mean()
    high = df_copy['high'].values.astype(np.float64)
    low = df_copy['low'].values.astype(np.float64)
    close = df_copy['close'].values.astype(np.float64)
    atr = talib.ATR(high, low, close, timeperiod=21)
    atr_series = pd.Series(atr, index=df_copy.index)
    raw = smoothed_imb / atr_series.replace(0, np.nan)
    df_copy['factor_volatility_adjusted_dollar_pressure'] = raw.rank(pct=True) * 2 - 1
    return df_copy['factor_volatility_adjusted_dollar_pressure']
```

---

## Factor 315

**Formula:** `EMA((C-O)*V, 10) * (1 + (C(-1)-C)/C(-1)) / ATR(21) then rank[-1,1]`

**Rationale:** Reduced EMA span from 21 to 10 to increase responsiveness of dollar pressure signal, aiming to improve IC while preserving the core RankIC strength from the parent's regime-adjusted dollar pressure mechanism. The ATR normalization and rank transformation remain unchanged.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_fear_adjusted_dollar_pressure_short_ema(df):
    df_copy = df.copy()
    dollar_imb = (df_copy['close'] - df_copy['open']) * df_copy['volume']
    smoothed_imb = dollar_imb.ewm(span=10, adjust=False).mean()
    price_decline = (df_copy['close'].shift(1) - df_copy['close']) / df_copy['close'].shift(1)
    fear_adj_imb = smoothed_imb * (1 + price_decline)
    high = df_copy['high'].values.astype(np.float64)
    low = df_copy['low'].values.astype(np.float64)
    close = df_copy['close'].values.astype(np.float64)
    atr = talib.ATR(high, low, close, timeperiod=21)
    atr_series = pd.Series(atr, index=df_copy.index)
    raw = fear_adj_imb / atr_series.replace(0, np.nan)
    df_copy['factor_fear_adjusted_dollar_pressure_short_ema'] = raw.rank(pct=True) * 2 - 1
    return df_copy['factor_fear_adjusted_dollar_pressure_short_ema']
```

---

## Factor 317

**Formula:** `rank( SMA((C-O)*V,21) * (1+(C(-1)-C)/C(-1)) / ATR(21) ) * 2 - 1`

**Rationale:** Mutation: simplified fear-adjusted dollar pressure by removing the volume abnormality multiplier and replacing EMA with SMA (21) for smoothing. Preserves parent's strong RankIC and RankICIR from the core economic mechanism (dollar pressure amplified by short-term price decline, normalized by ATR). The SMA reduces responsiveness, targeting improved IC and ICIR by reducing noise. Single core mechanism with no modifiers for compactness.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_simplified_fear_pressure(df):
    df_copy = df.copy()
    # Dollar imbalance
    dollar_imb = (df_copy['close'] - df_copy['open']) * df_copy['volume']
    # Simple moving average of dollar imbalance (21 days) instead of EMA
    smoothed_imb = dollar_imb.rolling(21, min_periods=21).mean()
    # Short-term price decline (positive when price falls)
    price_decline = (df_copy['close'].shift(1) - df_copy['close']) / df_copy['close'].shift(1)
    # Fear-adjusted dollar pressure
    fear_adj_imb = smoothed_imb * (1 + price_decline)
    # ATR for normalization
    high = df_copy['high'].values.astype(np.float64)
    low = df_copy['low'].values.astype(np.float64)
    close = df_copy['close'].values.astype(np.float64)
    atr = talib.ATR(high, low, close, timeperiod=21)
    atr_series = pd.Series(atr, index=df_copy.index)
    raw = fear_adj_imb / atr_series.replace(0, np.nan)
    # Rank transform for cross-sectional stability
    df_copy['factor_simplified_fear_pressure'] = raw.rank(pct=True) * 2 - 1
    return df_copy['factor_simplified_fear_pressure']
```

---

## Factor 331

**Formula:** `((close - (high+low+close)/3) * volume) / rolling_mean(high-low,20) -> rolling_zscore(20)`

**Rationale:** Preserves the core dollar imbalance relative to volatility range, but replaces rank normalization with rolling z-score to improve linear correlation (IC) while maintaining the economic mechanism. The z-score may better capture cross-sectional ordering than rank pct in this context.

```python
def factor_dollar_pressure_range_zscore(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    typical_price = (high + low + close) / 3
    dollar_imbalance = (close - typical_price) * volume
    range_ = high - low
    range_mean = range_.rolling(20, min_periods=20).mean().replace(0, np.nan)
    raw = dollar_imbalance / range_mean
    raw_mean = raw.rolling(20, min_periods=20).mean()
    raw_std = raw.rolling(20, min_periods=20).std(ddof=1)
    zscore = (raw - raw_mean) / raw_std.replace(0, np.nan)
    df_copy['factor_dollar_pressure_range_zscore'] = zscore
    return df_copy['factor_dollar_pressure_range_zscore']
```

---

## Factor 332

**Formula:** `(close - ts_delay(close,5)) * (volume / ts_mean(volume,20))`

**Rationale:** Mutation of crossover_liquidity_mom_simple_2: replaced volume rolling median with rolling mean to potentially improve linear correlation (IC) by using a more responsive denominator, while preserving the core liquidity-gated momentum mechanism that provides RankIC stability. The parent strength is the volume-based rank ordering; the bottleneck is low IC. This small window change from median to mean aims to enhance the linear signal without adding complexity.

```python
def factor_liquidity_mom_simple_2_mutated(df):
    df_copy = df.copy()
    close = df_copy["close"]
    volume = df_copy["volume"]
    mom = close - close.shift(5)
    vol_ratio = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    factor = mom * vol_ratio
    df_copy["factor_liquidity_mom_simple_2_mutated"] = factor
    return df_copy["factor_liquidity_mom_simple_2_mutated"]
```

---

## Factor 343

**Formula:** `(0.5 - EWM((close - low) / (high - low), span=5)) * (volume / ts_mean(volume, 20))`

**Rationale:** Crossover of intraday reversal (parent1) and volume-scaled momentum (parent2). Preserves the reversal's rank-stable signal (RankIC) while using the volume ratio from parent2 to amplify reversals on high volume, targeting improved ICIR and MI. The product is economically meaningful: reversals are more reliable when accompanied by above-average volume.

```python
def factor_reversal_volume_gated(df):
    df_copy = df.copy()
    range_ = df_copy['high'] - df_copy['low']
    safe_range = range_.replace(0.0, np.nan)
    intra_pos = (df_copy['close'] - df_copy['low']) / safe_range
    smoothed = intra_pos.ewm(span=5, adjust=False).mean()
    volume_ratio = df_copy['volume'] / df_copy['volume'].rolling(20).mean().replace(0.0, np.nan)
    factor = (0.5 - smoothed) * volume_ratio
    df_copy['factor_reversal_volume_gated'] = factor
    return df_copy['factor_reversal_volume_gated']
```

---

## Factor 345

**Formula:** `(0.5 - EWMA((close-low)/(high-low), span=3)) * rank_63(volume/SMA_21(volume)) * (1.0 if ATR5/ATR20 > 1 else 0.5)`

**Rationale:** Crossover of reversal-volume-confirmed (qualified parent, strong RankIC and ICIR) with a volatility regime gate (roughness from rejected parent). The reversal mechanism and volume rank are preserved as primary. The roughness ratio (ATR5/ATR20) acts as a lightweight modifier: when short-term volatility exceeds long-term volatility (roughness>1), the signal is fully applied; otherwise it is halved. This targets improved MI by capturing nonlinear volatility-dependent reversal strength while maintaining the parent's rank stability.

**Tools:** library_functions=np.where, pd.concat

```python
def factor_reversal_volume_volregime(df):
    df_copy = df.copy()
    # Primary mechanism from crossover-reversal-volume-001: reversal with volume confirmation
    range_ = df_copy['high'] - df_copy['low']
    safe_range = range_.replace(0.0, np.nan)
    intra_pos = (df_copy['close'] - df_copy['low']) / safe_range
    smoothed = intra_pos.ewm(span=3, adjust=False).mean()
    reversal = 0.5 - smoothed
    vol_sma_21 = df_copy['volume'].rolling(21, min_periods=21).mean()
    vol_ratio = df_copy['volume'] / vol_sma_21.replace(0, np.nan)
    vol_rank = vol_ratio.rolling(63, min_periods=63).rank(pct=True)
    # Lightweight modifier from rough_mom_crossover_001: volatility regime gate via roughness
    tr = pd.concat([df_copy['high'] - df_copy['low'],
                    (df_copy['high'] - df_copy['close'].shift(1)).abs(),
                    (df_copy['low'] - df_copy['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr5 = tr.rolling(5, min_periods=5).mean()
    atr20 = tr.rolling(20, min_periods=20).mean()
    roughness = atr5 / atr20.replace(0.0, np.nan)
    regime = np.where(roughness > 1.0, 1.0, 0.5)
    factor = reversal * vol_rank * regime
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_reversal_volume_volregime'] = factor
    return df_copy['factor_reversal_volume_volregime']
```

---

## Factor 349

**Formula:** `(0.5 - EWMA((close-low)/(high-low), span=3)) * rank_21(volume/SMA_21(volume))`

**Rationale:** Mutated from crossover-reversal-volume-002 by removing the volatility regime modifier (roughness/ATR ratio) to simplify the factor to one core mechanism (intraday reversal) and one lightweight modifier (volume confirmation rank). The volume rank window is reduced from 63 to 21 to match the volume SMA window, improving consistency and reducing lookback. This targets improved IC and RankIC by focusing on the most robust reversal-volume interaction without the volatility regime complexity.

```python
def factor_reversal_volume_simple(df):
    df_copy = df.copy()
    # core: reversal via smoothed close position
    range_ = df_copy['high'] - df_copy['low']
    safe_range = range_.replace(0.0, np.nan)
    intra_pos = (df_copy['close'] - df_copy['low']) / safe_range
    smoothed = intra_pos.ewm(span=3, adjust=False).mean()
    reversal = 0.5 - smoothed
    # volume confirmation: rank of volume/SMA(21) over 21 days
    vol_sma_21 = df_copy['volume'].rolling(21, min_periods=21).mean()
    vol_ratio = df_copy['volume'] / vol_sma_21.replace(0, np.nan)
    vol_rank = vol_ratio.rolling(21, min_periods=21).rank(pct=True)
    # combine
    factor = reversal * vol_rank
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_reversal_volume_simple'] = factor
    return df_copy['factor_reversal_volume_simple']
```

---

## Factor 364

**Formula:** `ratio = log(avg_up_range / avg_down_range) over 10 days, min 5 periods`

**Rationale:** Mutated from range-asym-smoothed-v1: reduced window to 10 and min_periods to 5 for shorter-term asymmetry; replaced symmetric ratio with log ratio to enhance linearity; removed EMA smoothing to avoid over-dampening. Preserves parent's core asymmetric range mechanism while targeting improved IC and RankIC. Repaired to avoid infinite values by clipping division and log inputs.

**Tools:** library_functions=np.clip, np.log

```python
def factor_range_asym_log_short(df):
    df_copy = df.copy()
    up_close = df_copy['close'] > df_copy['close'].shift(1)
    down_close = ~up_close
    range_ = df_copy['high'] - df_copy['low']
    win = 10
    min_per = 5
    up_sum = range_.where(up_close, 0).rolling(win, min_periods=min_per).sum()
    down_sum = range_.where(down_close, 0).rolling(win, min_periods=min_per).sum()
    up_count = up_close.rolling(win, min_periods=min_per).sum()
    down_count = down_close.rolling(win, min_periods=min_per).sum()
    up_avg = up_sum / up_count.replace(0, np.nan)
    down_avg = down_sum / down_count.replace(0, np.nan)
    # Safe ratio: avoid division by zero by clipping down_avg lower bound
    ratio = up_avg / down_avg.clip(lower=1e-12)
    # Clip ratio to avoid log(0) or negative
    ratio_clipped = np.clip(ratio, 1e-12, None)
    factor = np.log(ratio_clipped)
    factor = factor.fillna(0)
    df_copy['factor_range_asym_log_short'] = factor
    return df_copy['factor_range_asym_log_short']
```

---

## Factor 365

**Formula:** `ratio = (avg_up_range - avg_down_range) / (avg_up_range + avg_down_range) over 20 days; vol_weight = vol_20_avg / rolling_max(vol_20_avg); factor = ratio * vol_weight then 5-period EMA smoothing`

**Rationale:** Crossover: preserves the volume-weighted asymmetric range mechanism from range-asym-vol-weight-v1 (strong RankIC/IC) and borrows EMA smoothing from range-asym-smoothed-v1 to reduce noise and improve MI. Targets improved mutual information while retaining rank stability.

```python
def factor_range_asym_vol_smooth(df):
    df_copy = df.copy()
    up_close = df_copy['close'] > df_copy['close'].shift(1)
    down_close = ~up_close
    range_ = df_copy['high'] - df_copy['low']
    win = 20
    min_per = 5
    up_sum = range_.where(up_close, 0).rolling(win, min_periods=min_per).sum()
    down_sum = range_.where(down_close, 0).rolling(win, min_periods=min_per).sum()
    up_count = up_close.rolling(win, min_periods=min_per).sum()
    down_count = down_close.rolling(win, min_periods=min_per).sum()
    up_avg = up_sum / up_count.replace(0, np.nan)
    down_avg = down_sum / down_count.replace(0, np.nan)
    ratio = (up_avg - down_avg) / (up_avg + down_avg + 1e-10)
    vol_avg = df_copy['volume'].rolling(20, min_periods=10).mean()
    vol_max = vol_avg.rolling(20, min_periods=10).max().replace(0, np.nan)
    vol_weight = vol_avg / vol_max
    factor = ratio * vol_weight
    factor = factor.ewm(span=5, adjust=False).mean()
    factor = factor.fillna(0)
    df_copy['factor_range_asym_vol_smooth'] = factor
    return df_copy['factor_range_asym_vol_smooth']
```

---

## Factor 366

**Formula:** `ratio = log(avg_up_range / avg_down_range) over 20 days; factor = ratio * (vol_20_avg / rolling_max(vol_20_avg))`

**Rationale:** Preserves the volume-weighted asymmetric range mechanism from parent (strength: rank stability) and targets improved IC by replacing symmetric ratio (difference/sum) with log ratio, enhancing linearity. Removes EMA smoothing to keep one lightweight modifier (volume weight) and maintain compactness, following mutation principle of minimal intervention.

**Tools:** library_functions=np.log

```python
def factor_range_asym_log_vol(df):
    df_copy = df.copy()
    up_close = df_copy['close'] > df_copy['close'].shift(1)
    down_close = ~up_close
    range_ = df_copy['high'] - df_copy['low']
    win = 20
    min_per = 5
    up_sum = range_.where(up_close, 0).rolling(win, min_periods=min_per).sum()
    down_sum = range_.where(down_close, 0).rolling(win, min_periods=min_per).sum()
    up_count = up_close.rolling(win, min_periods=min_per).sum()
    down_count = down_close.rolling(win, min_periods=min_per).sum()
    up_avg = up_sum / up_count.replace(0, np.nan)
    down_avg = down_sum / down_count.replace(0, np.nan)
    ratio = np.log((up_avg + 1e-12) / (down_avg + 1e-12))
    vol_avg = df_copy['volume'].rolling(20, min_periods=10).mean()
    vol_max = vol_avg.rolling(20, min_periods=10).max().replace(0, np.nan)
    vol_weight = vol_avg / vol_max
    factor = ratio * vol_weight
    factor = factor.fillna(0)
    df_copy['factor_range_asym_log_vol'] = factor
    return df_copy['factor_range_asym_log_vol']
```

---

## Factor 375

**Formula:** `log( SMA_5( sqrt(mean_{20}(neg_ret^2)) / sqrt(mean_{20}(pos_ret^2)) ) )`

**Rationale:** Smoothed downside volatility asymmetry: apply a 5-day moving average to the ratio of downside to upside volatility to reduce noise and improve linear correlation with returns, targeting IC and MI improvement while preserving the stable rank characteristics from the parent.

**Tools:** library_functions=np.log, np.sqrt

```python
def factor_downside_vol_asymmetry_smoothed(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    negative_sq = ret.where(ret < 0, 0) ** 2
    positive_sq = ret.where(ret > 0, 0) ** 2
    window = 20
    down_var = negative_sq.rolling(window=window, min_periods=window).mean()
    up_var = positive_sq.rolling(window=window, min_periods=window).mean()
    ratio = np.sqrt(down_var) / np.sqrt(up_var.replace(0, np.nan))
    smoothed_ratio = ratio.rolling(window=5, min_periods=5).mean()
    factor = np.log(smoothed_ratio + 1e-10)
    factor.name = "factor_downside_vol_asymmetry_smoothed"
    return factor.fillna(0)
```

---

## Factor 389

**Formula:** `if close > SMA20 then z-score of 10-day momentum (over 60-day window) else intraday mean reversion`

**Rationale:** Mutation of MUT-003: preserve regime-switch core but normalize momentum component via trailing z-score to improve ICIR and IC stability. Keep reversal unchanged. Targeted at improving RankIC and ICIR while maintaining strong MI.

**Tools:** library_functions=np.where

```python
def factor_regime_switch_zscore_momentum(df):
    df_copy = df.copy()
    mom = df_copy['close'] / df_copy['close'].shift(10) - 1
    mom_mean = mom.rolling(60, min_periods=10).mean()
    mom_std = mom.rolling(60, min_periods=10).std(ddof=1)
    mom_z = (mom - mom_mean) / mom_std.replace(0.0, np.nan)
    rev = (df_copy['open'] - df_copy['close']) / df_copy['open']
    sma20 = df_copy['close'].rolling(20).mean()
    trend_up = df_copy['close'] > sma20
    factor = np.where(trend_up, mom_z, rev)
    df_copy['factor_regime_switch_zscore_momentum'] = factor
    return df_copy['factor_regime_switch_zscore_momentum']
```

---

## Factor 390

**Formula:** `drawdown = (close - rolling_max(close,60)) / rolling_max(close,60); vol = rolling_std(close,20); factor = - drawdown / vol`

**Rationale:** Simplified drawdown depth normalized by volatility, removing the volume trend component to focus on the core mean-reversion mechanism. The drawdown (relative to 60-day high) divided by 20-day volatility captures the severity of recent decline relative to recent noise. Negative sign ensures that deep drawdown yields a positive signal. The ablation reduces complexity and aims to improve IC and ICIR while preserving the rank-order stability (RankIC) from the parent. Cross-sectional rank is retained to standardize relative comparison across stocks.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_drawdown_normalized(df):
    df_copy = df.copy()
    trailing_high = df_copy['close'].rolling(60, min_periods=60).max()
    drawdown = (df_copy['close'] - trailing_high) / trailing_high
    vol = df_copy['close'].rolling(20, min_periods=20).std(ddof=1)
    vol = vol.replace(0.0, np.nan)
    factor = - drawdown / vol
    factor = factor.replace([np.inf, -np.inf], np.nan)
    df_copy['factor_drawdown_normalized'] = factor
    return df_copy['factor_drawdown_normalized']
```

---

## Factor 392

**Formula:** `if close > SMA20 then (close/shift(close,10)-1) else ((open-close)/open)`

**Rationale:** Simplified adaptive regime-switching factor: in uptrend (close above 20-day SMA) use 10-day momentum, otherwise use intraday mean reversion (open-close)/open. Removes volatility normalization to reduce noise and target improved IC and MI while preserving the core economic intuition of the parent and RankIC stability.

**Tools:** library_functions=np.where

```python
def factor_regime_switch_simple(df):
    df_copy = df.copy()
    sma20 = df_copy['close'].rolling(20, min_periods=20).mean()
    mom = df_copy['close'] / df_copy['close'].shift(10) - 1
    rev = (df_copy['open'] - df_copy['close']) / df_copy['open']
    trend_up = df_copy['close'] > sma20
    factor = np.where(trend_up, mom, rev)
    df_copy['factor_regime_switch_simple'] = factor
    return df_copy['factor_regime_switch_simple']
```

---

## Factor 402

**Formula:** `factor = close_10d_return * (volume / rolling_20d_mean_volume - 1)`

**Rationale:** Mutation of APVC-MUT-001: replaces robust z-score (median/MAD) with a simple volume deviation ratio (volume / rolling mean - 1) to simplify and reduce computational overhead while preserving the core price-volume coherence mechanism. The parent strength in RankIC is expected to be maintained because the relative ordering of the product remains similar, while IC may benefit from less aggressive clipping of volume extremes.

```python
def factor_volume_confirmed_momentum_simple_volume_deviation(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(10)
    vol_mean = df_copy['volume'].rolling(20).mean()
    vol_dev = df_copy['volume'] / vol_mean.replace(0, np.nan) - 1
    factor = ret * vol_dev
    factor.name = "factor_volume_confirmed_momentum_simple_volume_deviation"
    return factor
```

---

## Factor 403

**Formula:** `factor = (close_10d_return) * robust_z(volume, 20d) * (1 - downside_stress_ratio(20d))`

**Rationale:** Mutation of CROSS-001: preserves the core price-volume coherence mechanism (10-day return multiplied by volume deviation) but replaces the standard volume z-score (mean/std) with a robust z-score using median and MAD. This reduces the influence of outlier volume observations, aiming to improve IC and MI stability without disrupting the RankIC strength of the parent.

```python
def factor_volume_stress_adjusted_momentum_robust(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(10)
    vol_median = df_copy['volume'].rolling(20).median()
    vol_mad = (df_copy['volume'] - vol_median).abs().rolling(20).median()
    vol_robust_z = (df_copy['volume'] - vol_median) / (vol_mad * 1.4826).replace(0, np.nan)
    vol_robust_z = vol_robust_z.clip(-3, 3)
    returns = df_copy['close'].pct_change()
    neg_returns = returns.clip(upper=0)
    downside_std = neg_returns.rolling(20, min_periods=20).std()
    total_std = returns.rolling(20, min_periods=20).std()
    ratio = downside_std / total_std.replace(0, np.nan)
    ratio = ratio.fillna(0.5)
    factor = ret * vol_robust_z * (1 - ratio)
    factor.name = "factor_volume_stress_adjusted_momentum_robust"
    return factor
```

---

## Factor 405

**Formula:** `factor = close_10d_return * robust_zscore(volume, 20d)`

**Rationale:** Mutation of CROSS-002: preserves the core robust volume momentum mechanism (10-day return multiplied by robust z-score of volume relative to 20-day median/MAD). Removes the downside stress dampener to simplify and focus on the primary price-volume coherence signal. This targets improved interpretability and metric stability by reducing unnecessary complexity. The parent strength of robust volume z-score for outlier resistance is preserved.

```python
def factor_simplified_robust_volume_momentum(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(10)
    vol_median = df_copy['volume'].rolling(20).median()
    vol_mad = (df_copy['volume'] - vol_median).abs().rolling(20).median()
    vol_robust_z = (df_copy['volume'] - vol_median) / vol_mad.replace(0, np.nan)
    vol_robust_z = vol_robust_z.clip(-3, 3)
    factor = ret * vol_robust_z
    factor.name = "factor_simplified_robust_volume_momentum"
    return factor
```

---

## Factor 407

**Formula:** `factor = pct_change(close,10) * robust_zscore(volume,20) * persistence(10)`

**Rationale:** Mutation of CROSS-002: preserves the core price-volume coherence mechanism (10-day return multiplied by volume deviation) but replaces the standard z-score (mean/std) with a robust z-score using median and MAD. This reduces the influence of outlier volume observations, aiming to improve IC stability without disrupting the RankICIR strength of the parent.

```python
def factor_volume_persistence_momentum_robust(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change(10)
    vol_med = df_copy['volume'].rolling(20).median()
    vol_mad = (df_copy['volume'] - vol_med).abs().rolling(20).median()
    vol_robust_z = (df_copy['volume'] - vol_med) / vol_mad.replace(0, np.nan)
    vol_robust_z = vol_robust_z.clip(-3, 3)
    daily_ret = df_copy['close'].pct_change()
    persistence = (daily_ret > 0).astype(float).rolling(10).sum() / 10
    factor = ret * vol_robust_z * persistence
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = "factor_volume_persistence_momentum_robust"
    return factor
```

---

## Factor 427

**Formula:** `EMA( (close/shift(close,15)-1) * (volume/rolling_mean(volume,15)), 5 )`

**Rationale:** Increased windows from 10 to 15 for momentum and volume ratio, and EMA span from 3 to 5, to capture longer-term volume-confirmed trends. Targets improved IC while preserving the parent's RankIC stability. Mutation from MUT-001 (window adjustment).

**Tools:** cross_sectional_transform=cs_zscore

```python
def factor_window_adjusted_momentum(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    momentum = close / close.shift(15) - 1
    vol_mean = volume.rolling(15, min_periods=15).mean()
    vol_regime = volume / vol_mean.replace(0.0, np.nan)
    factor = momentum * vol_regime
    smoothed = factor.ewm(span=5, adjust=False).mean()
    smoothed = smoothed.fillna(0.0)
    df_copy['factor_window_adjusted_momentum'] = smoothed
    return df_copy['factor_window_adjusted_momentum']
```

---

## Factor 428

**Formula:** `log(ts_std(neg_ret,20)/ts_std(pos_ret,20)) * (volume/ts_mean(volume,20)) smoothed with ewm(span=5)`

**Rationale:** Crossover between M-001 (asymmetric volatility smoothed) and CROSS-001 (volume ratio from volume-momentum factor). Preserves M-001's core mechanism of downside vs upside volatility asymmetry measured by log ratio of rolling standard deviations of negative and positive returns, smoothed with EMA. Borrows CROSS-001's volume ratio (volume / 20-day mean) as a continuous multiplier to amplify the asymmetry signal during high volume periods, enhancing information content and aiming to improve IC while maintaining rank stability.

**Tools:** library_functions=np.log

```python
def factor_asym_vol_volume_confirmed(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    returns = close.pct_change()
    pos_returns = returns.where(returns > 0, 0.0)
    neg_returns = abs(returns.where(returns < 0, 0.0))
    pos_vol = pos_returns.rolling(20, min_periods=10).std()
    neg_vol = neg_returns.rolling(20, min_periods=10).std()
    ratio = neg_vol / pos_vol.replace(0, np.nan)
    log_ratio = np.log(ratio.clip(lower=1e-12))
    smoothed = log_ratio.ewm(span=5, min_periods=5, adjust=False).mean()
    vol_ratio = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    factor = smoothed * vol_ratio
    df_copy['factor_asym_vol_volume_confirmed'] = factor
    return df_copy['factor_asym_vol_volume_confirmed']
```

---

## Factor 432

**Formula:** `log(ts_std(neg_ret,20)/ts_std(pos_ret,20)) smoothed(span=5) * (1 if volume > ts_mean(volume,20) else 0.5)`

**Rationale:** Mutation of XO-003: preserve asymmetric volatility core (log ratio of downside to upside rolling std) with EMA smoothing, but replace continuous volume ratio with a binary high-volume indicator (1.0 when volume above 20-day MA, 0.5 otherwise). This reduces noise from low-volume periods, targeting improvement in IC while maintaining rank stability from the core mechanism.

**Tools:** library_functions=np.log, np.where

```python
def factor_asym_vol_high_volume_confirmed(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    returns = close.pct_change()
    pos_returns = returns.where(returns > 0, 0.0)
    neg_returns = abs(returns.where(returns < 0, 0.0))
    pos_vol = pos_returns.rolling(20, min_periods=10).std()
    neg_vol = neg_returns.rolling(20, min_periods=10).std()
    ratio = neg_vol / pos_vol.replace(0, np.nan)
    log_ratio = np.log(ratio.clip(lower=1e-12))
    smoothed = log_ratio.ewm(span=5, min_periods=5, adjust=False).mean()
    vol_ma = volume.rolling(20, min_periods=20).mean()
    vol_confirmation = np.where(volume > vol_ma, 1.0, 0.5)
    factor = smoothed * vol_confirmation
    df_copy['factor_asym_vol_high_volume_confirmed'] = factor
    return df_copy['factor_asym_vol_high_volume_confirmed']
```

---

## Factor 445

**Formula:** `-(close - max(close, 60)) / max(close, 60) / ATR(20)`

**Rationale:** This mutation adjusts the ATR normalization window from 14 to 20 days to smooth volatility estimates, potentially improving IC and ICIR while preserving the core drawdown-based oversold detection and cross-sectional rank stability. The longer ATR period reduces noise in the volatility denominator, which may enhance linear predictability and consistency of the signal.

**Tools:** cross_sectional_transform=cs_rank; library_functions=pd.Series, talib.ATR

```python
def factor_drawdown_atr_normalized_20(df):
    df_copy = df.copy()
    rolling_max = df_copy["close"].rolling(window=60, min_periods=60).max()
    drawdown = (df_copy["close"] - rolling_max) / rolling_max
    high = df_copy["high"].values.astype(float)
    low = df_copy["low"].values.astype(float)
    close = df_copy["close"].values.astype(float)
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=20), index=df_copy.index)
    normalized = -drawdown / atr.replace(0, np.nan)
    df_copy["factor_drawdown_atr_normalized_20"] = normalized.fillna(0)
    return df_copy["factor_drawdown_atr_normalized_20"]
```

---

## Factor 446

**Formula:** `(close - max(close,60)) / max(close,60) / std(close,60)`

**Rationale:** Mutation from drawdown_std_norm_001: increased volatility normalization window from 20 to 60 days to match the drawdown lookback period. A longer volatility window provides a more consistent risk adjustment, reducing noise in the normalization and potentially improving IC and ICIR while preserving the rank stability (parent strength: RankIC=0.0378, RankICIR=0.2959). The core drawdown mechanism remains unchanged.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_drawdown_std60_normalized(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(window=60, min_periods=60).max()
    drawdown = (df_copy['close'] - rolling_max) / rolling_max
    vol = df_copy['close'].rolling(window=60, min_periods=60).std(ddof=1)
    normalized_drawdown = drawdown / vol.replace(0, np.nan)
    df_copy['factor_drawdown_std60_normalized'] = normalized_drawdown.fillna(0)
    return df_copy['factor_drawdown_std60_normalized']
```

---

## Factor 448

**Formula:** `(-(close - max(close,60)) / max(close,60)) / ATR(60)`

**Rationale:** Crossover: primary mechanism from drawdown_atr_norm_001 (drawdown depth normalized by 60-period ATR), with sign flip from alpha-mutation-001 to make oversold severity a positive signal. The 60-period ATR aligns with the drawdown window for consistent risk adjustment, while the negative sign improves direction interpretation as oversold buying opportunity. Cross-sectional rank emphasizes relative severity.

**Tools:** cross_sectional_transform=cs_rank; library_functions=pd.Series, talib.ATR

```python
def factor_drawdown_atr_normalized_v2(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(window=60, min_periods=60).max()
    drawdown = (df_copy['close'] - rolling_max) / rolling_max
    high = df_copy['high'].values.astype(float)
    low = df_copy['low'].values.astype(float)
    close = df_copy['close'].values.astype(float)
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=60), index=df_copy.index)
    normalized = -drawdown / atr.replace(0, np.nan)
    df_copy['factor_drawdown_atr_normalized_v2'] = normalized.fillna(0)
    return df_copy['factor_drawdown_atr_normalized_v2']
```

---

## Factor 450

**Formula:** `(-(close - max(close,60)) / max(close,60)) / ATR(20)`

**Rationale:** Mutation from crossover_drawdown_atr_002: reduced ATR window from 60 to 20 to increase responsiveness to recent volatility, potentially improving IC while preserving the core drawdown-volatility normalization and cs_rank transform for cross-sectional stability. The drawdown window remains at 60 to maintain the original oversold detection horizon.

**Tools:** cross_sectional_transform=cs_rank; library_functions=pd.Series, talib.ATR

```python
def factor_drawdown_atr_20_normalized(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(window=60, min_periods=60).max()
    drawdown = (df_copy['close'] - rolling_max) / rolling_max
    high = df_copy['high'].values.astype(float)
    low = df_copy['low'].values.astype(float)
    close = df_copy['close'].values.astype(float)
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=20), index=df_copy.index)
    normalized = -drawdown / atr.replace(0, np.nan)
    df_copy['factor_drawdown_atr_20_normalized'] = normalized.fillna(0)
    return df_copy['factor_drawdown_atr_20_normalized']
```

---

## Factor 467

**Formula:** `mean_5( if(close > max_10(high).shift(1), (close/delay(close,1)-1) * (volume/mean_10(volume)), 0 ) )`

**Rationale:** Mutation of apvc_mutation_001: shortened volume ratio window from 20 to 10 to align with breakout horizon, and changed aggregation from sum to mean to produce a smoother signal. Aims to improve ICIR and IC while preserving RankIC strength.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_volume_adjusted_breakout_v3(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(1) - 1
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(10, min_periods=10).mean()
    highest_high = df_copy['high'].rolling(10, min_periods=10).max().shift(1)
    breakout = df_copy['close'] > highest_high
    signal = breakout * ret * vol_ratio
    factor = signal.rolling(5).mean()
    factor.name = 'factor_volume_adjusted_breakout_v3'
    return factor
```

---

## Factor 468

**Formula:** `sum_5( if(close > max_20(high).shift(1) and volume > mean_20(volume) and log(vol_ratio_risk_10) < median_20(log(vol_ratio_risk_10)), (close/delay(close,1)-1) * (volume/mean_20(volume)), 0 ) )`

**Rationale:** Mutation of crossover_volume_asymmetry_001: Shortens the volatility estimation window for downside and upside return std from 20 to 10 days to increase responsiveness of the risk filter, aiming to improve IC and ICIR while preserving the core breakout mechanism and RankIC strength.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_volume_adjusted_breakout_window_001(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    highest_high = df_copy['high'].rolling(20, min_periods=20).max().shift(1)
    breakout = df_copy['close'] > highest_high
    neg_ret = ret.where(ret < 0, 0)
    pos_ret = ret.where(ret > 0, 0)
    neg_vol = neg_ret.rolling(10, min_periods=5).std(ddof=1)
    pos_vol = pos_ret.rolling(10, min_periods=5).std(ddof=1)
    vol_ratio_risk = neg_vol / pos_vol.replace(0, np.nan)
    vol_ratio_risk = np.log(vol_ratio_risk.clip(lower=1e-12))
    vol_risk_median = vol_ratio_risk.rolling(20, min_periods=20).median()
    low_risk = vol_ratio_risk < vol_risk_median
    coherence_signal = (breakout & (vol_ratio > 1) & low_risk) * ret * vol_ratio
    factor = coherence_signal.rolling(5).sum()
    factor.name = 'factor_volume_adjusted_breakout_window_001'
    return factor
```

---

## Factor 470

**Formula:** `sum_5( if(close > max_20(high).shift(1) and volume > mean_20(volume) and log(neg_std/pos_std) < median_20(log(neg_std/pos_std)), (close/delay(close,10)-1) * (volume/mean_20(volume)), 0 ) )`

**Rationale:** Crossover of qualified parent1 (volume-adjusted breakout with downside risk filter) and rejected parent2 (10-day momentum with volume weight). Preserve parent1's core breakout and low-risk filter, but replace daily return with 10-day return from parent2 to capture longer trend momentum, aiming to improve IC while maintaining the strong RankICIR from parent1.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_volume_adjusted_breakout_with_vol_filter_v2(df):
    df_copy = df.copy()
    ret_10 = df_copy['close'].pct_change(10)
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    highest_high = df_copy['high'].rolling(20, min_periods=20).max().shift(1)
    breakout = df_copy['close'] > highest_high
    ret_daily = df_copy['close'].pct_change()
    neg_ret = ret_daily.where(ret_daily < 0, 0)
    pos_ret = ret_daily.where(ret_daily > 0, 0)
    neg_vol = neg_ret.rolling(20, min_periods=10).std(ddof=1)
    pos_vol = pos_ret.rolling(20, min_periods=10).std(ddof=1)
    vol_ratio_risk = neg_vol / pos_vol.replace(0, np.nan)
    vol_ratio_risk = np.log(vol_ratio_risk.clip(lower=1e-12))
    vol_risk_median = vol_ratio_risk.rolling(20, min_periods=20).median()
    low_risk = vol_ratio_risk < vol_risk_median
    coherence_signal = (breakout & (vol_ratio > 1) & low_risk) * ret_10 * vol_ratio
    factor = coherence_signal.rolling(5).sum()
    factor.name = 'factor_volume_adjusted_breakout_with_vol_filter_v2'
    return factor
```

---

## Factor 474

**Formula:** `sum_5( if(close > max_10(high).shift(1) and volume > mean_10(volume) and log(neg_std_20/pos_std_20) < median_20(...), (close/delay(close,5)-1) * (volume/mean_10(volume)), 0 ) )`

**Rationale:** Mutation of crossover_volume_asymmetry_002: Shortened breakout and volume confirmation windows from 20 to 10 to increase signal responsiveness, and shortened the return horizon from 10 to 5 days to capture shorter-term momentum. The goal is to improve IC while preserving the strong RankIC and RankICIR from the parent's risk filter mechanism.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_volume_adjusted_breakout_v3(df):
    df_copy = df.copy()
    ret_5 = df_copy['close'].pct_change(5)
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(10, min_periods=10).mean()
    highest_high = df_copy['high'].rolling(10, min_periods=10).max().shift(1)
    breakout = df_copy['close'] > highest_high
    ret_daily = df_copy['close'].pct_change()
    neg_ret = ret_daily.where(ret_daily < 0, 0)
    pos_ret = ret_daily.where(ret_daily > 0, 0)
    neg_vol = neg_ret.rolling(20, min_periods=10).std(ddof=1)
    pos_vol = pos_ret.rolling(20, min_periods=10).std(ddof=1)
    vol_ratio_risk = neg_vol / pos_vol.replace(0, np.nan)
    vol_ratio_risk = np.log(vol_ratio_risk.clip(lower=1e-12))
    vol_risk_median = vol_ratio_risk.rolling(20, min_periods=20).median()
    low_risk = vol_ratio_risk < vol_risk_median
    coherence_signal = (breakout & (vol_ratio > 1) & low_risk) * ret_5 * vol_ratio
    factor = coherence_signal.rolling(5).sum()
    factor.name = 'factor_volume_adjusted_breakout_v3'
    return factor
```

---

## Factor 488

**Formula:** `log(sigma_down(30)/sigma_up(30)) * (volume / MA_volume(20))`

**Rationale:** Continuous volume scaling replaces hard threshold to capture finer participation dynamics, enhancing nonlinear interaction while preserving asymmetric volatility as core risk signal. Targets improved IC and MI, maintains RankIC stability.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_asym_vol_volume_cont_30(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    returns = close.pct_change()
    up_returns = returns.copy()
    up_returns[up_returns < 0] = 0
    down_returns = returns.copy()
    down_returns[down_returns > 0] = 0
    up_vol = up_returns.rolling(30, min_periods=30).std()
    down_vol = down_returns.abs().rolling(30, min_periods=30).std()
    asym = np.log(down_vol / up_vol.replace(0, np.nan))
    vol_ma = volume.rolling(20, min_periods=20).mean()
    vol_ratio = volume / vol_ma.replace(0, np.nan)
    factor = asym * vol_ratio
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_asym_vol_volume_cont_30'] = factor
    return df_copy['factor_asym_vol_volume_cont_30']
```

---

## Factor 489

**Formula:** `log(down_vol(60) / up_vol(60)) * (volume / MA_volume(20))`

**Rationale:** Mutation: replaced binary volume gate (1.0 or 0.5) with a continuous volume ratio (volume / 20-day MA) to improve IC and ICIR by capturing finer participation dynamics, while preserving the parent's core asymmetric volatility mechanism and rank stability. The ratio is clipped to avoid extreme values.

**Tools:** cross_sectional_transform=cs_zscore; library_functions=np.log

```python
def factor_asym_vol_cont_gate_30(df):
    df_copy = df.copy()
    returns = df_copy['close'].pct_change()
    down_returns = returns.clip(upper=0)
    up_returns = returns.clip(lower=0)
    down_vol = down_returns.rolling(60, min_periods=20).std(ddof=0).clip(lower=1e-10)
    up_vol = up_returns.rolling(60, min_periods=20).std(ddof=0).clip(lower=1e-10)
    asym_vol = np.log(down_vol / up_vol)
    vol_ma = df_copy['volume'].rolling(20, min_periods=20).mean().replace(0, np.nan)
    vol_ratio = (df_copy['volume'] / vol_ma).clip(upper=3.0, lower=0.0)
    factor = asym_vol * vol_ratio
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_asym_vol_cont_gate_30'] = factor
    return df_copy['factor_asym_vol_cont_gate_30']
```

---

## Factor 491

**Formula:** `log(down_vol(30)/up_vol(30))`

**Rationale:** Asymmetric volatility over 30-day window captures downside vs upside risk, simplified from the parent by removing the volume z-score which added noise. A shorter window improves responsiveness while maintaining the core risk asymmetry. Cross-sectional rank identifies stocks with greatest fragility relative to peers.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_asym_vol_30(df):
    df_copy = df.copy()
    returns = df_copy['close'].pct_change()
    down_returns = returns.clip(upper=0)
    up_returns = returns.clip(lower=0)
    down_vol = down_returns.rolling(30, min_periods=15).std(ddof=0)
    up_vol = up_returns.rolling(30, min_periods=15).std(ddof=0)
    asym_vol = np.log(down_vol / up_vol.replace(0, np.nan))
    df_copy['factor_asym_vol_30'] = asym_vol
    return df_copy['factor_asym_vol_30']
```

---

## Factor 501

**Formula:** `- (SMA(high-low,5) / SMA(high-low,20)) * I(volume > SMA(volume,30)*1.5)`

**Rationale:** Repair of vol_regime_mut_001: replaced failing active alpha_tools call with manual volume regime classification using rolling mean and thresholds. The factor now captures expanding volatility ratio only when volume is in a high regime (above 1.5x its 30-day average), reducing noise and focusing on informed liquidity flow events. Preserves the core volatility ratio mechanism and the negative sign for mean reversion.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_vol_regime_vol_ratio(df):
    df_copy = df.copy()
    short_vol = (df_copy["high"] - df_copy["low"]).rolling(5).mean()
    long_vol = (df_copy["high"] - df_copy["low"]).rolling(20).mean()
    vol_ratio = short_vol / long_vol.replace(0, np.nan)
    vol_mean = df_copy["volume"].rolling(30).mean()
    high_vol_mask = (df_copy["volume"] > vol_mean * 1.5).astype(float)
    factor = -vol_ratio * high_vol_mask
    df_copy["factor_vol_regime_vol_ratio"] = factor
    return df_copy["factor_vol_regime_vol_ratio"]
```

---

## Factor 503

**Formula:** `log(1+down_vol_60) - log(1+up_vol_60) * (volume / vol_ma60) * ((high - low) / close)`

**Rationale:** Crossover of asymmetric volatility (parent 1) with relative range (parent 2) to enhance detection of informed selling pressure. The primary mechanism captures downside vs upside volatility asymmetry with volume confirmation over 60 days. The borrowed lightweight idea adds the relative intraday range (high-low)/close, amplifying the signal when the range is wide, consistent with informed liquidity flow. Target: improve IC and RankIC by incorporating range-based intensity.

**Tools:** cross_sectional_transform=cs_zscore; library_functions=np.isnan, np.log, np.sqrt, np.where, pd.Series

```python
def factor_vol_asym_confirmed_range(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down_ret = np.where(ret < 0, ret, np.nan)
    down_var = pd.Series(np.where(np.isnan(down_ret), 0, down_ret**2), index=df_copy.index).rolling(60, min_periods=40).mean()
    down_vol = np.sqrt(down_var)
    up_ret = np.where(ret > 0, ret, np.nan)
    up_var = pd.Series(np.where(np.isnan(up_ret), 0, up_ret**2), index=df_copy.index).rolling(60, min_periods=40).mean()
    up_vol = np.sqrt(up_var)
    asym = np.log(1 + down_vol) - np.log(1 + up_vol)
    vol_ma60 = df_copy["volume"].rolling(60, min_periods=60).mean()
    vol_ratio = df_copy["volume"] / vol_ma60.replace(0, np.nan)
    rel_range = (df_copy["high"] - df_copy["low"]) / df_copy["close"].replace(0, np.nan)
    factor = asym * vol_ratio * rel_range
    df_copy["factor_vol_asym_confirmed_range"] = factor
    return df_copy["factor_vol_asym_confirmed_range"]
```

---

## Factor 505

**Formula:** `log(1+down_vol_60) - log(1+up_vol_60) * (volume / vol_ma40) * ((high - low) / typical_price)`

**Rationale:** Mutation of alpha_crossover_002. Preserve core asymmetric volatility with volume confirmation, and adjust the relative range denominator from close to typical price to reduce noise. Reduce volume MA window from 60 to 40 for more responsive volume confirmation. Target: improve RankIC and ICIR by making the volume signal more adaptive and the range less sensitive to closing price fluctuations.

**Tools:** cross_sectional_transform=cs_zscore; library_functions=np.isnan, np.log, np.sqrt, np.where, pd.Series

```python
def factor_vol_asym_confirmed_range_v2(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down_ret = np.where(ret < 0, ret, np.nan)
    down_var = pd.Series(np.where(np.isnan(down_ret), 0, down_ret**2), index=df_copy.index).rolling(60, min_periods=40).mean()
    down_vol = np.sqrt(down_var)
    up_ret = np.where(ret > 0, ret, np.nan)
    up_var = pd.Series(np.where(np.isnan(up_ret), 0, up_ret**2), index=df_copy.index).rolling(60, min_periods=40).mean()
    up_vol = np.sqrt(up_var)
    asym = np.log(1 + down_vol) - np.log(1 + up_vol)
    vol_ma40 = df_copy["volume"].rolling(40, min_periods=40).mean()
    vol_ratio = df_copy["volume"] / vol_ma40.replace(0, np.nan)
    typical_price = (df_copy["high"] + df_copy["low"] + df_copy["close"]) / 3.0
    rel_range = (df_copy["high"] - df_copy["low"]) / typical_price.replace(0, np.nan)
    factor = asym * vol_ratio * rel_range
    df_copy["factor_vol_asym_confirmed_range_v2"] = factor
    return df_copy["factor_vol_asym_confirmed_range_v2"]
```

---

## Factor 517

**Formula:** `log(avg_down_range / avg_up_range) over 20 days`

**Rationale:** Simplifies the asymmetric volatility factor by removing the volume confirmation modifier, focusing solely on the core asymmetry mechanism. This reduces noise and may improve IC and MI while preserving the RankIC strength of the parent.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.log

```python
def factor_asym_ratio_pure(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down = ret < 0
    up = ret >= 0
    range_ = df_copy["high"] - df_copy["low"]
    down_range = range_.where(down, 0)
    up_range = range_.where(up, 0)
    down_sum = down_range.rolling(20, min_periods=10).sum()
    up_sum = up_range.rolling(20, min_periods=10).sum()
    down_count = down.rolling(20, min_periods=10).sum()
    up_count = up.rolling(20, min_periods=10).sum()
    down_avg = down_sum / down_count.replace(0, np.nan)
    up_avg = up_sum / up_count.replace(0, np.nan)
    ratio = down_avg / up_avg.replace(0, np.nan)
    factor = np.log(ratio.clip(lower=1e-6))
    df_copy["factor_asym_ratio_pure"] = factor
    return df_copy["factor_asym_ratio_pure"]
```

---

## Factor 519

**Formula:** `log(avg_down_range/avg_up_range) * (1.2 if volume_regime == high else 1) with window 15`

**Rationale:** Mutates asymmetric volatility factor by shortening the trailing window from 20 to 15 bars for the asymmetry calculation, making the signal more responsive to recent volatility regimes. The volume-based amplification is slightly reduced from 1.3 to 1.2 to preserve rank stability while improving IC and MI capture. The core mechanism and volume regime gate are preserved.

**Tools:** cross_sectional_transform=cs_rank; library_functions=alpha_tools.classify_volume_regime, np.log, np.where

```python
def factor_asym_vol_regime(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down = ret < 0
    up = ret >= 0
    range_ = df_copy["high"] - df_copy["low"]
    down_range = range_.where(down, 0)
    up_range = range_.where(up, 0)
    down_sum = down_range.rolling(15, min_periods=8).sum()
    up_sum = up_range.rolling(15, min_periods=8).sum()
    down_count = down.rolling(15, min_periods=8).sum()
    up_count = up.rolling(15, min_periods=8).sum()
    down_avg = down_sum / down_count.replace(0, np.nan)
    up_avg = up_sum / up_count.replace(0, np.nan)
    ratio = down_avg / up_avg.replace(0, np.nan)
    asym = np.log(ratio.clip(lower=1e-6))
    vol_regime = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    factor = np.where(vol_regime == 1, asym * 1.2, asym)
    df_copy["factor_asym_vol_regime"] = factor
    return df_copy["factor_asym_vol_regime"]
```

---

## Factor 521

**Formula:** `log( down_avg/up_avg ) with volume regime amplification`

**Rationale:** Preserves the core asymmetry mechanism (down-day vs up-day range ratio) from the parent, but simplifies the volume component by replacing continuous volume weighting with a binary regime gate using classify_volume_regime. This reduces noise from arbitrary volume scaling and applies amplification/attenuation only in high/low volume states, targeting better robustness and stability. The log ratio captures nonlinear asymmetry; cs_rank enhances cross-sectional differentiation.

**Tools:** cross_sectional_transform=cs_rank; library_functions=alpha_tools.classify_volume_regime, np.log, np.where

```python
def factor_asym_vol_gated(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down = ret < 0
    up = ret >= 0
    range_ = df_copy["high"] - df_copy["low"]
    down_range = range_.where(down, 0)
    up_range = range_.where(up, 0)
    down_avg = down_range.rolling(20, min_periods=10).sum() / down.rolling(20, min_periods=10).sum().replace(0, np.nan)
    up_avg = up_range.rolling(20, min_periods=10).sum() / up.rolling(20, min_periods=10).sum().replace(0, np.nan)
    ratio = down_avg / up_avg.replace(0, np.nan)
    raw_asym = np.log(ratio.clip(lower=1e-6))
    volume_regime = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    high_vol = volume_regime == "high"
    low_vol = volume_regime == "low"
    factor = np.where(high_vol, raw_asym * 1.5, np.where(low_vol, raw_asym * 0.5, raw_asym))
    df_copy["factor_asym_vol_gated"] = factor
    return df_copy["factor_asym_vol_gated"]
```

---

## Factor 522

**Formula:** `log(avg_down_range/avg_up_range) * (1 + 0.3 * clip(vol_dev,-1,1)) * (0.7 if low_volume_regime else 1)`

**Rationale:** Preserves the asymmetric volatility core (log ratio of down/up range averages) with continuous volume confirmation. Uses the volume regime classification tool to attenuate the signal on low volume days (regime 0) when price moves may be less reliable, targeting improved IC and MI by reducing noise.

**Tools:** cross_sectional_transform=cs_rank; library_functions=alpha_tools.classify_volume_regime, np.clip, np.log, np.where

```python
def factor_asym_vol_attenuate(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    down = ret < 0
    up = ret >= 0
    range_ = df_copy["high"] - df_copy["low"]
    down_range = range_.where(down, 0)
    up_range = range_.where(up, 0)
    down_sum = down_range.rolling(20, min_periods=10).sum()
    up_sum = up_range.rolling(20, min_periods=10).sum()
    down_count = down.rolling(20, min_periods=10).sum()
    up_count = up.rolling(20, min_periods=10).sum()
    down_avg = down_sum / down_count.replace(0, np.nan)
    up_avg = up_sum / up_count.replace(0, np.nan)
    ratio = down_avg / up_avg.replace(0, np.nan)
    asym = np.log(ratio.clip(lower=1e-6))
    vol = df_copy["volume"]
    vol_ema = vol.ewm(span=20, min_periods=10).mean()
    volu_dev = (vol - vol_ema) / vol_ema.replace(0, np.nan)
    weight = np.clip(volu_dev, -1, 1)
    factor = asym * (1 + 0.3 * weight)
    vol_regime = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    # classify_volume_regime returns 0 for low volume, 1 for normal, 2 for high
    low_vol = vol_regime == 0
    factor = np.where(low_vol, factor * 0.7, factor)
    df_copy["factor_asym_vol_attenuate"] = factor
    return df_copy["factor_asym_vol_attenuate"]
```

---

## Factor 543

**Formula:** `pressure = ((close - open) / open) * volume; smoothed = EMA10(pressure); halve on low-volume days`

**Rationale:** Preserves the core volume-weighted intraday momentum mechanism from parent. Adds a lightweight volume regime adjustment: on low-volume days (detected by classify_volume_regime), the signal is halved to reduce noise from thin trading. The EMA(10) smoothing is retained for stability. This targets improving RankIC and RankICIR by filtering out low-quality observations.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.where

```python
def factor_volume_regime_intraday_momentum(df):
    df_copy = df.copy()
    intraday_ret = (df_copy["close"] - df_copy["open"]) / df_copy["open"].replace(0, np.nan)
    pressure = intraday_ret * df_copy["volume"]
    smoothed = pressure.ewm(span=10, adjust=False).mean()
    is_high_vol, is_low_vol, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = np.where(is_low_vol > 0, smoothed * 0.5, smoothed)
    df_copy["factor"] = factor
    return df_copy["factor"].fillna(0).rename("factor_volume_regime_intraday_momentum")
```

---

## Factor 544

**Formula:** `EMA( ((close-open)/open) * volume, 10 )`

**Rationale:** Mutation of factor_intraday_pressure by replacing the simple moving average with an exponential moving average to give more weight to recent observations, preserving the core volume-weighted intraday momentum mechanism while improving responsiveness and potentially IC.

**Tools:** library_functions=pd.Series, talib.EMA

```python
def factor_intraday_pressure_ema(df):
    df_copy = df.copy()
    intraday_ret = (df_copy["close"] - df_copy["open"]) / df_copy["open"].replace(0, np.nan)
    pressure = intraday_ret * df_copy["volume"]
    # Fix: fill NaN and ensure float64 for talib
    momentum = pd.Series(talib.EMA(pressure.fillna(0).values.astype(np.float64), timeperiod=10), index=df_copy.index)
    return momentum.fillna(0).rename("factor_intraday_pressure_ema")
```

---

## Factor 546

**Formula:** `vol_ratio = volume / rolling_median(volume, 20); raw = rolling_std(vol_ratio, 15); factor = if liquidity_high then raw else 0.7*raw`

**Rationale:** Preserves the strong RankICIR of parent CRS_001_G2 by retaining its volume variability core (volume/median ratio and rolling std). Improves IC and RankIC by increasing the std window from 10 to 15 for smoother variability estimates, and adjusting the low-liquidity multiplier from 0.5 to 0.7 to retain more signal. These compact changes target the metric bottleneck (IC and RankIC) without weakening rank stability.

**Tools:** style_gates=style_gate_liquidity_high; library_functions=np.where

```python
def factor_volume_abnormality_liquidity_gated(df):
    df_copy = df.copy()
    vol = df_copy['volume']
    median_vol = vol.rolling(20, min_periods=20).median()
    vol_ratio = vol / median_vol.replace(0, np.nan)
    raw = vol_ratio.rolling(15, min_periods=15).std()
    high_liquidity = df_copy['style_gate_liquidity_high'] > 0
    factor = np.where(high_liquidity, raw, raw * 0.7)
    df_copy['factor_volume_abnormality_liquidity_gated'] = factor
    return df_copy['factor_volume_abnormality_liquidity_gated']
```

---

## Factor 558

**Formula:** `log(ewm_mean(down_squared_return) / ewm_mean(up_squared_return)) if style_gate_resvol_high > 0 else 0`

**Rationale:** Asymmetric volatility between up and down returns is often more pronounced during periods of high residual volatility. Replacing the simple rolling mean with exponential weighted mean (EWM) gives more weight to recent observations, making the volatility asymmetry signal more responsive to recent changes while preserving the core economic mechanism. Gating on style_gate_resvol_high focuses the signal on regimes where divergence in volatility components is most informative.

**Tools:** cross_sectional_transform=cs_rank; style_gates=style_gate_resvol_high; library_functions=np.log, np.where, pd.Series

```python
def factor_asym_vol_down_up_volatility_gated(df, window=20):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    pos_var = np.where(ret > 0, ret**2, np.nan)
    neg_var = np.where(ret < 0, ret**2, np.nan)
    pos_var_mean = pd.Series(pos_var, index=df_copy.index).ewm(span=window, min_periods=5).mean()
    neg_var_mean = pd.Series(neg_var, index=df_copy.index).ewm(span=window, min_periods=5).mean()
    ratio = np.log(neg_var_mean / pos_var_mean.replace(0, np.nan))
    asymmetry = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    is_high_resvol = df_copy['style_gate_resvol_high'] > 0
    factor = np.where(is_high_resvol, asymmetry, 0.0)
    df_copy['factor_asym_vol_down_up_volatility_gated'] = factor
    return df_copy['factor_asym_vol_down_up_volatility_gated']
```

---

## Factor 559

**Formula:** `if style_gate_resvol_high then EMA5(ATR20/ATR60) else 0`

**Rationale:** Crossover taking the direct ATR ratio mechanism from mutation_gen1_001 (parent2) as primary, and borrowing the EMA smoothing lightweight idea from ARV_002_M1 (parent1) to reduce noise in the ratio, potentially improving IC and ICIR while preserving RankICIR. The core economic mechanism (volatility regime expansion/compression during high residual volatility) remains intact.

**Tools:** cross_sectional_transform=cs_rank; style_gates=style_gate_resvol_high; library_functions=np.nan_to_num, np.where, pd.Series, talib.ATR, talib.EMA

```python
def factor_smoothed_atr_ratio_gated(df):
    df_copy = df.copy()
    atr20 = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=20)
    atr60 = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=60)
    ratio = atr20 / atr60.replace(0, np.nan)
    smoothed_ratio = pd.Series(talib.EMA(ratio.values, timeperiod=5), index=df_copy.index)
    smoothed_ratio = smoothed_ratio.fillna(0)
    resvol_high = df_copy['style_gate_resvol_high'] > 0
    factor = np.where(resvol_high, smoothed_ratio, 0)
    factor = np.nan_to_num(factor, nan=0)
    df_copy['factor_smoothed_atr_ratio_gated'] = factor
    return df_copy['factor_smoothed_atr_ratio_gated']
```

---

## Factor 570

**Formula:** `rank( EMA((C-O)*V,21) * (1+(C(-1)-C)/C(-1)) / ATR(21) ) * 2 - 1`

**Rationale:** Mutation: replaced SMA(21) with EMA(21) for dollar imbalance smoothing to reduce lag and improve IC while preserving the core fear-adjusted pressure mechanism and strong RankIC/RankICIR. Single core mechanism, no modifiers.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_pressure_ema_mutation(df):
    df_copy = df.copy()
    # Dollar imbalance
    dollar_imb = (df_copy['close'] - df_copy['open']) * df_copy['volume']
    # Exponential moving average of dollar imbalance (21 days) instead of SMA
    smoothed_imb = dollar_imb.ewm(span=21, min_periods=21, adjust=False).mean()
    # Short-term price decline (positive when price falls)
    price_decline = (df_copy['close'].shift(1) - df_copy['close']) / df_copy['close'].shift(1)
    # Fear-adjusted dollar pressure
    fear_adj_imb = smoothed_imb * (1 + price_decline)
    # ATR for normalization
    high = df_copy['high'].values.astype(np.float64)
    low = df_copy['low'].values.astype(np.float64)
    close = df_copy['close'].values.astype(np.float64)
    atr = talib.ATR(high, low, close, timeperiod=21)
    atr_series = pd.Series(atr, index=df_copy.index)
    raw = fear_adj_imb / atr_series.replace(0, np.nan)
    # Rank transform for cross-sectional stability
    df_copy['factor_pressure_ema_mutation'] = raw.rank(pct=True) * 2 - 1
    return df_copy['factor_pressure_ema_mutation']
```

---

## Factor 571

**Formula:** `R(drawdown) * R(V/MA20)`

**Rationale:** Simplified mutation of crossover_factor_001: removed duration and recovery components to focus on the core drawdown-volume interaction. Preserves rank stability through rank transforms while concentrating the economic signal on volume-confirmed drawdown depth. Aims to improve IC and MI by reducing noise from auxiliary terms.

```python
def factor_drawdown_vol_simple(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    running_max = close.expanding().max()
    drawdown = (running_max - close) / running_max
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma
    rank_dd = drawdown.rank(pct=True)
    rank_vol = vol_ratio.rank(pct=True)
    factor = rank_dd * rank_vol
    factor.name = "factor_drawdown_vol_simple"
    return factor
```

---

## Factor 572

**Formula:** `ema( (range^2/close) * ts_rank(volume,20), 10 )`

**Rationale:** Ablation of cube root compression to improve linear predictive power (IC) while preserving rank stability. The cube root dampened signal extremes but also reduced linear correlation with future returns. Removing it allows the smoothed product of squared range and volume rank to express full magnitude, improving IC potential.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_squared_range_volume_rank_smoothed_no_cuberoot(df):
    df_copy = df.copy()
    range_ = df_copy['high'] - df_copy['low']
    range_sq = range_ ** 2
    vol_rank = df_copy['volume'].rolling(20, min_periods=20).rank(pct=True)
    raw = (range_sq / df_copy['close']) * vol_rank
    signal = raw.ewm(span=10, min_periods=10, adjust=False).mean()
    signal.name = 'factor_squared_range_volume_rank_smoothed_no_cuberoot'
    return signal
```

---

## Factor 573

**Formula:** `sign(ema(range^2/close,10)) * |ema(...)|^(1/3)`

**Rationale:** Simplified parent by removing volume rank and resvol gate to focus on the core volatility amplification mechanism. Squared range relative to close captures daily volatility scaled by price level, smoothed with EMA to reduce noise, and cube root compresses outliers. Cross-sectional rank ensures comparability across stocks. Removing the volume rank and gate reduces complexity and potential noise, targeting improved IC while preserving rank stability.

**Tools:** cross_sectional_transform=cs_rank; library_functions=np.abs, np.sign, pd.Series

```python
def factor_squared_range_close_ema_cuberoot(df):
    df_copy = df.copy()
    range_ = df_copy['high'] - df_copy['low']
    range_sq = range_ ** 2
    raw = range_sq / df_copy['close']
    smoothed = raw.ewm(span=10, min_periods=10, adjust=False).mean()
    signal = np.sign(smoothed) * np.abs(smoothed) ** (1/3)
    signal = pd.Series(signal, index=df_copy.index, name='factor_squared_range_close_ema_cuberoot').fillna(0)
    return signal
```

---

## Factor 576

**Formula:** `log( EWMA(down_ret^2,20)^0.5 / EWMA(up_ret^2,20)^0.5 ) * vol_ratio(20)`

**Rationale:** Crossover: primary mechanism from asym-vol-ratio-ewma-001 captures downside/upside volatility asymmetry using EWMA; lightweight modifier from candidate_3 uses continuous vol_ratio from classify_volume_regime as a volume-conviction weight. The product amplifies the asymmetry signal during high volume periods, combining crash risk with volume pressure.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log, np.where, pd.Series

```python
def factor_asym_vol_gated_by_volume_pressure(df):
    df_copy = df.copy()
    close = df_copy["close"]
    ret = close.pct_change()
    up_ret = np.where(ret > 0, ret, np.nan)
    down_ret = np.where(ret < 0, ret, np.nan)
    span = 20
    up_var = pd.Series(up_ret**2, index=df_copy.index).ewm(span=span, min_periods=5).mean()
    down_var = pd.Series(down_ret**2, index=df_copy.index).ewm(span=span, min_periods=5).mean()
    up_vol = up_var.pow(0.5)
    down_vol = down_var.pow(0.5)
    ratio = down_vol / up_vol.replace(0, np.nan)
    asym = np.log(ratio.clip(lower=1e-6))
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = asym * vol_ratio
    factor = factor.fillna(0)
    df_copy["factor_asym_vol_gated_by_volume_pressure"] = factor
    return df_copy["factor_asym_vol_gated_by_volume_pressure"]
```

---

## Factor 586

**Formula:** `rank(depth) * rank(vol/vol_ma20) + rank(duration) + rank(delta_depth_10)`

**Rationale:** Crossover of seed_factor_103 (primary) and seed_factor_71 (lightweight idea). Preserves the depth*vol nonlinear interaction which targets mutual information, while replacing expanding max with rolling max (window=252, min_periods=20) from parent 71 to reduce look-ahead and improve robustness. Duration and recovery change remain additive.

```python
def factor_drawdown_volume_geometry_rolling(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    rolling_max = close.rolling(window=252, min_periods=20).max()
    depth = (rolling_max - close) / rolling_max
    new_high = close == rolling_max
    group = new_high.cumsum()
    duration = df_copy.groupby(group).cumcount()
    depth_chg = depth.diff(10).fillna(0)
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma
    rank_depth = depth.rank()
    rank_duration = duration.rank()
    rank_recovery = depth_chg.rank()
    rank_vol = vol_ratio.rank()
    depth_vol = rank_depth * rank_vol
    factor = depth_vol + rank_duration + rank_recovery
    df_copy['factor_drawdown_volume_geometry_rolling'] = factor
    return df_copy['factor_drawdown_volume_geometry_rolling']
```

---

## Factor 587

**Formula:** `If style_gate_resvol_high then ROC20 else ROC5, multiplied by volume ratio from classify_volume_regime`

**Rationale:** Combines residual volatility regime gating with volume confirmation for momentum. In high residual volatility regimes, a longer momentum window (20-day ROC) reduces noise; otherwise, a shorter window (5-day ROC) captures faster trends. Volume ratio from the classify_volume_regime tool confirms the signal, preserving the parent's core volume-confirmed momentum intuition while adding regime adaptation via an external style gate.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.where, pd.Series, talib.ROC

```python
def factor_resvol_volume_momentum(df):
    df_copy = df.copy()
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    fast_mom = pd.Series(talib.ROC(df_copy["close"], timeperiod=5), index=df_copy.index)
    slow_mom = pd.Series(talib.ROC(df_copy["close"], timeperiod=20), index=df_copy.index)
    raw_mom = np.where(high_resvol, slow_mom, fast_mom)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    factor = raw_mom * vol_ratio
    return pd.Series(factor, index=df_copy.index, name='factor_resvol_volume_momentum')
```

---

## Factor 589

**Formula:** `if resvol_high: normalized(ROC(close,20)/volatility) else normalized(ROC(close,10)/volatility) * (vol_ratio - 1)`

**Rationale:** Mutates the regime-gated momentum with volume deviation by normalizing momentum by its own trailing volatility to improve signal-to-noise ratio. Preserves the core regime-adaptive momentum window selection and volume confirmation via classify_volume_regime tool. The volatility adjustment targets improved RankIC and MI stability without altering the economic intuition. Parent strength: regime-adaptive momentum mechanism. Bottleneck: stability and nonlinear information.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.where, talib.ROC

```python
def factor_regime_vol_adjusted_momentum_deviation(df):
    df_copy = df.copy()
    high_resvol = df_copy["style_gate_resvol_high"]
    short_mom = talib.ROC(df_copy["close"], timeperiod=10)
    long_mom = talib.ROC(df_copy["close"], timeperiod=20)
    short_vol = short_mom.rolling(20, min_periods=20).std().replace(0, np.nan)
    long_vol = long_mom.rolling(20, min_periods=20).std().replace(0, np.nan)
    short_mom_norm = short_mom / short_vol
    long_mom_norm = long_mom / long_vol
    mom = np.where(high_resvol > 0, long_mom_norm, short_mom_norm)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = mom * (vol_ratio - 1)
    df_copy["factor_regime_vol_adjusted_momentum_deviation"] = factor
    return df_copy["factor_regime_vol_adjusted_momentum_deviation"]
```

---

## Factor 601

**Formula:** `factor = (close/shift10 - 1) * sum(positive daily returns over 10 days) * (1 + 0.3 * vol_ratio)`

**Rationale:** Mutation of parent factor_persistent_volume_ratio_momentum: replaced binary persistence count (number of positive daily returns) with a continuous sum of positive daily returns over the same 10-day window. This reduces noise from the discrete count and provides a smoother measure of directional persistence, aiming to improve RankIC and RankICIR stability while preserving the core mechanism of momentum confirmed by volume participation.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.where, pd.Series

```python
def factor_persistent_volume_sum_momentum(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(10) - 1
    daily_ret = df_copy['close'].pct_change()
    positive_ret = np.where(daily_ret > 0, daily_ret, 0.0)
    persistence = pd.Series(positive_ret, index=df_copy.index).rolling(10).sum()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    weight = 1 + 0.3 * vol_ratio
    factor = ret * persistence * weight
    df_copy['factor_persistent_volume_sum_momentum'] = factor
    return df_copy['factor_persistent_volume_sum_momentum']
```

---

## Factor 603

**Formula:** `-ewm(|ret(t)-ret(t-1)|, span=10) * clip(vol_ratio, 0.5, 1.5)`

**Rationale:** Mutate parent factor_return_change_smoothness by replacing rolling mean with EWMA for faster response, flipping direction so smoother price action (lower jitter) predicts continuation, and using continuous bounded volume scaling. Targets Price-Volatility Behavior layer: low return jitter confirmed by active volume indicates conviction.

**Tools:** library_functions=alpha_tools.classify_volume_regime

```python
def factor_ewma_smoothness_volume(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    abs_diff_ret = (ret - ret.shift(1)).abs()
    smoothness = abs_diff_ret.ewm(span=10, min_periods=10).mean()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    vol_weight = vol_ratio.clip(lower=0.5, upper=1.5)
    factor = -smoothness * vol_weight
    df_copy["factor_ewma_smoothness_volume"] = factor
    return df_copy["factor_ewma_smoothness_volume"]
```

---

## Factor 604

**Formula:** `(close / close.shift(10) - 1) * clip(alpha_tools.classify_volume_regime(volume, window=20).vol_ratio, 0.5, 2.0)`

**Rationale:** Crossover: primary mechanism from factor_volume_confirmed_return (return scaled by volume ratio) with lightweight transfer of the active tool classify_volume_regime from factor_persistent_volume_ratio_momentum for smoother EMA-based volume ratio. The tool vol_ratio replaces the manual rolling mean ratio to improve robustness while preserving the balanced IC/MI profile of the primary parent. Clipping retains bounded scaling.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.clip

```python
def factor_volume_confirmed_return_tool(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(10) - 1
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    vol_ratio_clipped = np.clip(vol_ratio, 0.5, 2.0)
    factor = ret * vol_ratio_clipped
    df_copy['factor_volume_confirmed_return_tool'] = factor
    return df_copy['factor_volume_confirmed_return_tool']
```

---

## Factor 606

**Formula:** `ret(10) * sum(pos_ret_10) * (1+0.5*vol_ratio) * 1/(1+rolling_mean(|ret(t)-ret(t-1)|,10))`

**Rationale:** Crossover: primary parent is qualified smoothed persistent volume ratio momentum (factor_smoothed_persistent_volume_ratio_momentum), secondary parent contributes the return change smoothness (smoothness) as a reliability weight. The primary mechanism (directional persistence weighted by volume ratio) is preserved; the smoothness reliability weight attenuates the signal when recent returns are erratic (high absolute change in returns), aiming to reduce noise and improve RankIC stability. This keeps the factor compact and interprets smoothness as a noise filter.

**Tools:** library_functions=alpha_tools.classify_volume_regime

```python
def factor_smoothed_persistent_momentum_smoothness_weighted(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(10) - 1
    persistence = (df_copy['close'].pct_change() > 0).rolling(10).sum()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    weight = 1 + 0.5 * vol_ratio
    daily_ret = df_copy['close'].pct_change()
    abs_change = (daily_ret - daily_ret.shift(1)).abs()
    smoothness = abs_change.rolling(10, min_periods=10).mean()
    reliability = 1.0 / (1.0 + smoothness)
    factor = ret * persistence * weight * reliability
    df_copy['factor_smoothed_persistent_momentum_smoothness_weighted'] = factor
    return df_copy['factor_smoothed_persistent_momentum_smoothness_weighted']
```

---

## Factor 620

**Formula:** `trend = (close - close.shift(10)) / rolling_std(close, 20); output = trend * (1.0 if resvol_high else 0.5)`

**Rationale:** This mutation preserves the parent's core normalized trend with regime-weighting, but replaces the liquidity gate with a residual volatility gate (style_gate_resvol_high) to capture periods of heightened idiosyncratic volatility that often accompany sustained trend signals. Additionally, the momentum window is extended from 5 to 10 days to reduce noise and improve signal-to-noise. The gate weight is kept as continuous attenuation (1.0 high, 0.5 low) to ensure cross-sectional variation.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.where

```python
def factor_resvol_smoothed_trend(df):
    df_copy = df.copy()
    momentum = df_copy["close"] - df_copy["close"].shift(10)
    vol = df_copy["close"].rolling(20).std(ddof=1).replace(0, np.nan)
    trend_signal = momentum / vol
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    weight = np.where(high_resvol, 1.0, 0.5)
    factor = trend_signal * weight
    df_copy["factor_resvol_smoothed_trend"] = factor
    return df_copy["factor_resvol_smoothed_trend"]
```

---

## Factor 631

**Formula:** `ema( ret_1d * log(vol_ratio_10) , span=5 )`

**Rationale:** Simplified mutation of parent g1_mutation_alpha_lag_response_001_6abf6c9a. Core hypothesis preserved: volume-amplified price adjustment due to delayed market response when volume is elevated. Replaces 5-day return with daily return for immediate reaction, removes ATR normalization which added noise, and uses log volume ratio from alpha_tools as weight. EMA smoothing (span=5) reduces jaggedness while preserving responsiveness. Targets improvement in IC and RankIC by using a cleaner signal. Parent strength target: the volume-regime-aware adjustment concept (nonlinear volume interaction) is preserved as the core mechanism.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log

```python
def factor_simplified_volume_lag(df):
    df_copy = df.copy()
    ret_1d = df_copy["close"].pct_change(1)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=10)
    log_vol = np.log(vol_ratio.clip(lower=1e-12))
    raw = ret_1d * log_vol
    factor = raw.ewm(span=5, min_periods=5, adjust=False).mean()
    df_copy["factor_simplified_volume_lag"] = factor
    return df_copy["factor_simplified_volume_lag"]
```

---

## Factor 632

**Formula:** `ret_5d * log(vol/vol_ma_20) / (ATR_14/close) * (1 + rolling_corr(ret_daily, vol_pct, 20))`

**Rationale:** Crossover combining the primary mechanism from parent g1_crossover_pvcoh_002_alpha_lag_response_001_85fca0ee (qualified, strong RankIC and RankICIR) with a lightweight log transform of the volume ratio borrowed from parent g1_mutation_alpha_lag_response_001_6abf6c9a. The primary mechanism uses delayed price adjustment captured by ret_5 * (vol/vol_ma) / (ATR/close) and then multiplies by a price-volume coherence weight (1+rolling_corr). The log transform of volume ratio compresses outliers and improves stationarity, aiming to improve IC while preserving the strong rank-order stability from the coherence term. No cross-sectional transform is needed as parent ranking evidence supports the raw mechanism.

**Tools:** library_functions=np.log, pd.concat

```python
def factor_vol_adjusted_lag_log_coherence(df):
    df_copy = df.copy()
    high_low = df_copy["high"] - df_copy["low"]
    high_pc = (df_copy["high"] - df_copy["close"].shift(1)).abs()
    low_pc = (df_copy["low"] - df_copy["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    norm_vol = atr / df_copy["close"].replace(0, np.nan)
    ret_5 = df_copy["close"].pct_change(5)
    vol_ma = df_copy["volume"].rolling(20, min_periods=20).mean()
    vol_ratio = df_copy["volume"] / vol_ma.replace(0, np.nan)
    log_vol_ratio = np.log(vol_ratio.clip(lower=1e-12))
    primary = ret_5 * log_vol_ratio / norm_vol.replace(0, np.nan)
    ret_daily = df_copy["close"].pct_change()
    vol_pct = df_copy["volume"].pct_change()
    corr = ret_daily.rolling(20, min_periods=20).corr(vol_pct)
    weight = 1 + corr.fillna(0)
    factor = primary * weight
    df_copy["factor_vol_adjusted_lag_log_coherence"] = factor
    return df_copy["factor_vol_adjusted_lag_log_coherence"]
```

---

## Factor 643

**Formula:** `factor = EMA_8( lag_ret * log( V / EMA_10(V) ) )`

**Rationale:** Replace SMA with EMA for volume normalization to achieve smoother and more responsive volume regime detection, preserving the core lagged return and volume interaction. The EMA reduces noise in the volume ratio, potentially improving linear association (IC) while maintaining the parent's strong rank stability (RankICIR=0.3329).

**Tools:** library_functions=np.log

```python
def factor_volume_adjusted_lag_return_v3(df):
    df_copy = df.copy()
    close = df_copy["close"]
    volume = df_copy["volume"]
    ret = close.pct_change()
    vol_ema = volume.ewm(span=10, min_periods=10, adjust=False).mean()
    vol_ratio = volume / vol_ema.replace(0, np.nan)
    log_vol_ratio = np.log(vol_ratio.clip(lower=1e-12))
    lag_ret = ret.shift(1)
    raw = lag_ret * log_vol_ratio
    factor = raw.ewm(span=8, min_periods=8, adjust=False).mean()
    factor.name = "factor_volume_adjusted_lag_return_v3"
    return factor
```

---

## Factor 644

**Formula:** `EMA_8( lag_ret * log(V / SMA_10(V)) )`

**Rationale:** Mutation from factor_volume_adjusted_lag_simple: reduced volume ratio window from 20 to 10 for more responsive volume regime detection, and increased EMA span from 5 to 8 for smoother, less noisy signal. This aims to improve IC and ICIR while preserving the RankICIR strength of the parent.

**Tools:** library_functions=np.log

```python
def factor_volume_adjusted_lag_simple(df):
    df_copy = df.copy()
    close = df_copy["close"]
    volume = df_copy["volume"]
    ret = close.pct_change()
    vol_ratio = volume / volume.rolling(10, min_periods=10).mean()
    log_vol_ratio = np.log(vol_ratio.replace(0, np.nan).clip(lower=1e-12))
    lag_ret = ret.shift(1)
    raw = lag_ret * log_vol_ratio
    factor = raw.ewm(span=8, min_periods=8, adjust=False).mean()
    factor = factor.fillna(0.0)
    factor.name = "factor_volume_adjusted_lag_simple"
    return factor
```

---

## Factor 646

**Formula:** `EMA_8( lag_ret * log(V / SMA_10(V)) * consensus(10) )`

**Rationale:** Preserve parent1's strong RankIC and RankICIR from its volume-adjusted lag return with window 10 and EMA 8. Borrow parent2's lightweight consensus between intraday return direction and volume change direction. The consensus acts as a regime weight: when both directions agree, the signal is amplified; when they disagree, it is attenuated. This adds nonlinear interaction to improve IC and ICIR without altering the core economic mechanism.

**Tools:** library_functions=alpha_tools.decompose_overnight_intraday, np.log, np.sign

```python
def factor_volume_adjusted_lag_consensus_v2(df):
    df_copy = df.copy()
    close = df_copy["close"]
    volume = df_copy["volume"]
    open_price = df_copy["open"]
    ret = close.pct_change()
    vol_ratio = volume / volume.rolling(10, min_periods=10).mean()
    log_vol_ratio = np.log(vol_ratio.replace(0, np.nan).clip(lower=1e-12))
    lag_ret = ret.shift(1)
    raw = lag_ret * log_vol_ratio
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(close, open_price)
    vol_diff = volume.diff()
    intraday_dir = np.sign(intraday_ret)
    vol_dir = np.sign(vol_diff)
    agreement = (intraday_dir == vol_dir).astype(float)
    consensus = agreement.rolling(10).mean()
    factor = raw * consensus
    factor = factor.ewm(span=8, min_periods=8, adjust=False).mean()
    factor = factor.fillna(0.0)
    factor.name = "factor_volume_adjusted_lag_consensus_v2"
    return factor
```

---

## Factor 656

**Formula:** `EMA_5( (close/shift(10)-1) * log(volume/EMA_20(volume)) / std_20(ret) )`

**Rationale:** Mutation of g1_crossover_then_mutation_atr_001_apvc_002_57113858: shortens the momentum horizon from 15 to 10 days to increase sensitivity to recent price changes while preserving the volume-confirmed momentum and volatility normalization core mechanism. The smoothing, log volume ratio, and volatility normalization window remain unchanged to maintain RankICIR strength.

**Tools:** library_functions=np.log

```python
def factor_smoothed_vol_adj_volume_momentum_10(df):
    df_copy = df.copy()
    ret = df_copy["close"] / df_copy["close"].shift(10) - 1
    vol_ema = df_copy["volume"].ewm(span=20, adjust=False).mean()
    vol_ratio = df_copy["volume"] / vol_ema.replace(0.0, np.nan)
    log_vol_ratio = np.log(vol_ratio.clip(lower=1e-12))
    ret_vol = ret.rolling(20, min_periods=20).std(ddof=0)
    raw_factor = (ret * log_vol_ratio) / ret_vol.replace(0.0, np.nan)
    smoothed = raw_factor.ewm(span=5, adjust=False).mean()
    df_copy["factor_smoothed_vol_adj_volume_momentum_10"] = smoothed.fillna(0)
    return df_copy["factor_smoothed_vol_adj_volume_momentum_10"]
```

---

## Factor 658

**Formula:** `corr(pct_return, pct_vol_change, 20) * vol_ratio_20`

**Rationale:** Mutates APVC-001 by using return-based price change (pct_change) and extending the correlation window to 20 days for more stable coherence estimation. Volume ratio continues to modulate weight. This targets higher IC and RankIC by reducing noise from price level drifts and improving stationarity. Parent strength target: price-volume coherence intuition.

**Tools:** library_functions=alpha_tools.classify_volume_regime

```python
def factor_vol_price_coherence_v2(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_chg_pct = df_copy["volume"].pct_change()
    roll_corr = ret.rolling(20, min_periods=20).corr(vol_chg_pct)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = roll_corr * vol_ratio
    factor = factor.fillna(0)
    df_copy["factor_vol_price_coherence_v2"] = factor
    return df_copy["factor_vol_price_coherence_v2"]
```

---

## Factor 676

**Formula:** `tanh( (ATR(20) / VWAP(30)) * (volume / EMA(volume, 30)) )`

**Rationale:** Mutation of factor_vwap_adjusted_range_tanh by extending all lookback windows (ATR to 20, VWAP to 30, volume EMA to 30) to produce a smoother, less noisy signal. This targets improved ICIR and RankICIR while preserving the core volatility-volume confirmation mechanism anchored to VWAP.

**Tools:** library_functions=np.tanh, talib.ATR, talib.EMA

```python
def factor_vwap_adjusted_range_tanh_smooth(df):
    df_copy = df.copy()
    # Increased ATR window from 14 to 20 for smoother volatility estimate
    atr = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=20)
    # Increased VWAP window from 20 to 30 for more stable anchor
    vwap = (df_copy['close'] * df_copy['volume']).rolling(30, min_periods=30).sum() / df_copy['volume'].rolling(30, min_periods=30).sum().replace(0, np.nan)
    norm_atr = atr / vwap
    # Increased volume EMA window from 20 to 30 for smoother ratio
    vol_ema = talib.EMA(df_copy['volume'], timeperiod=30)
    vol_ratio = df_copy['volume'] / vol_ema
    raw = norm_atr * vol_ratio
    factor = np.tanh(raw)
    df_copy['factor_vwap_adjusted_range_tanh_smooth'] = factor
    return df_copy['factor_vwap_adjusted_range_tanh_smooth']
```

---

## Factor 678

**Formula:** `tanh( (ATR(14) / VWAP(14)) * (volume / EMA(volume, 30)) )`

**Rationale:** Mutate the parent by lengthening ATR and VWAP windows from 10 to 14 for a more robust volatility estimate, and increasing volume EMA from 20 to 30 to reduce noise in volume confirmation. This targets improved IC while preserving the core trend-volume confirmation mechanism and RankIC stability. The tanh transformation is retained to maintain rank consistency.

**Tools:** library_functions=np.tanh, talib.ATR, talib.EMA

```python
def factor_vwap_adjusted_range_tanh_mutated_v2(df):
    df_copy = df.copy()
    atr = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
    vwap = (df_copy['close'] * df_copy['volume']).rolling(14).sum() / df_copy['volume'].rolling(14).sum().replace(0, np.nan)
    norm_atr = atr / vwap
    vol_ema = talib.EMA(df_copy['volume'], timeperiod=30)
    vol_ratio = df_copy['volume'] / vol_ema
    raw = norm_atr * vol_ratio
    factor = np.tanh(raw)
    df_copy['factor_vwap_adjusted_range_tanh_mutated_v2'] = factor
    return df_copy['factor_vwap_adjusted_range_tanh_mutated_v2']
```
---

## Factor 692

**Formula:** `sma_down = SMA(open - low, 40); sma_up = SMA(high - open, 40); ratio = sma_down / sma_up`

**Rationale:** Simplifies the intraday asymmetry measure by using simple moving averages instead of exponential, reducing sensitivity to recent outliers and improving stability. Window increased to 40 for a more persistent asymmetry signal.

**Tools:** library_functions=pd.Series, talib.SMA

```python
def factor_asym_intraday_sma(df):
    df_copy = df.copy()
    upside = (df_copy["high"] - df_copy["open"]).astype(float)
    downside = (df_copy["open"] - df_copy["low"]).astype(float)
    sma_up = pd.Series(talib.SMA(upside.values, timeperiod=40), index=df_copy.index)
    sma_down = pd.Series(talib.SMA(downside.values, timeperiod=40), index=df_copy.index)
    ratio = sma_down / sma_up.replace(0, np.nan)
    df_copy["factor_asym_intraday_sma"] = ratio.fillna(0)
    return df_copy["factor_asym_intraday_sma"]
```

---

## Factor 693

**Formula:** `(close/shift(21)-1) / (if liquidity_high then std(negative_returns,21) else std(negative_returns,63)) * 1/(1+EMA(downside_open_low)/EMA(upside_high_open))`

**Rationale:** Combines liquidity-adaptive downside volatility momentum (parent ADT-002) with intraday asymmetry weighting from parent ava_002. The momentum factor is normalized by downside volatility with window selected by liquidity regime. The intraday asymmetry ratio (EMA of downside/upside range) adjusts the signal continuously: when intraday downside pressure is high (sellers dominate), the momentum is attenuated, reducing false signals. This targets IC improvement while preserving the rank ICIR strength of the primary mechanism.

**Tools:** style_gates=style_gate_liquidity_high; library_functions=np.where, pd.Series, talib.EMA

```python
def factor_liquidity_adaptive_asym_momentum(df):
    df_copy = df.copy()
    close = df_copy['close']
    returns = close.pct_change()
    mom = close / close.shift(21) - 1.0
    down_ret = np.where(returns < 0, returns, np.nan)
    down_ret_series = pd.Series(down_ret, index=df_copy.index)
    vol_short = down_ret_series.rolling(21, min_periods=5).std(ddof=1)
    vol_long = down_ret_series.rolling(63, min_periods=5).std(ddof=1)
    high_liquidity = df_copy['style_gate_liquidity_high'] > 0
    vol = np.where(high_liquidity, vol_short, vol_long)
    vol = np.where(vol == 0.0, np.nan, vol)
    upside = (df_copy['high'] - df_copy['open']).astype(float)
    downside = (df_copy['open'] - df_copy['low']).astype(float)
    ema_up = pd.Series(talib.EMA(upside.values, timeperiod=20), index=df_copy.index)
    ema_down = pd.Series(talib.EMA(downside.values, timeperiod=20), index=df_copy.index)
    ratio = ema_down / ema_up.replace(0, np.nan)
    asym_weight = 1.0 / (1.0 + ratio.fillna(np.nan))
    factor = mom / vol * asym_weight
    df_copy['factor_liquidity_adaptive_asym_momentum'] = factor.fillna(np.nan)
    return df_copy['factor_liquidity_adaptive_asym_momentum']
```

---

## Factor 695

**Formula:** `(close / shift(21) - 1) / (if liquidity_high then std(negative_returns,21) else std(negative_returns,63))`

**Rationale:** Simplified mutation of parent: removed intraday asymmetry weighting to isolate the core liquidity-adaptive downside volatility normalized momentum. Ablation tests whether the asymmetry component added noise; expect IC improvement while preserving RankICIR stability of the primary mechanism.

**Tools:** style_gates=style_gate_liquidity_high; library_functions=np.where, pd.Series

```python
def factor_liquidity_adaptive_momentum(df):
    df_copy = df.copy()
    close = df_copy['close']
    returns = close.pct_change()
    mom = close / close.shift(21) - 1.0
    down_ret = np.where(returns < 0, returns, np.nan)
    down_ret_series = pd.Series(down_ret, index=df_copy.index)
    vol_short = down_ret_series.rolling(21, min_periods=5).std(ddof=1)
    vol_long = down_ret_series.rolling(63, min_periods=5).std(ddof=1)
    high_liquidity = df_copy['style_gate_liquidity_high'] > 0
    vol = np.where(high_liquidity, vol_short, vol_long)
    vol = np.where(vol == 0.0, np.nan, vol)
    factor = mom / vol
    df_copy['factor_liquidity_adaptive_momentum'] = factor
    return df_copy['factor_liquidity_adaptive_momentum']
```

---

## Factor 697

**Formula:** `(close/shift(21)-1) / (if liquidity_high then std(downside_returns,21) else std(downside_returns,63))`

**Rationale:** Crossover of parent 2's liquidity-adaptive momentum mechanism with parent 1's use of downside volatility instead of total volatility. Downside volatility better captures trend risk and improves risk adjustment, while the liquidity-adaptive window tailors the volatility estimation to market conditions. The factor measures momentum strength per unit of downside risk.

**Tools:** style_gates=style_gate_liquidity_high; library_functions=np.where

```python
def factor_downside_vol_liquidity_momentum(df):
    df_copy = df.copy()
    close = df_copy['close']
    momentum = close / close.shift(21) - 1.0
    returns = close.pct_change()
    downside_returns = returns.where(returns < 0, np.nan)
    vol_short = downside_returns.rolling(21, min_periods=5).std(ddof=1)
    vol_long = downside_returns.rolling(63, min_periods=5).std(ddof=1)
    high_liquidity = df_copy['style_gate_liquidity_high'] > 0
    vol = np.where(high_liquidity, vol_short, vol_long)
    vol = np.where(vol == 0.0, np.nan, vol)
    factor = momentum / vol
    df_copy['factor_downside_vol_liquidity_momentum'] = factor
    return df_copy['factor_downside_vol_liquidity_momentum']
```

---

## Factor 708

**Formula:** `ADX(14) * sign(close_10d_chg) * clip(vol_ratio_EMA(20), 0.5, 2.0)`

**Rationale:** Mutation of parent g1_crossover_alpha_daily_trend_gen0_cand2_rev_vol_asym_002_c03340f0: preserve core directional trend strength (ADX + sign) and volume confirmation. Replace SMA-based volume ratio with EMA-based vol_ratio from classify_volume_regime tool for smoother volume weighting, keeping the clip to bound influence. This aims to preserve high RankICIR while improving IC stability.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.sign, pd.Series, talib.ADX

```python
def factor_adx_trend_vol_ema(df):
    df_copy = df.copy()
    high = df_copy["high"].values.astype(np.float64)
    low = df_copy["low"].values.astype(np.float64)
    close = df_copy["close"].values.astype(np.float64)
    adx = talib.ADX(high, low, close, timeperiod=14)
    dir_sign = np.sign(df_copy["close"].pct_change(10))
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    vol_ratio_clipped = vol_ratio.clip(0.5, 2.0)
    factor = pd.Series(adx * dir_sign * vol_ratio_clipped, index=df_copy.index, name="factor_adx_trend_vol_ema")
    factor = factor.fillna(0)
    return factor
```

---

## Factor 718

**Formula:** `rolling_20_corr(pct_change(close), pct_change(volume)) * alpha_tools.classify_volume_regime(volume, 20).vol_ratio`

**Rationale:** Preserves the rolling correlation between returns and volume changes as the core mechanism. Adds a lightweight modifier using the volume regime ratio (vol_ratio) from classify_volume_regime to amplify the correlation during high-volume periods and attenuate during low-volume periods. This targets improved IC and RankIC by focusing the signal on periods of informed trading where price-volume coherence is stronger. The correlation remains bounded, and the vol_ratio multiplier is centered around 1, preserving the parent's robustness.

**Tools:** library_functions=alpha_tools.classify_volume_regime

```python
def factor_corr_volume_regime(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_pct = df_copy["volume"].pct_change()
    corr = ret.rolling(20, min_periods=20).corr(vol_pct)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = corr * vol_ratio
    factor.name = "factor_corr_volume_regime"
    return factor
```

---

## Factor 719

**Formula:** `EMA5( shift1(ret) * log(vol/SMA20(vol)) * classify_volume_regime_multiplier )`

**Rationale:** Crossover: primary mechanism from parent1 (lagged return weighted by log volume ratio) captures delayed price adjustment in abnormal volume. Lightweight modifier from parent2: volume regime multiplier using classify_volume_regime attenuates signal in low-volume noise periods (mult=0.3) and amplifies in high-volume informed periods (mult=vol_ratio_cont). This aims to preserve parent1's IC and MI while improving RankICIR via noise reduction.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log, np.where

```python
def factor_lag_volume_ratio_regime_adaptive(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    ret = close.pct_change()
    ret_lag = ret.shift(1)
    avg_vol = volume.rolling(20, min_periods=20).mean()
    vol_ratio = np.log(volume / avg_vol)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    core = ret_lag * vol_ratio
    is_high_vol, is_low_vol, vol_ratio_cont = alpha_tools.classify_volume_regime(volume, window=20)
    mult = np.where(is_high_vol > 0, vol_ratio_cont, 1.0)
    mult = np.where(is_low_vol > 0, 0.3, mult)
    factor = core * mult
    factor = factor.rolling(5, min_periods=5).mean()
    factor.name = 'factor_lag_volume_ratio_regime_adaptive'
    return factor
```

---

## Factor 733

**Formula:** `EMA( (close/shift(close,5)-1) / trailing_vol(20) * log(vol_ratio) , span=10)`

**Rationale:** Mutates factor_volume_adaptive_momentum_smoothed by adding a trailing 20-day volatility normalization to the 5-day return. This preserves the core volume-adaptive momentum and smoothing while targeting improved IC by making the return component risk-adjusted. The parent's strength in RankIC and RankICIR is preserved through the same log volume weighting and EMA smoothing.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log

```python
def factor_volume_adaptive_momentum_scaled_vol(df):
    df_copy = df.copy()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    ret5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    daily_ret = df_copy["close"].pct_change()
    vol_20 = daily_ret.rolling(20).std()
    ret5_scaled = ret5 / vol_20.replace(0, np.nan)
    vol_weight = np.log(vol_ratio.clip(lower=0.1, upper=10))
    weighted_ret = ret5_scaled * vol_weight
    factor = weighted_ret.ewm(span=10, min_periods=10).mean()
    df_copy["factor_volume_adaptive_momentum_scaled_vol"] = factor
    return df_copy["factor_volume_adaptive_momentum_scaled_vol"]
```

---

## Factor 735

**Formula:** `factor = EMA( (close/shift(close,5)-1) * log(vol_ratio) , span=15)`

**Rationale:** Preserves the primary mechanism from parent 1: 5-day momentum weighted by log volume ratio. Borrows from parent 2 the longer EMA smoothing span of 15 to further reduce noise and improve ICIR and RankICIR stability, while retaining the core volume-adaptive momentum strength.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log

```python
def factor_volume_adaptive_momentum_smoothed_v3(df):
    df_copy = df.copy()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    ret5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    vol_weight = np.log(vol_ratio.clip(lower=0.1, upper=10))
    weighted_ret = ret5 * vol_weight
    factor = weighted_ret.ewm(span=15, min_periods=15).mean()
    df_copy["factor_volume_adaptive_momentum_smoothed_v3"] = factor
    return df_copy["factor_volume_adaptive_momentum_smoothed_v3"]
```

---

## Factor 737

**Formula:** `factor = EMA( (close/shift(close,5)-1) * log(vol_ratio) , span=10) with vol_ratio from window=10`

**Rationale:** Mutation of factor_volume_adaptive_momentum_smoothed_v3: reduce volume regime window from 20 to 10 for more responsive volume adaptation, and correspondingly reduce EMA span from 15 to 10 to align smoothing with faster volume signal. This targets improved IC by reacting more quickly to volume changes, while preserving the core volume-adaptive momentum strength from the parent.

**Tools:** library_functions=alpha_tools.classify_volume_regime, np.log

```python
def factor_volume_adaptive_momentum_fast_vol(df):
    df_copy = df.copy()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=10, high_threshold=1.5, low_threshold=0.6)
    ret5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    vol_weight = np.log(vol_ratio.clip(lower=0.1, upper=10))
    weighted_ret = ret5 * vol_weight
    factor = weighted_ret.ewm(span=10, min_periods=10).mean()
    df_copy["factor_volume_adaptive_momentum_fast_vol"] = factor
    return df_copy["factor_volume_adaptive_momentum_fast_vol"]
```

---

## Factor 747

**Formula:** `EWMA( |intraday_ret| * vol_ratio, span=10 )`

**Rationale:** Mutation from factor_vol_regime_impact_smoothed: replaces rolling mean with exponential weighted moving average to reduce lag and improve signal responsiveness while preserving the core economic mechanism of intraday price impact scaled by volume regime. Parent strength target: RankICIR stability.

**Tools:** library_functions=alpha_tools.classify_volume_regime, alpha_tools.decompose_overnight_intraday

```python
def factor_vol_regime_impact_ema(df):
    df_copy = df.copy()
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    is_high_vol, is_low_vol, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    impact = abs(intraday_ret) * vol_ratio
    factor = impact.ewm(span=10, min_periods=10, adjust=False).mean()
    df_copy["factor_vol_regime_impact_ema"] = factor
    return df_copy["factor_vol_regime_impact_ema"]
```

---

## Factor 749

**Formula:** `-CV(volume, window) where window = 20 if high resvol else 10`

**Rationale:** Mutates the parent fixed 10-day window to an adaptive window based on residual volatility. In high residual volatility, a longer 20-day window captures more stable volume consistency, reducing noise. In low volatility, the shorter 10-day window retains sensitivity. This preserves the core volume smoothness mechanism while improving signal quality by matching window length to market noise. Aims to improve IC and MI while maintaining strong RankICIR.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.where

```python
def factor_adaptive_volume_smoothness(df):
    df_copy = df.copy()
    high_resvol = df_copy['style_gate_resvol_high'] > 0
    cv_short = df_copy['volume'].rolling(10, min_periods=10).std(ddof=1) / df_copy['volume'].rolling(10, min_periods=10).mean().replace(0, np.nan)
    cv_long = df_copy['volume'].rolling(20, min_periods=20).std(ddof=1) / df_copy['volume'].rolling(20, min_periods=20).mean().replace(0, np.nan)
    factor = -np.where(high_resvol, cv_long, cv_short)
    df_copy['factor_adaptive_volume_smoothness'] = factor
    return df_copy['factor_adaptive_volume_smoothness']
```

---

## Factor 761

**Formula:** `EMA(ret(5) * robust_z(range,30) * vol_ratio(volume,20), 15)`

**Rationale:** Preserve the lagged return and robust range normalization from the parent, but increase the range window to 30 for more stable robust z-score, and increase EMA period to 15 for stronger smoothing. These changes target the IC bottleneck (weaker than RankIC) by reducing noise in the linear correlation while preserving the core economic mechanism of gradual price adjustment following volume- and volatility-driven moves.

**Tools:** library_functions=alpha_tools.classify_volume_regime, talib.EMA

```python
def factor_lag_vol_ratio_smoothed_v2(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(5) - 1
    range_ = df_copy['high'] - df_copy['low']
    range_med = range_.rolling(window=30, min_periods=30).median()
    range_mad = (range_ - range_med).abs().rolling(window=30, min_periods=30).mean()
    range_z = (range_ - range_med) / range_mad.replace(0.0, np.nan)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    raw_factor = ret * range_z * vol_ratio
    factor = talib.EMA(raw_factor, timeperiod=15)
    df_copy['factor_lag_vol_ratio_smoothed_v2'] = factor
    return df_copy['factor_lag_vol_ratio_smoothed_v2']
```

---

## Factor 763

**Formula:** `EMA(ret(5) * robust_z(range,20) * vol_ratio(volume,20), 10)`

**Rationale:** Simplifies the parent by removing the log transform of vol_ratio, using the continuous volume/EMA ratio directly. Preserves the core mechanism of robust range z-score and lagged return multiplied by relative volume intensity. The log transform may add noise; removing it targets improvement in IC and ICIR while maintaining the economic intuition of delayed adjustment following volume- and volatility-driven moves. Parent strength: robust range normalization and vol_ratio.

**Tools:** library_functions=alpha_tools.classify_volume_regime, talib.EMA

```python
def factor_lag_vol_robust_volratio(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['close'].shift(5) - 1
    range_ = df_copy['high'] - df_copy['low']
    range_med = range_.rolling(window=20, min_periods=20).median()
    range_mad = (range_ - range_med).abs().rolling(window=20, min_periods=20).mean()
    range_z = (range_ - range_med) / range_mad.replace(0.0, np.nan)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    raw = ret * range_z * vol_ratio
    factor = talib.EMA(raw, timeperiod=10)
    df_copy['factor_lag_vol_robust_volratio'] = factor
    return df_copy['factor_lag_vol_robust_volratio']
```

---

## Factor 777

**Formula:** `EMA5( (close/shift(close,5)-1) * log(volume/EMA30(volume)) * (ATR20/close) )`

**Rationale:** Captures delayed adjustment between volatility, volume, and returns. Lagged return (5-day) is amplified by high relative volume (log volume ratio) and high volatility (ATR/close), then smoothed with EMA5 to reduce noise. The product expresses how past return signals strengthen when both volume and volatility are elevated, reflecting slower information diffusion.

**Tools:** library_functions=np.log, talib.ATR

```python
def factor_lag_vol_volatility_adjusted(df):
    df_copy = df.copy()
    ret_5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    ema_vol = df_copy["volume"].ewm(span=30, min_periods=30, adjust=False).mean()
    log_vol_ratio = np.log(df_copy["volume"] / ema_vol.replace(0, np.nan))
    atr_20 = talib.ATR(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=20)
    vol_adj = atr_20 / df_copy["close"].replace(0, np.nan)
    raw = ret_5 * log_vol_ratio * vol_adj
    factor = raw.ewm(span=5, min_periods=5, adjust=False).mean()
    df_copy["factor_lag_vol_volatility_adjusted"] = factor
    return df_copy["factor_lag_vol_volatility_adjusted"]
```

---

## Factor 778

**Formula:** `raw = (close/shift(close,5)-1) * log(volume/EMA30(volume)) * (ATR20/close); factor = if high_resvol then EMA10(raw) else EMA5(raw)`

**Rationale:** Preserves the core mechanism of lagged return amplified by log volume ratio and volatility adjustment. Adds a lightweight regime adaptation: in high residual volatility states (style_gate_resvol_high), a longer EMA smoothing (10 days) reduces noise; otherwise, the original EMA5 preserves responsiveness. This targets improved stability in turbulent periods without weakening the parent's strength.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.log, np.where, talib.ATR

```python
def factor_lag_vol_volatility_adjusted_gate(df):
    df_copy = df.copy()
    ret_5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    ema_vol = df_copy["volume"].ewm(span=30, min_periods=30, adjust=False).mean()
    log_vol_ratio = np.log(df_copy["volume"] / ema_vol.replace(0, np.nan))
    atr_20 = talib.ATR(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=20)
    vol_adj = atr_20 / df_copy["close"].replace(0, np.nan)
    raw = ret_5 * log_vol_ratio * vol_adj
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    factor = np.where(high_resvol, raw.ewm(span=10, min_periods=10, adjust=False).mean(), raw.ewm(span=5, min_periods=5, adjust=False).mean())
    df_copy["factor_lag_vol_volatility_adjusted_gate"] = factor
    return df_copy["factor_lag_vol_volatility_adjusted_gate"]
```

---

## Factor 779

**Formula:** `EMA5/10( (close/shift(close,5)-1) * log(classify_volume_regime(volume,30).vol_ratio) * (ATR20/close) ), conditional on resvol_high gate`

**Rationale:** Simplifies the parent by replacing the manual EMA volume ratio with the classify_volume_regime active tool, reducing code while preserving the economic mechanism: lagged return amplified by normalized volume and volatility, with regime-adaptive smoothing for robustness in high residual volatility states.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.clip, np.log, np.where, talib.ATR

```python
def factor_lag_response_vol_tool_adaptive(df):
    df_copy = df.copy()
    ret_5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=30)
    log_vol_ratio = np.log(np.clip(vol_ratio, 1e-12, None))
    atr_20 = talib.ATR(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=20)
    vol_adj = atr_20 / df_copy["close"].replace(0, np.nan)
    raw = ret_5 * log_vol_ratio * vol_adj
    ema5 = raw.ewm(span=5, min_periods=5, adjust=False).mean()
    ema10 = raw.ewm(span=10, min_periods=10, adjust=False).mean()
    resvol_high_gate = df_copy["style_gate_resvol_high"]
    factor = np.where(resvol_high_gate > 0, ema10, ema5)
    df_copy["factor_lag_response_vol_tool_adaptive"] = factor
    return df_copy["factor_lag_response_vol_tool_adaptive"]
```

---

## Factor 780

**Formula:** `EMA5( (close/shift(close,5)-1) * log(volume/EMA30(volume)) * np.where(resvol_high, ATR10/close, ATR20/close) )`

**Rationale:** Preserves the core mechanism of lagged return amplified by log volume ratio. Adds a lightweight regime adaptation: in high residual volatility states, a shorter ATR window (10) is used to adjust the volatility scaling, making the factor more responsive to recent volatility spikes; in low residual volatility, a longer ATR window (20) provides stable scaling. This targets improved adaptation to changing volatility regimes without altering the core economic signal.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.log, np.where, talib.ATR

```python
def factor_lag_volatility_regime_adapted(df):
    df_copy = df.copy()
    ret_5 = df_copy["close"] / df_copy["close"].shift(5) - 1
    ema_vol = df_copy["volume"].ewm(span=30, min_periods=30, adjust=False).mean()
    log_vol_ratio = np.log(df_copy["volume"] / ema_vol.replace(0, np.nan))
    atr_short = talib.ATR(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=10)
    atr_long = talib.ATR(df_copy["high"], df_copy["low"], df_copy["close"], timeperiod=20)
    vol_adj_short = atr_short / df_copy["close"].replace(0, np.nan)
    vol_adj_long = atr_long / df_copy["close"].replace(0, np.nan)
    resvol_high = df_copy["style_gate_resvol_high"]
    vol_adj = np.where(resvol_high > 0, vol_adj_short, vol_adj_long)
    raw = ret_5 * log_vol_ratio * vol_adj
    factor = raw.ewm(span=5, min_periods=5, adjust=False).mean()
    df_copy["factor_lag_volatility_regime_adapted"] = factor
    return df_copy["factor_lag_volatility_regime_adapted"]
```

---

## Factor 790

**Formula:** `(21-day high - close) / 21-day high / 21-day close std`

**Rationale:** Mutation: ablation of volume confirmation modifier from parent factor. The parent's volume ratio weighted drawdown depth may introduce noise; removing it focuses on the core drawdown/vol signal which was the strength of the grandparent lineage. Preserves 21-day lookbacks and cs_rank to maintain rank stability (parent strength). Targets improving IC and ICIR by simplifying.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_drawdown_vol_norm_clean(df):
    df_copy = df.copy()
    high_21 = df_copy['high'].rolling(21, min_periods=21).max()
    drawdown = (high_21 - df_copy['close']) / high_21.replace(0, np.nan)
    vol_std = df_copy['close'].rolling(21, min_periods=21).std(ddof=1).replace(0, np.nan)
    factor = drawdown / vol_std
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = 'factor_drawdown_vol_norm_clean'
    return factor
```

---

## Factor 805

**Formula:** `stress = -rolling_skew(returns, 40); factor = stress * (volume / rolling_mean(volume, 20)) * np.where(resvol_high, 1.2, 0.8)`

**Rationale:** Preserve core negative skew stress (tail-risk) multiplied by volume ratio. Add a lightweight residual-volatility regime gate to amplify the factor during high-residual-volatility periods when tail-risk premiums are elevated, and attenuate during low-resvol periods to reduce noise. This targets IC and MI improvement while preserving the rank ordering strength (RankIC 0.0309, RankICIR 0.3274).

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.where

```python
def factor_negative_skew_stress_gated(df):
    df_copy = df.copy()
    returns = df_copy['close'].pct_change()
    rolling_skew = returns.rolling(40, min_periods=40).skew()
    stress = -rolling_skew
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    factor = stress * vol_ratio
    resvol_high = df_copy['style_gate_resvol_high']
    weight = np.where(resvol_high > 0, 1.2, 0.8)
    factor = factor * weight
    factor.name = 'factor_negative_skew_stress_gated'
    return factor
```

---

## Factor 806

**Formula:** `factor = EMA3( -rolling_skew(returns,30) * (volume / EMA20(volume)) )`

**Rationale:** Mutation of parent g2_crossover_g1_mutation_cogalpha_tailrisk_002_faf7abab: simplified by removing the log transform from volume ratio to recover linear volume confirmation strength, reduced skew window from 40 to 30 days for more timely tail-risk capture, and added a 3-day EMA smoothing to reduce noise. Preserves the core negative-skew times volume-ratio mechanism. Target bottleneck: improve low MI (0.010) while retaining parent RankICIR strength (~0.33).

```python
def factor_negative_skew_volume_raw_smoothed(df):
    df_copy = df.copy()
    returns = df_copy['close'].pct_change()
    skew = returns.rolling(30, min_periods=30).skew()
    stress = -skew
    vol_ema = df_copy['volume'].ewm(span=20, adjust=False).mean()
    vol_ratio = df_copy['volume'] / vol_ema.replace(0.0, np.nan)
    raw = stress * vol_ratio
    factor = raw.ewm(span=3, adjust=False).mean()
    factor.name = 'factor_negative_skew_volume_raw_smoothed'
    return factor
```

---

## Factor 809

**Formula:** `EMA_gated(lag_ret_5 * log(volume/EMA15(volume))) gated by resvol_high (smoothing span 5 or 10)`

**Rationale:** Mutate parent by removing the liquidity_high gate (lightweight modifier) to simplify and avoid overfitting to a single market regime. Adjust smoothing windows from (3,8) to (5,10) to reduce noise while preserving the core volume-confirmed lagged return mechanism. Keep the resvol_high gate to adapt smoothing span based on residual volatility regime. Parent strength target: RankIC and ICIR stability from parent; aim to improve robustness by eliminating an unnecessary auxiliary parameter.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.log, np.where

```python
def factor_lagret_volratio_gated_smooth(df):
    df_copy = df.copy()
    lag_ret = df_copy['close'].pct_change(5).shift(1)
    vol_ema = df_copy['volume'].ewm(span=15, adjust=False).mean()
    vol_ratio = np.log((df_copy['volume'] / vol_ema).clip(lower=1e-12))
    raw = lag_ret * vol_ratio
    use_long_smooth = (df_copy['style_gate_resvol_high'] > 0)
    smooth_short = raw.ewm(span=5, adjust=False).mean()
    smooth_long = raw.ewm(span=10, adjust=False).mean()
    factor = np.where(use_long_smooth, smooth_long, smooth_short)
    df_copy['factor_lagret_volratio_gated_smooth'] = factor
    return df_copy['factor_lagret_volratio_gated_smooth']
```

---

## Factor 825

**Formula:** `EMA5( (close/shift5 - 1) * log(volume/EMA20(volume)) * (1 + ATR20/close) )`

**Rationale:** This factor captures the delayed adjustment between volatility, volume, and returns. Lagged 5-day return is multiplied by log volume ratio (to emphasize unusual volume) and amplified by the current volatility level (ATR/close). The EMA5 smoothing models the gradual incorporation of information into price, representing lagged response. The combination targets stable rank ordering across IC, RankIC, ICIR, RankICIR, and MI.

**Tools:** library_functions=np.log, pd.DataFrame

```python
def factor_vol_lag_ret_smoothed(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    # Lagged 5-day return
    lag_ret_5 = close / close.shift(5) - 1.0
    # Log volume ratio relative to 20-day EMA
    vol_ema = volume.ewm(span=20, min_periods=20).mean()
    vol_ratio = np.log(volume / vol_ema.replace(0, np.nan))
    # ATR / close as volatility measure
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean()
    vol_factor = atr / close.replace(0, np.nan)
    # Combine: lagged return * log volume ratio * (1 + volatility factor)
    raw = lag_ret_5 * vol_ratio * (1.0 + vol_factor)
    # Smooth with EMA5 to capture delayed adjustment
    df_copy['factor_vol_lag_ret_smoothed'] = raw.ewm(span=5, min_periods=5).mean()
    return df_copy['factor_vol_lag_ret_smoothed']
```

---

## Factor 826

**Formula:** `EMA5( (close/shift5-1) * log(volume/EMA20(volume)) * (1+ATR20/close) * (1.3 if resvol_high else 0.9) )`

**Rationale:** Mutates the trend adjustment from a fixed EMA200 binary to a regime-aware amplification based on resvol_high gate. In high residual volatility environments, the signal is amplified 1.3x to capture stronger trend effects; in low volatility, it is attenuated to 0.9x to reduce noise. This preserves the elite lagged-response mechanism (lagret_5 * log volume ratio * volatility amplification) while making the trend conditional on market stress.

**Tools:** library_functions=np.log, np.where, pd.DataFrame, pd.Series

```python
def factor_lagret_vol_trend_gated(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    lag_ret_5 = close / close.shift(5) - 1.0
    vol_ema = volume.ewm(span=20, min_periods=20).mean()
    vol_ratio = np.log(volume / vol_ema.replace(0, np.nan))
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean()
    vol_factor = atr / close.replace(0, np.nan)
    raw = lag_ret_5 * vol_ratio * (1.0 + vol_factor)
    high_resvol = df_copy.get('style_gate_resvol_high', pd.Series(0, index=df_copy.index))
    trend_mult = np.where(high_resvol > 0, 1.3, 0.9)
    raw_trend = raw * trend_mult
    factor = raw_trend.ewm(span=5, min_periods=5).mean()
    df_copy['factor_lagret_vol_trend_gated'] = factor
    return df_copy['factor_lagret_vol_trend_gated']
```

---

## Factor 828

**Formula:** `EMA5( (close/shift5-1) * log(volume/EMA10(volume)) * (1+ATR10/close) * (1.25 if resvol_high else 0.85) )`

**Rationale:** Reduced volume ratio and ATR windows from 20 to 10 to capture shorter-term volatility and volume dynamics while preserving the core lagged response mechanism. Adjusted trend multipliers slightly to 1.25/0.85 for finer regime adaptation. The resvol_high gate still amplifies the signal in high residual volatility regimes.

**Tools:** library_functions=np.log, np.where, pd.DataFrame, pd.Series

```python
def factor_lagret_vol_trend_gated_v2(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    lag_ret_5 = close / close.shift(5) - 1.0
    vol_ema = volume.ewm(span=10, min_periods=10).mean()
    vol_ratio = np.log(volume / vol_ema.replace(0, np.nan))
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
    atr = tr.rolling(10, min_periods=10).mean()
    vol_factor = atr / close.replace(0, np.nan)
    raw = lag_ret_5 * vol_ratio * (1.0 + vol_factor)
    high_resvol = df_copy.get('style_gate_resvol_high', pd.Series(0, index=df_copy.index))
    trend_mult = np.where(high_resvol > 0, 1.25, 0.85)
    raw_trend = raw * trend_mult
    factor = raw_trend.ewm(span=5, min_periods=5).mean()
    df_copy['factor_lagret_vol_trend_gated_v2'] = factor
    return df_copy['factor_lagret_vol_trend_gated_v2']
```

---

## Factor 830

**Formula:** `EMA5( (close/shift5-1) * log(volume/EMA20(volume)) * (1+ATR20/close) * (1.3 if resvol_high else 0.9) )`

**Rationale:** Mutation: Changed the regime gate from style_gate_momentum_high to style_gate_resvol_high to better align with the elite parent's successful pattern, using multipliers 1.3x for high residual volatility and 0.9x for low to capture stronger trend effects under market stress while attenuating noise. This preserves the core lagged-response mechanism of lagret_5 * log volume ratio * volatility amplification.

**Tools:** library_functions=np.log, np.where, pd.DataFrame, pd.Series

```python
def factor_vol_lag_ret_resvol_gated(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    lag_ret_5 = close / close.shift(5) - 1.0
    vol_ema = volume.ewm(span=20, min_periods=20).mean()
    vol_ratio = np.log(volume / vol_ema.replace(0, np.nan))
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean()
    vol_factor = atr / close.replace(0, np.nan)
    raw = lag_ret_5 * vol_ratio * (1.0 + vol_factor)
    high_resvol = df_copy.get('style_gate_resvol_high', pd.Series(0, index=df_copy.index))
    trend_mult = np.where(high_resvol > 0, 1.3, 0.9)
    raw_trend = raw * trend_mult
    factor = raw_trend.ewm(span=5, min_periods=5).mean()
    df_copy['factor_vol_lag_ret_resvol_gated'] = factor
    return df_copy['factor_vol_lag_ret_resvol_gated']
```

---

## Factor 853

**Formula:** `EMA(daily_ret,10) * clip(1 - ATR5/ATR20, -1, 1)`

**Rationale:** Simplification of parent: removed the signed square root transformation on the contraction term and added a clip to [-1,1] to bound extreme values. This preserves the core trend-contraction interaction while reducing unnecessary nonlinearity, potentially improving rank stability and interpretability. Keeps cross_sectional_transform as none to maintain raw payoff mechanism.

**Tools:** library_functions=np.clip, talib.ATR, talib.EMA

```python
def factor_smoothed_trend_vol_contraction_simplified(df):
    df_copy = df.copy()
    daily_ret = df_copy['close'].pct_change()
    trend = talib.EMA(daily_ret, timeperiod=10)
    atr5 = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=5)
    atr20 = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=20)
    contraction = 1 - atr5 / atr20.replace(0, np.nan)
    contraction = np.clip(contraction, -1, 1)
    factor = trend * contraction
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = 'factor_smoothed_trend_vol_contraction_simplified'
    return factor
```

---

## Factor 865

**Formula:** `(EMA(20) - EMA(40)) / EMA(40)`

**Rationale:** Simplified trend factor using a single EMA cross (20/40) to capture persistent momentum direction. Removes the liquidity gate and short-term cross from the parent, which introduced noise and complexity. The single EMA cross preserves the core trend-following mechanism while improving robustness and interpretability.

```python
def factor_ema_trend_simple(df):
    df_copy = df.copy()
    close = df_copy['close']
    fast_ema = close.ewm(span=20, adjust=False).mean()
    slow_ema = close.ewm(span=40, adjust=False).mean()
    factor = (fast_ema - slow_ema) / slow_ema.replace(0, np.nan)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_copy['factor_ema_trend_simple'] = factor
    return df_copy['factor_ema_trend_simple']
```

---

## Factor 879

**Formula:** `if style_gate_resvol_high > 0 then |overnight_ret+intraday_ret| * vol_ratio else (ema(close,10)-ema(close,30))/ema(close,30) * clip(vol_ratio, 0.5, 2.0)`

**Rationale:** Preserves the regime-gated structure from parent 2 (fast mean-reversion in high resvol, slow trend in low resvol) but replaces the fast signal with the total daily price impact (absolute overnight+intraday return scaled by volume ratio) borrowed from parent 1. This substitution captures a more comprehensive price impact signal in volatile periods, aiming to improve IC while maintaining rank stability from the trend component.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, alpha_tools.decompose_overnight_intraday, np.abs, np.where

```python
def factor_crossover_vol_impact_trend_gated(df):
    df_copy = df.copy()
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    is_high_vol, is_low_vol, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    vol_factor = vol_ratio.clip(lower=0.5, upper=2.0)
    ema_short = df_copy["close"].ewm(span=10, adjust=False).mean()
    ema_long = df_copy["close"].ewm(span=30, adjust=False).mean()
    slow_trend = (ema_short - ema_long) / ema_long
    slow_trend_adjusted = slow_trend * vol_factor
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    total_ret = overnight_ret + intraday_ret
    fast_impact = np.abs(total_ret) * vol_ratio
    df_copy["factor_crossover_vol_impact_trend_gated"] = np.where(high_resvol, fast_impact, slow_trend_adjusted)
    return df_copy["factor_crossover_vol_impact_trend_gated"]
```

---

## Factor 881

**Formula:** `if resvol_high then (close - (high+low)/2)/(high-low) * (V/EMA(V,20)) else (EMA(C,20)-EMA(C,50))/EMA(C,50)`

**Rationale:** Crossover preserving parent ARG-001 mutation's regime-gated mean-reversion/trend structure and borrowing parent AVS-001 mutation's volume ratio as a lightweight weight on the fast mean-reversion signal to enhance reversal detection when volume is elevated relative to its norm. The slow trend component remains unchanged, keeping the design compact while targeting IC improvement via volume confirmation.

**Tools:** style_gates=style_gate_resvol_high; library_functions=np.where

```python
def factor_volratio_resvol_trend_gated(df):
    df_copy = df.copy()
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    mid = (df_copy["high"] + df_copy["low"]) / 2
    fast_ret = (df_copy["close"] - mid) / (df_copy["high"] - df_copy["low"] + 1e-8)
    ema_short = df_copy["close"].ewm(span=20, adjust=False).mean()
    ema_long = df_copy["close"].ewm(span=50, adjust=False).mean()
    slow_trend = (ema_short - ema_long) / ema_long
    ema_vol = df_copy["volume"].ewm(span=20, adjust=False).mean()
    vol_ratio = df_copy["volume"] / ema_vol.replace(0, np.nan)
    fast_ret_scaled = fast_ret * vol_ratio
    df_copy["factor_volratio_resvol_trend_gated"] = np.where(high_resvol, fast_ret_scaled, slow_trend)
    return df_copy["factor_volratio_resvol_trend_gated"]
```

---

## Factor 892

**Formula:** `if resvol_high: factor = mean(5, (close-open)/(high-low)) * vol_ratio; else: factor = ROC(10) * vol_ratio`

**Rationale:** Mutation of regime_vol_adaptive_trend_v3: increased intraday pressure window to 5 for smoother mean-reversion signal, and simplified momentum branch to use raw 10-day rate of change (removed double smoothing) for more responsive trend capture. Volume ratio scaling is preserved as the lightweight modifier, and regime switching via style_gate_resvol_high remains the core mechanism.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.where, pd.Series

```python
def factor_regime_vol_adaptive_trend_v4(df):
    df_copy = df.copy()
    high_resvol = df_copy['style_gate_resvol_high'] > 0
    imp = (df_copy['close'] - df_copy['open']) / (df_copy['high'] - df_copy['low'] + 1e-8)
    imp = imp.rolling(5, min_periods=5).mean()
    mom = df_copy['close'] / df_copy['close'].shift(10) - 1
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    factor = np.where(high_resvol, imp * vol_ratio, mom * vol_ratio)
    factor = pd.Series(factor, index=df_copy.index)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    df_copy['factor_regime_vol_adaptive_trend_v4'] = factor
    return df_copy['factor_regime_vol_adaptive_trend_v4']
```

---

## Factor 894

**Formula:** `if resvol_high: factor = mean(3, (close-open)/(high-low)) * vol_ratio; else: factor = mean(5, ROC(10)) * vol_ratio`

**Rationale:** Crossover of regime_vol_adaptive_trend_v2 (primary) and regime_vol_adaptive_trend_v3 (secondary). Primary mechanism preserved: regime-switching between intraday price impact and smoothed momentum, both scaled by volume ratio from classify_volume_regime. Lightweight modifier borrowed from secondary: shorter impact window (3 vs 5) to capture faster mean reversion in high residual volatility regimes, while keeping momentum window at 5 to maintain trend stability. Targets improved IC and RankIC responsiveness without sacrificing the parent strength of continuous volume-confirmed signals.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.where, pd.Series

```python
def factor_regime_vol_adaptive_trend_v4(df):
    df_copy = df.copy()
    high_resvol = df_copy['style_gate_resvol_high'] > 0
    imp = (df_copy['close'] - df_copy['open']) / (df_copy['high'] - df_copy['low'] + 1e-8)
    imp = imp.rolling(3, min_periods=3).mean()
    mom = df_copy['close'] / df_copy['close'].shift(10) - 1
    mom = mom.rolling(5, min_periods=5).mean()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    factor = np.where(high_resvol, imp * vol_ratio, mom * vol_ratio)
    factor = pd.Series(factor, index=df_copy.index)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    df_copy['factor_regime_vol_adaptive_trend_v4'] = factor
    return df_copy['factor_regime_vol_adaptive_trend_v4']
```

---

## Factor 907

**Formula:** `(50 - RSI(14)) * vol_ratio, where vol_ratio = volume / EMA(volume, 20)`

**Rationale:** Combine RSI mean reversion with volume confirmation: scale the RSI reversal signal by the volume ratio (current volume / EMA volume) to amplify signals when volume is high, as strong volume often confirms reversal potential. This replaces the previous binary resvol gate with a continuous volume adjustment.

**Tools:** library_functions=alpha_tools.classify_volume_regime, pd.Series, talib.RSI

```python
def factor_rsi_volume_confirmed_reversal(df):
    df_copy = df.copy()
    close_float = df_copy["close"].values.astype(np.float64)
    rsi = talib.RSI(close_float, timeperiod=14)
    rsi = pd.Series(rsi, index=df_copy.index)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    signal = 50 - rsi
    factor = signal * vol_ratio
    df_copy["factor_rsi_volume_confirmed_reversal"] = factor
    return df_copy["factor_rsi_volume_confirmed_reversal"]
```

---

## Factor 911

**Formula:** `(50 - RSI(14)) * (1.5 if style_gate_resvol_high else 0.5) * vol_ratio`

**Rationale:** Combine two parents: preserve the residual volatility regime gating (resvol_high) from the qualified parent lineage, and borrow the volume ratio multiplier from the volume-confirmed reversal parent. The resvol gate amplifies the RSI mean-reversion signal in high-residual-volatility regimes where overreactions revert strongly, while the volume ratio emphasizes periods of abnormal trading activity that often accompany price dislocations. The volume ratio is used directly (not squared) as a lightweight modifier to avoid over-weighting.

**Tools:** style_gates=style_gate_resvol_high; library_functions=alpha_tools.classify_volume_regime, np.where, pd.Series, talib.RSI

```python
def factor_rsi_reversal_resvol_volume(df):
    df_copy = df.copy()
    close_float = df_copy["close"].values.astype(np.float64)
    rsi = talib.RSI(close_float, timeperiod=14)
    rsi = pd.Series(rsi, index=df_copy.index)
    raw_signal = 50 - rsi
    high_resvol = df_copy["style_gate_resvol_high"] > 0
    gate_scaled = raw_signal * np.where(high_resvol, 1.5, 0.5)
    is_high_vol, is_low_vol, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20, high_threshold=1.5, low_threshold=0.6)
    factor = gate_scaled * vol_ratio
    df_copy["factor_rsi_reversal_resvol_volume"] = factor
    return df_copy["factor_rsi_reversal_resvol_volume"]
```

---

## Factor 935

**Formula:** `(1 - depth_20) * (1 - duration_20) * (SMA(range,5)/SMA(range,30)) * (volume/SMA(volume,20))`

**Rationale:** Crossover preserves the primary drawdown geometry mechanism from the qualified parent (depth, duration, range ratio) which provides strong RankIC and RankICIR. The lightweight volume ratio from the secondary parent adds a volume confirmation dimension to improve linear IC and ICIR, targeting the bottleneck of low IC. The composite signal is high when drawdowns are shallow, short, and accompanied by elevated volume (panic-driven reversals), and low when drawdowns are deep, prolonged, or occur on low volume. Cross-sectional rank standardizes the raw composite for consistent ordinal comparison across assets.

**Tools:** cross_sectional_transform=cs_rank

```python
def factor_drawdown_depth_duration_range_vol(df):
    df_copy = df.copy()
    high20 = df_copy['close'].rolling(20).max()
    depth = (high20 - df_copy['close']) / high20.replace(0.0, np.nan)
    in_dd = df_copy['close'] < high20
    duration = in_dd.rolling(20, min_periods=20).sum() / 20.0
    range_ = df_copy['high'] - df_copy['low']
    avg_short = range_.rolling(5, min_periods=5).mean()
    avg_long = range_.rolling(30, min_periods=30).mean()
    range_ratio = avg_short / avg_long.replace(0.0, np.nan)
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    factor = (1.0 - depth) * (1.0 - duration) * range_ratio * vol_ratio
    df_copy['factor_drawdown_depth_duration_range_vol'] = factor
    return df_copy['factor_drawdown_depth_duration_range_vol']
```

---

## Factor 987

**Formula:** `-drawdown / volatility`

**Rationale:** Preserves parent's core intuition of drawdown severity normalized by recent volatility. Simplifies by removing the duration component, focusing solely on drawdown depth relative to short-term volatility. This reduction in complexity aims to reduce noise and improve signal-to-noise ratio, targeting the low IC observed in the parent.

```python
def factor_drawdown_depth_normalized(df):
    df_copy = df.copy()
    rolling_max = df_copy["close"].rolling(252, min_periods=50).max()
    drawdown = df_copy["close"] / rolling_max - 1
    depth = -drawdown
    ret = df_copy["close"].pct_change()
    vol = ret.rolling(20, min_periods=10).std()
    factor = depth / vol.replace(0, np.nan)
    factor = factor.fillna(0)
    df_copy["factor_drawdown_depth_normalized"] = factor
    return df_copy["factor_drawdown_depth_normalized"]
```

---

## Factor 1038

**Formula:** `drawdown = close / rolling_max(close,60) - 1; normalized_drawdown = drawdown / rolling_std(returns,20); factor = normalized_drawdown`

**Rationale:** Pure drawdown depth normalized by trailing volatility. Removes the volume consistency multiplier which introduced threshold instability, targeting robustness. Drawdown signals mean-reversion potential after severe drops.

```python
def factor_drawdown_recovery(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(60, min_periods=60).max()
    drawdown = df_copy['close'] / rolling_max - 1.0
    returns = df_copy['close'].pct_change()
    vol = returns.rolling(20, min_periods=20).std().replace(0, np.nan)
    factor = drawdown / vol
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = "factor_drawdown_recovery"
    return factor
```

---

## Factor 1040

**Formula:** `drawdown = close/rolling_max(close,60)-1; vol = rolling_std(returns,20); normalized_drawdown = drawdown/vol; persistence = ATR(40)/ATR(10); factor = normalized_drawdown * clip(persistence,0.5,2.0)`

**Rationale:** Combine normalized drawdown (primary parent strength) with volatility persistence ratio (ATR40/ATR10) from secondary parent. The persistence ratio modulates the drawdown signal: when longer-term volatility is elevated relative to short-term volatility, the drawdown is given higher weight, capturing environments where volatility clustering may amplify mean-reversion. This is a compact crossover that preserves the core drawdown-recovery geometry while adding a lightweight volatility persistence adjustment.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_drawdown_vol_persistence(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(60, min_periods=60).max()
    drawdown = df_copy['close'] / rolling_max - 1.0
    returns = df_copy['close'].pct_change()
    vol = returns.rolling(20, min_periods=20).std().replace(0, np.nan)
    normalized_drawdown = drawdown / vol
    high = df_copy['high'].values.astype('float64')
    low = df_copy['low'].values.astype('float64')
    close = df_copy['close'].values.astype('float64')
    atr40 = talib.ATR(high, low, close, timeperiod=40)
    atr10 = talib.ATR(high, low, close, timeperiod=10)
    atr40 = pd.Series(atr40, index=df_copy.index)
    atr10 = pd.Series(atr10, index=df_copy.index)
    persistence = atr40 / atr10.replace(0, np.nan)
    persistence = persistence.clip(lower=0.5, upper=2.0)
    factor = normalized_drawdown * persistence
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = "factor_drawdown_vol_persistence"
    return factor
```

---

## Factor 1042

**Formula:** `drawdown = close/rolling_max(close,60)-1; normalized_drawdown = drawdown / ATR(20); factor = fillna(replace(normalized_drawdown, inf, nan),0)`

**Rationale:** Simplify the parent's dual-mechanism (drawdown normalized by returns std multiplied by volatility persistence ratio) to a single core mechanism: drawdown depth normalized by ATR. ATR captures recent true-range volatility more robustly than returns std, and the removal of the persistence ratio halves the number of tunable windows, reducing overfitting risk while preserving the mean-reversion intuition. This targets improved IC without sacrificing the parent's strong RankICIR.

**Tools:** library_functions=pd.Series, talib.ATR

```python
def factor_drawdown_atr_recovery(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(60, min_periods=60).max()
    drawdown = df_copy['close'] / rolling_max - 1.0
    high = df_copy['high'].values.astype('float64')
    low = df_copy['low'].values.astype('float64')
    close = df_copy['close'].values.astype('float64')
    atr20 = talib.ATR(high, low, close, timeperiod=20)
    atr20 = pd.Series(atr20, index=df_copy.index).replace(0, np.nan)
    normalized_drawdown = drawdown / atr20
    factor = normalized_drawdown.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = "factor_drawdown_atr_recovery"
    return factor
```

---

## Factor 1044

**Formula:** `factor = drawdown / vol, where drawdown = close/rolling_max(60)-1, vol = rolling_std(returns,20)`

**Rationale:** Simplified to a single core mechanism: drawdown depth normalized by trailing volatility, capturing mean-reversion potential after severe drops. Removing the volatility persistence ratio modifier reduces complexity and potential instability from conditional division, targeting improved robustness and rank stability.

```python
def factor_drawdown_normalized(df):
    df_copy = df.copy()
    rolling_max = df_copy['close'].rolling(60, min_periods=60).max()
    drawdown = df_copy['close'] / rolling_max - 1.0
    returns = df_copy['close'].pct_change()
    vol = returns.rolling(20, min_periods=20).std().replace(0, np.nan)
    factor = drawdown / vol
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor.name = "factor_drawdown_normalized"
    return factor
```
