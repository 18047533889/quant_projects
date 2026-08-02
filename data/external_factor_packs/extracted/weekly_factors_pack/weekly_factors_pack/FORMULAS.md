# 因子公式清单

## factor_mut_hc004_volpctrank_s12_clip2_5_volwin6

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_53a3e3f7`
- train IC / RankIC: 0.0272 / 0.0499
- test RankIC: N/A

```
ema( (close - delay(close, 1))/delay(close, 1) * rank_pct6(volume) * ((high-low)/close) , span=12 ) clip ±2.5
```

Mutation adjusting volume rank window from 8 to 6 to increase responsiveness of the volume confirmation signal, targeting improved IC while preserving the strong RankICIR (0.4182) observed in the parent. Keeps smoothing span (12) and clipping (±2.5) unchanged.

---

## factor_crossover_herding_stress_span12_volrank_clip3

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_b0ca3bdd`
- train IC / RankIC: 0.0269 / 0.0492
- test RankIC: N/A

```
ema( ts_pct(close) * rank_pct8(volume) * ((high-low)/close), span=12 ) clip ±3
```

Crossover preserving primary parent mut_herding_momentum_stress_span12's core stress momentum mechanism with smoothing span 12 and clipping at ±3 to maintain strong RankICIR (0.4211), and borrowing the volume percentile rank normalization (rolling 8-day rank) from secondary parent crossover_hc004_volpctrank_s12_clip2_5 to potentially improve ranking stability and reduce outlier impact. Targets improved IC and RankIC while preserving the high RankICIR.

---

## factor_mut_hc004_volpctrank_w8

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_b719027d`
- train IC / RankIC: 0.0269 / 0.0492
- test RankIC: N/A

```
ema( (close - delay(close, 1))/delay(close, 1) * rank_pct8(volume) * ((high-low)/close) , span=12 )
```

Mutation of mut_hc004_volpctrank_w12: reduced volume rank window from 12 to 8 to increase responsiveness of the volume confirmation signal, targeting improved IC while preserving the smoothing span (12) and clipping threshold (±3) that support strong RankICIR. The parent-strength target is RankICIR=0.4211, and the bottleneck is IC=0.0270. This compact parameter change avoids adding new components and keeps the core volume rank stress momentum mechanism intact.

---

## factor_vol_price_log_range_gate_clipped_boost_5day_mut_001

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_27a292b1`
- train IC / RankIC: 0.0241 / 0.0483
- test RankIC: N/A

```
EMA12( pct_change * log1p(volume/SMA5(volume)) * (1 + min(max(0, zscore5(range)), 2.0)) )
```

Mutation of parent vol_price_log_range_gate_clipped_boost_5day_cross_001: preserve core volume-weighted price change with log volume compression, 5-day rolling windows, and clipped range z-score boost (cap 2.0). Increase EMA smoothing span from 10 to 12 to further reduce noise and target improvement in ICIR while maintaining strong RankIC and RankICIR.

---

## factor_vol_price_log_range_gate_clipped_10day_mut_003

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_c1a04bc5`
- train IC / RankIC: 0.0246 / 0.0479
- test RankIC: N/A

```
EMA12( pct_change * log1p(volume/SMA5(volume)) * (1 + cap(zscore10(range), 0, 2)) )
```

Mutation of vol_price_log_range_gate_clipped_10day_mut_002: preserve core volume-weighted price change with log volume compression and clipped range z-score boost. Increase EMA smoothing span from 8 to 12 to further reduce noise and improve ICIR and RankICIR stability while maintaining strong IC and RankIC.

---

## factor_vol_surge_regime_gated

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_c4358b90`
- train IC / RankIC: 0.0249 / 0.0456
- test RankIC: 0.0397

```
ema(intraday_return * safe_div(numerator=volume, denominator=ts_max(volume, 20)) * where(high - low > ts_quantile(high - low, 20, 0.5), 1, 0), 20)
```

Primary mechanism from imbalance_002: intraday return scaled by volume surge to capture directional pressure. Lightweight modifier from avr_002_mut001: volatility regime indicator (range > 20-day median) to filter out low-volatility periods where the signal may be noisy. This combination targets improved ICIR while preserving the core directional strength.

---

## factor_volume_confirmed_momentum_tanh_gated

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_0a4e4a21`
- train IC / RankIC: 0.0368 / 0.0453
- test RankIC: 0.0054

```
(sigmoid(2 * (- safe_div(numerator=2 * close - high - low, denominator=high - low) * safe_div(numerator=ts_mean(high - low, 5), denominator=ts_mean(close, 5)))) * 2 - 1)
```

Preserve strong IC and RankIC from parent crossover_vcmi_1 (volume-confirmed momentum gated by intraday position). Borrow tanh nonlinearity from parent crossover_liquidity_tanh_reversal_001 to compress extreme outlier values, improving MI and rank stability without altering the core momentum mechanism.

---

## factor_intraday_return_volume_surge

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_6b0f76d3`
- train IC / RankIC: 0.0254 / 0.0452
- test RankIC: 0.0380

```
ema(intraday_return * safe_div(numerator=volume, denominator=ts_max(volume, 20)), 20)
```

Identify acute directional pressure by scaling intraday return (close/open - 1) by a volume surge factor: volume divided by its 20-day rolling maximum. Large positive returns with volume surge indicate strong buying; negative returns with volume surge indicate strong selling. Smoothed with EWM to reduce noise.

---

## factor_herding_momentum_stress

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_ef639934`
- train IC / RankIC: 0.0242 / 0.0451
- test RankIC: N/A

```
ema( (close - delay(close, 1))/delay(close, 1) * (volume / ts_mean(volume,5)) * ((high-low)/close) , span=5 )
```

Volume-confirmed directional momentum amplified by intraday stress (range/close). The product captures herding behavior when price moves attract volume, and stress range magnifies crowded episodes. EMA(5) smoothing stabilizes the signal for rank consistency, while clipping controls outliers.

---

## factor_vol_weighted_pct_range_clip_ema10

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_948c7055`
- train IC / RankIC: 0.0239 / 0.0450
- test RankIC: N/A

```
EMA10( pct_change * (volume/SMA20(volume)) * (1 + cap(zscore10(range), 0, 2)) )
```

Mutation of price_volume_coherence_range_gate_clip_cross_001: preserve core volume-weighted price change with SMA20 volume ratio and clipped range z-score boost. Increase EMA span from 5 to 10 to reduce noise and improve ICIR and RankICIR stability while maintaining strong RankIC and IC.

---

## factor_volume_confirmed_momentum

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_d37770d2`
- train IC / RankIC: 0.0249 / 0.0432
- test RankIC: 0.0094

```
(daily_return) * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)))
```

Volume-confirmed price momentum: amplify returns when volume is above its 20-day average, dampen when below. Exponential smoothing stabilizes the signal, balancing IC and RankICIR.

---

## factor_directional_vol_volbase25_smooth_dir_range10

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_1476a21c`
- train IC / RankIC: 0.0297 / 0.0432
- test RankIC: N/A

```
range_pct = (high-low)/close; smooth_range = ema(range_pct, span=10); vol_ratio = volume / ts_mean(volume,25); smooth_vol = ema(vol_ratio, span=15); direction = (close-open)/close; smooth_direction = ema(direction, span=3); factor = smooth_range * smooth_vol * smooth_direction
```

Mutation of alpha-crossover-gen13-001. Preserves core directional volume-adjusted volatility mechanism (smooth_range EWMA span=8 increased to span=10, volume ratio baseline 25-day, single EWMA vol_ratio smoothing span=15, direction, smooth_direction EWMA span=3). The mutation increases the EWMA span for range smoothing from 8 to 10 to further stabilize the range component, aiming to improve IC consistency while preserving the strong rank-ordering strength (RankIC, RankICIR) inherited from the pa

---

## factor_adaptive_vol_directional_smooth_range10

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_f1858c32`
- train IC / RankIC: 0.0297 / 0.0432
- test RankIC: N/A

```
smooth_range = ema((high-low)/close, span=10); vol_ratio = volume / ts_mean(volume,25); smooth_vol = ema(vol_ratio, span=15); direction=(close-open)/close; smooth_direction=ema(direction, span=3); factor = smooth_range * smooth_vol * smooth_direction
```

Mutation of alpha-crossover-gen14-001. Preserves core directional volume-adjusted volatility mechanism (smooth_range EWMA span=10, volume ratio baseline 25-day, smooth_direction EWMA span=3). Simplifies volume smoothing by replacing the adaptive sigmoid blending of two EWMA with a single EWMA (span=15) as in the effective parent alpha-mutation-gen13-002, aiming to reduce complexity and potentially improve IC stability while preserving strong rank-ordering strength (RankIC, RankICIR). Only one co

---

## factor_intraday_volume_spike

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_e9cbfd57`
- train IC / RankIC: 0.0253 / 0.0430
- test RankIC: 0.0330

```
ema(intraday_return * safe_div(numerator=volume, denominator=ts_mean(volume, 20)), 10)
```

Measures intraday directional pressure scaled by volume relative to its 20-day average. Volume spikes relative to average indicate conviction behind the move. Smoothed with a shorter 10-period EWMA to balance responsiveness and noise reduction, targeting improved ICIR while preserving the core mechanism of the parent.

---

## factor_herding_vol_stress

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_340f0ad9`
- train IC / RankIC: 0.0169 / 0.0410
- test RankIC: 0.0269

```
ema(safe_div(numerator=close - open, denominator=high - low) * safe_div(numerator=volume, denominator=ts_quantile(volume, 10, 0.5)) * safe_div(numerator=high - low, denominator=close), 5)
```

Crossover: primary mechanism from mutation_herding_v2 (directional herding scaled by volume ratio, smoothed with EMA) combined with volatility normalization from candidate_004 (stress_range = (high-low)/close). The stress_range scaling amplifies the herding signal in high-volatility regimes, aiming to improve ICIR and RankICIR while preserving the parent's strong RankIC and IC.

---

## factor_intraday_reversal

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_e5e44222`
- train IC / RankIC: 0.0341 / 0.0394
- test RankIC: 0.0054

```
- safe_div(numerator=2 * close - high - low, denominator=high - low) * safe_div(numerator=ts_mean(high - low, 5), denominator=ts_mean(close, 5))
```

Intraday overreaction detection using the relative close position within the daily range, scaled by recent average range. Extreme positions (close near high/low) with wider ranges signal exhaustion and subsequent reversal.

---

## factor_volume_weighted_price_change

- week: `week2` · fitness: `elite` · id: `raw_ca_hash_83b7560a`
- train IC / RankIC: 0.0244 / 0.0391
- test RankIC: N/A

```
EMA5( (close / delay(close, 1) - 1) * (volume / SMA20(volume)) )
```

Volume-weighted price change captures the coherence between price movement and volume. When price moves in the same direction as volume concentration, the signal is amplified. EMA smoothing reduces turnover and improves stability.

---

## factor_volume_adjusted_momentum

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_32d50b39`
- train IC / RankIC: 0.0240 / 0.0385
- test RankIC: N/A

```
ema(daily_return) * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)), 5)
```

Volume-adjusted momentum: multiplies daily price return by the ratio of current volume to its 20-day rolling average. High volume confirms price movement, low volume attenuates it. An EWMA(5) smoothing stabilizes the signal, capturing price-volume coherence in short-term momentum.

---

## factor_directional_vol_smooth_range_mut

- week: `week2` · fitness: `elite` · id: `raw_ca_hash_338dd16c`
- train IC / RankIC: 0.0246 / 0.0385
- test RankIC: N/A

```
smooth_range = ewma((high-low)/close, span=8); vol_ratio = volume / rolling_mean(volume,25); fast=ewma(vol_ratio,10); slow=ewma(vol_ratio,30); weight = sigmoid(5*(vol_ratio-0.85)); blend=weight*fast+(1-weight)*slow; direction=(close-open)/close; factor = smooth_range * blend * direction
```

Mutation of parent alpha-crossover-gen10-001 by increasing the EWMA span for the range component from 5 to 8. Preserves the core directional volume-adjusted volatility mechanism with adaptive sigmoid blending, targeting improved IC stability through smoother range estimation. The parent's strong rank-ordering (RankIC, RankICIR) is maintained by keeping the volume adjustment and blending unchanged.

---

## factor_vol_gated_momentum_v3

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_0f53ef7d`
- train IC / RankIC: 0.0216 / 0.0377
- test RankIC: 0.0298

```
ts_pct(close, 5) * (1.5 - 0.5 * ts_rank(ts_std(daily_return, 20), 20))
```

Refined volatility-gated momentum using inverse volatility weighting: short-term return (5-day) is weighted by a function that gives higher weight to low-volatility periods. Volatility is measured as the 20-day rolling standard deviation of daily returns. The weight is 1.5 - 0.5 * time-series rank of volatility percentile, so low volatility receives weight near 1.0 and high volatility near 0.5. This targets improved IC and ICIR by reducing the impact of noisy high-volatility regimes while preser

---

## factor_directional_vol_adaptive_crossover

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_2fc67cd0`
- train IC / RankIC: 0.0242 / 0.0375
- test RankIC: N/A

```
range_pct = (high-low)/close; smooth_range = ema(range_pct, span=5); vol_ratio = volume / ts_mean(volume,20); smooth_vol_ratio = if vol_ratio > 0.9 then ema(vol_ratio,10) else ema(vol_ratio,30); direction = (close-open)/close; factor = smooth_range * smooth_vol_ratio * direction
```

Crossover of primary (alpha_mutation_gen6_002) and secondary (alpha-mutation-gen6-002). Preserves primary's core directional volume-adjusted volatility mechanism with smooth_range (EWMA span=5) and continuous direction. Borrows the faster EWMA spans (10/30) for the volume ratio from the secondary parent to make the volume regime switching more distinctive, targeting improved IC while maintaining rank-ordering strength.

---

## factor_vol_gated_momentum_volume_weighted

- week: `week1` · fitness: `elite` · id: `raw_ca_hash_61b8fb3c`
- train IC / RankIC: 0.0233 / 0.0365
- test RankIC: 0.0321

```
ts_pct(close, 5) * (0.5 + 0.5 * ts_rank(safe_div(numerator=ts_mean(true_range, 14), denominator=close), 20)) * safe_div(numerator=volume, denominator=ts_mean(volume, 20))
```

Combines volatility-gated momentum from vol_gated_momentum_mut_1 with volume conviction from alpha-liquidity-v1-002-mut-001. The primary mechanism is continuous volatility weighting of short-term return; the lightweight modifier is relative volume to enhance signal in high-volume regimes. This crossover targets improving IC and ICIR while preserving the strong RankICIR and MI of the volatility parent.

---

## factor_volume_confirmed_momentum_clipped

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_50cd89c1`
- train IC / RankIC: 0.0249 / 0.0362
- test RankIC: 0.0298

```
(ts_pct(close, 5)) * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)))
```

Volume-confirmed momentum with clipped extremes to improve robustness. The core mechanism (momentum scaled by volume ratio) is preserved from parent1, while the clipping from parent2 reduces outlier influence, targeting better ICIR stability.

---

## factor_vol_regime_breakout_continuous

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_c83fcf6c`
- train IC / RankIC: 0.0309 / 0.0354
- test RankIC: 0.0093

```
ema((safe_div(numerator=close - low, denominator=high - low) - 0.5) * safe_div(numerator=high - low, denominator=ts_quantile(high - low, 20, 0.5)), 5)
```

Preserves the parent's core idea of intraday breakout strength conditioned on volatility regime, but replaces the binary volatility gate with a continuous scaling factor (range / 20-day median). This avoids discontinuities and captures more nuanced volatility information, targeting improved MI while maintaining the ICIR and RankICIR strengths of the parent.

---

## factor_vol_regime_trend_smoothed

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_7c569498`
- train IC / RankIC: 0.0168 / 0.0349
- test RankIC: 0.0441

```
ema(ts_pct(close, 5) * safe_div(numerator=volume, denominator=ts_max(volume, 20)) * (safe_div(numerator=high - low, denominator=ts_quantile(high - low, 20, 0.5)) - 1), 5)
```

Preserves the core regime-dependent trend-volume interaction from parent. Replaces binary high_vol indicator with continuous volatility deviation to retain more information, and applies a short EMA smoothing to reduce noise and improve rank stability.

---

## factor_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15_stressclip25

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_c6320c09`
- train IC / RankIC: 0.0246 / 0.0345
- test RankIC: N/A

```
((((df.copy())["high"]) - ((df.copy())["low"])) - (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).mean()) / (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).std(ddof=1).clipema(lower=0, 15) * ((df.copy())["volume"]) / ((df.copy())["((df.copy())["volume"])"]).rolling(20, min_periods=20).mean() * (((df.copy())["close"]) - ((df.copy())["open"])) / (((df.copy())["high"]) - ((df.copy())["low"]))
```

Crossover: primary parent (elite, mutation_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15_001) provides core stress accumulation, volume weight, and directional clip at 3. Secondary parent (qualified, mutation_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15_dirclip25_001) contributes the concept of tighter control (clip value 2.5), applied here to the stress component instead of directional strength, to limit extreme stress influence while preserving the strong RankIC and RankICIR

---

## factor_stress_volume_cont_weight_cont_dir_w20_clip2_5_w40_ewm15_stressclip25

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_c4485ee6`
- train IC / RankIC: 0.0246 / 0.0345
- test RankIC: N/A

```
((((df.copy())["high"]) - ((df.copy())["low"])) - (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).mean()) / (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).std(ddof=1).clipema(lower=0, 15) * ((df.copy())["volume"]) / ((df.copy())["((df.copy())["volume"])"]).rolling(20, min_periods=20).mean() * (((df.copy())["close"]) - ((df.copy())["open"])) / (((df.copy())["high"]) - ((df.copy())["low"]))
```

Mutation of crossover child: preserve core stress accumulation, continuous volume weight, and ewm-smoothed stress with upper clip 2.5. Reduce directional strength clipping from ±3 to ±2.5 to mitigate extreme directional bets, targeting improved IC while maintaining the strong RankIC and RankICIR. No new components introduced.

---

## factor_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15

- week: `week2` · fitness: `elite` · id: `raw_ca_hash_c7206013`
- train IC / RankIC: 0.0246 / 0.0345
- test RankIC: N/A

```
((((df.copy())["high"]) - ((df.copy())["low"])) - (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).mean()) / (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).std(ddof=1).clipema(lower=0, 15) * ((df.copy())["volume"]) / ((df.copy())["((df.copy())["volume"])"]).rolling(20, min_periods=20).mean() * (((df.copy())["close"]) - ((df.copy())["open"])) / (((df.copy())["high"]) - ((df.copy())["low"]))
```

Mutation of parent mutation_stress_volume_cont_weight_cont_dir_w20_clip3_w40_001: preserve core stress accumulation, continuous volume weight, and clipped directional strength. Increase the ewm smoothing span from 10 to 15 to further smooth the stress component, reducing noise and targeting improved IC and ICIR while maintaining the strong RankIC and RankICIR. No new components introduced.

---

## factor_stress_volume_cont_weight_cont_dir_w25_clip2_w40_ewm15

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_a96b0966`
- train IC / RankIC: 0.0245 / 0.0344
- test RankIC: N/A

```
((((df.copy())["high"]) - ((df.copy())["low"])) - (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).mean()) / (((df.copy())["((df.copy())["high"])"]) - ((df.copy())["((df.copy())["low"])"])).rolling(40, min_periods=40).std(ddof=1).clipema(lower=0, 15) * ((df.copy())["volume"]) / ((df.copy())["((df.copy())["volume"])"]).rolling(25, min_periods=25).mean() * (((df.copy())["close"]) - ((df.copy())["open"])) / (((df.copy())["high"]) - ((df.copy())["low"]))
```

Mutation of qualified parent: preserve core stress accumulation (40-day range z-score, ewm smoothed span 15), continuous volume weight (volume/25-day mean, cap 2), and directional strength (close-open)/range. Adjust directional clip from ±3 to ±2 to reduce extreme directional bets, targeting improved IC and ICIR while maintaining the strong RankIC. No new components introduced.

---

## factor_stress_range_volume_surge

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_f2c0364c`
- train IC / RankIC: 0.0156 / 0.0334
- test RankIC: 0.0487

```
safe_div(numerator=high - low, denominator=close) * safe_div(numerator=volume, denominator=ts_mean(volume, 20))
```

Mutation of factor_continuous_stress_range: removed the tanh modulation to simplify the interaction between stress range and volume surge. The core economic mechanism of tail-risk expansion through intraday volatility (high-low) amplified by abnormal volume is preserved. The ablation targets improved RankIC stability by reducing non-linear distortion from tanh, while maintaining the core multiplicative structure.

---

## factor_pressure_tanh

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_b9907c26`
- train IC / RankIC: 0.0160 / 0.0331
- test RankIC: 0.0258

```
sigmoid(2 * safe_div(numerator=safe_div(numerator=close - low, denominator=high - low) * volume - ts_mean(safe_div(numerator=close - low, denominator=high - low) * volume, 20), denominator=ts_std(safe_div(numerator=close - low, denominator=high - low) * volume, 20))) * 2 - 1
```

Mutation of pressure_zscore: replace signed sqrt with tanh to bound extremes and improve nonlinear stability while preserving volume-weighted intraday pressure core. Preserves parent strength (RankICIR) and targets MI improvement through bounded nonlinearity.

---

## factor_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_bc141edf`
- train IC / RankIC: 0.0138 / 0.0328
- test RankIC: N/A

```
(-1.0) * (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) > ((df.copy())['open'])) == 1, np.nan).rolling(20, min_periods=3).median() / (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) <= ((df.copy())['open'])) == 1, np.nan).rolling(20, min_periods=3).median() * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)))
```

Mutation of mut_cro_asym_vol_ratio_med_20_minp3_vol6_001: preserve core asymmetry ratio and volume ratio mechanism, but increase EMA smoothing span from 3 to 5 to reduce noise and potentially improve IC while maintaining rank stability. Targeting IC improvement without sacrificing RankICIR.

---

## factor_herding_simple

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_718ea675`
- train IC / RankIC: 0.0172 / 0.0327
- test RankIC: 0.0212

```
ema(safe_div(numerator=close - open, denominator=high - low) * safe_div(numerator=volume, denominator=ts_quantile(volume, 10, 0.5)), 5)
```

Mutation of crossover_herding_vol_stress_001: removed the stress_range component (high-low)/close to simplify the factor and reduce noise. The parent's strength in RankIC and ICIR is partly from the herding direction and volume confirmation, and the stress_range may have introduced instability. By ablating this modifier, we aim to improve IC and MI while preserving the core herding mechanism and its strong RankIC profile.

---

## factor_adaptive_momentum_gate_v2

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_946bc42a`
- train IC / RankIC: 0.0173 / 0.0325
- test RankIC: 0.0281

```
safe_div(numerator=ts_pct(close, 5), denominator=ts_std(daily_return, 30))
```

Mutation of alpha_regime_gating_1: shortened momentum lookback from 10 to 5 days to capture higher-frequency trends, preserving volatility-gated momentum intuition with long (30-day) volatility estimate for stability. Targets improved IC and MI while maintaining RankIC.

---

## factor_volume_confirmed_momentum_binary_gate

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_777e86d4`
- train IC / RankIC: 0.0290 / 0.0320
- test RankIC: 0.0060

```
(ema(daily_return * safe_div(numerator=volume, denominator=ts_mean(volume, 20)), 10)) * (sign(close - (high + low) / 2))
```

Volume-confirmed momentum with a binary intraday gate. The continuous intraday position from the parent is replaced by a binary indicator (1 if close is above midpoint, -1 if below) to reduce noise and improve MI and RankIC stability while preserving the core economic mechanism of volume-confirmed momentum gated by intraday price position.

---

## factor_intraday_reversal_crossover_v1

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_881db7aa`
- train IC / RankIC: 0.0305 / 0.0320
- test RankIC: 0.0055

```
ts_mean(- safe_div(numerator=2 * close - high - low, denominator=high - low) * safe_div(numerator=ts_mean(high - low, 5), denominator=ts_mean(close, 5)), 3)
```

Crossover preserving the elite intraday reversal mechanism from parent1 (factor_intraday_reversal_repair_0) and borrowing the lightweight 3-period SMA smoothing from parent2 (factor_intraday_reversal_smoothed_v2_mutation) to reduce noise and improve rank stability while maintaining the core reversal signal.

---

## factor_range_position_volume_gated_smoothed

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_05d8d226`
- train IC / RankIC: 0.0302 / 0.0315
- test RankIC: 0.0067

```
ema((safe_div(numerator=close - low, denominator=high - low)) - 0.5, 3) * safe_div(numerator=volume, denominator=ts_mean(volume, 20))
```

Crossover: primary mechanism from AR_001_m1 (range position with EWM smoothing) and lightweight modifier from VGR-001-MUT-001 (volume ratio) applied after smoothing. This combines the clean smoothing of parent1 with volume gating of parent2, aiming to preserve high RankIC and IC while potentially improving MI through the order-of-operations change.

---

## factor_mut_cro_asym_vol_ratio_med_15_minp3_vol10

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_bcbd6919`
- train IC / RankIC: 0.0150 / 0.0311
- test RankIC: N/A

```
(-1.0) * (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) > ((df.copy())['open'])) == 1, np.nan).rolling(15, min_periods=3).median() / (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) <= ((df.copy())['open'])) == 1, np.nan).rolling(15, min_periods=3).median() * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)))
```

Parent strength: high RankICIR (~0.3795) from robust asymmetry and volume ratio mechanism. Bottleneck: low IC (0.0138). Mutation: increase volume ratio median window from 6 to 10 to obtain a more stable volume ratio, aiming to reduce noise and improve IC while preserving rank stability. All other parameters unchanged.

---

## factor_imbalance_volume_ratio_ewm

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_263e9c66`
- train IC / RankIC: 0.0285 / 0.0310
- test RankIC: 0.0053

```
ema(safe_div(numerator=close - (high + low) / 2, denominator=high - low) * safe_div(numerator=volume, denominator=ts_mean(volume, 20)), 5)
```

Preserve the core imbalance and volume confirmation mechanism. Simplify the volume adjustment by removing the absolute volume multiplication, using only the volume ratio (volume / 20-day average). This reduces the influence of large volume outliers that may cause nonlinearity, targeting an improvement in IC while maintaining rank stability from the parent. The parent strength in RankIC (0.0308) and RankICIR (0.3339) is preserved.

---

## factor_range_position_volume_gated_smoothed_mut

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_d8e639c3`
- train IC / RankIC: 0.0306 / 0.0309
- test RankIC: 0.0067

```
ema((safe_div(numerator=close - low, denominator=high - low)) - 0.5, 3) * safe_div(numerator=volume, denominator=ts_mean(volume, 20))
```

Mutation of CROSS_VGR_AR_001: increase EWM smoothing span from 3 to 5 to further reduce noise and improve rank stability (RankIC, RankICIR) while preserving the core volume-gated range position reversal mechanism. This targets the bottleneck of moderate RankICIR by smoothing more aggressively.

---

## factor_imbalance_volume_ewm_mut

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_15a03771`
- train IC / RankIC: 0.0142 / 0.0308
- test RankIC: 0.0060

```
ema(safe_div(numerator=close - (high + low) / 2, denominator=high - low) * volume * (1 + safe_div(numerator=volume, denominator=ts_mean(volume, 10)) - 1), 5)
```

Preserves the core imbalance-volume interaction from the parent. Shortens the volume baseline from 20 to 10 days and replaces the 5-day SMA with an exponential moving average (span=5) to improve responsiveness and potentially boost IC while maintaining RankIC stability.

---

## factor_mut_cro_asym_vol_ratio_med_20_minp3_vol10_ema2

- week: `week2` · fitness: `qualified` · id: `raw_ca_hash_ae211225`
- train IC / RankIC: 0.0139 / 0.0304
- test RankIC: N/A

```
(-1.0) * (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) > ((df.copy())['open'])) == 1, np.nan).rolling(20, min_periods=3).median() / (((df.copy())['((df.copy())['((df.copy())['high'])'])']) - ((df.copy())['((df.copy())['((df.copy())['low'])'])'])).where((((df.copy())['((df.copy())['((df.copy())['((df.copy())['close'])'])'])']) <= ((df.copy())['open'])) == 1, np.nan).rolling(20, min_periods=3).median() * (safe_div(numerator=volume, denominator=ts_mean(volume, 20)))
```

Mutation of cro_asym_vol_ratio_med_20_minp3_vol6_ema2_001. Preserve core asymmetry ratio mechanism (window 20) and EMA2 smoothing from parent. Increase volume median window from 6 to 10 to obtain a smoother denominator for the volume ratio, reducing noise in the volume component. This targets improved IC by providing a more stable volume signal while preserving the high RankIC and RankICIR inherited from the parents. The change is a compact window adjustment.

---
