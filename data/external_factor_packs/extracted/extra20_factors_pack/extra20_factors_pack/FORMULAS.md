# Extra 20 公式清单

## 1. cogalpha_20260630074841_f29a08ef

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630074841_f29a08ef`
- train IC / RankIC: 0.0270 / 0.0515

```
EWM( pct_change(close) * rank_pct8(volume) * ((high-low)/close), span=12 ) clip ±3
```



---

## 2. cogalpha_20260630075856_c4558a55

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630075856_c4558a55`
- train IC / RankIC: 0.0238 / 0.0501

```
EMA12( pct_change * log1p(volume/SMA5(volume)) * (1 + clip(zscore10(range), 0, 2)) )
```



---

## 3. cogalpha_20260630073345_a6787278

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630073345_a6787278`
- train IC / RankIC: 0.0233 / 0.0489

```
EWM( (close - close.shift(1))/close.shift(1) * (volume / SMA(volume,5)) * ((high-low)/close) , span=5 )
```



---

## 4. cogalpha_20260630025411_56a5dd19

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630025411_56a5dd19`
- train IC / RankIC: 0.0362 / 0.0489

```
tanh(ema( (close/shift(close,1)-1) * (volume/sma(volume,20)), 10 ) * ((close-low)/(high-low)))
```



---

## 5. cogalpha_20260630092749_cc36ec98

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630092749_cc36ec98`
- train IC / RankIC: 0.0249 / 0.0474

```
ewm(halflife=7, min_periods=7).mean((close/open-1) * clipped(volume/rolling_median(volume,10), 0.3, 2.0))
```



---

## 6. cogalpha_20260630090734_2c371ded

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630090734_2c371ded`
- train IC / RankIC: 0.0229 / 0.0469

```
smoothed(price_change * (volume / SMA(volume,20)) * (1 + zscore(range,10).clip(-2,2)), 10)
```



---

## 7. cogalpha_20260630093800_9903647e

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630093800_9903647e`
- train IC / RankIC: 0.0245 / 0.0467

```
0.95 * ewm(halflife=7, min_periods=7).mean((close/open-1) * clipped(volume/rolling_median(volume,10), 0.3, 3.0)) + 0.05 * rolling(10, min_periods=10).mean((close/open-1) * clipped(volume/rolling_median(volume,10), 0.3, 3.0))
```



---

## 8. cogalpha_20260630084303_9d606e59

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630084303_9d606e59`
- train IC / RankIC: 0.0295 / 0.0466

```
smooth_range = ewm((high-low)/close, span=10); vol_ratio = volume / rolling_mean(volume,25); fast=ewm(vol_ratio,10); slow=ewm(vol_ratio,30); weight = sigmoid(5*(vol_ratio-0.85)); smooth_vol = weight*fast + (1-weight)*slow; direction=(close-open)/close; smooth_direction=ewm(direction, span=3); factor = smooth_range * smooth_vol * smooth_direction
```



---

## 9. cogalpha_20260630080916_c4bd4f91

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630080916_c4bd4f91`
- train IC / RankIC: 0.0228 / 0.0466

```
EMA10( pct_change * (volume/SMA20(volume)) * (1 + clip(zscore10(range), 0, 2)) )
```



---

## 10. cogalpha_20260630083323_69492d5e

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630083323_69492d5e`
- train IC / RankIC: 0.0294 / 0.0463

```
range_pct = (high-low)/close; smooth_range = ewm(range_pct, span=10); vol_ratio = volume / rolling_mean(volume,25); smooth_vol = ewm(vol_ratio, span=15); direction = (close-open)/close; smooth_direction = ewm(direction, span=3); factor = smooth_range * smooth_vol * smooth_direction
```



---

## 11. cogalpha_20260630085240_b9415e04

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630085240_b9415e04`
- train IC / RankIC: 0.0294 / 0.0463

```
smooth_range = ewm((high-low)/close, span=10); vol_ratio = volume / rolling_mean(volume,25); smooth_vol = ewm(vol_ratio, span=15); direction=(close-open)/close; smooth_direction=ewm(direction, span=3); factor = smooth_range * smooth_vol * smooth_direction
```



---

## 12. cogalpha_20260702070916_4a2b9199

- campaign: `cogalpha_ashare_pv_202606300915_w04` · id: `cogalpha_20260702070916_4a2b9199`
- train IC / RankIC: 0.0294 / 0.0463

```
-EMA_5(intraday_return) * clip( std_20(close_ret) / std_60(close_ret), 0.5, 2.0 )
```



---

## 13. cogalpha_20260630091249_346a8977

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630091249_346a8977`
- train IC / RankIC: 0.0224 / 0.0462

```
smoothed(price_change * (volume / SMA(volume,20)) * (1 + zscore(range,10).clip(-2,2)), 8)
```



---

## 14. cogalpha_20260630021752_f66cda18

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630021752_f66cda18`
- train IC / RankIC: 0.0248 / 0.0460

```
EWM( (close/open - 1) * ( volume / rolling_max(volume,20) ), 20 )
```



---

## 15. cogalpha_20260630095214_ad397993

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630095214_ad397993`
- train IC / RankIC: 0.0255 / 0.0458

```
tanh(EWMA( (close/open-1) * (volume/SMA(volume,20)), span=20, adjust=False ))
```



---

## 16. cogalpha_20260630090227_5baad9b2

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630090227_5baad9b2`
- train IC / RankIC: 0.0225 / 0.0456

```
smoothed(price_change * (volume / SMA(volume,20)) * (1 + zscore(range,20)), 10)
```



---

## 17. cogalpha_20260630100633_a472187d

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630100633_a472187d`
- train IC / RankIC: 0.0261 / 0.0453

```
tanh(EWMA( (close/open-1) * (volume/SMA(volume,20)), span=10, adjust=false ))
```



---

## 18. cogalpha_20260702043307_1c1f15bf

- campaign: `cogalpha_ashare_pv_202606300915_w04` · id: `cogalpha_20260702043307_1c1f15bf`
- train IC / RankIC: 0.0252 / 0.0449

```
- ewm(intraday_ret, span=5)
```



---

## 19. cogalpha_20260630092237_0ad794d1

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630092237_0ad794d1`
- train IC / RankIC: 0.0222 / 0.0447

```
smoothed(ret * (volume / rolling_median(volume, 20)))
```



---

## 20. cogalpha_20260630024020_a436b80f

- campaign: `cogalpha_ashare_pv_202606301800` · id: `cogalpha_20260630024020_a436b80f`
- train IC / RankIC: 0.0230 / 0.0446

```
ema( (close / shift(close,1) - 1) * (volume / sma(volume,20)), 10 )
```



---
