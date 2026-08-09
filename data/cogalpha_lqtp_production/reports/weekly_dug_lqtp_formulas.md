# 本周新挖 / 扩展候选 · LQTP 公式

更新：2026-08-09T04:28:59.022380+00:00

## 本周入选（有 DSL）

### `alpha_20260702_afd40784`
- RankIC(本地面板): 0.04657378851014425
- 回测: 平台
- 平台 mean_ic=0.0520 sharpe=3.5707438871538737
- 源 DSL: `ts_corr(close, volume, 10) * (0 - 1) * ts_std(returns, 20)`
- LQTP: `ts_corr(close, volume, 10) * (0 - 1) * ts_std(ret, 20)`

### `alpha_20260702_16104691`
- RankIC(本地面板): 0.04493095267685021
- 回测: 平台
- 平台 mean_ic=0.0524 sharpe=1.6640367217268248
- 源 DSL: `rank((0 - 1) * ts_mean((high - low) / (close + 0.001), 10)) * rank(delay(close, 60) / (close + 0.001))`
- LQTP: `rank((0 - 1) * ts_mean((high - low) / (close + 0.001), 10)) * rank(delay(close, 60) / (close + 0.001))`

### `alpha_20260701_549b43e6`
- RankIC(本地面板): 0.04172905582793453
- 回测: 平台
- 平台 mean_ic=-0.0459 sharpe=-2.4917721345820456
- 源 DSL: `ts_rank(ts_delta(close, 5), 20) * ts_std(returns, 20)`
- LQTP: `ts_rank(ts_delta(close, 5), 20) * ts_std(ret, 20)`

### `alpha_20260703_9e3b2920`
- RankIC(本地面板): 0.04010747442006768
- 回测: 平台
- 平台 mean_ic=0.0460 sharpe=3.5158800040224754
- 源 DSL: `((0 - 1) * rank(ts_std(high, 10))) * ts_corr(high, volume, 10)`
- LQTP: `((0 - 1) * rank(ts_std(high, 10))) * ts_corr(high, volume, 10)`

### `alpha_20260703_99eb3d11`
- RankIC(本地面板): 0.04010747442006766
- 回测: 平台
- 平台 mean_ic=0.0460 sharpe=3.5158800040224754
- 源 DSL: `(0 - 1) * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)`
- LQTP: `(0 - 1) * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)`

### `alpha_20260702_68529a49`
- RankIC(本地面板): 0.03903759416010873
- 回测: 平台
- 平台 mean_ic=0.0438 sharpe=2.0600026729238365
- 源 DSL: `rank(ts_delta(close, 20)) * (0 - 1) * ts_std(returns, 20)`
- LQTP: `rank(ts_delta(close, 20)) * (0 - 1) * ts_std(ret, 20)`

### `alpha_20260703_c6169b2e`
- RankIC(本地面板): 0.03661320565454456
- 回测: 平台
- 平台 mean_ic=-0.0417 sharpe=-3.0276766334995138
- 源 DSL: `close / (low + 0.001)`
- LQTP: `close / (low + 0.001)`

### `alpha_20260703_6c174054`
- RankIC(本地面板): 0.03608381369193557
- 回测: 平台
- 平台 mean_ic=-0.0388 sharpe=1.0159805803247846
- 源 DSL: `(high - open) / (open + 0.001)`
- LQTP: `(high - open) / (open + 0.001)`

### `alpha_20260702_576e0a0b`
- RankIC(本地面板): 0.03299228649940185
- 回测: 平台
- 平台 mean_ic=-0.0366 sharpe=-1.7444274738216503
- 源 DSL: `(close - ts_mean(close, 24)) / (ts_mean(close, 24) + 0.001)`
- LQTP: `(close - ts_mean(close, 24)) / (ts_mean(close, 24) + 0.001)`

### `alpha_20260702_a1c128d5`
- RankIC(本地面板): 0.03266357065152101
- 回测: 平台
- 平台 mean_ic=0.0364 sharpe=3.3732231111878233
- 源 DSL: `rank((0 - 1) * ts_corr(close, volume, 20))`
- LQTP: `rank((0 - 1) * ts_corr(close, volume, 20))`

### `alpha_20260701_75f1066f`
- RankIC(本地面板): 0.03097235213204809
- 回测: 平台
- 平台 mean_ic=0.0336 sharpe=2.502838240511448
- 源 DSL: `rank(ts_delta(volume, 5)) * (0 - 1) * ts_std(returns, 10)`
- LQTP: `rank(ts_delta(volume, 5)) * (0 - 1) * ts_std(ret, 10)`

### `alpha_20260702_844ceabc`
- RankIC(本地面板): 0.030868651502659567
- 回测: 平台
- 平台 mean_ic=-0.0279 sharpe=-1.0068605532287147
- 源 DSL: `ts_sum(ts_max(ts_delta(close, 1), 0), 60) / (ts_sum(ts_max((0 - 1) * ts_delta(close, 1), 0), 60) + 0.001)`
- LQTP: `ts_sum(where(ts_delta(close, 1) > 0, ts_delta(close, 1), 0), 60) / (ts_sum(where((0 - 1) * ts_delta(close, 1) > 0, (0 - 1) * ts_delta(close, 1), 0), 60) + 0.001)`

### `alpha_20260702_6d5f848e`
- RankIC(本地面板): 0.03040189045794825
- 回测: 平台
- 平台 mean_ic=-0.0319 sharpe=-1.749155677654132
- 源 DSL: `rank((close - delay(close, 5)) / (delay(close, 5) + 0.001)) * rank(ts_corr(close, volume, 5))`
- LQTP: `rank((close - delay(close, 5)) / (delay(close, 5) + 0.001)) * rank(ts_corr(close, volume, 5))`

### `alpha_20260701_726efa4d`
- RankIC(本地面板): 0.029622792258442603
- 回测: 平台
- 平台 mean_ic=-0.0317 sharpe=-1.4835872296229917
- 源 DSL: `(close - ts_mean(close, 20)) / (ts_std(close, 20) + 0.001)`
- LQTP: `(close - ts_mean(close, 20)) / (ts_std(close, 20) + 0.001)`

### `alpha_20260701_5685f3e9`
- RankIC(本地面板): 0.02914426153339675
- 回测: 平台
- 平台 mean_ic=0.0310 sharpe=1.261141884033367
- 源 DSL: `rank((0 - 1) * ts_std(returns, 10)) * rank(ts_sum(returns, 6))`
- LQTP: `rank((0 - 1) * ts_std(ret, 10)) * rank(ts_sum(ret, 6))`

### `alpha_20260701_5f690b22`
- RankIC(本地面板): 0.02898057152141599
- 回测: 平台
- 平台 mean_ic=-0.0321 sharpe=-1.8141557856553425
- 源 DSL: `ts_delta(close, 10) / (delay(close, 10) + 0.001)`
- LQTP: `ts_delta(close, 10) / (delay(close, 10) + 0.001)`

### `alpha_20260702_89d0956b`
- RankIC(本地面板): 0.02755906769123201
- 回测: 平台
- 平台 mean_ic=0.0343 sharpe=1.5730834945183731
- 源 DSL: `ts_mean(close, 60) / (delay(close, 1) + 0.001)`
- LQTP: `ts_mean(close, 60) / (delay(close, 1) + 0.001)`

### `alpha_20260701_2f5c25fe`
- RankIC(本地面板): 0.027213954690921364
- 回测: 平台
- 平台 mean_ic=0.0287 sharpe=0.8865669415769459
- 源 DSL: `rank((0 - 1) * ts_std(returns, 20)) * ts_rank(ts_delta(close, 5), 20)`
- LQTP: `rank((0 - 1) * ts_std(ret, 20)) * ts_rank(ts_delta(close, 5), 20)`

### `alpha_20260701_abf0beb7`
- RankIC(本地面板): 0.02696511653096264
- 回测: 平台
- 平台 mean_ic=-0.0291 sharpe=-1.7005236825709713
- 源 DSL: `ts_delta(close, 5) / (delay(close, 5) + 0.001)`
- LQTP: `ts_delta(close, 5) / (delay(close, 5) + 0.001)`

### `alpha_20260701_d332134b`
- RankIC(本地面板): 0.02587686426836177
- 回测: 平台
- 平台 mean_ic=-0.0287 sharpe=-1.0241206089428594
- 源 DSL: `ts_mean(ts_delta(close, 1), 20) / (ts_std(ts_delta(close, 1), 20) + 0.001)`
- LQTP: `ts_mean(ts_delta(close, 1), 20) / (ts_std(ts_delta(close, 1), 20) + 0.001)`

### `alpha_20260702_21fb502a`
- RankIC(本地面板): 0.024158827627327923
- 回测: 平台
- 平台 mean_ic=-0.0256 sharpe=-1.2167000096262945
- 源 DSL: `(close - ts_mean(close, 6)) / (ts_mean(close, 6) + 0.001)`
- LQTP: `(close - ts_mean(close, 6)) / (ts_mean(close, 6) + 0.001)`

### `alpha_20260702_98ed259d`
- RankIC(本地面板): 0.021326508780514383
- 回测: 平台
- 平台 mean_ic=-0.0242 sharpe=-3.182255363255693
- 源 DSL: `rank(ts_corr(open, volume, 10)) * ts_rank(ts_delta(close, 5), 10)`
- LQTP: `rank(ts_corr(open, volume, 10)) * ts_rank(ts_delta(close, 5), 10)`

### `alpha_20260701_75057f17`
- RankIC(本地面板): 0.021292954130173746
- 回测: 平台
- 平台 mean_ic=0.0240 sharpe=1.92659901614966
- 源 DSL: `ts_std(returns, 60) * (0 - 1) * sign(ts_delta(close, 5))`
- LQTP: `ts_std(ret, 60) * (0 - 1) * sign(ts_delta(close, 5))`

### `alpha_20260702_f204de74`
- RankIC(本地面板): 0.02062019882934937
- 回测: 平台
- 平台 mean_ic=-0.0257 sharpe=-0.08937913936469842
- 源 DSL: `(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 0.001)`
- LQTP: `(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 0.001)`

### `alpha_20260702_2e48ae98`
- RankIC(本地面板): 0.020227308737052415
- 回测: 平台
- 平台 mean_ic=-0.0215 sharpe=-2.0521046266480183
- 源 DSL: `(close - open) / (open + 0.001)`
- LQTP: `(close - open) / (open + 0.001)`

## 扩展候选（平台去重后）

### `alpha_20260702_0430ef8f`
- pack_rank_ic=0.3797704589684442 platform_ic=-0.016153559934530493 sharpe=-2.6301278991892247
- LQTP: `(close - vwap) / (vwap + 0.001)`

### `alpha_20260701_ef61652c`
- pack_rank_ic=0.3604250078161817 platform_ic=-0.014727296912346961 sharpe=-2.2093948594873374
- LQTP: `(2 * close - high - low) / (open + 0.001)`

### `alpha_20260701_a7140261`
- pack_rank_ic=0.30208035277069545 platform_ic=0.020500357214315364 sharpe=-0.4250973543341091
- LQTP: `close / (high + 0.001)`

### `alpha_20260702_970118c9`
- pack_rank_ic=0.24072022947953614 platform_ic=-0.018149860501005262 sharpe=-1.7651077475477392
- LQTP: `(close - open) / ((high - low) + 0.001)`

### `alpha_20260702_45d2ecdb`
- pack_rank_ic=0.1747762683464782 platform_ic=-0.006101818401420559 sharpe=-1.0322210826125089
- LQTP: `rank(ts_delta(log(volume), 2)) * rank(ts_delta((close - open) / (open + 0.001), 6))`

### `alpha_20260702_72b9ca02`
- pack_rank_ic=0.11540709550992412 platform_ic=-0.01851988953281646 sharpe=22.452348163351424
- LQTP: `ts_rank(ts_delta(close, 5), 10)`

### `alpha_20260701_f5d88d1a`
- pack_rank_ic=0.10663348861325994 platform_ic=-0.012045821554823328 sharpe=-0.4714182495526786
- LQTP: `rank((close - open) / (open + 0.001)) * rank((ts_max(high, 5) - close) / (close + 0.001))`

### `alpha_20260702_87e60a49`
- pack_rank_ic=0.09703101105714418 platform_ic=-0.020278085441296237 sharpe=-0.1950006035703034
- LQTP: `(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001)`

### `alpha_20260702_cd182da7`
- pack_rank_ic=0.09703101105714418 platform_ic=-0.014771822377082838 sharpe=-0.1385966131474179
- LQTP: `(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 0.001)`

### `alpha_20260702_cd00ad0b`
- pack_rank_ic=0.09691147532342681 platform_ic=-0.018507774300581658 sharpe=-0.3179612048146237
- LQTP: `rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 0.001))`

### `alpha_20260702_8164b3d1`
- pack_rank_ic=0.0816529489936624 platform_ic=-0.01927833044842909 sharpe=10.71357883585848
- LQTP: `ts_rank(ts_delta(close, 5), 20)`

### `alpha_20260702_d4af1618`
- pack_rank_ic=0.07967151006647083 platform_ic=-0.019439584499157517 sharpe=10.168358156017272
- LQTP: `ts_rank(ts_delta(close, 5), 21)`

### `evoalpha_20260701141121_36be5f53`
- pack_rank_ic=0.075332 platform_ic=0.041757694946578924 sharpe=3.1660307022969074
- LQTP: `low / (close + 0.001)`

### `evoalpha_20260701141121_1aecc16d`
- pack_rank_ic=0.072387 platform_ic=0.04063727910049005 sharpe=1.6017485977253052
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
