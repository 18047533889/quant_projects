# 本周新挖 / 外部 DSL → 平台公式清单

- 挑选 80（周选命中 25 + 额外 55）
- 可提交平台 80

## `alpha_20260701_2f5c25fe`
- status: `ready` native=True weekly=True
- pack RankIC: 0.08045801510740583
- DSL: `rank((0 - 1) * ts_std(returns, 20)) * ts_rank(ts_delta(close, 5), 20)`
- LQTP: `rank((0 - 1) * ts_std(returns, 20)) * ts_rank(ts_delta(close, 5), 20)`

## `alpha_20260701_549b43e6`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03576506520943361
- DSL: `ts_rank(ts_delta(close, 5), 20) * ts_std(returns, 20)`
- LQTP: `ts_rank(ts_delta(close, 5), 20) * ts_std(returns, 20)`

## `alpha_20260701_5685f3e9`
- status: `ready` native=True weekly=True
- pack RankIC: 0.05355876874889347
- DSL: `rank((0 - 1) * ts_std(returns, 10)) * rank(ts_sum(returns, 6))`
- LQTP: `rank((0 - 1) * ts_std(returns, 10)) * rank(ts_sum(returns, 6))`

## `alpha_20260701_5f690b22`
- status: `ready` native=True weekly=True
- pack RankIC: 0.046235174623399176
- DSL: `ts_delta(close, 10) / (delay(close, 10) + 0.001)`
- LQTP: `ts_delta(close, 10) / (delay(close, 10) + 0.001)`

## `alpha_20260701_726efa4d`
- status: `ready` native=True weekly=True
- pack RankIC: 0.07507786771680396
- DSL: `(close - ts_mean(close, 20)) / (ts_std(close, 20) + 0.001)`
- LQTP: `(close - ts_mean(close, 20)) / (ts_std(close, 20) + 0.001)`

## `alpha_20260701_75057f17`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03871405493419674
- DSL: `ts_std(returns, 60) * (0 - 1) * sign(ts_delta(close, 5))`
- LQTP: `ts_std(returns, 60) * (0 - 1) * sign(ts_delta(close, 5))`

## `alpha_20260701_75f1066f`
- status: `ready` native=True weekly=True
- pack RankIC: 0.04153037582694815
- DSL: `rank(ts_delta(volume, 5)) * (0 - 1) * ts_std(returns, 10)`
- LQTP: `rank(ts_delta(volume, 5)) * (0 - 1) * ts_std(returns, 10)`

## `alpha_20260701_abf0beb7`
- status: `ready` native=True weekly=True
- pack RankIC: 0.08686451043844302
- DSL: `ts_delta(close, 5) / (delay(close, 5) + 0.001)`
- LQTP: `ts_delta(close, 5) / (delay(close, 5) + 0.001)`

## `alpha_20260701_d332134b`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03747392248938837
- DSL: `ts_mean(ts_delta(close, 1), 20) / (ts_std(ts_delta(close, 1), 20) + 0.001)`
- LQTP: `ts_mean(ts_delta(close, 1), 20) / (ts_std(ts_delta(close, 1), 20) + 0.001)`

## `alpha_20260702_16104691`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03916407026581727
- DSL: `rank((0 - 1) * ts_mean((high - low) / (close + 0.001), 10)) * rank(delay(close, 60) / (close + 0.001))`
- LQTP: `rank((0 - 1) * ts_mean((high - low) / (close + 0.001), 10)) * rank(delay(close, 60) / (close + 0.001))`

## `alpha_20260702_21fb502a`
- status: `ready` native=True weekly=True
- pack RankIC: 0.14921601530619974
- DSL: `(close - ts_mean(close, 6)) / (ts_mean(close, 6) + 0.001)`
- LQTP: `(close - ts_mean(close, 6)) / (ts_mean(close, 6) + 0.001)`

## `alpha_20260702_2e48ae98`
- status: `ready` native=True weekly=True
- pack RankIC: 0.2666002874498402
- DSL: `(close - open) / (open + 0.001)`
- LQTP: `(close - open) / (open + 0.001)`

## `alpha_20260702_576e0a0b`
- status: `ready` native=True weekly=True
- pack RankIC: 0.04842383996632039
- DSL: `(close - ts_mean(close, 24)) / (ts_mean(close, 24) + 0.001)`
- LQTP: `(close - ts_mean(close, 24)) / (ts_mean(close, 24) + 0.001)`

## `alpha_20260702_68529a49`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03337160082309406
- DSL: `rank(ts_delta(close, 20)) * (0 - 1) * ts_std(returns, 20)`
- LQTP: `rank(ts_delta(close, 20)) * (0 - 1) * ts_std(returns, 20)`

## `alpha_20260702_6d5f848e`
- status: `ready` native=True weekly=True
- pack RankIC: 0.05547749736240848
- DSL: `rank((close - delay(close, 5)) / (delay(close, 5) + 0.001)) * rank(ts_corr(close, volume, 5))`
- LQTP: `rank((close - delay(close, 5)) / (delay(close, 5) + 0.001)) * rank(ts_corr(close, volume, 5))`

## `alpha_20260702_844ceabc`
- status: `ready` native=True weekly=True
- pack RankIC: 0.033601188617677136
- DSL: `ts_sum(ts_max(ts_delta(close, 1), 0), 60) / (ts_sum(ts_max((0 - 1) * ts_delta(close, 1), 0), 60) + 0.001)`
- LQTP: `ts_sum(ts_max(ts_delta(close, 1), 0), 60) / (ts_sum(ts_max((0 - 1) * ts_delta(close, 1), 0), 60) + 0.001)`

## `alpha_20260702_89d0956b`
- status: `ready` native=True weekly=True
- pack RankIC: 0.04776545585869006
- DSL: `ts_mean(close, 60) / (delay(close, 1) + 0.001)`
- LQTP: `ts_mean(close, 60) / (delay(close, 1) + 0.001)`

## `alpha_20260702_98ed259d`
- status: `ready` native=True weekly=True
- pack RankIC: 0.039251086004827594
- DSL: `rank(ts_corr(open, volume, 10)) * ts_rank(ts_delta(close, 5), 10)`
- LQTP: `rank(ts_corr(open, volume, 10)) * ts_rank(ts_delta(close, 5), 10)`

## `alpha_20260702_a1c128d5`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03278313413982634
- DSL: `rank((0 - 1) * ts_corr(close, volume, 20))`
- LQTP: `rank((0 - 1) * ts_corr(close, volume, 20))`

## `alpha_20260702_afd40784`
- status: `ready` native=True weekly=True
- pack RankIC: 0.03995190796493969
- DSL: `ts_corr(close, volume, 10) * (0 - 1) * ts_std(returns, 20)`
- LQTP: `ts_corr(close, volume, 10) * (0 - 1) * ts_std(returns, 20)`

## `alpha_20260702_f204de74`
- status: `ready` native=True weekly=True
- pack RankIC: 0.09703101105714418
- DSL: `(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 0.001)`
- LQTP: `(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 0.001)`

## `alpha_20260703_6c174054`
- status: `ready` native=True weekly=True
- pack RankIC: 0.07040151242856924
- DSL: `(high - open) / (open + 0.001)`
- LQTP: `(high - open) / (open + 0.001)`

## `alpha_20260703_99eb3d11`
- status: `ready` native=True weekly=True
- pack RankIC: 0.04534810729409932
- DSL: `(0 - 1) * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)`
- LQTP: `(0 - 1) * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)`

## `alpha_20260703_9e3b2920`
- status: `ready` native=True weekly=True
- pack RankIC: 0.04534810729409932
- DSL: `((0 - 1) * rank(ts_std(high, 10))) * ts_corr(high, volume, 10)`
- LQTP: `((0 - 1) * rank(ts_std(high, 10))) * ts_corr(high, volume, 10)`

## `alpha_20260703_c6169b2e`
- status: `ready` native=True weekly=True
- pack RankIC: 0.23339166771830339
- DSL: `close / (low + 0.001)`
- LQTP: `close / (low + 0.001)`

## `alpha_20260702_0430ef8f`
- status: `ready` native=True weekly=False
- pack RankIC: 0.3797704589684442
- DSL: `(close - vwap) / (vwap + 0.001)`
- LQTP: `(close - vwap) / (vwap + 0.001)`

## `alpha_20260701_ef61652c`
- status: `ready` native=True weekly=False
- pack RankIC: 0.3604250078161817
- DSL: `(2 * close - high - low) / (open + 0.001)`
- LQTP: `(2 * close - high - low) / (open + 0.001)`

## `alpha_20260701_a7140261`
- status: `ready` native=True weekly=False
- pack RankIC: 0.30208035277069545
- DSL: `close / (high + 0.001)`
- LQTP: `close / (high + 0.001)`

## `alpha_20260702_970118c9`
- status: `ready` native=True weekly=False
- pack RankIC: 0.24072022947953614
- DSL: `(close - open) / ((high - low) + 0.001)`
- LQTP: `(close - open) / ((high - low) + 0.001)`

## `alpha_20260702_45d2ecdb`
- status: `ready` native=True weekly=False
- pack RankIC: 0.1747762683464782
- DSL: `rank(ts_delta(log(volume), 2)) * rank(ts_delta((close - open) / (open + 0.001), 6))`
- LQTP: `rank(ts_delta(logvolume, 2)) * rank(ts_delta((close - open) / (open + 0.001), 6))`

## `alpha_20260702_72b9ca02`
- status: `ready` native=True weekly=False
- pack RankIC: 0.11540709550992412
- DSL: `ts_rank(ts_delta(close, 5), 10)`
- LQTP: `ts_rank(ts_delta(close, 5), 10)`

## `alpha_20260701_f5d88d1a`
- status: `ready` native=True weekly=False
- pack RankIC: 0.10663348861325994
- DSL: `rank((close - open) / (open + 0.001)) * rank((ts_max(high, 5) - close) / (close + 0.001))`
- LQTP: `rank((close - open) / (open + 0.001)) * rank((ts_max(high, 5) - close) / (close + 0.001))`

## `alpha_20260702_87e60a49`
- status: `ready` native=True weekly=False
- pack RankIC: 0.09703101105714418
- DSL: `(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001)`
- LQTP: `(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001)`

## `alpha_20260702_cd182da7`
- status: `ready` native=True weekly=False
- pack RankIC: 0.09703101105714418
- DSL: `(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 0.001)`
- LQTP: `(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 0.001)`

## `alpha_20260702_cd00ad0b`
- status: `ready` native=True weekly=False
- pack RankIC: 0.09691147532342681
- DSL: `rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 0.001))`
- LQTP: `rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 0.001))`

## `alpha_20260702_c06aea44`
- status: `ready` native=True weekly=False
- pack RankIC: 0.08578569863869512
- DSL: `ts_rank(turnover / (ts_mean(amount, 20) + 0.001), 20) * ts_rank(ts_delta(close, 5), 20)`
- LQTP: `ts_rank(turnover / (ts_mean(amount, 20) + 0.001), 20) * ts_rank(ts_delta(close, 5), 20)`

## `alpha_20260702_8164b3d1`
- status: `ready` native=True weekly=False
- pack RankIC: 0.0816529489936624
- DSL: `ts_rank(ts_delta(close, 5), 20)`
- LQTP: `ts_rank(ts_delta(close, 5), 20)`

## `alpha_20260702_d4af1618`
- status: `ready` native=True weekly=False
- pack RankIC: 0.07967151006647083
- DSL: `ts_rank(ts_delta(close, 5), 21)`
- LQTP: `ts_rank(ts_delta(close, 5), 21)`

## `evoalpha_20260701141121_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701161309_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701175237_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701182212_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701194321_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701204104_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701213731_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701223043_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701232408_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702001949_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702011645_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702020855_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702030717_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702035629_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702045502_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702054334_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702064243_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702072948_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702083207_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702091907_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702102052_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702110833_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702120955_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702125614_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702135802_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702144425_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702154632_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702163049_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702181853_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702200809_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702215434_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260702234217_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260703013105_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260703031716_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260703050616_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260703065317_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260703084031_36be5f53`
- status: `ready` native=True weekly=False
- pack RankIC: 0.075332
- DSL: `low / (close + 0.001)`
- LQTP: `low / (close + 0.001)`

## `evoalpha_20260701141121_1aecc16d`
- status: `ready` native=True weekly=False
- pack RankIC: 0.072387
- DSL: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`

## `evoalpha_20260701161309_1aecc16d`
- status: `ready` native=True weekly=False
- pack RankIC: 0.072387
- DSL: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`

## `evoalpha_20260701182212_1aecc16d`
- status: `ready` native=True weekly=False
- pack RankIC: 0.072387
- DSL: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`

## `evoalpha_20260701204104_1aecc16d`
- status: `ready` native=True weekly=False
- pack RankIC: 0.072387
- DSL: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`

## `evoalpha_20260701223043_1aecc16d`
- status: `ready` native=True weekly=False
- pack RankIC: 0.072387
- DSL: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
- LQTP: `rank(close / (low + 0.001)) * (0 - 1) * rank(ts_std(ts_delta(close, 5), 20))`
