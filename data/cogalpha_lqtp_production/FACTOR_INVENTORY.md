# 合并因子清单

## 文件位置

- 7/26 报告包: `data/cogalpha_lqtp_production/reports_7_26/`
- 7/26 原始 zip: `data/cogalpha_lqtp_production/archives/report_7_26.zip`
- Top30 筛选总报告: `data/cogalpha_lqtp_production/reports/cogalpha_factor_screening_top30.html`
- 7/18 生产报告: `data/cogalpha_lqtp_production/reports/`
- 合并清单 JSON: `data/cogalpha_lqtp_production/factor_inventory_merged.json`

## 各池统计

| 来源 | 数量 | 说明 |
|------|------|------|
| CogAlpha 主批 (dsl_catalog) | 176 条 / **170** 唯一 | 7/18 评估，170 有报告 |
| Week2 价量目录 | 37 | 独立 slug 命名 |
| **7/26 优选 30** | **30** | 从 **179** 池子按 |RankIC| 贪心去重选出 |
| screening Top30 (综合分) | 30 | results/1~4 批次 factor_NNN |
| screening 文档内全部 | 66 | 含 RankIC 榜等附录 |

## 7/26 优选 30（report_7_26）

| # | factor_name | route | mean_rank_ic |
|---|-------------|-------|--------------|
| 1 | `factor_intraday_pressure_asym_diff` | python | 0.0729 |
| 2 | `alphasage_b0e9545fa66ab30a` | factor_engine | 0.0620 |
| 3 | `factor_tanh_thrust_dynamic_smooth` | python | 0.0619 |
| 4 | `factor_mut_hc004_volpctrank_s12_clip2_5_volwin6` | python | 0.0616 |
| 5 | `alphasage_43727f7f7b2ba487` | factor_engine | 0.0583 |
| 6 | `alphasage_89d1bc9b29c9d35a` | factor_engine | 0.0567 |
| 7 | `factor_ema_vol_asymmetry_rank` | python | 0.0552 |
| 8 | `factor_herding_volume_momentum_continuous` | python | 0.0506 |
| 9 | `factor_vol_regime_trend_smoothed` | python | 0.0499 |
| 10 | `factor_volume_adaptive_momentum` | python | 0.0492 |
| 11 | `factor_volatility_compression` | python | 0.0491 |
| 12 | `factor_volume_weighted_range_skew` | python | 0.0472 |
| 13 | `factor_volume_confirmed_ewm` | python | 0.0465 |
| 14 | `factor_volume_confirmed_up_capture` | python | 0.0455 |
| 15 | `factor_volume_surge_persistence` | python | 0.0453 |
| 16 | `factor_intraday_adx_momentum` | python | 0.0438 |
| 17 | `factor_crash_vol_spike_volregime_boost` | python | 0.0435 |
| 18 | `factor_simple_convex_momentum_log_volume` | python | 0.0431 |
| 19 | `factor_volume_adjusted_divergence_ema10` | python | 0.0422 |
| 20 | `alphasage_baec4247c71a031c` | factor_engine | 0.0396 |
| 21 | `alphasage_99e8c98dc92fba46` | factor_engine | 0.0394 |
| 22 | `factor_mut_cro_asym_vol_ratio_med_15_minp3_vol10` | python | 0.0386 |
| 23 | `factor_volatility_asymmetry_spike_adj_ema5` | python | 0.0383 |
| 24 | `factor_momentum_volratio` | python | 0.0380 |
| 25 | `factor_directional_vol_volbase25_smooth_dir_range10` | python | 0.0360 |
| 26 | `alphasage_154a51fe952b5243` | factor_engine | 0.0350 |
| 27 | `alphasage_3aaad1a1bf9ce8a2` | factor_engine | 0.0335 |
| 28 | `alphasage_5de2f47bc5cd6c66` | factor_engine | 0.0321 |
| 29 | `alphasage_58079cceee41ce6d` | factor_engine | 0.0321 |
| 30 | `alphasage_3e74b4fa2831dc36` | factor_engine | 0.0301 |

## screening 综合 Top30（factor_718 等）

| # | 批次 | 因子 |
|---|------|------|
| 1 | results/3 | `factor_718` |
| 2 | results/3 | `factor_658` |
| 3 | results/2 | `factor_489` |
| 4 | results/1 | `factor_314` |
| 5 | results/2 | `factor_572` |
| 6 | results/4 | `factor_805` |
| 7 | results/1 | `factor_122` |
| 8 | results/1 | `factor_89` |
| 9 | results/1 | `factor_570` |
| 10 | results/3 | `factor_749` |
| 11 | results/1 | `factor_315` |
| 12 | results/1 | `factor_544` |
| 13 | results/4 | `factor_892` |
| 14 | results/2 | `factor_573` |
| 15 | results/4 | `factor_790` |
| 16 | results/1 | `factor_215` |
| 17 | results/1 | `factor_70` |
| 18 | results/1 | `factor_299` |
| 19 | results/1 | `factor_209` |
| 20 | results/1 | `factor_302` |
| 21 | results/1 | `factor_247` |
| 22 | results/1 | `factor_248` |
| 23 | results/4 | `factor_828` |
| 24 | results/3 | `factor_646` |
| 25 | results/3 | `factor_735` |
| 26 | results/1 | `factor_546` |
| 27 | results/4 | `factor_879` |
| 28 | results/3 | `factor_643` |
| 29 | results/1 | `factor_104` |
| 30 | results/3 | `factor_644` |

## 与 CogAlpha 170 批关系

- 7/26 优选 30 中，与 `factor_*` 主批 **同名重叠**: **0** 个
- screening 的 `factor_718` 等是 **results/1~4 挖掘批次编号**，与主批 `factor_persistence_ewma` 等是 **不同命名体系**
- 7/26 池子来源 `factor_pool`（179 候选），含 hash 名 CogAlpha 因子 + AlphaSage DSL 因子
