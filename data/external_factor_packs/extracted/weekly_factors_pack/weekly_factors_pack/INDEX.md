# 周报因子打包（CogAlpha A 股价量）

生成时间: 2026-08-02T09:24:55.520357+00:00

## 目录

- `week1_previous_elite/`：第一周 PPT 的 8 个 factor_name
- `week2_new_rank_ic_gt_0p03/`：第二周新增（train Rank IC > 3%）
- `all_rank_ic_gt_0p03/`：全部去重后集合
- `summary.csv` / `INDEX.md` / `FORMULAS.md`

## 口径

- 来源 campaign: `llm_ashare_pv_cogalpha_2026q2`
- 去重: 按 `factor_name`，优先 `standard/` 架，取 train Rank IC 最高的一条
- train: CogAlpha fitness **2018-01-01 ~ 2021-12-31**
- lqtp_test: 有则约 **2023-01-01 ~ 2026-06-15**（含 orientation flip）

- week1: 8
- week2: 34
- total: 42
- elite / qualified: 11 / 31

## 汇总表

| week | fitness | factor_name | train_IC | train_RankIC | test_RankIC | candidate_id |
|------|---------|-------------|----------|--------------|-------------|--------------|
| week2 | qualified | `factor_mut_hc004_volpctrank_s12_clip2_5_volwin6` | 0.0272 | 0.0499 |  | `raw_ca_hash_53a3e3f7` |
| week2 | qualified | `factor_crossover_herding_stress_span12_volrank_clip3` | 0.0269 | 0.0492 |  | `raw_ca_hash_b0ca3bdd` |
| week2 | qualified | `factor_mut_hc004_volpctrank_w8` | 0.0269 | 0.0492 |  | `raw_ca_hash_b719027d` |
| week2 | qualified | `factor_vol_price_log_range_gate_clipped_boost_5day_mut_001` | 0.0241 | 0.0483 |  | `raw_ca_hash_27a292b1` |
| week2 | qualified | `factor_vol_price_log_range_gate_clipped_10day_mut_003` | 0.0246 | 0.0479 |  | `raw_ca_hash_c1a04bc5` |
| week2 | qualified | `factor_vol_surge_regime_gated` | 0.0249 | 0.0456 | 0.0397 | `raw_ca_hash_c4358b90` |
| week2 | qualified | `factor_volume_confirmed_momentum_tanh_gated` | 0.0368 | 0.0453 | 0.0054 | `raw_ca_hash_0a4e4a21` |
| week1 | elite | `factor_intraday_return_volume_surge` | 0.0254 | 0.0452 | 0.0380 | `raw_ca_hash_6b0f76d3` |
| week1 | elite | `factor_herding_momentum_stress` | 0.0242 | 0.0451 |  | `raw_ca_hash_ef639934` |
| week2 | qualified | `factor_vol_weighted_pct_range_clip_ema10` | 0.0239 | 0.0450 |  | `raw_ca_hash_948c7055` |
| week1 | elite | `factor_volume_confirmed_momentum` | 0.0249 | 0.0432 | 0.0094 | `raw_ca_hash_d37770d2` |
| week1 | elite | `factor_directional_vol_volbase25_smooth_dir_range10` | 0.0297 | 0.0432 |  | `raw_ca_hash_1476a21c` |
| week2 | qualified | `factor_adaptive_vol_directional_smooth_range10` | 0.0297 | 0.0432 |  | `raw_ca_hash_f1858c32` |
| week2 | qualified | `factor_intraday_volume_spike` | 0.0253 | 0.0430 | 0.0330 | `raw_ca_hash_e9cbfd57` |
| week2 | qualified | `factor_herding_vol_stress` | 0.0169 | 0.0410 | 0.0269 | `raw_ca_hash_340f0ad9` |
| week1 | elite | `factor_intraday_reversal` | 0.0341 | 0.0394 | 0.0054 | `raw_ca_hash_e5e44222` |
| week2 | elite | `factor_volume_weighted_price_change` | 0.0244 | 0.0391 |  | `raw_ca_hash_83b7560a` |
| week1 | elite | `factor_volume_adjusted_momentum` | 0.0240 | 0.0385 |  | `raw_ca_hash_32d50b39` |
| week2 | elite | `factor_directional_vol_smooth_range_mut` | 0.0246 | 0.0385 |  | `raw_ca_hash_338dd16c` |
| week2 | qualified | `factor_vol_gated_momentum_v3` | 0.0216 | 0.0377 | 0.0298 | `raw_ca_hash_0f53ef7d` |
| week1 | elite | `factor_directional_vol_adaptive_crossover` | 0.0242 | 0.0375 |  | `raw_ca_hash_2fc67cd0` |
| week1 | elite | `factor_vol_gated_momentum_volume_weighted` | 0.0233 | 0.0365 | 0.0321 | `raw_ca_hash_61b8fb3c` |
| week2 | qualified | `factor_volume_confirmed_momentum_clipped` | 0.0249 | 0.0362 | 0.0298 | `raw_ca_hash_50cd89c1` |
| week2 | qualified | `factor_vol_regime_breakout_continuous` | 0.0309 | 0.0354 | 0.0093 | `raw_ca_hash_c83fcf6c` |
| week2 | qualified | `factor_vol_regime_trend_smoothed` | 0.0168 | 0.0349 | 0.0441 | `raw_ca_hash_7c569498` |
| week2 | qualified | `factor_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15_stressclip25` | 0.0246 | 0.0345 |  | `raw_ca_hash_c6320c09` |
| week2 | qualified | `factor_stress_volume_cont_weight_cont_dir_w20_clip2_5_w40_ewm15_stressclip25` | 0.0246 | 0.0345 |  | `raw_ca_hash_c4485ee6` |
| week2 | elite | `factor_stress_volume_cont_weight_cont_dir_w20_clip3_w40_ewm15` | 0.0246 | 0.0345 |  | `raw_ca_hash_c7206013` |
| week2 | qualified | `factor_stress_volume_cont_weight_cont_dir_w25_clip2_w40_ewm15` | 0.0245 | 0.0344 |  | `raw_ca_hash_a96b0966` |
| week2 | qualified | `factor_stress_range_volume_surge` | 0.0156 | 0.0334 | 0.0487 | `raw_ca_hash_f2c0364c` |
| week2 | qualified | `factor_pressure_tanh` | 0.0160 | 0.0331 | 0.0258 | `raw_ca_hash_b9907c26` |
| week2 | qualified | `factor_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5` | 0.0138 | 0.0328 |  | `raw_ca_hash_bc141edf` |
| week2 | qualified | `factor_herding_simple` | 0.0172 | 0.0327 | 0.0212 | `raw_ca_hash_718ea675` |
| week2 | qualified | `factor_adaptive_momentum_gate_v2` | 0.0173 | 0.0325 | 0.0281 | `raw_ca_hash_946bc42a` |
| week2 | qualified | `factor_volume_confirmed_momentum_binary_gate` | 0.0290 | 0.0320 | 0.0060 | `raw_ca_hash_777e86d4` |
| week2 | qualified | `factor_intraday_reversal_crossover_v1` | 0.0305 | 0.0320 | 0.0055 | `raw_ca_hash_881db7aa` |
| week2 | qualified | `factor_range_position_volume_gated_smoothed` | 0.0302 | 0.0315 | 0.0067 | `raw_ca_hash_05d8d226` |
| week2 | qualified | `factor_mut_cro_asym_vol_ratio_med_15_minp3_vol10` | 0.0150 | 0.0311 |  | `raw_ca_hash_bcbd6919` |
| week2 | qualified | `factor_imbalance_volume_ratio_ewm` | 0.0285 | 0.0310 | 0.0053 | `raw_ca_hash_263e9c66` |
| week2 | qualified | `factor_range_position_volume_gated_smoothed_mut` | 0.0306 | 0.0309 | 0.0067 | `raw_ca_hash_d8e639c3` |
| week2 | qualified | `factor_imbalance_volume_ewm_mut` | 0.0142 | 0.0308 | 0.0060 | `raw_ca_hash_15a03771` |
| week2 | qualified | `factor_mut_cro_asym_vol_ratio_med_20_minp3_vol10_ema2` | 0.0139 | 0.0304 |  | `raw_ca_hash_ae211225` |
