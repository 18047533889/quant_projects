# R62 — 全量算子向量化收尾（batch-4，subagent 并行）

日期：2026-09-22（qs-server-c /home/sunhaiwei/quant_projects）

## 概述

R61 后剩余 ≥40ms/标的 的逐行算子全部处理完毕。本轮由 4 个并行 subagent（+主线程修复）完成，
覆盖 13 个源文件、约 45 个算子。所有改动均通过与 HEAD 权威的逐位 A/B 对拍
（13 个对抗面板：NaN 洞/常数/全零/重尾/1e±300/单调/tie/短序列等）。

## 各集群成果（单标的耗时，750 行面板）

| 文件 | 算子 | 优化前 → 优化后 |
|---|---|---|
| extreme_tail | ts_mean_excess_slope | 103ms → 1.5ms |
| extreme_tail | ts_extremal_index / ts_hill_tail_index / ts_gpd_shape_pwm | 27/12/10ms → ~1ms |
| moments_ext | ts_hartigan_dip | 82ms → 24.4ms（ECDF 批量，exact dip 有本质 O(n²) 下限） |
| topology_ext | ts_persistence_entropy_h0 | 94ms → 5.7ms（one-hot cumsum 批量） |
| topology_ext | ts_persistence_entropy_h1 | 348ms → 167ms（lock-step 批量 F2 归约；union-find 本质迭代，2×） |
| intrinsic_dimension | ts_delay_intrinsic_dimension | 150ms → 3.1ms |
| advanced_structure | ts_kramers_moyal_drift / diffusion | 105/104ms → ~1ms |
| candle_state_space | ts_matrix_profile_motif_age / novelty | 79/77ms → 0.6ms（修复了 agent 版本的距离单位 bug） |
| filter_smooth | ts_butterworth_lowpass_causal | 69ms(冷启动噪声)/0.6ms → 0.50ms（有限段批量 sosfilt） |
| rough_vol | ts_vol_pvariation_roughness / ts_vol_scaling_break | 99/77ms → 1.4/1.6ms（修复 agent 遗留的调用名丢失 + 零宽窗口崩溃） |
| robust_scale | ts_qn_scale | 102ms → 27.3ms（按 run 长度分组 + 批量 partition） |
| complexity_ext | forbidden_ordinal_pattern ×3 | 120ms → ~2ms |
| advanced_quantile_dynamics | ts_quantilogram / ts_extremogram | 25/15ms → ~2ms |
| alpha_language_distribution | wasserstein/quantile_transport/mmd/es/location 6 ops | 30ms 量级 → ~2ms |
| alpha_language_state | ts_sign_persistence | 60ms → ~2ms |
| recurrence_analysis | rqa 4 ops | 47ms → ~2ms |
| prospect_theory / advanced_expectile / alpha_language_shape / multifractal / multiscale_trend / state_geometry / regression_models / memory_ext / overhaul(base) | 各集群余量 | → ~2ms 级 |
| alpha_language_shape | ts_monotonicity（主线程补刀） | 127ms → 1.6ms（79×，padded 窗口 + 上三角广播差分） |

## 全量重扫结论（1429 个 daily 算子，503 个可计时）

- ≥30ms/标的 的慢点从 R60 的 71 个降到 **23 个**，其中 18 个是 `intra_*` 分钟线算子
  （需分钟线面板语义，不适用日线 sweep 口径）。
- 真正的 daily 剩余慢点仅 5 个：ts_persistence_entropy_h1 108ms（本质迭代）、
  ts_monotonicity 50ms（本轮已修 →1.6ms）、ts_rqa_determinism/laminarity_fixed_rr ~31ms、
  ts_quantile_kurtosis 31ms。

## 验证

- A/B：r62_ab_{alpha,misc1,misc2,seasonal,verify,fs2,vol,qn2,mono,topo2}.py 全部 TOTAL_FAIL 0。
- pytest：1542 passed / 26 failed → 修复 rough_vol 7 个后剩 19 个全部为预存失败
  （state_event 16 个 = ConditionBool 契约预存、r21 治理 2 个、v9_filter_cpu_bridges 1 个
  = source-hash 钉死，均与 R62 改动无关，已做 HEAD 基线对照确认）。
- 治理检查 GOVERNANCE_OK（migrated=1429 不变）。

## 教训（batch-5 参考）

- subagent 改共享文件后必须跑该文件全部 pytest——rough_vol 的调用名丢失、
  candle 的距离单位 bug 都是 A/B 抓到/测试抓到的。
- 加载 HEAD 权威副本需屏蔽 register/declare_stateful 副作用（r62_load_old.py）。
- /tmp 会被服务器清理：权威备份用 `git show HEAD:` 现做，不要依赖陈旧 .orig。
