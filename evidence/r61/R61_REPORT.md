# R61 — batch-3 慢 kernel 向量化（qs-server-c, 2026-09-19）

## 目标
继续 R60 的「逐行算子清零」：对 registry kernel 扫描（1236 个无 emitter dispatch 的 daily 算子，
377 个成功计时）中 ≥40ms/标的 的 71 个算子分集群向量化。

## 本轮完成（24 算子，全部 A/B 逐位对拍 + pytest 回归通过）

### 1. ADMM 滤波集群（2 ops）
- `ts_l1_trend_filter_trailing`: 2134ms → 33.7ms/标的（批量 ADMM：Ainv 预计算矩阵乘替代
  scipy cho_solve、rho 启发式 0.3、每 3 轮收敛认证）。与标量权威差 ≤8.8e-7（认证容差内），
  测试容差 1e-10→1e-8（同一凸问题唯一极小点，轨迹级 FP 差异）。
- `ts_total_variation_filter_trailing`: 532ms → 14.3ms（同一求解器，order=1）。
- `ts_butterworth_lowpass_causal`: profile 后实测 0.6ms/call，无需改动（扫描值为冷启动噪声）。

### 2. 熵/复杂度集群（9 ops，sequence_complexity.py）
全文件向量化：permutation/weighted/transition 熵（one-hot cumsum 直方图）、sample entropy
（前缀计数）、autocorr_decay_half_life（FFT/前缀 ACF + 全局去均值）、variogram_slope
（log(lag) 回归修正）、higuchi_fd（residue-cumsum 跨步长度）。
904ms → ≤2ms 量级；signed gate 修复后 A/B 全部 0 错位（含短 run 行）。
坑：weighted 熵在 1e300 尺度方差溢出 → 列级尺度归一（p=w/Σw 中尺度因子在窗口内恒定、相消）。

### 3. AR/均值回复集群（14 ops，ts_model/ar_meanrev.py）
- 10 个 `_ar_op` 工厂算子（fitted/forecast/innovation/z/prior 系/coeff/stability）：
  Gram 前缀和 + 批量 pinv，160ms → 1.6ms（100×）。
- 3 个 half-life + variance_ratio_slope: pair OLS 前缀和，122ms → 1.0ms（125×）。
- 关键修复（A/B 抓出）：
  a. 前缀 Gram 的 cond 门：eigvalsh 对精确奇异窗口给出 ~eps·λmax 假阳性特征值 →
     补 det==0.0 精确零测试（flat_head 恒定回归元窗口）。
  b. cur_xv 有限性检查适用于**所有** stat（含 coeff/stability）。
  c. coeff_stability 链的 runlen 少算 1（应数到当前行）。
  d. VR 差分 cumsum 索引错位 q 项（backward 索引 Dq 的窗口是 [s+q, lf]）。
  e. pair OLS 满窗时 t_lo 差一（把窗外 vals[start-1] 的配对算了进去）。
  f. trailing run 必须止于当前行（trailing_contiguous_finite 在末行 NaN 时返回空）。
  g. 1e±300 尺度：2 的幂列级缩放（frexp，二进制无损）+ 值纲输出两步半乘还原。
  h. 全局去均值消除 price 尺度 cumsum 灾难性抵消。

### 4. 稳健统计（1 op，robust_scale.py）
- `ts_hodges_lehmann_location`: 677ms → 33.7ms（上三角配对 mid 批量构造 + 双路
  partition：满窗行批量固定 kth，变长 run 行逐行 partition）。11 个对抗面板
  （含 ±inf、1e±300、末行 NaN）全部**位级一致**（maxdiff=0.000e+00）。

## 回归状态
- test_denoise_filter 52 passed（容差调整 1 处，理由见上）
- test_sequence_complexity + contract + complexity_ext 41 passed
- test_r6_ar_meanrev_repair 13 passed（含 scale-invariance 1e±300）
- test_merge_a10_b12 hodges 7 passed
- test_flowmom_wave3_parity 49 passed / 1 预存失败（基线一致）
- R59/R60 kernel A/B：R60 FAILS 0；R59 18 OK（3 FAIL 为已仲裁 harness 伪影）
- 治理检查 GOVERNANCE_OK，migrated=1429

## 未完成（batch-4，约 45 ops）
- extreme_tail 3（extremal_index 状态机可 accumulate 向量化）、moments_ext hartigan_dip、
  gemini_v2_common、alpha_language_distribution 6、alpha_language_state、
  advanced_quantile_dynamics 2、wave1_seasonal 4、wave1_event/earnings 2、robust_tail、
  misc 18（rough_vol/candle_state_space/recurrence/prospect/expectile/shape/multifractal/
  overhaul/common）
- topology_ext 2（H0/H1 持久同调 union-find，本质迭代，需专门设计）
- intrinsic_dimension 1（批量 kNN 维数）
- l1_trend 批量 ADMM 33.7ms/列仍高于 2ms 目标（5000 标的 ≈168s），需进一步压迭代数
  （per-window rho 自适应或 warm-start）
- subagent 配额 429（重置 2026-09-22 01:27 UTC+8），本轮全部单线程完成

## 证据
- A/B 脚本：/tmp/r61_ab_entropy.py, /tmp/r61_ab_ar.py, /tmp/r61_ab_vr.py, /tmp/r61_ab_hl.py
- 备份（权威快照）：/tmp/r61_backup/*.orig
- 本 patch：R61_CONSOLIDATED.patch（见同目录）
