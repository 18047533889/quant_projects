# 当前 F48 六指标默认 auto：证据与边界

本说明只覆盖当前固定请求，不是全部指标、任意规模或生产任务的性能承诺。

## 固定请求

- 实际 COS 因子来源，形状 T=2586、N=5461、F=48。
- 六指标：quantile_curvature、quantile_tail_asymmetry、quantile_adjacent_spread、quantile_extreme_cliff、top_quantile_cliff、bottom_quantile_cliff。
- 请求 tile 上限16，实际来源/GPU准入宽度5。测量包含整个批量评估调用，不是单内核耗时。
- 所有比较包含逐指标独立公式对照、数值、有效样本掩码及观测计数；完整覆盖288个因子-指标结果。

## 已完成的 ABBA 与默认 auto

报告：`real_cos_f48_linear_shape_profile_current_root_20261004_e20e97d6c317.json`。

CPU 两次：114.31550755997887、121.6592540779966 秒。
CUDA 两次：66.55510398000479、66.59511034999741 秒。
本请求的中位数比约1.77倍；四次独立对照均通过、OOM重试为0。
显式资格 auto 和随后默认 auto 均选择 CUDA，后者命中缓存。

## 已完成的独立新进程 provider 验证

报告：`real_cos_f48_linear_shape_provider_current_root_20261004_0fac516e100b.json`。

第一次普通 auto 为 report_candidate / candidate_validated / supplied；第二次为 process_cache / not_checked / cache_hit。
两次均 qualified_current_source、qualification_applied=true、backend_used=cuda、OOM重试0、实际宽度5。
协调者独立读取报告并断言：两次完整 context 均等于两个 ABBA profile context；六指标的 values_sha256、finite_mask_sha256、observation_counts_sha256 均等于两个 CUDA profile；覆盖计数完整。
这证明新进程配置 provider 后普通 auto 能使用当前资格，不证明实际生产 worker 已配置该 provider。

## 已完成的三组随机配对稳定性补充

三组随机配对使用 seed=20261004，完整报告为 `real_cos_f48_linear_shape_randomized_current_root_20261004_3e7a1ace149f.json`。
权威执行 session30326 已退出0；六次完整独立对照与测量回执通过，三组均 CUDA 胜出。
CPU 秒数：116.03170669701649、118.97837923798943、123.98216264403891。
CUDA 秒数：66.87503061397001、64.19567865901627、65.0652852130006。
中位数 CPU118.97837923798943 / CUDA65.0652852130006，约1.83倍；aggregate_pass=true，与 ABBA 赢家一致。
随机报告自身 qualification_available=false，只补充稳定性证据，不安装或扩大资格。

## 下一阶段不能绕过的门槛

全部24来源指标、F61等其他形状、任意指标子集、派生来源、多标签以及实际生产调用接入仍需各自独立资格、真实比较和资源准入。源码变化会使旧资格失效，不能用本报告为新源码背书。
