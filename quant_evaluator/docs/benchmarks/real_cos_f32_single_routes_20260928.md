# F32 单指标 CPU/CUDA/auto 真实批量验证（2026-09-28）

来源：DataAccess 绑定的 COS landing manifest
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`，
同一批 32 个 verified 因子，2586 日 × 5461 股 × 32 因子，float64。
标签为决策 t、执行 t+1、`AdjVwap(t+2)/AdjVwap(t+1)-1`；
所有调用经过公开 `evaluate`，设备 NVIDIA L20，BLAS 单线程。

## 交错 A/B

每项按 CPU/CUDA strict/CUDA strict/CPU 四轮运行，每轮一次完整请求。
比较器逐项检查制品类型、值、有限/有效掩码、计数、来源及逐因子 MetricValue；
四轮配置哈希均一致，所有比较均通过。

| 指标 | CPU 两端（秒） | CUDA 两轮（秒） | CUDA pool 峰值（字节） | 配置哈希 |
| --- | ---: | ---: | ---: | --- |
| `rank_ic` | 84.987 / 85.301 | 39.063 / 39.537 | 9,451,395,584 | `b51545a37b09cd8949344c445d56b48fbc8732c84f902bf786d6fd3797aebdca` |
| `quantile_spread` | 79.905 / 79.638 | 36.385 / 36.332 | 1,436,371,456 | `5d4e2c1261dce54a263b70112441fe017c55959068a167e7126fe26975d36059` |
| `factor_turnover_rate` | 69.621 / 68.740 | 36.150 / 36.696 | 1,129,165,824 | `79829164328dd18bbd3bc34de93bfd493d1f3e6e521481d03f7b958d9c0ba716` |

先行单轮 CPU/CUDA 完整制品对拍也分别通过：Rank IC
85.269/38.485 秒、分位价差 78.999/36.608 秒、换手率
69.623/36.336 秒。交错批次的父进程 RSS 峰值 13,793,408 KiB，
Rank IC 的 CPU/CUDA worker 峰值约 23,670,196/25,307,988 KiB。

## 自动路由复测

加入精确单指标路由后，重新载入相同绑定面板；每项用一次 CPU 参考
和两次 `auto`，两次 `auto` 都实际走 CUDA，配置哈希一致，
完整制品均与 CPU 参考对拍通过。

| 指标 | CPU 参考（秒） | `auto` 两轮（秒） | 路由原因 |
| --- | ---: | ---: | --- |
| `rank_ic` | 83.612 | 38.518 / 40.230 | `certified_single_metric_real_cos_f32_rank_ic` |
| `quantile_spread` | 80.443 | 37.389 / 36.927 | `certified_single_metric_real_cos_f32_quantile_spread` |
| `factor_turnover_rate` | 69.195 | 36.431 / 36.548 | `certified_single_metric_real_cos_f32_factor_turnover_rate` |

## 边界

只对此精确形状、这三个单指标各自的默认参数、float64、无特殊输入、
单 NVIDIA L20 使用自动 CUDA。`rank_ic` 要求实际及有效空闲显存
至少 14 GiB；其余两项至少 8 GiB。设备/显存/形状/dtype/参数不符
均回到 CPU。单项认证不证明任意多指标组合最快；完整 rank、分位
及默认三指标集合有各自独立的整批 A/B。服务器同时运行其它任务，
这些秒数不是其它硬件的性能承诺，也不证明 164 项指标全无 bug。
