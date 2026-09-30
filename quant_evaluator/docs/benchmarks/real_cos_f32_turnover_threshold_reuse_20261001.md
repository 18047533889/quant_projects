# COS 真实32因子换手率批量评估

原始报告：`real_cos_f32_turnover_threshold_reuse_20261001.json`。
我们在 server-c 用 2586 日期 × 5461 股票 × 32 因子运行 CPU、严格 CUDA 和 auto。
日期跨度为 2016-01-04 至 2026-08-25，输入 dtype 为 float64；F32 指32个因子。
因子有限值比例为 .76619，标签为 .77386。研究数据没有上游 PIT/可投资性认证。

## 版本边界

本报告记录阈值单次排序改动 `4457677b4` 之后的运行。
我们在运行结束后才加入 GPU 上半区插值修复 `8db5c15be` 和零复制哈希 `6c300eb29`。
因此本报告只证明当时版本的性能与容差一致性，不能作为后续版本的真实数据复测。
GPU 与 CPU 的 artifact SHA 不同；我们核验数值容差、轴、有效掩码、计数和 provenance。

| 子进程顺序 | 首次 evaluate（秒） | 第二次 evaluate（秒） |
| --- | --- | --- |
| CPU | 55.4731 | 54.4377 |
| CUDA strict | 18.0967 | 17.5276 |
| auto | 18.1227 | 16.3214 |
| auto | 18.0699 | 16.5039 |
| CUDA strict | 18.3084 | 16.7031 |
| CPU | 55.1938 | 53.8541 |

两次 auto 都选择 CUDA，理由为 `certified_single_metric_real_cos_f32_factor_turnover_rate`。
报告记载 `parity_pass=true`；比较使用 rtol=1e-8、atol=1e-10，并不声称逐位相同。
CUDA 子进程峰值 VRAM 为 2,018,205,696 字节，RSS 约 12,346,644 KiB。
evaluate 计时不包含外层报告序列化，加载数据耗时约 122.87 秒。

该结果覆盖这批数据的单指标换手率请求。
不同指标组合、股票数、缺失模式和未认证的冷请求仍需各自的后端比较。
