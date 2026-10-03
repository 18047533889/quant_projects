# 真实 F48 当前后端验证（2026-10-03）

固定 manifest：`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
从已有 COS 落值选取 48 个不同内容哈希因子，实际 shape 为 2586 天 × 5461 股票 × 48 因子，
float64；使用有界 COS source、两个读取 worker、512 MiB 预取预算、4096 MiB source 预算、
GPU 每块 2 因子。该批清单字节合计 3,620,857,809；不构造完整 F48 值立方体副本。

## 当前实测及 auto

| 调用 | CPU（秒） | CUDA（秒） | ordinary auto（秒） |
|---|---:|---:|---:|
| CPU-first 当前对照 | 233.8842 | 54.9888 | 未请求 |
| CUDA-first 后续核验 | 231.6482 | 52.4593 | 59.4330 |

CPU-first 的 CUDA 相对当前 CPU 约快 4.25 倍；这不是新旧版本提速，
三个指标为 rank_ic、quantile_spread、factor_turnover_rate，不包含 Pearson。
计时是 source API 调用 wall time，不含前置公共轴扫描、标签读取、
source 工厂构造及计时结束后的 close/final verify。

CUDA-first 测试可能与另一个窗口的短暂因子扫描/标签准备重叠；
对方已核验自己的远端重复进程确实终止。该轮数值核验保留，
耗时仅作探索记录，不作为独立稳定基准或扩大认证范围的依据。
本测试没有控制服务器所有其他业务进程负载。

ordinary auto 未注入候选，真实选择 `cuda`，原因
`bounded_f48_mixed_three_gpu_tile2`，有效分块宽度 2。
CUDA/auto 都处理完整 24 块，OOM retries = 0，
设备 peak 2,452,759,552 字节；这是记录的运行峰值，不是所有输入的保证。

## 数值对照

三指标共 144 个标量比较通过，有限值掩码与观测数量一致：

| 指标 | CPU/CUDA 最大绝对差异 |
|---|---:|
| rank_ic | 1.07553e-16 |
| quantile_spread | 2.92735e-18 |
| factor_turnover_rate | 3.88578e-16 |

auto 与 CPU 对照也通过。两个报告的 QE Python 来源摘要相同：
`b13bb54505d6da85b29726dd4736e4da71369a4f2074774282f9e0388a97046f`，
166 文件范围在运行前后未变化。该范围不认证外部 DataAccess/FO、原生库或已导入内存代码；
研究来源验证也不是 PIT/生产因子认证。

## 读取瓶颈与安全边界

CPU-first 的 CUDA source 读取等待 wall sum 为 48.0006 秒（API 共 54.9888 秒）；
CPU 同项为 13.6095 秒（API 共 233.8842 秒）。CPU 的计算较慢，
能与提前读取更充分重叠；GPU 已主要受数据读取影响。
`bound_factor_read` 是多个预取任务累积时间，不能与其他阶段直接相加当总耗时。
不能凭融合 Pearson 核心调用约 5.5 倍的单独结果推断这里整批又快 5.5 倍。
后续优先验证 IO 并发/块宽、完整 Pearson 链以及其他请求形状；不直接扩大 auto 认证区间。

之前的内存拒绝回执仍保留为历史安全门槛证据；本轮内存恢复后才成功启动。
未降低 32 GiB source 准入门槛、默认读取预算或精度要求。

报告：
- [CPU-first](benchmarks/real_cos_f48_fused_cpu_first_retry_20261003.json)
- [CUDA-first + ordinary auto](benchmarks/real_cos_f48_fused_cuda_first_auto_20261003.json)

复现环境仅作用于当前测试进程：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_source_batch \
  --factors 48 --tile-size 2 --days 2586 --assets 5461 \
  --source-adapter cos --cos-prefetch auto --run-order cuda-first --verify-auto \
  --max-object-mib 128 --max-total-mib 4096 --max-source-memory-mib 4096 \
  --max-prefetch-memory-mib 512 --output /tmp/f48-current-verification.json
```

全部指标最快、所有场景无 bug、完整 FE 原生复用与已知 FE 极值递推问题仍未闭合。
