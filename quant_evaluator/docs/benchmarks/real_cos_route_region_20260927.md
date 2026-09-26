# 真实 COS 因子多年全市场后端区域实测（2026-09-27）

输入沿用 [全历史基准](real_cos_factor_batch_20260927.md)：已验证 COS landing manifest 绑定的两条真实因子、DataAccess 注册的 A 股交易日历和复权 VWAP。完整输入 2586 日 × 5461 股 × 2 因子，2016-01-04 至 2026-08-25；所有日历交易日均有本地 AdjVwap 分区。每个形状从完整输入截取最近 T 日、排序后的前 N 股、前 F 因子。一次读取 COS 后逐形状测试，加载 20.87 秒，父进程峰值 RSS 2.81 GiB。设备为共享的 NVIDIA L20。

以下每格为冷运行 / 热运行（秒）。CPU 和 `cuda_strict` 进程独立；形状之间交替启动顺序，每个后端各测一次冷、一次热。

| T × N × F | rank_ic CPU | rank_ic CUDA | CUDA 峰值显存 |
|---|---:|---:|---:|
| 1000 × 5000 × 1 | 1.384 / 1.132 | 2.401 / 0.296 | 0.81 GB |
| 1000 × 5461 × 2 | 3.117 / 2.632 | 1.454 / 0.616 | 5.46 GB |
| 1800 × 5000 × 2 | 4.916 / 4.226 | 2.023 / 1.001 | 5.05 GB |
| 2400 × 5461 × 1 | 3.319 / 2.710 | 1.616 / 0.784 | 8.60 GB |
| 2400 × 5000 × 2 | 6.328 / 5.522 | 2.530 / 1.335 | 7.88 GB |
| 2586 × 5461 × 2 | 7.109 / 6.157 | 2.722 / 1.544 | 11.23 GB |

| T × N × F | quantile_spread CPU | quantile_spread CUDA | CUDA 峰值显存 |
|---|---:|---:|---:|
| 1000 × 5000 × 1 | 0.971 / 0.599 | 0.923 / 0.268 | 0.45 GB |
| 1000 × 5461 × 2 | 1.665 / 1.138 | 1.401 / 0.545 | 0.55 GB |
| 1800 × 5000 × 2 | 2.606 / 1.791 | 1.927 / 0.886 | 0.61 GB |
| 2400 × 5461 × 1 | 1.959 / 1.426 | 1.449 / 0.691 | 0.63 GB |
| 2400 × 5000 × 2 | 3.212 / 2.304 | 2.304 / 1.178 | 0.69 GB |
| 2586 × 5461 × 2 | 3.779 / 2.628 | 2.585 / 1.374 | 0.76 GB |

12/12 个指标/形状组合的公共 `evaluate()` CPU/CUDA 输出对拍通过，逐项核对形状、有限值 mask、数值（rtol 1e-8、atol 1e-10）、计数、有效性 mask、MetricValue 的有效性/样本数/样本单位、artifact 来源计数及输入因子哈希。全部热运行中 CUDA 更快；`rank_ic` 约 3.46–4.27 倍，`quantile_spread` 约 1.91–2.23 倍。T1000 N5000 F1 的 `rank_ic` 冷启动 CUDA 较慢，说明一次性小批任务仍需单独考虑启动成本。

这些离散点支持在 L20 显存充裕时，为 T1000–2586、N5000–5461、F1/F2 的两项指标建立有界 CUDA auto 候选区域；离散实测不是该区域每个尺寸、所有 GPU、混合指标批次都更快的数学证明。路由还需遵守显存容量、输入 dtype、特殊参数和既有回退契约。本次因子为研究来源，并未认证上游 PIT 或可投资性；共享服务器负载可能影响时间。完整数值、逐项 parity 和显存字节数见 [JSON](real_cos_route_region_20260927.json)。

复现：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_route_region \
  --limit 6 --output quant_evaluator/docs/benchmarks/real_cos_route_region_20260927.json
```
