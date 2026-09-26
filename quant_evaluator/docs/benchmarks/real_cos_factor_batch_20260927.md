# 真实 COS 因子全历史 QE 批量 A/B（2026-09-27）

输入由 `scripts/load_real_cos_factor_batch.py` 只读取得。DataAccess 精确对象研究入口绑定 SHA256 为 `00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66` 的 landing manifest；两条因子分别为 `weekly_4cbd7ca6dccc61dc` 与 `weekly_db5616cd85a477b3`。两条因子对象下载后通过 manifest 字节数、内容 SHA256、前后对象元数据校验，临时对象自动清理。标签走已登记 `ashare_calendar` 和 `ashare_stock_daily_adj`，从现有本地镜像读取。

精确 COS 身份：

- manifest: `cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66/landing_manifest.json`
- factor 1: `cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values/35ad096b62be98ce5766fe3df4e651bd7930ee377683d6282171a6984ed14a59/weekly_4cbd7ca6dccc61dc.parquet`（5,400,494 字节）
- factor 2: `cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values/674763066b67e4ed032122b51545a183f8a851b2488ff43ba2a81ab25c07df5d/weekly_db5616cd85a477b3.parquet`（3,828,456 字节）
完整重叠输入为 2016-01-04 至 2026-08-25，2586 个决策交易日 × 5461 只 A 股 × 2 因子。决策日 t 的标签是 `AdjVwap(t+2)/AdjVwap(t+1)-1`；日历 t、t+1、t+2 均核对已有分区。已登记日历内缺失的 AdjVwap 交易日分区为 0；2024-09-16/17 是日历标记的非交易日，未进入标签。因子有效值比例 76.55%，标签有效值比例 77.39%。这份来源记录不等于上游 PIT 或可投资性认证。

| 指标 | CPU 热运行 | CUDA 热运行 | auto 热运行 / 实际后端 | 当前最快 |
|---|---:|---:|---:|---|
| rank_ic | 6.173 s | 1.565 s | 6.103 s / CPU | CUDA 约 3.94 倍 |
| quantile_spread | 2.614 s | 1.386 s | 2.649 s / CPU | CUDA 约 1.89 倍 |
| factor_turnover_rate | 2.584 s | 4.240 s | 2.593 s / CPU | CPU 约 1.64 倍 |

CPU、`cuda_strict` 和 `auto` 对三项的数值、有限值 mask、计数、有效性和输入哈希逐项对拍通过；CUDA 相对 CPU 最大绝对差依次为 `3.47e-18`、`1.07e-18`、`2.78e-17`。输入加载 21.11 s，父进程峰值 RSS 2.76 GiB。执行设备 NVIDIA L20（46 GiB）；测试启动时约 9.3 GiB 显存已被其他进程占用。

每个后端测一次冷运行和一次热运行；这足以暴露 `auto` 在多年全市场规模漏选前两项 CUDA 的缺口，但单次热测还不足以把速度倍率作为跨设备永久规则。需要交错顺序复测并验证多指标共享 GPU 批次。完整机器可读结果在 `real_cos_factor_batch_20260927.json`。复现命令：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_factor_batch \
  --days 0 --assets 5500 --repeats 1 \
  --output quant_evaluator/docs/benchmarks/real_cos_factor_batch_20260927.json
```
