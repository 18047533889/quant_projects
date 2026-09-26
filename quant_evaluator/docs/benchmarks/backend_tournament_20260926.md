# IC kernel backend tournament, 2026-09-26

This measures the public compute_daily_ic kernel API, not the higher-level
evaluate() facade, whose CPU/CUDA routing differs. The 1250-day, 5000-stock
inputs are seeded synthetic panels with 4% missing factors, 3% missing labels,
ties, and a weak common signal. They are not COS or production returns.
Values are float32; calls include validity masks, layout conversion,
GPU transfers, and output synchronization.

Script: scripts/benchmark_backend_tournament.py. Results:
backend_tournament_20260926_baseline.json (before Polars correction),
backend_tournament_20260926_after_small.json, and
backend_tournament_20260926_after_full.json. The after-full run started
2026-09-26 09:02:44 UTC. Script SHA256:
84d10484281ff574affe817ff0ce3d642e871c989a25808936ac5e56156e174c.
Polars backend SHA256:
38213cf41abbafef907da1906bb5f7ac4df15f3cf05cae468308c8472d6d9265.

Warm median elapsed milliseconds, three calls after one cold call, L20 GPU:

| Shape / method | exact | Numba | Polars | GPU |
|---|---:|---:|---:|---:|
| 1250 x 5000 x 1 Pearson | 72.28 | **9.54** | 139.85 | 51.17 |
| 1250 x 5000 x 1 Spearman | 603.40 | **59.27** | 256.64 | 90.88 |
| 1250 x 5000 x 8 Pearson | 820.61 | **219.41** | 2191.19 | 806.04 |
| 1250 x 5000 x 8 Spearman | 5026.31 | **562.61** | 3785.06 | 1106.98 |

All 16 after-full outputs pass against exact: identical finite masks and
valid counts, and IC values within rtol 1e-8 / atol 1e-10. Before the Polars
float64 correction, every tested Pearson Polars case failed that tolerance.
The baseline file may span the concurrent source edit and is evidence of the
discovered mismatch, not a frozen-version speed comparison. At F=8, Polars
peak process RSS was about 9 GiB; Numba about 0.8 GiB. Cold times and input
construction times are recorded separately in JSON.

Machine snapshot during the after-full run (2026-09-26 09:03:59 UTC):
load average 11.72 / 7.05 / 5.49, 50 GiB available RAM, GPU 7779 MiB used,
37679 MiB free, 0% instantaneous utilization. Other active jobs included
new_mining_intake.py (~133% CPU), new_mining_publish.py (~100%),
run_daily_gfn_24h_live.py (~97%), and another model worker (~96%). The
benchmark Polars worker itself used multiple CPU cores (~908% at snapshot).
Shared-host contention makes millisecond differences unsuitable as a permanent
dispatch threshold; the parity and order-of-magnitude gaps are useful now.

## 真实注册行情档

另用已配置的 DataAccess 注册数据集 ashare_stock_daily_adj，仅读取镜像中实际
存在的日期分区。范围为 2023-09-01 至 2026-08-27：723 个日分区，
3,714,828 行、5,314 只不同股票；长于 3 自然日的相邻分区间隔有 18 处。
读取分成 155 个短窗口，每次最多 7 个文件、4 万行、4 MiB 返回量，全部仍经
DataAccess 授权和预算校验。没有把未取得的分区填成观测值，也没有修改 DataAccess。

为了检验 IC 内核，构造两个简单研究信号：决策日 t 的 5 日和 20 日滞后动量只用
t-1 及更早的 AdjVwap；标签用 t 至下一实际观测交易日的 AdjVwap 收益。
前置窗口和末端标签裁剪后，指标输入为 T=701、N=5314、F=2。它们只是测试信号，
没有 PIT、可交易性或生产因子认证；节假日后的标签跨自然日。

结果文件：backend_tournament_20260926_real_registered.json，开始于
2026-09-26 09:13:12 UTC；脚本 SHA256 为
bdd7c95f1b52d7f9016eb58ec849759c3362c74913630dcbb9fdaac0812a5d6e，
Polars 源码 SHA256 与上表相同。DataAccess 读取及构造合计 5.77 秒，只做一次，
后续各后端冷/热调用计时不含这 5.77 秒。

| 真实行情 IC | exact | Numba | Polars | GPU |
|---|---:|---:|---:|---:|
| Pearson 热运行中位数，毫秒 | 161.9 | **95.0** | 284.2 | 333.7 |
| Spearman 热运行中位数，毫秒 | 986.9 | **165.2** | 491.1 | 411.4 |

四后端在两种 IC 上全部通过 finite mask、valid_counts 严格一致及
rtol 1e-8 / atol 1e-10 数值对拍。此档 Numba 最快，且过程峰值 RSS 约
0.58 GiB；Polars 约 2.2 至 2.5 GiB。读取后机器快照
2026-09-26 09:14:06 UTC：load 5.79 / 6.11 / 5.73，可用 RAM 50 GiB，
GPU 占用 9050 MiB、空闲 36408 MiB，瞬时利用率 0%；
同时运行的 mining intake、daily GFN 和另一模型任务分别约用
130%、97%、96% CPU。这些是共享机器上的单轮测量，暂不据此固定自动路由阈值。

The optional real sample uses the existing bounded DataAccess loader from
factor_optimizer/examples/cos_batch_audit.py (2 factors, 500 dates, 256
stocks, at most 8 MiB per factor). Its fixed manifest currently raises
ValidationError: exact object metadata is missing or mismatched before any
metric is computed. The failed attempt is recorded in
backend_tournament_20260926_real_cos.json; no alternate COS URI was used.
