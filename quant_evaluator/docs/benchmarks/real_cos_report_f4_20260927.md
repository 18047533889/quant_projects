# 真实 COS 因子完整报告适配层 A/B：F4 / 生产 tile=4（2026-09-27）

使用与 F8 相同的 manifest `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`，按 DataAccess loader 的大小排序规则取其中最小四条已验证因子。面板为 **2586 日 × 5461 股 × 4 因子**，日期 2016-01-04 至 2026-08-25；每个对象上限 64 MiB、总对象上限 256 MiB。加载耗时 30.776 秒，未缺失 AdjVwap 分区。

报告口径与 F8 一致：标签为决策日 t、t+1 执行、`AdjVwap(t+2)/AdjVwap(t+1)-1`；训练方向截至 2018-06-30（607 个交易日）；10 组、每组至少 20 只股票、最少 20 个 IC 期；单边佣金率 0.0001（1 bp）。完整报告冷/热耗时及子进程峰值 RSS 如下，RSS 含 fork 继承的只读面板：

| 后端 | 冷 | 热 | 峰值 RSS | 实际后端 / IC 口径 |
|---|---:|---:|---:|---|
| 精确 CPU | 18.139 s | 15.574 s | 6.70 GiB | `cpu` / `exact_float64` |
| Numba | **8.927 s** | **7.428 s** | 3.55 GiB | `numba` / `numba_float64_ulp_variant` |
| CUDA strict | 10.674 s | 7.797 s | 6.42 GiB | `cuda` / `cuda_float64_ulp_variant` |
| auto | 9.102 s | 7.687 s | 3.56 GiB | Numba / `numba_float64_ulp_variant`，`measured_report_shape_numba` |

方向训练结果四条均为 -1：`weekly_db5616cd85a477b3`、`weekly_4cbd7ca6dccc61dc`、`weekly_c83002810f9e5643`、`weekly_d4cc0b4a9424e834`。Numba、CUDA strict、auto 分别与精确 CPU 对全部四条因子的 20 个 `ReportEvaluation` 字段逐项比较，共 80 项/后端；容差 `rtol=1e-8, atol=1e-10`，三组 `failed` 均为空。最大 RankIC 绝对差均为 `5.55e-17`；最大分组收益绝对差：Numba `0`、CUDA `3.47e-16`、auto `0`。字段清单和逐后端比较记录见[机器可读结果](real_cos_report_f4_20260927.json)。

**生产建议：保留 F4/tile=4 的 auto→Numba 路由。** 本次真实 F4 报告 A/B 中，Numba 冷、热耗时均最快；auto 也确实选择 Numba。CUDA 热运行比 Numba 慢 0.369 秒（约 5.0%），冷运行慢 1.747 秒（约 19.6%），并使用约 2.9 GiB 更多峰值 RSS。auto 热耗时比单独 Numba 高 0.259 秒，但两者走同一 Numba 后端；在单次冷/热测量和共享服务器条件下，这点差距不足以支持修改路由。此前 F8 热运行 CUDA 略快，冷运行 Numba 更快；F8 不改变 F4 生产 tile 的结论。

结果仅支持本机这次真实面板条件下的路由判断，不代表其他机器或重复吞吐排名。运行只读加载，没有写出因子面板、标签、报告数组或生产因子。

复现命令（只向指定路径写小型 JSON 摘要）：

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m factor_engine.scripts.benchmark_real_cos_report_adapter \
  --factors 4 --days 0 --assets 5500 \
  --manifest-sha256 b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864 \
  --max-object-mib 64 --max-total-mib 256 \
  --output quant_evaluator/docs/benchmarks/real_cos_report_f4_20260927.json
```
