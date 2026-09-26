# 真实 COS 因子完整报告适配层 A/B：F8（2026-09-27）

对 manifest SHA256 `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864` 绑定的 8 个已验证研究因子执行完整 `evaluate_report_batch`，面板形状为 **2586 日 × 5461 股 × 8 因子**，日期从 2016-01-04 到 2026-08-25。每个对象上限 64 MiB，总对象上限 256 MiB；本次标签为决策日 t、t+1 执行、`AdjVwap(t+2)/AdjVwap(t+1)-1`。缺失 AdjVwap 分区为 0。因子和标签有限值比例分别为 77.11% 和 77.39%。

方向只用截至 2018-06-30 的 607 个训练交易日推断，报告按 10 组、每组至少 20 只股票、至少 20 个 IC 期计算，单边佣金率为 0.0001（1 bp）。下表是独立顺序子进程中的冷、热完整报告耗时；RSS 包括 fork 继承的只读输入。

| 后端 | 冷 | 热 | 子进程峰值 RSS | 实际后端 / IC 口径 |
|---|---:|---:|---:|---|
| 精确 CPU | 35.622 s | 30.911 s | 13.37 GiB | `cpu` / `exact_float64` |
| Numba | 18.791 s | 17.753 s | 8.19 GiB | `numba` / `numba_float64_ulp_variant` |
| CUDA strict | 20.830 s | **16.268 s** | 12.12 GiB | `cuda` / `cuda_float64_ulp_variant` |
| auto | 18.802 s | 17.278 s | 8.23 GiB | Numba / `numba_float64_ulp_variant`，`measured_report_shape_numba` |

对拍以精确 CPU 为参考。Numba、CUDA strict、auto 各自都对全部 8 个因子逐项比较了完整 `ReportEvaluation` 的 20 个字段（160 个字段值/后端），容差 `rtol=1e-8, atol=1e-10`；三组 `failed` 均为空。字段包括方向、RankIC 序列及统计量、分组收益与净值、多空收益与净值、夏普、年化收益、累计收益、最大回撤、胜率、有效收益期数和佣金率。最大 RankIC 绝对差分别为 Numba `2.22e-16`、CUDA `1.67e-16`、auto `2.22e-16`；最大分组收益绝对差分别为 `0`、`3.47e-16`、`0`。

训练得出的方向为：

| 因子 | 方向 |
|---|---:|
| `weekly_db5616cd85a477b3` | -1 |
| `weekly_4cbd7ca6dccc61dc` | -1 |
| `weekly_c83002810f9e5643` | -1 |
| `weekly_d4cc0b4a9424e834` | -1 |
| `weekly_23629be243c54739` | -1 |
| `weekly_7632208952b859e5` | -1 |
| `weekly_6f27ea3106cc03c0` | -1 |
| `weekly_5c866fa073382822` | +1 |

因此，F8 的 `auto` 确实选择了 Numba CPU，但本次热运行中 CUDA strict 快 1.01 s（约 5.7%）；冷运行则 Numba 快 2.04 s（约 9.8%）。按完整报告热耗时比较，CUDA strict 是本次最快后端；按冷耗时比较，Numba 最快。auto 的 Numba 路由仍比精确 CPU 热运行快约 1.8 倍，且峰值 RSS 低约 5.1 GiB，但不能据本次单次冷/热样本断言跨机器或长期吞吐排名。

完整来源 SHA、逐后端/逐字段差异摘要、计时和 RSS 见[机器可读结果](real_cos_report_f8_20260927.json)。运行只读加载；没有保存原始因子面板、标签、报告数组或生产因子。来源为研究链路数据，不代表上游 PIT 或可投资性认证。

复现命令（只向指定路径写入小型 JSON 摘要，不写原始面板）：

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m factor_engine.scripts.benchmark_real_cos_report_adapter \
  --factors 8 --days 0 --assets 5500 \
  --manifest-sha256 b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864 \
  --max-object-mib 64 --max-total-mib 256 \
  --output quant_evaluator/docs/benchmarks/real_cos_report_f8_20260927.json
```
