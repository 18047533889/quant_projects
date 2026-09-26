# 真实 COS 因子完整报告适配层 A/B（2026-09-27）

通过 DataAccess 的 manifest-bound 研究入口只读加载两条已验证因子；完整重叠面板为 2016-01-04 至 2026-08-25、2586 日 × 5461 股 × 2 因子。因子 SHA256 分别为
`35ad096b62be98ce5766fe3df4e651bd7930ee377683d6282171a6984ed14a59`（`weekly_4cbd7ca6dccc61dc`，5,400,494 字节）和
`674763066b67e4ed032122b51545a183f8a851b2488ff43ba2a81ab25c07df5d`（`weekly_db5616cd85a477b3`，3,828,456 字节）。对应的标签为决策日 t、t+1 执行、`AdjVwap(t+2)/AdjVwap(t+1)-1`；训练方向截至 2018-06-30；十组、每组至少 20 只股票、最少 20 个 IC 日，单边佣金 1 bp。固定方向未传入，两因子由训练窗推得方向均为 -1。

| 完整 `evaluate_report_batch` 后端 | 冷运行 | 热运行 | 子进程峰值 RSS | 实际路径 |
|---|---:|---:|---:|---|
| 精确 CPU | 9.426 s | 8.180 s | 4.03 GB | `exact_float64` |
| Numba CPU | 4.701 s | 3.781 s | 2.74 GB | `numba_float64_ulp_variant` |
| CUDA strict | 5.618 s | 4.017 s | 4.01 GB | `cuda_float64_ulp_variant` |
| 默认 auto | 4.779 s | 3.907 s | 2.74 GB | Numba；`measured_report_shape_numba` |

Numba、CUDA 和 auto 相对精确 CPU 的 **20 个 `ReportEvaluation` 字段 × 2 因子全部通过** `rtol=1e-8, atol=1e-10`，包括方向、整条 IC、分组收益及净值、多空收益及净值、交易成本与各项绩效标量。最大绝对 RankIC 差为 `2.78e-17`；CUDA 分组收益最大差 `3.47e-16`，Numba 为 0。Numba/CUDA 的浮点求和次序可改变最低有效位及内容哈希，因此报告逐因子记录实际后端和 IC 数值口径。完整字段核对与耗时见 [机器可读结果](real_cos_report_adapter_20260927.json)。

每个后端在独立顺序 fork 子进程内执行冷、热各一次；峰值 RSS 含继承的只读输入。输入加载 22.710 秒，父进程峰值 RSS 2.88 GB。服务器有其他 CPU/GPU 作业，单次计时只支持本机本次条件下的选择；T2700–3500 或 N5500–6000 的 auto 路由属于有边界外推，尚未做相同真实数据 A/B。研究来源不构成上游 PIT 或可投资性认证。本测试没有写入原始面板、报告 manifest 或生产因子。

复现命令（只读数据；仅把小型结果 JSON 写到指定路径）：

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m factor_engine.scripts.benchmark_real_cos_report_adapter \
  --days 0 --assets 5500 \
  --output quant_evaluator/docs/benchmarks/real_cos_report_adapter_20260927.json
```
