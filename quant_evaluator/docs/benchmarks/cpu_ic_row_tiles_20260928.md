# CPU Spearman IC 行块内存 A/B（2026-09-28）

本实验只比较 `compute_daily_ic(..., method="spearman", backend="exact")` 的
旧版整面板秩数组与新版 128 个 `(交易日, 因子)` 行一块的计算。旧版源码固定为
`9f16b217f6739063278c18771762204557a98abc`，复跑脚本为
`quant_evaluator/scripts/benchmark_cpu_ic_row_tiles.py`。输入由固定随机种子生成：
600 日 × 2000 股 × 12 因子，float64，因子 8%、标签 4% 的随机缺失，
并显式传入 validity mask；无需 COS 或外部数据。

在 server-c 同一 Python 环境中以 `old, new, new, old` 四个独立进程交错运行，
OpenBLAS/OMP/MKL 均限制单线程。每轮包含输入构造后的计算耗时和该进程 RSS
高水位；输出数组的原始字节与计数共同计算 SHA256。

| 路径 | 两轮计算耗时 | 两轮 RSS 高水位 | 输出 SHA256 |
| --- | ---: | ---: | --- |
| 旧版 | 1.601、1.577 秒 | 1,655,756、1,656,104 KiB | `ca03cb180670d3cd8ef333c45baff5d42fc8afe24990dee9b475aca92e8aa943` |
| 新版 | 1.332、1.358 秒 | 524,916、524,960 KiB | 同上 |

此样本新版内存峰值约为旧版的 32%，耗时约少 16%。另有
`tests/metrics/test_daily_ic_vectorized_equivalence.py` 对带 ties、NaN、
validity、常数列及跨多个 128 行 tile 的输入，与历史逐行参考实现做逐位等价。
这是中等规模合成样本的证据，不是 F32 全市场生产性能认证；后者仍需在安全
内存余量下进行完整公开入口 A/B。
