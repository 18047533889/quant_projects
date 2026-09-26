# QuantEvaluator 公共入口 CPU/CUDA A/B（2026-09-26）

本报告只评估 `evaluate()` 公共入口的性能与数值对拍，不代表已启用自动路由，也不证明所有指标在所有输入下正确。正式工作树：server-c `/home/sunhaiwei/quant_projects`。未 commit、push 或部署。

## 数据和测法

- 来源：已注册的 `ashare_stock_daily_adj` 调整行情，经现有 DataAccess 和配置的本地 COS 镜像读取。原始 3,714,828 行、723 个日分区；整理后为 2023-09-01 至 2026-08-27、701 个交易日、5,314 只股票、2 个滞后价格信号因子。标签为次日调整 VWAP 收益。这是真实行情轴和真实信号/标签，但不是实盘交易回测。
- 先导全量：公共入口 38 个 CPU/CUDA 可达指标，每指标独立进程、先冷启动再 2 次热调用；38/38 可执行且数值对拍通过。原始记录：[全量 JSON](./public_backend_routes_20260926_full.json)。
- 针对性复测：挑首轮优势约 ≥1.3 倍的 17 项和 CUDA 明显落后的 `factor_turnover_rate`，共 18 项。各后端单独进程，1 次冷启动 + 3 次热调用；指标之间交替 CPU→CUDA / CUDA→CPU 顺序，90 秒进程超时；结果取热调用中位数。原始记录：[交错复测 JSON](./public_backend_routes_20260926_interleaved.json)，脚本为 `quant_evaluator/scripts/benchmark_public_backend_routes.py`，使用 `--alternate-order --repeats 3`。
- 对拍同时检查形状、有限值位置、数值（rtol=1e-8, atol=1e-10）、counts、valid mask、每因子的有效性/观测数/单位、provenance observation counts、输入因子哈希，以及严格 CUDA 路由确实触达 GPU。复测 18/18 通过，最大绝对数值差 8.88e-16。

## 大样本下稳定胜出的单指标

下表是交错复测的热调用中位数。16 个 CUDA 候选不仅中位数更快，且各自 3 次热调用中最慢的 CUDA 仍快于最快的 CPU。倍数为 CPU/CUDA，越大表示 CUDA 越快。

| 指标 | CPU ms | CUDA ms | CUDA 加速 |
|---|---:|---:|---:|
| `coverage` | 656.5 | 434.6 | 1.51× |
| `daily_quantile_monotonicity_rate` | 1522.8 | 482.6 | 3.16× |
| `daily_quantile_monotonicity_series` | 1019.9 | 489.8 | 2.08× |
| `ic_ir` | 1489.3 | 611.1 | 2.44× |
| `ic_median` | 1488.5 | 621.2 | 2.40× |
| `ic_std` | 1543.6 | 648.2 | 2.38× |
| `pearson_ic` | 673.5 | 481.4 | 1.40× |
| `pearson_ic_series` | 690.7 | 510.5 | 1.35× |
| `pearson_ic_std` | 675.5 | 501.2 | 1.35× |
| `quantile_monotonicity` | 994.0 | 532.6 | 1.87× |
| `quantile_returns_daily` | 1016.2 | 498.2 | 2.04× |
| `quantile_returns_full` | 1002.3 | 508.6 | 1.97× |
| `quantile_spread` | 1531.6 | 506.3 | 3.03× |
| `rank_ic` | 1509.7 | 626.9 | 2.41× |
| `rank_ic_series` | 1500.8 | 621.8 | 2.41× |
| `turnover` | 967.1 | 527.3 | 1.83× |

`factor_turnover_rate` 应优先 CPU：851.6 ms 对 CUDA 1259.0 ms，CPU 约快 1.48 倍，3 次热调用区间也未重叠。`pearson_ic_ir` 本轮 CUDA 527.5 ms 对 CPU 661.8 ms，但优势从首轮 1.43× 降到 1.25×，不进入保守 CUDA 白名单。其余组合、日历/滚动风险和暴露度指标的首轮差距小，暂保留 CPU；不能把单轮 2 次的薄优势视为确定收益。

## 路由边界与限制

- 上述选择只适用于本机这类全量股票、多年历史、大样本的已对拍 profile。短历史、少股票、不同因子数、不同设备、显存压力、并发负载或不同数据分布均需重新测量。共享机器负载会使毫秒级差异不稳定。
- 单指标加速不等于多指标混合请求加速。`evaluate()` 内有输入准备、共享计算和传输开销；自动路由须对整批成本做估计与 A/B，而不能简单按指标拆分后端。
- 组合收益由真实行情和滞后信号构造的是研究诊断 `ProbePortfolioArtifact`，并非经过交易执行认证的实盘收益。暴露度使用真实时间/股票轴上的确定性合成 6 风格 `ExposurePanel`，不能据此宣称真实风险数据的性能或正确性。日历风险基于观测交易日快照。
- CPU/CUDA 的 `config_hash` 与 artifact provenance 可相同，但因浮点末位差，`ScalarMetricArtifact` 内容哈希可不同。未来自动路由须记录实际后端、profile/决策版本和输入形状，并审查缓存/证据身份语义，不应把跨后端 artifact 哈希当成恒等。
- 测试覆盖的是这组数据、参数、容差和路径；未证明极端缺失值、不同分位数参数、所有投资组合定义或所有硬件上的正确性。上线前还需针对路由实现跑回归与混合请求 A/B。
