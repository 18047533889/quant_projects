# 单配置 Accurate Benchmark

`accurate-benchmark` 用于用一套冻结配置快速、可重复地验证一个目标仓位文件。
它与 `standard_accurate_v2` 复用相同的数据预检、准确撮合、公司行动、绩效计算、
可视化和 Barra 暴露分析，但只执行一次交易回测。

## 快速运行

在工作区根目录执行：

```powershell
python -m vectorbt_qs accurate-benchmark --positions=benchmarks\gtja191_alpha191_meanvar_hist\results\gtja191_alpha191_meanvar_hist_v1\target_positions.parquet --barra-root=v2_sbi_mvl_fullA --output-root=outputs\accurate_benchmark --start=2024-01-01
```

可用 CLI 参数：

| 参数 | 是否必填 | 含义 |
|---|---:|---|
| `--positions` | 是 | 目标仓位宽表 parquet |
| `--barra-root` | 是 | Barra 风险模型根目录 |
| `--output-root` | 是 | 回测结果根目录 |
| `--start` | 否 | 起始日期，默认 `2024-01-01` |
| `--end` | 否 | 结束日期，默认使用仓位数据末日 |

单配置不提供 `--workers`：只有一次交易计算，多进程不会缩短执行时间。

## 冻结参数

| 参数 | 固定值 | 口径 |
|---|---:|---|
| `execution_mode` | `accurate` | 原始价格、真实股数 |
| `benchmark_index` | `000852.SH` | 中证 1000 |
| `price_type` | `vwap` | 执行日未复权 VWAP |
| `freq` | `1D` | 日频目标仓位 |
| `slippage` | `0.001` | 单边 0.1%，10 bp |
| `costs.commission` | `0.0005` | 双边佣金 0.05%，5 bp |
| `costs.stamp_tax` | `0.0005` | 仅卖出收取 0.05% |
| `costs.transfer_fee` | `0.00001` | 双边 0.001% |
| `costs.minimum_commission` | `5.0` | 5 元/笔 |
| `init_cash` | `10000000` | 1000 万元 |
| `signal_time` | `close` | 收盘后得到目标仓位 |
| `execution_lag` | `1` | 下一交易日执行 |
| `limit_check_mode` | `strict` | 严格整日封板判断 |
| `lot_size` | `100` | 100 股/手 |
| `max_participation_rate` | `0.10` | 最多占当日成交量 10% |
| `performance_year_days` | `252` | 252 个交易日/年 |
| `risk_free_rate` | `0.0` | 年化无风险利率 0% |

其中 `0.05%` 手续费按佣金理解。买入总比例费用为佣金加过户费；卖出还会增加
印花税。公司行动继续读取 `lqtp_data/StockDividend`，现金分红进入现金，送股和
转增进入真实股数。

## 输出目录

结果目录直接使用仓位文件的父文件夹名：

```text
输入：
benchmarks/某策略_v1/target_positions.parquet

输出：
用户提供的 output-root/
└── 某策略_v1/
    ├── effective_config.json
    ├── performance_stats.csv
    ├── nav_curves.parquet
    ├── equity_curve.png
    ├── portfolio_dashboard.html
    └── exposure/
```

不会追加 `benchmark_index_..._price_type_...` 参数目录。重复使用相同输出根目录和
相同仓位父目录名时，已存在的同名结果文件会被本次回测覆盖。
