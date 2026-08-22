# vectorbt_qs Benchmark 性能报告

> 生成时间: 2026-07-18 18:36:09
> 环境: Windows, Python 3.11, conda 项目环境

## 测试对象

| 项目 | 值 |
|---|---|
| Benchmark | B001 gtja191_alpha191_topn_v1 |
| 信号源 | GTJA191 Alpha191 |
| 优化器 | topn_long_only_equal_weight |
| 权重矩阵 | 2294 天 × 5122 只 |
| 日期范围 | 2017-01-11 ~ 2026-06-26 |
| 日均持仓 | 50 只 (min=50, max=50) |

## 耗时分析

| 步骤 | 耗时 (秒) | 占比 |
|---|---|---|
| 读取 target_positions | 0.3 | 0.6% |
| 回测 (含行情加载) | 40.0 | 85.8% |
| 绩效报告 | 6.3 | 13.5% |
| 总计 | 46.5 | 100.0% |

## 回测绩效

| 指标 | 值 |
|---|---|
| Start Value | 10,000,000.00 |
| End Value | 263,891.95 |
| Total Return [%] | -97.36 |
| Sharpe Ratio | -1.71 |
| Max Drawdown [%] | 97.64 |
| Total Trades | 93863 |
| Win Rate [%] | 46.46 |
| Expectancy | -103.77 |
| Annual Return [%] | N/A |
| Annual Volatility [%] | N/A |
| Calmar Ratio | -0.45 |

## 系统信息

| 项目 | 值 |
|---|---|
| 行情数据源 | data_access (DuckDB) + lqtp_data 本地 parquet |
| 行情加载列数 | 11 列 (OHLCV + 涨跌停 + 复权因子) |
| 数据量 | ~2300 个 parquet 文件, ~6GB |
| vectorbt 版本 | 1.x (源码) |
| 约束层 | A股停牌过滤 + 涨跌停价格裁剪 + 费率矩阵 |
