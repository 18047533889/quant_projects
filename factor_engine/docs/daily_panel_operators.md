# Daily panel 通用算子

本组算子只包含能够逐股票、逐交易日生成数值的同形 panel 运算。所有时序窗口均为
`[t-window+1, t]`，不使用负向 shift、centered rolling、bfill、未来窗口或全样本统计。

总体统计检验和绩效汇总（例如 t 检验、KS、ADF、总体 Sharpe、总体最大回撤、
regression summary）仍属于 `research_operators`，不得进入 daily DSL。

## 条件滚动

| 算子 | 分类 | 核心语义 |
|---|---|---|
| `ts_count_if(condition, window, min_periods=1)` | `time_series_condition` | True 计 1，False 计 0，缺失不进入有效条件观察数 |
| `ts_sum_if(x, condition, window, min_periods=1)` | `time_series_condition` | 只累计 condition=True 且 x 有效的观察；无匹配返回 NaN |
| `ts_mean_if(x, condition, window, min_periods=1)` | `time_series_condition` | False 不进入分子或分母 |
| `ts_std_if(x, condition, window, min_periods=2, ddof=1)` | `time_series_condition` | 条件样本标准差；有效数须达到 `max(min_periods, ddof+1)` |

## 事件状态

| 算子 | 分类 | 核心语义 |
|---|---|---|
| `ts_last_if(x, condition, window)` | `time_series_event` | 返回窗口内最近一次条件成立且 x 有效时的值 |
| `ts_days_since(condition, max_lookback=None)` | `time_series_event` | 当前日 True 为 0，按交易数据行计数 |
| `ts_true_streak(condition)` | `time_series_event` | 连续 True 长度；False 或 NaN 重置为 0 |

## 横截面分层与中性化

| 算子 | 分类 | 核心语义 |
|---|---|---|
| `cs_bucket(x, buckets=10, ascending=True)` | `cross_sectional` | 每日平均百分位排名映射到 `1..buckets`；并列进入同一桶 |
| `cs_multi_resid(y, x1, x2, ..., add_intercept=True, min_obs=None)` | `cross_sectional_regression` | 每日一次多元 OLS；联合过滤缺失，奇异或病态矩阵返回 NaN |
| `cs_wls_resid(y, x, weight, add_intercept=True, min_obs=5)` | `cross_sectional_regression` | 每日严格正权重 WLS；不静默退化为 OLS |

`cs_multi_resid` 是一次联合回归，不由多次顺序 `cs_resid` 拼接，因此没有暴露变量顺序依赖。

## 报告期

`period_lag(x, period_id, periods=1)` 属于 `fundamental_period`。它按截至当前交易日已经出现的
报告期顺序滞后，而不是按固定交易日数滞后。同一报告期内结果保持稳定；`period_id` 缺失返回
NaN。输入必须先按真实公告时间完成 PIT 对齐，引擎不会把报告期末日期当作可用日期。

## 滚动统计信号

| 算子 | 分类 | 核心语义 |
|---|---|---|
| `ts_regression_tstat(y, x, window, min_periods=None, add_intercept=True)` | `time_series_regression` | 滚动回归斜率除以其标准误 |
| `ts_trend_tstat(x, window, min_periods=None)` | `time_series_regression` | 窗口内重新生成时间序号后的趋势斜率 t 值 |
| `ts_partial_corr(x, y, z, window, min_periods=None)` | `time_series_regression` | 使用 x/y/z 共同有效样本，控制 z 后的滚动偏相关 |
| `ts_max_drawdown(x, window, min_periods=2)` | `time_series_risk` | 正价格或净值的窗口内最大回撤，输出范围通常为 `[-1, 0]` |
| `ts_nth_value(x, window, n=1, order="largest", min_periods=None)` | `time_series_order` | 第 N 大或第 N 小有效值 |

自由度不足、零方差、零标准误、非正价格、奇异设计矩阵等无定义情形统一返回 NaN。

## 后端

以上 16 个 canonical 均提供：

- Pandas panel runtime；
- 显式 Polars panel runtime；
- Polars long-table 编译与执行；
- DuckDB SQL 下推。

DuckDB 和 Polars 的实现必须通过 `tests/operators/test_daily_panel_ops.py` 的参考结果、
缺失边界和未来数据突变测试后，才可进一步申请 production fast-path 认证。
