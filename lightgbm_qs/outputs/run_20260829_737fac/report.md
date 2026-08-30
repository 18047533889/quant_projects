# 737 因子全链路结果（2026-08-29 13:00）

## 版本
- **候选池**: 2262 LQTP + 915 本地(fm247/fmqa/cogfull/cogshort/cogneutral/optfac/factmat) = 3177
- **筛后**: rank_ic>0.015（vwap-to-vwap 10日靶，flip 后），**不去重** → 737 进模型
- **训练**: LightGBM 滚动前向 walk-forward（首切 2020-01，每 3 月，purge 10 交易日）
- **组合**: topk=30, max_weight=0.05, turnover≤0.30, tc=5bp, Vwap T+1 执行
- **回测**: vectorbt_qs accurate 模式, 1592 交易日 2020-01-23 .. 2026-08-24

## 结果（vs 中证等权基准）
| 指标 | 策略 | 基准 |
|---|---|---|
| 总收益 | **458.3%** | 14.6% |
| 超额收益 | 387.1% | — |
| Sharpe | 1.209 | — |
| 超额 Sharpe | 1.455 | — |
| 最大回撤 | -31.1% | — |

## 文件
- equity_curve.png / excess_curve.png / benchmark_curve.png
- equity_curves.csv (date, strategy_nav, benchmark_nav, excess_nav, *_ret)
- backtest_final_nav_curves.parquet（原始 NAV）

## 后续（自动进行中）
- LQTP 补算剩余 ~511 个因子完成后将自动重跑全链（auto_chain.sh）
