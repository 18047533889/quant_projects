---
name: quant-risk-management
description: 组合/因子风险管理的正确方式 — 用 quant_evaluator 注册的风险 metric（max_drawdown/cvar_95/cvar_99/ic_ir/hac_tstat/turnover）做风险度量，用 riskfolio_qs 的 tracking_error/行业中性/集中度约束做事前风控，用 vectorbt_qs 的 portfolio_report 做回测层风险复核。禁止手写 VaR/Sharpe/回撤。当任务涉及"风险管理、回撤、VaR、CVaR、跟踪误差、行业中性、集中度、风险预算、尾部风险"时先读此 skill。
version: 1.0.0
---

# 量化风险管理（quant_evaluator + riskfolio_qs + vectorbt_qs）

**前提**：已读 quant-platform-workflow（收益口径）+ quant-ml-optimization（优化/回测链路）。

## 三层风险架构（对应三层库，别混用）

| 层 | 库 | 管什么 | 入口 |
|---|---|---|---|
| 因子层 | quant_evaluator | 因子风险特征（IC 稳定性/回撤/换手/尾部） | `quant_evaluator.registry.metrics._REGISTRY` |
| 组合层 | riskfolio_qs | 事前风控（TE 上限/中性/集中度/换手） | `OptimizationPipeline` + `runtime_overrides` |
| 执行层 | vectorbt_qs | 事后实盘口径（真实回撤/费用/拒单） | `portfolio_report(pf)` |

## 因子层风险度量（quant_evaluator 51 个注册 metric，先查 `_REGISTRY`）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from quant_evaluator.registry.metrics import _REGISTRY
ids = sorted(_REGISTRY.to_dict())     # 51 个，先查再写，别自定义
```

常用风险域 metric（按 domain 选）：
- **回撤域**：`max_drawdown`、`calmar_ratio`、`drawdown_duration`
- **尾部风险域**：`cvar_95`、`cvar_99`、`skewness`、`kurtosis`（**手写 VaR/CVaR 禁止**）
- **换手/成本域**：`turnover`、`turnover_rate`、`turnover_cost`、`turnover_adjusted_ic`、`turnover_stability`（高频因子必须有换手成本意识）
- **稳定性域**：`ic_stability`、`rank_stability`、`subsample_stability`、`coverage_stability`
- **显著性域**：`ic_ir`、`hac_tstat`、`hac_pvalue`（HAC 校正的多重检验，防假发现，见 debug-pit-leakage）
- **集中度域**：`hhi_concentration`、`hhi_effective_n`
- 底层函数：`quant_evaluator.metrics.compute_daily_ic / compute_icir / compute_ic_tstat / assign_quantiles / compute_turnover / compute_coverage`

## 组合层事前风控（riskfolio_qs runtime_overrides）

指增类优化器（meanvar/minvar_enhance_*）默认带硬约束，全在 `runtime_overrides` 里开：

```python
runtime_overrides = {
    "tracking_error_cap_annual": 0.03,   # 主动风险硬约束，默认 3% 年化 TE
    "turnover_cap": 0.20,                # 单边换手上限
    "single_name_max": 0.03,             # 个股最大权重
    "active_weight_abs_max": 0.02,       # 单股主动权重上限（传 None 关闭）
    "industry_neutral_mode": "strict",   # off | strict | band
    "style_neutral_mode": "band",
    "style_band": 0.10,
    "mcap_band": 0.10,                   # 市值中性 band
    "enforce_tracking_error_cap": True,
}
```

求解后**必须检查** `output.summary`（每日期一行）：
- `solve_status` ∈ {optimal, optimal_inaccurate}（`failed` 不能用）
- `feasible_flag` 全 True；`fallback_used` False
- `max_constraint_violation <= 1e-5`
- `predicted_tracking_error_annual`（预测年化 TE，与 cap 对比）
- `tracking_error_cap_violation`（True = 超了，要调参）
- `max_abs_industry_active_exposure` / `max_abs_style_active_exposure`（中性约束执行度）
- `active_share`（主动份额）
- `predicted_portfolio_volatility`

## 执行层风险复核（vectorbt_qs portfolio_report）

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report
pf = run_backtest("ashare", target_positions, config={
    "execution_mode": "accurate",
    "costs": {"commission": 0.00025, "stamp_tax": 0.0005,
              "transfer_fee": 0.00001, "minimum_commission": 5.0},
    "benchmark_index": "000300.SH",
})
stats = portfolio_report(pf)   # 252 交易日口��
```

报告含（真实盘口径）：`Max Drawdown [%]`、`Max Drawdown Duration`、`Sharpe Ratio`（252 重算）、`Calmar Ratio`、`Sortino Ratio`、`Total Fees Paid`、`Win Rate [%]`、`Profit Factor`、`Expectancy`、`Max Gross Exposure [%]`、挂基准后：`Tracking Error [%]` / `Information Ratio` / `Annualized Active Return [%]`。

**拒绝单审计**（accurate 模式）：`pf._qs_execution_log` 里记录 suspend/limit_down/limit_up/cash/volume/price_bound 拒单，是真实可交易性的最诚实信号——如果拒单过多，说明权重矩阵没尊重停牌/涨跌停/流动性，要回优化层加 `tradable` 掩码或调 `max_participation_rate`。

## A 股特有风险清单（必查）

1. **停牌**：`is_suspend` 日买入/卖出均被拒 → 优化层用 `tradable=False` 冻结（w_i=w_{i-1}），避免停牌标的权重漂移。
2. **涨跌停**：涨停买/跌停卖拒单；`limit_check_mode=strict`（整日封板判定，需原始 High/Low）比 `execution` 保守。
3. **整手**：100 股一手，部分减仓整手、清仓允许零股 → 单票最小可交易规模=1手，权重别设太小。
4. **成交量容量**：`max_participation_rate=0.10`（V2 冻结值），超过当日 Volume 10% 的部分 `partial_volume` 排队到下一调仓日 → 高权重小票风险大。
5. **流动性/换手**：日频调仓因子必须看 `turnover_cost` / `turnover_adjusted_ic`，不能只看裸 IC。
6. **尾部**：因子/策略 OOS 段要重算 `cvar_95`/`cvar_99`，收益分布偏厚尾时 Sharpe 会高估。
7. **多重检验**：多因子筛选用 `hac_tstat`/`hac_pvalue` 校正，防"筛出来只是运气"。

## 风险预算思路（方法论，配合 factor_optimizer 实现）

- 组合层：按因子 IC_IR × 换手成本设定每因子权重上界（`alpha_weight`/`risk_weight` 权衡，`soft_cost_scale=0.05` 默认）。
- 用 `factor_optimizer.contracts`（ObjectiveSpec/SearchBudget/create_split_aware_evaluation_fn）做带风控约束的参数寻优，objective 可以同时含 rank_ic（分子）与 max_drawdown/cvar（分母）。
- 回���层对比：**同一权重矩阵**至少跑两档成���（如 2.5bp vs 5bp 佣金）+ 两档基准（000300/000905）验证稳健性（参考 vectorbt_qs `accurate-batch` 冻结 24 组设计）。

## 禁止

- ❌ 手写 VaR / CVaR / Sharpe / Calmar / 回撤（quant_evaluator metrics 或 portfolio_report 都有）
- ❌ 用裸 `pf.stats()`（365 天年化）报告 Sharpe/年化，必须 `portfolio_report`（252 交易日）
- ❌ 忽略拒单日志就把优化结果当"可交易收益"
- ❌ 用未复权收益算任何风险指标
