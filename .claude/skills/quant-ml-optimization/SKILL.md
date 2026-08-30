---
name: quant-ml-optimization
description: 组合优化与资产配置的正确方式 — 必须用 riskfolio_qs（OptimizationPipeline / BarraPrecomputedAdapter / CLI）+ vectorbt_qs 回测（run_backtest / portfolio_report），禁止手写 cvxpy QP 或换手 blend。当任务涉及"组合优化、资产配置、风险平价、均值方差、指数增强、目标权重、仓位生成、持仓优化、回测绩效"时��读此 skill。
version: 1.0.0
---

# 组合优化与资产配置（riskfolio_qs + vectorbt_qs）

**前提**：已读 quant-platform-workflow + quant-ml-modeling（预测信号来源 = lightgbm_qs 已入选因子 / ML 模型 OOS 预测）。

## 链路图（唯一正确调用链）

```
lightgbm_qs 预测/信号 → riskfolio_qs（OptimizationPipeline，出 target_positions）→ vectorbt_qs（run_backtest，出净值/Sharpe/报告）
```

**三条并行的真实链路**（谁也没调谁，别混）：
- `lightgbm_qs/scripts/portfolio_and_backtest.py`：纯 cvxpy QP（Charnes-Cooper max-Sharpe），**不依赖 riskfolio**，直接吃 `predictions.parquet` + `vwap_trad_adj.parquet`。
- `riskfolio_qs`：优化层，吃 alpha 信号 → 出目标权重。
- `vectorbt_qs`：回测/执行层，吃 target_positions 宽表 → 出 vbt.Portfolio。

## 关键事实（先读，避免浪费时间）

- **riskfolio_qs 不是 riskfolio-lib 封装**。v0.2 起直接使用 cvxpy，全库无 `import riskfolio`。
- **没有 HRP / Risk Parity / Black-Litterman / Max Sharpe / CVaR**。只有 3 个凸目标 + topN 规则：
  `meanvar_enhance_index` / `minvar_enhance_index`（→CLI 路由到 precomputed）、
  `meanvar_enhance_barra_precomputed` / `minvar_enhance_barra_precomputed`（消费 B/F/D 因子���式，不展开 N×N）、
  `meanvar_enhance_hist` / `minvar_enhance_hist`（历史协方差）、
  `meanvar_absolute_return`（绝对收益）、`topn_long_only_equal_weight`（兜底）、`topn_long_short_equal_weight`（**draft**）。
  需要 RiskParity/BL → 要么走 `factor_optimizer` 自定义，要么明确告知做不到，不许手写伪实现。
- 场景路由：`index_enhancement` → meanvar_enhance_barra_precomputed；`conservative_index_enhancement` → minvar；`absolute_return` → meanvar_absolute_return；`fallback` → topN 等权。

## 路线 1：CLI（最省事，YAML 驱动，推荐生产用）

`examples/cli/barra_precomputed.yaml`（真实结构）：

```yaml
config_version: 1
adapter:
  type: barra_precomputed
  risk_root: ../../../v1_sbi_fullA        # 含 manifest.yaml(product=barra_lite) 的目录
  benchmark_index: 000905.SH
  minimum_benchmark_weight_coverage: 0.95
  alpha_input_type: expected_return       # score | expected_return
  alpha_horizon_days: 5
  calendar_name: XSHG
  risk_data_lag_periods: 1
  exposure_data_lag_periods: 1
  annualization_factor: 252
  load_factor_returns: false
inputs:
  alpha:
    path: ../../../outputs/Ridge_h5_pred.parquet   # 只接受 alpha 文件（long 或 wide）
    layout: long                                   # long: date_column/asset_column/value_column 必给
    date_column: TradeDate
    asset_column: Symbol
    value_column: pred
    start_date: 2025-12-24
    end_date: 2025-12-24
smoother:
  mode: never
run:
  optimizer_name: meanvar_enhance_barra_precomputed
  data_version_hash: ridge_h5_2025-12-24
  allow_fallback: false
  runtime_overrides:
    tracking_error_cap_annual: 0.03
    turnover_cap: 0.20
    single_name_max: 0.03
    active_weight_abs_max: 0.02
    style_band: 0.10
output:
  directory: output
  overwrite: false
```

```bash
cd /home/sunhaiwei/quant_projects
.venv/bin/python -m riskfolio_qs optimize -c riskfolio_qs/examples/cli/barra_precomputed.yaml --overwrite
# 或 run_from_config(config_path, output_dir=..., overwrite=...) 脚本内调用
```

产物���`target_positions.parquet` / `trades.parquet` / `summary.parquet` / `metadata.json` / `run_manifest.yaml`。

**CLI 坑**：不接受成本类 override（`default_linear_cost_bps/default_impact_cost/impact_cost_penalty/enable_impact_cost` 直接报错，CLI 固定单边 5bps）；只有 alpha 一个外部输入文件。

## 路线 2：Pipeline + BarraPrecomputedAdapter（Python API，指增推荐）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from riskfolio_qs.adapters import BarraPrecomputedAdapter
from riskfolio_qs.runners.pipeline import OptimizationPipeline
from riskfolio_qs.smoothers.signal_smoother import SignalSmoother

bundle = BarraPrecomputedAdapter(
    risk_root="../v1_sbi_fullA",        # 含 manifest.yaml(product=barra_lite)
    alpha_df=alpha,                     # date × asset
    benchmark_df=benchmark,             # date × asset，列必须与 alpha 完全一致
    prev_positions_df=previous,         # 必须含一个 ≤ 首日 alpha 的 as-of 行
    tradable_df=tradable,               # bool 或 0/1
    alpha_input_type="expected_return",
    alpha_horizon_days=5,
    calendar_name="XSHG",
    risk_data_lag_periods=0,
    exposure_data_lag_periods=0,
    load_factor_returns=False,
    default_linear_cost_bps=5.0,
    minimum_annual_specific_volatility=0.05,
).build_bundle()

output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
    bundle,
    optimizer_name="meanvar_enhance_barra_precomputed",
    runtime_overrides={
        "tracking_error_cap_annual": 0.03,
        "turnover_cap": 0.20,
        "single_name_max": 0.03,
        "active_weight_abs_max": 0.02,   # 传 None 可关闭
        "style_band": 0.10,
    },
    data_version_hash="ridge_h5_v1",
)
w = output.target_positions        # date × asset 目标权重
trades = output.trades             # MultiIndex(date, asset)，列 delta_weight
summary = output.summary           # solve_status/feasible_flag/turnover/预测TE...
```

**验收条件**（来自 tests 断言）：`meta["optimizer_name"]` 匹配；`summary["solve_status"].isin({"optimal","optimal_inaccurate"}).all()`；`summary["feasible_flag"].all()`；`fallback_used=false`。

## 路线 3：Pipeline + data_access（历史协方差指增/绝对收益）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from riskfolio_qs.adapters.data_access_inputs import (
    load_portfolio_inputs, load_historical_market_returns,
)
from riskfolio_qs.core.contracts import InputBundle, OptimizationContext
from riskfolio_qs.runners.pipeline import OptimizationPipeline
from riskfolio_qs.smoothers.signal_smoother import SignalSmoother

pi = load_portfolio_inputs(
    alpha, benchmark_index="000905.SH",
    minimum_benchmark_weight_coverage=0.95, include_benchmark=True,
)
historical = load_historical_market_returns(
    alpha, lookback_days=60, market_data_lag_periods=0,
)
bundle = InputBundle(
    alpha=alpha,
    market=historical.returns,       # 已是日收益（StockDailyBar.Return/10000），不要 pct_change
    benchmark=pi.benchmark,
    tradable=pi.tradable,
    prev_positions=pi.benchmark.iloc[[0]].copy(),
    linear_cost_bps=None,
    metadata=OptimizationContext(
        alpha_is_absolute_return=False,
        benchmark_name="000905.SH",
        calendar_name="XSHG",
        alpha_input_type="score",
        alpha_horizon_days=1,
        market_input_type="return",  # 关键：已是收益就不能再用 price（会二次差分）
        risk_covariance_units="daily_variance",
        annualization_factor=252.0,
    ),
)
output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
    bundle, scenario="index_enhancement",
)
```

## vwap 后复权口径接入（重要）

**riskfolio_qs 自身不读 `vwap_trad_adj.parquet`**。它内置收益口径是 `ashare_stock_daily.Return/10000`。
要严格用 vwap-to-vwap 后复��口径做优化输入（与评估/训练 label 对齐），从 lightgbm_qs 面板读出后手动喂：

```python
import pandas as pd
vwap = pd.read_parquet('/home/sunhaiwei/quant_projects/lightgbm_qs/data/panel/vwap_trad_adj.parquet')
# 收益 = vwap.pct_change()，作为 market 传入，并把 market_input_type 设 "return"
```

## 回测与报告（vectorbt_qs）

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report, compare_reports
from vectorbt_qs.mvp.visualization import build_nav_curves, export_plotly_dashboard

pf = run_backtest(
    "ashare", w,                      # w = riskfolio_qs 输出的 target_positions
    config={
        "init_cash": 10_000_000.0,
        "execution_mode": "accurate", # accurate=真实约束；fast=无摩擦因子研究
        "costs": {"commission": 0.00025, "stamp_tax": 0.0005,
                  "transfer_fee": 0.00001, "minimum_commission": 5.0},
        "limit_check_mode": "strict",
        "benchmark_index": "000300.SH",   # 000300/000905/000852
        "max_participation_rate": 0.10,
    },
)
stats = portfolio_report(pf)          # 252 交易日口径（别用裸 pf.stats() 365 天）
nav = build_nav_curves(pf)            # strategy_nav / benchmark_nav / excess_nav
export_plotly_dashboard(pf, "output/portfolio_dashboard.html", title="Portfolio")
# 批量对比：compare_reports({"策略A": pf, "策略B": pf})
```

## 优化器输出 → 回测的衔接细节

- `riskfolio_qs.target_positions` 已是 date × asset 宽表，**可直接**给 `run_backtest`（列名须带交易所后缀 `000001.SZ`，index 是交易日）。
- A 股只做多：权重非负、每行和 ≤ 1（超了 run_backtest 直接 raise）。
- 未选中标的**填 0** 而非 NaN（NaN = 保持持仓不动，会导致敞口叠加）。
- 真实交易约束（涨跌停拒单/整手/分红送转/容量）在 **vectorbt_qs 回测层**；优化层只有 `tradable` 冻结 + 线性成本。别指望优化器处理涨跌停。

## runtime_overrides 白名单（约 25 个，别传白名单外 key）

`top_n`（注意不是 topn_n）、`single_name_max`、`active_weight_abs_max`、`turnover_cap`、`turnover_penalty`、`industry_neutral_mode`、`style_neutral_mode`、`style_band`、`mcap_band`、`alpha_weight`、`risk_weight`、`impact_cost_penalty`、`enable_impact_cost`、`solver_name`、`max_iters`、`solver_tol`、`on_solve_failure`、`soft_cost_scale`、`score_return_scale_bps`、`default_linear_cost_bps`、`default_impact_cost`、`hist_cov_method`、`hist_cov_halflife_days`、`hist_cov_min_observations`、`enforce_tracking_error_cap`、`tracking_error_cap_annual`。

## InputBundle 格式硬要求

- **全部宽表**：行 = DatetimeIndex（单调增无重复），列 = 资产代码，**列必须与 alpha 一字不差（含顺序）**，否则 validator 抛 `columns mismatch`。
- `F` = MultiIndex(date, asset) × factor；`factor_cov` = MultiIndex(date, factor_i) × factor_j；`style` = MultiIndex columns (factor, asset)。
- `tradable` 布尔矩阵，缺失行视为不可交易；`specific_var` 是**方差**（非波动率）且必须 >0；`benchmark` 每行和 = budget(1.0)。

## 禁止

- ❌ 手写 cvxpy QP / Charnes-Cooper / 换手 blend 绕过 riskfolio_qs（除已有 `portfolio_and_backtest.py` 那套冻结管线外）
- ❌ 承诺 HRP / Risk Parity / Black-Litterman / CVaR / Max Sharpe（riskfolio_qs 没有，别编 API）
- ❌ 手写 Sharpe/VaR/回撤（用 `portfolio_report` / `quant_evaluator` metrics）
- ❌ 未复权收益进优化器（vwap 后复权口径见上；data_access 路线用 `StockDailyBar.Return/10000` 时勿再 pct_change）
- ❌ 把目标权重矩阵用 NaN 表示空仓（填 0）
