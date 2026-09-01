# vectorbt_qs — A 股/美股组合回测系统

基于 vendored [vectorbt](https://github.com/polakowo/vectorbt) **1.1.0** 的 A 股/美股
组合回测系统：外层实现数据适配、市场约束、公司行动（现金分红/送转）与统一入口，
两种执行口径（fast 因子筛选 / accurate 真实成交）。

**版本:** 0.4.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs (私有)
**文档索引:** [`docs/README.md`](docs/README.md) ｜ 从零跑通: [`docs/accurate_batch_quickstart.md`](docs/accurate_batch_quickstart.md)

## 两种执行口径

| | Fast | Accurate |
|---|---|---|
| 用途 | 因子筛选（大量因子/分组） | 候选策略复核 |
| 成交单位 | 分数股 | A 股 100 股整手 |
| 现金/持仓 | 按目标权重再平衡 | 逐日维护真实现金与股数 |
| 费用/滑点 | 固定为零 | 方向性费率、最低佣金、滑点 |
| 停牌/涨跌停 | 忽略或目标层近似 | 按真实状态拒单 |
| 公司行动 | — | 除权除息日计入现金分红/送转 |
| 收益口径 | VWAP 信号 → 下一交易日执行 | VWAP / Open 执行，真实股数 |

两者：收盘生成信号 → 下一交易日执行；A 股默认只做多（各行权重和 ≤ 1）。

## 安装与运行

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs.git
cd vectorbt_qs
pip install -r requirements.txt && pip install -e .
# 数据层（推荐）：pip install -e ../data_access
```

```bash
python -m vectorbt_qs list                       # 列内置 benchmark
python -m vectorbt_qs backtest -b B001 --plot    # 跑 benchmark
python -m vectorbt_qs run configs/config.example.yaml
python -m vectorbt_qs batch configs/config.batch.example.yaml   # 参数网格
python -m vectorbt_qs accurate-batch --positions target_positions.parquet \
    --barra-root /path/v2_sbi_mvl_fullA --output-root out --workers 4  # 24 组固定报告
```

Python API：

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

target_weights = pd.DataFrame(data=..., index=pd.DatetimeIndex(...), columns=[...])
pf = run_backtest("ashare", target_weights, config={"init_cash": 10_000_000})
stats = portfolio_report(pf)
```

## 输入格式

目标权重矩阵：index = 交易日、columns = 代码（`000001.SZ`）、values = 权重
（0.05 = 多头 5%、-0.03 = 空头、0 = 平仓、NaN = 维持）。
详见 [`docs/target_positions_format.md`](docs/target_positions_format.md)。

## 关键模块

```
cli.py                # CLI：backtest / run / batch / list / accurate-batch
configs/              # 示例 YAML（accurate/fast/batch profile）
mvp/engine/           # runner、execution（numba 状态机）、batch、profiles
mvp/data/             # data_access 优先适配器、COS fallback
mvp/constraints/      # A 股 / 美股约束规则
mvp/analysis/         # Barra-lite B/f 暴露 + 实验性归因
contracts/            # BacktestRequest/Artifact、ledger、orders、trades DTO
vectorbt/             # vendored vectorbt 1.1.0（源码，无上游 git）
docs/                 # accurate 快速上手、模式、batch、暴露、目标仓位格式
tests/                # 回归测试
```

## 数据需求（accurate）

- `lqtp_data/StockDailyBar`（原始 OHLCV + 复权因子 + 停牌 + 涨跌停）
- `lqtp_data/StockDividend`（现金分红、送转，除权除息日入账）
- `lqtp_data/IndexDailyBar` + `IndexConstituent`（基准净值 / 成分）
- `lqtp_data/StockIndustry`（行业名）
- 可选 Barra-lite 风险模型（`v2_sbi_mvl_fullA`）：exposure/factor_returns/factor_cov/specific_risk

## A 股交易规则覆盖

停牌过滤、涨跌停（涨停不买/跌停不卖）、T+1 可卖、印花税（仅卖出）、佣金+最低佣金、
滑点、100 股整手、现金分红与送转、单股单日成交量 10% 上限。

## 性能

- B001 topN 等权：2294 天 × 5122 股 ≈ 46s
- B006 meanvar：1433 × 300 ≈ 20s；B007 XGB：599 × 5086 ≈ 15s

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（`mvp/data/adapter.py` 行情读取）。
- **被谁调用**：`lightgbm_qs`（`backtest_final_vectorbt.py`：`set_data_root` /
  `run_backtest` / `portfolio_report` / `build_nav_curves`）。

## 相关仓库

- **data_access** — 数据层（优先读路径）
- **riskfolio_qs** — 上游组合优化，产出 `target_positions.parquet`
- **factor_engine / quant_evaluator** — 上游因子计算与评估
