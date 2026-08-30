---
name: quant-data-quality
description: 量化数据质量审计与清洗 — 复权口径一致性（audit_adj_consistency）、缺失/停牌/涨跌停处理、样本权重、面板对齐、单位陷阱（Return 1/10000）、tradable 掩码。当任务涉及"数据质量、复权检查、缺失值、停牌、涨跌停、对齐、清洗、audit"时先读此 skill。
version: 1.0.0
---

# 量化数据质量（审计 + 清洗）

**前提**：已读 quant-platform-workflow（口径硬约束）+ debug-pit-leakage（泄漏联动）。

## 先跑现成 audit 脚本（别手查）

```bash
cd /home/sunhaiwei/quant_projects
# 复权口径一致性（label 对拍：未复权 vs 后复权）
.venv/bin/python lightgbm_qs/scripts/audit_adj_consistency.py
# PIT / look-ahead / period policy
.venv/bin/python scripts/audit_pit_and_period_policy.py
.venv/bin/python scripts/audit_r28_lookahead_static.py
# 算子数学契约 / 语义保持
.venv/bin/python scripts/audit_operator_math_contract.py
.venv/bin/python scripts/audit_execution_semantic_preservation.py
```

## 复权口径（最高优先级）

- **label/收益一律后复权 AdjVwap vwap-to-vwap**：`jobs/adj_vwap_common.adj_fwd_return()`；因子输入用 `_adj` 数据集 + AdjX 字段（AdjClose/AdjOpen/AdjHigh/AdjLow/AdjVwap）。
- 未复权跳空会被算成真收益（历史实测：未复权 10 日收益 -16.9% vs 后复权 +1.17%）。
- **复权因子跳变日**（除权除息日）检查：`ashare_stock_daily_adj` 里当天收益不应出现异常尖峰；`Factor` 字段分段常数。
- 除权除息日公司行为：vectorbt_qs accurate 回测依赖 `StockDividend`（CashDividend/StockDividend/StockTransfer），缺表/缺 schema 直接失败。

## 单位陷阱（必查）

- `ashare_stock_daily.Return` 单位是 **1/10000**（bps），`/10000` 才得到日收益（riskfolio_qs 规则 `HISTORICAL_RETURN_RULE = "StockDailyBar.Return / 10000"`）。
- `specific_var` 是**方差**不是波动率（riskfolio_qs 输入），且必须 >0、无 inf。
- `factor_lake` / `factor_lake_staging` 表因子值若来自不同口径（未复权混后复权）→ 评估前先合并清洗（`merge_all_factors.py` 真源）。

## 缺失 / 停牌 / 涨跌停

- 停牌日：`IsSuspend` 列；买入/卖出均应拒（回测层），因子值处理参考 `ashare_stock_status`。停牌标的权重冻结（w_i=w_{i-1}）。
- 涨跌停：涨停买/跌停卖拒单；`limit_check_mode=strict` 需要原始 High/Low；回测层 `max_participation_rate` 控制单股订单容量（V2 冻结 10%）。
- 面板 reindex 后 NaN 位置漂移 → MultiIndex (date, asset) 排序一致性检查（`factor_engine` vs pandas 对拍）。
- 缺失行在 riskfolio_qs 里视为不可交易（`tradable` 布尔矩阵），优化层自动冻结。

## 样本权重 / 覆盖度

- 覆盖度检查用 quant_evaluator metric：`factor_coverage` / `return_coverage` / `joint_coverage` / `coverage` / `coverage_stability`。
- riskfolio_qs 默认门禁：`min_alpha_coverage=0.80`、`min_benchmark_coverage=0.95`、`min_exposure_coverage=0.95`，低于直接 fail_fast。
- 因子上新样本（新股/新数据段）时，检查覆盖率随时间衰减（`coverage_stability`），避免"最近缺数据导致因子假失效"。

## 常见数据问题 → 修法

| 现象 | 病因 | 修法 |
|---|---|---|
| 收益异常尖峰 | 未复权/复权跳变 | 换 `_adj` 数据集 + AdjX 字段 |
| label 比真值高一个量级 | Return 单位 1/10000 没除 | `/10000` |
| 停牌日被计为亏损 | 未 mask IsSuspend | 剔除/冻结 |
| 因子 OOS 突然失效 | 新样本覆盖率掉 | 看 coverage_stability，补数据 |
| 回测拒单爆炸 | 权重给了停牌/涨停标的 | 优化层加 tradable 掩码 |

## 禁止

- ❌ 用未复权 `ashare_stock_daily` 算收益/因子（口径硬约束）
- ❌ 忽略单位直接 /10000 或不除就进模型
- ❌ 把停牌/涨跌停日的"价格不变"当真实收益
- ❌ 报告覆盖度不查 coverage metric
