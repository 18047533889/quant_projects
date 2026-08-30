---
name: debug-pit-leakage
description: 量化代码 debug 专用 — PIT/look-ahead/前视泄漏/复权口径错误的排查清单与修复模式，配合项目既有 audit 脚本。当因子结果"好得离谱"、IC 异常高、回测收益不真实、或任务提到"检查泄漏/look-ahead/PIT/口径"时使用。
version: 1.0.0
---

# 量化代码 Debug：泄漏与口径排查

**触发场景**：IC 异常高（>0.1 截面 rank_ic 要怀疑）、回测曲线过于平滑、因子在 OOS 失效、"好得离谱"的结果。历史教训：EvoAlpha14 混合口径假 IC 0.19 → 统一后复权后归零。

## 排查清单（按出现频率）

### 1. 收益口径错误（最常见）
- 检查 label：**必须后复权 AdjVwap vwap-to-vwap**（`jobs/adj_vwap_common.adj_fwd_return()`）
- `shift(-1)` 是 1 期**前向**收益；`shift(+1)` 是错的（变成当期）
- 未复权跳空会被算成真收益：未复权 10 日收益 -16.9% vs 后复权 +1.17%（历史实测）
- 用 `lightgbm_qs/scripts/audit_adj_consistency.py` 对拍

### 2. Look-ahead（未来函数）
- `shift` 方向：因子值 t 行只能用 ≤t 的数据
- rolling 窗口右对齐（默认）ok；`center=True` 禁用
- `.cumsum()` 后再 shift 的组合检查
- 静态扫描：`scripts/audit_r28_lookahead_static.py`、`scripts/audit_pit_and_period_policy.py`

### 3. 选择泄漏（selection leakage）
- 全样本选特征 → walk-forward 训练 = 未来信息进特征集（P0-B 问题 1，已修）
- 正确：逐折 train 段内选因子（`factor_selection.py --fold`）
- 预处理 fit 必须 train-only（`modeling.leakage_guard.TrainOnlyFitGuard`）

### 4. 数据对齐错误
- date×asset 面板 reindex 后 NaN 位置漂移
- 停牌/涨跌停日因子值处理（`ashare_stock_status`、`IsSuspend` 列）
- tradable 集合：`lightgbm_qs/data/panel/vwap_trad.parquet` 的列
- 复权因子跳变日（除权除息日）的值检查

### 5. 重复样本/重叠窗口
- 10 日前瞻收益逐日滚动 → 相邻样本高度重叠；多重检验校正用 `quant_evaluator.metrics.multiple_testing`（hac_tstat/hac_pvalue metric）

## 项目内现成 audit 工具（先跑再手查）

```bash
cd /home/sunhaiwei/quant_projects
# 口径
.venv/bin/python lightgbm_qs/scripts/audit_adj_consistency.py
# PIT / look-ahead / period policy
.venv/bin/python scripts/audit_pit_and_period_policy.py
.venv/bin/python scripts/audit_r28_lookahead_static.py
# 算子数学契约 / 语义保持
.venv/bin/python scripts/audit_operator_math_contract.py
.venv/bin/python scripts/audit_execution_semantic_preservation.py
```

## 修复模式

| 病因 | 修法 |
|---|---|
| label 用了未复权 | 换 `ashare_stock_daily_adj` / `adj_fwd_return()` |
| 因子值未复权输入 | 数据源换 `_adj` 数据集 + AdjX 字段（见 quant-factor-landing） |
| 全样本选择 | 改 walk-forward per-fold 选择 |
| 预处理全样本 fit | `fit_preprocessing(train_ds, spec)` → FrozenPreprocessing + guard |
| shift 方向反 | 收益 `pct_change().shift(-N)`；因子滞后 `shift(1)` |
| 停牌日污染 | mask 掉 `IsSuspend` / tradable 外资产 |

## 验证修复

1. 修复前后同一因子 rank_ic 对比（应下降到合理区间 0.01~0.05）
2. 分十层单调性（`quantile_returns` metric）
3. OOS 段（walk-forward fold）与 IS 段差异合理
4. 证据 YAML：修复内容、测试、仍 NOT_RUN 的项如实写

## 底层 debug 通用技巧（配合 debugging skill）

- 复现最小化：单股票单时段先跑通
- 数值边界：除零 → `quant_evaluator` 内部 protected_div；NaN 传播路径
- pandas vs factor_engine 结果对拍时，注意 MultiIndex (date, asset) 排序一致性