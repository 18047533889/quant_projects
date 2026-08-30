---
name: quant-platform-workflow
description: 主趁库体系强制约束 — 读数据/算因子/评估/预处理/优化/资产化必须走用户自有平台库（data_access → factor_engine → quant_evaluator → factor_preprocess → factor_optimizer → factor_assets），禁止绕库手写。任何涉及 A 股因子、行情读取、收益口径、回测的任务，先读本 skill。
alwaysApply: true
version: 1.0.0
---

# 量化平台强制工作流（主趁库体系）

**这是丁运磊的硬性规则（2026-08-28 原话：不准自己瞎写，别自己瞎写别的库落值结果不用我的库）。**
凡涉及因子/行情/评估/回测，一律按下面链路走，禁止绕过库手写 numpy/pandas 等价实现。

## 库链路（每一步用哪个库）

| 步骤 | 必须用的库 | 入口 |
|---|---|---|
| 1. 读数据（行情/财务/COS） | `data_access` | `from data_access import get_store; store = get_store(); store.read_frame(...)` |
| 2. 算因子值/落值 | `factor_engine` | DSL 或 `FactorEngine.run()`，见 quant-factor-landing skill |
| 3. 因子评估（RankIC/IR/分十层/回撤） | `quant_evaluator` | `quant_evaluator.registry.metrics`（51 个注册 metric），见 quant-evaluate-factor skill |
| 4. 预处理（zscore/winsor/中性化） | `factor_preprocess` | `from factor_preprocess.registry.transforms import create_default_registry` |
| 5. 参数寻优 | `factor_optimizer` | `factor_optimizer.grammar.MutationSpec` / `contracts.objective.ObjectiveSpec` / `contracts.search_budget.SearchBudget` |
| 6. 聚类/筛选/资产化 | `factor_assets` | `factor_assets.clustering.families.ClusterResult/ClusterArtifact`、`FactorSet` |

## 硬性口径（违反 = 结果作废）

1. **收益一律 vwap-to-vwap 且必须后复权**：`AdjVwap.pct_change().shift(-1)`。
   统一入口：`/home/sunhaiwei/quant_projects/jobs/adj_vwap_common.py`
   （`load_adj_vwap()` / `adj_fwd_return()`）。
   **禁止**未复权 `StockDailyBar.Vwap`、禁止 close 收益。
2. **读 COS 只走 admin-cos**：`sudo -n /usr/local/libexec/quantsociety-cos/admin-cos <cmd> cos://qs-cold/<prefix>`（bucket=`qs-cold`，支持 ls -r / cat / du / whoami）。不用 research-cos。
3. **因子输入数据也用后复权**：数据集用 `ashare_stock_daily_adj`（AdjOpen/AdjHigh/AdjLow/AdjClose/AdjPreClose/AdjAmount/AdjVwap；Volume/Factor 原样）。
4. **评估前先挂收益口径**：任何 rank_ic/IC 计算的 label 必须来自 `adj_fwd_return()`（10日前瞻时用 `adj.shift(-10)/adj - 1`，见 `lightgbm_qs/scripts/build_adj_label.py`）。

## 项目路径与环境

- 仓库根：`/home/sunhaiwei/quant_projects`（**monorepo 双副本**：`runtime/`、`cleaned_operators/`、`ir/`、`storage/` 在根目录和 `factor_engine/` 下各一份，改一处必须 `cp` 镜像到另一处保持 byte-identical）
- Python 环境：**必须用** `/home/sunhaiwei/quant_projects/.venv/bin/python`（系统 python3 缺 polars，`load_all` 会 fail）
- 因子值落盘目录：`weekly_backtest_output/factor_matrices_all/`（date×asset wide parquet）
- 因子池：`lightgbm_qs/data/factor_pools/`（fm247/fmqa/cogfull/cogshort/cogneutral/delivery/optfac/factmat/lqtp）
- 合并筛选真源：`lightgbm_qs/scripts/merge_all_factors.py`（rank_ic>0.015 + 完全去重 corr>0.98）

## 操作前必查

```python
# 1) 检查算子在不在白名单（1478 个算子）
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
import factor_engine.cleaned_operators as co
co.load_all(include_research=False)   # 必须先 load
from factor_engine.api.operator_registry import build_dsl_allowlist
ops = build_dsl_allowlist()   # dict/dict-keys，算子名在里面才能用

# 2) 检查评估指标在不在注册表（51 个）
from quant_evaluator.registry.metrics import _REGISTRY
ids = _REGISTRY.to_dict()   # rank_ic / pearson_ic / ic_ir / quantile_spread / max_drawdown ...
```

算子/指标不在注册表里 → 先在对应库里加实现并走证据流程，**不许临时手写**。

## 数据集速查（data_access/config/datasets.yaml 登记的 73 个）

常用 A 股：`ashare_stock_daily`（未复权）/ `ashare_stock_daily_adj`（**后复权，默认用它**）/ `ashare_stock_minute_adj` / `ashare_index_daily` / `ashare_index_constituent` / `ashare_stock_industry` / `ashare_stock_valuation_daily` / `ashare_stock_indicator` / `ashare_stock_list` / `ashare_universe_daily` / 财务四表 `ashare_stock_{balance,income,cashflow,capital_daily}`。
美股：`us_stock_daily` 等。因子湖：`factor_lake` / `factor_lake_staging`（写）/ `factor_matrix`。
完整清单：`grep -E "^[a-z_0-9]+:" data_access/config/datasets.yaml`。

## 禁止清单

- ❌ 绕过 `data_access` 直接 `pd.read_parquet` 读行情
- ❌ 手写 corr/std/rank 算 RankIC（必须 `quant_evaluator.metrics.ic` 或注册 metric）
- ❌ 用未复权 Vwap / Close 收益
- ❌ 用 research-cos
- ❌ 全数据集无 time_range 拉取（COS remote 必带 `time_range`/`read_time_ranges`）
- ❌ `git checkout/restore/stash/clean/reset/rebase`（working tree is truth）