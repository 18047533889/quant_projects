---
name: quant-ml-modeling
description: 机器学习建模的正确方式 — 用 modeling/ 库（trainer/dataset/split/leakage_guard/hyperparams）+ lightgbm_qs 现成管线 + factor_preprocess/factor_optimizer，禁止手写 walk-forward 和泄漏防护。当任务涉及"LightGBM 训练、特征矩阵、walk-forward、超参寻优、组合优化"时先读此 skill。
version: 1.0.0
---

# ML 建模标准流程（modeling/ + lightgbm_qs）

**前提**：已读 quant-platform-workflow（主趁库约束）。特征 = 已评估入选的因子；label = 后复权 vwap-to-vwap 前瞻收益。

## 我们已有的建模库（modeling/）

| 模块 | 用途 | 入口 |
|---|---|---|
| dataset | 面板数据集封装 | `modeling.dataset.PanelDataset` |
| trainer | 训练编排 + TrainResult + rank_ic | `modeling.trainer.TrainResult`、`fit_preprocessing(PanelDataset, PreprocessingSpec)` |
| split | 日期切分（唯一权威） | `date_bounded_split` / `split_by_date_cutoff` / `assert_date_authoritative` |
| leakage_guard | 泄漏防护 | `TrainOnlyFitGuard`、`assert_frozen_preprocessing` |
| learners | 模型库 | elastic_net / pcr / pls / mixture_of_experts / regime（`modeling.learners`） |
| hyperparams | 超参契约 | `modeling.hyperparams` |
| evaluation | 评估 | `modeling.evaluation` |
| artifact / model_catalog | 产物与目录 | — |

**LightGBM GPU 现成训练脚本**：`lightgbm_qs/scripts/train_lightgbm_gpu.py`（`import lightgbm as lgb` + walk-forward）。

## 特征矩阵构建（不重造）

1. 因子池合并筛选真源：`lightgbm_qs/scripts/merge_all_factors.py`（全部池 → rank_ic>0.015 → 特征矩阵，行=(date,asset) 列=因子，保留 NaN）
2. 全特征补全 + 完全去重：`build_full_features.py`（corr>0.98 簇内留 rank_ic 最高）
3. 预处理：`preprocess_features.py` + **必须用 `factor_preprocess`**：

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from factor_preprocess.registry.transforms import create_default_registry
reg = create_default_registry()
# 43 个 transform：cs_zscore / cs_winsor / cs_rank / cs_demean / cs_scale /
# ols_neutralize / industry_neutral / size_neutral / dual_neutral /
# rolling_zscore / realized_volatility / kalman_local_level / hp_filter ...
```

4. 复权对齐：特征/label 全部后复权口径（`ashare_stock_daily_adj` / `jobs/adj_vwap_common.py`）

## Walk-forward（唯一正确模式，防泄漏）

```python
from modeling.split import date_bounded_split, split_by_date_cutoff, assert_date_authoritative
from modeling.leakage_guard import TrainOnlyFitGuard, assert_frozen_preprocessing
```

- 切分：`date_bounded_split`（日期权威，`assert_date_authoritative` 校验）
- 10 日前瞻 label → cut 前 `PURGE_TRADING_DAYS=10` purge（见 `factor_selection.py`）
- **TrainOnlyFitGuard**：预处理/标准化只能 fit train 段
- 因子选择也逐折（walk-forward selection），**禁止全样本选特征**
- 参考实现：`lightgbm_qs/scripts/factor_selection.py --fold <cut>` / `train_final.py`

## 超参寻优（必须走 factor_optimizer）

```python
from factor_optimizer.grammar.mutation_spec import MutationSpec, ParameterSpec, ParameterRole
from factor_optimizer.contracts.objective import ObjectiveSpec, ObjectiveDirection
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan, create_split_aware_evaluation_fn
```

- Objective：`ObjectiveSpec`（direction: max rank_ic 等）；budget：`SearchBudget`/`BudgetTracker`
- 现成调参产物：`lightgbm_qs/scripts/best_params.json` / `best_params_full2.json`
- `train_opt.py`（Optuna 包装 rank_ic objective）也在管线里——新寻优优先接 factor_optimizer 契约

## 组合优化 / 回测（必须用我们自己的）

- 组合回测：`lightgbm_qs/scripts/portfolio_and_backtest.py`（基准 = 后复权 `vwap_trad_adj.parquet`）
- 回测引擎：`vectorbt_qs`（`vectorbt_qs.mvp.engine.runner.run_backtest / portfolio_report`，A股约束：停牌/涨跌停/手续费，参考 `vectorbt_qs/mvp/examples/ashare_demo.py`）
- 组合优化：`riskfolio_qs`
- 单策略运行登记：`single_asset_backtest_runs` 数据集（strategy_id + version）

## 禁止

- ❌ 绕过 `modeling.split`/`leakage_guard` 手写 train_test_split
- ❌ 全样本 fit 预处理（标准化必须 train-only fit）
- ❌ 手写 Sharpe/VaR/回撤（用 `quant_evaluator` metrics）
- ❌ 用 sklearn 默认 shuffle split 做时间序列（日期切分唯一权威是 `modeling.split`）