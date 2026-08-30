---
name: quant-evaluate-factor
description: 因子评估的正确方式 — 必须用 quant_evaluator（51 个注册 metric：rank_ic/pearson_ic/ic_ir/quantile_spread/max_drawdown/turnover 等）+ 后复权 vwap-to-vwap 收益 label，禁止手写 RankIC。当任务涉及"评估因子、算 IC、分层回测、因子筛选、去重"时先读此 skill。
version: 1.0.0
---

# 因子评估标准流程（quant_evaluator）

**前提**：已读 quant-platform-workflow（收益口径）+ quant-factor-landing（因子值来源）。

## 第一步：挂正确的 label（收益口径）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
# 1日前瞻收益（后复权 AdjVwap vwap-to-vwap，硬性口径）
from jobs.adj_vwap_common import load_adj_vwap, adj_fwd_return
fwd1 = adj_fwd_return()                       # AdjVwap.pct_change().shift(-1)

# 10日前瞻（因子池筛选口径）：adj.shift(-10)/adj - 1，参考 lightgbm_qs/scripts/build_adj_label.py
vwap = load_adj_vwap()
fwd10 = vwap.shift(-10) / vwap - 1.0
```

**禁止**：`StockDailyBar.Vwap`（未复权）、close 收益、手写 corr 算 IC。

## 第二步：用 quant_evaluator 的 metric（不许自定义）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from quant_evaluator.registry.metrics import _REGISTRY
ids = sorted(_REGISTRY.to_dict())     # 51 个注册 metric，先查有没有
```

常用 metric ID（按 domain）：
- **IC 域**：`rank_ic`、`rank_ic_cross_section`、`rank_ic_time_series`、`pearson_ic`、`spearman_ic`、`ic_summary`、`mean_ic`、`ic_std`、`ic_ir`、`ic_decay`、`ic_stability`、`rolling_ic`、`autocorrelation_ic`、`ic_autocorr_lag1`
- **分位数域**：`quantile_returns`、`quantile_returns_full`、`quantile_spread`、`quantile_stability`
- **换手域**：`turnover`、`turnover_rate`、`turnover_cost`、`turnover_adjusted_ic`、`turnover_stability`
- **回撤域**：`max_drawdown`、`calmar_ratio`、`drawdown_duration`
- **尾部风险域**：`cvar_95`、`cvar_99`、`skewness`、`kurtosis`
- **覆盖域**：`factor_coverage`、`return_coverage`、`joint_coverage`、`coverage`、`coverage_stability`
- **稳定性域**：`ic_stability`、`rank_stability`、`subsample_stability`、`turnover_stability`、`coverage_stability`
- **HHI 域**：`hhi_concentration`、`hhi_effective_n`
- 其他：`hac_tstat`、`hac_pvalue`、`half_life`、`block_bootstrap_ci`、`factor_turnover_rate`

底层函数（`quant_evaluator.metrics`）：`compute_daily_ic` / `compute_mean_ic` / `compute_icir` / `compute_ic_tstat` / `compute_ic_decay` / `assign_quantiles` / `compute_quantile_returns` / `compute_top_bottom_spread` / `compute_turnover` / `compute_coverage`。

## 第三步：Evaluator 批量评估（多因子）

```python
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef   # 以实际模块为准（tests/test_consistency.py 有示例）
from quant_evaluator.contracts.label_bundle import LabelBundle

ev = Evaluator(enable_cache=False)
# FactorBatch(factor_ids, time_axis=AxisRef("t","int",T), asset_axis=AxisRef("a","str",N), values=...)
# LabelBundle(target_id="r", values=..., horizon=1, decision_time=..., label_start_time=..., label_end_time=...)
result = ev.evaluate(factor_batch, label_bundle, metric_specs=[{"metric_id": "rank_ic"}, ...])
```

单因子快速路径（因子池已 wide parquet 的）：参考 `lightgbm_qs/scripts/factor_selection.py`（walk-forward 逐折，train 段才算统计量，防选择泄漏）与 `merge_all_factors.py`（逐因子 rank_ic，不进大 DataFrame）。

## 第四步：筛选与去重（走既有管线，不重造）

- **门槛**：rank_ic > 0.015（全历史或逐折 train 段）
- **完全去重**：抽样(≤40k行)算相关，corr>0.98 簇内留 rank_ic 最高者（`lightgbm_qs/scripts/build_full_features.py` 第[5]段）
- **聚类去冗余**：`factor_assets.clustering`（ClusterResult/ClusterArtifact）+ `jobs/cluster_factors_470.py`
- **walk-forward 选择**（防选择泄漏，研究正确模式）：`factor_selection.py --fold 2019-04-01`

## 输出规范

- 评估结果 CSV：`name,pool,rank_ic,n_dates,drop`（同 `data/build/rankic_all_factors.csv` 格式）
- 报告：`lightgbm_qs/scripts/rankic_report.py`
- 涉及结论的证据：evidence YAML（git_sha、测试、NOT_RUN 项如实写）

## 禁止

- ❌ `pd.Series.corr` / `scipy.stats.spearmanr` 手搓 RankIC（必须走 `quant_evaluator.metrics`）
- ❌ 全样本选特征喂 walk-forward 训练（选择泄漏，P0-B 已修，别回退）
- ❌ 用未复权收益算 IC