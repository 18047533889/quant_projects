---
name: quant-factor-eval-deep
description: 因子池深度评估与组合 — 横截面 IC/分层/多空、完全去重（corr>0.98 簇内留 rank_ic 最高）、聚类去冗余、线性因子与深度学习因子的互补性分析、因子池构建。全部走 quant_evaluator + lightgbm_qs 既有管线，禁止手写 RankIC。当任务涉及"因子池、因子组合、因子去重、聚类去冗余、多因子、因子库、新因子评估、深度因子"时先读此 skill。
version: 1.0.0
---

# 因子池深度评估与组合

**前提**：已读 quant-evaluate-factor（单因子评估）+ quant-platform-workflow（口径）。

## 因子池真源与筛选管线（不重造）

```
候选因子(任意来源，含深度学习因子) → merge_all_factors.py → rank_ic>0.015 → build_full_features.py(corr>0.98完全去重) → 特征矩阵 → ML/组合
```

```bash
cd /home/sunhaiwei/quant_projects
# 1) 全部因子落盘 wide parquet 后，合并 + 逐因子 rank_ic（不进大 DataFrame，防内存爆炸）
.venv/bin/python lightgbm_qs/scripts/merge_all_factors.py
# 2) 完全去重：抽样 ≤40k 行算相关，corr>0.98 簇内留 rank_ic 最高者
.venv/bin/python lightgbm_qs/scripts/build_full_features.py
# 3) 聚类去冗余（factor_assets.clustering）
.venv/bin/python jobs/cluster_factors_470.py
```

## 评估每个候选（量化深度）

对因子池每个因子，必须用 quant_evaluator（51 注册 metric）算一套，不只是 rank_ic：

- **IC 域**：`rank_ic`、`rank_ic_cross_section`、`rank_ic_time_series`、`pearson_ic`、`ic_summary`、`mean_ic`、`ic_std`、`ic_ir`、`ic_decay`、`ic_stability`、`rolling_ic`、`autocorrelation_ic`、`ic_autocorr_lag1`
- **分位数域**：`quantile_returns`、`quantile_returns_full`、`quantile_spread`、`quantile_stability`（**分十层单调性**是判断因子真伪的关键）
- **换手域**：`turnover`、`turnover_rate`、`turnover_cost`、`turnover_adjusted_ic`（高 IC 高换手 = 不可用）
- **尾部/稳定性**：`max_drawdown`、`cvar_95`、`hac_tstat`、`hac_pvalue`、`half_life`、`block_bootstrap_ci`

**好因子的标准**（经验）：截面 rank_ic 0.01~0.05 合理；>0.1 必须查泄漏（debug-pit-leakage）；分十层单调、Q1-Q10 价差稳定、OOS 段不崩。

## 深度因子 vs 线性因子的互补性分析

当池里同时有线性因子（factor_engine DSL 出的）和深度学习因子（alphaprobe RL / LightGBM 隐层）时：

1. **相关分析**：corr 矩阵看深度因子是否被线性因子覆盖。corr>0.7 说明只是线性重构，价值低；corr 低 + 各自 IC 显著 → 真互补。
2. **增量 IC**：把深度因子加入已含线性因子的模型，看 OOS rank_ic / IC_IR 提升（参考 `merge_all_factors.py` 之后喂 `train_lightgbm_gpu.py` 对比）。
3. **特征重要性**：LightGBM `feature_importance` 看深度因子是否进 top 特征；进不了说明冗余。
4. **分层叠加**：线性因子分层内再按深度因子分层，看是否还有单调性（条件独立性检验）。

## 组合信号（多因子 → 一信号）

- **不要简单等权平均**。至少按 `ic_ir` 加权，或 `rank` 加权。
- 完整方法：因子矩阵 → `factor_preprocess`（cs_zscore / cs_winsor / ols_neutralize / industry_neutral）→ 线性/ML 组合。
- 组合信号评估：`quantile_returns` + `quantile_spread` + `turnover_adjusted_ic` 全套（同单因子标准）。
- 组合信号才是喂给 riskfolio_qs 的 alpha（`alpha_input_type=score` 会被 `score_return_scale_bps` 标准化）。

## 输出规范

- 评估结果 CSV：`name,pool,rank_ic,n_dates,drop`（同 `data/build/rankic_all_factors.csv` 格式）。
- 报告：`lightgbm_qs/scripts/rankic_report.py`。
- 结论证据 YAML：git_sha、metric 版本、测试项、NOT_RUN 项如实写。

## 禁止

- ❌ `pd.Series.corr` / `scipy.stats.spearmanr` 手搓 RankIC（走 quant_evaluator.metrics）
- ❌ 全样本选因子喂 walk-forward（选择泄漏）
- ❌ 只看 rank_ic 不看换手/回撤/分层单调性就入库
- ❌ 深度因子不查与线性因子的相关性就直接并入特征矩阵
