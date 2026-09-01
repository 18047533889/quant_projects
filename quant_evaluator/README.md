# quant_evaluator — 因子评估器

批量因子评估器，返回类型化证据包。注册 **60+ 指标**（rank_ic / pearson_ic / ic_ir / hac_tstat /
quantile_spread / max_drawdown / cvar_95/99 / turnover* / ic_autocorr_lag1 / half_life /
block_bootstrap_ci / ...），对显式 `LabelBundle`（严格时序）评估因子批次。

**版本:** 0.0.1a1 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator (私有)

## 安装与第一次评估

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator.git
cd quant_evaluator
pip install -e ".[full]"        # 或 [polars] / [stats] / [reporting]
```

```python
from quant_evaluator.api import EvaluationRequest, EvaluationBundle
from quant_evaluator.contracts.label_bundle import LabelBundle

label = LabelBundle(
    target_id="fwd_vwap_5d",
    values=forward_returns,            # VWAP_{t+H}/VWAP_t - 1  （vwap-to-vwap）
    horizon=5,
    price_convention="vwap_to_vwap",   # 平台硬性口径，不要改
)
req = EvaluationRequest(
    batch_or_factor_ids=...,
    label_bundle=label,
    metric_ids=("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage"),
)
bundle: EvaluationBundle = evaluate(req)   # 经 runtime.evaluator
```

核心契约：
- `LabelBundle` — 显式前向标签；**时序全部由调用方提供**；默认 `price_convention="vwap_to_vwap"`。
- `FactorBatch` — `(时间 × 资产)` 轴的因子值 + validity。
- `EvaluationBundle` — `metric_values` / `diagnostics` / `grouped_metrics` / `series_refs` / `split_ref`。
- `SealedSplitRef` — 可选 sealed train/valid/test 切分绑定。

## 指标（60+）

权威清单：`docs/METRIC_REGISTRY_COVERAGE.csv`（metric_id / implementation / artifact kind /
report panel / registered|tested|production 标记 / institutional domain）。

常用：`rank_ic`、`pearson_ic`、`rank_ic_cross_section`、`rank_ic_time_series`、`ic_ir`、`ic_std`、
`ic_decay`、`ic_autocorr_lag1`、`ic_median`、`mean_ic`、`spearman_ic`、`quantile_returns`、
`quantile_spread`、`quantile_stability`、`max_drawdown`、`drawdown_duration`、`calmar_ratio`、
`cvar_95/99`、`var_95/99`、`turnover`、`turnover_rate`、`turnover_cost`、`turnover_adjusted_ic`、
`turnover_stability`、`factor_turnover_rate`、`half_life`、`rank_stability`、`coverage`、
`factor_coverage`、`joint_coverage`、`return_coverage`、`hhi_concentration`、`hhi_effective_n`、
`block_bootstrap_ci`、`hac_tstat`、`hac_pvalue`、`autocorrelation_ic`、`rolling_ic`、
`subsample_stability`、`benjamini_hochberg_correction`、`bonferroni_correction`、`fdr`，
以及 `metrics/interactions/` 下的 pairwise / conditional / substitution 交互指标。

## 架构

```
api/           EvaluationRequest / EvaluationBundle / MetricValue / FactorDiagnosis
contracts/     LabelBundle, FactorBatch, SealedSplitRef, evidence_status,
               metric_artifacts, treatment_evaluation, quantile_policy
registry/      MetricSpec + MetricRegistry（可 seal），presets（factor_core / ...）
metrics/       ic, quantile, drawdown, turnover, risk(VaR/CVaR), stats, exposure,
               quality, temporal, robustness, interactions
runtime/       Evaluator、streaming、parallel executor、cache、budgets
adapters/      factor_engine（FactorBatch/identity）、data_access（universe/日历）
               —— lazy import，可选依赖，缺装不影响 core
reporting/     tear_sheet、library_reports、chart_spec、artifacts
kernels/       numba 快路径 + reference bridge
docs/          METRIC_REGISTRY_COVERAGE.csv、METRIC_COVERAGE_COMPILER.csv
tests/         29 个测试文件（全量 332 passed / 2 skipped）
```

## 依赖与接口（谁 import 谁）

- **依赖（可选适配器）**：`factor_engine`（`adapters/factor_engine.py`）、
  `data_access`（`adapters/data_access.py`）；lazy import，core 不硬依赖。
- **被谁调用**：
  - `factor_optimizer` → 评估证据（`adapters/quant_evaluator.py`）
  - `factor_assets` → `QEEvidenceProvider`（bundle → EvidenceRef）
  - `quant_platform` → EvidenceStatus / 候选流水线
  - `modeling` → 模型 OOS 独立 IC 校验

## 硬性规则

- **禁止**自行推断标签 / 位移 / 填充 —— QE 只消费 `LabelBundle` 给定的时序。
- **vwap-to-vwap** 是全局收益口径（`LabelBundle.price_convention`）。
- 指标注册表 seal 后不可变 —— 新增指标需新注册路径，不许静默改。

## 相关仓库

- **factor_engine** — 上游因子值
- **data_access** — universe / 日历 provider
- **factor_optimizer / factor_assets / modeling** — 证据消费者（见上）
