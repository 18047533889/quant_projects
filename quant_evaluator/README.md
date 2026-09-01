# quant_evaluator

Batch factor evaluator returning typed evidence bundles. Registers **60+ metrics**
(rank_ic, pearson_ic, ic_ir, hac_tstat, quantile_spread, max_drawdown, cvar_95/99,
turnover*, ic_autocorr_lag1, half_life, block_bootstrap_ci, ...) and evaluates a
batch of factors against an explicit `LabelBundle` with strict timing.

**Version:** 0.0.1a1 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator (private)

> 中文用户见平台交接文档 `HANDOVER.md`；指标权威清单：`docs/METRIC_REGISTRY_COVERAGE.csv`。

## Install & first evaluation

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_evaluator.git
cd quant_evaluator
pip install -e ".[full]"        # or [polars] / [stats] / [reporting]
```

```python
from quant_evaluator.api import EvaluationRequest, EvaluationBundle
from quant_evaluator.contracts.label_bundle import LabelBundle

label = LabelBundle(
    target_id="fwd_vwap_5d",
    values=forward_returns,            # VWAP_{t+H}/VWAP_t - 1  (vwap-to-vwap)
    horizon=5,
    price_convention="vwap_to_vwap",   # default; do not change
)
req = EvaluationRequest(
    batch_or_factor_ids=...,
    label_bundle=label,
    metric_ids=("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage"),
)
bundle: EvaluationBundle = evaluate(req)   # via runtime.evaluator
```

Key contracts:
- `LabelBundle` — explicit forward labels; **all timing supplied by caller**;
  default `price_convention="vwap_to_vwap"` (the platform hard rule).
- `FactorBatch` — factor values on `(time × asset)` axes + validity.
- `EvaluationBundle` — `metric_values`, `diagnostics`, `grouped_metrics`,
  `series_refs`, `split_ref`, warnings.
- `SealedSplitRef` — optional sealed train/valid/test split binding.

## Metrics (60+)

Authoritative list: `docs/METRIC_REGISTRY_COVERAGE.csv` (metric_id, implementation,
artifact kind, report panel, registered/tested/production flags, institutional domain).

Highlights: `rank_ic`, `pearson_ic`, `rank_ic_cross_section`, `rank_ic_time_series`,
`ic_ir`, `ic_std`, `ic_decay`, `ic_autocorr_lag1`, `ic_median`, `mean_ic`,
`spearman_ic`, `quantile_returns`, `quantile_spread`, `quantile_stability`,
`max_drawdown`, `drawdown_duration`, `calmar_ratio`, `cvar_95/99`, `var_95/99`,
`turnover`, `turnover_rate`, `turnover_cost`, `turnover_adjusted_ic`,
`turnover_stability`, `factor_turnover_rate`, `half_life`, `rank_stability`,
`coverage`, `factor_coverage`, `joint_coverage`, `return_coverage`,
`hhi_concentration`, `hhi_effective_n`, `block_bootstrap_ci`, `hac_tstat`,
`hac_pvalue`, `autocorrelation_ic`, `rolling_ic`, `subsample_stability`,
`benjamini_hochberg_correction`, `bonferroni_correction`, `fdr`, ... plus
interaction/conditional/substitution metrics under `metrics/interactions/`.

## Architecture

```
api/           EvaluationRequest / EvaluationBundle / MetricValue / FactorDiagnosis
contracts/     LabelBundle, FactorBatch, SealedSplitRef, evidence_status,
               metric_artifacts, treatment_evaluation, quantile_policy
registry/      MetricSpec + MetricRegistry (seal-able), presets (factor_core / ...)
metrics/       ic, quantile, drawdown, turnover, risk (VaR/CVaR), stats, exposure,
               quality, temporal, robustness, interactions (pairwise/conditional/...)
runtime/       Evaluator, streaming evaluator, parallel executor, cache, budgets
adapters/      factor_engine (FactorBatch/identity provider), data_access
               (universe/calendar provider) — lazy import, optional deps
reporting/     tear_sheet, library_reports, chart_spec, artifacts
kernels/       numba fast paths + reference bridge
docs/          METRIC_REGISTRY_COVERAGE.csv, METRIC_COVERAGE_COMPILER.csv
tests/         29 test files (full suite 332 passed / 2 skipped)
```

## Hard rules

- **Never infer labels/shifts/fills** — QE consumes `LabelBundle` timing as-is.
- **vwap-to-vwap** is the platform-wide return convention (`LabelBundle.price_convention`).
- Metric registry is **sealed after build** — adding a metric requires a new
  registration path, not silent mutation.

## Related repos

- **factor_engine** — upstream factor values (via `adapters/factor_engine.py`)
- **data_access** — universe/calendar provider (via `adapters/data_access.py`)
- **factor_optimizer** — consumes `EvaluationBundle` as evidence
- **factor_assets** — wraps evaluation evidence as `EvidenceRef` (QEEvidenceProvider)
- **modeling** — independent IC computation for model OOS evaluation
