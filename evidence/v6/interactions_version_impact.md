# V6 interaction semantics and impact evidence

Verified public symbols:

- `quant_evaluator.metrics.FactorRelation`
- `quant_evaluator.metrics.FactorRelationEvidence`
- `quant_evaluator.metrics.IncrementalICDiagnostics`
- `quant_evaluator.metrics.SAME_DATE_DESCRIPTIVE`
- `quant_evaluator.metrics.compute_substitution_effect`
- `quant_evaluator.metrics.detect_substitutable_factors`
- `quant_evaluator.metrics.compute_marginal_contribution`
- `quant_evaluator.metrics.compute_conditional_ic`
- `quant_evaluator.metrics.compute_incremental_ic`
- `quant_evaluator.metrics.compute_partial_ic`
- `quant_evaluator.metrics.compute_pairwise_artifacts`

Method descriptors:

- Estimation scope: `SAME_DATE_DESCRIPTIVE`
- Pearson definition: `raw_value_residuals_then_pearson_correlation`
- Spearman definition: `raw_value_residuals_then_spearman_correlation`
- Projection authority: `centered_wls_svd.v1`
- Relation orientation: `frozen_as_supplied`
- Relation sample scope: `paired_common_support`
- Pairwise uncertainty scale: `raw_correlation_hac_daily_corr`

Schema and behavior impact:

- Conditional sample counts change from `(T,Q)` group counts to `(T,F,Q)` joint evaluation counts.
- Substitution decisions now require signed similarity, non-weak utility evidence, and bounded paired lift; the legacy ratio is retired.
- Marginal, substitution, and complementarity utilities use common support.
- Sparse pair APIs lazily consume explicit candidate identities and enforce a hard 10,000-pair default budget; dense and rolling outputs preflight a 10,000,000-element budget.
- Same-date residual/partial association cannot authorize OOS model admission; the FA residual-novelty consumer fails closed unless explicitly used for research diagnostics.
- Duplicate and near-collinear controls use rank-aware SVD subspace projection on CPU and GPU; saturated and zero-residual cases remain non-computed.

Historical impact:

- Recompute substitution/de-duplication decisions made with the prior ratio rule.
- Recompute comparisons whose baseline and combined statistics used different support.
- Recompute pairwise edges affected by an unrelated all-missing factor.
- Do not reuse same-date residual novelty as production/OOS admission evidence.
- Preserve underlying factor values and unaffected pair evidence.

Verification: `59 passed in 5.89s` in `evidence/v6/interactions_final_suite.txt`.
