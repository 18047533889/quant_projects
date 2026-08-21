"""
Core metrics module for quant_evaluator.

Reference implementations: coverage, IC, quantile, turnover, IC summary,
exposure, temporal, robustness, distribution, portfolio statistics,
multiple testing corrections, factor interactions, and advanced statistical tests.
"""

from quant_evaluator.metrics.quality import (
    compute_coverage,
    compute_per_time_coverage,
)
from quant_evaluator.metrics.ic import (
    compute_daily_ic,
    compute_mean_ic,
    compute_mean_ic_value,
    compute_ic_std,
)
from quant_evaluator.metrics.quantile import (
    assign_quantiles,
    compute_quantile_returns,
    compute_top_bottom_spread,
)
from quant_evaluator.metrics.turnover import (
    compute_turnover,
    compute_turnover_series,
    estimate_turnover_from_ranks,
)
from quant_evaluator.metrics.ic_summary import (
    compute_icir,
    compute_ic_tstat,
    compute_ic_decay,
    compute_ic_stability,
)
from quant_evaluator.metrics.exposure import (
    compute_factor_loadings,
    compute_sector_exposure,
    compute_style_exposure,
    compute_concentration_hhi,
)
from quant_evaluator.metrics.temporal import (
    compute_autocorrelation,
    compute_ic_autocorrelation,
    compute_rank_stability,
    compute_mean_rank_stability,
    compute_factor_turnover_rate,
    compute_half_life,
)
from quant_evaluator.metrics.robustness import (
    compute_subsample_ic,
    compute_subsample_ic_std,
    compute_hac_variance,
    compute_hac_tstat,
    compute_block_bootstrap_ci,
)
from quant_evaluator.metrics.distribution import (
    compute_skewness,
    compute_kurtosis,
    detect_outliers_iqr,
    detect_outliers_zscore,
    compute_outlier_ratio,
    compute_higher_moments,
)
from quant_evaluator.metrics.portfolio_stats import (
    compute_long_short_returns,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
    compute_calmar_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)
from quant_evaluator.metrics.multiple_testing import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    holm_bonferroni_correction,
    sidak_correction,
    compute_fdr,
)
from quant_evaluator.metrics.probe_portfolio import (
    construct_long_short_portfolio,
    compute_equal_weighted_returns,
    compute_cap_weighted_returns,
    compute_long_short_equal_weighted,
    compute_long_short_cap_weighted,
    compute_portfolio_weights,
    compute_portfolio_concentration,
    compute_turnover_from_positions,
)
from quant_evaluator.metrics.interactions.pairwise import (
    compute_pairwise_correlation,
    compute_rolling_pairwise_correlation,
    compute_correlation_matrix,
)
from quant_evaluator.metrics.interactions.conditional import (
    compute_conditional_ic,
    compute_incremental_ic,
    compute_partial_ic,
)
from quant_evaluator.metrics.interactions.substitution import (
    compute_substitution_effect,
    detect_substitutable_factors,
    compute_marginal_contribution,
)
from quant_evaluator.metrics.interactions.complementarity import (
    compute_complementarity_score,
    detect_complementary_pairs,
    compute_interaction_strength,
)

__all__ = [
    "compute_coverage",
    "compute_per_time_coverage",
    "compute_daily_ic",
    "compute_mean_ic",
    "compute_mean_ic_value",
    "compute_ic_std",
    "assign_quantiles",
    "compute_quantile_returns",
    "compute_top_bottom_spread",
    "compute_turnover",
    "compute_turnover_series",
    "estimate_turnover_from_ranks",
    "compute_icir",
    "compute_ic_tstat",
    "compute_ic_decay",
    "compute_ic_stability",
    "compute_factor_loadings",
    "compute_sector_exposure",
    "compute_style_exposure",
    "compute_concentration_hhi",
    "compute_autocorrelation",
    "compute_ic_autocorrelation",
    "compute_rank_stability",
    "compute_mean_rank_stability",
    "compute_factor_turnover_rate",
    "compute_half_life",
    "compute_subsample_ic",
    "compute_subsample_ic_std",
    "compute_hac_variance",
    "compute_hac_tstat",
    "compute_block_bootstrap_ci",
    "compute_skewness",
    "compute_kurtosis",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "compute_outlier_ratio",
    "compute_higher_moments",
    "compute_long_short_returns",
    "compute_sharpe_ratio",
    "compute_maximum_drawdown",
    "compute_calmar_ratio",
    "compute_sortino_ratio",
    "compute_win_rate",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    "holm_bonferroni_correction",
    "sidak_correction",
    "compute_fdr",
    "construct_long_short_portfolio",
    "compute_equal_weighted_returns",
    "compute_cap_weighted_returns",
    "compute_long_short_equal_weighted",
    "compute_long_short_cap_weighted",
    "compute_portfolio_weights",
    "compute_portfolio_concentration",
    "compute_turnover_from_positions",
    "compute_pairwise_correlation",
    "compute_rolling_pairwise_correlation",
    "compute_correlation_matrix",
    "compute_conditional_ic",
    "compute_incremental_ic",
    "compute_partial_ic",
    "compute_substitution_effect",
    "detect_substitutable_factors",
    "compute_marginal_contribution",
    "compute_complementarity_score",
    "detect_complementary_pairs",
    "compute_interaction_strength",
]
