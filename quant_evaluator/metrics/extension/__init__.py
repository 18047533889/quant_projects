"""QE extension metric families (QE-EXT-SPEC-1.0).

Families (each implemented in its own module; shared §6.6 tail primitives
in ``_tail``):
- M01 paired net performance / M02 out-of-sample deltas (``paired_performance``,
  ``oos_delta``), CPU kernels only;
- M03 path risk (``path_risk``: cdar / ced / drawdown_budget_exceedance /
  joint_drawdown_occupancy) and M04 book-conditional tail loss
  (``book_risk``), with CPU reference / CPU fast / opt-in CUDA backends.

Existing metric semantics elsewhere in the repo are untouched (§1.1).
"""

from quant_evaluator.metrics.extension.paired_performance import (
    BlockStartsPlan,
    METRIC_VERSION,
    PairedMetricResult,
    default_hac_max_lag,
    paired_cer_delta,
    paired_es_improvement,
    paired_mdd_improvement,
    paired_net_sharpe_delta,
    paired_sharpe_bootstrap_ci,
    paired_sharpe_hac_ci,
    paired_sharpe_studentized_ci,
)
from quant_evaluator.metrics.extension.oos_delta import (
    OOSAblationTaskResult,
    OOSDeltaResult,
    OOSFitManifest,
    oos_ablation_summary,
    oos_daily_mse_improvement,
    oos_r2_gain,
    oos_rank_ic_delta,
)
from quant_evaluator.metrics.extension.path_risk import (
    block_plan,
    cdar,
    ced,
    drawdown_budget_exceedance,
    joint_drawdown_occupancy,
    merge_path,
    path_summary,
    summary_mdd,
)
from quant_evaluator.metrics.extension.book_risk import (
    book_conditional_loss,
    book_downside_beta,
    book_tail_loss_delta,
)

__all__ = [
    # M01/M02
    "METRIC_VERSION",
    "BlockStartsPlan",
    "PairedMetricResult",
    "OOSFitManifest",
    "OOSDeltaResult",
    "OOSAblationTaskResult",
    "default_hac_max_lag",
    "paired_net_sharpe_delta",
    "paired_sharpe_hac_ci",
    "paired_sharpe_bootstrap_ci",
    "paired_sharpe_studentized_ci",
    "paired_cer_delta",
    "paired_mdd_improvement",
    "paired_es_improvement",
    "oos_rank_ic_delta",
    "oos_r2_gain",
    "oos_daily_mse_improvement",
    "oos_ablation_summary",
    # M03
    "block_plan",
    "cdar",
    "ced",
    "drawdown_budget_exceedance",
    "joint_drawdown_occupancy",
    "merge_path",
    "path_summary",
    "summary_mdd",
    # M04
    "book_conditional_loss",
    "book_downside_beta",
    "book_tail_loss_delta",
]
