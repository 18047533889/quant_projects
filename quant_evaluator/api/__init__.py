"""
Public API for quant_evaluator.
"""

from quant_evaluator.api.requests import (
    EvaluationRequest,
    EvaluationBundle,
    MetricValue,
    FactorDiagnosis,
)
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario
from quant_evaluator.api.ic_decay import ICDecayResult, compute_ic_decay
from quant_evaluator.api.horizons import (
    HorizonEvaluationBundle, HorizonSummaryRow, MultiHorizonSummary,
    evaluate_horizons, build_forward_return_label_bundles,
    summarize_horizons, evaluate_factor_multi_horizon,
)

__all__ = [
    "EvaluationRequest",
    "MetricInstance", "EvaluationScenario",
    "EvaluationBundle",
    "MetricValue",
    "FactorDiagnosis",
    "FactorValueRef",
    "LabelBundleRef",
    "HorizonEvaluationBundle", "HorizonSummaryRow", "MultiHorizonSummary",
    "ICDecayResult", "compute_ic_decay",
    "evaluate_horizons", "build_forward_return_label_bundles",
    "summarize_horizons", "evaluate_factor_multi_horizon",
]
