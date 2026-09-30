"""quant_evaluator package."""

from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.api.requests import EvaluationRequest, EvaluationBundle
from quant_evaluator.runtime.auto_calibration import AutoCalibrationOptions
from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario

# backend contracts exposed for callers choosing GPU execution
from quant_evaluator.contracts.backend_policy import (
    BackendPolicy,
    PrecisionPolicy,
    GPUExecutionPolicy,
)
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.evaluate_many import evaluate_many
from quant_evaluator.api.ic_decay import ICDecayResult, compute_ic_decay
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.api.horizons import (
    HorizonEvaluationBundle, MultiHorizonSummary,
    evaluate_horizons, evaluate_factor_multi_horizon,
)

__all__ = [
    "evaluate",
    "EvaluationRequest", "EvaluationBundle", "MetricInstance", "EvaluationScenario",
    "AutoCalibrationOptions",
    "BackendPolicy",
    "PrecisionPolicy",
    "GPUExecutionPolicy",
    "BatchEvaluationBundle",
    "evaluate_many",
    "HorizonEvaluationBundle", "MultiHorizonSummary",
    "ICDecayResult", "compute_ic_decay",
    "evaluate_factor_source_batch",
    "evaluate_horizons", "evaluate_factor_multi_horizon",
]
