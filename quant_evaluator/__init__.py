"""quant_evaluator package."""

from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.api.requests import EvaluationRequest, EvaluationBundle
from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario

# backend contracts exposed for callers choosing GPU execution
from quant_evaluator.contracts.backend_policy import (
    BackendPolicy,
    PrecisionPolicy,
    GPUExecutionPolicy,
)
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.horizons import HorizonEvaluationBundle, evaluate_horizons

__all__ = [
    "evaluate",
    "EvaluationRequest", "EvaluationBundle", "MetricInstance", "EvaluationScenario",
    "BackendPolicy",
    "PrecisionPolicy",
    "GPUExecutionPolicy",
    "BatchEvaluationBundle",
    "HorizonEvaluationBundle",
    "evaluate_horizons",
]
