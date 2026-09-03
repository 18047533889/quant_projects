"""quant_evaluator package."""

from quant_evaluator.runtime.evaluator import evaluate

# backend contracts exposed for callers choosing GPU execution
from quant_evaluator.contracts.backend_policy import (
    BackendPolicy,
    PrecisionPolicy,
    GPUExecutionPolicy,
)
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle

__all__ = [
    "evaluate",
    "BackendPolicy",
    "PrecisionPolicy",
    "GPUExecutionPolicy",
    "BatchEvaluationBundle",
]
