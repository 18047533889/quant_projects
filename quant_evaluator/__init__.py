"""
quant_evaluator: Pure Evidence/Metric/Diagnosis Engine

Batch-first evaluator returning typed evidence bundles. Consumes explicit FactorBatch
and LabelBundle from caller; never infers labels, shifts, or fills.

Public API:
    - evaluate: Batch evaluation facade
    - EvaluationRequest, EvaluationBundle: Primary contracts
    - FactorBatch, LabelBundle: Input contracts
    - FactorDiagnosis, MetricValue: Result types
    - ContractError, DataError, CapabilityError: Error taxonomy
"""

__version__ = "0.0.1a1"

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import (
    QuantEvaluatorError,
    ContractError,
    DataError,
    CapabilityError,
    DurableCacheCapabilityError,
    InsufficientObservations,
    UnsupportedMetricError,
    TimingContractError,
)
from quant_evaluator.api.requests import (
    EvaluationRequest,
    EvaluationBundle,
    MetricValue,
    FactorDiagnosis,
)

__all__ = [
    "__version__",
    "FactorBatch",
    "AxisRef",
    "LabelBundle",
    "EvaluationRequest",
    "EvaluationBundle",
    "MetricValue",
    "FactorDiagnosis",
    "QuantEvaluatorError",
    "ContractError",
    "DataError",
    "CapabilityError",
    "DurableCacheCapabilityError",
    "InsufficientObservations",
    "UnsupportedMetricError",
    "TimingContractError",
]
