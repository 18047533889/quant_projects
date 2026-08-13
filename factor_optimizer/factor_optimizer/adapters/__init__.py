"""Adapter protocols for FE and QE integration."""

from .factor_engine import (
    FactorEngineAdapter,
    OptionalDependencyMissing as FEOptionalDependencyMissing,
    create_fe_adapter,
)
from .quant_evaluator import (
    QuantEvaluatorAdapter,
    OptionalDependencyMissing as QEOptionalDependencyMissing,
    create_qe_adapter,
    create_mock_qe_adapter,
)

__all__ = [
    "FactorEngineAdapter",
    "QuantEvaluatorAdapter",
    "FEOptionalDependencyMissing",
    "QEOptionalDependencyMissing",
    "create_fe_adapter",
    "create_qe_adapter",
    "create_mock_qe_adapter",
]

