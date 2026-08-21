"""
Core error taxonomy for quant_evaluator.
"""

from typing import Optional


class QuantEvaluatorError(Exception):
    """Base exception for quant_evaluator."""
    pass


class ContractError(QuantEvaluatorError):
    """Data contract or schema violation."""
    pass


class SchemaVersionError(ContractError):
    """Schema version mismatch or unsupported version."""
    pass


class MissingInputError(ContractError):
    """Required input field is missing."""
    pass


class InvalidContractError(ContractError):
    """Input violates a contract constraint."""
    pass


class TimingContractError(ContractError):
    """Timing/ordering constraint violated."""
    pass


class SnapshotMismatchError(ContractError):
    """Snapshot or context reference mismatch."""
    pass


class CapabilityError(QuantEvaluatorError):
    """Requested capability is not available."""
    pass


class UnsupportedMetricError(CapabilityError):
    """Requested metric is not implemented."""
    pass


class OptionalDependencyMissing(CapabilityError):
    """Optional dependency (DA, FE, etc.) is not installed."""
    pass


class DurableCacheCapabilityError(CapabilityError):
    """Requested cache persistence is unavailable or unsafe for the runtime mode."""
    pass


class DataError(QuantEvaluatorError):
    """Data or evidence quality issue."""
    pass


class InsufficientObservations(DataError):
    """Not enough valid observations for metric computation."""
    pass


class InvalidValidityMask(DataError):
    """Validity mask is malformed or contradictory."""
    pass


class MissingLabelError(DataError):
    """Label data is missing or incomplete."""
    pass


class EvidenceUnavailableError(DataError):
    """Evidence cannot be computed or retrieved."""
    pass


class NumericalFailure(QuantEvaluatorError):
    """Numerical computation failed (overflow, NaN, Inf, etc.)."""
    pass


class OverflowOrNonFiniteError(NumericalFailure):
    """Result is infinite, NaN, or overflowed."""
    pass
