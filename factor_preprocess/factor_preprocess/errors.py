"""
Core error taxonomy for factor_preprocess.

Follows the error hierarchy specified in CONTRACT_FREEZE_DRAFT.md Section 7.
All typed errors inherit from FactorPreprocessError base.
"""

from typing import Optional


class FactorPreprocessError(Exception):
    """Base exception for factor_preprocess."""
    pass


# ============================================================================
# Contract Errors
# ============================================================================


class ContractError(FactorPreprocessError):
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


# ============================================================================
# Capability Errors
# ============================================================================


class CapabilityError(FactorPreprocessError):
    """Requested capability is not available."""
    pass


class UnsupportedTransformError(CapabilityError):
    """Requested transform is not implemented."""
    pass


class OptionalDependencyMissing(CapabilityError):
    """Optional dependency (FA, DA, etc.) is not installed."""

    def __init__(self, package_name: str, feature_name: str):
        self.package_name = package_name
        self.feature_name = feature_name
        super().__init__(
            f"Optional dependency '{package_name}' is required for {feature_name}. "
            f"Install it with: pip install {package_name}"
        )


# ============================================================================
# Data/Evidence Errors
# ============================================================================


class DataError(FactorPreprocessError):
    """Data or evidence quality issue."""
    pass


class InsufficientObservations(DataError):
    """Not enough valid observations for fitting or transform."""
    pass


class InvalidValidityMask(DataError):
    """Validity mask is malformed or contradictory."""
    pass


class MissingFittedStateError(DataError):
    """Required fitted state is missing or unavailable."""
    pass


class MissingFittedStateError(DataError):
    """Required fitted state is missing or unavailable."""
    pass


class StaleFittedStateError(DataError):
    """Fitted state is outdated or bound to wrong fit window."""
    pass


class UnknownRegimeError(DataError):
    """Encountered a regime label that was not seen at fit time."""
    pass


class SupervisedTargetRequiredError(DataError):
    """A supervised transform requires a target/label series."""
    pass


class UnsupportedTargetError(DataError):
    """A label-free transform was given a target it does not accept."""
    pass


# ============================================================================
# Execution Errors
# ============================================================================


class ExecutionError(FactorPreprocessError):
    """Execution or computation failed."""
    pass


class NumericalFailure(ExecutionError):
    """Numerical computation failed (overflow, NaN, Inf, etc.)."""
    pass


class OverflowOrNonFiniteError(NumericalFailure):
    """Result is infinite, NaN, or overflowed."""
    pass


class BudgetExceededError(ExecutionError):
    """Computational budget or resource limit exceeded."""
    pass


class CancellationError(ExecutionError):
    """Operation was cancelled."""
    pass


# ============================================================================
# Governance Errors
# ============================================================================


class GovernanceError(FactorPreprocessError):
    """Governance policy or constraint violation."""
    pass


class FullSampleFitError(GovernanceError):
    """Full-sample fitting before split is forbidden."""
    pass


class FittedStateMismatchError(GovernanceError):
    """Fitted state does not match expected universe or features."""
    pass


class ContractChangeRequired(GovernanceError):
    """Operation requires contract schema change."""
    pass


__all__ = [
    # Base
    "FactorPreprocessError",
    # Contract
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    # Capability
    "CapabilityError",
    "UnsupportedTransformError",
    "OptionalDependencyMissing",
    # Data
    "DataError",
    "InsufficientObservations",
    "InvalidValidityMask",
    "MissingFittedStateError",
    "StaleFittedStateError",
    "UnknownRegimeError",
    "SupervisedTargetRequiredError",
    "UnsupportedTargetError",
    # Execution
    "ExecutionError",
    "NumericalFailure",
    "OverflowOrNonFiniteError",
    "BudgetExceededError",
    "CancellationError",
    # Governance
    "GovernanceError",
    "FullSampleFitError",
    "FittedStateMismatchError",
    "ContractChangeRequired",
]
