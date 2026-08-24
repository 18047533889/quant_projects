"""
Core error taxonomy for factor_assets.

Follows the error hierarchy specified in CONTRACT_FREEZE_DRAFT.md Section 7.
All typed errors inherit from FactorAssetsError base.
"""

from typing import Optional


class FactorAssetsError(Exception):
    """Base exception for factor_assets."""
    pass


# ============================================================================
# Contract Errors
# ============================================================================


class ContractError(FactorAssetsError):
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


class CapabilityError(FactorAssetsError):
    """Requested capability is not available."""
    pass


class UnsupportedTransformError(CapabilityError):
    """Requested transform or aggregation is not implemented."""
    pass


class OptionalDependencyMissing(CapabilityError):
    """Optional dependency (QE, FE, DA, etc.) is not installed."""

    def __init__(self, package_name: str, adapter_name: str):
        self.package_name = package_name
        self.adapter_name = adapter_name
        super().__init__(
            f"Adapter '{adapter_name}' requires optional package '{package_name}'. "
            f"Install with: pip install factor_assets[adapters]"
        )


# ============================================================================
# Data/Evidence Errors
# ============================================================================


class DataError(FactorAssetsError):
    """Data or evidence quality issue."""
    pass


class InsufficientObservations(DataError):
    """Not enough valid observations for analysis."""
    pass


class InvalidValidityMask(DataError):
    """Validity mask is malformed or contradictory."""
    pass


class EvidenceUnavailableError(DataError):
    """Evidence cannot be computed or retrieved."""
    pass


class StaleEvidenceError(DataError):
    """Evidence is outdated or bound to stale state."""
    pass


# ============================================================================
# Execution Errors
# ============================================================================


class ExecutionError(FactorAssetsError):
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


class GovernanceError(FactorAssetsError):
    """Governance policy or constraint violation."""
    pass


class LifecycleConflictError(GovernanceError):
    """Factor lifecycle state conflict."""
    pass


class DuplicateIdentityError(GovernanceError):
    """Factor identity already exists."""
    pass


class CollisionError(GovernanceError):
    """Hash or identity collision detected."""
    pass


class ContractChangeRequired(GovernanceError):
    """Operation requires contract schema change."""
    pass


__all__ = [
    # Base
    "FactorAssetsError",
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
    "EvidenceUnavailableError",
    "StaleEvidenceError",
    # Execution
    "ExecutionError",
    "NumericalFailure",
    "OverflowOrNonFiniteError",
    "BudgetExceededError",
    "CancellationError",
    # Governance
    "GovernanceError",
    "LifecycleConflictError",
    "DuplicateIdentityError",
    "CollisionError",
    "ContractChangeRequired",
]
