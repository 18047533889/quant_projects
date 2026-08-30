"""
Core error taxonomy for factor_optimizer.

Follows the error hierarchy specified in CONTRACT_FREEZE_DRAFT.md Section 7.
All typed errors inherit from FactorOptimizerError base.
"""

from typing import Optional


class FactorOptimizerError(Exception):
    """Base exception for factor_optimizer."""
    pass


# ============================================================================
# Contract Errors
# ============================================================================


class ContractError(FactorOptimizerError):
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


class CapabilityError(FactorOptimizerError):
    """Requested capability is not available."""
    pass


class CapabilityForgeryError(CapabilityError):
    """A capability's identity/authorization fields were forged or altered."""
    pass


class UnsupportedMutationError(CapabilityError):
    """Requested mutation type is not implemented."""
    pass


class OptionalDependencyMissing(CapabilityError):
    """Optional dependency (FE, QE, etc.) is not installed."""

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


class DataError(FactorOptimizerError):
    """Data or evidence quality issue."""
    pass


class InsufficientObservations(DataError):
    """Not enough valid observations for optimization."""
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


class ExecutionError(FactorOptimizerError):
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
    """Search or optimization was cancelled."""
    pass


# ============================================================================
# Governance Errors
# ============================================================================


class GovernanceError(FactorOptimizerError):
    """Governance policy or constraint violation."""
    pass


class IllegalMutationError(GovernanceError):
    """Mutation violates FE legality or grammar constraints."""
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


class TreatmentIntegrityError(GovernanceError):
    """Treatment integrity evidence is missing, stale, tampered, or failing.

    R55 P0-9: the integrity gates of the treatment pipeline are fail-closed —
    a treatment candidate may be scored / admitted only against a complete,
    self-consistent, passing
    :class:`factor_optimizer.contracts.treatment_integrity.TreatmentIntegrityEvidence`.
    Anything else (no evidence at all, evidence bound to another treatment,
    a tampered payload, NOT_RUN, or a failed check) rejects the candidate.
    """
    pass


__all__ = [
    # Base
    "FactorOptimizerError",
    # Contract
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    # Capability
    "CapabilityError",
    "CapabilityForgeryError",
    "UnsupportedMutationError",
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
    "IllegalMutationError",
    "DuplicateIdentityError",
    "CollisionError",
    "ContractChangeRequired",
    "TreatmentIntegrityError",
]
