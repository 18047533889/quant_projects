"""
Test error taxonomy for factor_optimizer.

Verifies:
- All error types are properly defined
- Error hierarchy is correct
- OptionalDependencyMissing has expected interface
- Errors can be raised and caught by type
"""

import pytest
from factor_optimizer.errors import (
    FactorOptimizerError,
    # Contract
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    # Capability
    CapabilityError,
    UnsupportedMutationError,
    OptionalDependencyMissing,
    # Data
    DataError,
    InsufficientObservations,
    InvalidValidityMask,
    EvidenceUnavailableError,
    StaleEvidenceError,
    # Execution
    ExecutionError,
    NumericalFailure,
    OverflowOrNonFiniteError,
    BudgetExceededError,
    CancellationError,
    # Governance
    GovernanceError,
    IllegalMutationError,
    DuplicateIdentityError,
    CollisionError,
    ContractChangeRequired,
)


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_base_error(self):
        """All errors inherit from FactorOptimizerError."""
        assert issubclass(ContractError, FactorOptimizerError)
        assert issubclass(CapabilityError, FactorOptimizerError)
        assert issubclass(DataError, FactorOptimizerError)
        assert issubclass(ExecutionError, FactorOptimizerError)
        assert issubclass(GovernanceError, FactorOptimizerError)

    def test_contract_errors(self):
        """Contract error subtypes."""
        assert issubclass(SchemaVersionError, ContractError)
        assert issubclass(MissingInputError, ContractError)
        assert issubclass(InvalidContractError, ContractError)
        assert issubclass(TimingContractError, ContractError)
        assert issubclass(SnapshotMismatchError, ContractError)

    def test_capability_errors(self):
        """Capability error subtypes."""
        assert issubclass(UnsupportedMutationError, CapabilityError)
        assert issubclass(OptionalDependencyMissing, CapabilityError)

    def test_data_errors(self):
        """Data error subtypes."""
        assert issubclass(InsufficientObservations, DataError)
        assert issubclass(InvalidValidityMask, DataError)
        assert issubclass(EvidenceUnavailableError, DataError)
        assert issubclass(StaleEvidenceError, DataError)

    def test_execution_errors(self):
        """Execution error subtypes."""
        assert issubclass(NumericalFailure, ExecutionError)
        assert issubclass(OverflowOrNonFiniteError, NumericalFailure)
        assert issubclass(BudgetExceededError, ExecutionError)
        assert issubclass(CancellationError, ExecutionError)

    def test_governance_errors(self):
        """Governance error subtypes."""
        assert issubclass(IllegalMutationError, GovernanceError)
        assert issubclass(DuplicateIdentityError, GovernanceError)
        assert issubclass(CollisionError, GovernanceError)
        assert issubclass(ContractChangeRequired, GovernanceError)


class TestErrorRaising:
    """Test raising and catching typed errors."""

    def test_raise_contract_error(self):
        """Contract errors can be raised and caught."""
        with pytest.raises(ContractError):
            raise MissingInputError("factor_id is required")

        with pytest.raises(FactorOptimizerError):
            raise TimingContractError("timing mismatch")

    def test_raise_capability_error(self):
        """Capability errors can be raised and caught."""
        with pytest.raises(CapabilityError):
            raise UnsupportedMutationError("Unknown mutation type")

    def test_raise_data_error(self):
        """Data errors can be raised and caught."""
        with pytest.raises(DataError):
            raise InsufficientObservations("Need at least 100 obs")

    def test_raise_execution_error(self):
        """Execution errors can be raised and caught."""
        with pytest.raises(ExecutionError):
            raise BudgetExceededError("Search budget exhausted")

        with pytest.raises(NumericalFailure):
            raise OverflowOrNonFiniteError("Result is NaN")

    def test_raise_governance_error(self):
        """Governance errors can be raised and caught."""
        with pytest.raises(GovernanceError):
            raise IllegalMutationError("Mutation violates grammar")


class TestOptionalDependencyMissing:
    """Test OptionalDependencyMissing error."""

    def test_constructor(self):
        """OptionalDependencyMissing stores package and feature."""
        err = OptionalDependencyMissing("factor_engine", "canonical_hash")
        assert err.package_name == "factor_engine"
        assert err.feature_name == "canonical_hash"
        assert "factor_engine" in str(err)
        assert "canonical_hash" in str(err)

    def test_raise_and_catch(self):
        """Can raise and catch OptionalDependencyMissing."""
        with pytest.raises(OptionalDependencyMissing) as exc_info:
            raise OptionalDependencyMissing("quant_evaluator", "evaluation")

        assert exc_info.value.package_name == "quant_evaluator"
        assert exc_info.value.feature_name == "evaluation"

    def test_catch_as_capability_error(self):
        """OptionalDependencyMissing can be caught as CapabilityError."""
        with pytest.raises(CapabilityError):
            raise OptionalDependencyMissing("factor_engine", "validate")


class TestErrorMessages:
    """Test error messages are informative."""

    def test_optional_dependency_message(self):
        """OptionalDependencyMissing has helpful message."""
        err = OptionalDependencyMissing("factor_engine", "mutation_validation")
        msg = str(err)
        assert "factor_engine" in msg
        assert "mutation_validation" in msg
        assert "pip install" in msg.lower() or "install" in msg.lower()
