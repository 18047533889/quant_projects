"""
Test error taxonomy for factor_assets.

Verifies:
- All error types are properly defined
- Error hierarchy is correct
- OptionalDependencyMissing has expected interface
- Errors can be raised and caught by type
"""

import pytest
from factor_assets.errors import (
    FactorAssetsError,
    # Contract
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    # Capability
    CapabilityError,
    UnsupportedTransformError,
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
    LifecycleConflictError,
    DuplicateIdentityError,
    CollisionError,
    ContractChangeRequired,
)


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_base_error(self):
        """All errors inherit from FactorAssetsError."""
        assert issubclass(ContractError, FactorAssetsError)
        assert issubclass(CapabilityError, FactorAssetsError)
        assert issubclass(DataError, FactorAssetsError)
        assert issubclass(ExecutionError, FactorAssetsError)
        assert issubclass(GovernanceError, FactorAssetsError)

    def test_contract_errors(self):
        """Contract error subtypes."""
        assert issubclass(SchemaVersionError, ContractError)
        assert issubclass(MissingInputError, ContractError)
        assert issubclass(InvalidContractError, ContractError)
        assert issubclass(TimingContractError, ContractError)
        assert issubclass(SnapshotMismatchError, ContractError)

    def test_capability_errors(self):
        """Capability error subtypes."""
        assert issubclass(UnsupportedTransformError, CapabilityError)
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
        assert issubclass(LifecycleConflictError, GovernanceError)
        assert issubclass(DuplicateIdentityError, GovernanceError)
        assert issubclass(CollisionError, GovernanceError)
        assert issubclass(ContractChangeRequired, GovernanceError)


class TestErrorRaising:
    """Test raising and catching typed errors."""

    def test_raise_contract_error(self):
        """Contract errors can be raised and caught."""
        with pytest.raises(ContractError):
            raise MissingInputError("factor_id is required")

        with pytest.raises(FactorAssetsError):
            raise TimingContractError("timing mismatch")

    def test_raise_capability_error(self):
        """Capability errors can be raised and caught."""
        with pytest.raises(CapabilityError):
            raise UnsupportedTransformError("Unknown aggregation type")

    def test_raise_data_error(self):
        """Data errors can be raised and caught."""
        with pytest.raises(DataError):
            raise InsufficientObservations("Need at least 100 obs")

        with pytest.raises(DataError):
            raise StaleEvidenceError("Evidence bound to stale HEAD")

    def test_raise_execution_error(self):
        """Execution errors can be raised and caught."""
        with pytest.raises(ExecutionError):
            raise BudgetExceededError("Similarity search budget exceeded")

        with pytest.raises(NumericalFailure):
            raise OverflowOrNonFiniteError("Distance is Inf")

    def test_raise_governance_error(self):
        """Governance errors can be raised and caught."""
        with pytest.raises(GovernanceError):
            raise LifecycleConflictError("Cannot retire production asset")

        with pytest.raises(GovernanceError):
            raise DuplicateIdentityError("Factor ID already exists")


class TestOptionalDependencyMissing:
    """Test OptionalDependencyMissing error."""

    def test_constructor(self):
        """OptionalDependencyMissing stores package and adapter."""
        err = OptionalDependencyMissing("quant_evaluator", "qe_adapter")
        assert err.package_name == "quant_evaluator"
        assert err.adapter_name == "qe_adapter"
        assert "quant_evaluator" in str(err)
        assert "qe_adapter" in str(err)

    def test_raise_and_catch(self):
        """Can raise and catch OptionalDependencyMissing."""
        with pytest.raises(OptionalDependencyMissing) as exc_info:
            raise OptionalDependencyMissing("factor_engine", "fe_adapter")

        assert exc_info.value.package_name == "factor_engine"
        assert exc_info.value.adapter_name == "fe_adapter"

    def test_catch_as_capability_error(self):
        """OptionalDependencyMissing can be caught as CapabilityError."""
        with pytest.raises(CapabilityError):
            raise OptionalDependencyMissing("data_access", "da_adapter")


class TestErrorMessages:
    """Test error messages are informative."""

    def test_optional_dependency_message(self):
        """OptionalDependencyMissing has helpful message."""
        err = OptionalDependencyMissing("quant_evaluator", "evidence_provider")
        msg = str(err)
        assert "quant_evaluator" in msg
        assert "evidence_provider" in msg
        assert "adapters" in msg.lower()
