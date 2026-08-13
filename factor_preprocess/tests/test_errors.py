"""
Test error taxonomy for factor_preprocess.

Verifies:
- All error types are properly defined
- Error hierarchy is correct
- OptionalDependencyMissing has expected interface
- Errors can be raised and caught by type
"""

import pytest
from factor_preprocess.errors import (
    FactorPreprocessError,
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
    MissingFittedStateError,
    StaleFittedStateError,
    # Execution
    ExecutionError,
    NumericalFailure,
    OverflowOrNonFiniteError,
    BudgetExceededError,
    CancellationError,
    # Governance
    GovernanceError,
    FullSampleFitError,
    FittedStateMismatchError,
    ContractChangeRequired,
)


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_base_error(self):
        """All errors inherit from FactorPreprocessError."""
        assert issubclass(ContractError, FactorPreprocessError)
        assert issubclass(CapabilityError, FactorPreprocessError)
        assert issubclass(DataError, FactorPreprocessError)
        assert issubclass(ExecutionError, FactorPreprocessError)
        assert issubclass(GovernanceError, FactorPreprocessError)

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
        assert issubclass(MissingFittedStateError, DataError)
        assert issubclass(StaleFittedStateError, DataError)

    def test_execution_errors(self):
        """Execution error subtypes."""
        assert issubclass(NumericalFailure, ExecutionError)
        assert issubclass(OverflowOrNonFiniteError, NumericalFailure)
        assert issubclass(BudgetExceededError, ExecutionError)
        assert issubclass(CancellationError, ExecutionError)

    def test_governance_errors(self):
        """Governance error subtypes."""
        assert issubclass(FullSampleFitError, GovernanceError)
        assert issubclass(FittedStateMismatchError, GovernanceError)
        assert issubclass(ContractChangeRequired, GovernanceError)


class TestErrorRaising:
    """Test raising and catching typed errors."""

    def test_raise_contract_error(self):
        """Contract errors can be raised and caught."""
        with pytest.raises(ContractError):
            raise MissingInputError("values array is required")

        with pytest.raises(FactorPreprocessError):
            raise TimingContractError("fit_end_time must precede transform dates")

    def test_raise_capability_error(self):
        """Capability errors can be raised and caught."""
        with pytest.raises(CapabilityError):
            raise UnsupportedTransformError("Unknown transform type")

    def test_raise_data_error(self):
        """Data errors can be raised and caught."""
        with pytest.raises(DataError):
            raise InsufficientObservations("Need at least 60 obs for fitting")

        with pytest.raises(DataError):
            raise MissingFittedStateError("Fitted state required for this transform")

    def test_raise_execution_error(self):
        """Execution errors can be raised and caught."""
        with pytest.raises(ExecutionError):
            raise BudgetExceededError("Transform budget exhausted")

        with pytest.raises(NumericalFailure):
            raise OverflowOrNonFiniteError("Covariance matrix has Inf")

    def test_raise_governance_error(self):
        """Governance errors can be raised and caught."""
        with pytest.raises(GovernanceError):
            raise FullSampleFitError("Full-sample fitting is forbidden")

        with pytest.raises(GovernanceError):
            raise FittedStateMismatchError("Fitted state universe mismatch")


class TestOptionalDependencyMissing:
    """Test OptionalDependencyMissing error."""

    def test_constructor(self):
        """OptionalDependencyMissing stores package and feature."""
        err = OptionalDependencyMissing("factor_assets", "factor_set_loading")
        assert err.package_name == "factor_assets"
        assert err.feature_name == "factor_set_loading"
        assert "factor_assets" in str(err)
        assert "factor_set_loading" in str(err)

    def test_raise_and_catch(self):
        """Can raise and catch OptionalDependencyMissing."""
        with pytest.raises(OptionalDependencyMissing) as exc_info:
            raise OptionalDependencyMissing("data_access", "exposure_fetch")

        assert exc_info.value.package_name == "data_access"
        assert exc_info.value.feature_name == "exposure_fetch"

    def test_catch_as_capability_error(self):
        """OptionalDependencyMissing can be caught as CapabilityError."""
        with pytest.raises(CapabilityError):
            raise OptionalDependencyMissing("factor_assets", "adapter")


class TestErrorMessages:
    """Test error messages are informative."""

    def test_optional_dependency_message(self):
        """OptionalDependencyMissing has helpful message."""
        err = OptionalDependencyMissing("data_access", "industry_exposure")
        msg = str(err)
        assert "data_access" in msg
        assert "industry_exposure" in msg
        assert "pip install" in msg.lower() or "install" in msg.lower()


class TestGovernanceErrors:
    """Test governance-specific errors."""

    def test_full_sample_fit_error(self):
        """FullSampleFitError prevents forbidden operation."""
        with pytest.raises(FullSampleFitError) as exc_info:
            raise FullSampleFitError(
                "Cannot fit on full dataset before train/test split"
            )

        assert "full" in str(exc_info.value).lower()

    def test_fitted_state_mismatch(self):
        """FittedStateMismatchError detects state violations."""
        with pytest.raises(FittedStateMismatchError) as exc_info:
            raise FittedStateMismatchError(
                "Expected 500 features, got 400"
            )

        assert "mismatch" in str(exc_info.value).lower() or "500" in str(exc_info.value)
