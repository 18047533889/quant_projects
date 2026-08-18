"""
Tests for adapter to factor_preprocess.
"""
import pytest
from datetime import datetime

from modeling_adapters.adapter import FactorPreprocessAdapter, is_factor_preprocess_available
from modeling_adapters.contracts import PreprocessContract, FitWindow, TransformMode
from modeling_adapters.errors import AdapterError, FutureLeakageError


class TestFactorPreprocessAdapter:
    """Tests for FactorPreprocessAdapter."""

    pytestmark = pytest.mark.skipif(
        not is_factor_preprocess_available(),
        reason="factor_preprocess not installed",
    )

    def test_adapter_creation(self):
        """Test that adapter can be created when factor_preprocess is available."""
        adapter = FactorPreprocessAdapter()
        assert adapter is not None

    def test_translate_stateless_contract(self):
        """Test translating stateless contract."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="rank_zscore",
            transforms=[
                {"name": "rank", "kind": "cross_sectional", "mode": "stateless"},
                {"name": "zscore", "kind": "cross_sectional", "mode": "stateless"},
            ],
            mode=TransformMode.STATELESS,
        )

        fp_policy = adapter.translate_contract(contract)

        assert fp_policy.policy_id == "modeling_rank_zscore"
        assert len(fp_policy.transforms) == 2

    def test_translate_fitted_contract(self):
        """Test translating fitted contract."""
        adapter = FactorPreprocessAdapter()

        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        contract = PreprocessContract(
            contract_id="fitted_scaler",
            transforms=[
                {"name": "scaler", "kind": "cross_sectional", "mode": "fitted"},
            ],
            mode=TransformMode.FITTED,
            fit_window=fit_window,
        )

        fp_policy = adapter.translate_contract(
            contract,
            application_period_start=datetime(2021, 1, 1),
        )

        assert fp_policy.policy_id == "modeling_fitted_scaler"
        assert len(fp_policy.transforms) == 1

    def test_translate_contract_validates_no_leakage(self):
        """Test that adapter validates temporal ordering."""
        adapter = FactorPreprocessAdapter()

        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
        )

        contract = PreprocessContract(
            contract_id="fitted",
            transforms=[{"name": "scaler", "kind": "cross_sectional", "mode": "fitted"}],
            mode=TransformMode.FITTED,
            fit_window=fit_window,
        )

        # Should fail: trying to apply to period before fit_end
        with pytest.raises(FutureLeakageError):
            adapter.translate_contract(
                contract,
                application_period_start=datetime(2020, 6, 1),
            )

    def test_translate_fit_window(self):
        """Test translating FitWindow to metadata dict."""
        adapter = FactorPreprocessAdapter()

        fit_window = FitWindow(
            fit_start=datetime(2020, 1, 1),
            fit_end=datetime(2020, 12, 31),
            universe_ref="top_3000",
        )

        metadata = adapter.translate_fit_window(fit_window)

        assert metadata["fit_start_time"] == datetime(2020, 1, 1)
        assert metadata["fit_end_time"] == datetime(2020, 12, 31)
        assert metadata["fit_universe_ref"] == "top_3000"

    def test_translate_contract_with_exposure(self):
        """Test translating contract that requires exposure."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="neutralized",
            transforms=[
                {"name": "neutralize", "kind": "neutralization", "mode": "stateless"},
            ],
            mode=TransformMode.STATELESS,
            requires_industry=True,
            requires_size=True,
        )

        fp_policy = adapter.translate_contract(contract)

        # Exposure requirements should be translated exactly
        assert fp_policy.requires_industry is True
        assert fp_policy.requires_size is True


class TestAdapterUnavailable:
    """Tests for when factor_preprocess is not available."""

    @pytest.mark.skipif(
        is_factor_preprocess_available(),
        reason="factor_preprocess is installed",
    )
    def test_adapter_creation_fails_without_factor_preprocess(self):
        """Test that adapter creation fails when factor_preprocess is unavailable."""
        with pytest.raises(AdapterError, match="not available"):
            FactorPreprocessAdapter()


class TestAdapterFailClosed:
    """Tests for fail-closed behavior on unknown inputs."""

    def test_unknown_transform_kind_raises_error(self):
        """Test that unknown transform kind raises AdapterError."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="bad_kind",
            transforms=[
                {"name": "my_transform", "kind": "unknown_kind", "mode": "stateless"},
            ],
            mode=TransformMode.STATELESS,
        )

        with pytest.raises(AdapterError, match="Unknown transform kind"):
            adapter.translate_contract(contract)

    def test_unknown_transform_mode_raises_error(self):
        """Test that unknown transform mode raises AdapterError."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="bad_mode",
            transforms=[
                {"name": "my_transform", "kind": "cross_sectional", "mode": "unknown_mode"},
            ],
            mode=TransformMode.STATELESS,
        )

        with pytest.raises(AdapterError, match="Unknown transform mode"):
            adapter.translate_contract(contract)

    def test_missing_transform_name_raises_error(self):
        """Test that missing transform name raises AdapterError."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="no_name",
            transforms=[
                {"kind": "cross_sectional", "mode": "stateless"},  # Missing 'name'
            ],
            mode=TransformMode.STATELESS,
        )

        with pytest.raises(AdapterError, match="missing 'name'"):
            adapter.translate_contract(contract)

    def test_missing_transform_kind_raises_error(self):
        """Test that missing transform kind raises AdapterError."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="no_kind",
            transforms=[
                {"name": "my_transform", "mode": "stateless"},  # Missing 'kind'
            ],
            mode=TransformMode.STATELESS,
        )

        with pytest.raises(AdapterError, match="missing 'kind'"):
            adapter.translate_contract(contract)

    def test_missing_transform_mode_raises_error(self):
        """Test that missing transform mode raises AdapterError."""
        adapter = FactorPreprocessAdapter()

        contract = PreprocessContract(
            contract_id="no_mode",
            transforms=[
                {"name": "my_transform", "kind": "cross_sectional"},  # Missing 'mode'
            ],
            mode=TransformMode.STATELESS,
        )

        with pytest.raises(AdapterError, match="missing 'mode'"):
            adapter.translate_contract(contract)
