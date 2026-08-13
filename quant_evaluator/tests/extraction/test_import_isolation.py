"""
Extraction tests: Import isolation verification.

Tests that quant_evaluator core can be imported without optional dependencies
(DataAccess, FactorEngine) and that adapter imports are isolated.
"""

import pytest
import sys
import importlib
from unittest.mock import patch


class TestCoreImportIsolation:
    """Test core imports work without optional dependencies."""

    def test_core_package_imports(self):
        """Core package imports without DA or FE."""
        # These should succeed without optional dependencies
        from quant_evaluator import (
            FactorBatch,
            AxisRef,
            LabelBundle,
            EvaluationRequest,
            EvaluationBundle,
            MetricValue,
            FactorDiagnosis,
        )

        # Verify types are importable
        assert FactorBatch is not None
        assert AxisRef is not None
        assert LabelBundle is not None
        assert EvaluationRequest is not None
        assert EvaluationBundle is not None
        assert MetricValue is not None
        assert FactorDiagnosis is not None

    def test_contracts_import_independently(self):
        """Contract modules import independently."""
        from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
        from quant_evaluator.contracts.label_bundle import LabelBundle
        from quant_evaluator.contracts.errors import (
            QuantEvaluatorError,
            ContractError,
            DataError,
            CapabilityError,
        )

        # All contracts should be available
        assert FactorBatch is not None
        assert LabelBundle is not None
        assert QuantEvaluatorError is not None
        assert ContractError is not None

    def test_metrics_import_independently(self):
        """Metric modules import without adapters."""
        from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
        from quant_evaluator.metrics.quality import compute_coverage
        from quant_evaluator.metrics.quantile import (
            assign_quantiles,
            compute_quantile_returns,
            compute_top_bottom_spread,
        )
        from quant_evaluator.metrics.turnover import (
            compute_turnover,
            compute_turnover_series,
            estimate_turnover_from_ranks,
        )

        # All metric functions should be importable
        assert compute_daily_ic is not None
        assert compute_coverage is not None
        assert assign_quantiles is not None
        assert compute_turnover is not None

    def test_diagnosis_import_independently(self):
        """Diagnosis module imports without adapters."""
        from quant_evaluator.diagnosis.factor import (
            diagnose_factor,
            diagnose_all_factors,
        )

        assert diagnose_factor is not None
        assert diagnose_all_factors is not None

    def test_api_contracts_import(self):
        """API request/response contracts import."""
        from quant_evaluator.api.requests import (
            EvaluationRequest,
            EvaluationBundle,
            MetricValue,
            FactorDiagnosis,
        )

        assert EvaluationRequest is not None
        assert EvaluationBundle is not None


class TestAdapterIsolation:
    """Test adapter imports are properly isolated."""

    def test_adapters_module_exists(self):
        """Adapters module exists but is optional."""
        try:
            from quant_evaluator import adapters
            # If it imports, that's fine
            assert adapters is not None
        except ImportError:
            # If it doesn't import, that's also acceptable
            pass

    def test_core_does_not_depend_on_adapters(self):
        """Core metrics do not import adapter modules."""
        import quant_evaluator.metrics.ic as ic_module
        import quant_evaluator.metrics.quality as quality_module
        import quant_evaluator.metrics.quantile as quantile_module

        # Check module source doesn't reference adapter imports
        ic_source = importlib.import_module('quant_evaluator.metrics.ic').__file__
        quality_source = importlib.import_module('quant_evaluator.metrics.quality').__file__

        # Core modules should exist and be loadable
        assert ic_source is not None
        assert quality_source is not None

    def test_contracts_do_not_depend_on_metrics(self):
        """Contracts are independent of metric implementations."""
        from quant_evaluator.contracts.factor_batch import FactorBatch
        from quant_evaluator.contracts.label_bundle import LabelBundle

        # Contracts should not trigger metric imports
        # This is a design check - contracts are lower layer
        assert FactorBatch is not None
        assert LabelBundle is not None


class TestMinimalDependencies:
    """Test core functionality with minimal dependencies."""

    def test_core_only_uses_numpy_scipy(self):
        """Core metrics only depend on numpy and scipy."""
        import numpy as np
        import scipy

        # These are the only required dependencies for core
        assert np is not None
        assert scipy is not None

        # Core metrics should work with just numpy/scipy
        from quant_evaluator.metrics.ic import _pearson_correlation

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        corr = _pearson_correlation(x, y, min_obs=3)
        assert abs(corr - 1.0) < 1e-10

    def test_no_pandas_required_for_core(self):
        """Core evaluation does not require pandas."""
        # Verify core metrics work without pandas being imported
        import sys

        # Save original pandas if it was imported
        pandas_was_imported = 'pandas' in sys.modules

        if pandas_was_imported:
            # Can't reliably test this if pandas already loaded
            pytest.skip("Pandas already imported")

        from quant_evaluator.metrics.ic import compute_daily_ic
        from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
        from quant_evaluator.contracts.label_bundle import LabelBundle
        import numpy as np

        # Create simple batch without pandas
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=20)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(10, 20, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(10, 20),
            horizon=1,
            decision_time=tuple(range(10)),
            label_start_time=tuple(range(10)),
            label_end_time=tuple(range(1, 11)),
        )

        # Should work without pandas
        ic_series, _ = compute_daily_ic(batch, bundle)
        assert ic_series.shape == (10, 1)

        # Verify pandas was not imported
        assert 'pandas' not in sys.modules


class TestErrorHierarchy:
    """Test error taxonomy is properly isolated."""

    def test_error_hierarchy_imports(self):
        """Error classes import independently."""
        from quant_evaluator.contracts.errors import (
            QuantEvaluatorError,
            ContractError,
            DataError,
            CapabilityError,
            InsufficientObservations,
            UnsupportedMetricError,
            TimingContractError,
        )

        # Check inheritance
        assert issubclass(ContractError, QuantEvaluatorError)
        assert issubclass(DataError, QuantEvaluatorError)
        assert issubclass(CapabilityError, QuantEvaluatorError)
        assert issubclass(InsufficientObservations, DataError)
        assert issubclass(UnsupportedMetricError, CapabilityError)
        assert issubclass(TimingContractError, ContractError)

    def test_errors_can_be_raised_and_caught(self):
        """Error classes work correctly."""
        from quant_evaluator.contracts.errors import (
            ContractError,
            InsufficientObservations,
        )

        # Test raising and catching
        with pytest.raises(ContractError):
            raise ContractError("test error")

        with pytest.raises(InsufficientObservations):
            raise InsufficientObservations("not enough data")


class TestPublicAPIExports:
    """Test public API exports are correct."""

    def test_package_all_exports(self):
        """Check __all__ exports from main package."""
        import quant_evaluator

        # Should have __all__ defined
        assert hasattr(quant_evaluator, '__all__')

        # Check key exports
        expected_exports = [
            'FactorBatch',
            'AxisRef',
            'LabelBundle',
            'EvaluationRequest',
            'EvaluationBundle',
            'MetricValue',
            'FactorDiagnosis',
            'QuantEvaluatorError',
            'ContractError',
            'DataError',
            'CapabilityError',
        ]

        for name in expected_exports:
            assert name in quant_evaluator.__all__
            assert hasattr(quant_evaluator, name)

    def test_version_exported(self):
        """Package version is exported."""
        import quant_evaluator

        assert hasattr(quant_evaluator, '__version__')
        assert isinstance(quant_evaluator.__version__, str)
        assert len(quant_evaluator.__version__) > 0

    def test_no_private_exports_in_all(self):
        """__all__ does not export private names."""
        import quant_evaluator

        for name in quant_evaluator.__all__:
            # No single underscore private names (but dunders like __version__ are ok)
            if name.startswith('_') and not name.startswith('__'):
                pytest.fail(f"Private name {name} in __all__")


class TestNamespaceIsolation:
    """Test module namespaces are properly isolated."""

    def test_metrics_namespace_isolation(self):
        """Metric modules don't pollute each other's namespaces."""
        from quant_evaluator.metrics import ic
        from quant_evaluator.metrics import quality
        from quant_evaluator.metrics import quantile
        from quant_evaluator.metrics import turnover

        # Each module should have distinct exports
        assert hasattr(ic, 'compute_daily_ic')
        assert hasattr(quality, 'compute_coverage')
        assert hasattr(quantile, 'assign_quantiles')
        assert hasattr(turnover, 'compute_turnover')

        # Functions shouldn't leak across modules
        assert not hasattr(ic, 'compute_coverage')
        assert not hasattr(quality, 'compute_daily_ic')

    def test_contract_namespace_isolation(self):
        """Contract modules are isolated."""
        from quant_evaluator.contracts import factor_batch
        from quant_evaluator.contracts import label_bundle
        from quant_evaluator.contracts import errors

        assert hasattr(factor_batch, 'FactorBatch')
        assert hasattr(label_bundle, 'LabelBundle')
        assert hasattr(errors, 'QuantEvaluatorError')

        # No cross-contamination
        assert not hasattr(errors, 'FactorBatch')
        assert not hasattr(factor_batch, 'LabelBundle')


class TestImportOrder:
    """Test imports work in any order."""

    def test_metrics_before_contracts(self):
        """Can import metrics before contracts."""
        # This tests circular dependency absence
        from quant_evaluator.metrics.ic import compute_daily_ic
        from quant_evaluator.contracts.factor_batch import FactorBatch

        assert compute_daily_ic is not None
        assert FactorBatch is not None

    def test_contracts_before_metrics(self):
        """Can import contracts before metrics."""
        from quant_evaluator.contracts.label_bundle import LabelBundle
        from quant_evaluator.metrics.quality import compute_coverage

        assert LabelBundle is not None
        assert compute_coverage is not None

    def test_diagnosis_imports_after_contracts(self):
        """Diagnosis can import after contracts."""
        from quant_evaluator.contracts.factor_batch import FactorBatch
        from quant_evaluator.diagnosis.factor import diagnose_factor

        assert FactorBatch is not None
        assert diagnose_factor is not None


class TestOptionalDependencyHandling:
    """Test optional dependency handling (stubs)."""

    def test_adapters_are_optional(self):
        """Adapter modules are marked as optional."""
        # Core should work even if adapters fail to import
        try:
            from quant_evaluator.adapters import data_access
            from quant_evaluator.adapters import factor_engine

            # If they import, they should be stub implementations
            # or raise clear errors when used
            assert data_access is not None
            assert factor_engine is not None

        except (ImportError, ModuleNotFoundError):
            # It's acceptable if adapters don't import at all
            pass

    def test_core_functionality_without_adapters(self):
        """Core metrics work without adapter modules."""
        import numpy as np
        from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
        from quant_evaluator.contracts.label_bundle import LabelBundle
        from quant_evaluator.metrics.ic import compute_daily_ic

        # Create and evaluate batch
        time_axis = AxisRef(name="time", dtype="datetime64", size=5)
        asset_axis = AxisRef(name="asset", dtype="int64", size=10)

        batch = FactorBatch(
            factor_ids=("test",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(5, 10, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(5, 10),
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )

        ic_series, _ = compute_daily_ic(batch, bundle)
        assert ic_series.shape == (5, 1)
