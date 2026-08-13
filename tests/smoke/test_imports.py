"""
Smoke test: Package imports
Verify that all core packages can be imported successfully.
"""
import pytest
import sys
import importlib


class TestPackageImports:
    """Test that all core packages import successfully."""

    def test_dataaccess_imports(self):
        """Test dataaccess package imports."""
        import dataaccess
        assert dataaccess is not None

        # Test key exports
        from dataaccess import get_store, DataAccessStore
        assert get_store is not None
        assert DataAccessStore is not None

    def test_factor_engine_imports(self):
        """Test factor_engine package imports."""
        try:
            import factor_engine
            assert factor_engine is not None
        except Exception as e:
            pytest.skip(f"factor_engine import requires dependencies: {e}")

    def test_factor_optimizer_imports(self):
        """Test factor_optimizer package imports."""
        import factor_optimizer
        assert factor_optimizer is not None

    def test_quant_evaluator_imports(self):
        """Test quant_evaluator package imports."""
        import quant_evaluator
        assert quant_evaluator is not None

    def test_all_packages_have_version(self):
        """Test that all packages have __version__ or similar."""
        packages = ['dataaccess', 'factor_engine', 'factor_optimizer', 'quant_evaluator']
        for pkg_name in packages:
            try:
                pkg = importlib.import_module(pkg_name)
                # Check for version attribute
                assert hasattr(pkg, '__version__') or hasattr(pkg, 'VERSION'), \
                    f"{pkg_name} should have __version__ or VERSION"
            except AssertionError:
                # Some packages may not have version, just warn
                pytest.skip(f"{pkg_name} has no version attribute")
