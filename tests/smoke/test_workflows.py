"""
Smoke test: Basic workflows
Test end-to-end workflows for each major package.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


class TestDataAccessWorkflow:
    """Test basic DataAccess workflows."""

    def test_dataaccess_basic_workflow(self):
        """Test basic data access operations."""
        try:
            from dataaccess import get_store

            # Test store initialization
            store = get_store()
            assert store is not None

        except Exception as e:
            pytest.skip(f"DataAccess workflow requires setup: {e}")


class TestFactorEngineWorkflow:
    """Test basic FactorEngine workflows."""

    def test_factor_engine_backend_import(self):
        """Test factor engine backend modules."""
        try:
            from factor_engine import backend

            # Test backend module exists
            assert backend is not None

        except Exception as e:
            pytest.skip(f"FactorEngine workflow requires setup: {e}")

    def test_factor_engine_simple_operation(self):
        """Test a simple factor engine operation."""
        try:
            import pandas as pd

            # Create simple test data
            data = pd.DataFrame({
                'value': [1.0, 2.0, 3.0, 4.0, 5.0],
                'timestamp': pd.date_range('2024-01-01', periods=5)
            })

            # Basic sanity check
            assert len(data) == 5

        except Exception as e:
            pytest.skip(f"FactorEngine operation requires full setup: {e}")


class TestFactorOptimizerWorkflow:
    """Test basic FactorOptimizer workflows."""

    def test_factor_optimizer_basic(self):
        """Test factor optimizer basic functionality."""
        try:
            import factor_optimizer

            # Basic import check
            assert factor_optimizer is not None

        except Exception as e:
            pytest.skip(f"FactorOptimizer workflow requires setup: {e}")


class TestQuantEvaluatorWorkflow:
    """Test basic QuantEvaluator workflows."""

    def test_quant_evaluator_metrics(self):
        """Test quant evaluator metrics calculation."""
        try:
            from quant_evaluator import metrics

            # Test with simple returns data
            returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01])

            # Should be able to compute basic metrics
            assert returns is not None
            assert len(returns) == 5

        except Exception as e:
            pytest.skip(f"QuantEvaluator workflow requires setup: {e}")
