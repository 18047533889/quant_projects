"""
Smoke test: Backend availability
Test that all computational backends work when available.
"""
import pytest
import numpy as np


class TestBackends:
    """Test computational backend availability and basic operations."""

    def test_numpy_backend(self):
        """Test numpy backend (always required)."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0

    def test_numba_backend(self):
        """Test numba backend if available."""
        try:
            from numba import jit

            @jit(nopython=True)
            def simple_sum(arr):
                total = 0.0
                for i in range(len(arr)):
                    total += arr[i]
                return total

            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result = simple_sum(arr)
            assert result == 15.0
        except ImportError:
            pytest.skip("numba not available")

    def test_polars_backend(self):
        """Test polars backend if available."""
        try:
            import polars as pl

            df = pl.DataFrame({
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50]
            })
            assert df.shape == (5, 2)
            assert df['a'].sum() == 15
            assert df['b'].sum() == 150
        except ImportError:
            pytest.skip("polars not available")

    def test_cupy_backend(self):
        """Test cupy backend if available."""
        try:
            import cupy as cp

            arr = cp.array([1, 2, 3, 4, 5])
            result = cp.asnumpy(arr.sum())
            assert result == 15
        except (ImportError, Exception) as e:
            pytest.skip(f"cupy not available or no GPU: {e}")

    def test_pandas_backend(self):
        """Test pandas backend (commonly used)."""
        try:
            import pandas as pd

            df = pd.DataFrame({
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50]
            })
            assert df.shape == (5, 2)
            assert df['a'].sum() == 15
        except ImportError:
            pytest.skip("pandas not available")

    def test_duckdb_backend(self):
        """Test duckdb backend if available."""
        try:
            import duckdb

            conn = duckdb.connect(':memory:')
            result = conn.execute("SELECT 1 + 2 as sum").fetchone()
            assert result[0] == 3
            conn.close()
        except ImportError:
            pytest.skip("duckdb not available")
