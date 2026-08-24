# -*- coding: utf-8 -*-
"""Integration tests for ts_std operator with ddof parameter."""
import pytest
import pandas as pd
import polars as pl
import numpy as np


def test_ts_std_operator_default_ddof():
    """Test TSStdNative operator with default ddof=1."""
    from factor_engine.cleaned_operators.common.polars_ts_basic import TSStdNative

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"value": data})

    op = TSStdNative()
    result = op._calculate_series(df, d=3)

    # Pandas reference with ddof=1 (default)
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=1)

    np.testing.assert_allclose(result["value"].to_numpy(), expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_operator_ddof_0():
    """Test TSStdNative operator with explicit ddof=0."""
    from factor_engine.cleaned_operators.common.polars_ts_basic import TSStdNative

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"value": data})

    op = TSStdNative()
    result = op._calculate_series(df, d=3, ddof=0)

    # Pandas reference with ddof=0
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=0)

    np.testing.assert_allclose(result["value"].to_numpy(), expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_operator_ddof_1_explicit():
    """Test TSStdNative operator with explicit ddof=1."""
    from factor_engine.cleaned_operators.common.polars_ts_basic import TSStdNative

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"value": data})

    op = TSStdNative()
    result = op._calculate_series(df, d=3, ddof=1)

    # Pandas reference with ddof=1
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=1)

    np.testing.assert_allclose(result["value"].to_numpy(), expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_operator_metadata():
    """Test TSStdNative operator metadata includes ddof parameter."""
    from factor_engine.cleaned_operators.common.polars_ts_basic import TSStdNative

    op = TSStdNative()

    # Check param_names includes ddof
    assert "ddof" in op.metadata.param_names
    assert "x" in op.metadata.param_names
    assert "d" in op.metadata.param_names

    # Check param_specs for ddof
    assert "ddof" in op.metadata.param_specs
    ddof_spec = op.metadata.param_specs["ddof"]
    assert ddof_spec.default == 1  # Default should be sample stddev
    assert ddof_spec.min == 0
    assert ddof_spec.max == 1


def test_ts_std_operator_with_nan_data():
    """Test TSStdNative operator handles NaN correctly with different ddof."""
    from factor_engine.cleaned_operators.common.polars_ts_basic import TSStdNative

    data = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]
    df = pl.DataFrame({"value": data})

    op = TSStdNative()

    # Test ddof=1
    result_ddof1 = op._calculate_series(df, d=4, ddof=1)
    s_pd = pd.Series(data)
    expected_ddof1 = s_pd.rolling(4).std(ddof=1)
    np.testing.assert_allclose(result_ddof1["value"].to_numpy(), expected_ddof1.values, rtol=1e-10, equal_nan=True)

    # Test ddof=0
    result_ddof0 = op._calculate_series(df, d=4, ddof=0)
    expected_ddof0 = s_pd.rolling(4).std(ddof=0)
    np.testing.assert_allclose(result_ddof0["value"].to_numpy(), expected_ddof0.values, rtol=1e-10, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
