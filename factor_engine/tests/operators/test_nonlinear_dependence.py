# -*- coding: utf-8 -*-
"""Public-contract tests for pairwise nonlinear-dependence operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CASES = (
    ("ts_distance_corr", {}),
    ("ts_distance_cov", {}),
    ("ts_mutual_information", {}),
    ("ts_lagged_mutual_information", {"lag": 1}),
    ("ts_upper_tail_coexceedance_probability", {"q": 0.8}),
    ("ts_lower_tail_coexceedance_probability", {"q": 0.2}),
)


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name}/pandas_numpy"
    return op


def _paired_frames(rows=90, cols=3, seed=42):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows)
    columns = [f"S{i}" for i in range(cols)]
    x = pd.DataFrame(rng.normal(size=(rows, cols)), index=index, columns=columns)
    y = pd.DataFrame(
        0.4 * x.to_numpy() + rng.normal(size=(rows, cols)), index=index, columns=columns
    )
    return x, y


@pytest.mark.parametrize(("name", "extra"), CASES)
def test_pairwise_operator_public_contract(name, extra):
    x, y = _paired_frames()
    result = _op(name).calculate(x=x, y=y, window=60, **extra)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == x.shape
    assert result.index.equals(x.index)
    assert result.columns.equals(x.columns)
    terminal = result.iloc[-1].to_numpy(dtype=float)
    assert np.all(np.isfinite(terminal))
    if "tail_coexceedance_probability" in name:
        assert np.all((terminal >= 0.0) & (terminal <= 1.0))


@pytest.mark.parametrize(("name", "extra"), CASES)
def test_pairwise_operator_nan_handling_is_deterministic(name, extra):
    x, y = _paired_frames(seed=123)
    x.iloc[25, 0] = np.nan
    y.iloc[28, 1] = np.nan
    first = _op(name).calculate(x=x, y=y, window=60, **extra)
    second = _op(name).calculate(x=x, y=y, window=60, **extra)
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-12, atol=1e-12)


def test_nonlinear_dependence_metadata_declares_both_inputs():
    for name, _extra in CASES:
        metadata = _op(name).metadata
        assert metadata.param_names[:2] == ["x", "y"]
        assert metadata.tags
