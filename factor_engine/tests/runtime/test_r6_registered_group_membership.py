"""Validate missing-membership behavior through final registered backends."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("missing", [None, "", np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_registered_missing_memberships(weighted, missing, backend):
    x = pd.DataFrame([[1., 3., 10., 20.]], columns=list("abcd"))
    valid = "A" if missing is None or isinstance(missing, str) else 1.
    group = pd.DataFrame([[missing, missing, valid, valid]], columns=x.columns)
    panels = [x, x * 0 + 1, group] if weighted else [x, group]
    name = "group_ex_self_weighted_mean" if weighted else "group_ex_self_mean"
    op = OperatorRegistry.get(name, backend, mode="research")
    assert op is not None
    if backend == "polars":
        panels = [pl.from_pandas(p) for p in panels]
    result = op.calculate(*panels)
    if backend == "polars":
        result = result.to_pandas()
    expected = pd.DataFrame([[np.nan, np.nan, 20., 10.]], columns=x.columns)
    pd.testing.assert_frame_equal(result, expected)
