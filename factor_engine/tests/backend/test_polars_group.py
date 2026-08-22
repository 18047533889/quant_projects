# -*- coding: utf-8 -*-
"""Parity tests for cross-sectional group / robust-regression Polars backends."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(333)
    index = pd.date_range("2022-01-01", periods=80, freq="D")
    x = pd.DataFrame(rng.normal(0.5, 2.0, (80, 4)), index=index, columns=["A", "B", "C", "D"])
    group = pd.DataFrame(
        rng.integers(0, 3, (80, 4)), index=index, columns=["A", "B", "C", "D"]
    ).astype(float)
    subgroup = pd.DataFrame(
        rng.integers(0, 2, (80, 4)), index=index, columns=["A", "B", "C", "D"]
    ).astype(float)
    weight = pd.DataFrame(rng.uniform(0.5, 2.0, (80, 4)), index=index, columns=["A", "B", "C", "D"])
    y = x * 0.7 + rng.normal(0, 0.5, (80, 4))
    return x, group, subgroup, weight, y


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_native_polars_group_matches_pandas(panels):
    x, g, sg, w, y = panels
    cases = [
        ("group_ex_self_mean", (x, g), {}),
        ("group_ex_self_weighted_mean", (x, w, g), {}),
        ("hierarchical_group_neutralize", (x, g, sg), {}),
        ("cs_robust_resid", (y, x), {"trim_ratio": 0.1, "add_intercept": True}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)


def test_native_polars_group_nan_warmup_matches_pandas():
    rng = np.random.default_rng(44)
    index = pd.date_range("2023-01-01", periods=60, freq="D")
    mask = rng.random((60, 4)) < 0.1
    x = pd.DataFrame(rng.normal(0.5, 2.0, (60, 4)), index=index, columns=["A", "B", "C", "D"])
    x[mask] = np.nan
    g = pd.DataFrame(rng.integers(0, 3, (60, 4)), index=index, columns=["A", "B", "C", "D"]).astype(float)
    sg = pd.DataFrame(rng.integers(0, 2, (60, 4)), index=index, columns=["A", "B", "C", "D"]).astype(float)
    w = pd.DataFrame(rng.uniform(0.5, 2.0, (60, 4)), index=index, columns=["A", "B", "C", "D"])
    y = x * 0.7 + rng.normal(0, 0.5, (60, 4))
    cases = [
        ("group_ex_self_mean", (x, g), {}),
        ("group_ex_self_weighted_mean", (x, w, g), {}),
        ("hierarchical_group_neutralize", (x, g, sg), {}),
        ("cs_robust_resid", (y, x), {"trim_ratio": 0.1, "add_intercept": True}),
    ]
    for name, args, kwargs in cases:
        _assert_parity(name, args, kwargs)
