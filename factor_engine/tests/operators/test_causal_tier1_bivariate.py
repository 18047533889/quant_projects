# -*- coding: utf-8 -*-
"""四则运算与价量 Tier-1 双变量 prefix-invariant 测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.operators.test_causal_operators import assert_bivariate_prefix_invariant, assert_prefix_invariant

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _dates(n: int):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _two(n: int = 6):
    idx = _dates(n)
    return (
        pd.DataFrame({"A": list(range(1, n + 1))}, index=idx),
        pd.DataFrame({"A": list(range(n, 0, -1))}, index=idx),
    )


def _op(name: str):
    return OperatorRegistry.get(name)


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide"])
def test_binary_ops_prefix_invariant(name: str):
    x, y = _two()
    assert_bivariate_prefix_invariant(_op(name).calculate, x, y)
