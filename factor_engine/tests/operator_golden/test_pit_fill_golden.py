# -*- coding: utf-8
"""填充算子 PIT / 因果 golden：禁前视、shape 保持。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy fill primitives were removed from the production runtime")

from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged, assert_prefix_invariant


def _run(canon: str, panel: pd.DataFrame, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None
    out = op.calculate(panel.copy(), **kwargs)
    assert_panel_shape_unchanged(panel, out)
    return out


def test_ffill_prefix_invariant(loaded):
    panel = pd.DataFrame(
        {"A": [np.nan, 1.0, np.nan, 3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    assert_prefix_invariant(lambda df: _run("ffill", df), panel)


def test_causal_linear_extrapolate_is_prefix_invariant(loaded):
    panel = pd.DataFrame(
        {"A": [np.nan, 2.0, 3.0, np.nan, 5.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    out = _run("causal_linear_extrapolate", panel)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[3, 0] == pytest.approx(4.0)
    assert_prefix_invariant(lambda df: _run("causal_linear_extrapolate", df), panel)


def test_fillna_const_preserves_shape_with_nan(loaded, golden_panel):
    out = _run("fillna_const", golden_panel, value=0.0)
    assert out.loc["2024-01-01", "C"] == 0.0
    assert out.loc["2024-01-03", "A"] == 0.0
    assert out.loc["2024-01-02", "A"] == pytest.approx(2.0)


def test_coalesce_prefers_left_non_nan(loaded):
    left = pd.DataFrame({"A": [np.nan, 2.0, 3.0]})
    right = pd.DataFrame({"A": [1.0, 99.0, 99.0]})
    op = OperatorRegistry.get("coalesce")
    out = op.calculate(left, right)
    assert out.iloc[0, 0] == pytest.approx(1.0)
    assert out.iloc[1, 0] == pytest.approx(2.0)
