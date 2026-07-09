# -*- coding: utf-8
"""截面 / 保护性算子 golden：shape 不变 + 固定 NaN / 边界行为。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.common.cs_broadcast import cs_rank_01
from cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged


def _run(canon: str, panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None, canon
    out = op.calculate(panel.copy(), *args, **kwargs)
    assert_panel_shape_unchanged(panel, out)
    return out


def test_rank_matches_cs_rank_01(golden_panel, loaded):
    out = _run("rank", golden_panel)
    expected = cs_rank_01(golden_panel)
    pd.testing.assert_frame_equal(out, expected)
    # 两有效值行：最低 → 0，最高 → 1
    assert out.loc["2024-01-01", "A"] == 0.0
    assert out.loc["2024-01-01", "B"] == 1.0
    assert np.isnan(out.loc["2024-01-01", "C"])


def test_rank_single_valid_row_all_half(golden_panel, loaded):
    row = golden_panel.loc[["2024-01-01"]].copy()
    row.loc["2024-01-01", "B"] = np.nan
    out = _run("rank", row)
    assert out.loc["2024-01-01", "A"] == 0.5
    assert out.loc["2024-01-01", "C"] == 0.5


def test_rank_pct_two_valid_values(golden_panel, loaded):
    out = _run("rank_pct", golden_panel)
    assert out.loc["2024-01-01", "A"] == 0.5
    assert out.loc["2024-01-01", "B"] == 1.0
    assert np.isnan(out.loc["2024-01-01", "C"])


def test_c_mean_broadcasts_row_stat(golden_panel, loaded):
    out = _run("c_mean", golden_panel)
    row_mean = golden_panel.mean(axis=1, skipna=True)
    for col in golden_panel.columns:
        pd.testing.assert_series_equal(out[col], row_mean, check_names=False)


def test_c_count_broadcasts_row_count(golden_panel, loaded):
    out = _run("c_count", golden_panel)
    n = golden_panel.count(axis=1)
    for col in golden_panel.columns:
        pd.testing.assert_series_equal(out[col], n.astype(float), check_names=False)


def test_protected_div_zero_denominator(golden_panel, loaded):
    num = golden_panel.copy()
    den = golden_panel.copy() * 0.0
    op = OperatorRegistry.get("protected_div", backend="pandas_numpy")
    out = op.calculate(num, den)
    assert_panel_shape_unchanged(num, out)
    assert (out.to_numpy() == 0.0).all()


def test_zscore_ddof1_two_values(golden_panel, loaded):
    sub = golden_panel.loc[["2024-01-01"], ["A", "B"]]
    out = _run("zscore", sub)
    assert out.loc["2024-01-01", "A"] == pytest.approx(-0.707107, rel=1e-5)
    assert out.loc["2024-01-01", "B"] == pytest.approx(0.707107, rel=1e-5)
