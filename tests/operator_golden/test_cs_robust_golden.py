# -*- coding: utf-8
"""截面 robust / 分位数算子 golden。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged


def _run(canon: str, panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None
    out = op.calculate(panel.copy(), *args, **kwargs)
    assert_panel_shape_unchanged(panel, out)
    return out


def test_c_percentile_broadcasts_row_quantile(golden_panel, loaded):
    out = _run("c_percentile", golden_panel, 0.5)
    for ts in golden_panel.index:
        row = golden_panel.loc[ts].dropna()
        if len(row) == 0:
            continue
        expected = row.quantile(0.5)
        for col in golden_panel.columns:
            assert out.loc[ts, col] == pytest.approx(expected, rel=1e-9)


def test_cs_mad_broadcasts_row_mad(golden_panel, loaded):
    out = _run("cs_mad", golden_panel)
    row = golden_panel.loc["2024-01-03"].dropna()
    med = row.median()
    mad = (row - med).abs().median()
    for col in golden_panel.columns:
        assert out.loc["2024-01-03", col] == pytest.approx(mad, rel=1e-9)


def test_cs_mad_zscore_equal_row_is_nan(golden_panel, loaded):
    row_panel = golden_panel.loc[["2024-01-02"], ["A", "B"]].copy()
    row_panel.loc["2024-01-02", "A"] = 2.0
    row_panel.loc["2024-01-02", "B"] = 2.0
    out = _run("cs_mad_zscore", row_panel)
    assert np.isnan(out.loc["2024-01-02", "A"])
    assert np.isnan(out.loc["2024-01-02", "B"])
