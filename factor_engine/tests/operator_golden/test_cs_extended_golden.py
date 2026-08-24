# -*- coding: utf-8
"""审查清单截面算子 golden：c_* / cs_* / winsorize / neutralize。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged


def _run(canon: str, panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None
    out = op.calculate(panel.copy(), *args, **kwargs)
    assert_panel_shape_unchanged(panel, out)
    return out


def test_c_std_broadcasts_row_std(golden_panel, loaded):
    out = _run("c_std", golden_panel)
    row_std = golden_panel.std(axis=1, skipna=True, ddof=1)
    for col in golden_panel.columns:
        pd.testing.assert_series_equal(out[col], row_std, check_names=False)


def test_c_sum_all_nan_row_is_null(loaded):
    panel = pd.DataFrame({"A": [np.nan, 1.0], "B": [np.nan, 2.0]})
    out = _run("c_sum", panel)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(3.0)


def test_cs_quantile_matches_c_percentile(golden_panel, loaded):
    a = _run("c_percentile", golden_panel, 0.5)
    b = _run("cs_quantile", golden_panel, 0.5)
    pd.testing.assert_frame_equal(a, b)


def test_cs_pct_rank_differs_from_rank(golden_panel, loaded):
    rank = _run("rank", golden_panel)
    pct = _run("cs_pct_rank", golden_panel)
    assert not np.allclose(
        rank.to_numpy(dtype=float),
        pct.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_rank_pct_is_pandas_pct_not_cs_rank_01(golden_panel, loaded):
    rank01 = _run("rank", golden_panel)
    pct = _run("rank_pct", golden_panel)
    assert not np.allclose(
        rank01.loc["2024-01-01"].to_numpy(),
        pct.loc["2024-01-01"].to_numpy(),
        equal_nan=True,
    )


def test_winsorize_clips_cross_section(loaded):
    panel = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 1000.0]})
    out = _run("winsorize", panel, lower=0.25, upper=0.75)
    assert out.iloc[1, 2] < 1000.0
    assert out.iloc[1, 0] >= 1.0


def test_neutralize_cs_demean_alias_semantics(loaded):
    panel = pd.DataFrame({"A": [1.0, 4.0], "B": [3.0, 6.0]})
    demean = _run("cs_demean", panel)
    neut = _run("neutralize", panel)
    pd.testing.assert_frame_equal(demean, neut)


def test_protected_div_zero_denominator(loaded):
    num = pd.DataFrame({"A": [1.0, 2.0]})
    den = pd.DataFrame({"A": [0.0, 2.0]})
    op = OperatorRegistry.get("protected_div")
    out = op.calculate(num, den)
    assert_panel_shape_unchanged(num, out)
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[1, 0] == pytest.approx(1.0)
