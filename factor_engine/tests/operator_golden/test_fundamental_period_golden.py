# -*- coding: utf-8
"""基本面 period-aware 算子 golden。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.fundamental.period_helpers import (
    compute_quarter,
    compute_yoy,
    quarter_from_cumulative,
    ttm_from_cumulative,
    ttm_from_quarterly,
    yoy_by_period,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged


def _fiscal_panel():
    """4 季累计值 + fiscal_quarter 1-4。"""
    idx = pd.date_range("2024-03-31", periods=4, freq="QE")
    cumulative = pd.DataFrame({"A": [100.0, 220.0, 350.0, 500.0]}, index=idx)
    fiscal_q = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    return cumulative, fiscal_q


def test_quarter_from_cumulative_matches_compute_quarter(loaded):
    cum, fq = _fiscal_panel()
    a = quarter_from_cumulative(cum, fq)
    b = compute_quarter(cum, fq)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[cum.index[0], "A"] == 100.0
    assert a.loc[cum.index[1], "A"] == 120.0


def test_ttm_from_cumulative_vs_quarterly_path(loaded):
    cum, fq = _fiscal_panel()
    quarterly = quarter_from_cumulative(cum, fq)
    direct = ttm_from_cumulative(cum, fq)
    via_q = ttm_from_quarterly(quarterly, fq)
    assert_panel_shape_unchanged(cum, direct)
    pd.testing.assert_frame_equal(direct, via_q)
    assert direct.loc[cum.index[3], "A"] == pytest.approx(500.0)


def test_yoy_by_period_matches_compute_yoy(loaded):
    cum, fq = _fiscal_panel()
    quarterly = quarter_from_cumulative(cum, fq)
    pd.testing.assert_frame_equal(
        yoy_by_period(quarterly, fq),
        compute_yoy(quarterly, fq),
    )


def test_ttm_from_cumulative_differs_from_naive_ttm_on_cumulative(loaded):
    """累计值直接 ttm 与 ttm_from_cumulative 语义不同（后者先 quarterify）。"""
    from factor_engine.cleaned_operators.fundamental.period_helpers import compute_ttm

    cum, fq = _fiscal_panel()
    naive = compute_ttm(cum, fq)
    proper = ttm_from_cumulative(cum, fq)
    assert not np.allclose(
        naive.to_numpy(dtype=float),
        proper.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_period_ops_registered(loaded):
    for canon in (
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
    ):
        assert OperatorRegistry.get(canon) is not None, canon
