# -*- coding: utf-8
"""Production core 时序 / 截面算子 PIT golden：前缀不变性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_prefix_invariant


def _run(canon: str, panel: pd.DataFrame, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None
    return op.calculate(panel.copy(), **kwargs)


@pytest.mark.parametrize(
    "canon,kwargs",
    [
        ("ts_mean", {"d": 2}),
        ("ts_std", {"d": 2}),
        ("ts_sum", {"d": 2}),
        ("ts_ema", {"d": 2}),
        ("ts_delay", {"d": 1}),
        ("ts_delta", {"d": 1}),
        ("ts_pct", {"d": 1}),
        ("rank", {}),
        ("zscore", {}),
        ("cs_demean", {}),
        ("ffill", {}),
    ],
)
def test_production_core_prefix_invariant(loaded, canon: str, kwargs: dict):
    panel = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=6),
    )
    assert_prefix_invariant(lambda df: _run(canon, df, **kwargs), panel)


def test_ts_rank_prefix_invariant(loaded):
    panel = pd.DataFrame(
        {"A": [3.0, 1.0, 2.0, 4.0, 5.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    assert_prefix_invariant(lambda df: _run("ts_rank", df, d=2), panel)
