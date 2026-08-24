# -*- coding: utf-8
"""算子 golden 共享 panel fixture。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all


@pytest.fixture(scope="module")
def loaded():
    load_all()
    yield


@pytest.fixture(scope="module")
def golden_panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 5.0],
            "B": [2.0, 2.0, 2.0, np.nan, 10.0],
            "C": [np.nan, 1.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )


def assert_panel_shape_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    assert after.shape == before.shape
    assert list(after.index) == list(before.index)
    assert list(after.columns) == list(before.columns)


def assert_prefix_invariant(
    calc,
    x: pd.DataFrame,
    *,
    atol: float = 1e-9,
) -> None:
    """追加未来样本后，历史各时点结果保持不变（PIT / 因果 golden）。"""
    import pytest

    full = calc(x)
    col = x.columns[0]
    for t in range(len(x)):
        trunc = x.iloc[: t + 1]
        part = calc(trunc)
        got = full.iloc[t][col]
        exp = part.iloc[t][col]
        if pd.isna(got) and pd.isna(exp):
            continue
        assert got == pytest.approx(exp, abs=atol), f"lookahead at t={t}: {got} != {exp}"
