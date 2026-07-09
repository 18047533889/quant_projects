# -*- coding: utf-8
"""Polars batch mirror 与 row 统计算子门禁。"""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

pytest.importorskip("polars")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


@pytest.mark.parametrize(
    "canonical",
    [
        "nan_to_num",
        "protected_log",
        "rank_transform",
        "product",
        "quarter",
        "ttm",
        "yoy",
        "price_spread_deviation",
        "ts_max_buildup",
        "row_kurt",
        "row_skew",
        "row_prod",
    ],
)
def test_batch_mirror_ops_have_polars(canonical: str):
    assert "polars" in OperatorRegistry.backends_for(canonical), canonical


def test_pandas_only_count_below_ceiling():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    pandas_only = [
        c for c in canon if "polars" not in OperatorRegistry.backends_for(c) and "pandas_numpy" in OperatorRegistry.backends_for(c)
    ]
    assert len(pandas_only) <= 45, f"仍有过量 pandas-only: {len(pandas_only)}"
