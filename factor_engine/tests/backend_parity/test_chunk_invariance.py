# -*- coding: utf-8
"""Chunk / partition 扫描不变性：全量 vs 分块 overlap 须 parity。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _panel(*, n_days: int = 20, overlap: int = 5, window: int = 3):
    dates = pd.date_range("2024-01-02", periods=n_days, freq="D")
    insts = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    rng = np.random.default_rng(7)
    close = pd.Series(10.0 + rng.normal(0, 0.2, len(idx)).cumsum(), index=idx)
    open_ = close - 0.1
    full = InMemorySeriesSource(data={"close": close, "open": open_})
    chunk_start = dates[max(0, n_days - overlap - (window - 1))]
    eval_start = dates[n_days - overlap]
    part_idx = idx[idx.get_level_values(0) >= chunk_start]
    eval_idx = idx[(idx.get_level_values(0) >= eval_start)]
    part = InMemorySeriesSource(
        data={
            "close": close.loc[part_idx],
            "open": open_.loc[part_idx],
        }
    )
    return full, part, eval_idx


def _run(source, expr, backend: str) -> pd.Series:
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


@pytest.mark.parametrize(
    "name,expr_builder",
    [
        ("ts_mean", lambda: F("ts_mean")(col("close"), 3)),
        ("ts_delay", lambda: F("ts_delay")(col("close"), 2)),
        ("ts_delta", lambda: F("ts_delta")(col("close"), 1)),
    ],
)
def test_chunk_overlap_invariance_polars_long(_loaded, name, expr_builder):
    full_src, part_src, eval_idx = _panel()
    expr = expr_builder()
    full_out = _run(full_src, expr, "polars_long").loc[eval_idx]
    part_out = _run(part_src, expr, "polars_long").loc[eval_idx]
    pd.testing.assert_series_equal(
        full_out.astype(float),
        part_out.astype(float),
        check_names=False,
        rtol=1e-5,
        atol=1e-5,
    )
