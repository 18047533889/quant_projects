# -*- coding: utf-8
"""代数性质与排序不变性测试（§十九）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.long_alignment import AlignmentError, assert_unique_keys
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


@pytest.fixture(scope="module")
def panel_source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    y = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    return InMemorySeriesSource(data={"x": x, "y": y})


def _run(source, expr, backend: str = "polars_long"):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def test_add_identity(_loaded, panel_source):
    x = _run(panel_source, col("x"))
    out = _run(panel_source, F("add")(col("x"), 0.0))
    pd.testing.assert_series_equal(x, out, check_names=False, rtol=1e-9, atol=1e-9)


def test_multiply_identity(_loaded, panel_source):
    x = _run(panel_source, col("x"))
    out = _run(panel_source, F("multiply")(col("x"), 1.0))
    pd.testing.assert_series_equal(x, out, check_names=False, rtol=1e-9, atol=1e-9)


def test_subtract_self_is_zero(_loaded, panel_source):
    out = _run(panel_source, F("subtract")(col("x"), col("x")))
    finite = out.dropna()
    assert (finite == 0.0).all()


def test_minimum_self(_loaded, panel_source):
    x = _run(panel_source, col("x"))
    out = _run(panel_source, F("minimum")(col("x"), col("x")))
    pd.testing.assert_series_equal(x, out, check_names=False, rtol=1e-9, atol=1e-9)


def test_maximum_self(_loaded, panel_source):
    x = _run(panel_source, col("x"))
    out = _run(panel_source, F("maximum")(col("x"), col("x")))
    pd.testing.assert_series_equal(x, out, check_names=False, rtol=1e-9, atol=1e-9)


def test_abs_non_negative(_loaded, panel_source):
    out = _run(panel_source, F("abs")(col("x")))
    finite = out.dropna()
    assert (finite >= 0).all()


def test_ts_delta_equals_x_minus_delay(_loaded, panel_source):
    n = 2
    delta = _run(panel_source, F("ts_delta")(col("x"), n))
    delay = _run(panel_source, F("ts_delay")(col("x"), n))
    x = _run(panel_source, col("x"))
    recon = x - delay
    pd.testing.assert_series_equal(delta, recon, check_names=False, rtol=1e-9, atol=1e-9)


def test_rank_affine_invariant_positive_scale(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
        ],
        names=["timestamp", "instrument"],
    )
    x = pd.Series([1.0, 2.0, 3.0], index=idx)
    src = InMemorySeriesSource(data={"x": x})
    r1 = _run(src, F("rank_pct")(col("x")))
    r2 = _run(src, F("rank_pct")(F("add")(F("multiply")(col("x"), 2.0), 5.0)))
    pd.testing.assert_series_equal(r1, r2, check_names=False, rtol=1e-9, atol=1e-9)


def test_output_stable_under_shuffled_input(_loaded, panel_source):
    x = panel_source.data["x"]
    shuffled = x.sample(frac=1.0, random_state=7)
    src_shuf = InMemorySeriesSource(data={"x": shuffled})
    a = _run(panel_source, F("ts_mean")(col("x"), 2))
    b = _run(src_shuf, F("ts_mean")(col("x"), 2))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9, atol=1e-9)


def test_duplicate_ts_inst_raises(_loaded):
    import polars as pl

    lf = pl.LazyFrame(
        {
            "ts": [1, 1, 2],
            "inst": ["A", "A", "B"],
            "_v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(AlignmentError):
        assert_unique_keys(lf, context="test")
