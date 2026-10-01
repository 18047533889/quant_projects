"""Standalone stable native event-sum kernel tests (no registry/source fixtures)."""
import math
import numpy as np
import polars as pl
import pytest
from factor_engine.backend.event_decay_expr import event_decay_sum_expr


@pytest.mark.parametrize("half_life", [1.0, 20.0, 1e10, 1e20])
@pytest.mark.parametrize("grouped", [False, True])
def test_stable_native_event_sum_long_history(half_life, grouped, monkeypatch):
    raw = [None if i % 73 == 0 else (i % 11 - 5) / 8.0 for i in range(4096)]
    weight = 0.5 ** (1.0 / half_life)
    expected, acc = [], 0.0
    for value in raw:
        acc = weight * acc + (0.0 if value is None else value)
        expected.append(acc)
    frame = pl.DataFrame({"x": raw, "g": ["A"] * len(raw)})
    def forbidden(*args, **kwargs):
        raise AssertionError("native core must not convert or call Python UDF")
    monkeypatch.setattr(pl.DataFrame, "to_numpy", forbidden)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", forbidden)
    monkeypatch.setattr(pl.Series, "to_numpy", forbidden)
    monkeypatch.setattr(pl.Expr, "map_elements", forbidden)
    monkeypatch.setattr(pl.Expr, "map_batches", forbidden)
    step = pl.int_range(pl.len()) + 1
    contribution = pl.col("x").fill_null(0.0)
    expr = event_decay_sum_expr(contribution, step, half_life=half_life, groups="g" if grouped else None)
    actual = frame.select(expr.alias("out"))["out"].to_list()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert all(math.isfinite(value) for value in actual)
    prefix = frame.head(211).select(expr.alias("out"))["out"].to_list()
    np.testing.assert_allclose(prefix, expected[:211], rtol=1e-10, atol=1e-10)
