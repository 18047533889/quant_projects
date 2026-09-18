import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.polars_expr_emitter import _normalize_on
from factor_engine.backend.sql_pushdown.emitter import (
    SqlDialect,
    _normalize_window_expr,
)
from factor_engine.cleaned_operators.common.elementwise import Normalize
from factor_engine.cleaned_operators.common.polars_daily_native import NormalizeNative


EXTREME = [-1.0e308, 0.0, 1.0e308]
EXPECTED = [0.0, 0.5, 1.0]
SUBNORMAL = [0.0, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2.0]


def test_pandas_normalize_opposite_sign_extrema_without_overflow():
    frame = pd.DataFrame([EXTREME], columns=["a", "b", "c"])
    actual = Normalize()._calculate_series(frame)
    np.testing.assert_allclose(actual.iloc[0].to_numpy(), EXPECTED, rtol=0, atol=0)


def test_pandas_normalize_keeps_subnormal_formula():
    frame = pd.DataFrame([SUBNORMAL], columns=["a", "b", "c"])
    actual = Normalize()._calculate_series(frame)
    np.testing.assert_allclose(actual.iloc[0].to_numpy(), EXPECTED, rtol=0, atol=0)


def test_polars_native_normalize_opposite_sign_extrema_without_overflow():
    frame = pl.DataFrame({"a": [EXTREME[0]], "b": [0.0], "c": [EXTREME[2]]})
    actual = NormalizeNative()._calculate_series(frame)
    assert actual.row(0) == pytest.approx(EXPECTED, rel=0, abs=0)


def test_polars_native_normalize_keeps_subnormal_formula():
    frame = pl.DataFrame({"a": [SUBNORMAL[0]], "b": [SUBNORMAL[1]], "c": [SUBNORMAL[2]]})
    actual = NormalizeNative()._calculate_series(frame)
    assert actual.row(0) == pytest.approx(EXPECTED, rel=0, abs=0)


def test_polars_emitter_normalize_opposite_sign_extrema_without_overflow():
    frame = pl.DataFrame({
        "ts": [1, 1, 1],
        "inst": ["a", "b", "c"],
        "_v": EXTREME,
    })
    actual = frame.with_columns(_normalize_on("_v").alias("out"))["out"].to_list()
    assert actual == pytest.approx(EXPECTED, rel=0, abs=0)


def test_polars_emitter_normalize_keeps_subnormal_formula():
    frame = pl.DataFrame({
        "ts": [1, 1, 1],
        "inst": ["a", "b", "c"],
        "_v": SUBNORMAL,
    })
    actual = frame.with_columns(_normalize_on("_v").alias("out"))["out"].to_list()
    assert actual == pytest.approx(EXPECTED, rel=0, abs=0)


def test_duckdb_sql_normalize_opposite_sign_extrema_without_overflow():
    import duckdb

    expr = _normalize_window_expr(
        value_col="v", partition="PARTITION BY ts", dialect=SqlDialect.DUCKDB
    )
    sql = (
        f"SELECT inst, {expr} AS out FROM (VALUES "
        "(1, 'a', -1e308), (1, 'b', 0.0), (1, 'c', 1e308)) t(ts, inst, v) "
        "ORDER BY inst"
    )
    with duckdb.connect() as connection:
        actual = [row[1] for row in connection.execute(sql).fetchall()]
    assert actual == pytest.approx(EXPECTED, rel=0, abs=0)


def test_duckdb_sql_normalize_keeps_subnormal_formula():
    import duckdb

    expr = _normalize_window_expr(
        value_col="v", partition="PARTITION BY ts", dialect=SqlDialect.DUCKDB
    )
    sql = (
        f"SELECT inst, {expr} AS out FROM (VALUES "
        "(1, 'a', CAST(? AS DOUBLE)), (1, 'b', CAST(? AS DOUBLE)), "
        "(1, 'c', CAST(? AS DOUBLE))) t(ts, inst, v) ORDER BY inst"
    )
    with duckdb.connect() as connection:
        actual = [
            row[1] for row in connection.execute(sql, SUBNORMAL).fetchall()
        ]
    assert actual == pytest.approx(EXPECTED, rel=0, abs=0)


def test_normalize_nan_inf_and_finite_singleton_are_all_missing():
    values = [np.nan, np.inf, 42.0]

    pandas_actual = Normalize()._calculate_series(
        pd.DataFrame([values], columns=["a", "b", "c"])
    )
    assert pandas_actual.iloc[0].isna().all()

    native_actual = NormalizeNative()._calculate_series(
        pl.DataFrame({"a": [values[0]], "b": [values[1]], "c": [values[2]]})
    )
    assert native_actual.row(0) == (None, None, None)

    emitter_frame = pl.DataFrame({
        "ts": [1, 1, 1],
        "inst": ["a", "b", "c"],
        "_v": values,
    })
    emitter_actual = emitter_frame.with_columns(
        _normalize_on("_v").alias("out")
    )["out"].to_list()
    assert emitter_actual == [None, None, None]

    import duckdb

    safe = "CASE WHEN v IS NOT NULL AND NOT isnan(v) AND NOT isinf(v) THEN v END"
    expr = _normalize_window_expr(
        value_col=safe, partition="PARTITION BY ts", dialect=SqlDialect.DUCKDB
    )
    sql = (
        f"SELECT inst, {expr} AS out FROM (VALUES "
        "(1, 'a', CAST(? AS DOUBLE)), (1, 'b', CAST(? AS DOUBLE)), "
        "(1, 'c', CAST(? AS DOUBLE))) t(ts, inst, v) ORDER BY inst"
    )
    with duckdb.connect() as connection:
        sql_actual = [
            row[1] for row in connection.execute(sql, values).fetchall()
        ]
    assert sql_actual == [None, None, None]
