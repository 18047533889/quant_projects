import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.polars_expr_emitter import _group_normalize_on
from factor_engine.backend.sql_pushdown.emitter import (
    SqlDialect,
    _group_minmax_expr,
)
from factor_engine.cleaned_operators.common.group import GroupNormalize
from factor_engine.cleaned_operators.common.group_polars import GroupNormalizePolars


EXTREME = [-1.0e308, 0.0, 1.0e308]
SUBNORMAL = [0.0, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2.0]


def _pandas_result(values):
    columns = [f"v{i}" for i in range(len(values))]
    frame = pd.DataFrame([values], columns=columns)
    group = pd.DataFrame([["g"] * len(values)], columns=columns)
    return GroupNormalize()._calculate_series(frame, group).iloc[0].to_numpy()


def _polars_eager_result(values):
    frame = pl.DataFrame({f"v{i}": [value] for i, value in enumerate(values)})
    group = pl.DataFrame({f"v{i}": ["g"] for i in range(len(values))})
    return np.asarray(
        GroupNormalizePolars()._calculate_series(frame, group).row(0), dtype=float
    )


def _polars_expression_result(values):
    frame = pl.DataFrame({
        "ts": [1] * len(values),
        "inst": list(range(len(values))),
        "grp": ["g"] * len(values),
        "_v": values,
    })
    return np.asarray(
        frame.with_columns(
            _group_normalize_on("_v", ("ts", "grp")).alias("out")
        )["out"].to_list(),
        dtype=float,
    )


def _duckdb_result(values):
    import duckdb

    expr = _group_minmax_expr(
        value_col="v",
        partition="PARTITION BY ts, grp",
        dialect=SqlDialect.DUCKDB,
    )
    rows = ", ".join(
        f"(1, {index}, 'g', CAST(? AS DOUBLE))" for index in range(len(values))
    )
    sql = f"SELECT inst, {expr} AS out FROM (VALUES {rows}) t(ts, inst, grp, v) ORDER BY inst"
    with duckdb.connect() as connection:
        return np.asarray(
            [row[1] for row in connection.execute(sql, list(values)).fetchall()],
            dtype=float,
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (EXTREME, [0.0, 0.5, 1.0]),
        (SUBNORMAL, [0.0, 0.5, 1.0]),
        ([7.0], [0.5]),
        ([7.0, 7.0, 7.0], [0.5, 0.5, 0.5]),
    ],
    ids=["opposite-sign-extreme", "subnormal", "singleton", "constant"],
)
@pytest.mark.parametrize(
    "implementation",
    [_pandas_result, _polars_eager_result, _polars_expression_result, _duckdb_result],
    ids=["pandas", "polars-eager", "polars-expression", "duckdb-sql"],
)
def test_group_normalize_numeric_boundaries(implementation, values, expected):
    np.testing.assert_allclose(implementation(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_group_normalize_missing_group_global_keeps_minmax_semantics(backend):
    columns = ["a", "b", "c"]
    if backend == "pandas":
        frame = pd.DataFrame([EXTREME], columns=columns)
        missing = pd.DataFrame([[np.nan] * 3], columns=columns)
        actual = GroupNormalize()._calculate_series(
            frame, missing, fallback_policy="global"
        ).iloc[0].to_numpy()
    else:
        frame = pl.DataFrame({column: [value] for column, value in zip(columns, EXTREME)})
        missing = pl.DataFrame({column: [None] for column in columns})
        actual = np.asarray(
            GroupNormalizePolars()._calculate_series(
                frame, missing, fallback_policy="global"
            ).row(0),
            dtype=float,
        )
    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0], rtol=0, atol=0)
