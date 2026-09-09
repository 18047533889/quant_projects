# -*- coding: utf-8 -*-
"""Q03: execute emitted DuckDB SQL against independent small-domain oracles.

These are research-kernel execution tests, not production certification.  They
use the real emitter identity/admission checks and never weaken production
policy when compilation is refused.
"""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.sql_pushdown.emitter import (
    SqlDialect,
    assert_emitter_identity_known,
    compile_plan_to_sql,
)
from factor_engine.planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _node(op: str, *inputs: PlanNode, **attrs) -> PlanNode:
    return PlanNode(op=op, inputs=list(inputs), attrs=attrs)


@pytest.fixture(scope="module", autouse=True)
def _registered_sql_surface() -> None:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()
    # This invokes the real identity policy. Empty is acceptable only in this
    # explicitly research-mode test; production would hard-fail on unknown.
    assert isinstance(assert_emitter_identity_known(production=False), str)


@pytest.fixture
def observations():
    frame = pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "i": ["A"] * 5 + ["B"] * 5,
            "x": [1.0, 2.0, None, 4.0, 5.0, -2.0, None, 3.0, 0.0, 4.0],
            "y": [10.0, 20.0, 30.0, None, 50.0, 1.0, 2.0, 3.0, 4.0, None],
        }
    )
    connection = duckdb.connect(":memory:")
    connection.register("observations", frame)
    yield connection, frame
    connection.close()


class _CountingConnection:
    def __init__(self, connection):
        self.connection = connection
        self.execute_count = 0

    def execute(self, query: str):
        self.execute_count += 1
        return self.connection.execute(query)


def _compile(plan: PlanNode):
    compiled = compile_plan_to_sql(
        plan,
        table="observations",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "observations" in compiled.query
    return compiled


def _assert_values(actual: pd.Series, expected: list[float | None]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual.tolist(), expected):
        if want is None:
            assert pd.isna(got)
        else:
            assert math.isclose(float(got), want, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (
            _node("add", _col("x"), _col("y")),
            [11.0, 22.0, None, None, 55.0, -1.0, None, 6.0, 4.0, None],
        ),
        (
            _node("square", _col("x")),
            [1.0, 4.0, None, 16.0, 25.0, 4.0, None, 9.0, 0.0, 16.0],
        ),
        (
            _node("ts_sum", _col("x"), d=3),
            [1.0, 3.0, 3.0, 6.0, 9.0, -2.0, -2.0, 1.0, 3.0, 7.0],
        ),
    ],
)
def test_emitted_sql_executes_once_and_matches_independent_oracle(
    observations, monkeypatch, record_property, plan: PlanNode, expected: list[float | None]
) -> None:
    # Any accidental pandas-operator fallback is a hard test failure.
    from factor_engine.cleaned_operators.base import SeriesOperator

    monkeypatch.setattr(
        SeriesOperator,
        "calculate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pandas calculate fallback used")
        ),
    )
    connection, _ = observations
    counted = _CountingConnection(connection)
    result = counted.execute(_compile(plan).query).fetchdf()
    record_property("actual_sql_execute_count", counted.execute_count)
    record_property("fetchdf_materializations", 1)
    record_property("output_rows", len(result))
    record_property("output_frame_deep_bytes", int(result.memory_usage(deep=True).sum()))
    record_property("transfer_bytes", "NOT_INSTRUMENTED")
    assert counted.execute_count == 1
    assert result[["ts", "inst"]].values.tolist() == [
        [t, i] for t, i in zip([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], ["A", "B"] * 5)
    ]
    # SQL orders by time then instrument; reorder the hand-written oracle from
    # instrument-major fixture order accordingly.
    oracle = [expected[j] for j in [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]
    _assert_values(result["value"], oracle)


def test_shared_window_subexpression_matches_independent_execution(
    observations, monkeypatch, record_property
) -> None:
    from factor_engine.cleaned_operators.base import SeriesOperator

    monkeypatch.setattr(
        SeriesOperator,
        "calculate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pandas calculate fallback used")
        ),
    )
    connection, _ = observations
    rolling = _node("ts_sum", _col("x"), d=3)
    shared = _node("add", rolling, rolling)

    shared_compiled = _compile(shared)
    independent_compiled = _compile(_node("ts_sum", _col("x"), d=3))
    shared_conn = _CountingConnection(connection)
    independent_conn = _CountingConnection(connection)
    shared_result = shared_conn.execute(shared_compiled.query).fetchdf()
    independent_result = independent_conn.execute(independent_compiled.query).fetchdf()
    record_property("actual_sql_execute_count", shared_conn.execute_count + independent_conn.execute_count)
    record_property("fetchdf_materializations", 2)
    record_property("transfer_bytes", "NOT_INSTRUMENTED")

    assert shared_conn.execute_count == 1
    assert independent_conn.execute_count == 1
    assert " s1 AS " in shared_compiled.query or "s1 AS(" in shared_compiled.query.replace(" ", "")
    np.testing.assert_allclose(
        shared_result["value"].to_numpy(dtype=float),
        2.0 * independent_result["value"].to_numpy(dtype=float),
        equal_nan=True,
    )
