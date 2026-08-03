# -*- coding: utf-8 -*-
"""Fastpath coverage for stable neutralize ops: size / industry_size."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from planner.logical_plan import PlanNode


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", inputs=(), attrs={"name": name})


def test_size_and_dual_backends_registered() -> None:
    from cleaned_operators.registry import OperatorRegistry

    for name in ("size_neutralize", "industry_size_neutralize", "group_neutralize"):
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends, (name, backends)
        assert "polars" in backends, (name, backends)
        assert "sql" in backends, (name, backends)


def test_size_neutralize_sql_emitter() -> None:
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(op="size_neutralize", inputs=[_col("close"), _col("mcap")])
    compiled = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert compiled is not None
    q = compiled.query.lower()
    assert "ln(" in q or "log(" in q
    assert "greatest" in q
    assert "covar" in q
    assert "< 3" in compiled.query


def test_industry_size_neutralize_sql_emitter() -> None:
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="industry_size_neutralize",
        inputs=[_col("close"), _col("industry"), _col("mcap")],
    )
    compiled = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert compiled is not None
    assert "PARTITION BY x.ts, g._v" in compiled.query
    q = compiled.query.lower()
    assert "ln(" in q or "log(" in q
    assert "greatest" in q


def test_size_neutralize_polars_matches_pandas() -> None:
    from cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=8)
    cols = [f"s{i}" for i in range(40)]
    x = pd.DataFrame(rng.normal(size=(8, 40)), index=idx, columns=cols)
    mcap = pd.DataFrame(
        np.exp(rng.normal(22, 1.0, size=(8, 40))), index=idx, columns=cols
    )
    pd_out = OperatorRegistry.get("size_neutralize").calculate(x, mcap)
    pl_out = OperatorRegistry.get("size_neutralize", "polars").calculate(
        __import__("polars").from_pandas(x.reset_index(drop=True)),
        __import__("polars").from_pandas(mcap.reset_index(drop=True)),
    )
    np.testing.assert_allclose(
        pl_out.to_numpy(), pd_out.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-9
    )


def test_industry_size_neutralize_polars_matches_pandas() -> None:
    from cleaned_operators.registry import OperatorRegistry
    import polars as pl

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-02", periods=6)
    cols = [f"s{i}" for i in range(30)]
    x = pd.DataFrame(rng.normal(size=(6, 30)), index=idx, columns=cols)
    ind = pd.DataFrame(
        np.tile(rng.integers(0, 5, size=30), (6, 1)).astype(float),
        index=idx,
        columns=cols,
    )
    mcap = pd.DataFrame(
        np.exp(rng.normal(22, 1.0, size=(6, 30))), index=idx, columns=cols
    )
    pd_out = OperatorRegistry.get("industry_size_neutralize").calculate(x, ind, mcap)
    pl_out = OperatorRegistry.get("industry_size_neutralize", "polars").calculate(
        pl.from_pandas(x.reset_index(drop=True)),
        pl.from_pandas(ind.reset_index(drop=True)),
        pl.from_pandas(mcap.reset_index(drop=True)),
    )
    np.testing.assert_allclose(
        pl_out.to_numpy(), pd_out.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-9
    )


def test_polars_long_emitter_size_and_dual() -> None:
    from backend.polars_expr_emitter import compile_plan_to_polars
    import polars as pl

    base = pl.DataFrame(
        {
            "ts": [1, 1, 1, 2, 2, 2],
            "inst": ["a", "b", "c", "a", "b", "c"],
            "close": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            "industry": [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
            "mcap": [10.0, 100.0, 10.0, 10.0, 100.0, 10.0],
        }
    ).lazy()
    size_plan = PlanNode(op="size_neutralize", inputs=[_col("close"), _col("mcap")])
    dual_plan = PlanNode(
        op="industry_size_neutralize",
        inputs=[_col("close"), _col("industry"), _col("mcap")],
    )
    size_res = compile_plan_to_polars(size_plan, base)
    dual_res = compile_plan_to_polars(dual_plan, base)
    assert size_res is not None
    assert dual_res is not None
    size_df = size_res.frame.collect()
    dual_df = dual_res.frame.collect()
    assert size_df.height == 6
    assert dual_df.height == 6
    assert size_df["_v"].null_count() == 0
