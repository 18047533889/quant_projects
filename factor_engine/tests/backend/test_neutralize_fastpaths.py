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
    for name in ("size_neutralize", "industry_size_neutralize", "group_neutralize"):
        assert "sql" in OperatorRegistry.backends_for(name), (name, "expected sql")


def test_size_neutralize_sql_emitter() -> None:
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(op="size_neutralize", inputs=[_col("close"), _col("mcap")])
    compiled = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert compiled is not None
    q = compiled.query.lower()
    assert "ln(" in q or "log(" in q
    assert "greatest" not in q  # R19-024: no silent clip to log(1)
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
    q = compiled.query.lower()
    assert "ln_dm" in compiled.query or "ln_dm" in q
    assert "greatest" not in q  # R19-024: no silent clip to log(1)
    assert "< 3" in compiled.query


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
    from cleaned_operators._numpy_kernels import industry_size_resid_panel_
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
    size_df = size_res.frame.collect().sort(["ts", "inst"])
    dual_df = dual_res.frame.collect().sort(["ts", "inst"])
    assert size_df.height == 6
    assert dual_df.height == 6
    assert size_df["_v"].null_count() == 0

    # Emitter must match FWL numpy kernel (not old sequential raw-ln residual).
    y = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=float)
    ind = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=float)
    mcap = np.array([[10.0, 100.0, 10.0], [10.0, 100.0, 10.0]], dtype=float)
    expected = industry_size_resid_panel_(y, ind, mcap).reshape(-1)
    got = dual_df["_v"].to_numpy()
    np.testing.assert_allclose(got, expected, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_industry_size_emitter_rejects_nonpositive_cap_like_kernel() -> None:
    """R19-024: non-positive mcap is missing, not silently clipped to log(1)."""
    from backend.polars_expr_emitter import compile_plan_to_polars
    import polars as pl

    base = pl.DataFrame(
        {
            "ts": [1, 1, 1],
            "inst": ["a", "b", "c"],
            "close": [1.0, 2.0, 3.0],
            "industry": [1.0, 1.0, 1.0],
            "mcap": [10.0, 0.0, -5.0],
        }
    ).lazy()
    plan = PlanNode(
        op="industry_size_neutralize",
        inputs=[_col("close"), _col("industry"), _col("mcap")],
    )
    res = compile_plan_to_polars(plan, base)
    assert res is not None
    df = res.frame.collect().sort("inst")
    # With only one legal cap in the industry, FWL residual sample is degenerate
    # or that row alone cannot form a 3-obs OLS; non-positive caps must be null.
    by_inst = {r["inst"]: r["_v"] for r in df.to_dicts()}
    assert by_inst["b"] is None
    assert by_inst["c"] is None


def test_size_neutralize_emitter_rejects_nonpositive_cap_like_kernel() -> None:
    """R19-024: size_neutralize emitter matches numpy legal-cap mask."""
    from backend.polars_expr_emitter import compile_plan_to_polars
    from cleaned_operators._numpy_kernels import size_resid_panel_
    import polars as pl

    base = pl.DataFrame(
        {
            "ts": [1, 1, 1, 1, 1, 1],
            "inst": ["a", "b", "c", "d", "e", "f"],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "mcap": [100.0, 200.0, 300.0, 0.0, float("nan"), 600.0],
        }
    ).lazy()
    plan = PlanNode(op="size_neutralize", inputs=[_col("close"), _col("mcap")])
    res = compile_plan_to_polars(plan, base)
    assert res is not None
    got = res.frame.collect().sort("inst")["_v"].to_numpy()
    expected = size_resid_panel_(
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        np.array([[100.0, 200.0, 300.0, 0.0, np.nan, 600.0]]),
    ).reshape(-1)
    np.testing.assert_allclose(got, expected, equal_nan=True, rtol=1e-9, atol=1e-9)
    assert np.isnan(got[3]) and np.isnan(got[4])


def test_size_and_industry_sql_duckdb_matches_numpy() -> None:
    """DuckDB SQL emitter vs numpy for size + industry_size neutralize."""
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    rows = [
        (1, "a", 1.0, 1.0, 10.0),
        (1, "b", 2.0, 1.0, 100.0),
        (1, "c", 3.0, 2.0, 10.0),
        (1, "d", 4.0, 2.0, 0.0),  # illegal cap
        (2, "a", 2.0, 1.0, 10.0),
        (2, "b", 4.0, 1.0, 100.0),
        (2, "c", 6.0, 2.0, 10.0),
        (2, "d", 8.0, 2.0, 50.0),
    ]
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE d AS SELECT * FROM (VALUES "
        + ", ".join(f"({t}, '{i}', {c}, {g}, {m})" for t, i, c, g, m in rows)
        + ") AS t(t, i, close, industry, mcap)"
    )
    size_plan = PlanNode(op="size_neutralize", inputs=[_col("close"), _col("mcap")])
    dual_plan = PlanNode(
        op="industry_size_neutralize",
        inputs=[_col("close"), _col("industry"), _col("mcap")],
    )
    size_sql = compile_plan_to_sql(
        size_plan, dataset="d", time_column="t", instrument_column="i"
    )
    dual_sql = compile_plan_to_sql(
        dual_plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert size_sql is not None and dual_sql is not None
    size_q = size_sql.query.replace("{{d}}", "d")
    dual_q = dual_sql.query.replace("{{d}}", "d")
    size_df = con.execute(size_q).df().sort_values(["ts", "inst"])
    dual_df = con.execute(dual_q).df().sort_values(["ts", "inst"])
    val_col = "value" if "value" in size_df.columns else "_v"

    from cleaned_operators._numpy_kernels import (
        industry_size_resid_panel_,
        size_resid_panel_,
    )

    y = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    ind = np.array([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    mcap = np.array([[10.0, 100.0, 10.0, 0.0], [10.0, 100.0, 10.0, 50.0]])
    np.testing.assert_allclose(
        size_df[val_col].to_numpy(),
        size_resid_panel_(y, mcap).reshape(-1),
        equal_nan=True,
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dual_df[val_col].to_numpy(),
        industry_size_resid_panel_(y, ind, mcap).reshape(-1),
        equal_nan=True,
        rtol=1e-9,
        atol=1e-9,
    )