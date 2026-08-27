# -*- coding: utf-8
"""Wave1-E: SQL→Arrow→polars engine-to-engine transfer end-to-end fixture.

Covers mission item 4 (engine-to-engine transfer) with a NON-VACUOUS parity
assertion: compile a real SQL pushdown plan, execute it in an in-memory DuckDB,
pull the Arrow result through ``polars.from_arrow`` (the exact boundary the
production executor uses — ``sql_pushdown.executor`` Arrow→polars), and assert
the transferred values match the pandas reference exactly.

Also pins the Wave1-E capability governance:
* ``_resolve_capability_path`` is the honest runtime probe — a canonical whose
  SQL whitelist entry has no real emitter/execution path is NOT SQL-capable.
* ``ts_ewm_corr``/``ts_ewm_cov`` remain polars-only (experimental pandas-EWM
  delegates, R20-P0-EWM-PAIRWISE) — they must never be claimed SQL-capable.

Does not touch the Wave1-H-owned EWMA/Wilder emitter block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

import duckdb

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
from factor_engine.backend.sql_pushdown.sql_registry import _resolve_capability_path
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.logical_plan import PlanNode

ensure_cleaned_loaded()

DATES = [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(8)]
_INST = ["A", "B"]


def _build_table(vals):
    rows = []
    half = len(vals) // 2
    for inst, offset in zip(_INST, (0, half)):
        for day, v in enumerate(vals[offset : offset + half]):
            rows.append({"ts": DATES[day].date(), "inst": inst, "close": float(v)})
    return pd.DataFrame(rows)


def _pandas_reference(op, vals, pkw):
    half = len(vals) // 2
    panel = pd.DataFrame({"A": vals[:half], "B": vals[half:]}, index=DATES[:half])
    inst = OperatorRegistry.get(op, "pandas_numpy")
    # ts_std keeps a hardcoded min_periods=1 in its kernel; only forward the
    # declared params (the kernel rejects undeclared min_periods, R5-06).
    call_kw = {k: v for k, v in pkw.items() if k != "min_periods"}
    return inst.calculate(panel, **call_kw).stack().sort_index()


# ts_std is whitelisted in SQL_IMPLEMENTED_CANONICALS and has a real emitter
# branch; this exercises the honest transfer path.  Its pandas reference
# contract is (x, window) only — min_periods is not a declared parameter, so we
# seed both sides with the same window and rely on the kernel's min_periods=1.
TRANSFER_CASE = ("ts_std", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                 dict(window=3))


def _run_sql_to_polars(op, vals, pkw):
    """compile -> duckdb exec -> Arrow -> polars collect (the executor boundary)."""
    import polars as pl

    plan = PlanNode(op=op, inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs=pkw)
    sql = compile_plan_to_sql(
        plan, dataset="test_panel", time_column="ts", instrument_column="inst"
    )
    assert sql is not None and sql.query.strip(), f"{op} compiled empty SQL"
    query = sql.query.replace("{{test_panel}}", "test_panel")

    con = duckdb.connect(":memory:")
    try:
        con.register("test_panel", _build_table(vals))
        arrow_raw = con.execute(query).arrow()
    finally:
        con.close()
    # modern duckdb returns a pyarrow RecordBatchReader
    arrow = arrow_raw.read_all() if hasattr(arrow_raw, "read_all") else arrow_raw
    assert arrow.num_rows > 0, "transfer produced zero rows"
    lf = pl.from_arrow(arrow)  # Arrow -> polars boundary
    assert "value" in lf.columns, f"missing value column: {lf.columns}"
    df = lf.select(["ts", "inst", "value"]).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index(["ts", "inst"])["value"].astype(float).sort_index()


def test_sql_to_polars_transfer_values_match_pandas_reference():
    op, vals, pkw = TRANSFER_CASE
    sql_series = _run_sql_to_polars(op, vals, pkw)
    pd_series = _pandas_reference(op, vals, pkw)
    pd_series.index = pd_series.index.set_names(["ts", "inst"])
    aligned = pd_series.reindex(sql_series.index)

    assert len(sql_series) == len(aligned) == 16, (len(sql_series), len(aligned))
    np.testing.assert_allclose(
        aligned.to_numpy(), sql_series.to_numpy(),
        rtol=1e-5, atol=1e-6, equal_nan=True,
        err_msg="engine-to-engine SQL->polars transfer != pandas reference",
    )


def test_resolve_capability_path_is_non_vacuous():
    op, vals, pkw = TRANSFER_CASE
    plan = PlanNode(op=op, inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs=pkw)
    assert _resolve_capability_path(plan, dialect="duckdb") is True
    # A canonical with no real emitter/execution path must probe False — the
    # whitelist alone must never imply capability (non-vacuous governance).
    plan2 = PlanNode(op="cs_rank_normalize", inputs=[PlanNode(op="column", attrs={"name": "close"})], attrs={})
    assert _resolve_capability_path(plan2, dialect="duckdb") is False


def test_ts_ewm_pairwise_stays_polars_only_and_never_sql():
    # R20-P0-EWM-PAIRWISE experimental pandas-EWM delegates: polars backend only,
    # and — absent a real emitter branch — never SQL-capable.
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    for canon in ("ts_ewm_corr", "ts_ewm_cov"):
        backends = set(OperatorRegistry.backends_for(canon))
        assert "pandas_numpy" not in backends, canon
        assert "polars" in backends, canon
        assert canon not in SQL_IMPLEMENTED_CANONICALS, (
            f"{canon} is not a real emitter branch; must not be SQL-implemented"
        )
