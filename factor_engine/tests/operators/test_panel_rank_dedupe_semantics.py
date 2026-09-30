"""The legacy panel_rank name must retain pandas percentile-rank semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _rank_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [7.0, np.nan, np.nan, np.nan],  # singleton -> 1.0
            [1.0, 1.0, 3.0, np.nan],        # tie average rank, NaN excluded
            [2.0, 2.0, 2.0, 2.0],           # all tied -> 0.625
            [np.nan, np.nan, np.nan, np.nan],
        ],
        index=["singleton", "ties_and_nan", "all_tied", "all_nan"],
        columns=["A", "B", "C", "D"],
    )


def test_panel_rank_dedupe_preserves_percentile_semantics_through_load_all():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.common.group import PanelRank
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    panel = _rank_panel()
    expected = panel.rank(pct=True, axis=1)

    # The registered class's pre-dedupe implementation is the semantic oracle.
    before_load = PanelRank().calculate(panel)
    pd.testing.assert_frame_equal(before_load, expected)
    assert before_load.loc["singleton", "A"] == 1.0
    assert before_load.loc["ties_and_nan", ["A", "B"]].tolist() == [0.5, 0.5]
    assert before_load.loc["all_tied"].tolist() == [0.625] * 4
    assert before_load.loc["all_nan"].isna().all()

    load_all()
    assert OperatorRegistry.resolve_canonical("panel_rank") == "cs_pct_rank"
    pandas_op = OperatorRegistry.get("panel_rank", "pandas_numpy", mode="any")
    assert pandas_op is not None
    pd.testing.assert_frame_equal(pandas_op.calculate(panel), expected)

    pl = pytest.importorskip("polars")
    polars_input = pl.DataFrame({column: panel[column].tolist() for column in panel.columns})
    polars_op = OperatorRegistry.get("panel_rank", "polars", mode="any")
    assert polars_op is not None
    np.testing.assert_allclose(
        polars_op.calculate(polars_input).to_numpy(),
        expected.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_panel_rank_sql_alias_uses_percentile_rank_semantics():
    duckdb = pytest.importorskip("duckdb")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql

    load_all()
    panel = _rank_panel()
    expected = panel.rank(pct=True, axis=1)
    source = pd.DataFrame(
        [
            {"ts": str(index), "inst": column, "close": panel.loc[index, column]}
            for index in panel.index
            for column in panel.columns
        ]
    )
    plan = PlanNode(
        op="panel_rank",
        inputs=(PlanNode(op="column", attrs={"name": "close"}),),
    )
    compiled = compile_plan_to_sql(
        plan, table="panel_rank_input", time_column="ts", instrument_column="inst",
    )
