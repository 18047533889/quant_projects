"""Independent finalized-registry tests for relation distribution repairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


DIST_NAMES = (
    "relation_distribution_skew",
    "relation_distribution_pearson_kurtosis",
    "relation_distribution_excess_kurtosis",
)


def _panel(values, *, columns=("A",)):
    return pd.DataFrame(values, index=pd.date_range("2024-01-02", periods=len(values)), columns=list(columns))


def _rank_panels():
    return tuple(_panel([[float(rank)], [float(rank)]]) for rank in range(1, 6))


def _entity_panels():
    cur = (_panel([[0.4], [0.5]]), _panel([[0.3], [0.2]]))
    cur_ids = (_panel([["A"], ["A"]]), _panel([["B"], ["B"]]))
    prev = (_panel([[0.25], [0.3]]), _panel([[0.5], [0.4]]))
    prev_ids = (_panel([["B"], ["B"]]), _panel([["A"], ["A"]]))
    return cur, cur_ids, prev, prev_ids


def test_finalized_owner_contracts_are_explicit_and_dynamic():
    load_all()
    for name in DIST_NAMES:
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert type(op).__module__.endswith("relation.distribution")
        assert tuple(op.metadata.panel_params) == ("relations",)
        assert "variadic" in op.metadata.tags
    mobility = OperatorRegistry.get("relation_rank_entity_mobility", "pandas_numpy")
    assert type(mobility).__module__.endswith("relation.distribution")
    assert "dynamic_inputs" in mobility.metadata.tags
    assert tuple(mobility.metadata.panel_params) == tuple(mobility.metadata.param_names)
    for name in ("group_quantile_spread", "group_tail_ratio"):
        metadata = OperatorRegistry.get(name, "pandas_numpy").metadata
        assert tuple(metadata.panel_params) == ("x", "group")
        assert tuple(metadata.scalar_params) == ("q_low", "q_high")


def test_distribution_positional_keyword_prefix_and_oracle():
    load_all()
    ranks = _rank_panels()
    expected = {
        "relation_distribution_skew": 0.0,
        "relation_distribution_pearson_kurtosis": 1.7,
        "relation_distribution_excess_kurtosis": -1.3,
    }
    for name in DIST_NAMES:
        op = OperatorRegistry.get(name)
        positional = op.calculate(*ranks)
        keyword = op.calculate(relations=list(ranks))
        pd.testing.assert_frame_equal(positional, keyword, obj=name)
        np.testing.assert_allclose(positional["A"], expected[name], atol=1e-12)
        prefix = op.calculate(*(panel.iloc[:1] for panel in ranks))
        pd.testing.assert_frame_equal(prefix, positional.iloc[:1], obj=f"{name}: prefix")


def test_entity_mobility_accepts_dynamic_prefix_and_matches_by_id():
    load_all()
    cur, cur_ids, prev, prev_ids = _entity_panels()
    args = (*cur, *cur_ids, *prev, *prev_ids)
    kwargs = {
        "s1": cur[0], "s2": cur[1], "sid1": cur_ids[0], "sid2": cur_ids[1],
        "p1": prev[0], "p2": prev[1], "psid1": prev_ids[0], "psid2": prev_ids[1],
    }
    op = OperatorRegistry.get("relation_rank_entity_mobility")
    positional = op.calculate(*args)
    keyword = op.calculate(**kwargs)
    pd.testing.assert_frame_equal(positional, keyword)
    np.testing.assert_allclose(positional["A"], [0.075, 0.1], atol=1e-12)
    prefix = op.calculate(*(panel.iloc[:1] for panel in args))
    pd.testing.assert_frame_equal(prefix, positional.iloc[:1])

    bad = cur[0].copy()
    bad.iloc[0, 0] = np.nan
    failed = op.calculate(bad, cur[1], *cur_ids, *prev, *prev_ids)
    assert np.isnan(failed.iloc[0, 0])
    with pytest.raises((TypeError, ValueError)):
        op.calculate(s1=cur[0], sid1=cur_ids[0], p1=prev[0])


def test_group_quantiles_default_keyword_positional_prefix_and_oracle():
    load_all()
    idx = pd.date_range("2024-01-02", periods=2)
    x = pd.DataFrame([[-2.0, -1.0, 1.0, 4.0], [-4.0, -2.0, 2.0, 8.0]], index=idx, columns=list("ABCD"))
    group = pd.DataFrame([["G"] * 4, ["G"] * 4], index=idx, columns=list("ABCD"))
    for name, expected in (("group_quantile_spread", 3.0), ("group_tail_ratio", 1.4)):
        op = OperatorRegistry.get(name)
        positional = op.calculate(x, group, 0.25, 0.75)
        keyword = op.calculate(x=x, group=group, q_low=0.25, q_high=0.75)
        pd.testing.assert_frame_equal(positional, keyword, obj=name)
        np.testing.assert_allclose(positional.iloc[0], expected, atol=1e-12)
        default = op.calculate(x, group)
        assert np.isfinite(default.to_numpy()).all()
        prefix = op.calculate(x.iloc[:1], group.iloc[:1], 0.25, 0.75)
        pd.testing.assert_frame_equal(prefix, positional.iloc[:1])
        with pytest.raises((TypeError, ValueError)):
            op.calculate(x, group, 0.8, 0.2)


def _polars_panel(frame):
    import polars as pl

    materialized = frame.copy()
    materialized.index.name = "date"
    return pl.from_pandas(materialized.reset_index())


def test_all_final_polars_backends_execute_positional_keyword_and_prefix():
    load_all()
    ranks = _rank_panels()
    pl_ranks = tuple(_polars_panel(panel) for panel in ranks)
    expected = {
        "relation_distribution_skew": 0.0,
        "relation_distribution_pearson_kurtosis": 1.7,
        "relation_distribution_excess_kurtosis": -1.3,
    }
    for name in DIST_NAMES:
        assert "polars" in OperatorRegistry.backends_for(name)
        op = OperatorRegistry.get(name, "polars")
        positional = op.calculate(*pl_ranks)
        keyword = op.calculate(relations=list(pl_ranks))
        np.testing.assert_allclose(positional["A"].to_numpy(), expected[name], atol=1e-12)
        np.testing.assert_allclose(keyword["A"].to_numpy(), positional["A"].to_numpy(), atol=1e-12)
        prefix = op.calculate(*(panel.head(1) for panel in pl_ranks))
        np.testing.assert_allclose(prefix["A"].to_numpy(), positional["A"].head(1).to_numpy(), atol=1e-12)

    cur, cur_ids, prev, prev_ids = _entity_panels()
    blocks = tuple(_polars_panel(panel) for panel in (*cur, *cur_ids, *prev, *prev_ids))
    mobility = OperatorRegistry.get("relation_rank_entity_mobility", "polars")
    positional = mobility.calculate(*blocks)
    kwargs = {
        "s1": blocks[0], "s2": blocks[1], "sid1": blocks[2], "sid2": blocks[3],
        "p1": blocks[4], "p2": blocks[5], "psid1": blocks[6], "psid2": blocks[7],
    }
    keyword = mobility.calculate(**kwargs)
    np.testing.assert_allclose(positional["A"].to_numpy(), [0.075, 0.1], atol=1e-12)
    np.testing.assert_allclose(keyword["A"].to_numpy(), positional["A"].to_numpy(), atol=1e-12)
    prefix = mobility.calculate(*(panel.head(1) for panel in blocks))
    np.testing.assert_allclose(prefix["A"].to_numpy(), [0.075], atol=1e-12)

    idx = pd.date_range("2024-01-02", periods=2)
    x = pd.DataFrame([[-2.0, -1.0, 1.0, 4.0], [-4.0, -2.0, 2.0, 8.0]], index=idx, columns=list("ABCD"))
    group = pd.DataFrame([["G"] * 4, ["G"] * 4], index=idx, columns=list("ABCD"))
    px, pg = _polars_panel(x), _polars_panel(group)
    for name, oracle in (("group_quantile_spread", 3.0), ("group_tail_ratio", 1.4)):
        op = OperatorRegistry.get(name, "polars")
        positional = op.calculate(px, pg, 0.25, 0.75)
        keyword = op.calculate(x=px, group=pg, q_low=0.25, q_high=0.75)
        np.testing.assert_allclose(positional.select(list("ABCD")).to_numpy()[0], oracle, atol=1e-12)
        np.testing.assert_allclose(keyword.select(list("ABCD")).to_numpy(), positional.select(list("ABCD")).to_numpy(), atol=1e-12)
        prefix = op.calculate(px.head(1), pg.head(1), 0.25, 0.75)
        np.testing.assert_allclose(prefix.select(list("ABCD")).to_numpy(), positional.select(list("ABCD")).head(1).to_numpy(), atol=1e-12)


def test_group_quantile_spread_sql_backend_executes_default_kwargs_and_prefix():
    import duckdb
    from factor_engine.backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
    from factor_engine.planner.logical_plan import PlanNode

    load_all()
    assert "sql" in OperatorRegistry.backends_for("group_quantile_spread")

    def column(name):
        return PlanNode(op="column", attrs={"name": name})

    def execute(frame, **attrs):
        plan = PlanNode(op="group_quantile_spread", inputs=[column("x"), column("grp")], attrs=attrs)
        compiled = compile_plan_to_sql(
            plan, table="observations", time_column="t", instrument_column="i", dialect=SqlDialect.DUCKDB,
        )
        assert compiled is not None
        con = duckdb.connect(":memory:")
        try:
            con.register("observations", frame)
            return con.execute(compiled.query).fetchdf().sort_values(["ts", "inst"]).reset_index(drop=True)
        finally:
            con.close()

    frame = pd.DataFrame({
        "t": [1] * 4 + [2] * 4,
        "i": list("ABCD") * 2,
        "x": [-2.0, -1.0, 1.0, 4.0, -4.0, -2.0, 2.0, 8.0],
        "grp": ["G"] * 8,
    })
    default = execute(frame)
    explicit = execute(frame, q_low=0.25, q_high=0.75)
    np.testing.assert_allclose(default["value"], [3.0] * 4 + [6.0] * 4, atol=1e-12)
    np.testing.assert_allclose(explicit["value"], default["value"], atol=1e-12)
    prefix = execute(frame[frame["t"] == 1])
    np.testing.assert_allclose(prefix["value"], default.loc[default["ts"] == 1, "value"], atol=1e-12)
