# -*- coding: utf-8
"""算子数值语义契约：NaN truthy、is_null/is_nan、对齐、scale、WindowSpec。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy rollout semantics superseded by canonical edge certification")

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.backend.numeric_semantics import (
    group_percentile_null_is_null,
    is_nan_excludes_null,
    truthy_nan_is_false,
    truthy_null_is_false,
    ts_argmax_empty_window_is_null,
    ts_sharpe_zero_std_is_null,
)
from factor_engine.backend.plan_params import PlanParamError
from factor_engine.backend.window_spec import WindowSpec
from factor_engine.cleaned_operators import load_all
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import FactorEngine
from tests.backend_parity.test_p0_edge_cases_triple_parity import duckdb_source, edge_source
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _col(name: str):
    return col(
        {
            "close": "Close",
            "flag": "Flag",
            "numer": "Numer",
            "open": "Open",
        }.get(name, name)
    )


def test_semantic_contract_flags():
    assert truthy_null_is_false()
    assert truthy_nan_is_false()
    assert is_nan_excludes_null()
    assert ts_argmax_empty_window_is_null()
    assert ts_sharpe_zero_std_is_null()
    assert group_percentile_null_is_null()


def test_nan_is_false_in_and_or(_loaded, edge_source, duckdb_source):
    idx = edge_source.data["close"].index
    nan_flag = pd.Series([float("nan")] * len(idx), index=idx)
    src = InMemorySeriesSource(
        data={
            **edge_source.data,
            "nan_flag": nan_flag,
        }
    )
    expr = F("and_")(col("nan_flag"), col("close"))
    pd_out = _run(src, expr, "pandas")["result"].sort_index()
    long_out = _run(src, expr, "polars_long")["result"].sort_index()
    assert (pd_out == 0.0).all()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False)


def test_is_null_vs_is_nan(_loaded, edge_source):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-03"), "A")],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([np.nan, 10.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    null_out = _run(src, F("is_null")(col("close")), "polars_long")["result"].sort_index()
    nan_out = _run(src, F("is_nan")(col("close")), "polars_long")["result"].sort_index()
    assert null_out.iloc[0] == 1.0
    assert null_out.iloc[1] == 0.0
    assert nan_out.iloc[1] == 0.0


def test_scale_to_parameter(_loaded, edge_source, duckdb_source):
    expr = F("scale")(col("close"), 2.0)
    pd_out = _run(edge_source, expr, "pandas")["result"].sort_index()
    long_out = _run(edge_source, expr, "polars_long")["result"].sort_index()
    sql_out = _run(duckdb_source, F("scale")(_col("close"), 2.0), "duckdb_sql")["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_binary_join_preserves_sparse_left_rows(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    left = pd.Series([1.0, 2.0, 3.0], index=idx)
    right_idx = idx[:2]
    right = pd.Series([10.0, 20.0], index=right_idx)
    src = InMemorySeriesSource(data={"left": left, "right": right})
    expr = F("add")(col("left"), col("right"))
    pd_out = _run(src, expr, "pandas")["result"].sort_index()
    long_out = _run(src, expr, "polars_long")["result"].sort_index()
    assert len(long_out) == len(pd_out) == 3
    assert math.isnan(long_out.loc[(pd.Timestamp("2024-01-02"), "B")])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_window_spec_rejects_dynamic_window(_loaded):
    node = PlanNode(
        op="ts_mean",
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="add", inputs=[], attrs={}),
        ],
    )
    with pytest.raises(PlanParamError, match="动态输入"):
        WindowSpec.from_plan_node(node)


def test_window_spec_rejects_fractional_ddof(_loaded):
    node = PlanNode(
        op="ts_std",
        attrs={"ddof": 1.5},
        inputs=[
            PlanNode(op="column", attrs={"name": "close"}, inputs=[]),
            PlanNode(op="literal", attrs={"value": 3}, inputs=[]),
        ],
    )
    with pytest.raises(PlanParamError, match="ddof"):
        WindowSpec.from_plan_node(node)


def test_backend_specific_capability(_loaded):
    from factor_engine.backend.evidence_provenance import evidence_artifact_valid
    from factor_engine.backend.operator_call_capability import CapabilityLevel, check_operator_call_capability

    ok_polars = check_operator_call_capability("rank", backend="polars_long", production=True)
    ok_duck = check_operator_call_capability("rank", backend="duckdb_sql", production=True)
    if evidence_artifact_valid():
        assert ok_polars.level == CapabilityLevel.PRODUCTION
        assert ok_duck.level == CapabilityLevel.PRODUCTION
    else:
        assert ok_polars.level == CapabilityLevel.RESEARCH
        assert ok_duck.level == CapabilityLevel.RESEARCH
    bad = check_operator_call_capability("ts_mad", backend="polars_long", production=True)
    assert bad.level == CapabilityLevel.RESEARCH


def test_duplicate_keys_raise_alignment_error(_loaded):
    import polars as pl
    from factor_engine.backend.long_alignment import AlignmentError, anchor_left_join_binary

    left = pl.LazyFrame(
        {"ts": [1, 1], "inst": ["A", "A"], "_v": [1.0, 2.0]},
        schema={"ts": pl.Int64, "inst": pl.String, "_v": pl.Float64},
    )
    right = pl.LazyFrame(
        {"ts": [1], "inst": ["A"], "_v": [10.0]},
        schema={"ts": pl.Int64, "inst": pl.String, "_v": pl.Float64},
    )
    with pytest.raises(AlignmentError, match="重复 key"):
        anchor_left_join_binary(left, right)


def test_expanding_std_high_dynamic_range(_loaded):
    """大绝对值 + 微小波动：expanding_std 不应因消减误差变 NULL。"""
    n = 500
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=n, freq="D"), ["A"]],
        names=["timestamp", "instrument"],
    )
    base = 1e12
    noise = pd.Series(np.linspace(-1e-6, 1e-6, n), index=idx)
    close = pd.Series(base + noise.values, index=idx)
    src = InMemorySeriesSource(data={"close": close})
    expr = F("expanding_std")(col("close"))
    pd_out = _run(src, expr, "pandas")["result"].sort_index()
    long_out = _run(src, expr, "polars_long")["result"].sort_index()
    tail = long_out.iloc[1:]
    assert tail.notna().all()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-8)


def test_ewm_spec_rejects_invalid_span(_loaded):
    from factor_engine.backend.ewm_spec import EwmSpec
    from factor_engine.backend.plan_params import PlanParamError
    from factor_engine.planner.logical_plan import PlanNode

    bad = PlanNode(op="ewm_mean", inputs=[PlanNode(op="column", attrs={"name": "x"})], attrs={"span": 2.5})
    with pytest.raises(PlanParamError):
        EwmSpec.from_plan_node(bad)
    neg = PlanNode(op="ewm_mean", inputs=[PlanNode(op="column", attrs={"name": "x"})], attrs={"span": -3})
    with pytest.raises(PlanParamError):
        EwmSpec.from_plan_node(neg)
