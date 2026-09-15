from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.common.static_adjacency import StaticAdjacency
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.direct_use import input_slot_specs
from factor_engine.planner.plan_hash import _typed_semantic_value
from factor_engine.runtime.operator_snapshot import _declared_parameter_schema


def _graph() -> StaticAdjacency:
    return StaticAdjacency(
        ("A", "B", "C", "D"),
        ((100, 1, 0, 1), (0, 100, 2, 0), (3, 0, 100, 1), (0, 0, 0, 100)),
    )


def test_directed_node_specific_oracle_and_no_self() -> None:
    ensure_cleaned_loaded()
    x = pd.DataFrame([[1, 2, 4, 8], [10, np.nan, 30, 40]], columns=list("ABCD"))
    op = OperatorRegistry.get("panel_peer_graph_aggregate", "pandas_numpy")
    out = op.calculate(x, _graph())
    np.testing.assert_allclose(out.iloc[0, :3], [5, 4, 4.5])
    assert np.isnan(out.iloc[0, 3])
    np.testing.assert_allclose(out.iloc[1, :3], [40, 30, 25])
    assert np.isnan(out.iloc[1, 3])
    assert len(set(out.iloc[0, :3])) > 1  # counterexample to old global-profile output


def test_dataframe_coercion_strict_threshold_and_weighted_direction() -> None:
    ensure_cleaned_loaded()
    graph = pd.DataFrame(_graph().matrix, index=list("ABCD"), columns=list("ABCD"))
    x = pd.DataFrame([[1, 2, 4, 8]], columns=list("ABCD"))
    op = OperatorRegistry.get("panel_peer_graph_aggregate", "pandas_numpy")
    mean = op.calculate(x, graph, threshold=1.0)  # equality is excluded
    assert np.isnan(mean.iloc[0, 0])
    np.testing.assert_allclose(mean.iloc[0, 1:3], [4, 1])
    assert np.isnan(mean.iloc[0, 3])
    weighted = op.calculate(x, graph, method="weighted_mean")
    np.testing.assert_allclose(weighted.iloc[0, :3], [5, 4, 11 / 4])


def test_polars_delegate_preserves_axes_and_matches_reference() -> None:
    ensure_cleaned_loaded()
    pdf = pd.DataFrame([[1.0, 2.0, 4.0, 8.0]], columns=list("ABCD"))
    graph = _graph()
    expected = OperatorRegistry.get("panel_peer_graph_aggregate", "pandas_numpy").calculate(pdf, graph)
    actual = OperatorRegistry.get("panel_peer_graph_aggregate", "polars").calculate(
        pl.from_pandas(pdf), graph
    )
    assert actual.columns == list(pdf.columns)
    np.testing.assert_allclose(actual.to_pandas(), expected)


def test_content_identity_schema_and_adjacency_slot() -> None:
    first = _graph()
    second = StaticAdjacency(first.symbols, first.matrix[:-1] + ((0, 1, 0, 100),))
    assert first.stable_hash() != second.stable_hash()
    assert _typed_semantic_value(first) != _typed_semantic_value(second)
    ensure_cleaned_loaded()
    op = OperatorRegistry.get("panel_peer_graph_aggregate", "pandas_numpy")
    schema, verified = _declared_parameter_schema(op.metadata.param_specs["similarity"])
    assert verified and schema["x-factor-engine-object"] == "static_adjacency"
    slots = input_slot_specs(
        "panel_peer_graph_aggregate",
        {"param_names": op.metadata.param_names},
        panel_params=op.metadata.panel_params,
        scalar_params=op.metadata.scalar_params,
    )
    similarity = next(slot for slot in slots if slot.parameter == "similarity")
    assert similarity.cardinality == "adjacency"
    assert similarity.axis_semantics == "source_symbol_x_destination_symbol"


def test_static_adjacency_rejects_ambiguous_symbols_and_implicit_weights() -> None:
    integer = StaticAdjacency((np.int64(1), 2), ((0, 1), (1, 0)))
    string = StaticAdjacency(("1", "2"), ((0, 1), (1, 0)))
    assert integer.symbols == (1, 2)
    assert integer.stable_hash() != string.stable_hash()
    with pytest.raises(TypeError, match="symbols must be strings or integers"):
        StaticAdjacency((pd.Timestamp("2024-01-01"),), ((0,),))
    with pytest.raises(TypeError, match="weights must be real numbers"):
        StaticAdjacency(("A",), (("1.0",),))
    with pytest.raises(TypeError, match="weights must be real numbers"):
        StaticAdjacency(("A",), ((True,),))


def test_fresh_load_and_actual_call() -> None:
    code = """
import pandas as pd
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.common.static_adjacency import StaticAdjacency
from factor_engine.cleaned_operators.registry import OperatorRegistry
ensure_cleaned_loaded()
x=pd.DataFrame([[1.,2.]],columns=['A','B'])
g=StaticAdjacency(('A','B'),((1.,1.),(1.,1.)))
out=OperatorRegistry.get('panel_peer_graph_aggregate','pandas_numpy').calculate(x,g)
assert out.iloc[0].tolist()==[2.,1.]
"""
    subprocess.run([sys.executable, "-c", code], check=True)
