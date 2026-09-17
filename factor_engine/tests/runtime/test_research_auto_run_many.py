from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api import col, ts_mean, ts_std
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.resource_broker import ResourceBroker


def _coherent_broker():
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=2,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    hard = broker.hard_memory_limit
    snapshot = replace(
        broker.snapshot(),
        hard_memory_limit=hard,
        cgroup_memory_current=0,
        host_mem_available=hard,
        process_rss=0,
        worker_rss=0,
        process_family_rss=0,
        process_family_pss=0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snapshot
    return broker


@pytest.mark.parametrize("enable_cse", [True, False])
def test_research_auto_run_many_executes_admitted_physical_routes_with_parity(enable_cse):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    close = col("close")
    factors = [
        Factor(name="mean", expr=ts_mean(close, 3)),
        Factor(name="std", expr=ts_std(close, 3)),
    ]
    perf = PerfConfig(max_workers=1, native_fusion=False)

    source = Source(values)
    source.instrument_filter = ("A", "B", "C")
    source.start_date = index.levels[0].min().tz_localize("UTC")
    source.end_date = index.levels[0].max().tz_localize("UTC")
    source.schema = {"close": "float64"}
    engine = FactorEngine(
        backend=build_backend("auto"), data_source=source, run_mode="research"
    )
    engine.resource_broker = _coherent_broker()
    auto_result = engine.run_many(factors, perf=perf, enable_cse=enable_cse)

    expected_mean = values.groupby(level="instrument").rolling(3, min_periods=1).mean()
    expected_mean.index = expected_mean.index.droplevel(0)
    expected_mean = expected_mean.reorder_levels(values.index.names).sort_index()
    pd.testing.assert_series_equal(
        auto_result["results"]["mean"], expected_mean, check_names=False
    )
    expected_std = values.groupby(level="instrument").rolling(3, min_periods=1).std()
    expected_std.index = expected_std.index.droplevel(0)
    expected_std = expected_std.reorder_levels(values.index.names).sort_index()
    pd.testing.assert_series_equal(
        auto_result["results"]["std"], expected_std, check_names=False
    )
    paths = auto_result["backend_paths"]
    assert all("physical_plan" in path for path in paths.values())
    assert any(path["actual_backend"] != "pandas" for path in paths.values())


def test_shape_evidence_reuses_anchor_columns_but_not_other_scope_or_axis():
    from factor_engine.api.source_ref import encode_source_ref, make_source_ref
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    anchor_a = PlanNode("column", attrs={"name": "close"})
    anchor_b = PlanNode("column", attrs={"name": "volume"})
    source_ref = PlanNode(
        "column",
        attrs={"name": encode_source_ref(make_source_ref("OtherTable", "close"))},
    )
    explicit_table = PlanNode("column", attrs={"name": "close", "table": "OtherTable"})
    changed_axis = PlanNode(
        "resample", inputs=(anchor_a,), attrs={"output_frequency": "monthly"}
    )
    nodes = {
        "a": anchor_a,
        "b": anchor_b,
        "ref": source_ref,
        "table": explicit_table,
        "axis": changed_axis,
    }
    graph = {"a": [], "b": [], "ref": [], "table": [], "axis": ["a"]}
    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
        nodes,
        graph,
        roots=(),
        global_rows=12,
        global_bytes=96,
        global_memory=192,
    )

    assert estimates["a"] == (12, 96, 192)
    assert estimates["b"] == (12, 96, 192)
    assert estimates["ref"] == (0, 0, 0)
    assert estimates["table"] == (0, 0, 0)
    assert estimates["axis"] == (0, 0, 0)


def test_only_run_many_may_defer_research_hybrid_to_physical_consumer():
    import pytest
    from factor_engine.runtime.engine import (
        PhysicalPlanRequiredError,
        _assert_public_batch_authority,
    )

    engine = FactorEngine(
        backend=build_backend("auto"), data_source=Source(pd.Series(dtype=float)),
        run_mode="research",
    )
    _assert_public_batch_authority(engine, research_physical_consumer=True)
    with pytest.raises(PhysicalPlanRequiredError):
        _assert_public_batch_authority(engine)
    with pytest.raises(PhysicalPlanRequiredError):
        engine.run(Factor(name="single", expr=col("close")))
    with pytest.raises(PhysicalPlanRequiredError):
        list(engine.run_many_iter([Factor(name="iter", expr=col("close"))]))


def test_global_optimizer_disambiguates_distinct_nodes_with_same_structural_id():
    from types import SimpleNamespace

    from factor_engine.planner.batch_global_optimizer import optimize_batch_global
    from factor_engine.planner.logical_plan import PlanNode

    left = PlanNode("column", attrs={"name": "close"}, node_id="same-column")
    right = PlanNode("column", attrs={"name": "close"}, node_id="same-column")
    root = PlanNode("add", inputs=(left, right), node_id="root")
    ctx = SimpleNamespace(
        run_mode="research",
        runtime_stats={
            "row_count_estimate": 12,
            "estimated_bytes": 96,
            "estimated_memory_bytes": 192,
        },
    )

    result = optimize_batch_global({"factor": root}, {}, {}, ctx)

    assert result.execution_ready
    assert len(result.discovered_node_ids) == 3
    assert len(set(result.discovered_node_ids.values())) == 3


def test_physical_regions_are_ordered_by_region_dependencies():
    from types import SimpleNamespace

    from factor_engine.backend.contracts import ExecutionKind
    from factor_engine.planner.backend_region import PhysicalBackend, Representation
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        NodeBackendChoice,
        PhysicalBatchGlobalOptimizer,
    )

    nodes = {
        "leaf": PlanNode("column", attrs={"name": "close"}),
        "middle": PlanNode("abs"),
        "root": PlanNode("add"),
    }
    graph = {"leaf": [], "middle": [], "root": ["leaf", "middle"]}

    def choice(node_id, backend, representation):
        return NodeBackendChoice(
            node_id=node_id,
            backend=backend,
            compute_cost_ms=1.0,
            transfer_from_children_ms=0.0,
            total_cost_ms=1.0,
            representation=representation,
            execution_kind=ExecutionKind.PANDAS_REFERENCE,
            production_certified=True,
        )

    choices = {
        "leaf": choice(
            "leaf", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG
        ),
        "middle": choice(
            "middle", PhysicalBackend.POLARS_PANEL, Representation.POLARS_LONG
        ),
        "root": choice(
            "root", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG
        ),
    }
    optimizer = PhysicalBatchGlobalOptimizer()

    plan = optimizer._build_physical_plan(
        choices,
        {},
        0.0,
        3.0,
        0.0,
        SimpleNamespace(),
        graph,
        ("root",),
        {node_id: (12, 96, 192) for node_id in nodes},
        nodes,
    )

    order = {region_id: index for index, region_id in enumerate(plan.topological_order)}
    assert all(
        order[edge.producer_region] < order[edge.consumer_region]
        for edge in plan.edges
    )


def test_shape_evidence_ignores_scalar_literal_when_propagating_panel_shape():
    import sys

    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    column = PlanNode("column", attrs={"name": "close"})
    literal = PlanNode("literal", attrs={"value": 0.5})
    operator = PlanNode("add", inputs=(column, literal))
    nodes = {"column": column, "literal": literal, "operator": operator}
    graph = {"column": [], "literal": [], "operator": ["column", "literal"]}

    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
        nodes,
        graph,
        roots=(),
        global_rows=12,
        global_bytes=96,
        global_memory=192,
    )

    assert estimates["column"] == (12, 96, 192)
    assert estimates["literal"] == (
        1, sys.getsizeof(0.5), sys.getsizeof(0.5)
    )
    assert estimates["operator"] == (12, 96, 192)


def _physical_choice(node_id, backend, representation):
    from factor_engine.backend.contracts import ExecutionKind
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        NodeBackendChoice,
    )

    return NodeBackendChoice(
        node_id=node_id,
        backend=backend,
        compute_cost_ms=1.0,
        transfer_from_children_ms=0.0,
        total_cost_ms=1.0,
        representation=representation,
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        production_certified=True,
    )


def test_unique_consumer_literal_is_embedded_in_consumer_region():
    from types import SimpleNamespace

    from factor_engine.planner.backend_region import PhysicalBackend, Representation
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    literal = PlanNode("literal", attrs={"value": 0.5})
    operator = PlanNode("add", inputs=(literal,))
    nodes = {"literal": literal, "operator": operator}
    graph = {"literal": [], "operator": ["literal"]}
    choices = {
        "literal": _physical_choice("literal", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG),
        "operator": _physical_choice("operator", PhysicalBackend.DUCKDB_SQL, Representation.DUCKDB_RELATION),
    }

    PhysicalBatchGlobalOptimizer()._co_locate_residency_neutral_nodes(
        choices, nodes, graph, estimate_rows=12, ctx=SimpleNamespace()
    )

    assert choices["literal"].backend == PhysicalBackend.DUCKDB_SQL
    assert choices["literal"].representation == Representation.DUCKDB_RELATION
    plan = PhysicalBatchGlobalOptimizer()._build_physical_plan(
        choices,
        {},
        0.0,
        2.0,
        0.0,
        SimpleNamespace(),
        graph,
        ("operator",),
        {"literal": (0, 0, 0), "operator": (12, 96, 192)},
        nodes,
    )
    assert len(plan.regions) == 1
    assert plan.regions[0].estimated_rows == 12


def test_literal_with_mixed_consumer_residencies_remains_unmoved():
    from types import SimpleNamespace

    from factor_engine.planner.backend_region import PhysicalBackend, Representation
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    literal = PlanNode("literal", attrs={"value": 0.5})
    left = PlanNode("add", inputs=(literal,))
    right = PlanNode("subtract", inputs=(literal,))
    nodes = {"literal": literal, "left": left, "right": right}
    graph = {"literal": [], "left": ["literal"], "right": ["literal"]}
    choices = {
        "literal": _physical_choice("literal", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG),
        "left": _physical_choice("left", PhysicalBackend.DUCKDB_SQL, Representation.DUCKDB_RELATION),
        "right": _physical_choice("right", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG),
    }

    PhysicalBatchGlobalOptimizer()._co_locate_residency_neutral_nodes(
        choices, nodes, graph, estimate_rows=12, ctx=SimpleNamespace()
    )

    assert choices["literal"].backend == PhysicalBackend.PANDAS_NUMPY
    assert choices["literal"].representation == Representation.PANDAS_LONG


def test_source_ref_shape_uses_only_exact_column_scan_cost():
    from types import SimpleNamespace

    from factor_engine.api.source_ref import encode_source_ref, make_source_ref
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    encoded = encode_source_ref(
        make_source_ref(
            "StockValuationDaily",
            "TurnoverRatio",
            dataset="ashare_stock_valuation_daily",
            market="ashare",
        )
    )
    source = PlanNode("column", attrs={"name": encoded})
    nodes = {"source": source}
    graph = {"source": []}
    cost = SimpleNamespace(
        estimated_rows=2760,
        projection_bytes=176640,
        selected_bytes=0,
    )

    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
        nodes,
        graph,
        roots=(),
        global_rows=999,
        global_bytes=999,
        global_memory=999,
        column_scan_costs={encoded: cost},
    )

    assert estimates["source"] == (2760, 176640, 176640)


def test_literal_shape_is_real_scalar_when_cross_backend_consumers_split_region():
    import sys

    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    literal = PlanNode("literal", attrs={"value": 20})
    nodes = {"literal": literal}

    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
        nodes,
        {"literal": []},
        roots=(),
        global_rows=2760,
        global_bytes=176640,
        global_memory=176640,
    )

    assert estimates["literal"] == (1, sys.getsizeof(20), sys.getsizeof(20))


def test_literal_shape_sizes_text_and_rejects_container_payloads():
    import sys

    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import (
        PhysicalBatchGlobalOptimizer,
    )

    text = "window-name"
    nodes = {
        "text": PlanNode("literal", attrs={"value": text}),
        "wide": PlanNode("literal", attrs={"value": list(range(1000))}),
    }
    estimates = PhysicalBatchGlobalOptimizer._derive_node_estimates(
        nodes,
        {"text": [], "wide": []},
        roots=(),
        global_rows=2760,
        global_bytes=176640,
        global_memory=176640,
    )

    assert estimates["text"] == (1, sys.getsizeof(text), sys.getsizeof(text))
    assert estimates["wide"] == (0, 0, 0)


def test_source_ref_shape_rejects_mismatched_column_scan_cost():
    from types import SimpleNamespace

    from factor_engine.api.source_ref import encode_source_ref, make_source_ref
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.multibackend.batch_global_optimizer import PhysicalBatchGlobalOptimizer

    encoded = encode_source_ref(make_source_ref("StockValuationDaily", "TurnoverRatio"))
    source = PlanNode("column", attrs={"name": encoded})
    wrong = encode_source_ref(make_source_ref("StockIncome", "NetProfit"))
