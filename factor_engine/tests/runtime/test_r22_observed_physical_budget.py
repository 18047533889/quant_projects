from dataclasses import replace
from types import SimpleNamespace as NS
import pandas as pd
import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.backend_region import PhysicalBackend, Representation, TransferEdge, TransferTransform
from factor_engine.planner.physical_factor_dag import PhysicalFactorTask, TASK_ROOT, TASK_SOURCE_SCAN, TASK_CSE_SHARED
from factor_engine.runtime.task_resource_contract import TaskResourceContract
from factor_engine.runtime.observed_physical_budget import make_observed_budget_refresher


def fixture():
    col = PlanNode("column", attrs={"name": "encoded_rank_source", "estimated_rows": 1000000,
                                    "estimated_bytes": 100000000, "estimated_memory_bytes": 100000000})
    mid = PlanNode("abs", inputs=(col,))
    root = PlanNode("neg", inputs=(mid,))
    regions = [
        NS(region_id=rid, node_ids=(nid,), estimated_memory_bytes=100000000)
        for rid, nid in (("source_region", "c"), ("middle_region", "m"), ("root_region", "factor"))
    ]
    optimization = NS(
        logical_roots={"factor": root}, logical_shared_nodes={},
        discovered_node_ids={id(col): "c", id(mid): "m", id(root): "factor"},
        physical_plan=NS(regions=regions, edges=[
            TransferEdge(edge_id=str(i), producer_region=source, consumer_region=target,
                         source_backend=PhysicalBackend.PANDAS_NUMPY, target_backend=PhysicalBackend.PANDAS_NUMPY,
                         source_representation=Representation.PANDAS_LONG, target_representation=Representation.PANDAS_LONG,
                         transform=TransferTransform.SAME_BACKEND_NATIVE, estimated_rows=1000000,
                         estimated_bytes=100000000, estimated_transfer_ms=0.)
            for i, (source, target) in enumerate([
                ("source_region", "middle_region"), ("middle_region", "root_region")])
        ]),
        per_node_choices={key: NS(backend=PhysicalBackend.PANDAS_NUMPY) for key in ("c", "m", "factor")},
    )
    logical = TaskResourceContract(peak_memory_bytes=80, output_bytes=16, uncertainty=1.)
    raw = TaskResourceContract(peak_memory_bytes=900000000, uncertainty=1.)
    scan = PhysicalFactorTask("scan", "source_scan", TASK_SOURCE_SCAN, resource_contract=raw,
                              consumers=("root",), required_columns=("encoded_rank_source",))
    task = PhysicalFactorTask("root", "neg", TASK_ROOT, factor_name="factor",
                              inputs=("scan",), resource_contract=replace(logical, peak_memory_bytes=500000000))
    plan = NS(physical_dag=NS(tasks={"scan": scan, "root": task}), meta={})
    value = pd.Series([1., 2., 3.])
    ref = NS(meta={"loaded_columns": {"encoded_rank_source": value}})
    scheduler = NS(_buffer_results={"scan": ref}, _wave_refs={0: ref},
                   _wave_source_tasks={0: ("scan",)}, _wave_pending_consumers={0: {"root"}})
    return optimization, plan, scheduler, logical, raw, value


def test_refinement_keeps_all_ancestor_regions_transfers_and_raw_scan_contract():
    optimization, plan, scheduler, logical, raw, value = fixture()
    refresh = make_observed_budget_refresher(optimization, {"root": logical})
    refresh(scheduler, plan, {"scan"}, {"root"})
    size = value.memory_usage(index=True, deep=True)
    ledger = plan.meta["observed_physical_budget"]["root"]
    assert ledger["status"] == "refined"
    assert ledger["workspace_bytes"] == 3 * size
    assert ledger["transfer_bytes"] == 2 * size
    assert plan.physical_dag.tasks["root"].resource_contract.peak_memory_bytes == 5 * size + 16
    assert plan.physical_dag.tasks["scan"].resource_contract is raw
    assert optimization.physical_plan.regions[0].estimated_memory_bytes == 100000000
    before = plan.physical_dag.tasks["root"]
    refresh(scheduler, plan, {"scan"}, {"root"})
    assert plan.physical_dag.tasks["root"] is before


@pytest.mark.parametrize("problem", ["uncommitted", "released", "other_owner", "missing", "object", "unknown_axis", "running", "untyped_edge"])
def test_unproved_inputs_never_reduce_contract(problem):
    optimization, plan, scheduler, logical, raw, value = fixture()
    committed, remaining = {"scan"}, {"root"}
    if problem == "uncommitted":
        committed.clear()
    elif problem == "released":
        scheduler._wave_pending_consumers.clear()
    elif problem == "other_owner":
        scheduler._wave_pending_consumers[0] = {"other-root"}
    elif problem == "untyped_edge":
        optimization.physical_plan.edges[0] = NS(producer_region="source_region", consumer_region="middle_region")
    elif problem == "missing":
        scheduler._buffer_results["scan"].meta["loaded_columns"].clear()
    elif problem == "object":
        scheduler._buffer_results["scan"].meta["loaded_columns"]["encoded_rank_source"] = pd.Series(["a", "b", "c"])
    elif problem == "unknown_axis":
        old = optimization.logical_roots["factor"]
        root = replace(old, attrs={"output_rows": 999, "output_axis": "unknown"})
        optimization.discovered_node_ids[id(root)] = "factor"
        optimization.logical_roots["factor"] = root
    elif problem == "running":
        remaining.clear()
    before = plan.physical_dag.tasks["root"]
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, committed, remaining)
    assert plan.physical_dag.tasks["root"] is before
    assert plan.physical_dag.tasks["scan"].resource_contract is raw


def test_shared_task_preserves_ancestor_workspace():
    optimization, plan, scheduler, logical, raw, value = fixture()
    task = plan.physical_dag.tasks.pop("root")
    plan.physical_dag.tasks["cse:m"] = replace(task, task_id="cse:m", task_type=TASK_CSE_SHARED)
    scheduler._wave_pending_consumers[0] = {"cse:m"}
    make_observed_budget_refresher(optimization, {"cse:m": logical})(scheduler, plan, {"scan"}, {"cse:m"})
    ledger = plan.meta["observed_physical_budget"]["cse:m"]
    assert ledger["status"] == "refined"
    assert ledger["workspace_bytes"] == 2 * value.memory_usage(index=True, deep=True)


def test_duplicate_scope_column_with_different_owners_stays_conservative():
    optimization, plan, scheduler, logical, raw, value = fixture()
    task = plan.physical_dag.tasks["root"]
    plan.physical_dag.tasks["root"] = replace(task, inputs=("scan", "scan2"))
    plan.physical_dag.tasks["scan2"] = replace(plan.physical_dag.tasks["scan"], task_id="scan2")
    ref = NS(meta={"loaded_columns": {"encoded_rank_source": value.copy()}})
    scheduler._buffer_results["scan2"] = ref
    scheduler._wave_refs[1] = ref
    scheduler._wave_source_tasks[1] = ("scan2",)
    scheduler._wave_pending_consumers[1] = {"root"}
    before = plan.physical_dag.tasks["root"]
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, {"scan", "scan2"}, {"root"})
    assert plan.physical_dag.tasks["root"] is before


def test_unknown_numeric_input_operator_never_assumes_numeric_output():
    optimization, plan, scheduler, logical, raw, value = fixture()
    root = replace(optimization.logical_roots["factor"], op="custom_variable_width")
    optimization.logical_roots["factor"] = root
    optimization.discovered_node_ids[id(root)] = "factor"
    before = plan.physical_dag.tasks["root"]
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, {"scan"}, {"root"})
    assert plan.physical_dag.tasks["root"] is before


def test_nonidentity_transfer_keeps_original_conversion_bound():
    optimization, plan, scheduler, logical, raw, value = fixture()
    optimization.physical_plan.edges[0] = replace(
        optimization.physical_plan.edges[0], requires_dtype_cast=True)
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, {"scan"}, {"root"})
    ledger = plan.meta["observed_physical_budget"]["root"]
    assert ledger["transfer_bytes"] >= 100000000 + value.memory_usage(index=True, deep=True)


def test_merged_region_keeps_every_intermediate_materialization():
    optimization, plan, scheduler, logical, raw, value = fixture()
    optimization.physical_plan.regions = [NS(region_id="all", node_ids=("c", "m", "factor"))]
    optimization.physical_plan.edges = []
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, {"scan"}, {"root"})
    ledger = plan.meta["observed_physical_budget"]["root"]
    assert ledger["workspace_bytes"] == 3 * value.memory_usage(index=True, deep=True)


def test_unrelated_object_column_does_not_block_numeric_observation():
    optimization, plan, scheduler, logical, raw, _ = fixture()
    scheduler._buffer_results["scan"].meta["loaded_columns"]["irrelevant"] = pd.Series(["a"])
    make_observed_budget_refresher(optimization, {"root": logical})(scheduler, plan, {"scan"}, {"root"})
    assert plan.meta["observed_physical_budget"]["root"]["status"] == "refined"
