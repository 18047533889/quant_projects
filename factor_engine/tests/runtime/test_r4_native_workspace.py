"""Centered native windows must be charged through real task admission."""
from types import SimpleNamespace as NS
import sys
import pytest
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.backend_region import PhysicalBackend
from factor_engine.runtime.multibackend.batch_global_optimizer import PhysicalBatchGlobalOptimizer as Optimizer
from factor_engine.runtime.batch_service import _bind_physical_task_budgets
from factor_engine.planner.physical_factor_dag import PhysicalFactorTask, TASK_ROOT
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def node(op, window):
    inputs = [PlanNode("column", attrs={"name": "x"})]
    if op == "ts_cov":
        inputs.append(PlanNode("column", attrs={"name": "y"}))
    inputs.append(PlanNode("literal", attrs={"value": window}))
    return PlanNode(op, inputs=tuple(inputs))


@pytest.mark.parametrize("backend,op,coefficient", [
    ("POLARS_LONG", "ts_cov", 192),
    ("POLARS_LONG", "ts_var", 128), ("POLARS_LONG", "ts_std", 128),
    ("DUCKDB_SQL", "ts_cov", 160), ("DUCKDB_SQL", "ts_kurt", 160),
    ("DUCKDB_SQL", "ts_var", 96), ("DUCKDB_SQL", "ts_std", 96),
])
def test_window_budget_reaches_root(backend, op, coefficient):
    n = node(op, 7)
    choice = NS(backend=getattr(PhysicalBackend, backend))
    workspace = 100 * (coefficient * 7 + 64)
    memory = Optimizer._node_execution_memory_bytes(n, choice, 100, 800)
    assert memory == workspace + 800
    task = PhysicalFactorTask("root", op, TASK_ROOT, factor_name="root",
        resource_contract=TaskResourceContract(peak_memory_bytes=800, output_bytes=80))
    scheduler = NS(physical_dag=NS(tasks={"root": task}), meta={})
    physical = NS(regions=(NS(region_id="native", node_ids=("root",),
                              estimated_memory_bytes=memory),), edges=())
    _bind_physical_task_budgets(scheduler, NS(physical_plan=physical))
    assert scheduler.physical_dag.tasks["root"].resource_contract.peak_memory_bytes == memory + 80


def test_two_centered_nodes_sum_workspaces_not_just_max():
    nodes = {"a": node("ts_cov", 7), "b": node("ts_std", 11)}
    choices = {key: NS(backend=PhysicalBackend.DUCKDB_SQL) for key in nodes}
    estimates = {"a": (100, 800, 800), "b": (200, 1600, 1600)}
    actual = Optimizer._region_execution_memory_bytes(list(nodes), nodes, choices, estimates)
    assert actual == 1600 + 100 * (160 * 7 + 64) + 200 * (96 * 11 + 64)


def test_workspace_zero_rows_and_saturation():
    n = node("ts_cov", 7)
    choice = NS(backend=PhysicalBackend.DUCKDB_SQL)
    assert Optimizer._node_additional_workspace_bytes(n, choice, 0) == 0
    assert Optimizer._node_additional_workspace_bytes(n, choice, -1) == 0
    assert Optimizer._node_execution_memory_bytes(n, choice, sys.maxsize, 800) == sys.maxsize


@pytest.mark.parametrize("op", ["ts_var", "ts_std"])
def test_panel_registry_route_is_not_charged_long_emitter_list_buffers(op):
    choice = NS(backend=PhysicalBackend.POLARS_PANEL)
    assert Optimizer._node_additional_workspace_bytes(node(op, 7), choice, 100) == 0


def test_panel_ts_cov_is_charged_for_native_staged_pairwise_buffers():
    choice = NS(backend=PhysicalBackend.POLARS_PANEL)
    assert Optimizer._node_additional_workspace_bytes(
        node("ts_cov", 7), choice, 100
    ) == 100 * (96 * 7 + 24)
