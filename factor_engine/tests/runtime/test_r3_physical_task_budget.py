"""Contract regression: physical workspace reaches actual ROOT admission."""
from types import SimpleNamespace as NS
import pytest
from factor_engine.runtime.batch_service import _bind_physical_task_budgets
from factor_engine.planner.physical_factor_dag import PhysicalFactorTask, TASK_CSE_SHARED, TASK_ROOT
from factor_engine.runtime.task_resource_contract import TaskResourceContract


def _fixture():
    tasks = {name: PhysicalFactorTask(name, "ts_kurt", TASK_ROOT, factor_name=name,
             resource_contract=TaskResourceContract(peak_memory_bytes=80, output_bytes=16))
             for name in ("a", "b")}
    scheduler = NS(physical_dag=NS(tasks=tasks), meta={})
    physical = NS(regions=(
        NS(region_id="input", node_ids=("x",), estimated_memory_bytes=128),
        NS(region_id="sql", node_ids=("a",), estimated_memory_bytes=80000),
        NS(region_id="cpu", node_ids=("b",), estimated_memory_bytes=256)),
        edges=(NS(consumer_region="sql", producer_region="input", estimated_bytes=64),))
    return scheduler, NS(physical_plan=physical)


def test_selected_workspace_replaces_panel_only_root_budget_without_charging_unrelated_root():
    scheduler, optimization = _fixture()
    ledger = _bind_physical_task_budgets(scheduler, optimization)
    first, other = (scheduler.physical_dag.tasks[key].resource_contract for key in ("a", "b"))
    assert first.peak_memory_bytes == 80000 + 128 + 64 + 16
    assert other.peak_memory_bytes == 256 + 16
    assert first.admissible_peak_bytes >= first.peak_memory_bytes
    assert ledger["a"]["transfer_bytes"] == 64
    assert ledger["a"]["region_ids"] == ("input", "sql")
    assert scheduler.meta["physical_resource_admission"] == ledger
    _bind_physical_task_budgets(scheduler, optimization)
    assert scheduler.physical_dag.tasks["a"].resource_contract.peak_memory_bytes == first.peak_memory_bytes


@pytest.mark.parametrize("failure", ["missing", "zero"])
def test_invalid_physical_budget_fails_before_dispatch(failure):
    scheduler, optimization = _fixture()
    if failure == "missing":
        optimization.physical_plan.regions[1].node_ids = ("wrong",)
    else:
        optimization.physical_plan.regions[1].estimated_memory_bytes = 0
    with pytest.raises(ValueError):
        _bind_physical_task_budgets(scheduler, optimization)


def test_shared_task_receives_selected_physical_workspace_budget():
    task = PhysicalFactorTask(
        "cse:shared", "ts_kurt", TASK_CSE_SHARED,
        resource_contract=TaskResourceContract(peak_memory_bytes=80, output_bytes=16),
    )
    scheduler = NS(physical_dag=NS(tasks={"cse:shared": task}), meta={})
    physical = NS(
        regions=(NS(region_id="sql", node_ids=("shared",), estimated_memory_bytes=80000),),
        edges=(),
    )
    ledger = _bind_physical_task_budgets(scheduler, NS(physical_plan=physical))
    contract = scheduler.physical_dag.tasks["cse:shared"].resource_contract
    assert contract.peak_memory_bytes == 80016
    assert ledger["cse:shared"]["workspace_bytes"] == 80000


def test_same_region_roots_reuse_one_ancestry_computation():
    tasks = {
        name: PhysicalFactorTask(
            name, "abs", TASK_ROOT, factor_name=name,
            resource_contract=TaskResourceContract(peak_memory_bytes=8),
        )
        for name in ("a", "b")
    }
    scheduler = NS(physical_dag=NS(tasks=tasks), meta={})
    physical = NS(
        regions=(NS(region_id="merged", node_ids=("a", "b"), estimated_memory_bytes=128),),
        edges=(),
    )
    ledger = _bind_physical_task_budgets(scheduler, NS(physical_plan=physical))
    assert ledger["a"]["region_ids"] is ledger["b"]["region_ids"]
    assert ledger["a"]["region_ids"] == ("merged",)
