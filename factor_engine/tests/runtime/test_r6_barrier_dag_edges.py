"""Barrier lowering must preserve reciprocal edges and valid critical paths."""
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.physical_lowerer import lower_root_plan
from factor_engine.planner.physical_factor_dag import PhysicalFactorDAG, PhysicalFactorTask
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()

def test_nested_barriers_have_only_real_reverse_edges():
    plan=PlanNode("cs_rank_gaussian",inputs=[PlanNode("ts_kurt",inputs=[PlanNode("column",attrs={"name":"x"})])])
    tasks=lower_root_plan(plan,factor_name="nested",rows=32,instruments=2,preferred_backend="pandas_numpy")
    dag=PhysicalFactorDAG(tasks={t.task_id:t for t in tasks})
    assert len(tasks)>=4
    for task in tasks:
        for successor in task.consumers:
            assert task.task_id in dag.tasks[successor].inputs
        for predecessor in task.inputs:
            assert task.task_id in dag.tasks[predecessor].consumers
    order=dag.topological_order()
    positions={task:i for i,task in enumerate(order)}
    for task in tasks:
        assert all(positions[p]<positions[task.task_id] for p in task.inputs)
    scheduler=object.__new__(AdaptiveBatchScheduler)
    scheduler._critical_path_cache=None
    costs={task.task_id:1. for task in tasks}
    scores=scheduler._critical_path_scores(dag,costs)
    assert scores[order[0]]==len(tasks)
    assert dag.critical_path_remaining_ms(order[0],costs)==len(tasks)
    assert scheduler._critical_path_scores(dag,costs) is scores

def test_cyclic_graph_keeps_explicit_error_not_keyerror():
    tasks={
        "a":PhysicalFactorTask("a","a","ROOT",inputs=("b",),consumers=("b",)),
        "b":PhysicalFactorTask("b","b","ROOT",inputs=("a",),consumers=("a",)),
    }
    dag=PhysicalFactorDAG(tasks=tasks)
    scheduler=object.__new__(AdaptiveBatchScheduler)
    scheduler._critical_path_cache=None
    with pytest.raises(RuntimeError,match="cycle"):
        scheduler._critical_path_scores(dag,{"a":1.,"b":1.})
    with pytest.raises(RuntimeError,match="cycle"):
        dag.critical_path_remaining_ms("a",{"a":1.,"b":1.})
