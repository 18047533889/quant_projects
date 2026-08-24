# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

from factor_engine.runtime.multibackend.batch_global_optimizer import (
    BatchGlobalOptimizer,
    OptimizationOpportunity,
)
from factor_engine.runtime.multibackend.streaming_executor import StreamingExecutionPlanner


class _OpportunityOptimizer(BatchGlobalOptimizer):
    def __init__(self, replacement=None):
        super().__init__(min_savings_bytes=1, min_savings_work=1)
        self._replacement = replacement

    def _detect_cross_root_cse(self, dag, cost_fn):
        return [
            OptimizationOpportunity(
                kind="cse",
                savings_bytes=100,
                savings_work=50.0,
                affected_roots=["root"],
            )
        ]

    def _apply_optimization(self, dag, opp):
        return dag if self._replacement is None else self._replacement


def _dag(op="add", dependencies=()):
    task = SimpleNamespace(
        task_type="root",
        dependencies=list(dependencies),
        node_ref=SimpleNamespace(op=op, inputs=[], params={}),
    )
    return SimpleNamespace(tasks={"root": task})


def test_optimizer_does_not_account_for_unchanged_dag():
    dag = _dag()
    result = _OpportunityOptimizer().optimize_batch(
        dag, enable_predicate_push=False, enable_constant_fold=False
    )

    assert len(result.opportunities) == 1
    assert result.applied_count == 0
    assert result.total_savings_bytes == 0
    assert result.total_savings_work == 0.0
    assert result.rewritten_dag is None


def test_optimizer_accounts_only_for_structural_rewrite():
    dag = _dag()
    rewritten = _dag(op="multiply")
    result = _OpportunityOptimizer(rewritten).optimize_batch(
        dag, enable_predicate_push=False, enable_constant_fold=False
    )

    assert result.applied_count == 1
    assert result.total_savings_bytes == 100
    assert result.total_savings_work == 50.0
    assert result.rewritten_dag is rewritten


def test_unknown_streaming_capability_fails_closed_until_registered():
    planner = StreamingExecutionPlanner()

    assert planner._is_streaming_capable("unclassified_operator") is False
    planner.register_streaming_operator("unclassified_operator")
    assert planner._is_streaming_capable("unclassified_operator") is True


def test_streaming_priority_preserves_dependency_order():
    tasks = {
        "materialize": SimpleNamespace(
            dependencies=[], node_ref=SimpleNamespace(op="sort")
        ),
        "stream": SimpleNamespace(
            dependencies=["materialize"], node_ref=SimpleNamespace(op="add")
        ),
        "independent_stream": SimpleNamespace(
            dependencies=[], node_ref=SimpleNamespace(op="multiply")
        ),
    }
    dag = SimpleNamespace(tasks=tasks)
    planner = StreamingExecutionPlanner()

    plan = planner.plan_streaming_execution(dag)

    assert plan.execution_order.index("materialize") < plan.execution_order.index("stream")
    assert plan.execution_order.index("independent_stream") < plan.execution_order.index(
        "materialize"
    )
