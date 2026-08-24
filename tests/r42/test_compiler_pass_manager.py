from __future__ import annotations

from dataclasses import dataclass

import pytest

from factor_engine.planner.compiler_pass import (
    CompilerCost,
    CompilerPassManager,
    IRKind,
    InvariantResult,
    NumericEquivalence,
    NumericPolicy,
    PassContext,
    PassContract,
    PassInvariantError,
    PassLegalityError,
    SemanticEquivalence,
)
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.optimizer import Optimizer


@dataclass(frozen=True)
class _Pass:
    contract: PassContract
    output: PlanNode

    def run(self, plan: PlanNode, context: PassContext) -> PlanNode:
        del plan, context
        return self.output


def _contract(**overrides) -> PassContract:
    values = {
        "name": "test_pass",
        "input_ir": IRKind.LOGICAL_QUERY_GRAPH,
        "output_ir": IRKind.OPTIMIZED_LOGICAL_GRAPH,
        "semantic_equivalence": SemanticEquivalence.VALUE_PRESERVING,
        "numeric_equivalence": NumericEquivalence.EXACT_VALUE,
    }
    values.update(overrides)
    return PassContract(**values)


def test_pass_manager_emits_typed_trace_and_stage_transition():
    before = PlanNode(op="column", attrs={"name": "close"})
    after = PlanNode(op="column", attrs={"name": "close"})
    result = CompilerPassManager([_Pass(_contract(), after)]).run(before)

    assert result.plan is after
    assert result.ir_kind is IRKind.OPTIMIZED_LOGICAL_GRAPH
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.name == "test_pass"
    assert trace.input_ir is IRKind.LOGICAL_QUERY_GRAPH
    assert trace.output_ir is IRKind.OPTIMIZED_LOGICAL_GRAPH
    assert trace.semantic_equivalence is SemanticEquivalence.VALUE_PRESERVING
    assert trace.numeric_equivalence is NumericEquivalence.EXACT_VALUE
    assert trace.changed is False
    assert trace.duration_ns >= 0


def test_pass_manager_fails_closed_on_invariant_violation():
    def reject(before, after, context):
        del before, after, context
        return InvariantResult("NO_NEW_COLUMN", False, "unexpected future column")

    before = PlanNode(op="column", attrs={"name": "close"})
    after = PlanNode(op="column", attrs={"name": "future_close"})
    compiler_pass = _Pass(_contract(invariants=(reject,)), after)

    with pytest.raises(PassInvariantError, match="unexpected future column"):
        CompilerPassManager([compiler_pass]).run(before)


def test_numeric_policy_blocks_relaxed_pass_in_production():
    plan = PlanNode(op="column", attrs={"name": "close"})
    relaxed = _Pass(
        _contract(numeric_equivalence=NumericEquivalence.TOLERANCE_EQUIVALENT),
        plan,
    )
    context = PassContext(production=True, numeric_policy=NumericPolicy.production_default())

    with pytest.raises(PassLegalityError, match="TOLERANCE_EQUIVALENT"):
        CompilerPassManager([relaxed]).run(plan, context=context)


def test_cost_ranking_selects_lowest_legal_candidate():
    plan = PlanNode(op="column", attrs={"name": "close"})

    def expensive(before, after, context):
        del before, after, context
        return CompilerCost(scan_bytes=1000, conversion_bytes=500)

    def cheap(before, after, context):
        del before, after, context
        return CompilerCost(scan_bytes=100, conversion_bytes=10)

    first = CompilerPassManager([_Pass(_contract(cost_estimator=expensive), plan)]).run(plan)
    second = CompilerPassManager([_Pass(_contract(cost_estimator=cheap), plan)]).run(plan)
    assert CompilerPassManager.choose_lowest_cost([first, second]) is second


def test_optimizer_is_integrated_with_declared_pass_pipeline():
    col = PlanNode(op="column", attrs={"name": "close"})
    plan = PlanNode(
        op="add",
        inputs=(col, PlanNode(op="literal", attrs={"value": 1.0})),
        semantic_attrs={"unit": "return", "grain": "daily"},
    )
    optimized, traces = Optimizer().optimize_with_pass_trace(plan)

    assert optimized == Optimizer().optimize(plan)
    assert [trace.name for trace in traces] == [
        "literal_fold",
        "parameter_validation",
        "composite_lowering",
        "post_lowering_literal_fold",
        "fastpath_rewrite",
        "parameter_canonicalization",
    ]
    assert traces[-1].output_ir is IRKind.OPTIMIZED_LOGICAL_GRAPH
    assert all(trace.invariants for trace in traces)
    assert optimized.semantic_attrs == plan.semantic_attrs


def test_legacy_lowering_trace_api_remains_compatible():
    plan = PlanNode(op="add", inputs=(
        PlanNode(op="column", attrs={"name": "close"}),
        PlanNode(op="literal", attrs={"value": 1.0}),
    ))
    final, folded, lowering_trace = Optimizer().optimize_with_trace(plan)

    assert isinstance(final, PlanNode)
    assert isinstance(folded, PlanNode)
    assert isinstance(lowering_trace, tuple)
