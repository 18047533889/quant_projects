"""Adapters that express the legacy optimizer pipeline as typed compiler passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from planner.compiler_pass import (
    DEFAULT_REWRITE_INVARIANTS,
    CompilerPassManager,
    IRKind,
    NumericEquivalence,
    PassContext,
    PassContract,
    SemanticEquivalence,
)
from planner.logical_plan import PlanNode


PassFunction = Callable[[PlanNode, PassContext], PlanNode]


@dataclass(frozen=True)
class FunctionCompilerPass:
    contract: PassContract
    function: PassFunction

    def run(self, plan: PlanNode, context: PassContext) -> PlanNode:
        return self.function(plan, context)


def build_optimizer_pass_manager(optimizer: object) -> CompilerPassManager:
    """Build the compatibility optimizer as an explicit, observable pass pipeline."""
    from planner.canonicalize_params import (
        canonicalize_plan_parameters,
        validate_plan_params,
    )
    from planner.composite_lowering import lower_composite_operators
    from planner.rewrite_fastpath import rewrite_plan_for_fastpath

    exact_contract = dict(
        input_ir=IRKind.LOGICAL_QUERY_GRAPH,
        output_ir=IRKind.LOGICAL_QUERY_GRAPH,
        semantic_equivalence=SemanticEquivalence.VALUE_PRESERVING,
        numeric_equivalence=NumericEquivalence.EXACT_VALUE,
        invariants=DEFAULT_REWRITE_INVARIANTS,
    )

    def fold(plan: PlanNode, context: PassContext) -> PlanNode:
        del context
        return optimizer._fold_literals(plan)  # type: ignore[attr-defined]

    def validate(plan: PlanNode, context: PassContext) -> PlanNode:
        validate_plan_params(plan, production=context.production)
        return plan

    def lower(plan: PlanNode, context: PassContext) -> PlanNode:
        del context
        return lower_composite_operators(plan)

    def rewrite(plan: PlanNode, context: PassContext) -> PlanNode:
        return rewrite_plan_for_fastpath(
            plan,
            allow_semantic_rewrites=bool(
                getattr(optimizer, "allow_semantic_rewrites", False)
                and not context.production
            ),
        )

    def canonicalize(plan: PlanNode, context: PassContext) -> PlanNode:
        del context
        return canonicalize_plan_parameters(plan)

    passes = (
        FunctionCompilerPass(
            PassContract(name="literal_fold", description="Fold finite literal arithmetic", **exact_contract),
            fold,
        ),
        FunctionCompilerPass(
            PassContract(name="parameter_validation", description="Apply planning-time parameter authority", **exact_contract),
            validate,
        ),
        FunctionCompilerPass(
            PassContract(name="composite_lowering", description="Expand certified composite operators", **exact_contract),
            lower,
        ),
        FunctionCompilerPass(
            PassContract(name="post_lowering_literal_fold", description="Fold literals exposed by lowering", **exact_contract),
            fold,
        ),
        FunctionCompilerPass(
            PassContract(
                name="fastpath_rewrite",
                input_ir=IRKind.LOGICAL_QUERY_GRAPH,
                output_ir=IRKind.LOGICAL_QUERY_GRAPH,
                semantic_equivalence=SemanticEquivalence.VALUE_PRESERVING,
                numeric_equivalence=NumericEquivalence.IEEE_EQUIVALENT,
                invariants=DEFAULT_REWRITE_INVARIANTS,
                description="Fuse certified fast-path graph patterns",
            ),
            rewrite,
        ),
        FunctionCompilerPass(
            PassContract(
                name="parameter_canonicalization",
                input_ir=IRKind.LOGICAL_QUERY_GRAPH,
                output_ir=IRKind.OPTIMIZED_LOGICAL_GRAPH,
                semantic_equivalence=SemanticEquivalence.IDENTITY,
                numeric_equivalence=NumericEquivalence.EXACT_VALUE,
                invariants=DEFAULT_REWRITE_INVARIANTS,
                description="Canonicalize operator and parameter spelling only",
            ),
            canonicalize,
        ),
    )
    return CompilerPassManager(passes)
