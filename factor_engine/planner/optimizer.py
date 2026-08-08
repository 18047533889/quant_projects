"""逻辑计划轻量优化：常量折叠、composite lowering 与 fastpath 改写。"""

from __future__ import annotations

import math

from .logical_plan import PlanNode


class Optimizer:
    """轻量计划优化；最终输出保证 canonical op/parameter spelling。"""

    def __init__(self, *, allow_semantic_rewrites: bool = False) -> None:
        self.allow_semantic_rewrites = bool(allow_semantic_rewrites)

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        folded = self._fold_literals(plan)
        from planner.composite_lowering import lower_composite_operators
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath
        from planner.canonicalize_params import (
            canonicalize_plan_parameters,
            validate_plan_params,
        )

        # R6 P0-04: parameters must be validated BEFORE composite lowering so a
        # lowering can never truncate a 5.9 or accept a NaN/Inf that runtime
        # validation rejects.  Lowering only reads already-canonicalized values.
        validate_plan_params(folded, production=production)
        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        rewritten = rewrite_plan_for_fastpath(
            refolded,
            allow_semantic_rewrites=(self.allow_semantic_rewrites and not production),
        )
        return canonicalize_plan_parameters(rewritten)

    def lower_only(self, plan: PlanNode) -> PlanNode:
        from planner.composite_lowering import lower_composite_operators
        from planner.canonicalize_params import (
            canonicalize_plan_parameters,
            validate_plan_params,
        )

        folded = self._fold_literals(plan)
        validate_plan_params(folded)
        lowered = lower_composite_operators(folded)
        return canonicalize_plan_parameters(self._fold_literals(lowered))

    def optimize_with_trace(
        self, plan: PlanNode, *, production: bool = False
    ) -> tuple[PlanNode, PlanNode, tuple[tuple[str, tuple[str, ...]], ...]]:
        from planner.composite_lowering import build_lowering_trace, lower_composite_operators
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath
        from planner.canonicalize_params import (
            canonicalize_plan_parameters,
            validate_plan_params,
        )

        folded = self._fold_literals(plan)
        validate_plan_params(folded, production=production)
        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        final = rewrite_plan_for_fastpath(
            refolded,
            allow_semantic_rewrites=(self.allow_semantic_rewrites and not production),
        )
        final = canonicalize_plan_parameters(final)
        trace = build_lowering_trace(folded, refolded)
        return final, folded, trace

    def _fold_literals(self, node: PlanNode) -> PlanNode:
        inputs = [self._fold_literals(c) for c in node.inputs]
        n = PlanNode(
            op=node.op,
            inputs=inputs,
            attrs=dict(node.attrs),
            node_id=node.node_id,
        )
        if n.op in ("add", "subtract", "sub", "multiply", "mul", "divide", "div") and len(inputs) == 2:
            a, b = inputs
            if a.op == "literal" and b.op == "literal":
                va, vb = float(a.attrs["value"]), float(b.attrs["value"])
                if n.op == "add":
                    out = va + vb
                elif n.op in ("subtract", "sub"):
                    out = va - vb
                elif n.op in ("multiply", "mul"):
                    out = va * vb
                else:
                    if vb == 0.0:
                        return n
                    out = va / vb
                # Audit #389 finite gate: 1e308 * 1e308 overflows to +inf during
                # constant folding.  Folding would manufacture an ``inf`` literal
                # that the runtime's numeric semantics may treat differently from
                # an evaluated multiply (e.g. protected_div / NaN handling).  When
                # the folded result is a non-finite float, DON'T fold — leave the
                # operator node in place and let the runtime apply its own
                # arithmetic semantics to the operands.
                if isinstance(out, float) and not math.isfinite(out):
                    return n
                return PlanNode(op="literal", attrs={"value": out}, inputs=[])
        if n.op == "nary_add" and inputs and all(c.op == "literal" for c in inputs):
            out = sum(float(c.attrs["value"]) for c in inputs)
            # Audit #389: same finite gate for nary sums (e.g. 1e308 + 1e308).
            if isinstance(out, float) and not math.isfinite(out):
                return n
            return PlanNode(
                op="literal",
                attrs={"value": out},
                inputs=[],
            )
        if n.op == "nary_mul" and inputs and all(c.op == "literal" for c in inputs):
            prod = 1.0
            for c in inputs:
                prod *= float(c.attrs["value"])
            # Audit #389: same finite gate for nary products.
            if isinstance(prod, float) and not math.isfinite(prod):
                return n
            return PlanNode(op="literal", attrs={"value": prod}, inputs=[])
        return n
