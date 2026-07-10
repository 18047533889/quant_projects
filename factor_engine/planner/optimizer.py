from __future__ import annotations

from .logical_plan import PlanNode


class Optimizer:
    """轻量计划优化：自底向上常量折叠等。"""

    def optimize(self, plan: PlanNode) -> PlanNode:
        folded = self._fold_literals(plan)
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath

        return rewrite_plan_for_fastpath(folded)

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
                if n.op in ("add",):
                    out = va + vb
                elif n.op in ("subtract", "sub"):
                    out = va - vb
                elif n.op in ("multiply", "mul"):
                    out = va * vb
                else:
                    if vb == 0.0:
                        return n
                    out = va / vb
                return PlanNode(op="literal", attrs={"value": out}, inputs=[])
        if n.op == "nary_add" and inputs and all(c.op == "literal" for c in inputs):
            total = sum(float(c.attrs["value"]) for c in inputs)
            return PlanNode(op="literal", attrs={"value": total}, inputs=[])
        if n.op == "nary_mul" and inputs and all(c.op == "literal" for c in inputs):
            prod = 1.0
            for c in inputs:
                prod *= float(c.attrs["value"])
            return PlanNode(op="literal", attrs={"value": prod}, inputs=[])
        return n
