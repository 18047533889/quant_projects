"""逻辑计划轻量优化：常量折叠与 fastpath 友好改写。"""

from __future__ import annotations

from .logical_plan import PlanNode


class Optimizer:
    """轻量计划优化：常量折叠与可选的语义保持 fastpath 改写。"""

    def __init__(self, *, allow_semantic_rewrites: bool = False) -> None:
        """创建优化器；默认不改变 divide/group_mean 的边界语义。"""
        self.allow_semantic_rewrites = bool(allow_semantic_rewrites)

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        """对逻辑计划执行优化流水线。

        参数：
            plan: 未经优化的逻辑计划根节点

        返回：
            常量折叠并完成 fastpath 改写后的计划根节点
        """
        folded = self._fold_literals(plan)
        from planner.composite_lowering import lower_composite_operators
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath

        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        return rewrite_plan_for_fastpath(
            refolded,
            allow_semantic_rewrites=(self.allow_semantic_rewrites and not production),
        )

    def lower_only(self, plan: PlanNode) -> PlanNode:
        """仅 composite lowering + 常量折叠（不含 fastpath rewrite）。"""
        from planner.composite_lowering import lower_composite_operators

        folded = self._fold_literals(plan)
        lowered = lower_composite_operators(folded)
        return self._fold_literals(lowered)

    def optimize_with_trace(
        self, plan: PlanNode, *, production: bool = False
    ) -> tuple[PlanNode, PlanNode, tuple[tuple[str, tuple[str, ...]], ...]]:
        """返回 ``(optimized_plan, pre_lowering_plan, lowering_trace)``。"""
        from planner.composite_lowering import build_lowering_trace, lower_composite_operators
        from planner.rewrite_fastpath import rewrite_plan_for_fastpath

        folded = self._fold_literals(plan)
        lowered = lower_composite_operators(folded)
        refolded = self._fold_literals(lowered)
        final = rewrite_plan_for_fastpath(
            refolded,
            allow_semantic_rewrites=(self.allow_semantic_rewrites and not production),
        )
        trace = build_lowering_trace(folded, refolded)
        return final, folded, trace

    def _fold_literals(self, node: PlanNode) -> PlanNode:
        """自底向上折叠二元/多元算术字面量子表达式。"""
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
