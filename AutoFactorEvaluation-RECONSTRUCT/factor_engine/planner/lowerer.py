"""IR → 逻辑计划：结构一一对应，仅把 ``tuple`` 子节点转为 ``list``。"""

from ir.nodes import IRNode

from .logical_plan import PlanNode


class Lowerer:
    """无优化的纯结构拷贝下降器。"""

    def to_logical_plan(self, ir: IRNode) -> PlanNode:
        def visit(node: IRNode) -> PlanNode:
            children = [visit(child) for child in node.inputs]
            return PlanNode(op=node.op, inputs=children, attrs=node.attrs.copy())

        return visit(ir)
