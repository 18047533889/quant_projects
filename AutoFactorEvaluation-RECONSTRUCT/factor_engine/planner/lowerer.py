"""IR → 逻辑计划：结构一一对应，仅把 ``tuple`` 子节点转为 ``list``。"""

from ir.nodes import IRNode

from .logical_plan import PlanNode


class Lowerer:
    """无优化的纯结构拷贝下降器。"""

    def to_logical_plan(self, ir: IRNode) -> PlanNode:
        """将 IR 树根节点递归下降为 ``PlanNode`` 逻辑计划。

        参数：
            ir: ``Analyzer.lower`` 产出的 IR 根节点

        返回：
            与 IR 同形的 ``PlanNode`` 树（``inputs`` 为 list）
        """

        def visit(node: IRNode) -> PlanNode:
            """递归访问 IR 子树并构造对应 PlanNode。"""
            children = [visit(child) for child in node.inputs]
            return PlanNode(op=node.op, inputs=children, attrs=node.attrs.copy())

        return visit(ir)
