"""调试后端：不访问真实数据，只把逻辑计划树打印成缩进文本，便于检查编译结果。"""

from factor_engine.planner.logical_plan import PlanNode

from .base import Backend
from .context import ExecutionContext


class DebugBackend(Backend):
    """将 ``PlanNode`` 递归格式化为字符串（每行 ``op`` + ``attrs``）。"""

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> str:
        """将逻辑计划格式化为缩进文本（不访问真实数据）。

        参数:
            plan: 逻辑计划根节点。
            ctx: 执行上下文（调试模式下未使用）。

        返回:
            可读的 plan 树字符串。
        """
        _ = ctx  # 调试模式不读数据源
        return self._render(plan)

    def _render(self, node: PlanNode, depth: int = 0) -> str:
        """深度优先打印子树。"""
        if node.op == "plan_ref":
            sid = node.attrs.get("sid", "")
            preview = (sid[:48] + "…") if len(sid) > 48 else sid
            return "  " * depth + f"plan_ref sid={preview!r}"
        lines = ["  " * depth + f"{node.op} {node.attrs}"]
        for child in node.inputs:
            lines.append(self._render(child, depth + 1))
        return "\n".join(lines)
