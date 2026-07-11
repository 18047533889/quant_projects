"""多因子公共子表达式消除（CSE）：将重复子树替换为 ``plan_ref``，定义存入 ``DAGPlan.shared_nodes``。"""

from __future__ import annotations

from planner.logical_plan import PlanNode
from planner.plan_hash import structural_key


def deep_copy_plan(node: PlanNode) -> PlanNode:
    return PlanNode(
        op=node.op,
        inputs=[deep_copy_plan(c) for c in node.inputs],
        attrs=dict(node.attrs),
        node_id=node.node_id,
    )


def _postorder(root: PlanNode) -> list[PlanNode]:
    out: list[PlanNode] = []

    def visit(n: PlanNode) -> None:
        for c in n.inputs:
            visit(c)
        out.append(n)

    visit(root)
    return out


def apply_cse(roots: list[PlanNode]) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """对多棵根计划做 CSE。

    出现次数大于 1 的结构键对应子树放入 ``shared_nodes``，各根中该子树一律替换为
    ``op="plan_ref"``、``attrs={"sid": <结构键>}``。

    结构键为 :func:`planner.plan_hash.structural_key` 的 JSON 串，可能较长但唯一稳定。
    """
    if not roots:
        return [], {}

    counts: dict[str, int] = {}  # 结构键 → 在森林中出现过几次
    first_seen: dict[str, PlanNode] = {}  # 首次出现时保留一份用于 shared_nodes

    for root in roots:
        memo: dict[int, str] = {}
        for n in _postorder(root):
            k = structural_key(n, memo)
            counts[k] = counts.get(k, 0) + 1
            if k not in first_seen:
                first_seen[k] = n

    shared_nodes: dict[str, PlanNode] = {}
    for k, c in counts.items():
        if c > 1:
            shared_nodes[k] = deep_copy_plan(first_seen[k])

    def rewrite(n: PlanNode, memo: dict[int, str]) -> PlanNode:
        k = structural_key(n, memo)
        if counts.get(k, 0) > 1:
            return PlanNode(op="plan_ref", attrs={"sid": k}, inputs=[])
        return PlanNode(
            op=n.op,
            attrs=dict(n.attrs),
            inputs=[rewrite(c, memo) for c in n.inputs],
            node_id=n.node_id,
        )

    new_roots = [rewrite(r, {}) for r in roots]
    return new_roots, shared_nodes
