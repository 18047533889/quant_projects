"""Rolling 语义 CSE：按 (canonical_op, column, window) 跨因子共享滚动子树。"""

from __future__ import annotations

import json
from typing import Any

from planner.cse import deep_copy_plan
from planner.logical_plan import PlanNode
from planner.plan_hash import structural_key
from planner.rolling_cache import (
    ROLLING_OPS,
    _first_col_ref,
    _window_from_attrs,
    _window_from_literals,
)

ROLLING_OP_CANONICAL: dict[str, str] = {
    "ts_std_dev": "ts_std",
    "ts_correlation": "ts_corr",
    "ts_covariance": "ts_cov",
    "m_cor": "ts_corr",
    "m_cov": "ts_cov",
    "delay": "ts_delay",
    "ts_delay": "ts_delay",
}

# 默认原则：只有 OperatorContract 明确声明的 cse_irrelevant_params 才能忽略，
# 目前没有任何声明的参数可忽略 —— 因此 ddof/min_periods/min_count 全部视为
# active 参数，必须进入语义键（#319）。
_ROLLING_ATTR_IGNORE = frozenset()

# window 已被显式放进 payload 主字段，故从 extra 中排除以避免重复承载；
# 其余所有 attrs（含 d/period/n/periods/lag/min_periods/min_count/ddof）
# 一律进 extra，不允许丢失（#319）。
_WINDOW_EXTRA_EXCLUDE = frozenset({"window"})


def canonical_rolling_op(op: str) -> str:
    """将 rolling 算子名映射为 canonical 形式。

    参数：
        op: 原始算子名

    返回：
        canonical 算子名；无别名时返回原 op
    """
    return ROLLING_OP_CANONICAL.get(op, op)


def _merge_windows(attrs_window: int | None, literal_window: int | None) -> int | None:
    """合并 attrs 与 literal 两路 window 解析结果。

    literal child input 是最终执行值（#318），优先；仅 attrs 解析到值时回退
    attrs；两者都非 None 且不一致时以 literal 为准（双源互斥时宁可分离也不
    错误合并不同 window 的子树）。
    """
    if literal_window is not None:
        return literal_window
    return attrs_window


def _second_col_ref(node: PlanNode) -> str | None:
    """提取双输入 rolling 算子第二路输入的列名。"""
    if len(node.inputs) < 2:
        return None
    return _first_col_ref(node.inputs[1])


def rolling_semantic_key(node: PlanNode) -> str | None:
    """``(canonical_op, col[, col2], window)`` 稳定语义键；非 rolling 返回 ``None``。

    参数：
        node: 候选 rolling 计划节点

    返回：
        JSON 语义键字符串；非 rolling 或无法解析主列时返回 ``None``
    """
    if node.op not in ROLLING_OPS:
        return None
    op = canonical_rolling_op(node.op)
    col = _first_col_ref(node)
    if not col:
        return None
    window = _merge_windows(
        _window_from_attrs(node.op, node.attrs),
        _window_from_literals(node),
    )
    payload: dict[str, Any] = {"op": op, "col": col, "window": window}
    col2 = _second_col_ref(node)
    if col2 is not None:
        payload["col2"] = col2
    extra = {
        k: node.attrs[k]
        for k in sorted(node.attrs)
        if k not in _ROLLING_ATTR_IGNORE and k not in _WINDOW_EXTRA_EXCLUDE
    }
    if extra:
        payload["attrs"] = extra
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _postorder(root: PlanNode) -> list[PlanNode]:
    """后序遍历计划树，返回节点列表。"""
    out: list[PlanNode] = []

    def visit(n: PlanNode) -> None:
        """后序递归访问单节点。"""
        for c in n.inputs:
            visit(c)
        out.append(n)

    visit(root)
    return out


def apply_rolling_cse(
    roots: list[PlanNode],
    existing_shared: dict[str, PlanNode] | None = None,
) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """结构 CSE 之后按 rolling 语义键合并仍可共享的子树。

    参数：
        roots: 多因子逻辑计划根节点列表
        existing_shared: 结构 CSE 已产出的 shared_nodes（在此基础上扩展）

    返回：
        ``(改写后的根列表, 更新后的 shared_nodes)``；
        语义相同但结构不同的 rolling 子树会被统一为 ``plan_ref``
    """
    if not roots:
        return [], dict(existing_shared or {})

    shared: dict[str, PlanNode] = dict(existing_shared or {})
    existing_sids = set(shared)
    sem_to_sid: dict[str, str] = {}
    sem_counts: dict[str, int] = {}
    sem_first_node: dict[str, PlanNode] = {}

    key_memo: dict[int, str] = {}
    for root in roots:
        for n in _postorder(root):
            sem = rolling_semantic_key(n)
            if sem is None:
                continue
            sem_counts[sem] = sem_counts.get(sem, 0) + 1
            if sem not in sem_first_node:
                sem_first_node[sem] = n
            sid = structural_key(n, key_memo)
            if sid in existing_sids:
                sem_to_sid.setdefault(sem, sid)

    for sem, count in sem_counts.items():
        if count <= 1:
            continue
        sid = sem_to_sid.get(sem)
        if sid is None:
            node = sem_first_node[sem]
            sid = structural_key(node, {})
            shared[sid] = deep_copy_plan(node)
            sem_to_sid[sem] = sid

    if not sem_to_sid:
        return roots, shared

    def rewrite(n: PlanNode, memo: dict[int, str]) -> PlanNode:
        """按 rolling 语义键将等价子树统一为 ``plan_ref``。"""
        sem = rolling_semantic_key(n)
        if sem and sem in sem_to_sid:
            sid = sem_to_sid[sem]
            if structural_key(n, memo) != sid:
                return PlanNode(op="plan_ref", attrs={"sid": sid}, inputs=[])
        return PlanNode(
            op=n.op,
            attrs=dict(n.attrs),
            inputs=[rewrite(c, memo) for c in n.inputs],
            node_id=n.node_id,
        )

    new_roots = [rewrite(r, key_memo) for r in roots]
    return new_roots, shared
