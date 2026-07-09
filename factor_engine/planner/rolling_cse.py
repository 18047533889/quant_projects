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
)

ROLLING_OP_CANONICAL: dict[str, str] = {
    "ts_std_dev": "ts_std",
    "ts_correlation": "ts_corr",
    "ts_covariance": "ts_cov",
    "m_cor": "ts_corr",
    "m_cov": "ts_cov",
    "ts_delay": "delay",
}

_ROLLING_ATTR_IGNORE = frozenset({"min_periods", "min_count", "ddof"})


def canonical_rolling_op(op: str) -> str:
    return ROLLING_OP_CANONICAL.get(op, op)


def _second_col_ref(node: PlanNode) -> str | None:
    if len(node.inputs) < 2:
        return None
    return _first_col_ref(node.inputs[1])


def rolling_semantic_key(node: PlanNode) -> str | None:
    """``(canonical_op, col[, col2], window)`` 稳定语义键；非 rolling 返回 ``None``。"""
    if node.op not in ROLLING_OPS:
        return None
    op = canonical_rolling_op(node.op)
    col = _first_col_ref(node)
    if not col:
        return None
    window = _window_from_attrs(node.op, node.attrs)
    payload: dict[str, Any] = {"op": op, "col": col, "window": window}
    col2 = _second_col_ref(node)
    if col2 is not None:
        payload["col2"] = col2
    extra = {
        k: node.attrs[k]
        for k in sorted(node.attrs)
        if k not in _ROLLING_ATTR_IGNORE and k not in {"window", "d", "period", "n", "periods", "lag"}
    }
    if extra:
        payload["attrs"] = extra
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _postorder(root: PlanNode) -> list[PlanNode]:
    out: list[PlanNode] = []

    def visit(n: PlanNode) -> None:
        for c in n.inputs:
            visit(c)
        out.append(n)

    visit(root)
    return out


def apply_rolling_cse(
    roots: list[PlanNode],
    existing_shared: dict[str, PlanNode] | None = None,
) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """结构 CSE 之后按 rolling 语义键合并仍可共享的子树。"""
    if not roots:
        return [], dict(existing_shared or {})

    shared: dict[str, PlanNode] = dict(existing_shared or {})
    existing_sids = set(shared)
    sem_to_sid: dict[str, str] = {}
    sem_counts: dict[str, int] = {}
    sem_first_node: dict[str, PlanNode] = {}

    for root in roots:
        for n in _postorder(root):
            sem = rolling_semantic_key(n)
            if sem is None:
                continue
            sem_counts[sem] = sem_counts.get(sem, 0) + 1
            if sem not in sem_first_node:
                sem_first_node[sem] = n
            sid = structural_key(n, {})
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

    new_roots = [rewrite(r, {}) for r in roots]
    return new_roots, shared
