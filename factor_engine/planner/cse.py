"""多因子公共子表达式消除（CSE）：将重复子树替换为 ``plan_ref``，定义存入 ``DAGPlan.shared_nodes``。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from planner.logical_plan import PlanNode
from planner.plan_hash import structural_key


def deep_copy_plan(node: PlanNode) -> PlanNode:
    """深拷贝整棵 ``PlanNode`` 子树。

    参数：
        node: 待拷贝的根节点

    返回：
        与输入同形、互不共享子节点引用的新树
    """
    return PlanNode(
        op=node.op,
        inputs=[deep_copy_plan(c) for c in node.inputs],
        attrs=dict(node.attrs),
        semantic_attrs=dict(node.semantic_attrs),
        node_id=node.node_id,
    )


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


def cse_consumer_counts(roots: list[PlanNode]) -> dict[str, int]:
    """统计每个共享子树 sid 在森林中的消费次数（Phase 5 R5）。

    运行时按引用计数回收：某 root 执行完即对其消费的 sid 减一，归零立即 evict，
    避免共享大 panel 常驻到整个 batch 结束。

    R6-150: 每个 sid 在**同一 root 内只计一次**。释放侧
    (``runtime.batch_service._release_root_cse``) 用 :func:`collect_consumed_sids`
    在单 root 内去重后一次性减一；若这里按出现次数计数（``factor=add(X,X)`` 时
    X 的 refcount=2），则 root 完成后只减 1，剩余 1 永不 evict。按 root 消费者去重
    后两侧语义一致：refcount == 消费该 sid 的 root 数。
    """
    counts: dict[str, int] = {}

    def walk(n: PlanNode, seen: set[str]) -> None:
        if getattr(n, "op", None) == "plan_ref":
            sid = str((n.attrs or {}).get("sid") or "")
            if sid and sid not in seen:
                seen.add(sid)
                counts[sid] = counts.get(sid, 0) + 1
            return
        for child in getattr(n, "inputs", ()) or ():
            walk(child, seen)

    for root in roots:
        walk(root, set())
    return counts


def collect_consumed_sids(root: PlanNode) -> list[str]:
    """返回单棵根计划消费的全部共享 sid（去重）。"""
    seen: set[str] = set()
    out: list[str] = []

    def walk(n: PlanNode) -> None:
        if getattr(n, "op", None) == "plan_ref":
            sid = str((n.attrs or {}).get("sid") or "")
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
            return
        for child in getattr(n, "inputs", ()) or ():
            walk(child)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# R20-044..046: scoped-universe classifier 不能 fail-open
# ---------------------------------------------------------------------------

#: 与 ``runtime.engine._CROSS_SECTIONAL_OPS`` 一致；加 ``cs_`` / ``group_`` 前缀
#: 与 registry category 判定。本模块是 fail-closed 权威实现。
_CROSS_SECTIONAL_OPS = frozenset(
    {
        "rank",
        "rank_pct",
        "zscore",
        "scale",
        "normalize",
        "winsorize",
        "quantile",
        "neutralize",
        "size_neutralize",
        "industry_size_neutralize",
        "industry_neutralize",
        "industry_rank",
    }
)


class PlanScopeCategory(str, Enum):
    """R20-044..046: 计划 scope 三态判定。

    - ``CROSS_SECTIONAL``：计划明确含截面算子（rank/zscore/neutralize/group/
      CS/kNN）；
    - ``NOT_CROSS_SECTIONAL``：计划明确不含；
    - ``UNKNOWN_SCOPE``：registry lookup 失败 / 未知算子 —— production 下必须
      当作 unresolved scope **hard fail**，绝不静默回落成「不是横截面」
      （否则 scoped-universe 因子会在全源上静默算截面）。
    """

    CROSS_SECTIONAL = "cross_sectional"
    NOT_CROSS_SECTIONAL = "not_cross_sectional"
    UNKNOWN_SCOPE = "unknown_scope"


def _node_scope_category(op: str) -> PlanScopeCategory:
    if not op:
        return PlanScopeCategory.NOT_CROSS_SECTIONAL
    if op.startswith("cs_") or op.startswith("group_"):
        return PlanScopeCategory.CROSS_SECTIONAL
    if op in _CROSS_SECTIONAL_OPS:
        return PlanScopeCategory.CROSS_SECTIONAL
    try:
        from cleaned_operators.registry import OperatorRegistry

        canonical = OperatorRegistry.resolve_canonical(op)
        if (
            canonical in _CROSS_SECTIONAL_OPS
            or canonical.startswith("cs_")
            or canonical.startswith("group_")
        ):
            return PlanScopeCategory.CROSS_SECTIONAL
        impl = OperatorRegistry.get(canonical)
        category = str(getattr(getattr(impl, "metadata", None), "category", "") or "")
        if category in {"cross_sectional", "group_neutralization"}:
            return PlanScopeCategory.CROSS_SECTIONAL
        return PlanScopeCategory.NOT_CROSS_SECTIONAL
    except Exception:
        # R20-044..046: registry lookup 异常 → UNKNOWN_SCOPE（不是「不是截面」）。
        return PlanScopeCategory.UNKNOWN_SCOPE


class UnresolvedScopeError(ValueError):
    """production 下计划含 UNKNOWN_SCOPE 节点（R20-044..046 fail-closed）。"""


def classify_plan_scope(plan: PlanNode) -> PlanScopeCategory:
    """R20-044..046: 全计划 scope 三态判定。

    任一节点 UNKNOWN_SCOPE → 整体 UNKNOWN_SCOPE（fail-closed：一个未知算子
    就足以让整个 plan 的 scope 判定不可信）。
    """
    if plan is None:
        return PlanScopeCategory.NOT_CROSS_SECTIONAL
    own = _node_scope_category(getattr(plan, "op", ""))
    if own is PlanScopeCategory.UNKNOWN_SCOPE:
        return PlanScopeCategory.UNKNOWN_SCOPE
    if own is PlanScopeCategory.CROSS_SECTIONAL:
        return PlanScopeCategory.CROSS_SECTIONAL
    for child in getattr(plan, "inputs", ()) or ():
        child_cat = classify_plan_scope(child)
        if child_cat is PlanScopeCategory.UNKNOWN_SCOPE:
            return PlanScopeCategory.UNKNOWN_SCOPE
        if child_cat is PlanScopeCategory.CROSS_SECTIONAL:
            return PlanScopeCategory.CROSS_SECTIONAL
    return PlanScopeCategory.NOT_CROSS_SECTIONAL


def assert_plan_scope_resolved(plan: PlanNode, *, production: bool = False) -> PlanScopeCategory:
    """R20-044..046: production 下 unresolved scope 必须是 ``UNKNOWN_SCOPE`` 硬失败。

    返回实际分类；production 且分类为 ``UNKNOWN_SCOPE`` 时抛
    ``UnresolvedScopeError``，不允许把「registry 查不到」静默当作「不是横截面」。
    """
    category = classify_plan_scope(plan)
    if production and category is PlanScopeCategory.UNKNOWN_SCOPE:
        raise UnresolvedScopeError(
            "production plan scope resolution is UNKNOWN_SCOPE: a registry "
            "lookup failed for an operator in the plan — refusing to treat the "
            "plan as non-cross-sectional (fail-closed)"
        )
    return category


# ---------------------------------------------------------------------------
# R20-058..061 / R20-062..066: CSE scope 绑定真实二级 SourceRef 依赖 +
# resolved universe membership
# ---------------------------------------------------------------------------


def cse_scope_key(
    scope: Any,
    *,
    source_dependency_hash: str = "",
    universe_membership_hash: str = "",
) -> str:
    """CSE 作用域键：``FactorExecutionScope`` + source_dependency_hash + universe membership。

    R20-058..061: ``_scope_from_factor`` 的 source scope 主要取 factor hint +
    anchor data source config，未纳入 ``source_dependency_hash(plan)``。两个
    factor 共享 anchor source 但二级 SourceRef 依赖不同时必须隔离 CSE，否则
    一次计算的结果会被错误复用到另一 factor。R20-062..066: 命名 universe 的
    实际 resolved membership 也必须进入 CSE scope（非空 instrument_filter 与
    相同 universe 标签不再能跨 membership 共享）。
    """
    from storage.data_scope import execution_scope_key as _exec_scope_key

    return _exec_scope_key(
        scope,
        source_dependency_hash=source_dependency_hash,
        universe_membership_hash=universe_membership_hash,
    )


def assert_cse_contracts_resolved(plan: PlanNode, *, production: bool = False) -> None:
    """R20-067..072: production CSE 前强制 ``ALL_OPERATOR_CONTRACTS_RESOLVED``。

    未注册算子语义契约 / field catalog hash 不可用的计划不得进入 production
    CSE（persistent cache 已部分避免 unresolved 写盘，但 CSE 的 shared_nodes
    会被消费，必须显式 fail-closed）。
    """
    if not production:
        return
    from planner.plan_hash import assert_plan_contracts_resolved

    assert_plan_contracts_resolved(plan, production=True)


def apply_cse(roots: list[PlanNode]) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """对多棵根计划做结构 CSE。

    出现次数大于 1 的结构键对应子树放入 ``shared_nodes``，各根中该子树一律替换为
    ``op="plan_ref"``、``attrs={"sid": <结构键>}``。

    结构键为 :func:`planner.plan_hash.structural_key` 的 JSON 串，可能较长但唯一稳定。

    参数：
        roots: 多因子逻辑计划根节点列表

    返回：
        ``(改写后的根列表, shared_nodes 字典)``，其中 shared_nodes 的键为结构 sid
    """
    if not roots:
        return [], {}

    counts: dict[str, int] = {}  # 结构键 → 在森林中出现过几次
    first_seen: dict[str, PlanNode] = {}  # 首次出现时保留一份用于 shared_nodes
    key_memo: dict[int, str] = {}

    for root in roots:
        for n in _postorder(root):
            k = structural_key(n, key_memo)
            counts[k] = counts.get(k, 0) + 1
            if k not in first_seen:
                first_seen[k] = n

    shared_nodes: dict[str, PlanNode] = {}
    for k, c in counts.items():
        if c > 1:
            shared_nodes[k] = deep_copy_plan(first_seen[k])

    def rewrite(n: PlanNode, memo: dict[int, str]) -> PlanNode:
        """将重复子树替换为 ``plan_ref`` 引用。"""
        k = structural_key(n, memo)
        if counts.get(k, 0) > 1:
            # R13 NEW-P0-21: a ``plan_ref`` must expose the shared subtree's
            # OUTPUT semantic contract (unit / grain / availability / price-basis
            # / source identity), not just the sid — downstream typing/PIT/SQL
            # layers read ``plan_ref.semantic_attrs`` as if they had the full
            # subtree.  The sid already binds semantics (structural_key embeds a
            # semantic digest), but the attrs must travel with the reference too.
            representative = first_seen[k]
            return PlanNode(
                op="plan_ref",
                attrs={"sid": k},
                inputs=[],
                semantic_attrs=dict(getattr(representative, "semantic_attrs", None) or {}),
            )
        return PlanNode(
            op=n.op,
            attrs=dict(n.attrs),
            semantic_attrs=dict(n.semantic_attrs),
            inputs=[rewrite(c, memo) for c in n.inputs],
            node_id=n.node_id,
        )

    new_roots = [rewrite(r, key_memo) for r in roots]
    return new_roots, shared_nodes
