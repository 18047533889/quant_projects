"""多因子公共子表达式消除（CSE）：将重复子树替换为 ``plan_ref``，定义存入 ``DAGPlan.shared_nodes``。"""

from __future__ import annotations

from enum import Enum
from typing import Any

import logging

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.plan_hash import structural_key

logger = logging.getLogger(__name__)


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


#: R32-P1-047: CSE / plan 深度上限（DSL 解析器已限 32；Direct Python API 构造
#: 的深 plan 在此兜底）。超限抛错，绝不让递归栈溢出。
MAX_PLAN_DEPTH = 512


class PlanDepthLimitError(ValueError):
    """计划深度超过 ``MAX_PLAN_DEPTH``（R32-P1-047 防 RecursionError）。"""


def assert_plan_depth_bounded(root: PlanNode, *, max_depth: int = MAX_PLAN_DEPTH) -> None:
    """R32-P1-047: 迭代遍历检查计划深度（不递归，栈安全）。

    深表达式在 CSE/planner 的递归遍历前先 fail-closed —— 超深抛
    ``PlanDepthLimitError``，绝不 RecursionError 崩溃。
    """
    stack: list[tuple[PlanNode, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > int(max_depth):
            raise PlanDepthLimitError(
                f"plan depth {depth} exceeds max_depth={max_depth}; refusing to "
                "recurse (deep expression protection, R32-P1-047)"
            )
        for child in getattr(node, "inputs", ()) or ():
            stack.append((child, depth + 1))


def _postorder(root: PlanNode) -> list[PlanNode]:
    """后序遍历计划树，返回节点列表（R32-P1-047: 迭代实现，栈安全）。"""
    out: list[PlanNode] = []
    stack: list[tuple[PlanNode, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            out.append(node)
            continue
        stack.append((node, True))
        for child in getattr(node, "inputs", ()) or ():
            stack.append((child, False))
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
        from factor_engine.cleaned_operators.registry import OperatorRegistry

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
    from factor_engine.storage.data_scope import execution_scope_key as _exec_scope_key

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
    from factor_engine.planner.plan_hash import assert_plan_contracts_resolved

    assert_plan_contracts_resolved(plan, production=True)


# ---------------------------------------------------------------------------
# R32-P0-008: cost-based CSE —— 出现两次不等于值得 materialize
# ---------------------------------------------------------------------------

#: leaf column 的"重算"成本 = 一次数据扫描（最高）；literal 成本 0（永不提取）。
_COLUMN_RECOMPUTE_COST = 4.0
_LITERAL_RECOMPUTE_COST = 0.0
#: materialize 一份共享 panel 的固定成本。
_MATERIALIZE_COST = 1.0


def _recompute_cost(node: PlanNode) -> float:
    """估算子树重算成本：数据扫描（column）最贵，其余按节点聚合。"""
    op = str(getattr(node, "op", "") or "")
    if op == "column":
        return _COLUMN_RECOMPUTE_COST
    if op in {"literal", "plan_ref", "materialized_series"}:
        return _LITERAL_RECOMPUTE_COST
    cost = 1.0
    for c in getattr(node, "inputs", ()) or ():
        cost += _recompute_cost(c)
    return cost


def _memory_cost(node: PlanNode) -> float:
    """估算共享 panel 内存成本 ≈ 子树节点数（至少 1）。"""
    size = 0
    stack = [node]
    while stack:
        n = stack.pop()
        size += 1
        stack.extend(getattr(n, "inputs", ()) or ())
    return float(max(1, size))


def cse_benefit(count: int, recompute: float, memory: float) -> float:
    """R32-P0-008: ``benefit = recompute_saved - materialize_cost - memory_cost``。

    ``recompute_saved = (count-1) * recompute``（原需 count 次，提取后 1 次）。
    literal（recompute=0）或列数少而重复次数低时 benefit<=0 → 不提取。
    """
    return (int(count) - 1) * float(recompute) - _MATERIALIZE_COST - float(memory)


def apply_cse(roots: list[PlanNode]) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """对多棵根计划做 cost-based 结构 CSE（R32-P0-007/008）。

    - 只提取 **profitable** 重复子树（``cse_benefit > 0``）——literal/列数少而
      重复次数低的 cheap 节点不再一视同仁提取（P0-008）；
    - shared definition 自己 rewrite 成 **nested shared DAG**（P0-007 方案 B）：
      父 subtree 内部重复的 child 也被替换为 ``plan_ref``，杜绝 orphan shared
      node（runtime 完整性校验拒绝无人引用的 shared）；
    - 结构键为 :func:`planner.plan_hash.structural_key` 的 JSON 串，唯一稳定。

    参数：
        roots: 多因子逻辑计划根节点列表

    返回：
        ``(改写后的根列表, shared_nodes 字典)``，其中 shared_nodes 的键为结构 sid
    """
    if not roots:
        return [], {}

    # R32-P1-047: 深 plan 在递归遍历前 fail-closed。
    for root in roots:
        assert_plan_depth_bounded(root)

    counts: dict[str, int] = {}  # 结构键 → 在森林中出现过几次
    first_seen: dict[str, PlanNode] = {}  # 首次出现时保留一份用于 shared_nodes
    key_memo: dict[int, str] = {}
    costs: dict[str, float] = {}

    for root in roots:
        for n in _postorder(root):
            k = structural_key(n, key_memo)
            counts[k] = counts.get(k, 0) + 1
            if k not in first_seen:
                first_seen[k] = n
            if k not in costs:
                costs[k] = _recompute_cost(n)

    # R32-P0-008: 只提取 profitable 子树。
    shared_keys = {
        k
        for k, c in counts.items()
        if c > 1 and cse_benefit(c, costs[k], _memory_cost(first_seen[k])) > 0
    }

    shared_nodes: dict[str, PlanNode] = {}

    def _shared_def_rewrite(n: PlanNode, memo: dict[int, str]) -> PlanNode:
        """改写一个 shared definition 的内部重复 child 为 ``plan_ref``。

        shared definition 的**根**保持真实节点（它就是被共享的那棵子树）；其
        内部重复 child 若也在 ``shared_keys`` 中则替换为 ``plan_ref`` —— 形成
        nested shared DAG（P0-007 方案 B），内部 child 因此有消费者，不再 orphan。
        """
        new_inputs: list[PlanNode] = []
        for c in n.inputs:
            ck = structural_key(c, memo)
            if ck in shared_keys:
                representative = first_seen[ck]
                new_inputs.append(
                    PlanNode(
                        op="plan_ref",
                        attrs={"sid": ck},
                        inputs=[],
                        semantic_attrs=dict(
                            getattr(representative, "semantic_attrs", None) or {}
                        ),
                    )
                )
            else:
                new_inputs.append(_shared_def_rewrite(c, memo))
        if not n.inputs:
            return n
        return PlanNode(
            op=n.op,
            attrs=dict(n.attrs),
            semantic_attrs=dict(n.semantic_attrs),
            inputs=new_inputs,
            node_id=n.node_id,
        )

    # 构建 nested shared DAG：shared definition 根保持真实节点，内部重叠 child
    # 改写为 plan_ref。
    for k in shared_keys:
        shared_nodes[k] = _shared_def_rewrite(
            first_seen[k], dict(key_memo)
        )

    def rewrite(n: PlanNode, memo: dict[int, str]) -> PlanNode:
        """将重复子树替换为 ``plan_ref`` 引用（根侧）。"""
        k = structural_key(n, memo)
        if k in shared_keys:
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
    # P0#7: emit CSE telemetry so reuse is never silent. ``len(shared_keys)`` is
    # the profitable shared-node count, ``counts`` are structural candidates,
    # and the reuse edge total is the sum of consumers across shared definitions.
    # ``nodes_released`` (execution-time refcount lifecycle) is counted separately
    # where actual release happens (see ``_release_consumed_sids``) — plan-time
    # apply_cse cannot know how many shared panels a run later evicts.
    reuse_edges = max(0, sum(counts.get(k, 0) for k in shared_keys) - len(shared_keys))
    try:
        from factor_engine.telemetry.execution_telemetry import cse_record

        cse_record(
            {
                "candidates": len(counts),
                "shared_nodes": len(shared_keys),
                "reuse_edges": reuse_edges,
            }
        )
    except Exception:  # pragma: no cover - telemetry must never break execution
        logger.debug("CSE telemetry emit failed (non-fatal)", exc_info=True)
    return new_roots, shared_nodes


def verify_cse_dag(
    roots: list[PlanNode], shared_nodes: dict[str, PlanNode]
) -> list[str]:
    """R32-P0-007: 校验 CSE 产出是合法 shared DAG。

    返回违规列表（空 = 合法）：
      - dangling ref：plan_ref 的 sid 在 shared_nodes 中不存在；
      - orphan shared：某 shared node 没有任何 plan_ref 消费者（既不在根中、
        也不被另一 shared node 引用）；
      - cycle：shared DAG 内循环引用。
    """
    violations: list[str] = []

    def _collect_plan_refs(root: PlanNode) -> set[str]:
        out: set[str] = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if str(getattr(n, "op", "")) == "plan_ref":
                out.add(str((n.attrs or {}).get("sid") or ""))
            stack.extend(getattr(n, "inputs", ()) or ())
        return out

    # dangling refs in roots + consumers map.
    consumers: dict[str, int] = {k: 0 for k in shared_nodes}
    for root in roots:
        for sid in _collect_plan_refs(root):
            if sid not in shared_nodes:
                violations.append(f"dangling plan_ref sid={sid!r} in root")
            elif sid in consumers:
                consumers[sid] += 1
    # shared definition 内部的 plan_ref 消费者（nested DAG）。
    for sid, node in shared_nodes.items():
        for inner_sid in _collect_plan_refs(node):
            if inner_sid not in shared_nodes:
                violations.append(
                    f"dangling plan_ref sid={inner_sid!r} inside shared def {sid!r}"
                )
            elif inner_sid == sid:
                violations.append(f"self-cycle in shared def {sid!r}")
            else:
                consumers[inner_sid] += 1
    for sid, count in consumers.items():
        if count == 0:
            violations.append(f"orphan shared node sid={sid!r}: zero consumers")
    return violations
