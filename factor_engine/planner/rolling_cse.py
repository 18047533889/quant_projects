"""Rolling 语义 CSE：按 (canonical_op, 每个 panel 输入结构语义键, 参数) 跨因子共享滚动子树。

R20-001..004: ``plan_ref`` 继承被共享子树的 output semantic contract；所有
planner rewrite 统一走 ``rewrite_node`` 保留 ``semantic_attrs`` / ``node_id`` /
provenance；新增 ``verify_plan_ref_semantics`` 校验 sid 指向节点的 output
semantic digest；release invariant ``SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0``。

R20-005..008: ``rolling_semantic_key`` 不再取第一个列名作为共享键 —— 每个 panel
输入子树的完整 ``structural_key`` 进入键，``ts_mean(close,20)`` 与
``ts_mean(log(close),20)`` / ``ts_mean(close/ts_mean(close,5),20)`` 不再误共享。
"""

from __future__ import annotations

import json
from typing import Any

from planner.cse import deep_copy_plan
from planner.logical_plan import PlanNode
from planner.plan_hash import structural_key
from planner.rewrite_fastpath import rewrite_node
from planner.rolling_cache import (
    _ensure_rolling_ops_refreshed,
    _resolve_canonical,
    _window_from_attrs,
    _window_from_literals,
    is_rolling_operator,
)

# R20-010..012: 仅作为 registry 尚未加载时的 bootstrap 兜底。registry 加载后，
# ``canonical_rolling_op`` 优先走 ``OperatorRegistry.resolve_canonical`` 的 alias
# 解析（_aliases.py 的权威映射），本表不参与生产判定。
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

# window 已被显式放进 payload 的 params.window，故从 extra 中排除以避免重复承载；
# 其余所有 attrs（含 d/period/n/periods/lag/min_periods/min_count/ddof）
# 一律进 params，不允许丢失（#319）。
_WINDOW_EXTRA_EXCLUDE = frozenset({"window"})

# R20-004: release invariant —— rolling CSE 不得丢失任何语义属性。等于 0 意味着
# ``apply_rolling_cse`` 前后每个 root 的 output semantic digest 完全一致。
SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE = 0


def canonical_rolling_op(op: str) -> str:
    """将 rolling 算子名映射为 canonical 形式。

    R20-010..012: 优先使用 registry 的 canonical alias 解析（``_aliases.py`` 的
    权威映射）；registry 未加载时退回 ``ROLLING_OP_CANONICAL`` bootstrap 表。

    参数：
        op: 原始算子名

    返回：
        canonical 算子名；无别名时返回原 op
    """
    resolved = _resolve_canonical(op)
    if resolved != str(op):
        return resolved
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


def _safe_scalar(v: Any) -> Any:
    """把 literal 值归一为 JSON 稳定标量（NaN/Inf 转确定字符串）。"""
    if isinstance(v, float) and not _isfinite(v):
        return f"nonfinite:{v}"
    return v


def _isfinite(v: float) -> bool:
    try:
        import math

        return math.isfinite(v)
    except (TypeError, ValueError):  # pragma: no cover
        return True


def _output_semantic_digest(node: PlanNode) -> str:
    """节点 output semantic_attrs 的稳定摘要；为空时返回 ``""``。

    与 ``plan_hash._semantic_digest`` 同构：排序后 JSON 序列化再 SHA-256，保证
    相同语义属性得到相同摘要、不同语义属性得到不同摘要。
    """
    if not node.semantic_attrs:
        return ""

    def _norm(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (set, frozenset)):
            return sorted((_norm(x) for x in v), key=repr)
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _norm(val) for k, val in sorted(v.items())}
        return repr(v)

    import hashlib

    payload = sorted((k, _norm(v)) for k, v in node.semantic_attrs.items())
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rolling_semantic_key(node: PlanNode, memo: dict[int, str] | None = None) -> str | None:
    """稳定语义键；非 rolling 返回 ``None``。

    R20-005..008: 键包含 ``canonical``、每个 panel 输入子树的完整
    ``structural_key``（不再只看第一个列名）、``params``（window + 全部 active
    attrs + 全部 literal 标量）以及 ``output_semantic_digest``。这与 structural
    CSE 一样精确 —— 只有 key 完全一致的两个子树才可能被 rolling CSE 共享。

    参数：
        node: 候选 rolling 计划节点
        memo: 可选的 ``structural_key`` memo（避免重复遍历共享子树）

    返回：
        JSON 语义键字符串；非 rolling 时返回 ``None``
    """
    if not is_rolling_operator(node.op):
        return None
    op = canonical_rolling_op(node.op)
    window = _merge_windows(
        _window_from_attrs(node.op, node.attrs),
        _window_from_literals(node),
    )
    # panel 输入 = 非 literal 子节点（scalar 参数以 literal child 或 attrs 传入）。
    panel_inputs = [
        c for c in node.inputs if getattr(c, "op", None) != "literal"
    ]
    scalar_literals = [
        _safe_scalar((c.attrs or {}).get("value"))
        for c in node.inputs
        if getattr(c, "op", None) == "literal" and (c.attrs or {}).get("value") is not None
    ]
    payload: dict[str, Any] = {
        "canonical": op,
        "input_semantic_keys": [structural_key(c, memo) for c in panel_inputs],
    }
    params: dict[str, Any] = {}
    if window is not None:
        params["window"] = window
    if scalar_literals:
        params["scalar_params"] = scalar_literals
    for k in sorted(node.attrs):
        if k in _ROLLING_ATTR_IGNORE or k in _WINDOW_EXTRA_EXCLUDE:
            continue
        params[k] = _safe_scalar(node.attrs[k])
    if params:
        payload["params"] = params
    semantic = _output_semantic_digest(node)
    if semantic:
        payload["output_semantic_digest"] = semantic
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


def verify_plan_ref_semantics(
    plan_ref: PlanNode,
    shared_nodes: dict[str, PlanNode],
) -> bool:
    """R20-003: 校验 ``plan_ref`` 的 sid 指向节点与引用声明语义一致。

    校验三条不变量：
    1. ``plan_ref`` 的 ``sid`` 在 ``shared_nodes`` 中可解析；
    2. ``sid`` 正是被引用共享子树的 ``structural_key``（sid 绑定语义）；
    3. ``plan_ref`` 声明的 output semantic digest 与被引用子树根节点一致。

    参数：
        plan_ref: 待校验的 ``plan_ref`` 节点
        shared_nodes: 结构 / rolling CSE 产出的 ``{sid: 共享子树}``

    返回：
        全部不变量满足时为 ``True``；任一不满足返回 ``False``
    """
    if plan_ref.op != "plan_ref":
        return True
    sid = str((plan_ref.attrs or {}).get("sid") or "")
    if not sid:
        return False
    shared = shared_nodes.get(sid)
    if shared is None:
        return False
    if structural_key(shared) != sid:
        return False
    if _output_semantic_digest(plan_ref) != _output_semantic_digest(shared):
        return False
    return True


def verify_plan_refs_semantics(
    roots: list[PlanNode],
    shared_nodes: dict[str, PlanNode],
) -> bool:
    """R20-003: 遍历整棵根计划，校验其中全部 ``plan_ref``。"""
    for root in roots:
        for n in _postorder(root):
            if n.op == "plan_ref" and not verify_plan_ref_semantics(n, shared_nodes):
                return False
    return True


def rolling_cse_output_semantic_digests(roots: list[PlanNode]) -> list[str]:
    """每个 root 的 output semantic digest（R20-004 不变量用）。

    参数：
        roots: 根计划列表

    返回：
        与 ``roots`` 对齐的 digest 列表（无 semantic_attrs 时为 ``""``）
    """
    return [_output_semantic_digest(r) for r in roots]


def assert_rolling_cse_semantics_preserved(
    before_roots: list[PlanNode],
    after_roots: list[PlanNode],
) -> None:
    """R20-004: 断言 rolling CSE 不改变任何 root 的 output semantic digest。

    比较前后每个 root 的 ``_output_semantic_digest``；任何不一致都会触发
    ``AssertionError``，配合 ``SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0`` 构成
    release invariant。

    参数：
        before_roots: rolling CSE 之前的根计划列表
        after_roots: rolling CSE 之后的根计划列表
    """
    assert SEMANTIC_ATTRS_LOST_BY_ROLLING_CSE == 0, (
        "rolling CSE must not drop semantic_attrs (release invariant)"
    )
    before = rolling_cse_output_semantic_digests(before_roots)
    after = rolling_cse_output_semantic_digests(after_roots)
    assert before == after, (
        "rolling CSE changed a root's output semantic digest: "
        f"{before!r} -> {after!r}"
    )


def apply_rolling_cse(
    roots: list[PlanNode],
    existing_shared: dict[str, PlanNode] | None = None,
) -> tuple[list[PlanNode], dict[str, PlanNode]]:
    """结构 CSE 之后按 rolling 语义键合并仍可共享的子树。

    R20-005..008: 语义键包含每个 panel 输入的完整 ``structural_key``，因此
    ``ts_mean(log(close),20)`` 与 ``ts_mean(close,20)`` 不会被误共享；只有语义
    完全等价的 rolling 子树才被统一为 ``plan_ref``。

    参数：
        roots: 多因子逻辑计划根节点列表
        existing_shared: 结构 CSE 已产出的 shared_nodes（在此基础上扩展）

    返回：
        ``(改写后的根列表, 更新后的 shared_nodes)``；
        语义相同但结构不同的 rolling 子树会被统一为 ``plan_ref``
    """
    if not roots:
        return [], dict(existing_shared or {})

    # R20-013: 确保派生集合已刷新（registry 已加载时覆盖 bootstrap 基集）。
    # 使用内存化的 ``_ensure_*`` 而非每次全量 ``refresh_rolling_ops``，避免
    # 每个 batch 都重扫整个 registry。
    _ensure_rolling_ops_refreshed()

    shared: dict[str, PlanNode] = dict(existing_shared or {})
    existing_sids = set(shared)
    sem_to_sid: dict[str, str] = {}
    sem_counts: dict[str, int] = {}
    sem_first_node: dict[str, PlanNode] = {}

    key_memo: dict[int, str] = {}
    for root in roots:
        for n in _postorder(root):
            sem = rolling_semantic_key(n, key_memo)
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
        sem = rolling_semantic_key(n, memo)
        if sem and sem in sem_to_sid:
            sid = sem_to_sid[sem]
            if structural_key(n, memo) != sid:
                # R20-001: ``plan_ref`` 继承被共享子树的 output semantic contract
                # （unit / grain / availability / price-basis / source identity），
                # 与 ``planner.cse.apply_cse`` 的处理一致。
                representative = shared.get(sid)
                return PlanNode(
                    op="plan_ref",
                    attrs={"sid": sid},
                    inputs=[],
                    semantic_attrs=dict(
                        getattr(representative, "semantic_attrs", None) or {}
                    ),
                )
        # R20-002: 统一走 ``rewrite_node`` —— 默认保留 semantic_attrs、node_id
        # 与 provenance，不再裸 ``PlanNode(...)`` 重建丢语义。
        return rewrite_node(
            n,
            inputs=[rewrite(c, memo) for c in n.inputs],
        )

    new_roots = [rewrite(r, key_memo) for r in roots]
    return new_roots, shared
