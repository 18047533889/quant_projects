# -*- coding: utf-8 -*-
"""R27-215..223/233: Native multi-root fusion —— 一次 native query 算多个因子。

要点
    - R27-215/216：同 source scope + backend 下多个根，一次 native select/query
      而不是每根调用 backend.execute（共享 scan / sort / window）。
    - R27-220 NativeFusionGroup 约束：same backend / same source / same
      time+universe / same semantic scope。
    - R27-221/222：fusion group 大小受内存限制，自适应 32/64/128/256 roots。
    - R27-223：multi-root output 直接送 matrix writer（避免拆 100 个 Series 再拼）。
    - 诚实性（R25 原则）：backend 不支持 multi-root 时**不宣称** fusion——
      ``can_fuse`` 返回 False，调度器走 per-root（计数 ``native_fusion_fallback``）。

R39（§7）
    - PERF-021：``adaptive_fusion_block_size`` 从单标量比较换成可测量变量驱动
      的校准模型（planner/fusion_size_model.py）。
    - PERF-022：fusion 失败不再整组退 per-root —— binary-split 只拆失败子组；
      NegativeFusionCache 让已知失败组合后续不再付失败成本。
    - PERF-023：fusion grouping 加入 window / rank-scope / source-column /
      derived-primitive locality（次级排序键，不弱化 can_fuse 保守检查）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from planner.fusion_size_model import (
    FUSION_BLOCKS,
    MODEL_WEIGHTS,
    FusionSizeFeatures,
    choose_fusion_block_size,
)
from planner.negative_fusion_cache import (
    NEGATIVE_FUSION_CACHE,
    NegativeFusionCacheKey,
)
from planner.physical_factor_dag import PhysicalFactorTask

#: 自适应 fusion block（R27-222 / R39-PERF-021：bucket 保留，选择模型驱动）。
_FUSION_BLOCKS = FUSION_BLOCKS

#: 可视为 rolling/window 表达式的算子（用于统计 window_expression_count 与
#: window locality；启发式集合，缺失只是低估，不影响正确性）。
_WINDOW_OPS = frozenset({
    "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_argmax", "ts_argmin",
    "ts_delay", "ts_delta", "ts_rank", "ts_zscore", "ts_decay_linear",
    "rolling_mean", "rolling_std", "rolling_sum", "rolling_min", "rolling_max",
    "ema", "wma", "wilder_smoothing", "rolling_rank", "ts_corr", "rolling_corr",
    "ts_regression", "ts_slope", "ts_skew", "ts_kurt", "ts_median", "ts_quantile",
    "ts_entropy", "rolling_apply", "ts_autocorr",
})

#: rank / group / cross-section scope 算子（用于 rank_scope locality）。
_RANK_SCOPE_OPS = frozenset({
    "rank", "cs_rank", "cs_zscore", "cross_section", "group_rank", "group_zscore",
    "neutralize", "demean", "cs_demean", "rank_within", "group_neutralize",
})

#: 列引用算子（用于 source-column locality）。
_COLUMN_OPS = frozenset({"column", "field", "raw", "read_column", "load_column"})

#: locality 权重（PERF-023）：window/rank/derived 优先于 source columns ——
#: 同 window 不同 source 的 group 应比同 source 不同 window 的 group 得分更高
#: （共享 sort / rolling state / window frame 的收益 > 只共享 scan）。
_LOCALITY_WEIGHTS: dict[str, float] = {
    "window": 2.0,
    "rank_scope": 1.5,
    "derived": 1.5,
    "source_columns": 1.0,
}


@dataclass(frozen=True)
class NativeFusionGroup:
    """同 backend + source + semantic scope 的可融合根组（R27-220）。"""

    group_id: int
    backend: str
    source_scope: str
    roots: tuple[str, ...]
    estimated_output_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "backend": self.backend,
            "source_scope": self.source_scope,
            "roots": list(self.roots),
            "estimated_output_bytes": self.estimated_output_bytes,
        }


def native_fusion_capability_map(ctx: Any | None) -> dict[str, bool]:
    """R33-P0-044：真实 backend capability map。

    只有实际实现并认证 ``execute_multi_roots`` 的 backend 才创建 fusion group；
    否则直接不计划（不在 runtime 再 fallback）。判定基于：
      1. backend 对象确有 ``execute_multi_roots``（callable）；
      2. 该 capability 有 certification 证据（``backend_certification`` 中
         ``execute_multi_roots`` 已认证）。
    """
    out: dict[str, bool] = {
        "duckdb_sql": False,
        "polars": False,
        "polars_panel": False,
        "polars_long": False,
        "pandas_numpy": False,
    }
    backend = getattr(ctx, "backend", None)
    if backend is None:
        return out
    multi = getattr(backend, "execute_multi_roots", None)
    if not callable(multi):
        return out
    # certification 证据：能拿到 certified capability 才算数（不靠名字猜）。
    certified = False
    try:
        from backend.backend_certification import backend_certifies

        certified = bool(backend_certifies(backend, "execute_multi_roots"))
    except Exception:
        # 无 certification 层时退化为「有 callable 实现」——但 caller 显式传
        # capability map 时以 caller 为准。
        certified = True
    if certified:
        for key in out:
            out[key] = True
    return out


# ---------------------------------------------------------------------------
# R39-P0-PERF-021: 可测量变量 size model
# ---------------------------------------------------------------------------


def _walk(node: Any) -> Any:
    """深度优先遍历计划树（不递归回环，计划是 DAG/tree）。"""
    if node is None:
        return
    yield node
    for child in getattr(node, "inputs", None) or ():
        yield from _walk(child)


def _node_attrs(node: Any) -> dict[str, Any]:
    return dict(getattr(node, "attrs", None) or {})


def _is_window_node(node: Any) -> bool:
    op = getattr(node, "op", "")
    attrs = _node_attrs(node)
    return op in _WINDOW_OPS or "window" in attrs


def _is_column_node(node: Any) -> bool:
    op = getattr(node, "op", "")
    attrs = _node_attrs(node)
    return op in _COLUMN_OPS or "name" in attrs or "column" in attrs


def _column_name(node: Any) -> str:
    attrs = _node_attrs(node)
    return str(attrs.get("name") or attrs.get("column") or getattr(node, "op", "?"))


def _node_text_len(node: Any) -> int:
    """SQL 不可用时，用表达式文本长度估算 sql_chars。"""
    s = str(getattr(node, "op", "?"))
    for k, v in _node_attrs(node).items():
        s += str(k) + repr(v)
    return len(s)


def _node_sql_chars(node: Any) -> int:
    attrs = _node_attrs(node)
    sql = attrs.get("sql") or attrs.get("sql_text")
    if sql:
        return len(str(sql))
    return _node_text_len(node)


def _task_output_bytes(task: Any) -> int:
    rc = getattr(task, "resource_contract", None)
    if rc is None:
        return 0
    return int(getattr(rc, "output_bytes", 0) or 0)


def extract_fusion_size_features(
    roots: list[Any],
    *,
    task_by_id: dict[str, PhysicalFactorTask] | None = None,
) -> FusionSizeFeatures:
    """从 fused plan/tasks 提取可测量规模变量（R39-P0-PERF-021）。

    - 遍历每个 root 的 ``node_ref`` 计划树统计 AST 节点数 / window 表达式数 /
      projection（去重 source column）数；
    - ``sql_chars``：有 emitted SQL 用其长度，否则用表达式文本长度估算；
    - ``estimated_intermediate_bytes``：优先累计节点级中间字节估算，否则
      ``output_bytes * 2`` 启发式；
    - ``output_bytes``：来自 resource_contract。
    """
    nodes: list[Any] = []
    out_bytes = 0
    for r in roots:
        task = r
        if isinstance(r, str) and task_by_id is not None:
            task = task_by_id.get(r)
        if task is None:
            continue
        out_bytes += _task_output_bytes(task)
        node = getattr(task, "node_ref", None)
        if node is None:
            continue
        nodes.extend(_walk(node))

    if not nodes:
        return FusionSizeFeatures(
            sql_chars=0,
            ast_node_count=0,
            window_expression_count=0,
            projection_count=len([r for r in roots if not isinstance(r, str) or True]),
            estimated_intermediate_bytes=max(0, out_bytes) * 2,
            output_bytes=out_bytes,
        )

    ast_node_count = len(nodes)
    window_expression_count = sum(1 for n in nodes if _is_window_node(n))
    projection_count = len({_column_name(n) for n in nodes if _is_column_node(n)})
    sql_chars = sum(_node_sql_chars(n) for n in nodes)
    intermediate = 0
    for n in nodes:
        attrs = _node_attrs(n)
        eb = attrs.get("estimated_intermediate_bytes") or attrs.get("estimated_bytes")
        if eb:
            intermediate += int(eb)
    if intermediate <= 0:
        intermediate = max(0, out_bytes) * 2
    return FusionSizeFeatures(
        sql_chars=sql_chars,
        ast_node_count=ast_node_count,
        window_expression_count=window_expression_count,
        projection_count=projection_count,
        estimated_intermediate_bytes=intermediate,
        output_bytes=out_bytes,
    )


def _features_from_legacy_scalars(
    expression_complexity: float, estimated_output_bytes: int, root_count: int
) -> FusionSizeFeatures:
    """把旧的单标量合成可测量特征（仅当 caller 未传 ``features`` 时兼容用）。"""
    complexity = max(1.0, float(expression_complexity))
    n = max(1, int(root_count))
    per_root_nodes = complexity * 3.0
    out = max(0, int(estimated_output_bytes))
    return FusionSizeFeatures(
        sql_chars=int(per_root_nodes * n * 80),
        ast_node_count=int(per_root_nodes * n),
        window_expression_count=int(per_root_nodes * n / 5),
        projection_count=n,
        estimated_intermediate_bytes=out * 2,
        output_bytes=out,
    )


def adaptive_fusion_block_size(
    *,
    root_count: int,
    expression_complexity: float = 1.0,
    estimated_output_bytes: int = 0,
    backend_compile_budget_bytes: int = 2 * 1024**3,
    features: FusionSizeFeatures | None = None,
) -> int:
    """R27-222 + R39-P0-PERF-021：模型驱动的自适应 fusion block size。

    - 有 ``features``（真实可测量变量）→ ``estimate_fusion_cost`` 校准模型选
      最小化 ``compile_ms + execute_ms + output_materialization_ms`` 的
      32/64/128/256 bucket；
    - 无 ``features``（旧 caller 只传 scalar）→ 由旧 scalar 合成特征后走同一
      模型（``backend_compile_budget_bytes`` 收缩编译成本权重 → 更小 block）；
    - 签名/返回契约保留：返回 32..256 的 int。
    """
    if features is None:
        features = _features_from_legacy_scalars(
            expression_complexity, estimated_output_bytes, root_count
        )
    # 预算越小 → 编译成本权重越大 → 倾向更小 block（维度一致地进入模型）。
    budget_scale = max(0.25, min(4.0, (2 * 1024**3) / max(1, backend_compile_budget_bytes)))
    weights: dict[str, float] | None = None
    if budget_scale != 1.0:
        weights = {
            k: v * budget_scale
            for k, v in MODEL_WEIGHTS.items()
            if k.startswith("compile_")
        }
    return choose_fusion_block_size(
        root_count=root_count, features=features, weights=weights
    )


def can_fuse_roots(
    roots: list[PhysicalFactorTask],
    *,
    require_same_backend: bool = True,
    require_same_source_scope: bool = True,
    require_same_execution_scope: bool = True,
) -> bool:
    """R27-220 约束：same backend / same source / same semantic scope。

    R31-P0-019：额外要求 **same execution_scope**——``execution_scope`` 是
    ``FactorExecutionScope.scope_key()`` 的 JSON（含 market / universe / calendar /
    decision_time_policy / frequency），不同执行作用域的根绝不能融合（结果语义
    不同）。任一不满足 → False（诚实：不宣称 fusion）。

    R39-PERF-023 只**加强/保持**这些检查，locality 只做组内排序的次级键，
    绝不放宽本函数（NEVER 融合语义不同的根）。
    """
    if len(roots) < 2:
        return False
    backends = {r.preferred_backend for r in roots}
    scopes = {r.source_scope for r in roots}
    snaps = {r.source_snapshot_id for r in roots}
    exec_scopes = {getattr(r, "execution_scope", "") for r in roots}
    if require_same_backend and len(backends) > 1:
        return False
    if require_same_source_scope and len(scopes) > 1:
        return False
    if require_same_execution_scope and len(exec_scopes) > 1:
        return False
    if len(snaps) > 1:
        return False
    return True


# ---------------------------------------------------------------------------
# R39-P1-PERF-023: kernel/window locality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootLocality:
    """一个 root 的 locality 摘要（从计划树提取）。"""

    window_signature: frozenset = frozenset()  # {(op, window), ...}
    source_columns: frozenset = frozenset()  # {column_name, ...}
    rank_scope: str = ""  # canonical group-by/rank scope key
    derived_primitives: frozenset = frozenset()  # {(op, window)/("cse", sid), ...}


def _extract_root_locality(root: Any) -> RootLocality:
    """提取一个 root 的 window / source-column / rank-scope / derived-primitive。"""
    node = getattr(root, "node_ref", None)
    windows: set[tuple[str, Any]] = set()
    columns: set[str] = set()
    rank_scope = ""
    derived: set[tuple[str, Any]] = set()
    for n in _walk(node):
        op = getattr(n, "op", "")
        attrs = _node_attrs(n)
        if _is_window_node(n):
            w = attrs.get("window", 0)
            try:
                w = int(w)
            except (TypeError, ValueError):
                w = 0
            windows.add((op, w))
            derived.add((op, w))
        if _is_column_node(n):
            columns.add(_column_name(n))
        if op in _RANK_SCOPE_OPS:
            rank_scope = repr(
                attrs.get("by") or attrs.get("group") or attrs.get("scope") or ""
            )
        cse_sid = attrs.get("cse_sid") or attrs.get("structural_key")
        if cse_sid is not None:
            derived.add(("cse", str(cse_sid)))
    return RootLocality(
        window_signature=frozenset(windows),
        source_columns=frozenset(columns),
        rank_scope=rank_scope,
        derived_primitives=frozenset(derived),
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def locality_score(
    roots: list[Any],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """PERF-023：roots 之间的 locality 得分（次级排序键，非硬过滤）。

    取两两共享率 × 权重：window 共享权重最高（共享 sort / rolling state /
    window frame 收益最大），其次 rank-scope / derived-primitive，source-column
    权重最低（只共享 scan）。返回 [0, sum(weights)]。
    """
    w = {**_LOCALITY_WEIGHTS, **(weights or {})}
    locs = [_extract_root_locality(r) for r in roots]
    n = len(locs)
    if n < 2:
        return 0.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def frac(pred: Any) -> float:
        hit = sum(1 for i, j in pairs if pred(locs[i], locs[j]))
        return hit / len(pairs)

    window_f = frac(lambda a, b: bool(a.window_signature & b.window_signature))
    rank_f = frac(lambda a, b: bool(a.rank_scope and a.rank_scope == b.rank_scope))
    derived_f = frac(lambda a, b: bool(a.derived_primitives & b.derived_primitives))
    source_f = frac(lambda a, b: _jaccard(a.source_columns, b.source_columns))
    return (
        w["window"] * window_f
        + w["rank_scope"] * rank_f
        + w["derived"] * derived_f
        + w["source_columns"] * source_f
    )


def _output_bytes_key(task: Any) -> tuple[int, str]:
    rc = getattr(task, "resource_contract", None)
    out = int(getattr(rc, "output_bytes", 0) or 0) if rc is not None else 0
    return (out, getattr(task, "task_id", ""))


def _locality_ordered_tasks(tasks: list[PhysicalFactorTask]) -> list[PhysicalFactorTask]:
    """组内按 locality 排序（PERF-023）。

    主分组键 (backend, source_scope, execution_scope) 不变；本函数只对组内
    成员重排：贪心链式选择「与上一个 root locality 最高」的下一个，让同
    window / 同 rank-scope / 同 derived primitive / 同 source column 的根相邻，
    以便 chunk 时优先聚合。确定性：等分平局按 output_bytes + task_id 决胜。
    """
    if len(tasks) <= 2:
        return sorted(tasks, key=_output_bytes_key, reverse=True)
    remaining = list(tasks)
    current = max(remaining, key=_output_bytes_key)
    remaining.remove(current)
    ordered = [current]
    while remaining:
        best = max(
            remaining,
            key=lambda t: (
                locality_score([current, t]),
                _output_bytes_key(t)[0],
                _output_bytes_key(t)[1],
            ),
        )
        remaining.remove(best)
        ordered.append(best)
        current = best
    return ordered


def plan_native_fusion_groups(
    roots: list[PhysicalFactorTask],
    *,
    fusion_block: int | None = None,
    backend_capability: dict[str, bool] | None = None,
) -> list[NativeFusionGroup]:
    """把 roots 聚成 NativeFusionGroup（R27-220/221 + R39-PERF-023）。

    - 先按 (backend, source_scope, execution_scope) 分组；
    - ``fusion_block`` 缺省走 :func:`adaptive_fusion_block_size`（R39-PERF-021
      模型驱动）——按 root 数 + 真实可测量特征选 32/64/128/256；
    - 组内按 :func:`_locality_ordered_tasks` 重排（window/rank/derived/source
      locality 作为次级排序键），每个 fusion group 不超过 ``fusion_block`` 个 root；
    - ``backend_capability[backend]`` 为 False 或缺失时**不**生成 fusion 组。
    """
    by_scope: dict[tuple[str, str, str], list[PhysicalFactorTask]] = {}
    for r in roots:
        key = (r.preferred_backend, r.source_scope, r.execution_scope)
        by_scope.setdefault(key, []).append(r)

    groups: list[NativeFusionGroup] = []
    gid = 0
    for (backend, source_scope, _exec_scope), tasks in sorted(by_scope.items()):
        # R33-P0-044：只有真实实现并认证 ``execute_multi_roots`` 的 backend 才计划
        # fusion；否则直接不计划（不 runtime 再 fallback）。
        if backend_capability is not None and not backend_capability.get(backend, False):
            continue
        # R33-P0-042：每 (backend, source_scope, execution_scope) 组独立 can_fuse
        #（一组不合格不影响其它组）；组内不足 2 root 不融合。
        if len(tasks) < 2:
            continue
        if not can_fuse_roots(tasks, require_same_backend=True,
                              require_same_source_scope=True,
                              require_same_execution_scope=True):
            continue
        # R33-P0-043：fusion_block **每 group 独立**计算（不沿用第一个 group 的）。
        total_out = sum(_task_output_bytes(t) for t in tasks)
        block = (
            fusion_block
            if fusion_block is not None
            else adaptive_fusion_block_size(
                root_count=len(tasks),
                expression_complexity=max(
                    1.0,
                    sum(
                        (t.estimated_cost or {}).get("total_work", 1.0) if isinstance(t.estimated_cost, dict) else 1.0
                        for t in tasks
                    )
                    / max(1, len(tasks)),
                ),
                estimated_output_bytes=total_out,
                features=extract_fusion_size_features(tasks),
            )
        )
        # R39-PERF-023：组内 locality 重排（次级键；不改变 can_fuse 结论）。
        tasks_sorted = _locality_ordered_tasks(tasks)
        for i in range(0, len(tasks_sorted), block):
            chunk = tasks_sorted[i : i + block]
            out_bytes = sum(_task_output_bytes(t) for t in chunk)
            groups.append(
                NativeFusionGroup(
                    group_id=gid,
                    backend=backend,
                    source_scope=source_scope,
                    roots=tuple(t.task_id for t in chunk),
                    estimated_output_bytes=out_bytes,
                )
            )
            gid += 1
    return groups


# ---------------------------------------------------------------------------
# R39-P0-PERF-022: binary-split recovery + negative cache
# ---------------------------------------------------------------------------


def _family_repr(node: Any) -> tuple:
    """只含算子结构的 family 表示（不含参数值）——负缓存 plan_family_hash。"""
    if node is None:
        return ("none",)
    op = getattr(node, "op", "?")
    children = tuple(_family_repr(c) for c in (getattr(node, "inputs", None) or ()))
    return (op, children)


def _param_repr(node: Any) -> tuple:
    """含参数值的表示（deterministic：attr 按 key 排序）——负缓存 parameter_shape。"""
    if node is None:
        return ("none",)
    op = getattr(node, "op", "?")
    attrs = _node_attrs(node)
    items = tuple(sorted((str(k), repr(v)) for k, v in attrs.items()))
    children = tuple(_param_repr(c) for c in (getattr(node, "inputs", None) or ()))
    return (op, items, children)


def _plan_family_hash(nodes: list[Any]) -> str:
    digest = hashlib.sha256(
        repr(tuple(_family_repr(n) for n in nodes)).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def _negative_key_for(
    backend: Any, tasks: list[PhysicalFactorTask]
) -> NegativeFusionCacheKey:
    """构建一个 fusion 子组的负缓存键（backend / version / family / param / source）。"""
    nodes = [getattr(t, "node_ref", None) for t in tasks]
    family_hash = _plan_family_hash(nodes)
    param_shape = tuple(repr(_param_repr(n)) for n in nodes)
    source_shape = tuple(
        (
            getattr(t, "source_scope", ""),
            getattr(t, "source_snapshot_id", ""),
            tuple(sorted(getattr(t, "required_columns", ()) or ())),
        )
        for t in tasks
    )
    backend_name = (
        getattr(backend, "name", None)
        or getattr(backend, "backend_name", None)
        or type(backend).__name__
    )
    backend_version = (
        getattr(backend, "version", None)
        or getattr(backend, "__version__", None)
        or "unknown"
    )
    return NegativeFusionCacheKey(
        backend=str(backend_name),
        backend_version=str(backend_version),
        plan_family_hash=family_hash,
        parameter_shape=param_shape,
        source_shape=source_shape,
    )


def _execute_fusion_recursive(
    roots: tuple[str, ...],
    *,
    backend: Any,
    task_by_id: dict[str, PhysicalFactorTask],
    ctx: Any,
    execute_root: Any,
    stats: dict[str, int],
) -> tuple[bool, dict[str, Any]]:
    """Binary-split fusion recovery（R39-P0-PERF-022）。

    - 尝试把 ``roots`` 一次 multi-root 执行；
    - 失败 → 标记 NegativeFusionCache（已知坏组合后续不再付失败成本）；
    - 非单根失败 → 二分拆成两半，**只对失败子组继续拆**（成功子组直接融合）；
    - 单根 multi 也失败 → per-root fallback。

    返回 ``(是否全部融合成功, {tid: result})``；``results`` 覆盖全部 roots
    （能融合的融合，不能的 per-root）。
    """
    tasks = [task_by_id[t] for t in roots]
    key = _negative_key_for(backend, tasks)
    if NEGATIVE_FUSION_CACHE.is_blocked(key):
        # 已知坏组合：跳过 multi-root，直接 per-root（不再付一次失败成本）。
        return False, {t: execute_root(task_by_id[t]) for t in roots}
    try:
        payload = backend.execute_multi_roots([t.node_ref for t in tasks], ctx)
        results = {t: payload[t] for t in roots if t in payload}
        if len(results) != len(roots):
            raise ValueError(
                f"execute_multi_roots returned incomplete payload "
                f"({len(results)}/{len(roots)})"
            )
        stats["native_fusion_executed"] = stats.get("native_fusion_executed", 0) + 1
        stats["roots_per_fusion"] = max(stats.get("roots_per_fusion", 0), len(roots))
        stats["native_fusion_roots_total"] = (
            stats.get("native_fusion_roots_total", 0) + len(roots)
        )
        return True, results
    except Exception:
        NEGATIVE_FUSION_CACHE.mark_blocked(key)
        if len(roots) <= 1:
            stats["native_fusion_fallback"] = stats.get("native_fusion_fallback", 0) + 1
            return False, {roots[0]: execute_root(task_by_id[roots[0]])}
        stats["native_fusion_binary_split"] = (
            stats.get("native_fusion_binary_split", 0) + 1
        )
        mid = len(roots) // 2
        _l_ok, l_res = _execute_fusion_recursive(
            roots[:mid], backend=backend, task_by_id=task_by_id, ctx=ctx,
            execute_root=execute_root, stats=stats,
        )
        _r_ok, r_res = _execute_fusion_recursive(
            roots[mid:], backend=backend, task_by_id=task_by_id, ctx=ctx,
            execute_root=execute_root, stats=stats,
        )
        results = {**l_res, **r_res}
        return (_l_ok and _r_ok), results


def execute_fusion_group(
    group: NativeFusionGroup,
    *,
    backend: Any,
    task_by_id: dict[str, PhysicalFactorTask],
    ctx: Any,
    execute_root: Any,
) -> dict[str, Any]:
    """执行一个 fusion group（R27-216/217/218 + R39-P0-PERF-022）。

    - backend 提供 ``execute_multi_roots`` → 一次 native select/query 产多个根
      输出；失败时 **binary-split**（只拆失败子组），NegativeFusionCache 跳过
      已知坏组合；仍失败的最小集 per-root 回退。
    - backend 不支持 multi-root → 诚实 per-root 回退并计数
      ``native_fusion_fallback``（R25 诚实原则：不宣称 fusion）。
    """
    results: dict[str, Any] = {}
    stats = dict(getattr(ctx, "runtime_stats", None) or {})
    multi = getattr(backend, "execute_multi_roots", None)
    if callable(multi):
        stats["native_fusion_planned"] = stats.get("native_fusion_planned", 0) + 1
        ctx.runtime_stats = stats  # type: ignore[attr-defined]
        _ok, results = _execute_fusion_recursive(
            tuple(group.roots),
            backend=backend,
            task_by_id=task_by_id,
            ctx=ctx,
            execute_root=execute_root,
            stats=stats,
        )
        ctx.runtime_stats = stats  # type: ignore[attr-defined]
        return results
    # backend 无 multi-root：诚实 per-root fallback。
    stats["native_fusion_planned"] = stats.get("native_fusion_planned", 0) + 1
    stats["native_fusion_fallback"] = stats.get("native_fusion_fallback", 0) + 1
    ctx.runtime_stats = stats  # type: ignore[attr-defined]
    for tid in group.roots:
        task = task_by_id[tid]
        results[tid] = execute_root(task)
    return results
