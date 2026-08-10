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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planner.physical_factor_dag import PhysicalFactorTask

#: 自适应 fusion block（R27-222）
_FUSION_BLOCKS = (32, 64, 128, 256)


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


def adaptive_fusion_block_size(
    *,
    root_count: int,
    expression_complexity: float = 1.0,
    estimated_output_bytes: int = 0,
    backend_compile_budget_bytes: int = 2 * 1024**3,
) -> int:
    """R27-222：自适应 fusion block size。

    表达式越复杂 / 输出越大 / backend 编译预算越小 → 用更小 block。
    返回 32/64/128/256 中最接近且不超 root_count 的值。
    """
    score = expression_complexity * max(1.0, estimated_output_bytes / (1024**3))
    budget = max(64 * 1024**2, backend_compile_budget_bytes / (256 * 1024**2))
    # complexity 高 → 小 block；预算小 → 小 block。
    block = 256
    for candidate in _FUSION_BLOCKS:
        if score * (256 / candidate) > budget or candidate > root_count:
            block = candidate
            break
    else:
        block = 256
    return max(32, min(root_count, block))


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


def plan_native_fusion_groups(
    roots: list[PhysicalFactorTask],
    *,
    fusion_block: int | None = None,
    backend_capability: dict[str, bool] | None = None,
) -> list[NativeFusionGroup]:
    """把 roots 聚成 NativeFusionGroup（R27-220/221）。

    - 先按 (backend, source_scope, execution_scope) 分组；
    - ``fusion_block`` 缺省走 :func:`adaptive_fusion_block_size`（R31-P0-021：
      自适应 block 是**实际默认**，不再固定 64）——按 root 数 + 输出字节 +
      表达式复杂度自动选 32/64/128/256；
    - 组内按 root 输出字节降序（先塞贵的），每个 fusion group 不超过
      ``fusion_block`` 个 root；
    - ``backend_capability[backend]`` 为 False 或缺失时**不**生成 fusion 组
      （该 backend 不支持 multi-root，R27-158 只对已有证据支持的 backend 做
      自动 routing）。
    """
    by_scope: dict[tuple[str, str, str], list[PhysicalFactorTask]] = {}
    for r in roots:
        key = (r.preferred_backend, r.source_scope, r.execution_scope)
        by_scope.setdefault(key, []).append(r)

    groups: list[NativeFusionGroup] = []
    gid = 0
    for (backend, source_scope, exec_scope), tasks in sorted(by_scope.items()):
        if backend_capability is not None and not backend_capability.get(backend, False):
            continue
        if fusion_block is None:
            total_out = sum(
                t.resource_contract.output_bytes if t.resource_contract else 0 for t in tasks
            )
            fusion_block = adaptive_fusion_block_size(
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
            )
        tasks_sorted = sorted(
            tasks,
            key=lambda t: (t.resource_contract.output_bytes if t.resource_contract else 0),
            reverse=True,
        )
        for i in range(0, len(tasks_sorted), fusion_block):
            chunk = tasks_sorted[i : i + fusion_block]
            out_bytes = sum(
                t.resource_contract.output_bytes if t.resource_contract else 0
                for t in chunk
            )
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


def execute_fusion_group(
    group: NativeFusionGroup,
    *,
    backend: Any,
    task_by_id: dict[str, PhysicalFactorTask],
    ctx: Any,
    execute_root: Any,
) -> dict[str, Any]:
    """执行一个 fusion group。

    若 backend 提供 ``execute_multi_roots``（一次性 native select/query 产多个
    根输出），调用它（R27-216/217/218）；否则诚实回退 per-root 并计数
    ``native_fusion_fallback``（R25 诚实原则：不宣称 fusion）。
    """
    results: dict[str, Any] = {}
    stats = dict(getattr(ctx, "runtime_stats", None) or {})
    multi = getattr(backend, "execute_multi_roots", None)
    if callable(multi):
        try:
            payload = multi(
                [task_by_id[tid].node_ref for tid in group.roots],
                ctx,
            )
            for tid in group.roots:
                if tid in payload:
                    results[tid] = payload[tid]
            if len(results) == len(group.roots):
                # R31-P0-022：真实 runtime event——fusion 真的执行了。
                stats["native_fusion_planned"] = stats.get("native_fusion_planned", 0) + 1
                stats["native_fusion_executed"] = stats.get("native_fusion_executed", 0) + 1
                stats["roots_per_fusion"] = max(
                    stats.get("roots_per_fusion", 0), len(group.roots)
                )
                stats["native_fusion_roots_total"] = (
                    stats.get("native_fusion_roots_total", 0) + len(group.roots)
                )
                ctx.runtime_stats = stats  # type: ignore[attr-defined]
                return results
        except Exception:
            # multi-root 执行失败 → 回退 per-root（结果仍要交付）。
            results = {}
    stats = dict(getattr(ctx, "runtime_stats", None) or {})
    stats["native_fusion_planned"] = stats.get("native_fusion_planned", 0) + 1
    stats["native_fusion_fallback"] = stats.get("native_fusion_fallback", 0) + 1
    ctx.runtime_stats = stats  # type: ignore[attr-defined]
    for tid in group.roots:
        task = task_by_id[tid]
        results[tid] = execute_root(task)
    return results
