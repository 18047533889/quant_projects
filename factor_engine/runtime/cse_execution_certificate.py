# -*- coding: utf-8 -*-
"""R42-015: CSE Execution Certificate — 编译期预生成共享子树执行元数据。

本模块为 CSE shared nodes 提供编译期构造的执行证书，避免运行期重复计算成本。

核心类型
    - :class:`CSEExecutionCertificate`: 单个 shared node 的不可变执行元数据
    - :class:`CSECertificateStore`: 批量 CSE 证书存储器

R42-015 要求
    compile/plan 阶段预生成证书：
    - size_estimate_bytes: 物化结果预估大小
    - recompute_cost_ms: 重算成本（供 LRU/spill 决策）
    - consumer_count: 消费者数量
    - lifetime_interval: (first_use_ordinal, last_use_ordinal)
    运行时 O(1) 读取，零成本重复计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CSEExecutionCertificate:
    """单个 CSE shared node 的执行证书（R42-015 预计算元数据）。

    Attributes:
        sid: Shared node ID (必须与 DAG shared_nodes key 一致)
        size_estimate_bytes: 物化结果预估大小（影响 admission/eviction）
        recompute_cost_ms: 重算成本（LRU 逐出决策）
        consumer_count: 消费该 shared node 的 root 数量
        lifetime_interval: (first_use_ordinal, last_use_ordinal) 消费顺序区间
        reuse_distance: Next-use distance 列表（R42-016 Belady-like eviction）
        plan_digest: Subplan semantic digest（验证运行时 node 一致性）
    """

    sid: str
    size_estimate_bytes: int
    recompute_cost_ms: float
    consumer_count: int
    lifetime_interval: tuple[int, int] = (0, 0)
    reuse_distance: tuple[int, ...] = ()
    plan_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "size_estimate_bytes": self.size_estimate_bytes,
            "recompute_cost_ms": self.recompute_cost_ms,
            "consumer_count": self.consumer_count,
            "lifetime_interval": list(self.lifetime_interval),
            "reuse_distance": list(self.reuse_distance),
            "plan_digest": self.plan_digest,
        }


@dataclass
class CSECertificateStore:
    """批量 CSE 证书存储器（R42-015 编译期预生成，运行时只读）。

    Attributes:
        certificates: sid -> CSEExecutionCertificate 映射
        total_shared_memory_bytes: 所有 shared nodes 内存预估总和
        critical_sids: 高 reuse / 高成本 shared nodes（优先保留）
    """

    certificates: dict[str, CSEExecutionCertificate] = field(default_factory=dict)
    total_shared_memory_bytes: int = 0
    critical_sids: frozenset[str] = field(default_factory=frozenset)

    def get(self, sid: str) -> CSEExecutionCertificate | None:
        """O(1) 查询 shared node 证书（运行时零成本）。"""
        return self.certificates.get(sid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificates": {k: v.to_dict() for k, v in self.certificates.items()},
            "total_shared_memory_bytes": self.total_shared_memory_bytes,
            "critical_sids": sorted(self.critical_sids),
        }


def build_cse_certificate_store(
    shared_nodes: dict[str, Any],
    consumer_map: dict[str, list[str]],
    ordinal_map: dict[str, int],
    *,
    estimate_cost_fn: Any = None,
) -> CSECertificateStore:
    """R42-015: 从 shared_nodes 和消费者依赖构造 CSE 证书存储器。

    Args:
        shared_nodes: CSE shared node dict (sid -> subplan)
        consumer_map: sid -> list[consumer_task_id] 消费者映射
        ordinal_map: task_id -> execution_ordinal 拓扑序映射
        estimate_cost_fn: Optional cost estimator (默认导入 backend.operator_cost)

    Returns:
        预生成的不可变证书存储器
    """
    if estimate_cost_fn is None:
        try:
            from factor_engine.backend.operator_cost import estimate_plan_cost

            estimate_cost_fn = estimate_plan_cost
        except ImportError:
            estimate_cost_fn = lambda plan: {"total_work": 0.0, "peak_live_memory_bytes": 0}

    certificates = {}
    total_memory = 0
    critical_sids = set()

    for sid, subplan in shared_nodes.items():
        consumers = consumer_map.get(sid, [])
        consumer_count = len(consumers)

        # 计算 cost 和 size
        try:
            cost_result = estimate_cost_fn(subplan)
            recompute_ms = float(cost_result.get("total_work", 0.0) or 0.0)
            size_bytes = int(cost_result.get("peak_live_memory_bytes", 0) or 0)
        except Exception:
            recompute_ms = 0.0
            size_bytes = 0

        # 计算 lifetime interval 和 reuse distance
        ordinals = sorted(ordinal_map.get(c, 0) for c in consumers)
        first_use = ordinals[0] if ordinals else 0
        last_use = ordinals[-1] if ordinals else 0

        # R42-016: reuse distance = 每次 use 到下次 use 的 ordinal 距离
        reuse_dist = tuple(ordinals[i + 1] - ordinals[i] for i in range(len(ordinals) - 1))

        # 生成 plan digest
        plan_digest = ""
        try:
            import hashlib
            plan_str = str(subplan)
            plan_digest = hashlib.sha256(plan_str.encode("utf-8")).hexdigest()[:16]
        except Exception:
            pass

        cert = CSEExecutionCertificate(
            sid=sid,
            size_estimate_bytes=size_bytes,
            recompute_cost_ms=recompute_ms,
            consumer_count=consumer_count,
            lifetime_interval=(first_use, last_use),
            reuse_distance=reuse_dist,
            plan_digest=plan_digest,
        )
        certificates[sid] = cert
        total_memory += size_bytes

        # Critical sids: consumer_count >= 3 或 recompute_cost 高
        if consumer_count >= 3 or recompute_ms > 500.0:
            critical_sids.add(sid)

    return CSECertificateStore(
        certificates=certificates,
        total_shared_memory_bytes=total_memory,
        critical_sids=frozenset(critical_sids),
    )
