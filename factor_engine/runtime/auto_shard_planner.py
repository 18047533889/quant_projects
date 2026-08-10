# -*- coding: utf-8 -*-
"""R36 §87..97/§100/§120：AutoShardPlanner —— 合法 task 超 envelope 时自动分片。

R36 P0-020（§100）：scheduler no-progress 后不能只报 stuck——若 ready task 的
P99 peak > SafeEnvelope 且语义上**可合法 shard**，先 replan smaller，再失败。

Shardability 是**语义属性**（§88，TaskResourceContract.shardable / shard_dimension
已携带）。本模块定义每类算子的合法 shard 维度（§89..96）：
    - elementwise        → time 或 asset
    - ts rolling         → asset 优先；time 必须带 warmup overlap
    - cross-section      → **只能 time**（asset 分片每片 rank 会改变因子）
    - group              → time（group 完整性）
    - stateful recursive → asset 优先；time 仅允许有 checkpoint state contract
    - full history       → 禁止 time shard
    - PCA / cs-model     → time blocks（每个日期完整 universe + 历史 window overlap）
    - minute→daily       → session/day shard（天然）

:func:`plan_shards` 返回 :class:`ShardPlan`（维度 + 分片数 + 每片 P99 ≤ envelope）。
非法 shard 请求必须拒绝（§90/91/94/95）——gate R36_ZERO_ILLEGAL_SHARD。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: 语义维度（TaskResourceContract.shard_dimension 的取值）
SHARD_ASSET = "asset"
SHARD_TIME = "time"
SHARD_SESSION = "session"
SHARD_NONE = None

#: 算子族 → 合法维度顺序（第一个是首选）。
_SHARD_POLICY: dict[str, tuple[str | None, ...]] = {
    "elementwise": (SHARD_ASSET, SHARD_TIME),
    "literal": (SHARD_ASSET, SHARD_TIME),
    "ts_rolling": (SHARD_ASSET, SHARD_TIME),     # time 需 warmup overlap
    "ts_window": (SHARD_ASSET, SHARD_TIME),
    "cs": (SHARD_TIME,),                          # §91：只能 time
    "group": (SHARD_TIME,),                       # §92：group 完整性 → time
    "stateful": (SHARD_ASSET,),                   # §93：asset 优先；time 需 checkpoint
    "full_history": (SHARD_ASSET,),               # §94：禁 time
    "model_cs": (SHARD_TIME,),                    # §95：PCA/cs-model → time blocks
    "minute_daily": (SHARD_SESSION,),
}

#: 未知算子族：保守不可 shard（§88：语义未知时禁止任意切分）。
_UNKNOWN_POLICY: tuple[str | None, ...] = (SHARD_NONE,)


def _policy_for(task: Any) -> tuple[str | None, ...]:
    """§89..96：按算子族语义给出合法维度（不 consult contract——语义真相）。"""
    task_type = str(getattr(task, "task_type", "") or "")
    if task_type.endswith("SOURCE_SCAN"):
        return (SHARD_TIME, SHARD_ASSET)
    op = str(getattr(task, "op", "") or "")
    if not op:
        return _UNKNOWN_POLICY
    if op.startswith("cs_") or "cross_section" in op:
        return _SHARD_POLICY["cs"]
    if op.startswith("ts_") and any(k in op for k in ("rank", "pct", "quantile", "zscore", "std", "demean")):
        return _SHARD_POLICY["ts_rolling"]
    if "state" in op or "recursive" in op or op in {"ewm", "kalman", "garch"}:
        return _SHARD_POLICY["stateful"]
    if op.startswith("ts_") or op.startswith("rolling"):
        return _SHARD_POLICY["ts_rolling"]
    return _SHARD_POLICY["elementwise"]


def legal_shard_dimensions(task: Any) -> tuple[str | None, ...]:
    """§88/91/94：该 task 语义上合法的 shard 维度（空/None = 不可 shard）。

    contract 声明的 ``shard_dimension`` 只有在与算子语义类**一致**时才是权威；
    冲突（如 cs 算子声明 asset shard，§91 会改变因子）→ 禁止 shard，保证
    gate R36_ZERO_ILLEGAL_SHARD。
    """
    policy = _policy_for(task)
    contract = getattr(task, "resource_contract", None)
    if contract is not None:
        if not getattr(contract, "shardable", False):
            return _UNKNOWN_POLICY
        declared = getattr(contract, "shard_dimension", None)
        if declared is not None:
            if declared in policy:
                return (declared,)
            return _UNKNOWN_POLICY  # 声明维度与语义冲突 → 不可 shard
    return policy


@dataclass(frozen=True)
class ShardPlan:
    """一个合法 shard 计划（§97 AutoShardPlanner 输出）。"""

    task_id: str
    dimension: str            # asset / time / session
    shard_count: int
    per_shard_peak_bytes: int
    original_peak_bytes: int
    needs_warmup: bool = False   # time shard 的 rolling/stateful 需要 overlap
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dimension": self.dimension,
            "shard_count": self.shard_count,
            "per_shard_peak_bytes": self.per_shard_peak_bytes,
            "original_peak_bytes": self.original_peak_bytes,
            "needs_warmup": self.needs_warmup,
            "reason": self.reason,
        }


class AutoShardPlanner:
    """§97：输入 task 语义 + P99 memory + SafeEnvelope + backend + history requirement
    → 输出 :class:`ShardPlan`（每片 P99 ≤ SafeEnvelope）。"""

    def __init__(self, *, min_shards: int = 2, max_shards: int = 64) -> None:
        self._min_shards = max(2, int(min_shards))
        self._max_shards = max(self._min_shards, int(max_shards))

    def plan_for_task(
        self,
        task: Any,
        *,
        safe_envelope_bytes: int,
        history_requirement: str = "none",
    ) -> ShardPlan | None:
        """给单个 task 出 shard 计划；不可 shard / 已能放下返回 None。

        §87：合法 task P99 peak > SafeEnvelope 不能 forever reject——auto shard。
        """
        contract = getattr(task, "resource_contract", None)
        peak = int(getattr(contract, "peak_memory_bytes", 0) or 0) or 0
        if peak <= 0:
            # 无契约：用 admissible 估算。
            peak = int(getattr(contract, "admissible_peak_bytes", 0) or 0) or 0
        if peak <= safe_envelope_bytes:
            return None
        dims = legal_shard_dimensions(task)
        if not dims or dims[0] is None:
            return None
        dim = dims[0]
        # 非法组合硬拒（gate R36_ZERO_ILLEGAL_SHARD）。
        if history_requirement == "full" and dim == SHARD_TIME:
            return None
        if dim == SHARD_ASSET and _is_cross_section(task):
            return None
        # 分片数：每片 peak ≈ peak / n ≤ safe_envelope。
        n = max(self._min_shards, int(math.ceil(peak / max(1, safe_envelope_bytes))))
        n = min(self._max_shards, n)
        per_shard = max(0, int(math.ceil(peak / n)))
        if per_shard > safe_envelope_bytes:
            # 即使最小分片也放不下 → 不可行（诚实 None）。
            return None
        needs_warmup = dim == SHARD_TIME and (
            _is_rolling(task) or _is_stateful(task)
        )
        reason = (
            f"peak {peak} > safe {safe_envelope_bytes}; auto-shard {dim} x{n}"
            + (" (warmup overlap)" if needs_warmup else "")
        )
        return ShardPlan(
            task_id=str(getattr(task, "task_id", "") or ""),
            dimension=dim,
            shard_count=n,
            per_shard_peak_bytes=per_shard,
            original_peak_bytes=peak,
            needs_warmup=needs_warmup,
            reason=reason,
        )

    def plan_for_tasks(
        self,
        tasks: list[Any],
        *,
        safe_envelope_bytes: int,
        history_requirement: str = "none",
    ) -> dict[str, ShardPlan]:
        """批量：返回 {task_id: ShardPlan}（只包含需要 shard 的）。"""
        out: dict[str, ShardPlan] = {}
        for task in tasks:
            plan = self.plan_for_task(
                task,
                safe_envelope_bytes=safe_envelope_bytes,
                history_requirement=history_requirement,
            )
            if plan is not None:
                out[plan.task_id] = plan
        return out


def _is_cross_section(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return op.startswith("cs_") or "cross_section" in op


def _is_rolling(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return op.startswith("ts_") or op.startswith("rolling") or op.startswith("ewm")


def _is_stateful(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return any(k in op for k in ("state", "recursive", "kalman", "garch", "ewm"))
