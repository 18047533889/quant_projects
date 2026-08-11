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


def build_time_blocks(
    time_range: tuple[Any, Any],
    shard_count: int,
) -> list[tuple[Any, Any]]:
    """把时间窗切成 ``shard_count`` 个连续块（按交易日分布）。"""
    import pandas as pd

    start, end = time_range
    try:
        days = list(pd.bdate_range(start, end))
    except Exception:
        return _fallback_time_blocks(start, end, shard_count)
    if len(days) < shard_count:
        shard_count = max(1, len(days))
    n = max(1, len(days) // shard_count)
    blocks: list[tuple[Any, Any]] = []
    for i in range(shard_count):
        lo = days[i * n]
        hi = days[min(len(days) - 1, (i + 1) * n - 1)]
        blocks.append((lo, hi))
    # 末块必须覆盖最后一天（``shard_count * n`` 可能 < len(days)，丢掉尾部日期）。
    if blocks and blocks[-1][1] != days[-1]:
        blocks[-1] = (blocks[-1][0], days[-1])
    return blocks


def _fallback_time_blocks(start: Any, end: Any, shard_count: int) -> list[tuple[Any, Any]]:
    import pandas as pd

    if start is None or end is None:
        return [(None, None)]
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    total = max(1, int((e - s).days))
    per = max(1, total // max(1, shard_count))
    blocks = []
    cursor = s
    for _ in range(shard_count):
        hi = min(e, cursor + pd.Timedelta(days=per))
        blocks.append((cursor, hi))
        cursor = hi + pd.Timedelta(days=1)
        if cursor >= e:
            break
    if blocks and blocks[-1][1] != e:
        blocks[-1] = (blocks[-1][0], e)
    return blocks


def split_instrument_universe(
    universe: list[Any],
    shard_count: int,
) -> list[list[Any]]:
    """把仪器全集切成 ``shard_count`` 个连续子集（确定性排序）。"""
    ordered = sorted(universe)
    n = max(1, len(ordered) // max(1, shard_count))
    out: list[list[Any]] = []
    for i in range(shard_count):
        out.append(ordered[i * n : (i + 1) * n])
    return out


class AutoShardPlanner:
    """§97：输入 task 语义 + P99 memory + SafeEnvelope + backend + history requirement
    → 输出 :class:`ShardPlan`（每片 P99 ≤ SafeEnvelope）。

    R38 P0-001/002（§4）：除 :class:`ShardPlan`（预算数字）外，还提供
    :meth:`build_shard_execution_plan` —— 真正可执行的切片计划（输入切片 +
    warmup overlap + 输出切片 + merge barrier + ShardMergeContract）。"""

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

    # -- R38 P0-001/002：真实可执行分片计划 --

    def build_shard_execution_plan(
        self,
        task: Any,
        *,
        safe_envelope_bytes: int,
        time_range: tuple[Any, Any] | None = None,
        instrument_universe: list[Any] | None = None,
        lookback_bars: int = 0,
        history_requirement: str = "none",
        failed_signature: str = "",
    ) -> "ShardExecutionPlan | None":
        """构造**真实可执行**的 :class:`ShardExecutionPlan`（§4 切片 + merge）。

        需要真实切片信息：
            - asset shard → ``instrument_universe``（必须给出仪器全集）；
            - time/session shard → ``time_range``（必须给出时间窗）；
            - rolling/stateful time shard → ``lookback_bars``（warmup overlap 长度）。
        信息不足 / 语义不合法 / 最小分片也放不下 → 返回 None（诚实不可分片）。
        """
        import pandas as pd

        from runtime.shard_execution_plan import (
            SHARD_ASSET,
            SHARD_SESSION,
            SHARD_TIME,
            DEFAULT_MERGE_CONTRACT,
            ShardDescriptor,
            ShardExecutionPlan,
            shape_signature,
        )

        base = self.plan_for_task(
            task,
            safe_envelope_bytes=safe_envelope_bytes,
            history_requirement=history_requirement,
        )
        if base is None:
            return None
        dim = base.dimension
        n = base.shard_count
        merge_policy = DEFAULT_MERGE_CONTRACT
        shard_task_ids: list[str] = []
        descriptors: list[ShardDescriptor] = []
        task_id = str(getattr(task, "task_id", "") or "")

        if dim == SHARD_ASSET:
            if not instrument_universe:
                return None  # 没有仪器全集，asset 切片无法真实构造
            subsets = split_instrument_universe(instrument_universe, n)
            for i, subset in enumerate(subsets):
                if not subset:
                    continue
                sid = f"{task_id}:shard:{i}"
                shard_task_ids.append(sid)
                descriptors.append(
                    ShardDescriptor(
                        shard_id=sid,
                        dimension=SHARD_ASSET,
                        input_slice=tuple(subset),
                        warmup_slice=None,
                        output_slice=None,
                        preserves_full_cross_section=False,
                        merge_order=i,
                    )
                )
        elif dim in (SHARD_TIME, SHARD_SESSION):
            if not time_range:
                return None  # 没有时间窗，time 切片无法真实构造
            blocks = build_time_blocks(time_range, n)
            for i, (lo, hi) in enumerate(blocks):
                sid = f"{task_id}:shard:{i}"
                shard_task_ids.append(sid)
                warmup_start = None
                needs_warmup = base.needs_warmup and lookback_bars > 0
                if needs_warmup:
                    warmup_start = lo - pd.Timedelta(days=int(lookback_bars) * 2)
                descriptors.append(
                    ShardDescriptor(
                        shard_id=sid,
                        dimension=dim,
                        input_slice=(lo, hi),
                        warmup_slice=(warmup_start or lo, hi),
                        output_slice=(lo, hi),
                        preserves_full_cross_section=(dim == SHARD_TIME),
                        merge_order=i,
                    )
                )
        else:
            return None

        if not descriptors:
            return None
        merge_task_id = f"{task_id}:merge"
        per_shard = base.per_shard_peak_bytes
        return ShardExecutionPlan(
            original_task_id=task_id,
            dimension=dim,
            shards=tuple(descriptors),
            merge_policy=merge_policy,
            merge_task_id=merge_task_id,
            shard_task_ids=tuple(shard_task_ids),
            per_shard_peak_bytes=per_shard,
            original_peak_bytes=base.original_peak_bytes,
            failed_shape_signature=failed_signature,
            reason=base.reason,
            time_range=time_range,
            instrument_universe=(
                tuple(instrument_universe) if instrument_universe else None
            ),
            safe_envelope_bytes=max(0, int(safe_envelope_bytes)),
        )

    def replan_after_oom(
        self,
        task: Any,
        *,
        safe_envelope_bytes: int,
        time_range: tuple[Any, Any] | None = None,
        instrument_universe: list[Any] | None = None,
        lookback_bars: int = 0,
        failed_shape_signature: str = "",
        min_shards_override: int | None = None,
    ) -> "ShardExecutionPlan | None":
        """OOM 后 smaller-shape replan（R38-P0-005）。

        用更小的 per-shard 目标（更多分片）重建计划，并保证：
            ``new_shape_signature != failed_shape_signature``
        否则返回 None（禁止同 shape 重试）。
        """
        from runtime.shard_execution_plan import shape_signature

        old = self._min_shards
        if min_shards_override is not None:
            self._min_shards = max(2, int(min_shards_override))
        try:
            plan = self.build_shard_execution_plan(
                task,
                safe_envelope_bytes=safe_envelope_bytes,
                time_range=time_range,
                instrument_universe=instrument_universe,
                lookback_bars=lookback_bars,
                failed_signature=failed_shape_signature,
            )
        finally:
            self._min_shards = old
        if plan is None:
            return None
        new_sig = shape_signature(
            task_id=plan.original_task_id,
            dimension=plan.dimension,
            shard_count=len(plan.shards),
            per_shard_peak_bytes=plan.per_shard_peak_bytes,
        )
        if failed_shape_signature and new_sig == failed_shape_signature:
            return None  # 同 shape 禁止重试
        return plan


def _is_cross_section(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return op.startswith("cs_") or "cross_section" in op


def _is_rolling(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return op.startswith("ts_") or op.startswith("rolling") or op.startswith("ewm")


def _is_stateful(task: Any) -> bool:
    op = str(getattr(task, "op", "") or "")
    return any(k in op for k in ("state", "recursive", "kalman", "garch", "ewm"))
