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
    """§89..96：按算子族语义给出合法维度（递归整棵 plan，最严格 barrier wins）。

    - SOURCE_SCAN → (time, asset)（扫描本身可任意切）。
    - 带 ``node_ref`` 的 task（ROOT / CSE shared / shard）→ 递归整棵 plan 取
      交集：``add(cs_rank(x), y)`` 顶层 op 是 ``add``，但内部 cs 节点把合法维度
      收窄到 time——**不能只看顶层 op 名**（stateful / group neutralization /
      PCA 同理，最严格 barrier wins）。
    - 无 plan 的裸 task → 按单 op 结构化分类（不 consult contract——语义真相）。
    """
    task_type = str(getattr(task, "task_type", "") or "")
    if task_type.endswith("SOURCE_SCAN"):
        return (SHARD_TIME, SHARD_ASSET)
    node_ref = getattr(task, "node_ref", None)
    if node_ref is not None:
        root = getattr(node_ref, "root", None) or node_ref
        try:
            from planner.physical_lowerer import plan_shard_semantics

            shardable, dim = plan_shard_semantics(root)
        except Exception:
            shardable, dim = False, None
        if not shardable or dim is None:
            return _UNKNOWN_POLICY
        return (dim,)
    op = str(getattr(task, "op", "") or "")
    if not op:
        return _UNKNOWN_POLICY
    return _single_op_policy(op)


def _single_op_policy(op: str) -> tuple[str | None, ...]:
    """单个 op 的合法 shard 维度（委托 planner 的结构化分类）。"""
    try:
        from planner.physical_lowerer import single_op_shard_policy

        policy = single_op_shard_policy(op)
        if policy:
            return tuple(policy)
    except Exception:
        pass
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
    calendar: Any | None = None,
) -> list[tuple[Any, Any]]:
    """把时间窗切成 ``shard_count`` 个连续块（按**真实交易日**分布）。

    ``calendar`` 传 :class:`storage.trading_calendar.TradingCalendar`（真实市场
    交易日，含节假日）时按真实交易日切分——春节/国庆等长假不会伪造工作日；
    否则回退 ``pd.bdate_range`` 工作日近似（research / 无日历）。quotient/
    remainder 切分，末块覆盖最后一个真实交易日（不丢尾部日期）。
    """
    import pandas as pd

    start, end = time_range
    days = _window_trading_days(start, end, calendar)
    if not days:
        return _fallback_time_blocks(start, end, shard_count)
    if len(days) < shard_count:
        shard_count = max(1, len(days))
    k = max(1, shard_count)
    q, r = divmod(len(days), k)
    blocks: list[tuple[Any, Any]] = []
    start_idx = 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        lo = days[start_idx]
        hi = days[start_idx + size - 1]
        blocks.append((lo, hi))
        start_idx += size
    # 末块必须覆盖最后一个真实交易日。
    if blocks and blocks[-1][1] != days[-1]:
        blocks[-1] = (blocks[-1][0], days[-1])
    return blocks


def _warmup_start_for(
    lo: Any,
    lookback_bars: int,
    *,
    calendar: Any | None,
) -> Any:
    """time shard 的 warmup overlap 起点。

    有真实交易日历时按 ``lookback_bars`` 个**真实 session** 回退（`calendar.offset`）；
    越界 / 无日历回退旧近似 ``lo - Timedelta(days=lookback_bars*2)``。
    """
    import pandas as pd

    lb = max(0, int(lookback_bars))
    if lb <= 0:
        return lo
    if calendar is not None:
        try:
            offset = getattr(calendar, "offset", None)
            if callable(offset):
                return offset(lo, -lb)
        except Exception:
            pass
    return lo - pd.Timedelta(days=lb * 2)


def _window_trading_days(
    start: Any,
    end: Any,
    calendar: Any | None,
) -> list[Any]:
    """[start, end] 内的真实交易日；无日历/不可用 → 空（调用方走 bdate fallback）。"""
    import pandas as pd

    if start is None or end is None:
        return []
    try:
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
    except Exception:
        return []
    if calendar is not None:
        try:
            days = getattr(calendar, "days", None)
            if days:
                return [d for d in days if s <= pd.Timestamp(d).normalize() <= e]
        except Exception:
            pass
        return []
    # 无真实日历：工作日近似（bdate_range，仅 research / 无日历环境）。
    try:
        return list(pd.bdate_range(s, e))
    except Exception:
        return []


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
    """把仪器全集切成 ``shard_count`` 个连续子集（确定性排序，**不丢尾部**）。

    quotient/remainder 切分（``np.array_split`` 语义）：前 ``r`` 片每片 ``q+1``
    个、其余 ``q`` 个，**覆盖全部仪器**。修复旧实现 ``len//shard_count`` +
    ``[i*n:(i+1)*n]`` 把余数尾部静默丢弃的 bug（如 5000 只切 3 片丢最后 2 只）。
    ``shard_count > 全集长度`` 时尾部为空片（调用方已跳过空片）。
    """
    ordered = sorted(universe)
    k = max(1, int(shard_count))
    n = len(ordered)
    if n == 0:
        return [[] for _ in range(k)]
    q, r = divmod(n, k)
    out: list[list[Any]] = []
    start = 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        out.append(ordered[start:start + size])
        start += size
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
        calendar: Any | None = None,
        attempt_id: int = 0,
    ) -> "ShardExecutionPlan | None":
        """构造**真实可执行**的 :class:`ShardExecutionPlan`（§4 切片 + merge）。

        需要真实切片信息：
            - asset shard → ``instrument_universe``（必须给出仪器全集）；
            - time/session shard → ``time_range``（必须给出时间窗）；
            - rolling/stateful time shard → ``lookback_bars``（warmup overlap 长度）。
        ``calendar`` 传真实交易日历（``TradingCalendar``）时 time shard 按真实
        交易日切分、warmup 按 ``lookback_bars`` 个**真实 session** 回退，而不是
        ``Timedelta(days=lookback_bars*2)`` 自然日近似。
        ``attempt_id``：OOM replan 第 2 次起，shard/merge task id 带 ``attempt:N``
        段（``root:x:attempt:2:shard:0``）——旧 attempt 的 in-flight future 与新
        DAG 不复用同名 task id，避免旧结果污染新 merge（P0-010 attempt 隔离）。
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
        # attempt 隔离：第 2 次起 id 带 ``:attempt:N:`` 段（attempt 1 保持旧命名，
        # 兼容既有测试）。旧 attempt 的 in-flight future 完成后按新 id 命名空间
        # 丢弃，绝不复用同一 task id 污染新 partial map。
        prefix = task_id
        if int(attempt_id or 0) > 1:
            prefix = f"{task_id}:attempt:{int(attempt_id)}"

        if dim == SHARD_ASSET:
            if not instrument_universe:
                return None  # 没有仪器全集，asset 切片无法真实构造
            subsets = split_instrument_universe(instrument_universe, n)
            for i, subset in enumerate(subsets):
                if not subset:
                    continue
                sid = f"{prefix}:shard:{i}"
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
            blocks = build_time_blocks(time_range, n, calendar=calendar)
            for i, (lo, hi) in enumerate(blocks):
                sid = f"{prefix}:shard:{i}"
                shard_task_ids.append(sid)
                warmup_start = None
                needs_warmup = base.needs_warmup and lookback_bars > 0
                if needs_warmup:
                    warmup_start = _warmup_start_for(
                        lo, lookback_bars, calendar=calendar
                    )
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
        merge_task_id = f"{prefix}:merge"
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
        calendar: Any | None = None,
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
                calendar=calendar,
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
