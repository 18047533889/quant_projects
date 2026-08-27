# -*- coding: utf-8 -*-
"""P3/P4: AutoMemoryBudget —— startup + resample memory budget authority。

问题：多处在硬编码 ``4 * 1024**3``（4GiB）作为固定内存默认值，直接当作预留预算
使用，而不是由统一的 ResourceBroker 从 live headroom 派生。本模块只定义**公式**：

    - ``HardMemoryLimit``   = min(cgroup memory.max, SLURM, RLIMIT,
                                   user-configured cap, host capacity)
    - ``EmergencyReserve``  = max(minimum_reserve, HardLimit × reserve_fraction)
    - ``SafeLiveBudget``    = min(cgroup_remaining, host_MemAvailable − reserve,
                                  configured_remaining)
    - ``ExecutionBudget``   = SafeLiveBudget × safety_factor
                              （0.65~0.75 起步，可校准至 0.80~0.85）

任何默认值都只是**上限/回退**，绝不是一个会被直接预留的固定 4GiB。

本模块不含任何 broker 状态；单权威由
:class:`factor_engine.runtime.resource_broker.ResourceBroker` 持有，本模块只做
纯计算（broker 消费本模块的 ``compute_auto_memory_budget``）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 统一 MemoryLeaseKind（P3/P4）：所有内存租约种类。
# ---------------------------------------------------------------------------


class MemoryLeaseKind(str, enum.Enum):
    """统一内存租约种类（P3/P4）。"""

    SOURCE_READ = "SOURCE_READ"            # 源数据读取缓冲
    READ_WAVE = "READ_WAVE"                # read wave（SOURCE_SCAN 批量读取）
    COMPUTE = "COMPUTE"                    # 计算峰值（pandas/duckdb/polars 工作区）
    CSE_CACHE = "CSE_CACHE"                # CSE 共享缓存
    TRANSFER_BUFFER = "TRANSFER_BUFFER"    # backend→backend / 远程传输缓冲
    RESULT_QUEUE = "RESULT_QUEUE"          # sink result queue
    WRITER_BATCH = "WRITER_BATCH"          # writer 攒批缓冲
    BACKEND_WORKSPACE = "BACKEND_WORKSPACE"  # backend 私有工作区
    SPILL_STAGING = "SPILL_STAGING"        # spill 落盘 staging

    # P0-10..12: FactorEngine 结果直写 COS 的内存租约种类（bounded-memory）。
    COS_READ_BUFFER = "COS_READ_BUFFER"            # COS 读取缓冲
    PARQUET_DECODE = "PARQUET_DECODE"              # parquet 解码工作区
    REMOTE_RANGE_BUFFER = "REMOTE_RANGE_BUFFER"    # 远程 range-read 缓冲
    FEATURE_BLOCK_ASSEMBLY = "FEATURE_BLOCK_ASSEMBLY"  # 因子 block 内存组装
    COS_UPLOAD_PART = "COS_UPLOAD_PART"            # COS 上传分片缓冲
    COS_UPLOAD_INFLIGHT = "COS_UPLOAD_INFLIGHT"    # COS 上传进行中（inflight）
    MANIFEST_BUFFER = "MANIFEST_BUFFER"            # manifest / 布局元数据缓冲

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# AutoMemoryBudget 公式常量。
# ---------------------------------------------------------------------------

#: EmergencyReserve 下限（GB）与占 HardLimit 的最小比例。
DEFAULT_MIN_RESERVE_GB = 8.0
DEFAULT_MIN_RESERVE_FRACTION = 0.15
#: EmergencyReserve 封顶（占 HardLimit 的比例，R27-029：小机器不被吃满）。
RESERVE_CAP_FRACTION = 0.35
#: ExecutionBudget = SafeLiveBudget × safety_factor（0.65~0.75 起步）。
DEFAULT_SAFETY_FACTOR = 0.70
#: 可校准上限（安全前提下逐步放开）。
MAX_SAFETY_FACTOR = 0.85
#: 仅当什么都探测不到时的绝对上限（是 **cap**，不是被预留的固定值）。
ABSOLUTE_FALLBACK_EXECUTION_CAP = 4 * 1024**3


@dataclass(frozen=True)
class AutoMemoryBudget:
    """一次 startup / resample 算出的内存预算。

    ``EmergencyReserve`` 是**预留**：从 ``SafeLiveBudget`` 中扣除，不作为
    ``ExecutionBudget`` 的一部分。``ExecutionBudget <= SafeLiveBudget``。
    """

    hard_memory_limit: int
    emergency_reserve: int
    safe_live_budget: int
    execution_budget: int
    safety_factor: float
    #: SafeLiveBudget 各候选来源（可读性/调试）。
    candidates: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_memory_limit": self.hard_memory_limit,
            "emergency_reserve": self.emergency_reserve,
            "safe_live_budget": self.safe_live_budget,
            "execution_budget": self.execution_budget,
            "safety_factor": round(self.safety_factor, 3),
            "candidates": dict(self.candidates),
        }


def compute_auto_memory_budget(
    *,
    hard_memory_limit: int,
    cgroup_current: int | None,
    host_mem_available: int,
    process_family_rss: int,
    min_reserve_gb: float = DEFAULT_MIN_RESERVE_GB,
    min_reserve_fraction: float = DEFAULT_MIN_RESERVE_FRACTION,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> AutoMemoryBudget:
    """P3/P4：从一次资源采样计算 AutoMemoryBudget（纯函数，无副作用）。

    - ``HardMemoryLimit`` 由调用方传入（通常来自 ``effective_memory_limit_bytes``，
      已含 cgroup/SLURM/RLIMIT/user-cap/host 的最严格 min）。
    - ``EmergencyReserve = max(min_reserve_abs, HardLimit × fraction)``，封顶
      ``RESERVE_CAP_FRACTION``。
    - ``SafeLiveBudget = min(cgroup_remaining, host_remaining, configured)``。
    - ``ExecutionBudget = SafeLiveBudget × safety_factor``。

    任何默认值都只是**上限/回退**；探测失败时基于 ``HardMemoryLimit`` 派生，绝不
    把一个固定 4GiB 直接当预留预算。固定 4GiB 仅作为绝对上限 ``ABSOLUTE_FALLBACK_EXECUTION_CAP``。
    """
    hard = max(0, int(hard_memory_limit))
    sf = min(float(safety_factor), MAX_SAFETY_FACTOR)
    sf = max(0.0, sf)

    # EmergencyReserve = max(minimum_reserve, HardLimit × fraction)，封顶 35%。
    raw_reserve = max(
        int(min_reserve_gb * 1024**3),
        int(hard * float(min_reserve_fraction)),
    )
    emergency = raw_reserve
    if hard:
        emergency = min(raw_reserve, int(hard * RESERVE_CAP_FRACTION))

    candidates: dict[str, int] = {}
    if cgroup_current is not None and hard:
        candidates["cgroup_remaining"] = max(0, hard - int(cgroup_current))
    candidates["host_remaining"] = max(0, int(host_mem_available) - emergency)
    candidates["configured_remaining"] = max(0, hard - int(process_family_rss))

    safe = min(candidates.values()) if candidates else 0
    # ExecutionBudget = SafeLiveBudget × safety_factor。
    execution = int(safe * sf)
    if execution <= 0:
        # 探测失败 → 基于 HardLimit 派生（不是固定 4GiB 预留）。
        execution = int(hard * sf)

    return AutoMemoryBudget(
        hard_memory_limit=hard,
        emergency_reserve=max(0, emergency),
        safe_live_budget=max(0, safe),
        execution_budget=max(0, execution),
        safety_factor=sf,
        candidates=candidates,
    )
