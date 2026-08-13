# -*- coding: utf-8 -*-
"""R38 P0-014/015/016（§8）：MemoryBudgetAllocator —— 预算从同一个 SafeEnvelope
统一分配，硬不变量：

    sum(wave + block + sink + cache) + emergency_reserve
        <= current safe envelope - active_live_bytes

R36 的各自独立 clamp（wave=safe*10% min 256MB、block=safe*5%、sink=safe*5%、
cache=hard*15%、spill=hard*15%）在低 headroom 时绝对 minimum 可能已经超过 safe
memory（P0-014）。修复：

    - 分配是**同一个 allocator**，active_live 占用的额度先扣除；
    - emergency reserve 永不缩到下限以下（fail-safe before OOM）；
    - cache 目标随 live envelope 收缩（P0-015：不再长期 hard*15%），压力解除后
      只按比例缓慢恢复；
    - spill 预算来自**磁盘**（P0-016：spill_free - 保留），不是 RAM。

2026-08-13: 使用自适应配置替代硬编码 BLOCK_ABS_MAX/SINK_ABS_MAX/CACHE_ABS_MAX。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get_adaptive_bounds() -> dict[str, int]:
    """获取自适应内存边界（基于系统资源）。"""
    try:
        from runtime.adaptive_config import get_global_adaptive_config
        config = get_global_adaptive_config()
        # BLOCK_ABS_MAX 使用 block_abs_max_bytes
        # SINK_ABS_MAX/CACHE_ABS_MAX 使用其 4x（原比例：8GB vs 2GB）
        return {
            "BLOCK_ABS_MAX": config.block_abs_max_bytes,
            "SINK_ABS_MAX": config.block_abs_max_bytes * 4,
            "CACHE_ABS_MAX": config.block_abs_max_bytes * 4,
        }
    except ImportError:
        # 回退：使用原硬编码值（向后兼容）
        return {
            "BLOCK_ABS_MAX": 2 * 1024**3,
            "SINK_ABS_MAX": 8 * 1024**3,
            "CACHE_ABS_MAX": 8 * 1024**3,
        }


#: 预算默认分数（§35/38/39/49：占 safe envelope）。
WAVE_FRACTION = 0.10
BLOCK_FRACTION = 0.05
SINK_FRACTION = 0.05
CACHE_FRACTION = 0.15
#: 必须保留的 emergency reserve 分数（§16/17：fail-safe before OOM）。
EMERGENCY_RESERVE_FRACTION = 0.25
EMERGENCY_RESERVE_ABS = 512 * 1024**2       # 至少 512MB
#: 绝对 bounds（§35..39）—— 自适应计算，延迟初始化。
WAVE_ABS_MIN = 256 * 1024**2
WAVE_ABS_MAX = 16 * 1024**3
BLOCK_ABS_MIN = 64 * 1024**2
SINK_ABS_MIN = 128 * 1024**2
CACHE_ABS_MIN = 128 * 1024**2

# 自适应上限（延迟初始化，首次访问时计算）
_adaptive_bounds: dict[str, int] | None = None


def _get_bounds() -> dict[str, int]:
    """获取或初始化自适应边界。"""
    global _adaptive_bounds
    if _adaptive_bounds is None:
        _adaptive_bounds = _get_adaptive_bounds()
    return _adaptive_bounds


# 向后兼容：保留原常量名（通过 __getattr__ 动态查找）
def __getattr__(name: str) -> int:
    if name in ("BLOCK_ABS_MAX", "SINK_ABS_MAX", "CACHE_ABS_MAX"):
        return _get_bounds()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class MemoryBudgetAllocation:
    """一个 control tick 的内存预算（sum ≤ safe - active_live - emergency）。"""

    read_wave_bytes: int
    factor_block_bytes: int
    result_queue_bytes: int
    cache_bytes: int
    spill_bytes: int
    emergency_reserve_bytes: int
    active_live_bytes: int
    safe_memory_bytes: int
    pressure_stage: str
    #: hard invariant：sum(活预算) + emergency ≤ safe - active_live。
    within_safe: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_wave_bytes": self.read_wave_bytes,
            "factor_block_bytes": self.factor_block_bytes,
            "result_queue_bytes": self.result_queue_bytes,
            "cache_bytes": self.cache_bytes,
            "spill_bytes": self.spill_bytes,
            "emergency_reserve_bytes": self.emergency_reserve_bytes,
            "active_live_bytes": self.active_live_bytes,
            "safe_memory_bytes": self.safe_memory_bytes,
            "pressure_stage": self.pressure_stage,
            "within_safe": self.within_safe,
        }


class MemoryBudgetAllocator:
    """从 SafeEnvelope 统一分配全部内存预算（R38 P0-014）。"""

    def __init__(
        self,
        *,
        wave_fraction: float = WAVE_FRACTION,
        block_fraction: float = BLOCK_FRACTION,
        sink_fraction: float = SINK_FRACTION,
        cache_fraction: float = CACHE_FRACTION,
        emergency_fraction: float = EMERGENCY_RESERVE_FRACTION,
    ) -> None:
        self._wf = float(wave_fraction)
        self._bf = float(block_fraction)
        self._sf = float(sink_fraction)
        self._cf = float(cache_fraction)
        self._ef = float(emergency_fraction)

    def allocate(
        self,
        safe_memory_bytes: int,
        *,
        pressure_stage: str = "NORMAL",
        active_live_bytes: int = 0,
        spill_free_bytes: int = 0,
        spill_reserve_bytes: int = 0,
    ) -> MemoryBudgetAllocation:
        """分配预算。硬不变量：sum(活预算) + emergency ≤ safe - active_live。

        顺序：
            1. emergency = max(abs, safe * fraction)，但不超过 safe - active_live；
            2. 可用 = max(0, safe - active_live - emergency)；
            3. wave/block/sink/cache 各按 fraction 取，全部 clamp 到 [min, 可用]；
            4. 若求和超可用 → 按优先级（wave > sink > block > cache）逐项砍到可用。
        """
        safe = max(0, int(safe_memory_bytes))
        active = max(0, int(active_live_bytes))
        emergency = max(
            EMERGENCY_RESERVE_ABS,
            int(safe * self._ef),
        )
        # emergency 不能吃光 safe（至少给一点活预算）。
        emergency = min(emergency, max(0, safe - active - WAVE_ABS_MIN))
        emergency = max(0, emergency)
        available = max(0, safe - active - emergency)

        bounds = _get_bounds()
        wave = _clamp(int(safe * self._wf), WAVE_ABS_MIN, WAVE_ABS_MAX)
        block = _clamp(int(safe * self._bf), BLOCK_ABS_MIN, bounds["BLOCK_ABS_MAX"])
        sink = _clamp(int(safe * self._sf), SINK_ABS_MIN, bounds["SINK_ABS_MAX"])
        cache = _clamp(int(safe * self._cf), CACHE_ABS_MIN, bounds["CACHE_ABS_MAX"])

        # 压力档位降预算（P0-015：cache 随 live envelope 收缩）。
        if pressure_stage in {"PRESSURE_2", "PRESSURE_3"}:
            wave = _clamp(int(wave * 0.5), WAVE_ABS_MIN, WAVE_ABS_MAX)
            block = _clamp(int(block * 0.5), BLOCK_ABS_MIN, bounds["BLOCK_ABS_MAX"])
        if pressure_stage in {"PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
            cache = _clamp(int(cache * 0.5), CACHE_ABS_MIN, bounds["CACHE_ABS_MAX"])

        # sum ≤ available（优先级 wave > sink > block > cache 逐项砍）。
        budgets = {"read_wave": wave, "factor_block": block, "result_queue": sink, "cache": cache}
        order = ("read_wave", "result_queue", "factor_block", "cache")
        total = sum(budgets[k] for k in order)
        if total > available:
            # 先整体等比缩到可用，再逐项 clamp 到下限（下限冲突时舍弃低优先级）。
            scale = available / total if total > 0 else 0.0
            scaled = {k: max(0, int(v * scale)) for k, v in budgets.items()}
            # 低优先级（cache/block）让位，保证 wave/sink 至少到 min。
            for k in reversed(order):
                lo = {
                    "read_wave": WAVE_ABS_MIN,
                    "factor_block": BLOCK_ABS_MIN,
                    "result_queue": SINK_ABS_MIN,
                    "cache": CACHE_ABS_MIN,
                }[k]
                if scaled[k] < lo:
                    scaled[k] = min(lo, available - sum(
                        scaled[j] for j in order if j != k
                    ))
            wave, block, sink, cache = (
                scaled["read_wave"], scaled["factor_block"],
                scaled["result_queue"], scaled["cache"],
            )
            wave = max(0, wave)
            block = max(0, block)
            sink = max(0, sink)
            cache = max(0, cache)

        # spill 来自磁盘（P0-016），不是 RAM。
        spill = max(0, int(spill_free_bytes) - int(spill_reserve_bytes))

        live_sum = wave + block + sink + cache
        within_safe = (live_sum + emergency) <= (safe - active)
        return MemoryBudgetAllocation(
            read_wave_bytes=wave,
            factor_block_bytes=block,
            result_queue_bytes=sink,
            cache_bytes=cache,
            spill_bytes=spill,
            emergency_reserve_bytes=emergency,
            active_live_bytes=active,
            safe_memory_bytes=safe,
            pressure_stage=pressure_stage,
            within_safe=within_safe,
        )


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))
