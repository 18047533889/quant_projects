# -*- coding: utf-8 -*-
"""R39-PERF-074：DuckDB query-class cohort 固定 profile 池。

当前 ``ExecutionResourcePlan.duckdb_threads`` 是单一标量（``auto()`` 里
``max(1, min(cpu, cpu // workers))``）。本模块引入固定 profile 池
``{1, 2, 4, 8}-thread``，Resource lease 根据 current concurrency / query shape /
scan bytes 选 profile，而不是静态 ``total_threads // max_concurrency``。

默认关闭（env ``FACTOR_ENGINE_COHORT_PROFILES=1`` 开启），避免改变默认行为：
- 未知 shape（scan_bytes<=0）→ ``select_cohort`` 返回 ``None``，调用方回退 legacy
  公式；
- 重 scan + 机器空闲 → 大 profile（8-thread）；并发升高自动降档。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

#: 固定 profile 池（thread 数）
COHORT_PROFILES: tuple[int, ...] = (1, 2, 4, 8)

#: 重 / 中 scan 字节阈值
_HEAVY_SCAN_BYTES = 512 * 1024 * 1024  # 512 MiB
_MEDIUM_SCAN_BYTES = 64 * 1024 * 1024  # 64 MiB

_ENABLE_ENV = "FACTOR_ENGINE_COHORT_PROFILES"


@dataclass(frozen=True)
class CohortQueryShape:
    """可供 cohort 选择的最小 shape 描述（未知字段用 0/None 表示不可用）。"""

    scan_bytes: int = 0
    estimated_rows: int = 0
    window: int | None = None
    is_fused_duckdb: bool = False


def cohort_profiles_enabled() -> bool:
    """env ``FACTOR_ENGINE_COHORT_PROFILES=1`` 开启（默认 OFF）。"""
    return os.environ.get(_ENABLE_ENV) == "1"


class QueryClassCohort:
    """固定 profile 池选择器。

    - 未知 shape（scan_bytes<=0）→ 返回 ``None``（调用方回退 legacy 公式）。
    - 重 scan + 机器空闲 → 8-thread；并发升高自动降档。
    """

    def select_cohort(
        self,
        query_shape: CohortQueryShape | None,
        scan_bytes: int,
        current_concurrency: int,
    ) -> int | None:
        """返回选中的 profile thread 数；未知 shape 返回 ``None``。"""
        scan_bytes = max(0, int(scan_bytes or 0))
        if scan_bytes <= 0 and query_shape is not None:
            scan_bytes = max(0, int(query_shape.scan_bytes or 0))
        if scan_bytes <= 0:
            return None  # 未知 shape → legacy 回退
        concurrency = max(1, int(current_concurrency or 1))
        if scan_bytes >= _HEAVY_SCAN_BYTES:
            if concurrency <= 1:
                return 8
            if concurrency <= 2:
                return 4
            return 2
        if scan_bytes >= _MEDIUM_SCAN_BYTES:
            if concurrency <= 1:
                return 4
            if concurrency <= 2:
                return 2
            return 1
        # 小 scan：低并发也只用小 profile，避免每 query 抢核
        if concurrency <= 1:
            return 2
        return 1


def select_cohort_threads(
    *,
    plan: Any | None = None,
    legacy_threads: int | None = None,
    query_shape: CohortQueryShape | None = None,
    scan_bytes: int = 0,
    current_concurrency: int = 0,
) -> int:
    """接线辅助：cohort 开启且 shape 可用时返回 profile，否则 legacy 线程数。

    ``plan`` 与 ``legacy_threads`` 二选一提供（plan 优先取 ``plan.duckdb_threads``）。
    """
    if not cohort_profiles_enabled():
        if plan is not None:
            return int(plan.duckdb_threads)
        return int(legacy_threads or 1)
    selected = QueryClassCohort().select_cohort(
        query_shape, scan_bytes, current_concurrency
    )
    if selected is not None:
        return selected
    if plan is not None:
        return int(plan.duckdb_threads)
    return int(legacy_threads or 1)
