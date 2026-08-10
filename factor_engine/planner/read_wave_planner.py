# -*- coding: utf-8 -*-
"""R27-063..068: ReadWavePlanner —— 内存有界的读波规划。

目标（R27-062/065/067/068）
    - **不再**全 batch 一次性 union 所有列 prefetch（source panel/cache 爆内存）。
    - 将 factor tasks 按 source_scope / dataset / snapshot / time_range / universe
      / column overlap / lookback 聚成 Read Waves。
    - wave 追求「共享最大、内存有限」：共享列越多越值得一起，但总驻留字节不能
      超过 wave_memory_budget（R27-064/065）。reuse_density 定义：
        reuse_density = shared_scan_bytes_saved / wave_memory_bytes
    - 每 wave 只投影该 wave 真实需要的列（R27-067 禁止 SELECT *）。
    - 依赖 DataAccess pushdown：time range / instrument filter / manifest prune /
      row-group prune（R27-068）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROOT,
    TASK_SOURCE_SCAN,
)

_DEFAULT_WAVE_MEMORY_BUDGET = 4 * 1024**3  # 4GiB 默认 wave 内存预算


@dataclass(frozen=True)
class ReadWave:
    """一个读波：一次 scan 喂多个 task（R27-066）。"""

    wave_id: int
    source_scope: str
    dataset: str
    snapshot_id: str
    columns: frozenset[str]
    time_range: tuple[str, str] | None
    task_ids: tuple[str, ...]
    estimated_scan_bytes: int = 0
    estimated_memory_bytes: int = 0
    reuse_density: float = 0.0
    locality_groups: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_id": self.wave_id,
            "source_scope": self.source_scope,
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "columns": sorted(self.columns),
            "time_range": list(self.time_range) if self.time_range else None,
            "task_ids": list(self.task_ids),
            "estimated_scan_bytes": self.estimated_scan_bytes,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "reuse_density": round(self.reuse_density, 4),
            "locality_groups": [list(g) for g in self.locality_groups],
        }


@dataclass
class ReadWavePlan:
    """读波计划：waves + 总览（R27-063/064）。"""

    waves: list[ReadWave] = field(default_factory=list)

    @property
    def total_scan_bytes(self) -> int:
        return sum(w.estimated_scan_bytes for w in self.waves)

    @property
    def total_wave_memory_bytes(self) -> int:
        return sum(w.estimated_memory_bytes for w in self.waves)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_count": len(self.waves),
            "total_scan_bytes": self.total_scan_bytes,
            "total_wave_memory_bytes": self.total_wave_memory_bytes,
            "waves": [w.to_dict() for w in self.waves],
        }


def _default_scan_bytes(columns: Iterable[str], rows_estimate: int = 500_000) -> int:
    """缺省 scan 字节估算：列数 × 行数 × 8B（manifest 给出前用）。"""
    return max(0, len(set(columns)) * max(0, rows_estimate) * 8)


def _source_scope_key(
    *,
    dataset: str,
    snapshot_id: str | None,
    source_scope: str,
    time_range: tuple[str, str] | None,
) -> str:
    """wave 聚类的 source scope 身份（R27-063/160）。"""
    return "::".join(
        [
            dataset,
            snapshot_id or "",
            source_scope or "",
            f"{time_range[0]}~{time_range[1]}" if time_range else "*",
        ]
    )


class ReadWavePlanner:
    """按 source scope / 列共享聚波，内存有界（R27-063..065）。"""

    def __init__(
        self,
        *,
        wave_memory_budget: int = _DEFAULT_WAVE_MEMORY_BUDGET,
        rows_estimate: int = 500_000,
    ) -> None:
        self.wave_memory_budget = max(1, wave_memory_budget)
        self.rows_estimate = max(1, rows_estimate)
        self._requests: list[_ReadRequest] = []

    # -- task 登记 --

    def register_scan_task(
        self,
        task_id: str,
        *,
        dataset: str,
        source_scope: str,
        snapshot_id: str | None,
        time_range: tuple[str, str] | None,
        columns: Iterable[str],
        estimated_scan_bytes: int | None = None,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        self._requests.append(
            _ReadRequest(
                task_id=task_id,
                dataset=dataset,
                source_scope=source_scope,
                snapshot_id=snapshot_id,
                time_range=time_range,
                columns=frozenset(columns),
                estimated_scan_bytes=(
                    estimated_scan_bytes
                    if estimated_scan_bytes is not None
                    else _default_scan_bytes(columns, self.rows_estimate)
                ),
                estimated_memory_bytes=(
                    estimated_memory_bytes
                    if estimated_memory_bytes is not None
                    else _default_scan_bytes(columns, self.rows_estimate)
                ),
            )
        )

    def _reset(self) -> None:
        self._requests: list[_ReadRequest] = []

    def plan(self) -> ReadWavePlan:
        """聚波：同 source scope 的请求按列共享贪心合并，内存有界。

        - 同 scope 的请求合并成一个 wave；共享列越多 reuse_density 越高。
        - 合并后驻留字节超预算就拆 wave（R27-128：locality 不能压过内存安全）。
        """
        requests = list(self._requests)
        self._reset()
        by_scope: dict[str, list[_ReadRequest]] = {}
        for req in requests:
            key = _source_scope_key(
                dataset=req.dataset,
                snapshot_id=req.snapshot_id,
                source_scope=req.source_scope,
                time_range=req.time_range,
            )
            by_scope.setdefault(key, []).append(req)

        waves: list[ReadWave] = []
        wave_id = 0
        for key, reqs in sorted(by_scope.items()):
            # 组内所有 request 的 dataset/snapshot/source_scope/time_range 相同
            #（按 key 分组构造保证）；以第一个为组代表，避免从含 '::' 的
            # source_scope 反向 split 出错。
            rep = reqs[0]
            # 贪心：按共享列从高到低合并，内存有界。
            candidates = sorted(
                reqs,
                key=lambda r: (-len(r.columns), r.task_id),
            )
            current: list[_ReadRequest] = []
            current_cols: set[str] = set()
            current_mem = 0
            for req in candidates:
                if current and current_mem + req.estimated_memory_bytes > self.wave_memory_budget:
                    waves.append(self._build_wave(wave_id, rep.dataset, rep.snapshot_id or "",
                                                  rep.source_scope, rep.time_range,
                                                  current, current_cols))
                    wave_id += 1
                    current = []
                    current_cols = set()
                    current_mem = 0
                current.append(req)
                current_cols |= req.columns
                current_mem += req.estimated_memory_bytes
            if current:
                waves.append(self._build_wave(wave_id, rep.dataset, rep.snapshot_id or "",
                                              rep.source_scope, rep.time_range,
                                              current, current_cols))
                wave_id += 1
        return ReadWavePlan(waves=waves)

    def _build_wave(
        self,
        wave_id: int,
        dataset: str,
        snapshot: str,
        source_scope: str,
        time_range: tuple[str, str] | None,
        reqs: list["_ReadRequest"],
        cols: set[str],
    ) -> ReadWave:
        total_scan = sum(r.estimated_scan_bytes for r in reqs)
        total_mem = sum(r.estimated_memory_bytes for r in reqs)
        # R27-065 reuse_density = shared_scan_bytes_saved / wave_memory_bytes
        shared_saved = total_scan - _default_scan_bytes(cols, self.rows_estimate)
        reuse_density = shared_saved / max(1, total_mem)
        return ReadWave(
            wave_id=wave_id,
            source_scope=source_scope,
            dataset=dataset,
            snapshot_id=snapshot,
            columns=frozenset(cols),
            time_range=time_range,
            task_ids=tuple(sorted(r.task_id for r in reqs)),
            estimated_scan_bytes=total_scan,
            estimated_memory_bytes=total_mem,
            reuse_density=round(reuse_density, 4),
        )


@dataclass
class _ReadRequest:
    """内部读请求（不暴露）。"""

    task_id: str
    dataset: str
    source_scope: str
    snapshot_id: str | None
    time_range: tuple[str, str] | None
    columns: frozenset[str]
    estimated_scan_bytes: int
    estimated_memory_bytes: int


def build_waves_from_dag(
    dag: Any,
    *,
    wave_memory_budget: int = _DEFAULT_WAVE_MEMORY_BUDGET,
    rows_estimate: int = 500_000,
    scan_cost_map: dict[str, Any] | None = None,
) -> ReadWavePlan:
    """从 PhysicalFactorDAG 构建读波（source_scope / dataset / snapshot 聚合）。

    ``scan_cost_map``：``task_id -> ScanCost``（R27-060/061：FE 在真正 read 前
    从 DataAccess 拿 selected_bytes / estimated_rows / remote）。
    """
    planner = ReadWavePlanner(
        wave_memory_budget=wave_memory_budget,
        rows_estimate=rows_estimate,
    )
    scan_cost_map = scan_cost_map or {}
    for task in dag.tasks.values():
        if task.task_type not in {TASK_SOURCE_SCAN, TASK_CSE_SHARED, TASK_ROOT}:
            continue
        cost = scan_cost_map.get(task.task_id)
        est_scan = cost.selected_bytes if cost is not None and cost.selected_bytes else None
        est_mem = cost.projection_bytes if cost is not None and cost.projection_bytes else None
        planner.register_scan_task(
            task.task_id,
            dataset=task.source_scope.split("::")[0] if "::" in task.source_scope else task.source_scope,
            source_scope=task.source_scope,
            snapshot_id=task.source_snapshot_id or None,
            time_range=None,
            columns=task.inputs if task.task_type == TASK_SOURCE_SCAN else (task.op,),
            estimated_scan_bytes=est_scan,
            estimated_memory_bytes=est_mem,
        )
    return planner.plan()
