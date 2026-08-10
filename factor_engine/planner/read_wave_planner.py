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
    source_scope_from_key,
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


def _scope_group_key(
    *,
    dataset: str,
    snapshot_id: str | None,
    source_scope: str,
    time_range: tuple[str, str] | None,
) -> str:
    """wave 聚类的 source scope 身份（R27-063/160）。

    R33-P0-013：dataset 用 typed 字段（SourceScopeId.dataset），**不**从含
    ``::`` 的 source_scope 反向 split；time_range 参与聚类（不同历史窗的 wave
    不混扫）。
    """
    return "::".join(
        [
            dataset,
            snapshot_id or "",
            source_scope or "",
            f"{time_range[0]}~{time_range[1]}" if time_range else "*",
        ]
    )


class ReadWavePlanner:
    """按 source scope / 列共享聚波，内存有界（R27-063..065）。

    R33-P0-014/015：wave 内存按 **union projected columns** 记账（共享列只算
    一份，不再按 request 内存求和）；贪心按 **marginal overlap**（新增共享最多、
    边际字节最小者先入波）。
    """

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
        instrument_scope: tuple[str, ...] | None = None,
        universe_id: str | None = None,
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
                instrument_scope=tuple(instrument_scope or ()),
                universe_id=universe_id or "",
            )
        )

    def _reset(self) -> None:
        self._requests: list[_ReadRequest] = []

    def plan(self) -> ReadWavePlan:
        """聚波：同 source scope 的请求按 marginal-overlap 贪心合并，内存有界。

        R33-P0-014：每个 wave 的驻留字节 = union 物理列 × 行数 × 8B（共享列只算
        一次），不是 request 内存求和——100 个因子共用 ``close`` 只驻留一份。
        R33-P0-015：候选按「与当前 union 的新增共享列 / 边际新增字节」排序，
        不再只按列数降序。
        """
        requests = list(self._requests)
        self._reset()
        by_scope: dict[str, list[_ReadRequest]] = {}
        for req in requests:
            key = _scope_group_key(
                dataset=req.dataset,
                snapshot_id=req.snapshot_id,
                source_scope=req.source_scope,
                time_range=req.time_range,
            )
            by_scope.setdefault(key, []).append(req)

        waves: list[ReadWave] = []
        wave_id = 0
        for key, reqs in sorted(by_scope.items()):
            rep = reqs[0]
            remaining = list(reqs)
            while remaining:
                current: list[_ReadRequest] = []
                current_cols: set[str] = set()
                current_mem = 0
                current_scan = 0
                # marginal-overlap 贪心：每步挑「与当前 union 重叠最多 / 边际
                # 字节最小」的下一个 request。
                while remaining:
                    current_cols = current_cols or set()
                    best_idx = _best_marginal(remaining, current_cols, self.rows_estimate)
                    cand = remaining.pop(best_idx)
                    union_cols = current_cols | cand.columns
                    new_mem = _default_scan_bytes(union_cols, self.rows_estimate)
                    if current and new_mem > self.wave_memory_budget:
                        # 超预算：该 request 留给下一个 wave。
                        remaining.append(cand)
                        break
                    current.append(cand)
                    current_cols = union_cols
                    current_mem = new_mem
                    current_scan += cand.estimated_scan_bytes
                if current:
                    waves.append(self._build_wave(
                        wave_id, rep.dataset, rep.snapshot_id or "", rep.source_scope,
                        rep.time_range, current, current_cols, current_mem, current_scan,
                    ))
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
        union_mem: int,
        total_scan: int,
    ) -> ReadWave:
        # R27-065 reuse_density = shared_scan_bytes_saved / wave_memory_bytes
        shared_saved = total_scan - _default_scan_bytes(cols, self.rows_estimate)
        reuse_density = shared_saved / max(1, union_mem)
        return ReadWave(
            wave_id=wave_id,
            source_scope=source_scope,
            dataset=dataset,
            snapshot_id=snapshot,
            columns=frozenset(cols),
            time_range=time_range,
            task_ids=tuple(sorted(r.task_id for r in reqs)),
            estimated_scan_bytes=total_scan,
            estimated_memory_bytes=union_mem,
            reuse_density=round(reuse_density, 4),
        )


def _best_marginal(
    candidates: list["_ReadRequest"],
    current_cols: set[str],
    rows_estimate: int,
) -> int:
    """R33-P0-015：marginal-overlap 评分——共享新增多、边际字节小者优先。"""
    best_idx = 0
    best_score = -1.0
    for idx, req in enumerate(candidates):
        union_cols = current_cols | req.columns
        marginal_bytes = _default_scan_bytes(union_cols, rows_estimate)
        if current_cols:
            shared_new = len(req.columns & current_cols)
            score = shared_new / max(1.0, float(marginal_bytes))
        else:
            # 空波起点：直接按列数降序（最宽的先撑起复用面）。
            score = len(req.columns) / max(1.0, float(marginal_bytes))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


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
    instrument_scope: tuple[str, ...] = ()
    universe_id: str = ""


def build_waves_from_dag(
    dag: Any,
    *,
    wave_memory_budget: int = _DEFAULT_WAVE_MEMORY_BUDGET,
    rows_estimate: int = 500_000,
    scan_cost_map: dict[str, Any] | None = None,
    scope_scan_cost_map: dict[str, Any] | None = None,
) -> ReadWavePlan:
    """从 PhysicalFactorDAG 构建读波（source_scope / dataset / snapshot 聚合）。

    R33-P0-010/011：``columns`` 取 **SOURCE_SCAN task 的 ``required_columns``**
    （真实物理列），不再用 ``task.op`` / ``task.inputs`` 推断；``time_range`` 取
    task 的真实加载窗口（R33-P0-012）。SOURCE_SCAN 之外只登记 CSE_SHARED/ROOT
    的 scope 身份（它们不贡献列，但共享同一 scope 的 wave 计数）。

    ``scan_cost_map``：``task_id -> ScanCost``；``scope_scan_cost_map``：
    ``source_scope -> ScanCost``。优先级 task_id > source_scope > 静态估算。
    """
    planner = ReadWavePlanner(
        wave_memory_budget=wave_memory_budget,
        rows_estimate=rows_estimate,
    )
    scan_cost_map = scan_cost_map or {}
    scope_scan_cost_map = scope_scan_cost_map or {}
    for task in dag.tasks.values():
        if task.task_type not in {TASK_SOURCE_SCAN, TASK_CSE_SHARED, TASK_ROOT}:
            continue
        cost = scan_cost_map.get(task.task_id)
        if cost is None:
            cost = scope_scan_cost_map.get(task.source_scope)
        if task.task_type == TASK_SOURCE_SCAN:
            spec = task.source_scan_spec
            columns = tuple(spec.required_columns) if spec is not None else task.required_columns
            dataset = spec.dataset if spec is not None else task.source_scope.split("::")[0]
            time_range = spec.time_range if spec is not None else task.time_range
            est_scan = cost.selected_bytes if cost is not None and cost.selected_bytes else None
            est_mem = cost.projection_bytes if cost is not None and cost.projection_bytes else None
            planner.register_scan_task(
                task.task_id,
                dataset=dataset,
                source_scope=task.source_scope,
                snapshot_id=task.source_snapshot_id or None,
                time_range=time_range,
                columns=columns,
                estimated_scan_bytes=est_scan,
                estimated_memory_bytes=est_mem,
                instrument_scope=spec.instrument_scope if spec is not None else task.instrument_scope,
                universe_id=spec.universe_id if spec is not None else None,
            )
        else:
            # CSE/ROOT 共享同一 scope 的 wave 计数（不新增列）。
            # R33-P0-013：dataset 用 typed SourceScopeId 字段，不 split("::")。
            try:
                _scope = source_scope_from_key(task.source_scope)
                dataset = _scope.dataset or task.source_scope
            except Exception:
                dataset = task.source_scope
            planner.register_scan_task(
                task.task_id,
                dataset=dataset,
                source_scope=task.source_scope,
                snapshot_id=task.source_snapshot_id or None,
                time_range=task.time_range,
                columns=(),
                estimated_scan_bytes=0,
                estimated_memory_bytes=0,
            )
    return planner.plan()
