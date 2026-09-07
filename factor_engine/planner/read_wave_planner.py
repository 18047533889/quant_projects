# -*- coding: utf-8 -*-
"""R27-063..068 + R39-P0-PERF-005..012 —— 读波规划：从「固定 500k×8B」升级为
真正的物理 footprint optimizer。

R27 目标
    - **不再**全 batch 一次性 union 所有列 prefetch（source panel/cache 爆内存）。
    - 将 factor tasks 按 source_scope / dataset / snapshot / time_range / universe
      / column overlap / lookback 聚成 Read Waves。
    - wave 追求「共享最大、内存有限」：共享列越多越值得一起，但总驻留字节不能
      超过 wave_memory_budget。reuse_density = shared_scan_bytes_saved / wave_memory_bytes。

R39 §4 升级
    - PERF-005：ReadWave 拆 ``source_tasks`` / ``consumer_tasks``；ROOT/CSE_SHARED
      不再作为零列 scan request（ZERO_COLUMN_READ_WAVE == 0；
      NON_SOURCE_TASK_IN_READ_WAVE_SOURCE_TASKS == 0）。
    - PERF-006：wave memory 消费真实投影列 footprint（manifest / parquet / ScanCost
      / fallback），不再固定 500k×8B。
    - PERF-007：marginal 评分用「边际新增字节」做 denominator + bounded local search。
    - PERF-008：cost-based superset coalescing（20d/60d/120d 只有成本证明有收益才合并）。
    - PERF-009：ScopeCompatibility 进入 packing key 与成本模型。
    - PERF-010：拆 baseline_duplicate / physical_union / saved / decoded_resident。
    - PERF-011：preferred_representation + consumer_backend_mask（SourceWaveExecutor
      消费，不再固定 pandas）。
    - PERF-012：typed wave 失败恢复（见 ``planner.wave_recovery``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from factor_engine.planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROOT,
    TASK_SOURCE_SCAN,
    source_scope_from_key,
)
from factor_engine.planner.projected_column_footprint import (
    ColumnFootprintProvider,
    ProjectedColumnFootprint,
    union_footprint_bytes,
    union_scan_bytes,
)
from factor_engine.planner.source_representation import (
    SourceRepresentation,
    consumer_backend_mask,
    preferred_representation_for_mask,
)
from factor_engine.planner.wave_recovery import (
    WaveRecoveryCategory,
    WaveRecoveryPlan,
    classify_wave_failure,
    hard_gate_failed_wave_to_n_root_scans,
    recover_wave_failure,
    would_fallback_to_n_root_scans,
)

_DEFAULT_WAVE_MEMORY_BUDGET = 4 * 1024**3  # 4GiB 默认 wave 内存预算
_DEFAULT_ROWS_ESTIMATE = 500_000


# ---------------------------------------------------------------------------
# PERF-009：scope compatibility
# ---------------------------------------------------------------------------


class ScopeCompatibility(Enum):
    """两个读请求 / 两个 wave 的 stock-pool 兼容性（R39-P0-PERF-009）。"""

    EXACT = "EXACT"
    UNION_COMPATIBLE = "UNION_COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"


# ---------------------------------------------------------------------------
# PERF-007：wave 成本模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveCostModel:
    """marginal 评分 / superset coalescing 的成本参数（R39-P0-PERF-007/008）。

    收益 = avoided scan + decode + open + conversion；
    成本 = incremental_live_bytes + memory_rent*lifetime + extra_superset_scan
          + scheduling_delay。所有字段可被测试用合成小数值覆盖。
    """

    open_scan_overhead_bytes: int = 200_000
    decode_per_byte: float = 1.0
    conversion_per_byte: float = 0.5
    scan_per_column_bytes: int = _DEFAULT_ROWS_ESTIMATE * 8
    decode_per_column_bytes: int = _DEFAULT_ROWS_ESTIMATE * 8
    conversion_per_column_bytes: int = (_DEFAULT_ROWS_ESTIMATE * 8) // 2
    memory_rent_per_byte: float = 0.01
    lifetime_factor: float = 1.0
    scheduling_delay_bytes: int = 10_000
    epsilon: float = 1.0
    #: 合并条件 superset < separate * ratio；ratio 越小越难合并。
    min_superset_benefit_ratio: float = 1.0
    #: 有名字 universe 的行数 ≈ 全 A × fraction（无 hierarchy 信息的保守估计）。
    universe_subset_fraction: float = 0.25


# ---------------------------------------------------------------------------
# ReadWave / ReadWavePlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadWave:
    """一个读波：一次 scan 喂多个 task（R27-066 + R39-P0-PERF-005..011）。

    ``source_tasks`` = 真实 SOURCE_SCAN task（唯一贡献列 / IO 的）；
    ``consumer_tasks`` = 由 dependency closure 计算的下游 ROOT / CSE_SHARED
    （零列，不注册 scan request）。``task_ids`` 是两者并集（向后兼容：外部
    构造只给 ``task_ids`` 时视为 source_tasks）。
    """

    wave_id: int
    source_scope: str
    dataset: str
    snapshot_id: str
    columns: frozenset[str]
    time_range: tuple[str, str] | None
    source_tasks: tuple[str, ...] = ()
    consumer_tasks: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    estimated_scan_bytes: int = 0
    estimated_memory_bytes: int = 0
    reuse_density: float = 0.0
    locality_groups: tuple[tuple[str, ...], ...] = ()
    instrument_scope: tuple[str, ...] = ()
    universe_id: str = ""
    # R39
    baseline_duplicate_scan_bytes: int = 0
    physical_union_scan_bytes: int = 0
    saved_scan_bytes: int = 0
    decoded_resident_bytes: int = 0
    superset_coalesce: bool = False
    preferred_representation: str = "pandas_columns"
    consumer_backend_mask: int = 0
    instrument_scope_union_extra_bytes: int = 0
    universe_scope_union_extra_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", frozenset(self.columns))
        object.__setattr__(self, "source_tasks", tuple(sorted(set(self.source_tasks))))
        object.__setattr__(self, "consumer_tasks", tuple(sorted(set(self.consumer_tasks))))
        if self.source_tasks or self.consumer_tasks:
            object.__setattr__(
                self,
                "task_ids",
                tuple(sorted(set(self.source_tasks) | set(self.consumer_tasks))),
            )
        elif self.task_ids:
            # legacy：外部只给 task_ids → 视为 source tasks。
            object.__setattr__(self, "source_tasks", tuple(sorted(set(self.task_ids))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_id": self.wave_id,
            "source_scope": self.source_scope,
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "columns": sorted(self.columns),
            "time_range": list(self.time_range) if self.time_range else None,
            "source_tasks": list(self.source_tasks),
            "consumer_tasks": list(self.consumer_tasks),
            "task_ids": list(self.task_ids),
            "estimated_scan_bytes": self.estimated_scan_bytes,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "reuse_density": round(self.reuse_density, 4),
            "locality_groups": [list(g) for g in self.locality_groups],
            "instrument_scope": list(self.instrument_scope),
            "universe_id": self.universe_id,
            "baseline_duplicate_scan_bytes": self.baseline_duplicate_scan_bytes,
            "physical_union_scan_bytes": self.physical_union_scan_bytes,
            "saved_scan_bytes": self.saved_scan_bytes,
            "decoded_resident_bytes": self.decoded_resident_bytes,
            "superset_coalesce": self.superset_coalesce,
            "preferred_representation": self.preferred_representation,
            "consumer_backend_mask": self.consumer_backend_mask,
            "instrument_scope_union_extra_bytes": self.instrument_scope_union_extra_bytes,
            "universe_scope_union_extra_rows": self.universe_scope_union_extra_rows,
        }


@dataclass
class ReadWavePlan:
    """读波计划：waves + 总览（R27-063/064 + R39-P0-PERF-009/010）。"""

    waves: list[ReadWave] = field(default_factory=list)
    instrument_scope_union_extra_bytes: int = 0
    universe_scope_union_extra_rows: int = 0
    baseline_duplicate_scan_bytes: int = 0
    physical_union_scan_bytes: int = 0
    saved_scan_bytes: int = 0

    @property
    def total_scan_bytes(self) -> int:
        return sum(w.estimated_scan_bytes for w in self.waves)

    @property
    def total_wave_memory_bytes(self) -> int:
        return sum(w.estimated_memory_bytes for w in self.waves)

    @property
    def total_baseline_duplicate_scan_bytes(self) -> int:
        return sum(w.baseline_duplicate_scan_bytes for w in self.waves)

    @property
    def total_physical_union_scan_bytes(self) -> int:
        return sum(w.physical_union_scan_bytes for w in self.waves)

    @property
    def total_saved_scan_bytes(self) -> int:
        return sum(w.saved_scan_bytes for w in self.waves)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_count": len(self.waves),
            "total_scan_bytes": self.total_scan_bytes,
            "total_wave_memory_bytes": self.total_wave_memory_bytes,
            "baseline_duplicate_scan_bytes": self.baseline_duplicate_scan_bytes,
            "physical_union_scan_bytes": self.physical_union_scan_bytes,
            "saved_scan_bytes": self.saved_scan_bytes,
            "instrument_scope_union_extra_bytes": self.instrument_scope_union_extra_bytes,
            "universe_scope_union_extra_rows": self.universe_scope_union_extra_rows,
            "waves": [w.to_dict() for w in self.waves],
        }


# ---------------------------------------------------------------------------
# 基础估算 / 范围工具
# ---------------------------------------------------------------------------


def _default_scan_bytes(columns: Iterable[str], rows_estimate: int = _DEFAULT_ROWS_ESTIMATE) -> int:
    """缺省 scan 字节估算：列数 × 行数 × 8B（manifest 给出前用，R39 仅 last-resort）。"""
    return max(0, len(set(columns)) * max(0, rows_estimate) * 8)


def _default_per_column_bytes(rows_estimate: int) -> int:
    return max(0, rows_estimate) * 8


def _scope_group_key(
    *,
    dataset: str,
    snapshot_id: str | None,
    source_scope: str,
    time_range: tuple[str, str] | None,
) -> str:
    """wave 聚类的 source scope 身份（R27-063/160，向后兼容保留）。"""
    return "::".join(
        [
            dataset,
            snapshot_id or "",
            source_scope or "",
            f"{time_range[0]}~{time_range[1]}" if time_range else "*",
        ]
    )


def _coarse_scope_key(req: "_ReadRequest") -> str:
    """super group：dataset/snapshot/source_scope（不含 time_range / scope cluster）。

    PERF-008/009 在同一 super group 内做跨时间窗 / 跨 scope 的 cost-based 合并。
    """
    return "::".join([req.dataset, req.snapshot_id or "", req.source_scope or ""])


def _scope_cluster_key(req: "_ReadRequest") -> str:
    """EXACT scope 身份：instrument_scope + universe_id（PERF-009 packing key）。"""
    return (
        f"inst={','.join(sorted(req.instrument_scope)) or '*'}"
        f"|univ={req.universe_id or '*'}"
    )


def _time_range_key(time_range: tuple[str, str] | None) -> str:
    return f"{time_range[0]}~{time_range[1]}" if time_range else "*"


def scope_compatibility(a: Any, b: Any) -> ScopeCompatibility:
    """两个请求 / wave 的 stock-pool 兼容性（R39-P0-PERF-009）。

    - universe_id 都非空且不同 → INCOMPATIBLE（未知 hierarchy，保守）。
    - instrument_scope 相同 → EXACT。
    - instrument_scope 相交或一方为空 → UNION_COMPATIBLE。
    - 互斥 → INCOMPATIBLE。
    """
    ua = str(getattr(a, "universe_id", "") or "")
    ub = str(getattr(b, "universe_id", "") or "")
    if ua and ub and ua != ub:
        return ScopeCompatibility.INCOMPATIBLE
    ia = frozenset(getattr(a, "instrument_scope", ()) or ())
    ib = frozenset(getattr(b, "instrument_scope", ()) or ())
    if ia == ib:
        return ScopeCompatibility.EXACT
    if ia and ib and not (ia & ib):
        return ScopeCompatibility.INCOMPATIBLE
    return ScopeCompatibility.UNION_COMPATIBLE


def _range_length_days(time_range: tuple[str, str] | None) -> int | None:
    """时间窗长度（天）。ISO date 可解析则返回天数，否则 None。"""
    if time_range is None:
        return None
    try:
        from datetime import datetime

        d0 = datetime.fromisoformat(time_range[0]).date()
        d1 = datetime.fromisoformat(time_range[1]).date()
        return max(1, (d1 - d0).days)
    except Exception:
        return None


def _union_range(
    a: tuple[str, str] | None,
    b: tuple[str, str] | None,
) -> tuple[str, str] | None:
    if a is None or b is None:
        return None
    if a == b:
        return a
    try:
        return (min(a[0], b[0]), max(a[1], b[1]))
    except Exception:
        return a


def _ranges_overlap(
    a: tuple[str, str] | None,
    b: tuple[str, str] | None,
) -> bool:
    if a is None or b is None:
        return True  # None = full range，视为重叠
    return not (a[1] < b[0] or b[1] < a[0])


def _range_sort_key(time_range: tuple[str, str] | None) -> tuple[str, str]:
    if time_range is None:
        return ("", "")
    return (time_range[0], time_range[1])


# ---------------------------------------------------------------------------
# ReadWavePlanner
# ---------------------------------------------------------------------------


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
    column_footprints: dict[str, ProjectedColumnFootprint] = field(default_factory=dict)
    axis_bytes: int = 0


@dataclass(frozen=True)
class _ConsumerCandidate:
    """ROOT / CSE_SHARED 消费者候选（PERF-005，不注册为 scan request）。"""

    task_id: str
    dataset: str
    snapshot_id: str
    source_scope: str
    time_range: tuple[str, str] | None
    backend: str = "pandas_numpy"


class ReadWavePlanner:
    """按 source scope / 列共享聚波，内存有界（R27-063..065 + R39 §4）。"""

    def __init__(
        self,
        *,
        wave_memory_budget: int = _DEFAULT_WAVE_MEMORY_BUDGET,
        rows_estimate: int = _DEFAULT_ROWS_ESTIMATE,
        axis_bytes: int = 0,
        metadata_bytes: int = 0,
        downstream_live_reserve: int = 0,
        output_reserve: int = 0,
        cost_model: WaveCostModel | None = None,
        universe_rows_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.wave_memory_budget = max(1, wave_memory_budget)
        self.rows_estimate = max(1, rows_estimate)
        self.per_column_bytes = _default_per_column_bytes(self.rows_estimate)
        self.axis_bytes = max(0, int(axis_bytes))
        self.metadata_bytes = max(0, int(metadata_bytes))
        self.downstream_live_reserve = max(0, int(downstream_live_reserve))
        self.output_reserve = max(0, int(output_reserve))
        self.cost_model = cost_model or WaveCostModel()
        self.universe_rows_estimator = universe_rows_estimator or (
            lambda u: (
                self.rows_estimate
                if not u
                else max(1, int(self.rows_estimate * self.cost_model.universe_subset_fraction))
            )
        )
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
        column_footprints: Mapping[str, ProjectedColumnFootprint] | None = None,
        axis_bytes: int = 0,
    ) -> None:
        cols = frozenset(columns)
        footprints = dict(column_footprints or {})
        self._requests.append(
            _ReadRequest(
                task_id=task_id,
                dataset=dataset,
                source_scope=source_scope,
                snapshot_id=snapshot_id,
                time_range=time_range,
                columns=cols,
                estimated_scan_bytes=(
                    estimated_scan_bytes
                    if estimated_scan_bytes is not None
                    else _default_scan_bytes(cols, self.rows_estimate)
                ),
                estimated_memory_bytes=(
                    estimated_memory_bytes
                    if estimated_memory_bytes is not None
                    else self._request_mem_bytes(cols, footprints, axis_bytes)
                ),
                instrument_scope=tuple(instrument_scope or ()),
                universe_id=universe_id or "",
                column_footprints=footprints,
                axis_bytes=axis_bytes or 0,
            )
        )

    def _reset(self) -> None:
        self._requests = []

    # -- footprint 记账（PERF-006） --

    def _union_footprints(self, reqs: list[_ReadRequest]) -> dict[str, ProjectedColumnFootprint]:
        out: dict[str, ProjectedColumnFootprint] = {}
        for r in reqs:
            for c, fp in r.column_footprints.items():
                out.setdefault(c, fp)
        return out

    def _request_mem_bytes(
        self,
        cols: frozenset[str],
        footprints: Mapping[str, ProjectedColumnFootprint],
        axis_bytes: int,
    ) -> int:
        """PERF-006：单 request 内存 = axis + union(decoded) + metadata + reserves。"""
        decoded = union_footprint_bytes(
            footprints, cols, per_column_fallback_bytes=self.per_column_bytes
        )
        axis = max(self.axis_bytes, int(axis_bytes or 0))
        return (
            axis
            + decoded
            + self.metadata_bytes
            + self.downstream_live_reserve
            + self.output_reserve
        )

    def _wave_memory_bytes(self, cols: Iterable[str], reqs: list[_ReadRequest]) -> int:
        """PERF-006：wave 驻留字节 = axis + union(decoded) + metadata + reserves。"""
        footprints = self._union_footprints(reqs)
        decoded = union_footprint_bytes(
            footprints, cols, per_column_fallback_bytes=self.per_column_bytes
        )
        axis = self.axis_bytes
        for r in reqs:
            axis = max(axis, r.axis_bytes)
        return (
            axis
            + decoded
            + self.metadata_bytes
            + self.downstream_live_reserve
            + self.output_reserve
        )

    def _physical_union_scan_bytes(
        self,
        cols: Iterable[str],
        reqs: list[_ReadRequest],
        time_range: tuple[str, str] | None,
    ) -> int:
        """PERF-010：一次 union 扫描的物理字节（IO admission 用）。"""
        footprints = self._union_footprints(reqs)
        scan = union_scan_bytes(
            footprints, cols, per_column_fallback_bytes=self.per_column_bytes
        )
        if scan <= 0:
            scan = _default_scan_bytes(cols, self.rows_estimate)
        return max(0, int(scan))

    # -- plan --

    def plan(self) -> ReadWavePlan:
        """聚波（R27-063/064 + R39-P0-PERF-005..010）。

        流程：
            1. 按 super group（dataset/snapshot/source_scope）分组；
            2. 组内按 EXACT scope cluster → 精确 time_range 子分组，做
               marginal-overlap 贪心（内存有界）；
            3. bounded local-search（PERF-007，合并子集列 wave）；
            4. cost-based superset / scope coalescing（PERF-008/009）。
        """
        requests = list(self._requests)
        self._reset()
        by_super: dict[str, list[_ReadRequest]] = {}
        for req in requests:
            by_super.setdefault(_coarse_scope_key(req), []).append(req)

        packed: list[tuple[ReadWave, list[_ReadRequest]]] = []
        wave_id = 0
        for _key, reqs in sorted(by_super.items()):
            group = self._pack_super_group(wave_id, reqs)
            wave_id += len(group)
            packed.extend(group)

        packed = self._bounded_local_search(packed)
        packed = self._superset_coalesce(packed)

        waves = [w for w, _ in packed]
        plan = ReadWavePlan(waves=waves)
        plan.baseline_duplicate_scan_bytes = sum(
            w.baseline_duplicate_scan_bytes for w in waves
        )
        plan.physical_union_scan_bytes = sum(
            w.physical_union_scan_bytes for w in waves
        )
        plan.saved_scan_bytes = sum(w.saved_scan_bytes for w in waves)
        plan.instrument_scope_union_extra_bytes = sum(
            w.instrument_scope_union_extra_bytes for w in waves
        )
        plan.universe_scope_union_extra_rows = sum(
            w.universe_scope_union_extra_rows for w in waves
        )
        return plan

    def _pack_super_group(
        self, start_id: int, reqs: list[_ReadRequest]
    ) -> list[tuple[ReadWave, list[_ReadRequest]]]:
        """super group 内：EXACT scope cluster → 精确 time_range → marginal 贪心。"""
        by_cluster: dict[str, list[_ReadRequest]] = {}
        for req in reqs:
            by_cluster.setdefault(_scope_cluster_key(req), []).append(req)

        packed: list[tuple[ReadWave, list[_ReadRequest]]] = []
        wid = start_id
        for _ck, cluster in sorted(by_cluster.items()):
            by_tr: dict[str, list[_ReadRequest]] = {}
            for req in cluster:
                by_tr.setdefault(_time_range_key(req.time_range), []).append(req)
            for _tr, tr_reqs in sorted(by_tr.items()):
                remaining = list(tr_reqs)
                while remaining:
                    current: list[_ReadRequest] = []
                    current_cols: set[str] = set()
                    current_mem = 0
                    current_scan = 0
                    while remaining:
                        best_idx = self._best_marginal(remaining, current_cols, current)
                        cand = remaining.pop(best_idx)
                        union_cols = current_cols | cand.columns
                        new_mem = self._wave_memory_bytes(union_cols, current + [cand])
                        if not current and new_mem > self.wave_memory_budget:
                            from factor_engine.runtime.resource_errors import ResourceBudgetExceeded
                            raise ResourceBudgetExceeded(
                                f"atomic read request {cand.task_id!r} requires {new_mem} bytes; "
                                f"read-wave budget is {self.wave_memory_budget} bytes"
                            )
                        if current and new_mem > self.wave_memory_budget:
                            remaining.append(cand)
                            break
                        current.append(cand)
                        current_cols = union_cols
                        current_mem = new_mem
                        current_scan += cand.estimated_scan_bytes
                    if current:
                        wave = self._build_wave(
                            wid,
                            tr_reqs[0].dataset,
                            tr_reqs[0].snapshot_id or "",
                            tr_reqs[0].source_scope,
                            tr_reqs[0].time_range,
                            current,
                            current_cols,
                            current_mem,
                            current_scan,
                        )
                        packed.append((wave, list(current)))
                        wid += 1
        return packed

    def _build_wave(
        self,
        wave_id: int,
        dataset: str,
        snapshot: str,
        source_scope: str,
        time_range: tuple[str, str] | None,
        reqs: list[_ReadRequest],
        cols: set[str],
        union_mem: int,
        total_scan: int,
    ) -> ReadWave:
        baseline = max(0, total_scan)
        physical_union = self._physical_union_scan_bytes(cols, reqs, time_range)
        saved = max(0, baseline - physical_union)
        reuse_density = saved / max(1, union_mem)
        inst = tuple(sorted(set().union(*[set(r.instrument_scope) for r in reqs])))
        univ = ""
        for r in reqs:
            if r.universe_id:
                univ = r.universe_id
                break
        return ReadWave(
            wave_id=wave_id,
            source_scope=source_scope,
            dataset=dataset,
            snapshot_id=snapshot,
            columns=frozenset(cols),
            time_range=time_range,
            source_tasks=tuple(sorted(r.task_id for r in reqs)),
            estimated_scan_bytes=baseline,
            estimated_memory_bytes=union_mem,
            reuse_density=round(reuse_density, 4),
            instrument_scope=inst,
            universe_id=univ,
            baseline_duplicate_scan_bytes=baseline,
            physical_union_scan_bytes=physical_union,
            saved_scan_bytes=saved,
            decoded_resident_bytes=union_mem,
        )

    # -- PERF-007 marginal 评分 --

    def _best_marginal(
        self,
        candidates: list[_ReadRequest],
        current_cols: set[str],
        current_reqs: list[_ReadRequest],
    ) -> int:
        """R39-P0-PERF-007：score = benefit / max(cost, epsilon)。

        - incremental_live_bytes = bytes(union) - bytes(current)（真正边际新增）；
        - benefit = avoided scan + decode + open + conversion；
        - cost = incremental_live_bytes + memory_rent*lifetime + extra_superset_scan
          + scheduling_delay。
        """
        best_idx = 0
        best_score = float("-inf")
        cm = self.cost_model
        # The current wave is invariant while candidates are scored. Rewalking
        # every accepted request twice per candidate makes wide shared-column
        # batches cubic in request count. Preserve first-footprint-wins and
        # max-axis semantics, but summarize the current wave only once.
        current_footprints = self._union_footprints(current_reqs)
        current_axis = max((r.axis_bytes for r in current_reqs), default=self.axis_bytes)
        current_axis = max(self.axis_bytes, current_axis)
        reserves = self.metadata_bytes + self.downstream_live_reserve + self.output_reserve
        current_mem = (current_axis + union_footprint_bytes(
            current_footprints, current_cols, per_column_fallback_bytes=self.per_column_bytes
        ) + reserves) if current_reqs else 0
        for idx, req in enumerate(candidates):
            union_cols = current_cols | req.columns
            footprints = dict(current_footprints)
            for column, footprint in req.column_footprints.items():
                footprints.setdefault(column, footprint)
            union_mem = max(current_axis, req.axis_bytes) + union_footprint_bytes(
                footprints, union_cols, per_column_fallback_bytes=self.per_column_bytes
            ) + reserves
            incremental_live_bytes = max(0, union_mem - current_mem)
            cost = (
                incremental_live_bytes
                + self._memory_rent(incremental_live_bytes)
                + self.cost_model.scheduling_delay_bytes
            )
            if current_cols:
                shared_new = len(req.columns & current_cols)
                benefit = self._pack_benefit(shared_new, req)
            else:
                # 空波起点：最宽列先撑起复用面。
                benefit = self._pack_benefit(0, req) + len(req.columns) * cm.scan_per_column_bytes
            score = benefit / max(cost, cm.epsilon)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _pack_benefit(self, shared_new: int, req: _ReadRequest) -> int:
        """PERF-007 收益：共享列避免的 scan + decode + open + conversion。"""
        cm = self.cost_model
        avoided_scan = shared_new * cm.scan_per_column_bytes
        avoided_decode = shared_new * cm.decode_per_column_bytes
        avoided_conversion = shared_new * cm.conversion_per_column_bytes
        avoided_open = cm.open_scan_overhead_bytes if shared_new > 0 else 0
        return int(avoided_scan + avoided_decode + avoided_conversion + avoided_open)

    def _memory_rent(self, incremental_bytes: int) -> int:
        return int(
            incremental_bytes * self.cost_model.memory_rent_per_byte * self.cost_model.lifetime_factor
        )

    # -- PERF-007 bounded local search --

    def _bounded_local_search(
        self, packed: list[tuple[ReadWave, list[_ReadRequest]]]
    ) -> list[tuple[ReadWave, list[_ReadRequest]]]:
        """一次 bounded 局部搜索：把列是另一 wave 列子集的 wave 并过去（减少 scan）。

        上界：≤ 50 次候选检查；≤ 2 次 swap 后停止。确定性（按插入序扫描）。
        """
        result = list(packed)
        checks = 0
        swaps = 0
        changed = True
        while changed and swaps < 2:
            changed = False
            n = len(result)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    checks += 1
                    if checks > 50:
                        return result
                    wi, ri = result[i]
                    wj, rj = result[j]
                    if not (wi.dataset == wj.dataset and wi.source_scope == wj.source_scope):
                        continue
                    if wi.time_range != wj.time_range:
                        continue
                    if scope_compatibility(wi, wj) is ScopeCompatibility.INCOMPATIBLE:
                        continue
                    if not ri or not wi.columns.issubset(wj.columns):
                        continue
                    merged_reqs = rj + ri
                    union_cols = wj.columns | wi.columns
                    new_mem = self._wave_memory_bytes(union_cols, merged_reqs)
                    if new_mem > self.wave_memory_budget:
                        continue
                    new_wave = self._build_wave(
                        wj.wave_id,
                        wj.dataset,
                        wj.snapshot_id,
                        wj.source_scope,
                        wj.time_range,
                        merged_reqs,
                        set(union_cols),
                        new_mem,
                        sum(r.estimated_scan_bytes for r in merged_reqs),
                    )
                    new_wave = replace(
                        new_wave,
                        superset_coalesce=wj.superset_coalesce or wi.superset_coalesce,
                        preferred_representation=wj.preferred_representation,
                        consumer_backend_mask=wj.consumer_backend_mask,
                        consumer_tasks=tuple(
                            sorted(set(wj.consumer_tasks) | set(wi.consumer_tasks))
                        ),
                    )
                    result[j] = (new_wave, merged_reqs)
                    result.pop(i)
                    swaps += 1
                    changed = True
                    break
                if changed:
                    break
        return result

    # -- PERF-008/009 superset + scope coalescing --

    def _superset_coalesce(
        self, packed: list[tuple[ReadWave, list[_ReadRequest]]]
    ) -> list[tuple[ReadWave, list[_ReadRequest]]]:
        """同 super group 内、时间窗重叠的 wave：cost-based 合并（PERF-008/009）。

        合并条件（PERF-008）：
            superset_extra_scan_cost < duplicate_open_decode_conversion_saved
        UNION_COMPATIBLE scope 额外要求成本模型证明收益（PERF-009）。
        """
        groups: dict[str, list[tuple[ReadWave, list[_ReadRequest]]]] = {}
        for item in packed:
            w, _ = item
            groups.setdefault(f"{w.dataset}::{w.snapshot_id}::{w.source_scope}", []).append(item)

        result: list[tuple[ReadWave, list[_ReadRequest]]] = []
        for _key, items in sorted(groups.items()):
            items = sorted(items, key=lambda it: _range_sort_key(it[0].time_range))
            merged: list[tuple[ReadWave, list[_ReadRequest]]] = []
            for item in items:
                if merged and self._should_coalesce(merged[-1], item):
                    merged[-1] = self._merge_waves(merged[-1], item)
                else:
                    merged.append(item)
            result.extend(merged)
        return result

    def _should_coalesce(
        self,
        a: tuple[ReadWave, list[_ReadRequest]],
        b: tuple[ReadWave, list[_ReadRequest]],
    ) -> bool:
        wa, _ra = a
        wb, _rb = b
        if not (
            wa.dataset == wb.dataset
            and wa.source_scope == wb.source_scope
            and wa.snapshot_id == wb.snapshot_id
        ):
            return False
        compat = scope_compatibility(wa, wb)
        if compat is ScopeCompatibility.INCOMPATIBLE:
            return False
        # The current executor does not pass a physical range/halo handle into
        # the source or trim a superset for each consumer.  Until that contract
        # exists, only identical physical ranges may share a wave.  The union
        # memory and cost gates below still preserve deliberate budget splits.
        if wa.time_range != wb.time_range:
            return False
        # 合并后的 union 内存必须仍在 budget 内（不能 OOM 反噬）。
        merged_reqs = _ra + _rb
        union_mem = self._wave_memory_bytes(wa.columns | wb.columns, merged_reqs)
        if union_mem > self.wave_memory_budget:
            return False
        cm = self.cost_model
        separate = self._separate_scan_cost(wa) + self._separate_scan_cost(wb)
        sup_range = _union_range(wa.time_range, wb.time_range)
        superset = self._superset_scan_cost(wa, wb, sup_range)
        if superset >= separate * cm.min_superset_benefit_ratio:
            return False
        if compat is ScopeCompatibility.UNION_COMPATIBLE:
            extra_bytes, extra_rows = self._scope_merge_metrics(wa, wb)
            saved = max(0, separate - superset)
            if extra_bytes + extra_rows * self.per_column_bytes >= saved:
                return False
        return True

    def _separate_scan_cost(self, w: ReadWave) -> int:
        cm = self.cost_model
        scan = max(0, w.physical_union_scan_bytes) + cm.open_scan_overhead_bytes
        return int(scan * (1 + cm.decode_per_byte + cm.conversion_per_byte))

    def _est_rows_for_range(self, time_range: tuple[str, str] | None) -> int:
        days = _range_length_days(time_range)
        if days is None:
            return self.rows_estimate
        return max(1, int(self.rows_estimate * (days / 365.0)))

    def _superset_scan_cost(
        self,
        w1: ReadWave,
        w2: ReadWave,
        sup_range: tuple[str, str] | None,
    ) -> int:
        """一个 superset scan 的总成本：把较宽 wave 的 physical scan 按行数缩放。"""
        cm = self.cost_model
        rows1 = self._est_rows_for_range(w1.time_range)
        rows2 = self._est_rows_for_range(w2.time_range)
        rows_sup = self._est_rows_for_range(sup_range)
        if rows1 >= rows2:
            base_wave, base_rows = w1, rows1
        else:
            base_wave, base_rows = w2, rows2
        scan = max(0, base_wave.physical_union_scan_bytes) * rows_sup / max(1, base_rows)
        scan = int(scan)
        return int((scan + cm.open_scan_overhead_bytes) * (1 + cm.decode_per_byte + cm.conversion_per_byte))

    def _scope_merge_metrics(
        self, w1: ReadWave, w2: ReadWave
    ) -> tuple[int, int]:
        """PERF-009：scope union 的额外字节 / 额外行（报告指标）。"""
        ia = set(w1.instrument_scope)
        ib = set(w2.instrument_scope)
        extra_inst = max(0, len(ia | ib) - max(len(ia), len(ib)))
        extra_bytes = extra_inst * self.per_column_bytes
        extra_rows = 0
        ua = w1.universe_id
        ub = w2.universe_id
        if ua != ub:
            rows_a = self.universe_rows_estimator(ua)
            rows_b = self.universe_rows_estimator(ub)
            extra_rows = abs(rows_a - rows_b)
        return extra_bytes, extra_rows

    def _merge_waves(
        self,
        a: tuple[ReadWave, list[_ReadRequest]],
        b: tuple[ReadWave, list[_ReadRequest]],
    ) -> tuple[ReadWave, list[_ReadRequest]]:
        wa, ra = a
        wb, rb = b
        cols = wa.columns | wb.columns
        sup_range = _union_range(wa.time_range, wb.time_range)
        reqs = ra + rb
        union_mem = self._wave_memory_bytes(cols, reqs)
        total_scan = sum(r.estimated_scan_bytes for r in reqs)
        wave = self._build_wave(
            wa.wave_id,
            wa.dataset,
            wa.snapshot_id,
            wa.source_scope,
            sup_range,
            reqs,
            set(cols),
            union_mem,
            total_scan,
        )
        extra_bytes, extra_rows = self._scope_merge_metrics(wa, wb)
        wave = replace(
            wave,
            superset_coalesce=True,
            preferred_representation=(
                wa.preferred_representation or wb.preferred_representation
            ),
            consumer_backend_mask=wa.consumer_backend_mask | wb.consumer_backend_mask,
            consumer_tasks=tuple(
                sorted(set(wa.consumer_tasks) | set(wb.consumer_tasks))
            ),
            instrument_scope_union_extra_bytes=(
                wave.instrument_scope_union_extra_bytes + extra_bytes
            ),
            universe_scope_union_extra_rows=(
                wave.universe_scope_union_extra_rows + extra_rows
            ),
        )
        return (wave, reqs)


# ---------------------------------------------------------------------------
# build_waves_from_dag（PERF-005/011 接线）
# ---------------------------------------------------------------------------


def _dataset_from_scope(task: Any) -> str:
    try:
        _s = source_scope_from_key(task.source_scope)
        return _s.dataset or task.source_scope
    except Exception:
        return task.source_scope


def _lookup_footprints(
    footprint_source: Any,
    dataset: str,
    snapshot_id: str | None,
    columns: Iterable[str],
) -> dict[str, ProjectedColumnFootprint]:
    """PERF-006 数据来源：(1) manifest stats / (2) parquet metadata / (3) ScanCost。"""
    if footprint_source is None:
        return {}
    if hasattr(footprint_source, "column_footprints"):
        try:
            result = footprint_source.column_footprints(dataset, snapshot_id, list(columns))
            if isinstance(result, Mapping):
                return {c: fp for c, fp in result.items() if isinstance(fp, ProjectedColumnFootprint)}
        except Exception:
            return {}
    if isinstance(footprint_source, Mapping):
        out: dict[str, ProjectedColumnFootprint] = {}
        for c in columns:
            fp = footprint_source.get((dataset, c))
            if fp is None:
                fp = footprint_source.get(c)
            if isinstance(fp, ProjectedColumnFootprint):
                out[c] = fp
        return out
    return {}


def _task_ancestors(dag: Any) -> dict[str, set[str]]:
    """DAG 每个 task 的传递前驱（inputs 闭包）——消费者归属用。"""
    tasks = getattr(dag, "tasks", {}) or {}
    ancestors: dict[str, set[str]] = {}
    for tid in tasks:
        anc: set[str] = set()
        stack = list(getattr(tasks[tid], "inputs", ()) or ())
        while stack:
            pid = stack.pop()
            if pid in anc:
                continue
            anc.add(pid)
            ptask = tasks.get(pid)
            if ptask is not None:
                stack.extend(getattr(ptask, "inputs", ()) or ())
        ancestors[tid] = anc
    return ancestors


def _attach_consumers(
    waves: list[ReadWave],
    consumer_candidates: list[_ConsumerCandidate],
    ancestors: dict[str, set[str]],
) -> list[ReadWave]:
    """PERF-005：consumer_tasks = dependency closure（与 wave.source_tasks 相交）。"""
    out: list[ReadWave] = []
    for w in waves:
        source_ids = set(w.source_tasks)
        consumers = [
            cc.task_id
            for cc in consumer_candidates
            if source_ids & ancestors.get(cc.task_id, set())
        ]
        out.append(replace(w, consumer_tasks=tuple(sorted(consumers))))
    return out


def _attach_representation(
    waves: list[ReadWave],
    consumer_candidates: list[_ConsumerCandidate],
) -> list[ReadWave]:
    """PERF-011：按下游 backend mask 选 preferred representation。"""
    cc_by_id = {cc.task_id: cc for cc in consumer_candidates}
    out: list[ReadWave] = []
    for w in waves:
        backends = [
            cc_by_id[t].backend for t in w.consumer_tasks if t in cc_by_id
        ]
        mask = consumer_backend_mask(backends)
        rep = preferred_representation_for_mask(mask)
        out.append(
            replace(
                w,
                consumer_backend_mask=mask,
                preferred_representation=rep.value,
            )
        )
    return out


def build_waves_from_dag(
    dag: Any,
    *,
    wave_memory_budget: int = _DEFAULT_WAVE_MEMORY_BUDGET,
    rows_estimate: int = _DEFAULT_ROWS_ESTIMATE,
    scan_cost_map: dict[str, Any] | None = None,
    scope_scan_cost_map: dict[str, Any] | None = None,
    footprint_source: Any = None,
    cost_model: WaveCostModel | None = None,
) -> ReadWavePlan:
    """从 PhysicalFactorDAG 构建读波（PERF-005..011）。

    - 只注册 **SOURCE_SCAN** task 为 scan request（真实列 / IO）；
    - ROOT / CSE_SHARED 成为 ``consumer_tasks``（dependency closure 归属），
      **不**注册零列 scan request（ZERO_COLUMN_READ_WAVE == 0）；
    - footprint 优先级：manifest stats（``footprint_source``）→ ScanCost →
      fallback heuristic。
    """
    planner = ReadWavePlanner(
        wave_memory_budget=wave_memory_budget,
        rows_estimate=rows_estimate,
        cost_model=cost_model,
    )
    scan_cost_map = scan_cost_map or {}
    scope_scan_cost_map = scope_scan_cost_map or {}
    consumer_candidates: list[_ConsumerCandidate] = []
    ancestors = _task_ancestors(dag)

    for task in dag.tasks.values():
        if task.task_type == TASK_SOURCE_SCAN:
            cost = scan_cost_map.get(task.task_id)
            if cost is None:
                cost = scope_scan_cost_map.get(task.source_scope)
            spec = task.source_scan_spec
            columns = tuple(spec.required_columns) if spec is not None else task.required_columns
            dataset = spec.dataset if spec is not None else _dataset_from_scope(task)
            time_range = spec.time_range if spec is not None else task.time_range
            est_scan = cost.selected_bytes if cost is not None and cost.selected_bytes else None
            est_mem = cost.projection_bytes if cost is not None and cost.projection_bytes else None
            footprints = _lookup_footprints(
                footprint_source, dataset, task.source_snapshot_id or None, columns
            )
            axis_bytes = 0
            if spec is not None:
                axis_bytes = getattr(spec, "expected_rows", 0) * 8
            planner.register_scan_task(
                task.task_id,
                dataset=dataset,
                source_scope=task.source_scope,
                snapshot_id=task.source_snapshot_id or None,
                time_range=time_range,
                columns=columns,
                estimated_scan_bytes=est_scan,
                estimated_memory_bytes=est_mem,
                instrument_scope=(
                    spec.instrument_scope if spec is not None else task.instrument_scope
                ),
                universe_id=spec.universe_id if spec is not None else None,
                column_footprints=footprints,
                axis_bytes=axis_bytes,
            )
        elif task.task_type in {TASK_CSE_SHARED, TASK_ROOT}:
            consumer_candidates.append(
                _ConsumerCandidate(
                    task_id=task.task_id,
                    dataset=_dataset_from_scope(task),
                    snapshot_id=task.source_snapshot_id or "",
                    source_scope=task.source_scope,
                    time_range=task.time_range,
                    backend=task.preferred_backend,
                )
            )

    plan = planner.plan()
    if consumer_candidates:
        plan.waves = _attach_consumers(plan.waves, consumer_candidates, ancestors)
        plan.waves = _attach_representation(plan.waves, consumer_candidates)
    return plan


# ---------------------------------------------------------------------------
# R39 hard-gate 判定函数
# ---------------------------------------------------------------------------


def count_zero_column_waves(plan: ReadWavePlan) -> int:
    """hard gate：ZERO_COLUMN_READ_WAVE == 0。"""
    return sum(1 for w in plan.waves if not w.columns)


def non_source_tasks_in_wave_source_tasks(dag: Any, plan: ReadWavePlan) -> int:
    """hard gate：NON_SOURCE_TASK_IN_READ_WAVE_SOURCE_TASKS == 0。"""
    tasks = getattr(dag, "tasks", {}) or {}
    n = 0
    for w in plan.waves:
        for t in w.source_tasks:
            task = tasks.get(t)
            if task is None or task.task_type != TASK_SOURCE_SCAN:
                n += 1
    return n


__all__ = [
    "ReadWave",
    "ReadWavePlan",
    "ReadWavePlanner",
    "ScopeCompatibility",
    "WaveCostModel",
    "build_waves_from_dag",
    "scope_compatibility",
    "count_zero_column_waves",
    "non_source_tasks_in_wave_source_tasks",
    # R39-PERF-012 re-exports（typed recovery）
    "WaveRecoveryCategory",
    "WaveRecoveryPlan",
    "classify_wave_failure",
    "recover_wave_failure",
    "would_fallback_to_n_root_scans",
    "hard_gate_failed_wave_to_n_root_scans",
]
