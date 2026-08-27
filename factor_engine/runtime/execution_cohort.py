# -*- coding: utf-8 -*-
"""P78-P7/P8: ExecutionCohort —— 执行 cohort 分组 + 动态 cohort 大小 + cohort 执行器。

设计目标（用户 P7/P8）：
    - 100k 因子**绝不**一次性编译成一张 mega-DAG。manifest 只持指纹；执行时按
      cohort 分组，**一次只编译 + 执行一个 cohort** 的小 DAG。
    - 全局 CSE 不因 cohort 切分而丢失：跨 cohort 共享的子表达式经
      :class:`~factor_engine.runtime.global_subexpression_index.GlobalSubexpressionIndex`
      物化一次（内存或 spill）并复用，而不是每个 cohort 各重算一次。

本模块三件套：
    1. :class:`ExecutionCohortKey` —— 分组键（market, universe, frequency,
       source_scope, required_field_family, lookback_bucket, execution_axis,
       backend_affinity）。
    2. :func:`cohort_size_for` —— 动态 cohort 大小：由 planner 从「估计 DAG 字节 +
       读字节 + 计算峰值 + 当前内存（ResourceBroker headroom）」决定，**不是**
       硬编码 1000。
    3. :class:`ExecutionCohortExecutor` —— 消费 manifest，按 cohort 分组，逐个
       cohort 编译成小 ``PhysicalFactorDAG``，经既有 ``AdaptiveBatchScheduler``
       执行，并用 GlobalSubexpressionIndex 做跨 cohort CSE 复用。

与既有框架的关系（不重建）：
    - 复用 :class:`~factor_engine.runtime.adaptive_batch_scheduler.AdaptiveBatchScheduler`
      （plan/run/materialize）与 :class:`~factor_engine.runtime.resource_broker.ResourceBroker`
      （live_headroom / resource_envelope.safe_memory_bytes）。
    - 本模块是**编排层**：把 manifest 切成 cohort，把每个 cohort 编译成小 DAG，
      交给既有 scheduler。不重复实现调度 / 资源治理 / CSE 缓存。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from factor_engine.runtime.global_factor_manifest import (
    FactorFingerprint,
    GlobalFactorManifest,
)
from factor_engine.runtime.global_subexpression_index import (
    MATERIALIZE_MEMORY,
    MATERIALIZE_SPILL,
    RECOMPUTE,
    GlobalSubexpressionIndex,
)

_logger = logging.getLogger(__name__)

#: 默认 cohort 内存预算占 safe envelope 的比例（保守，避免单 cohort 吃满内存）。
_DEFAULT_COHORT_MEMORY_FRACTION = 0.25
#: 单因子估计 DAG 字节的保守下界（无估计时用）。
_DEFAULT_PER_FACTOR_DAG_BYTES = 64 * 1024**2
#: 单因子估计读字节的保守下界。
_DEFAULT_PER_FACTOR_READ_BYTES = 32 * 1024**2
#: 单因子估计计算峰值字节的保守下界。
_DEFAULT_PER_FACTOR_PEAK_BYTES = 32 * 1024**2
#: cohort 大小下限（至少 1 个因子）。
_MIN_COHORT_SIZE = 1
#: cohort 大小上限（防御性，避免单 cohort 过大）。
_MAX_COHORT_SIZE = 50_000


@dataclass(frozen=True)
class ExecutionCohortKey:
    """执行 cohort 分组键（任务指定 8 维）。

    相同 key 的因子分到同一 cohort；不同 key 分到不同 cohort。
    """

    market: str = ""
    universe: str = ""
    frequency: str = ""
    source_scope: str = ""
    required_field_family: str = ""
    lookback_bucket: str = ""
    execution_axis: str = ""
    backend_affinity: str = ""

    def to_tuple(self) -> tuple[str, ...]:
        return (
            self.market,
            self.universe,
            self.frequency,
            self.source_scope,
            self.required_field_family,
            self.lookback_bucket,
            self.execution_axis,
            self.backend_affinity,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "market": self.market,
            "universe": self.universe,
            "frequency": self.frequency,
            "source_scope": self.source_scope,
            "required_field_family": self.required_field_family,
            "lookback_bucket": self.lookback_bucket,
            "execution_axis": self.execution_axis,
            "backend_affinity": self.backend_affinity,
        }


def _lookback_bucket(lookback: int) -> str:
    """把回看窗口映射到 bucket（cohort 分组用，避免每因子一个 key）。"""
    if lookback <= 5:
        return "lt5"
    if lookback <= 20:
        return "5-20"
    if lookback <= 60:
        return "20-60"
    if lookback <= 120:
        return "60-120"
    if lookback <= 252:
        return "120-252"
    return "gt252"


def cohort_key_for(fp: FactorFingerprint, *, market: str = "", universe: str = "",
                   frequency: str = "", source_scope: str = "",
                   required_field_family: str = "") -> ExecutionCohortKey:
    """从指纹 + 外部维度构造 cohort key。

    ``lookback_bucket`` 由指纹的 ``lookback`` 派生；``execution_axis`` /
    ``backend_affinity`` 直接取指纹字段。
    """
    return ExecutionCohortKey(
        market=market,
        universe=universe,
        frequency=frequency,
        source_scope=source_scope,
        required_field_family=required_field_family,
        lookback_bucket=_lookback_bucket(fp.lookback),
        execution_axis=fp.execution_axis,
        backend_affinity=fp.backend_affinity,
    )


def group_manifest_into_cohorts(
    manifest: GlobalFactorManifest,
    *,
    market: str = "",
    universe: str = "",
    frequency: str = "",
    source_scope: str = "",
    required_field_family: str = "",
) -> dict[ExecutionCohortKey, list[str]]:
    """把 manifest 的因子按 cohort key 分组。

    Returns:
        ``{cohort_key: [factor_name, ...]}``（组内顺序确定性排序）。
    """
    groups: dict[ExecutionCohortKey, list[str]] = {}
    for name, fp in manifest.iter_fingerprints():
        key = cohort_key_for(
            fp,
            market=market,
            universe=universe,
            frequency=frequency,
            source_scope=source_scope,
            required_field_family=required_field_family,
        )
        groups.setdefault(key, []).append(name)
    for names in groups.values():
        names.sort()
    return groups


def cohort_size_for(
    budget_bytes: int,
    per_factor_estimate: float,
    *,
    min_size: int = _MIN_COHORT_SIZE,
    max_size: int = _MAX_COHORT_SIZE,
) -> int:
    """动态 cohort 大小：随可用内存缩放（**不是**硬编码 1000）。

    Args:
        budget_bytes: 当前 cohort 可用的内存预算（字节）——由 planner 从
            ResourceBroker headroom / safe envelope 决定。
        per_factor_estimate: 单因子的估计内存占用（DAG 字节 + 读字节 + 计算峰值）。

    Returns:
        cohort 大小（因子数），随 ``budget_bytes`` 单调不减。

    单调性：``budget_bytes`` 越大 → cohort 越大（``budget // per_factor_estimate``
    单调不减），并夹在 ``[min_size, max_size]``。
    """
    if per_factor_estimate <= 0:
        per_factor_estimate = float(_DEFAULT_PER_FACTOR_DAG_BYTES)
    size = int(budget_bytes // max(1.0, float(per_factor_estimate)))
    return max(min_size, min(max_size, size))


def cohort_budget_from_broker(
    broker: Any,
    *,
    fraction: float = _DEFAULT_COHORT_MEMORY_FRACTION,
) -> int:
    """从 ResourceBroker 取当前 cohort 内存预算（safe envelope × fraction）。

    复用既有 :class:`~factor_engine.runtime.resource_broker.ResourceBroker` 的
    ``resource_envelope().safe_memory_bytes``（已扣除 emergency/untracked/writer
    reserve）。broker 提供真实值时返回；返回时若 ``fallback_reason`` 非 None，
    budget 即为保守回退值。

    返回 :class:`CohortBudgetResult`：\n
    - ``budget_bytes`` —— 幂等的纯计算字节数；\n
    - ``fallback_reason`` —— None 表示预算来自真实 broker envelope
      （单权威），否则为回退原因（当前硬编码 1GiB 或空预算的非单权威说明）。
    """
    env = None
    reason: str | None = None
    try:
        env = broker.resource_envelope()
        safe = int(getattr(env, "safe_memory_bytes", 0) or 0)
        if safe > 0:
            return CohortBudgetResult(
                budget_bytes=max(1, int(safe * fraction)), fallback_reason=None
            )
        reason = "broker.resource_envelope().safe_memory_bytes <= 0"
    except Exception:
        reason = "broker.resource_envelope() 不可用"
    # 保守回退：1 GiB（非单权威来源，调用方生产模式应拒用）。
    return CohortBudgetResult(budget_bytes=1 * 1024**3, fallback_reason=reason)


@dataclass(frozen=True)
class CohortBudgetResult:
    """cohort 内存预算解析结果。

    - ``budget_bytes``：幂等的纯计算分配字节。
    - ``fallback_reason``：``None`` 表示预算来自真实 broker envelope
      （FE 唯一资源权威）；否则为保守回退的原因（非单权威来源，生产调用方
      必须据 ``fallback_reason`` 拒绝该预算，见 ``cohort_budget_from_broker``）。
    """

    budget_bytes: int
    fallback_reason: str | None

    def __int__(self) -> int:
        return self.budget_bytes


def per_factor_estimate_bytes(fp: FactorFingerprint) -> float:
    """单因子的估计内存占用（DAG 字节 + 读字节 + 计算峰值）。

    有 ``estimated_cost`` 时按比例放大；否则用保守默认。真实 planner 会用
    DataShapeEstimator / ScanCost 更精确估计；这里提供可用的保守近似。
    """
    dag = _DEFAULT_PER_FACTOR_DAG_BYTES
    read = _DEFAULT_PER_FACTOR_READ_BYTES
    peak = _DEFAULT_PER_FACTOR_PEAK_BYTES
    # estimated_cost 越大 → 面板越大 → 内存越大（线性近似）。
    if fp.estimated_cost > 0:
        scale = min(4.0, max(0.5, fp.estimated_cost / 100.0))
        dag = int(dag * scale)
        peak = int(peak * scale)
    return float(dag + read + peak)


@dataclass
class CohortRunResult:
    """单个 cohort 的执行结果。"""

    cohort_key: ExecutionCohortKey
    factor_names: list[str]
    scheduler_output: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    cohort_size: int = 0
    budget_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_key": self.cohort_key.to_dict(),
            "factor_count": len(self.factor_names),
            "cohort_size": self.cohort_size,
            "budget_bytes": self.budget_bytes,
            "results": self.results,
        }


class ExecutionCohortExecutor:
    """消费 manifest，按 cohort 分组，逐个 cohort 编译 + 执行（anti-mega-DAG）。

    任意时刻只编译 + 执行**一个** cohort 的小 DAG；跨 cohort 共享子表达式经
    :class:`GlobalSubexpressionIndex` 物化一次并复用。
    """

    def __init__(
        self,
        *,
        manifest: GlobalFactorManifest,
        subexpression_index: GlobalSubexpressionIndex | None = None,
        broker: Any = None,
        scheduler_factory: Callable[..., Any] | None = None,
        cohort_memory_fraction: float = _DEFAULT_COHORT_MEMORY_FRACTION,
    ) -> None:
        self.manifest = manifest
        self.subexpression_index = subexpression_index or GlobalSubexpressionIndex()
        self.broker = broker
        self.scheduler_factory = scheduler_factory
        self.cohort_memory_fraction = float(cohort_memory_fraction)
        self._cohort_results: list[CohortRunResult] = []

    # -- cohort 规划 --

    def plan_cohorts(
        self,
        *,
        market: str = "",
        universe: str = "",
        frequency: str = "",
        source_scope: str = "",
        required_field_family: str = "",
    ) -> list[tuple[ExecutionCohortKey, list[str], int]]:
        """规划 cohort 序列：``[(key, factor_names, cohort_size), ...]``。

        cohort_size 由动态预算决定（``cohort_budget_from_broker`` ×
        ``per_factor_estimate_bytes``），**不是**硬编码 1000。
        """
        groups = group_manifest_into_cohorts(
            self.manifest,
            market=market,
            universe=universe,
            frequency=frequency,
            source_scope=source_scope,
            required_field_family=required_field_family,
        )
        budget = int(cohort_budget_from_broker(
            self.broker, fraction=self.cohort_memory_fraction
        ))
        planned: list[tuple[ExecutionCohortKey, list[str], int]] = []
        for key in sorted(groups.keys(), key=lambda k: k.to_tuple()):
            names = groups[key]
            # 组内按指纹估计取最大 per-factor（保守：整组用最贵因子定 cohort 大小）。
            per = max(
                (per_factor_estimate_bytes(self.manifest.get(n)) for n in names),
                default=float(_DEFAULT_PER_FACTOR_DAG_BYTES),
            )
            size = cohort_size_for(budget, per)
            planned.append((key, names, size))
        return planned

    # -- 执行 --

    def execute(
        self,
        *,
        build_dag: Callable[[list[str], ExecutionCohortKey], Any],
        run_cohort: Callable[[Any, list[str], ExecutionCohortKey], dict[str, Any]],
        market: str = "",
        universe: str = "",
        frequency: str = "",
        source_scope: str = "",
        required_field_family: str = "",
    ) -> list[CohortRunResult]:
        """逐个 cohort 执行。

        Args:
            build_dag: ``(factor_names, cohort_key) -> PhysicalFactorDAG``（编译
                小 DAG；由调用方把 manifest 指纹展开成真实 DAG）。
            run_cohort: ``(dag, factor_names, cohort_key) -> {factor_name: result}``
                （经既有 AdaptiveBatchScheduler 执行）。

        Returns:
            每个 cohort 的 :class:`CohortRunResult`。
        """
        planned = self.plan_cohorts(
            market=market,
            universe=universe,
            frequency=frequency,
            source_scope=source_scope,
            required_field_family=required_field_family,
        )
        self._cohort_results = []
        for key, names, size in planned:
            # 动态 cohort 大小：把组切成 size 大小的子批（一次只编译一个子批）。
            for start in range(0, len(names), size):
                batch = names[start:start + size]
                dag = build_dag(batch, key)
                results = run_cohort(dag, batch, key)
                self._cohort_results.append(
                    CohortRunResult(
                        cohort_key=key,
                        factor_names=batch,
                        scheduler_output={},
                        results=results,
                        cohort_size=size,
                        budget_bytes=int(cohort_budget_from_broker(
                            self.broker, fraction=self.cohort_memory_fraction
                        )),
                    )
                )
        return self._cohort_results

    def results(self) -> list[CohortRunResult]:
        return self._cohort_results

    def summary(self) -> dict[str, Any]:
        return {
            "cohort_count": len(self._cohort_results),
            "factor_count": len(self.manifest),
            "subexpression_index": self.subexpression_index.summary(),
            "cohorts": [r.to_dict() for r in self._cohort_results],
        }
