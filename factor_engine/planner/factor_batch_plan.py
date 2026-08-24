# -*- coding: utf-8 -*-
"""R30-P0-003: FactorBatchDataPlan + SourceDemandGroup —— 批量因子 source 规划。

.. deprecated:: R32-P0-090
    **DEPRECATED**: FactorBatchPlan 被 :class:`planner.read_wave_planner.ReadWavePlanner`
    替代。ReadWavePlanner 是生产实现的**唯一权威**，提供：

    - 真实物理 footprint optimizer（非固定 500k×8B）
    - Cost-based superset coalescing
    - Typed SourceScopeId / ColumnSourceBinding
    - Wave memory budget 闭环控制
    - Backend-specific representation 选择

    FactorBatchPlan 保留用于**向后兼容旧 batch grouping 逻辑**，新代码应直接使用
    ``ReadWavePlanner`` 或 ``build_waves_from_dag()``。

R30 分工：FE 提供每个因子的 :class:`FactorSourcePlan`（P0-002），本模块把它们
按真实数据需求归组，得到整批的 :class:`SourceDemandGroup` 列表：

- ``SourceDemandGroup``：一个可复用扫描的最小需求单元，``compatibility_key()``
  覆盖除 fields/time_range 之外的全部维度；``mergeable_with()`` 判同组。
- ``FactorBatchDataPlan``：整批汇总（factors / source_groups / canonical_fields
  / time_range / universe / market / experiment snapshot）+ ``group_summary()``。
- ``plan_from_factors(...)``：按 compatibility key 归组，同组 fields 并集；
  内部用稳定 digest 做分组 key（不依赖对象 hash）。

**拒绝错误合并**（这些必须分到不同组，绝不合并）：

- US quarterly 与 TTM（不同 frequency / timeframe）；
- raw 与 backward-adjusted price（不同 price_basis）；
- 不同 universe；
- 不同 decision policy（pit_policy）；
- 不同 security scope。

``batch_data_request(plan)`` 适配到既有 ``planner.batch_data_request`` IR
（懒 import，失败返回 ``None`` —— 并发 session 可能改签名）。

本模块**只做 IR 与分组统计，不读数据、不引入 DuckDB**。对既有 planner 模块
全部防御性 import（try/except + getattr 兜底）。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FactorBatchDataPlan",
    "SourceDemandGroup",
    "batch_data_request",
    "plan_from_factors",
]

# R32-P0-090: Deprecation warning
warnings.warn(
    "FactorBatchPlan is deprecated since R32-P0-090. "
    "Use planner.read_wave_planner.ReadWavePlanner instead for production workloads. "
    "ReadWavePlanner provides true physical footprint optimization, cost-based coalescing, "
    "and typed source bindings.",
    DeprecationWarning,
    stacklevel=2,
)

# 防御性 import：并发 session 可能改 factor_source_plan；不可得时用本地兜底。
try:
    from factor_engine.planner.factor_source_plan import FactorSourcePlan as _FactorSourcePlan
    from factor_engine.planner.factor_source_plan import stable_digest as _stable_digest
except Exception:  # pragma: no cover - 极端 import 环境
    _FactorSourcePlan = None  # type: ignore[assignment]

    def _stable_digest(payload: Any) -> str:  # 本地兜底
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()[:24]


def _fmt_bound(value: Any) -> Any:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _fmt_tr(time_range: Any) -> Any:
    if time_range is None:
        return None
    as_tuple = getattr(time_range, "as_tuple", None)
    if callable(as_tuple):
        try:
            return list(time_range.as_tuple())
        except Exception:  # noqa: BLE001
            return str(time_range)
    if isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        return [_fmt_bound(time_range[0]), _fmt_bound(time_range[1])]
    return str(time_range)


def _sorted_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(sorted(str(v) for v in (values or ())))


# ---------------------------------------------------------------------------
# 分组维度提取（自包含；全部 getattr 兜底）
# ---------------------------------------------------------------------------


def _pit_dict(plan: Any) -> dict[str, Any]:
    pit = getattr(plan, "pit_requirements", None) or {}
    return dict(pit) if isinstance(pit, Mapping) else {}


def _extra(plan: Any) -> dict[str, Any]:
    return dict(getattr(plan, "_extra", None) or {})


def _extract_frequency(plan: Any) -> str:
    freq = getattr(plan, "required_frequency", None)
    if freq:
        return str(freq)
    pit = _pit_dict(plan)
    freq = pit.get("frequency")
    return str(freq) if freq else "daily"


def _extract_timeframe(plan: Any) -> str:
    extra = _extra(plan)
    tf = extra.get("timeframe")
    if tf:
        return str(tf)
    tf = getattr(plan, "timeframe", None)
    if tf:
        return str(tf)
    pit = _pit_dict(plan)
    tf = pit.get("timeframe")
    if tf:
        return str(tf)
    cov = getattr(plan, "coverage_requirements", None) or {}
    if isinstance(cov, Mapping) and cov.get("timeframe"):
        return str(cov["timeframe"])
    grain = getattr(plan, "required_grain", None)
    if isinstance(grain, str) and grain:
        return grain
    return "default"


def _extract_pit_policy(plan: Any) -> str:
    pit = _pit_dict(plan)
    for key in ("policy", "pit_policy", "decision_policy"):
        value = pit.get(key)
        if value:
            return str(value)
    extra = _extra(plan)
    if extra.get("pit_policy"):
        return str(extra["pit_policy"])
    return "none"


def _extract_universe(plan: Any) -> str:
    extra = _extra(plan)
    value = extra.get("universe")
    if value:
        return str(value)
    value = getattr(plan, "universe", None)
    if value:
        return str(value)
    pit = _pit_dict(plan)
    value = pit.get("universe")
    return str(value) if value else ""


def _extract_security_scope(plan: Any) -> str:
    extra = _extra(plan)
    value = extra.get("security_scope")
    if value:
        return str(value)
    value = getattr(plan, "security_scope", None)
    if value:
        return str(value)
    pit = _pit_dict(plan)
    value = pit.get("security_scope")
    return str(value) if value else ""


def _extract_snapshot(plan: Any) -> str:
    extra = _extra(plan)
    value = extra.get("source_snapshot")
    if value:
        return str(value)
    value = getattr(plan, "source_snapshot", None)
    return str(value) if value else ""


def _extract_price_basis(plan: Any) -> str:
    pb = getattr(plan, "price_basis", None)
    if pb:
        return str(pb)
    pit = _pit_dict(plan)
    pb = pit.get("price_basis")
    return str(pb) if pb else "raw"


def _extract_aggregation(plan: Any) -> str:
    aggs = getattr(plan, "aggregations", None) or ()
    if aggs:
        return str(aggs[0])
    extra = _extra(plan)
    return str(extra.get("aggregation_recipe", "") or "")


def _demand_groups_from_plan(
    plan: Any,
) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    """把一个 :class:`FactorSourcePlan` 拆成 (dims, fields) 序列。

    每个 ``source_dataset`` 一个 dims（secondary fundamental / universe 源独立
    成组）；``fields`` = 该 factor 的 ``leaf_concepts``。
    """
    datasets = tuple(getattr(plan, "source_datasets", None) or ())
    if not datasets:
        datasets = ("default",)
    base: dict[str, Any] = {
        "market": str(getattr(plan, "market", "") or ""),
        "source_snapshot": _extract_snapshot(plan),
        "frequency": _extract_frequency(plan),
        "timeframe": _extract_timeframe(plan),
        "pit_policy": _extract_pit_policy(plan),
        "universe": _extract_universe(plan),
        "security_scope": _extract_security_scope(plan),
        "price_basis": _extract_price_basis(plan),
        "aggregation_recipe": _extract_aggregation(plan),
        "filters": _sorted_tuple(getattr(plan, "filters", None) or ()),
    }
    fields = tuple(str(c) for c in (getattr(plan, "leaf_concepts", None) or ()))
    out: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for ds in datasets:
        dims = dict(base, dataset=str(ds))
        out.append((dims, fields))
    return out


def _dims_digest(dims: Mapping[str, Any]) -> str:
    """分组 key 的稳定 digest：集合类值先排序（同一语义 → 同一 key）。"""
    normalized: dict[str, Any] = {}
    for k, v in dims.items():
        if isinstance(v, (tuple, list, set, frozenset)):
            normalized[str(k)] = tuple(sorted(str(x) for x in v))
        else:
            normalized[str(k)] = str(v)
    return _stable_digest(normalized)


def _infer_market(source_plans: Sequence[Any]) -> str:
    for plan in source_plans:
        m = getattr(plan, "market", None)
        if m:
            return str(m)
    return ""


def _canonical_fields(
    source_plans: Sequence[Any],
    source_groups: Sequence["SourceDemandGroup"],
) -> dict[str, list[str]]:
    """按 (dataset, price_basis, timeframe) 折叠字段并集（排序去重）。"""
    buckets: dict[str, set[str]] = {}
    for g in source_groups:
        key = f"{g.dataset}::{g.price_basis}::{g.timeframe}"
        buckets.setdefault(key, set()).update(g.fields or ())
    return {k: sorted(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


@dataclass
class SourceDemandGroup:
    """一个可复用扫描的最小需求单元。

    ``compatibility_key()`` 覆盖除 fields/time_range 外的全部维度；两个 group
    仅当 ``compatibility_key()`` 相等才可合并（``mergeable_with``）。
    """

    dataset: str
    market: str = ""
    source_snapshot: str = ""
    frequency: str = "daily"
    timeframe: str = "default"
    pit_policy: str = "none"
    universe: str = ""
    security_scope: str = ""
    price_basis: str = "raw"
    aggregation_recipe: str = ""
    filters: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    time_range: Any = None
    instruments: tuple[str, ...] = ()

    def compatibility_key(self) -> tuple[Any, ...]:
        """除 fields/time_range 外的全部维度（全可哈希）。"""
        return (
            str(self.dataset),
            str(self.market),
            str(self.source_snapshot),
            str(self.frequency),
            str(self.timeframe),
            str(self.pit_policy),
            str(self.universe),
            str(self.security_scope),
            str(self.price_basis),
            str(self.aggregation_recipe),
            _sorted_tuple(self.filters),
        )

    def mergeable_with(self, other: "SourceDemandGroup") -> bool:
        return self.compatibility_key() == other.compatibility_key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "market": self.market,
            "source_snapshot": self.source_snapshot,
            "frequency": self.frequency,
            "timeframe": self.timeframe,
            "pit_policy": self.pit_policy,
            "universe": self.universe,
            "security_scope": self.security_scope,
            "price_basis": self.price_basis,
            "aggregation_recipe": self.aggregation_recipe,
            "filters": list(self.filters or ()),
            "fields": list(self.fields or ()),
            "n_fields": len(self.fields or ()),
            "time_range": _fmt_tr(self.time_range),
            "instruments": list(self.instruments or ()),
        }


@dataclass
class FactorBatchDataPlan:
    """整批因子合并后的 source 规划（R30-P0-003）。"""

    factors: tuple[Any, ...] = ()
    source_groups: tuple[SourceDemandGroup, ...] = ()
    canonical_fields: dict[str, list[str]] = field(default_factory=dict)
    time_range: Any = None
    universe_id: str = ""
    market: str = ""
    experiment_snapshot_id: str = ""

    def group_summary(self) -> dict[str, Any]:
        """批量归组统计。``source_group_count << factor_count`` 由调用方判。"""
        factor_count = len(self.factors)
        concepts: set[str] = set()
        for plan in self.factors:
            concepts.update(
                str(c) for c in (getattr(plan, "leaf_concepts", None) or ())
            )
        source_group_count = len(self.source_groups)
        total_fields = sum(len(g.fields or ()) for g in self.source_groups)
        avg = (
            round(total_fields / source_group_count, 2)
            if source_group_count
            else 0.0
        )
        return {
            "factor_count": factor_count,
            "unique_concepts": len(concepts),
            "source_group_count": source_group_count,
            "physical_scan_count": source_group_count,  # 合理估计 = 组数
            "avg_fields_per_group": avg,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": [
                str(getattr(f, "factor_id", f)) for f in self.factors
            ],
            "source_groups": [g.to_dict() for g in self.source_groups],
            "canonical_fields": dict(self.canonical_fields),
            "time_range": _fmt_tr(self.time_range),
            "universe_id": self.universe_id,
            "market": self.market,
            "experiment_snapshot_id": self.experiment_snapshot_id,
            "summary": self.group_summary(),
        }


def plan_from_factors(
    source_plans: Sequence[Any],
    time_range: Any = None,
    universe_id: str = "",
    market: str = "",
    experiment_snapshot_id: str | None = None,
) -> FactorBatchDataPlan:
    """按 compatibility key 归组整批因子。

    同组 fields 并集；**不同 frequency/timeframe、price_basis、universe、
    pit_policy、security scope 自动分到不同组**（compatibility key 全等才合并）。
    分组 key 用稳定 digest，不依赖对象 hash。
    """
    source_plans = tuple(source_plans)
    market = str(market or "") or _infer_market(source_plans)

    buckets: dict[str, dict[str, Any]] = {}
    for plan in source_plans:
        for dims, fields in _demand_groups_from_plan(plan):
            key = _dims_digest(dims)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "dims": dims,
                    "fields": set(),
                    "factor_ids": set(),
                    "instruments": tuple(),
                }
                buckets[key] = bucket
            bucket["fields"].update(fields or ())
            bucket["factor_ids"].add(str(getattr(plan, "factor_id", plan)))

    source_groups: list[SourceDemandGroup] = []
    for key, bucket in buckets.items():  # key 不参与业务，只作去重
        dims = bucket["dims"]
        source_groups.append(
            SourceDemandGroup(
                dataset=str(dims["dataset"]),
                market=str(dims["market"]),
                source_snapshot=str(dims["source_snapshot"]),
                frequency=str(dims["frequency"]),
                timeframe=str(dims["timeframe"]),
                pit_policy=str(dims["pit_policy"]),
                universe=str(dims["universe"]),
                security_scope=str(dims["security_scope"]),
                price_basis=str(dims["price_basis"]),
                aggregation_recipe=str(dims["aggregation_recipe"]),
                filters=tuple(dims["filters"] or ()),
                fields=tuple(sorted(bucket["fields"])),
                time_range=time_range,
                instruments=tuple(dims.get("instruments", ()) or ()),
            )
        )

    canonical_fields = _canonical_fields(source_plans, source_groups)
    return FactorBatchDataPlan(
        factors=source_plans,
        source_groups=tuple(source_groups),
        canonical_fields=canonical_fields,
        time_range=time_range,
        universe_id=str(universe_id or ""),
        market=market,
        experiment_snapshot_id=str(experiment_snapshot_id or ""),
    )


# ---------------------------------------------------------------------------
# 适配既有 BatchDataRequest（R31/R39 IR）——懒 import，失败返回 None
# ---------------------------------------------------------------------------


def _as_time_range(time_range: Any) -> Any:
    """把 plan 的 time_range 适配为既有 typed ``TimeRange``（失败返回 None）。"""
    try:
        from factor_engine.planner.source_binding import TimeRange
    except Exception:  # pragma: no cover
        return None
    if isinstance(time_range, TimeRange):
        return time_range
    if time_range is None:
        return None
    if isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        try:
            return TimeRange(start=time_range[0], end=time_range[1])
        except Exception:  # pragma: no cover
            return None
    return None


def batch_data_request(plan: Any) -> Any | None:
    """把 :class:`FactorBatchDataPlan` 适配为既有 ``BatchDataRequest``。

    懒 import + 全防御：任何一步不可用/签名变化 → 返回 ``None``（调用方按
    无成本规划降级），绝不抛异常。
    """
    try:
        from factor_engine.planner.batch_data_request import BatchDataRequest, SourceScanGroup
        from factor_engine.planner.physical_factor_dag import SourceScopeId
    except Exception:  # pragma: no cover
        return None
    if plan is None:
        return None
    groups = []
    for i, g in enumerate(getattr(plan, "source_groups", None) or ()):
        scope = SourceScopeId(
            dataset=str(getattr(g, "dataset", "") or ""),
            snapshot_id=str(getattr(g, "source_snapshot", "") or ""),
            market=str(getattr(g, "market", "") or ""),
        )
        try:
            groups.append(
                SourceScanGroup(
                    group_id=i,
                    dataset=str(getattr(g, "dataset", "") or ""),
                    source_scope=scope,
                    snapshot_id=str(getattr(g, "source_snapshot", "") or ""),
                    fields=tuple(getattr(g, "fields", None) or ()),
                    time_range=_as_time_range(getattr(plan, "time_range", None)),
                    instrument_scope=tuple(getattr(g, "instruments", None) or ())
                    or None,
                    scan_cost=None,
                )
            )
        except Exception:  # pragma: no cover - 并发改签名
            return None
    all_fields = tuple(
        sorted({f for g in groups for f in g.fields})
    )
    try:
        return BatchDataRequest(fields=all_fields, groups=groups)
    except Exception:  # pragma: no cover
        return None
