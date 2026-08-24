"""data_access.read.data_read_identity —— R21 DataReadIdentity（DA-P0-01..04 修复）。

DataAccess reads were not bound to a stable identity: a read's provenance was
only the ``DataSnapshot`` (data facts) + ``ReadLineage`` (dataset/columns/range),
with no binding to availability / mining / unit / PIT / calendar / universe /
source snapshot / grain / session / decision clock. Two reads of the same
dataset with different PIT visibility, a different calendar world, or a
different universe were indistinguishable.

This module introduces an immutable ``DataReadIdentity`` that binds the full
read context, plus a fail-closed validator for production SemanticField reads
(availability / mining / unit / PIT must be declared before a read is allowed).

DA-P0-01..04 red-team fixes (R21-DA-IDENTITY-FIX):

1. **DA-P0-01** —— 原 ``_availability_grain(fields)`` 在循环里覆盖 per-field
   availability/grain：多字段读（price same_day、roe next_trading_day、
   announcement knowledge_time）塌缩成单一 availability，且依赖字段顺序。
   现在每个字段编译成 ``ResolvedFieldIdentity``（logical_name / physical_name /
   dataset / market / availability / temporal_model / event_time / knowledge_time /
   grain / source_unit / canonical_unit / scale / pit_fidelity /
   revision_policy），**按逻辑名规范排序**后整体 hash——混合 availability 语义
   全量保留，字段顺序无关。

2. **DA-P0-02** —— ``digest`` 原为 caller 可覆盖（``if not self.digest`` 才重算，
   ``DataReadIdentity(..., digest="whatever")`` 被直接接受且不校验）。现在
   ``digest=init=False``：始终内部推导；``from_dict`` 反序列化时重算并校验
   传入 digest 必须一致。

3. **DA-P0-03** —— provenance 原 fail-open：``_revision_of()`` 出错返回 None、
   ``_universe_snapshot_id()`` except 返回 None、calendar 可能 None、source
   snapshot 可能缺失。现在 production/PIT-strict 下**必须 fail closed**：
   revision 不可得 → raise；calendar 不可得 → raise；required universe 不可得
   → raise；物理 source snapshot 不可得 → raise。research 允许显式 UNKNOWN
   状态（``provenance_status`` 明确标注）。

4. **DA-P0-04** —— 原 digest 含 session/request_id：同一份数据/字段/时间/
   universe/PIT/source 今天读与明天读身份不同，破坏缓存复用/可复现。现在
   **拆分**：
     - ``DataReadContentIdentity``：dataset、fields(ResolvedFieldIdentity[])、
       revision、source_snapshot、calendar、universe、PIT、time_range、
       instrument_filter、columns —— 内容 hash **不含执行身份**；
     - ``ReadExecutionIdentity``：request_id、session、user/executor、timestamp、
       trace_id —— 独立记录，不进内容 hash。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_access.core.exceptions import ValidationError
from data_access.r30._shared import stable_digest_full
from data_access.r30.calendar_snapshot import canonical


# ---------------------------------------------------------------------------
# DA-P0-01：ResolvedFieldIdentity —— 每个字段的完整语义身份。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedFieldIdentity:
    """一个已解析字段的完整语义身份（DA-P0-01）。

    携带 availability / temporal_model / event_time / knowledge_time / grain /
    unit / scale / pit_fidelity / revision_policy 等全部语义维度。多字段读按
    逻辑名排序后整体 hash，混合 availability（same_day + next_trading_day +
    financial PIT）全量保留，顺序无关。
    """

    logical_name: str
    physical_name: str | None = None
    dataset: str | None = None
    market: str | None = None
    availability: str | None = None
    temporal_model: str | None = None
    event_time: str | None = None
    knowledge_time: str | None = None
    grain: str | None = None
    source_unit: str | None = None
    canonical_unit: str | None = None
    scale: float | None = None
    pit_fidelity: str | None = None
    revision_policy: str | None = None

    def _tuple(self) -> tuple[str, ...]:
        return (
            str(self.logical_name),
            canonical(self.physical_name),
            canonical(self.dataset),
            canonical(self.market),
            canonical(self.availability),
            canonical(self.temporal_model),
            canonical(self.event_time),
            canonical(self.knowledge_time),
            canonical(self.grain),
            canonical(self.source_unit),
            canonical(self.canonical_unit),
            canonical(self.scale),
            canonical(self.pit_fidelity),
            canonical(self.revision_policy),
        )

    @property
    def content_canonical(self) -> str:
        """整字段的确定性标量（进内容 hash 前的最后一道折叠）。"""
        return "|".join(self._tuple())

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "physical_name": self.physical_name,
            "dataset": self.dataset,
            "market": self.market,
            "availability": self.availability,
            "temporal_model": self.temporal_model,
            "event_time": self.event_time,
            "knowledge_time": self.knowledge_time,
            "grain": self.grain,
            "source_unit": self.source_unit,
            "canonical_unit": self.canonical_unit,
            "scale": self.scale,
            "pit_fidelity": self.pit_fidelity,
            "revision_policy": self.revision_policy,
        }


def _field_event_time(f: Any) -> str | None:
    """字段的 event_time：只接受显式时间列（dtype 像时间或物理名含 time/date）。

    #P0-38：值字段的 time_role=event_time 只是行级标注，不是时间轴列——绝不从
    值字段推导 event_time。``knowledge_time`` 是**可见时间**（数据发布时刻），
    可以安全取（与 calendar/PIT 语义同源）。
    """
    ev = getattr(f, "event_time", None)
    if ev:
        return str(ev)
    dtype = str(getattr(f, "dtype", "") or "").lower()
    if any(tok in dtype for tok in ("time", "date")):
        role = getattr(f, "time_role", None)
        if role in (None, "event_time"):
            name = getattr(f, "physical_name", None) or getattr(f, "logical_name", None)
            if name and any(tok in str(name).lower() for tok in ("time", "date")):
                return str(name)
    return None


def _field_revision_policy(f: Any) -> str | None:
    """字段的 revision 策略。

    R45 closure：字段既无 ``revision_policy`` 也无 ``duplicate_policy`` 时，**不再
    乐观缺省 ``latest_revision``**。对 A 股财务，``latest_revision`` = 仓库里现在
    的值，**不是**历史上市场可知的值——把它当默认会掩盖 PIT 前视。因此缺省改为
    ``UNKNOWN``，并要求 production PIT 字段显式声明（见
    ``assert_semantic_field_production_ready``）。research 非 PIT 读若确实要用
    latest，由调用方显式声明，不在这里静默注入。
    """
    rp = getattr(f, "revision_policy", None)
    if rp:
        return str(rp)
    dp = getattr(f, "duplicate_policy", None)
    if dp:
        return str(dp)
    return "UNKNOWN"


def _field_identity(f: Any) -> ResolvedFieldIdentity:
    """把任意字段对象（SemanticField 或带属性的占位对象）编译成字段身份。"""
    return ResolvedFieldIdentity(
        logical_name=str(getattr(f, "logical_name", None) or "?"),
        physical_name=(
            str(getattr(f, "physical_name", None))
            if getattr(f, "physical_name", None) is not None
            else None
        ),
        dataset=(
            str(getattr(f, "dataset", None))
            if getattr(f, "dataset", None) is not None
            else None
        ),
        market=(
            str(getattr(f, "market", None))
            if getattr(f, "market", None) is not None
            else None
        ),
        availability=(
            str(getattr(f, "availability", None))
            if getattr(f, "availability", None) not in (None, "", "unknown")
            else None
        ),
        temporal_model=(
            str(getattr(f, "temporal_model", None))
            if getattr(f, "temporal_model", None)
            else None
        ),
        event_time=_field_event_time(f),
        knowledge_time=(
            str(getattr(f, "knowledge_time", None))
            if getattr(f, "knowledge_time", None)
            else None
        ),
        grain=(
            str(getattr(f, "grain", None))
            if getattr(f, "grain", None)
            else None
        ),
        source_unit=(
            str(getattr(f, "source_unit", None))
            if getattr(f, "source_unit", None)
            else None
        ),
        canonical_unit=(
            str(getattr(f, "canonical_unit", None))
            if getattr(f, "canonical_unit", None)
            else None
        ),
        scale=(
            float(getattr(f, "scale", None))
            if getattr(f, "scale", None) is not None
            else None
        ),
        pit_fidelity=(
            str(getattr(f, "pit_fidelity", None))
            if getattr(f, "pit_fidelity", None)
            else None
        ),
        revision_policy=_field_revision_policy(f),
    )


def _sorted_field_identities(
    fields: Iterable[Any] | None,
) -> tuple[ResolvedFieldIdentity, ...]:
    """字段身份集合：按逻辑名规范排序（顺序无关），tuple 冻结。"""
    if not fields:
        return ()
    out = [_field_identity(f) for f in fields]
    out.sort(key=lambda r: (str(r.logical_name), str(r.physical_name)))
    return tuple(out)


def _pit_relevant(fields: tuple[ResolvedFieldIdentity, ...]) -> bool:
    """R45：这些字段的可见行集是否受 decision_clock / availability_cutoff 影响。

    只要任一字段带 PIT 语义（availability 非 same_day、或 pit_fidelity 非
    effective_only/unsupported、或 temporal_model 是 financial/event），decision
    clock 就可能改变可见行 → 必须进内容身份 hash。纯 same_day 面板字段不受
    decision clock 影响，不进内容 hash（保持缓存复用）。
    """
    for f in fields:
        if f.availability not in (None, "same_day"):
            return True
        if f.pit_fidelity not in (None, "effective_only", "unsupported"):
            return True
        tm = (f.temporal_model or "").lower()
        if any(tok in tm for tok in ("financial", "event")):
            return True
    return False


# ---------------------------------------------------------------------------
# DA-P0-04：内容身份 vs 执行身份。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataReadContentIdentity:
    """一次读取的**内容**身份（DA-P0-04）——不含任何执行上下文。

    覆盖：dataset、fields（ResolvedFieldIdentity[]）、revision、source snapshot、
    calendar、universe、PIT、time_range、instrument_filter、columns。
    同一份数据/字段/时间/universe/PIT/source 无论何时、由谁、以什么 request_id
    读取，内容 hash 都相同 → 缓存复用/可复现性成立。
    """

    dataset: str
    revision: str | None = None
    # R45 DEPRECATED：单值 availability 只用于向后兼容/单字段读的首 availability。
    # production 消费方必须用 per-field ``ResolvedFieldIdentity.availability``
    # （``fields`` 里的 FieldReadPolicySetIdentity），不得消费本单值。
    availability: str = "same_day"
    calendar_identity: str | None = None
    universe_snapshot: str | None = None
    source_snapshot: str | None = None
    # R45 DEPRECATED：单值 grain 同理，production 用 per-field grain。
    grain: str | None = None
    columns: tuple[str, ...] | None = None
    time_range: tuple[str | None, str | None] | None = None
    instrument_filter: tuple[str, ...] | None = None
    fields: tuple[ResolvedFieldIdentity, ...] = ()
    # R45：decision_clock / availability_cutoff 在**能改变可见行集**（PIT /
    # availability cutoff）时进入内容身份 hash——否则只进执行身份。见
    # ``_pit_relevant``。
    decision_clock: str | None = None
    availability_cutoff: str | None = None
    digest: str = field(init=False)
    # provenance 状态：production 下必须是 available；research 下 UNKNOWN 显式记录。
    provenance_status: str = "available"      # available / unknown
    provenance_notes: tuple[str, ...] = ()
    _schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValidationError("DataReadContentIdentity.dataset is required")
        if isinstance(self.columns, list):
            object.__setattr__(self, "columns", tuple(self.columns))
        if isinstance(self.instrument_filter, list):
            object.__setattr__(self, "instrument_filter", tuple(self.instrument_filter))
        if isinstance(self.time_range, list):
            object.__setattr__(self, "time_range", _stable_range(self.time_range))
        elif self.time_range is not None:
            object.__setattr__(self, "time_range", _stable_range(self.time_range))
        if isinstance(self.fields, list):
            object.__setattr__(self, "fields", tuple(self.fields))
        if isinstance(self.provenance_notes, list):
            object.__setattr__(self, "provenance_notes", tuple(self.provenance_notes))
        # DA-P0-01：fields 规范化成 ResolvedFieldIdentity[]（SemanticField /
        # ResolvedFieldIdentity 均可），按逻辑名排序（顺序无关）。
        if self.fields:
            normalized = _sorted_field_identities(self.fields)
            if normalized != tuple(self.fields):
                object.__setattr__(self, "fields", normalized)
        if not self.revision:
            object.__setattr__(self, "revision", None)
        if not self.calendar_identity:
            object.__setattr__(self, "calendar_identity", None)
        if not self.universe_snapshot:
            object.__setattr__(self, "universe_snapshot", None)
        if not self.source_snapshot:
            object.__setattr__(self, "source_snapshot", None)
        if not self.grain:
            object.__setattr__(self, "grain", None)
        if not self.availability:
            object.__setattr__(self, "availability", "same_day")
        if self.provenance_status not in ("available", "unknown"):
            raise ValidationError(
                f"provenance_status 必须是 available/unknown，收到 {self.provenance_status!r}"
            )
        # R45：decision_clock / availability_cutoff 只有能改变可见行集（PIT /
        # availability cutoff）时才进内容身份 hash。纯执行语义（如仅用于审计的
        # decision_clock 快照）不进内容身份——此时字段置 None，保证非 PIT 读的
        # 内容身份跨 decision_clock 相同（DA-P0-04 执行身份独立）。
        pit_relevant = _pit_relevant(self.fields)
        dc = self.decision_clock if pit_relevant else None
        ac = self.availability_cutoff if pit_relevant else None
        object.__setattr__(self, "decision_clock", dc)
        object.__setattr__(self, "availability_cutoff", ac)
        # DA-P0-02：digest 始终内部推导，绝不接受 caller 传入值。
        object.__setattr__(
            self,
            "digest",
            stable_digest_full(
                canonical("data_access.DataReadContentIdentity"),
                canonical(self._schema_version),
                canonical(self.dataset),
                canonical(self.revision),
                canonical(self.availability),
                canonical(self.calendar_identity),
                canonical(self.universe_snapshot),
                canonical(self.source_snapshot),
                canonical(self.grain),
                canonical(self.columns),
                canonical(self.time_range),
                canonical(self.instrument_filter),
                canonical(tuple(f.content_canonical for f in self.fields)),
                canonical(dc),
                canonical(ac),
                canonical(self.provenance_status),
                canonical(tuple(self.provenance_notes)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "availability": self.availability,
            "calendar_identity": self.calendar_identity,
            "universe_snapshot": self.universe_snapshot,
            "source_snapshot": self.source_snapshot,
            "grain": self.grain,
            "columns": self.columns,
            "time_range": self.time_range,
            "instrument_filter": self.instrument_filter,
            "fields": [f.to_dict() for f in self.fields],
            "digest": self.digest,
            "decision_clock": self.decision_clock,
            "availability_cutoff": self.availability_cutoff,
            "provenance_status": self.provenance_status,
            "provenance_notes": list(self.provenance_notes),
            "_schema_version": self._schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DataReadContentIdentity":
        """反序列化：重算 digest 并校验传入 digest 必须一致（DA-P0-02）。"""
        data = dict(raw)
        supplied = data.get("digest")
        data.pop("digest", None)
        if "fields" in data and data["fields"] is not None:
            data["fields"] = [
                ResolvedFieldIdentity(**dict(fd)) for fd in data["fields"]
            ]
        obj = cls(**data)
        if supplied is not None and supplied != obj.digest:
            raise ValidationError(
                f"DataReadContentIdentity digest 不匹配（DA-P0-02）："
                f"供给 {supplied!r} != 重算 {obj.digest!r} —— digest 只能由内部推导。"
            )
        return obj


@dataclass(frozen=True)
class ReadExecutionIdentity:
    """一次读取的**执行**身份（DA-P0-04）——绝不进内容 hash。

    记录 request_id / session / user(executor) / timestamp / trace_id，独立
    于 ``DataReadContentIdentity`` 保存，用于审计/追踪；同一内容在不同
    request_id/session 下执行身份不同但内容身份相同。
    """

    request_id: str | None = None
    session: str | None = None
    executor: str | None = None
    timestamp: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session": self.session,
            "executor": self.executor,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class DataReadIdentity:
    """Immutable identity of one DataAccess read（DA-P0-01..04 修复后）。

    Binds the full read context so two reads that differ in any of these
    dimensions are provably different:

        - ``dataset`` / ``revision``      data + version
        - ``fields`` / ``availability``   per-field semantic visibility +
                                          row granularity（DA-P0-01：混合语义全保留）
        - ``calendar_identity``           PIT world (frozen calendar snapshot)
        - ``universe_snapshot``           stock pool identity
        - ``source_snapshot``             exact physical source snapshot digest
        - ``content``                     ``DataReadContentIdentity``（不含执行身份）

    ``digest``（内容 hash）只来自 ``content``（DA-P0-04）；执行上下文
    （request_id/session/executor/timestamp/trace_id）放在 ``execution``，
    不进 digest。``digest`` 由内部推导（DA-P0-02），caller 传入任何值都被忽略。
    """

    dataset: str
    revision: str | None = None
    # R45 DEPRECATED：单值 availability/grain 只向后兼容；production 用 per-field。
    availability: str = "same_day"
    calendar_identity: str | None = None
    universe_snapshot: str | None = None
    source_snapshot: str | None = None
    grain: str | None = None
    session: str | None = None
    decision_clock: str | None = None
    # R45：availability_cutoff（PIT 可见性截止）在能改变可见行集时进内容身份。
    availability_cutoff: str | None = None
    columns: tuple[str, ...] | None = None
    time_range: tuple[str | None, str | None] | None = None
    instrument_filter: tuple[str, ...] | None = None
    fields: tuple[ResolvedFieldIdentity, ...] = ()
    provenance_status: str = "available"      # available / unknown（DA-P0-03）
    provenance_notes: tuple[str, ...] = ()
    digest: str = field(init=False, default="")
    content: DataReadContentIdentity | None = field(init=False, default=None)
    execution: ReadExecutionIdentity | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValidationError("DataReadIdentity.dataset is required")
        if isinstance(self.columns, list):
            object.__setattr__(self, "columns", tuple(self.columns))
        if isinstance(self.instrument_filter, list):
            object.__setattr__(self, "instrument_filter", tuple(self.instrument_filter))
        if isinstance(self.time_range, list):
            object.__setattr__(self, "time_range", _stable_range(self.time_range))
        elif self.time_range is not None:
            object.__setattr__(self, "time_range", _stable_range(self.time_range))
        if isinstance(self.fields, list):
            object.__setattr__(self, "fields", tuple(self.fields))

        # ---- DA-P0-02：digest 始终内部推导，忽略任何 caller 传入值。 ----
        if isinstance(self.fields, list):
            object.__setattr__(self, "fields", tuple(self.fields))
        if self.fields:
            normalized = _sorted_field_identities(self.fields)
            if normalized != tuple(self.fields):
                object.__setattr__(self, "fields", normalized)

        # ---- DA-P0-02：digest 始终内部推导，忽略任何 caller 传入值。 ----
        content = DataReadContentIdentity(
            dataset=self.dataset,
            revision=self.revision or None,
            availability=self.availability or "same_day",
            calendar_identity=self.calendar_identity or None,
            universe_snapshot=self.universe_snapshot or None,
            source_snapshot=self.source_snapshot or None,
            grain=self.grain or None,
            columns=self.columns,
            time_range=self.time_range,
            instrument_filter=self.instrument_filter,
            fields=self.fields,
            decision_clock=self.decision_clock,
            availability_cutoff=self.availability_cutoff,
            provenance_status=self.provenance_status,
            provenance_notes=self.provenance_notes,
        )
        object.__setattr__(self, "content", content)

        # ---- DA-P0-04：执行身份独立记录，不进内容 hash。 ----
        execution = _execution_identity(
            session=self.session,
            decision_clock=self.decision_clock,
        )
        object.__setattr__(self, "execution", execution)

        # 内容 hash（稳定、与执行身份无关）。
        object.__setattr__(self, "digest", content.digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "availability": self.availability,
            "calendar_identity": self.calendar_identity,
            "universe_snapshot": self.universe_snapshot,
            "source_snapshot": self.source_snapshot,
            "grain": self.grain,
            "columns": self.columns,
            "time_range": self.time_range,
            "instrument_filter": self.instrument_filter,
            "fields": [f.to_dict() for f in self.fields],
            "digest": self.digest,
            "content": self.content.to_dict() if self.content is not None else None,
            "execution": self.execution.to_dict() if self.execution is not None else None,
            "provenance_status": self.provenance_status,
            "provenance_notes": list(self.provenance_notes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DataReadIdentity":
        """反序列化：重算 content digest 并校验（DA-P0-02）。"""
        data = dict(raw)
        supplied = data.get("digest")
        data.pop("digest", None)
        data.pop("content", None)
        data.pop("execution", None)
        if "fields" in data and data["fields"] is not None:
            data["fields"] = [
                ResolvedFieldIdentity(**dict(fd)) for fd in data["fields"]
            ]
        obj = cls(**data)
        if supplied is not None and supplied != obj.digest:
            raise ValidationError(
                f"DataReadIdentity digest 不匹配（DA-P0-02）："
                f"供给 {supplied!r} != 重算 {obj.digest!r} —— digest 只能由内部推导。"
            )
        return obj


# ---------------------------------------------------------------------------
# Provenance helpers（DA-P0-03：production fail-closed / research UNKNOWN）。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductionExecutionContext:
    """R45: explicit production/strict execution context.

    The strict/production mode is **never** inferred from "did an import
    succeed"（旧 ``_is_strict()`` 的 ``except Exception: return False`` 会在
    strict-infra 导入失败时 fail-open 成 research 语义）。生产调用路径显式传入
    本上下文；research 保留兼容默认（``from_authority`` 走 RuntimeModeIdentity
    单一权威，不依赖任何单个 import 是否成功）。
    """

    strict: bool
    source: str = "explicit"

    @classmethod
    def from_authority(cls) -> "ProductionExecutionContext":
        """从 RuntimeModeIdentity 单一权威解析（不 try/except 单个 import）。"""
        from data_access.runtime.mode_identity import is_strict_semantics_authority

        return cls(strict=bool(is_strict_semantics_authority()), source="authority")


def _revision_of(
    store: Any,
    dataset: str,
    params: Mapping[str, Any] | None = None,
    *,
    strict: bool | None = None,
) -> tuple[str | None, str | None]:
    """Derive a stable revision token from the dataset manifest.

    Returns ``(revision, note)``. ``strict=True``（production/PIT-strict）时
    不可得 → raise（fail closed）；research 返回 ``(None, note)`` 由调用方标注
    UNKNOWN。
    """
    fn = getattr(store, "manifest_version", None)
    if not callable(fn):
        note = "revision: store 无 manifest_version"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下无法解析 dataset={dataset!r} 的 revision：{note}"
            )
        return None, note
    try:
        mv = fn(dataset, **(params or {}))
    except Exception as exc:
        note = f"revision: manifest_version 失败: {type(exc).__name__}"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下无法解析 dataset={dataset!r} 的 revision：{note}"
            ) from exc
        return None, note
    if not isinstance(mv, dict):
        note = "revision: manifest_version 返回非 dict"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下无法解析 dataset={dataset!r} 的 revision：{note}"
            )
        return None, note
    if mv.get("has_manifest"):
        gen = mv.get("manifest_generation_id")
        if gen:
            return f"{dataset}:gen:{gen}", None
        token = (
            mv.get("source_epoch")
            or mv.get("manifest_epoch")
            or mv.get("manifest_built_epoch")
            or mv.get("dataset_version")
            or mv.get("partition_version")
        )
        if token is not None:
            return f"{dataset}:{token}", None
    # 未发布/无 manifest 的数据集（如测试用 StaticDataset）：用 dataset + 列 +
    # 物理对象集的稳定 token 兜底，避免 research/legacy 无谓地标 UNKNOWN。
    if mv.get("has_manifest") is False:
        try:
            resolve_paths = getattr(store, "_resolve_raw_paths", None)
            if callable(resolve_paths):
                ds_obj = None
                registry = getattr(store, "_registry", None)
                if registry is not None:
                    try:
                        ds_obj = registry.get(dataset)
                    except Exception:
                        ds_obj = None
                paths = resolve_paths(
                    ds_obj, time_range=None, params=dict(params or {})
                )
                files = tuple(sorted(str(p) for p in paths)) if paths else ()
                if files:
                    # R45 closure：不能只按文件**数量** hash——100 个不同文件会
                    # 塌缩成同一个 digest。改为对排序后的
                    # ``[(object_key, etag, size)]`` 做稳定 hash；etag/size 任一
                    # 不可得 → 返回 UNKNOWN / 不可缓存，而不是 count-only digest。
                    import hashlib

                    entries: list[tuple[str, str | None, int | None]] = []
                    for p in files:
                        etag = None
                        size = None
                        try:
                            st = Path(p).stat()
                            size = st.st_size
                        except OSError:
                            size = None
                        entries.append((p, etag, size))
                    if all(e[1] is not None or e[2] is not None for e in entries):
                        payload = "\n".join(
                            f"{k}|{e}|{s}" for k, e, s in entries
                        )
                        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                        return f"{dataset}:objects:{digest}", None
                    # etag/size 都不可得 → 无法形成内容级 digest，标 UNKNOWN。
                    note = (
                        "revision: 无 manifest 且对象 etag/size 不可得，"
                        "无法形成内容级 digest（不可缓存）"
                    )
                    if _resolve_strict(strict):
                        raise ValidationError(
                            f"production/strict 下无法解析 dataset={dataset!r} 的 revision：{note}"
                        )
                    return None, note
        except Exception:
            pass
    note = "revision: manifest 存在但无 generation/epoch token"
    if _resolve_strict(strict):
        raise ValidationError(
            f"production/strict 下无法解析 dataset={dataset!r} 的 revision：{note}"
        )
    return None, note


def _universe_snapshot_id(
    store: Any,
    universe: str | None,
    *,
    strict: bool | None = None,
) -> tuple[str | None, str | None]:
    """Universe 快照身份（DA-P0-03）。universe 未指定 → (None, None)（不要求）。

    ``strict=True`` 且 universe 指定但解析失败 → raise（fail closed）；
    research 返回 ``(None, note)``。
    """
    if not universe:
        return None, None
    try:
        from data_access.r30.universe_snapshot import UniverseSnapshot

        snap = UniverseSnapshot.from_store(store, universe)
        return snap.snapshot_id, None
    except Exception as exc:
        note = f"universe: {universe!r} 快照解析失败: {type(exc).__name__}"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下无法解析 required universe={universe!r}：{note}"
            ) from exc
        return None, note


def _calendar_identity(store: Any, *, strict: bool | None = None) -> tuple[str | None, str | None]:
    """Calendar 身份（DA-P0-03）。strict 下不可得 → raise（fail closed）。"""
    fn = getattr(store, "calendar_snapshot_id", None)
    if not callable(fn):
        note = "calendar: store 无 calendar_snapshot_id"
        if _resolve_strict(strict):
            raise ValidationError(f"production/strict 下无法解析 calendar identity：{note}")
        return None, note
    try:
        return str(fn()), None
    except Exception as exc:
        note = f"calendar: calendar_snapshot_id 失败: {type(exc).__name__}"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下无法解析 calendar identity：{note}"
            ) from exc
        return None, note


def _source_snapshot_id(
    prepared: Any,
    source_snapshot: Any,
    *,
    strict: bool | None = None,
) -> tuple[str | None, str | None]:
    """物理 source snapshot digest（DA-P0-03）。

    ``prepared``（PreparedRead）优先；无 prepared 时用 ``source_snapshot``。
    两者都没有（或没有 digest）→ strict 下 raise（fail closed），research
    返回 ``(None, note)``。
    """
    if prepared is not None and source_snapshot is None:
        src = getattr(prepared, "resolved_source_snapshot", None)
        if src is not None:
            source_snapshot = src
    if source_snapshot is not None:
        digest = getattr(source_snapshot, "content_digest", None)
        if digest:
            return str(digest), None
        note = "source_snapshot: 对象存在但无 content_digest"
        if _resolve_strict(strict):
            raise ValidationError(
                f"production/strict 下物理 source snapshot 缺少 content_digest：{note}"
            )
        return None, note
    note = "source_snapshot: 无 prepared/resolved source snapshot"
    if _resolve_strict(strict):
        raise ValidationError(
            f"production/strict 下物理 source snapshot 不可得：{note}"
        )
    return None, note


def _resolve_strict(strict: bool | None) -> bool:
    """strict 判定：显式传参优先；缺省用进程/请求级权威。

    R45：不再用 ``_is_strict()`` 的 try/except fail-open。缺省走
    ``ProductionExecutionContext.from_authority()``（RuntimeModeIdentity 单一
    权威），生产调用路径应显式传入 ``ProductionExecutionContext``。
    """
    if strict is not None:
        return bool(strict)
    return ProductionExecutionContext.from_authority().strict


# ---------------------------------------------------------------------------
# Execution context helpers（DA-P0-04）。
# ---------------------------------------------------------------------------

def _execution_request_id() -> str | None:
    try:
        from data_access.security.execution_context import current_execution_context

        ctx = current_execution_context()
        if ctx is not None:
            return ctx.request_id
    except Exception:
        pass
    return None


def _execution_trace_id() -> str | None:
    try:
        from data_access.security.execution_context import current_execution_context

        ctx = current_execution_context()
        if ctx is not None:
            trace = getattr(ctx, "trace_id", None)
            if trace:
                return str(trace)
            rid = ctx.request_id
            if rid:
                return f"trace:{rid}"
    except Exception:
        pass
    return None


def _execution_executor() -> str | None:
    try:
        from data_access.security.execution_context import current_execution_context

        ctx = current_execution_context()
        if ctx is not None:
            principal = ctx.principal
            if principal is not None:
                pid = getattr(principal, "principal_id", None)
                if pid:
                    return str(pid)
    except Exception:
        pass
    return None


def _execution_identity(
    *,
    session: str | None = None,
    decision_clock: str | None = None,
) -> ReadExecutionIdentity:
    """编译执行身份（request_id / session / executor / timestamp / trace_id）。

    ``session`` 显式传入优先；否则回退当前执行上下文的 request_id。
    ``timestamp`` 只在有 request_id（真实请求上下文）时才记录——测试/离线构建
    无请求时不引入随机时间戳，保证确定性可复现。
    """
    rid = session or _execution_request_id()
    trace = _execution_trace_id()
    executor = _execution_executor()
    ts = None
    if rid is not None and trace is not None:
        ts = datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return ReadExecutionIdentity(
        request_id=rid,
        session=session,
        executor=executor,
        timestamp=ts,
        trace_id=trace,
    )


# ---------------------------------------------------------------------------
# Range 稳定化（不动）。
# ---------------------------------------------------------------------------

def _stable_range(
    time_range: tuple[Any, Any] | None,
) -> tuple[str | None, str | None] | None:
    """稳定化 time_range 的两个端点 → (isoformat | None | str, 同左)。

    端点是 Timestamp/datetime/date → isoformat()；None → None；其它 → str()。
    返回 None 当且仅当输入本身为 None。
    """
    if time_range is None:
        return None
    start, end = time_range
    return (_stable_endpoint(start), _stable_endpoint(end))


def _stable_endpoint(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return str(value)


# ---------------------------------------------------------------------------
# Builders。
# ---------------------------------------------------------------------------

def build_data_read_identity(
    store: Any,
    *,
    dataset: str,
    columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    universe: str | None = None,
    decision_clock: str | None = None,
    availability_cutoff: str | None = None,
    prepared: Any = None,
    fields: Iterable[Any] | None = None,
    source_snapshot: Any = None,
    strict: bool | None = None,
) -> DataReadIdentity:
    """Build the immutable identity of one read from store + request context.

    DA-P0-03：production/PIT-strict（``strict=True`` 或缺省按权威判定）下，
    revision / calendar / required universe / 物理 source snapshot 任一不可得
    → raise（fail closed）。research 下缺失项标注 ``provenance_status="unknown"``
    并附 ``provenance_notes``。
    """
    strict_mode = _resolve_strict(strict)

    revision, rev_note = _revision_of(store, dataset, strict=strict_mode)
    calendar_id, cal_note = _calendar_identity(store, strict=strict_mode)
    universe_id, uni_note = _universe_snapshot_id(store, universe, strict=strict_mode)
    source_digest, src_note = _source_snapshot_id(
        prepared, source_snapshot, strict=strict_mode
    )

    notes: list[str] = []
    for note in (rev_note, cal_note, uni_note, src_note):
        if note:
            notes.append(note)
    if notes:
        notes.sort()

    resolved_fields = _sorted_field_identities(fields)
    if not resolved_fields and columns:
        resolved_fields = _sorted_field_identities(_placeholder_fields(columns, dataset))
    first = resolved_fields[0] if resolved_fields else None
    availability = (
        first.availability
        if first is not None and first.availability
        else "same_day"
    )
    grain = first.grain if first is not None else None

    # R45：production 读绝不消费 legacy 单值 availability/grain——必须用 per-field
    # FieldReadPolicySetIdentity（``fields`` 里的 ResolvedFieldIdentity）。单值只
    # 是向后兼容/单字段读的首值，多字段混合语义下它必然塌缩，production 禁止。
    if strict_mode and resolved_fields:
        _assert_production_uses_field_policies(resolved_fields, availability, grain)

    provenance_status = "unknown" if notes else "available"

    # DA-P0-04：session/request_id 不再参与内容 hash。执行身份独立记录。
    identity = DataReadIdentity(
        dataset=dataset,
        revision=revision,
        availability=availability,
        calendar_identity=calendar_id,
        universe_snapshot=universe_id,
        source_snapshot=source_digest,
        grain=grain,
        session=None,  # 显式 session 由调用方 set 后走 __post_init__ 记录执行身份
        decision_clock=decision_clock,
        availability_cutoff=availability_cutoff,
        columns=tuple(columns) if columns is not None else None,
        time_range=_stable_range(time_range),
        instrument_filter=tuple(instrument_filter) if instrument_filter is not None else None,
        fields=resolved_fields,
        provenance_status=provenance_status,
        provenance_notes=tuple(notes),
    )
    # 重建 execution（当前 request 上下文的 request_id 也记录进去）。
    object.__setattr__(
        identity,
        "execution",
        _execution_identity(
            session=_session_id(),
            decision_clock=decision_clock,
        ),
    )
    return identity


def _placeholder_fields(columns: Sequence[str], dataset: str) -> list[Any]:
    """没有 catalog 元数据时按列名生成占位字段（供多字段 identity 排序用）。"""
    from data_access.read.semantic_catalog import SemanticField

    return [
        SemanticField(logical_name=str(c), dataset=dataset, physical_name=str(c))
        for c in columns
    ]


def _assert_production_uses_field_policies(
    fields: tuple[ResolvedFieldIdentity, ...],
    legacy_availability: str,
    legacy_grain: str | None,
) -> None:
    """R45：production 读必须消费 per-field FieldReadPolicySetIdentity。

    单值 ``availability`` / ``grain`` 在多字段混合语义下必然塌缩（DA-P0-01 已
    证明），production 消费它会把不同可见性/粒度的字段当成同一语义。这里在
    production/strict 下拒绝「多字段读却依赖单值」的路径——单字段读（fields
    长度 1）仍允许单值作为该字段的首值。
    """
    if len(fields) <= 1:
        return
    # 多字段读：单值 availability/grain 无法表达混合语义 → production 拒绝。
    raise ValidationError(
        "production/strict 读禁止消费 legacy 单值 availability/grain："
        f"多字段读（{len(fields)} 个字段）的可见性/粒度必须用 per-field "
        "FieldReadPolicySetIdentity（ResolvedFieldIdentity.availability / .grain），"
        f"单值 availability={legacy_availability!r} / grain={legacy_grain!r} 会塌缩"
        "混合语义（DA-P0-01）。"
    )


def _session_id() -> str | None:
    """兼容旧 helper：当前 request 上下文的 request_id（只进执行身份）。"""
    return _execution_request_id()


def assert_semantic_field_production_ready(field: Any, *, strict: bool | None = None) -> None:
    """Production/strict: a SemanticField must declare availability / mining /
    unit / PIT before it can be read. Missing any → fail closed.

    R21: production SemanticField reads must not silently default to
    ``same_day`` availability / ``True`` mining / no unit / no PIT fidelity —
    those defaults hide look-ahead and unit-mixing bugs.

    R45: strict 判定走 ``ProductionExecutionContext``（显式传参或 RuntimeModeIdentity
    单一权威），不再 try/except 单个 import 推断——import 失败绝不 fail-open 成
    research 语义。
    """
    if strict is None:
        strict = ProductionExecutionContext.from_authority().strict
    if not strict:
        return
    missing: list[str] = []
    if not getattr(field, "availability", None):
        missing.append("availability")
    if getattr(field, "mining_allowed", None) is None:
        missing.append("mining_allowed")
    if not getattr(field, "source_unit", None) or not getattr(field, "canonical_unit", None):
        missing.append("unit(source_unit/canonical_unit)")
    if not getattr(field, "pit_fidelity", None):
        missing.append("pit_fidelity")
    if missing:
        raise ValidationError(
            f"production/strict 读要求 SemanticField "
            f"'{getattr(field, 'logical_name', '?')}' 声明完整语义，缺失: {missing} "
            "（R21 DataReadIdentity fail-closed）"
        )


__all__ = [
    "DataReadIdentity",
    "DataReadContentIdentity",
    "ReadExecutionIdentity",
    "ResolvedFieldIdentity",
    "ProductionExecutionContext",
    "assert_semantic_field_production_ready",
    "build_data_read_identity",
]
