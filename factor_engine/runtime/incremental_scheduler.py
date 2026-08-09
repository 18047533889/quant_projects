# -*- coding: utf-8 -*-
"""数据更新事件 → 受影响因子 → 最小增量重算窗口。

R10 #51: the scheduler must NOT reuse one engine for all affected factors.
Each factor is rebuilt from the catalog's FULL spec (data sources, filters,
composite config, universe) and materialized on a fresh ``FactorEngine`` whose
data-source config matches the catalog — so two factors with different source
configs never share an engine.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cleaned_operators.operator_policy import effective_lookback
from runtime.incremental import build_incremental_plan
from storage.catalog import FactorCatalog

logger = logging.getLogger(__name__)


class OperatorManifestUnavailable(RuntimeError):
    """算子实现 manifest 无法构建（P0-37）。

    production 下算子 manifest 是因子 lineage 的一部分——「无法证明算子实现」=
    lineage 不完整，必须让物化失败而不是静默省略。research 下只 warning 并跳过。
    """


class CalendarFingerprintUnavailable(RuntimeError):
    """日历指纹无法构建（P0-38/P0-39）。

    production 下依赖交易日/会话/可用性的因子必须绑定日历 digest；构建失败 =
    lineage 缺失，必须让物化失败。
    """


class CalendarOffsetError(RuntimeError):
    """forward-impact 窗口扩展遇到日历错误（P0-42）。

    production 下绝不允许把 ``t0`` 当作 ``t0+19``（零扩展 = 静默漏算）；必须
    raise。research 下按自然日过扩（保守，绝不少于 N 个交易日）。
    """


class DependencyDatasetResolutionError(RuntimeError):
    """被引用字段的物理 dataset 无法解析（P0-40/P0-41）。

    production 下 Composite / 多源字段的物理 dataset 未知 = 依赖边无法建立 =
    DataEvent 永远匹配不到该因子（增量重算静默失效），必须 fail。无 anchor
    兜底。
    """


@dataclass(frozen=True)
class DataEvent:
    """源数据字段更新事件 (R10 #53 extended).

    Keeps the original ``(dataset, column, updated_date)`` positional
    constructor working; the R10 fields (``field_id``, revision window,
    snapshot before/after, ``revision_kind``, ``deleted_keys``) are optional and
    default to ``None``.
    """

    dataset: str
    column: str
    updated_date: str
    field_id: str | None = None
    affected_start: str | None = None
    affected_end: str | None = None
    snapshot_before: Any = None
    snapshot_after: Any = None
    revision_kind: str | None = None
    #: 被删除的源数据 key。允许 ``(datetime, asset)`` 键对或裸 key（裸 key 在
    #: planner 里以受影响日期为删除日期）；planner 会归一成 tombstone 键对。
    deleted_keys: tuple[Any, ...] | None = None
    #: P0-18 ledger: event identity / ordering / status.  ``snapshot_before``
    #: must match the ledger's last committed snapshot for the same (dataset,
    #: field) chain, or the event is stale (rejected).  ``status`` is one of
    #: ``pending`` / ``committed`` / ``failed`` / ``rejected``.
    event_id: str | None = None
    sequence: int | None = None
    status: str | None = None
    #: R14 #5 P1 lease：持有 pending 预留的 worker 标识 / 尝试号。``owner`` 建议用
    #: ``(hostname, pid)``；``attempt_id`` 每次 begin 取号。lease 过期后其他 worker
    #: 才能 takeover（否则 ``in_flight`` 永久卡死）。
    owner: str | None = None
    attempt_id: str | None = None

    @property
    def field(self) -> str:
        """Canonical field the event touches (``field_id`` falls back to ``column``)."""
        return self.field_id or self.column

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "column": self.column,
            "updated_date": self.updated_date,
            "field_id": self.field_id or self.column,
            "affected_start": self.affected_start,
            "affected_end": self.affected_end,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "revision_kind": self.revision_kind,
            "deleted_keys": list(self.deleted_keys) if self.deleted_keys else None,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "status": self.status,
        }


def _strict_int(value: Any, name: str) -> int | None:
    """严格整数校验（Section 十五 / R13）。

    ``DataEvent.sequence`` 是强一致性的排序字段：``int(2.9)`` 截断成 ``2`` 会把
    两个不同事件折叠成同一序号、破坏 chain-head 判定。**绝不截断**——非整数值
    （含 bool）直接拒绝。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"DataEvent.{name} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"DataEvent.{name} must be an integer, got non-integral float {value!r}"
            )
        return int(value)
    # strings / numpy ints / Decimal …
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"DataEvent.{name} must be an integer, got {value!r}") from exc
    # 字符串形式 ``2.9`` / ``"2.5"`` 虽可 int() 但会截断——拒绝。
    if isinstance(value, str) and value.strip() != str(parsed).strip():
        raise ValueError(
            f"DataEvent.{name} must be an integer, got non-integral {value!r}"
        )
    return parsed


def normalize_data_event(event: DataEvent | dict[str, Any]) -> DataEvent:
    """Coerce a dict (or ``DataEvent``) to a :class:`DataEvent` (R10 #53).

    Section 十五 / R13: ``sequence`` is a strong-consistency ordering field —
    strict integer validation (never truncate ``2.9`` -> ``2``).
    """
    if isinstance(event, DataEvent):
        return event
    raw = dict(event)
    return DataEvent(
        dataset=str(raw["dataset"]),
        column=str(raw["column"]),
        updated_date=str(raw["updated_date"]),
        field_id=str(raw["field_id"]) if raw.get("field_id") else None,
        affected_start=str(raw["affected_start"]) if raw.get("affected_start") else None,
        affected_end=str(raw["affected_end"]) if raw.get("affected_end") else None,
        snapshot_before=raw.get("snapshot_before"),
        snapshot_after=raw.get("snapshot_after"),
        revision_kind=str(raw["revision_kind"]) if raw.get("revision_kind") else None,
        deleted_keys=tuple(raw["deleted_keys"]) if raw.get("deleted_keys") else None,
        event_id=str(raw["event_id"]) if raw.get("event_id") else None,
        sequence=_strict_int(raw.get("sequence"), "sequence"),
        status=str(raw["status"]) if raw.get("status") else None,
    )


@dataclass(frozen=True)
class FactorUpdatePlan:
    """单因子增量重算计划。"""

    factor_id: str
    column: str
    lookback_bars: int
    since: str | None
    end_date: str | None
    load_start: str | None
    load_end: str | None
    source_dataset: str | None
    is_full_run: bool
    field_id: str | None = None
    #: 源删除 key 归一成的 ``(datetime, asset)`` tombstone 键对（R11）。
    deleted_keys: tuple[tuple[str, str], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "factor_id": self.factor_id,
            "column": self.column,
            "field_id": self.field_id or self.column,
            "lookback_bars": self.lookback_bars,
            "since": self.since,
            "end_date": self.end_date,
            "load_start": self.load_start,
            "load_end": self.load_end,
            "source_dataset": self.source_dataset,
            "is_full_run": self.is_full_run,
            "deleted_keys": (
                [list(k) for k in self.deleted_keys] if self.deleted_keys else None
            ),
        }


class FactorSemanticIdentityMismatch(ValueError):
    """Rebuilt factor / data source does not match the catalog definition.

    ``runtime/factor_identity.py`` does not exist yet (R10); the check uses the
    identity fields available today (``source_expr`` vs catalog ``expression``,
    best-effort IR hash vs catalog ``ast_hash``).
    """


def _resolve_source_dataset(data_source: Any) -> str | None:
    return getattr(data_source, "dataset", None)


def _physical_dataset_for_logical_table(table: str) -> str | None:
    """逻辑表名（``StockDailyBar``）→ 物理 DataAccess dataset（``ashare_stock_daily``）。

    #收官轮 P0：edge 的物理 ``source_dataset`` 绝不能写成 FactorEngine 的 logical
    table name——``DataEvent.dataset`` 是物理 dataset，表名写错会导致
    ``DependencyCatalog.factors_for_event``（``source_dataset == event.dataset``）
    匹配不上、增量重算静默失效。优先 ``FieldSpec.dataset``（市场已解析正确），
    这里只兜底：默认单市场 registry 优先，再逐 market registry。
    """
    if not table:
        return None
    try:
        from fields import get_field_registry

        spec = get_field_registry().resolve_table(table)
        if spec is not None and getattr(spec, "dataset", None):
            return str(spec.dataset)
    except Exception:  # pragma: no cover - registry 缺失不阻塞 edge 构建
        pass
    try:
        from fields.market_registry import multi_market_registry

        mm = multi_market_registry()
        for market in getattr(mm, "markets", ("ashare", "us")):
            try:
                spec = mm.registry_for(market).resolve_table(table)
            except Exception:  # pragma: no cover - 单 market registry 不可用跳过
                spec = None
            if spec is not None and getattr(spec, "dataset", None):
                return str(spec.dataset)
    except Exception:  # pragma: no cover - 多市场 registry 缺失不阻塞
        pass
    return None


def _implementation_digest(impl: Any) -> str | None:
    """Deterministic digest of the ACTUAL operator kernel + logical contract.

    P0-35 / P1-36: never hash ``type(instance)``'s source file.  Overhaul ops
    are the same ``PandasFunctionOperator`` / ``PolarsFunctionOperator`` wrapper
    class whose real kernel lives in ``_fn`` — hashing the wrapper's source file
    would make every kernel edit invisible AND every wrapper-base edit invalidate
    hundreds of unrelated ops.  Resolution order:

      1. ``_fn`` — the wrapped kernel callable (bridge / function operators):
         ``_fn_payload`` hashes its bytecode + closure + defaults;
      2. class-defined kernel methods (``_calculate_series`` /
         ``_calculate_scalar`` / ``calculate``) via ``_impl_source_hash`` — code
         objects, never whole source files.

    The logical-contract hash (``_contract_hash``) is always appended so a
    contract change invalidates even when the kernel is unchanged.
    """
    import hashlib

    try:
        from cleaned_operators.registry import (
            _contract_hash,
            _fn_payload,
            _impl_source_hash,
        )
    except Exception:  # pragma: no cover - registry always importable
        return None
    try:
        kernel_fn = getattr(impl, "_fn", None)
        if callable(kernel_fn):
            payload = _fn_payload(kernel_fn, type(impl))
        else:
            payload = _impl_source_hash(impl)
        if not payload:
            return None
        contract = _contract_hash(impl) or ""
        return hashlib.sha256(f"{payload}::{contract}".encode("utf-8")).hexdigest()[:16]
    except Exception:  # pragma: no cover - 任一失败视为不可哈希
        return None


def _operator_manifest_from_ir(
    ir: Any, *, production: bool = False
) -> list[dict[str, str]] | None:
    """P2-02: operator semantic manifest for a factor's IR.

    Records every operator canonical used, its declared semantic version and a
    digest of its ACTUAL kernel + logical contract (``_implementation_digest``)
    — so an operator upgrade invalidates the factor even when source data is
    unchanged.  ``None`` when the IR has no operators (caller stores nothing).

    P0-37: a *build failure* (registry unavailable / digest uncomputable) is NOT
    silently omitted — in production it raises ``OperatorManifestUnavailable``
    (materialization must fail: no operator lineage = not a complete factor);
    research logs a warning and skips.
    """
    if ir is None:
        return None
    try:
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.registry import OperatorRegistry

        ensure_cleaned_loaded()
        seen: dict[str, dict[str, str]] = {}

        def walk(node: Any) -> None:
            op = getattr(node, "op", None)
            if op and op not in {"column", "literal", "plan_ref"}:
                try:
                    canonical = OperatorRegistry.resolve_canonical_optional(op) or str(op)
                except Exception:
                    canonical = str(op)
                impl = None
                try:
                    impl = OperatorRegistry.get(canonical)
                except Exception:
                    impl = None
                if impl is None:
                    if production:
                        raise OperatorManifestUnavailable(
                            f"operator manifest: no implementation for canonical "
                            f"{canonical!r} (operator lineage unavailable)"
                        )
                    logger.warning(
                        "operator manifest: no implementation for %r (research, "
                        "recorded without implementation_hash)",
                        canonical,
                    )
                source_hash = _implementation_digest(impl) if impl is not None else None
                if source_hash is None and production:
                    raise OperatorManifestUnavailable(
                        f"operator manifest: cannot compute implementation digest "
                        f"for canonical {canonical!r} (P0-35/P0-37 fail-closed)"
                    )
                if source_hash is None:
                    logger.warning(
                        "operator manifest: implementation digest unavailable for "
                        "%r (research, recorded without implementation_hash)",
                        canonical,
                    )
                seen.setdefault(
                    canonical,
                    {
                        "canonical": canonical,
                        "semantic_version": str(
                            getattr(getattr(impl, "metadata", None), "semantic_version", None)
                            or "1.0"
                        ),
                        "implementation_hash": source_hash,
                    },
                )
            for child in getattr(node, "inputs", ()) or ():
                walk(child)

        walk(ir)
        return sorted(seen.values(), key=lambda m: m["canonical"]) if seen else None
    except OperatorManifestUnavailable:
        raise
    except Exception as exc:
        if production:
            raise OperatorManifestUnavailable(
                f"operator manifest build failed (P0-37): {type(exc).__name__}: {exc}"
            ) from exc
        logger.warning("operator manifest build failed (research, skipped): %s", exc)
        return None


def _calendar_scope_payload(data_source: Any, calendar: Any) -> dict[str, Any]:
    """日历指纹的额外作用域维度：会话段 / 时区 / DST / early-close 表。

    这些字段大多数 data source / calendar 上不存在（``getattr`` 返回 None），
    存在则必须进入 digest——同一组交易日在不同时区 / 会话段 / early-close 规则
    下不是同一个执行日历。
    """
    sess = getattr(data_source, "session_calendar", None)
    if sess is None:
        sess = getattr(calendar, "session_calendar", None)
    session_segments = None
    if sess is not None:
        segs = getattr(sess, "segments", None)
        if segs:
            session_segments = [tuple(str(x) for x in s) for s in segs]
    early_close = None
    for obj in (data_source, calendar):
        val = getattr(obj, "early_close", None)
        if val is None:
            val = getattr(obj, "early_close_days", None)
        if val is not None:
            try:
                early_close = sorted(str(v) for v in val)
            except TypeError:
                early_close = str(val)
            break
    dst_rules = None
    for obj in (data_source, calendar):
        val = getattr(obj, "dst_rules", None)
        if val is None:
            val = getattr(obj, "dst", None)
        if val is not None:
            dst_rules = str(val)
            break
    return {
        "session_segments": session_segments,
        "timezone": getattr(data_source, "timezone", None)
        or getattr(calendar, "timezone", None),
        "dst_rules": dst_rules,
        "early_close": early_close,
        "calendar_semantic_version": getattr(calendar, "semantic_version", None) or "1",
    }


def _calendar_version(data_source: Any, *, production: bool = False) -> str | None:
    """P2-04: calendar-contract fingerprint for the factor's lineage.

    P0-38: the old ``v{len}:{first}:{last}`` fingerprint is insufficient — an
    interior holiday swap leaves ``len/first/last`` unchanged.  The new digest
    hashes the FULL ordered trading days plus session segments / timezone / DST
    rules / early-close table / calendar semantic version, so any calendar
    contract change shifts the fingerprint and invalidates rolling history /
    intraday aggregation / availability dependencies.

    P0-39: a *build failure* (market-bound but calendar unresolvable, or digest
    computation error) is NOT silently ``None`` — in production it raises
    ``CalendarFingerprintUnavailable`` (a factor depending on trading days must
    bind a calendar digest); research logs a warning and skips.
    """
    try:
        market = (
            getattr(data_source, "market", None)
            or getattr(data_source, "market_code", None)
        )
        if not market:
            from storage.trading_calendar import infer_market

            market = infer_market(
                universe=getattr(data_source, "universe", None),
                dataset=getattr(data_source, "dataset", None),
            )
        if not market:
            # 无市场/无日历绑定：该数据源不依赖交易日历，无需指纹。
            return None
        from storage.trading_calendar import get_trading_calendar

        calendar = get_trading_calendar(market)
        if calendar is None:
            if production:
                raise CalendarFingerprintUnavailable(
                    f"calendar fingerprint: market {market!r} has no resolvable "
                    "trading calendar (P0-39 fail-closed)"
                )
            logger.warning(
                "calendar fingerprint: no trading calendar for market %r "
                "(research, skipped)",
                market,
            )
            return None
        days = getattr(calendar, "days", None)
        if not days:
            if production:
                raise CalendarFingerprintUnavailable(
                    f"calendar fingerprint: market {market!r} calendar has no "
                    "trading days (P0-39 fail-closed)"
                )
            return None
        ordered = [str(pd.Timestamp(d).normalize().date()) for d in days]
        payload = {
            "market": str(market),
            "calendar_id": getattr(data_source, "calendar_id", None)
            or getattr(data_source, "calendar", None),
            "trading_days": ordered,
            **_calendar_scope_payload(data_source, calendar),
        }
        import hashlib

        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:16]
    except CalendarFingerprintUnavailable:
        raise
    except Exception as exc:
        if production:
            raise CalendarFingerprintUnavailable(
                f"calendar fingerprint build failed (P0-39): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        logger.warning(
            "calendar fingerprint build failed (research, skipped): %s", exc
        )
        return None


def _dependency_rows_for_event(
    catalog: Any,
    event: DataEvent,
) -> list[dict[str, Any]]:
    """Dependency rows for an event — edges first, legacy column index fallback.

    ``catalog`` may be a ``runtime.dependency_catalog.DependencyCatalog`` (R10
    edge-aware) or a raw ``storage.catalog.FactorCatalog`` (legacy rows).
    """
    if hasattr(catalog, "factors_for_event"):
        return catalog.factors_for_event(event)
    return catalog.list_factors_for_column(event.field_id or event.column)


class DeletePredicateError(ValueError):
    """A bare deleted key (no datetime) cannot be auto-dated (P0-19).

    Production demands ``(datetime, asset)`` or a structured ``DeletePredicate``;
    guessing ``affected_start`` as the delete date has no clear semantics when
    ``affected_start != affected_end``.
    """


def _coerce_deleted_keys(
    event: DataEvent,
    *,
    production: bool = False,
) -> tuple[tuple[str, str], ...] | None:
    """把 ``DataEvent.deleted_keys`` 归一成 ``(datetime, asset)`` tombstone 键对。

    ``ParquetMaterializer._append_tombstones`` 需要 ``(datetime, asset)`` 键对；
    裸 key（单字符串）以事件受影响日期（``affected_start`` 或 ``updated_date``）
    作为删除日期，避免把删除日错记为事件抵达日。

    P0-19: production 拒绝裸 key —— 裸 key 的删除日期是"猜"的，``affected_start !=
    affected_end`` 时没有充分语义；必须显式 ``(datetime, asset)`` 或结构化谓词。
    """
    raw = event.deleted_keys
    if not raw:
        return None
    default_date = event.affected_start or event.updated_date
    out: list[tuple[str, str]] = []
    for key in raw:
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            out.append((str(key[0]), str(key[1])))
        else:
            if production:
                raise DeletePredicateError(
                    f"production: deleted key {key!r} has no datetime — a bare key "
                    "cannot be auto-dated (affected_start may != affected_end). "
                    "Provide (datetime, asset) or a structured DeletePredicate."
                )
            out.append((str(default_date), str(key)))
    return tuple(out)


def _extend_date_forward(
    end_date: str | None,
    bars: int | None,
    *,
    market: str | None,
    cap: str | None = None,
    production: bool = False,
) -> str | None:
    """把 ``end_date`` 向未来扩 ``bars`` 个交易日（P0-01 forward impact）。

    ``bars`` 为 None（unbounded）时返回 ``cap``（最新数据日）；否则按交易日历
    前移 ``bars`` 并把结果封顶在 ``cap``（不越过可用数据的最新日）。

    P0-42: 日历错误时**绝不零扩展**（把 ``t0`` 当 ``t0+19`` = 静默漏算）。
    production 下 raise ``CalendarOffsetError``；research 下按自然日过扩（至少
    ``bars`` 个自然日，必 >= ``bars`` 个交易日），保证不会比正确窗口更窄。
    """
    if bars is None:
        return cap
    if not end_date or bars <= 0:
        return end_date
    try:
        from storage.trading_calendar import get_trading_calendar, trading_day_offset

        calendar = get_trading_calendar(market)
        shifted = trading_day_offset(pd.Timestamp(end_date), int(bars), calendar=calendar)
        out = str(pd.Timestamp(shifted).date())
        if cap:
            return min(out, str(pd.Timestamp(cap).date()))
        return out
    except Exception as exc:
        if production:
            raise CalendarOffsetError(
                f"forward-impact window extension failed (P0-42): end_date="
                f"{end_date!r} bars={bars} market={market!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # research: 自然日过扩（保守上界），绝不零扩展。
        shifted_calendar_days = pd.Timestamp(end_date) + pd.Timedelta(days=int(bars))
        out = str(pd.Timestamp(shifted_calendar_days).date())
        if cap:
            return min(out, str(pd.Timestamp(cap).date()))
        return out


def plan_updates_from_data_event(
    catalog: Any,
    event: DataEvent,
    *,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    production: bool = False,
) -> list[FactorUpdatePlan]:
    """``column``/``field_id`` 更新后，查询 catalog 中受影响因子并计算 lookback 窗口。

    修订窗口以**事件受影响的数据日期**（``affected_start``/``affected_end``）为
    基准，而不是事件抵达日期（``updated_date``）。历史修订示例：2026-08-09 收到
    修订、实际受影响的是 2024-06-01 财务数据——若从 ``updated_date`` 起重算，
    2024→2026 的下游因子会全部漏算。``since`` 传 ``affected_start``，由
    ``build_incremental_plan`` 再往前推 dependency lookback。

    P0-01: 输入修订的影响是**向未来传播**的 —— ``ts_mean(x,20)`` 中 ``x[t0]``
    被修订会同时影响 ``t0..t0+19`` 的输出。因此 ``recompute_end`` 先按因子的
    forward impact 向未来扩展（``ForwardImpactRequirement``），再让
    ``build_incremental_plan`` 对这个输出窗口向后扩 warmup；不能只往历史方向
    扩而把输出停在 ``affected_end``。P0-02: 删除事件同样走 forward 传播。
    """
    deps = _dependency_rows_for_event(catalog, event)
    recompute_start = event.affected_start or event.updated_date
    recompute_end = event.affected_end or end_date
    deleted_keys = _coerce_deleted_keys(event, production=production)
    plans: list[FactorUpdatePlan] = []
    for dep in deps:
        source_dataset = dep.get("source_dataset")
        if source_dataset and str(source_dataset) != str(event.dataset):
            continue
        factor_id = str(dep["factor_id"])
        lookback = int(dep.get("lookback") or 0)
        # P0-01 / P0-43: per-factor forward impact — the catalog read path now
        # carries a tri-state ``ForwardImpactRequirement`` (finite / unbounded /
        # unknown).  ``effective_forward_bars`` resolves it to scheduler bars:
        # ``None`` = unbounded (reach cap / latest data), ``unknown`` in
        # production resolves conservatively to unbounded (mandatory reindex).
        from runtime.dependency_catalog import (
            ForwardImpactRequirement,
            effective_forward_bars,
        )

        forward = dep.get("forward_impact")
        forward_req = (
            forward
            if isinstance(forward, ForwardImpactRequirement)
            else ForwardImpactRequirement.unknown()
        )
        forward_bars = effective_forward_bars(forward_req, production=production)
        eff_end = _extend_date_forward(
            recompute_end,
            forward_bars,
            market=market,
            cap=end_date,
            production=production,
        )
        inc = build_incremental_plan(
            factor_id=factor_id,
            analysis_lookback=lookback,
            watermark={"end_date": recompute_start},
            since=recompute_start,
            end_date=eff_end,
            lookback_extra=lookback_extra,
            market=market,
            factor_freq=dep.get("frequency"),
            forward_impact=forward_bars,
        )
        plans.append(
            FactorUpdatePlan(
                factor_id=factor_id,
                column=event.column,
                field_id=event.field_id or event.column,
                lookback_bars=int(inc.lookback_bars),
                since=recompute_start,
                end_date=eff_end,
                load_start=None if inc.load_start is None else inc.load_start.isoformat(),
                load_end=None if inc.load_end is None else inc.load_end.isoformat(),
                source_dataset=source_dataset,
                is_full_run=bool(inc.is_full_run),
                deleted_keys=deleted_keys,
            )
        )
    return plans


def _edges_from_analysis(
    factor_id: str,
    analysis: Any,
    data_source: Any,
    *,
    production: bool = False,
) -> list[Any]:
    """Build rich dependency edges from the resolved physical source manifest.

    R11: each ``FieldSpec`` in ``analysis.referenced_fields`` carries its
    physical DataAccess dataset (``spec.dataset``), its FactorEngine logical
    table (``spec.table``), canonical ``field_id`` and physical column
    (``spec.source_name``).  The edge ``source_dataset`` MUST be the physical
    dataset the field lives in — NOT the anchor ``data_source.dataset``: a
    composite factor reading ``StockDailyBar.close`` + ``StockValuationDaily.pe_ratio``
    must record the second edge on ``ashare_stock_valuation_daily``, or
    ``DependencyCatalog.factors_for_event()`` (which queries
    ``source_dataset == event.dataset``) can never find it.

    P0-40: logical→physical dataset resolution must NEVER anchor-fallback.  A
    Composite source's valuation child resolving to the daily anchor would make
    a valuation event never trigger the factor — the dependency would silently
    point at the wrong dataset.  P0-41: a real dependency whose physical dataset
    is unresolvable must not be silently dropped (``continue``).  In production
    both fail with ``DependencyDatasetResolutionError``; research logs a warning
    and skips the unresolved edge.
    """
    from runtime.dependency_catalog import FactorDependencyEdge, UNBOUNDED_FORWARD_IMPACT

    ref_fields = getattr(analysis, "referenced_fields", {}) or {}
    # P0-01: forward impact of the whole factor — any changed source field
    # propagates with the factor's forward reach (None = unbounded, stored as
    # the -1 sentinel so legacy ``None`` keeps meaning "not recorded").
    from runtime.execution_contract import factor_forward_impact

    fwd = None
    try:
        fwd = factor_forward_impact(getattr(analysis, "ir", None))
    except Exception:  # pragma: no cover - 无 IR 时保守视为无前向传播
        fwd = None
    fwd_stored = UNBOUNDED_FORWARD_IMPACT if fwd is None else max(0, int(fwd))
    # P2-03: field-catalog version enters the edge lineage — a catalog change
    # (unit / transform / calendar contract) shifts this hash and makes the
    # dependency identity differ even when the parquet data is unchanged.
    field_catalog_hash = None
    try:
        from fields import FIELD_REGISTRY

        field_catalog_hash = str(FIELD_REGISTRY.catalog_hash() or "")
    except Exception:  # pragma: no cover - registry 缺失时无版本
        field_catalog_hash = None
    edges: list[FactorDependencyEdge] = []
    for name, spec in ref_fields.items():
        # #收官轮 P0/P0-40: 物理 source_dataset 必须解析成 DataAccess 物理
        # dataset。``spec.dataset``（市场正确）→ ``TableSpec(spec.table).dataset``。
        # **永远不要**把 logical table name（``StockDailyBar``）写进物理
        # ``source_dataset``（DataEvent 匹配不上），**也不许** anchor-fallback
        # （valuation child 解析到 daily 会让 valuation 事件永远触发不了因子）。
        physical_ds = getattr(spec, "dataset", None)
        if not physical_ds:
            physical_ds = _physical_dataset_for_logical_table(
                getattr(spec, "table", None)
            )
        if not physical_ds:
            if production:
                raise DependencyDatasetResolutionError(
                    f"factor {factor_id!r} references field {name!r} whose "
                    f"physical dataset is unresolvable "
                    f"(table={getattr(spec, 'table', None)!r}) — no anchor "
                    f"fallback (P0-40/P0-41 fail-closed)"
                )
            logger.warning(
                "dependency edge: field %r of factor %r has no physical dataset "
                "(table=%r) — skipping (research)",
                name,
                factor_id,
                getattr(spec, "table", None),
            )
            continue
        field_id = str(getattr(spec, "field_id", None) or name)
        # P1-10: derive the REAL snapshot semantics from the field's transform /
        # temporal model instead of a blanket "point".
        transform = getattr(spec, "transform", None)
        if transform in {"asof", "asof_backward"}:
            snap = "asof"
        elif transform in {"financial_asof", "financial_lag"}:
            snap = "financial_pit"
        elif transform in {"ffill", "forward_fill"}:
            snap = "asof"
        else:
            snap = str(getattr(spec, "snapshot_semantics", None) or "point")
        edges.append(
            FactorDependencyEdge(
                factor_id=factor_id,
                source_dataset=str(physical_ds),
                logical_table=getattr(spec, "table", None),
                field_id=field_id,
                physical_field=str(
                    getattr(spec, "source_name", None) or field_id
                ),
                transform=transform,
                snapshot_semantics=snap,
                lookback=0,
                forward_impact=fwd_stored,
                field_catalog_hash=field_catalog_hash,
            )
        )
    return edges


def record_factor_dependency_from_analysis(
    catalog: FactorCatalog,
    *,
    factor_id: str,
    analysis: Any,
    data_source: Any,
    frequency: str | None = None,
    full_definition: dict[str, Any] | None = None,
    production: bool = False,
) -> None:
    """从 ``AnalysisResult`` 写入依赖 catalog（legacy 行 + R10 edges [+ full spec]）。

    R11: the whole factor dependency definition is written atomically through
    ``DependencyCatalog.record_factor_manifest`` (single BEGIN IMMEDIATE
    transaction), so a concurrent ``factors_for_event`` reader or a crash can
    never observe a partially-replaced edge set (0/1/5 edges).

    P0-37 / P0-39 / P0-40 / P0-41: ``production=True`` turns every lineage build
    failure (operator manifest, calendar fingerprint, unresolvable physical
    dataset) into a hard error — never silently omit lineage.
    """
    from runtime.dependency_catalog import DependencyCatalog

    referenced_columns = getattr(analysis, "referenced_columns", set()) or set()
    lookback = int(getattr(analysis, "lookback", 0))
    source_dataset = _resolve_source_dataset(data_source)
    edges = _edges_from_analysis(
        factor_id, analysis, data_source, production=production
    )
    # P2-02/P2-04: operator semantic manifest + calendar version enter the
    # full definition so an operator upgrade / calendar-contract change
    # invalidates the factor even when source data is unchanged.
    manifest = _operator_manifest_from_ir(
        getattr(analysis, "ir", None), production=production
    )
    cal_version = _calendar_version(data_source, production=production)
    full_def = dict(full_definition or {})
    if manifest:
        full_def["operator_manifest"] = manifest
    if cal_version:
        full_def["calendar_version"] = cal_version
    dep_catalog = (
        catalog if isinstance(catalog, DependencyCatalog) else DependencyCatalog(catalog)
    )
    dep_catalog.record_factor_manifest(
        factor_id,
        edges=edges,
        referenced_columns=referenced_columns,
        lookback=lookback,
        frequency=frequency,
        source_dataset=source_dataset,
        full_definition=full_def or None,
    )


def factor_from_catalog_info(info: dict) -> Any:
    """catalog 行 → ``Factor``（需含 ``expression``；R10 支持 surface/dialect/universe）。

    R14 #4：除 ``parse_factor`` 的基本元数据外，还要从 full definition 恢复
    **canonical execution scope**（``FactorExecutionScopeHint``：market /
    universe_id / frequency / calendar_id / decision_time_policy）——否则事件增量
    rebuild 后 ``_scope_from_factor`` 只剩 ``factor.freq``/``factor.universe`` 兜底，
    ``decision_time_policy=eod`` / market=A / calendar=SSE 全部丢失，重建出的因子
    与落库时的执行语义分裂。
    """
    import dataclasses

    from api.dsl_parser import parse_factor
    from api.factor import FactorExecutionScopeHint

    expression = info.get("expression")
    if not expression or not str(expression).strip():
        raise ValueError(f"因子 {info.get('factor_id')!r} 缺少 expression，无法重建")
    factor_id = str(info.get("factor_id") or info.get("name") or "unknown")
    factor = parse_factor(
        str(expression),
        name=factor_id,
        freq=str(info.get("frequency") or "1d"),
        universe=info.get("universe"),
        description=info.get("description"),
        surface=str(info.get("surface") or "daily"),
        dialect=str(info.get("dialect") or "native"),
        dialect_version=info.get("dialect_version"),
    )
    scope_fields = {
        "market": info.get("market"),
        "universe_id": info.get("universe") or info.get("universe_id"),
        "frequency": info.get("frequency") or info.get("freq"),
        "calendar_id": info.get("calendar") or info.get("calendar_id"),
        "decision_time_policy": info.get("decision_policy")
        or info.get("decision_time_policy"),
    }
    if any(v is not None and str(v) != "" for v in scope_fields.values()):
        hint = FactorExecutionScopeHint(
            **{
                k: (str(v) if v is not None else None)
                for k, v in scope_fields.items()
            }
        )
        factor = dataclasses.replace(factor, semantic_identity=hint)
    return factor


# ---------------------------------------------------------------------------
# R10 #51: per-factor engine reconstruction
# ---------------------------------------------------------------------------

def _default_engine_factory(
    full_def: dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    market: str | None = None,
    run_mode: str | None = None,
) -> Any:
    """Build a fresh ``FactorEngine`` for one factor's full definition.

    The data source is reconstructed from the catalog's ``data_source_config``,
    so two factors with different source configs are guaranteed to run on their
    own engine.  Overridable via ``engine_factory=`` for tests / alternate
    backends.
    """
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import DataSourceBuildContext, build_data_source

    ds_config = dict(full_def.get("data_source_config") or {})
    if not ds_config:
        raise ValueError(
            f"因子 {full_def.get('factor_id')!r} 缺少 data_source_config，无法重建引擎"
        )
    if "type" not in ds_config:
        ds_config["type"] = "parquet"
    mode = run_mode or full_def.get("run_mode")
    build_context = DataSourceBuildContext(
        run_mode=mode,
        market=market or full_def.get("market"),
        calendar_id=full_def.get("calendar"),
        pit_enforce=bool(full_def.get("pit_enforce")),
    )
    data_source = build_data_source(ds_config, build_context=build_context)
    backend = build_backend(str(full_def.get("backend") or "pandas"))
    return FactorEngine(backend=backend, data_source=data_source, run_mode=mode)


def _build_engine_for_factor(
    full_def: dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    market: str | None = None,
    engine_factory: Any | None = None,
) -> Any:
    factory = engine_factory or _default_engine_factory
    return factory(
        full_def,
        lake_root=lake_root,
        market=market,
        run_mode=full_def.get("run_mode"),
    )


def _factor_scope_field(factor: Any, hint_attr: str, direct_attr: str) -> str:
    """从 factor 读取 scope 字段：优先 ``semantic_identity`` 提示，回退因子级属性。

    R14 #4：rebuild 后 canonical scope 挂在 ``factor.semantic_identity``（
    ``FactorExecutionScopeHint``）。``Factor`` dataclass 本身没有
    decision_time_policy 等字段，只读因子级属性会让 declared 有值、actual 恒空。
    """
    hint = getattr(factor, "semantic_identity", None)
    if hint is not None:
        v = getattr(hint, hint_attr, None)
        if v is not None and str(v) != "":
            return str(v)
    v = getattr(factor, direct_attr, None)
    return str(v) if v is not None and str(v) != "" else ""


def _verify_factor_semantic_identity(
    factor: Any,
    full_def: dict[str, Any],
    *,
    production: bool = False,
) -> None:
    """Verify a rebuilt factor matches the catalog identity — FAIL-CLOSED (P0-17).

    P0-17: in production, "cannot prove identity" IS an identity-verification
    failure — a hash-computation error must never be silently skipped and passed.
    Checks, when the catalog records them: ``source_expr``, IR ``ast_hash``,
    surface / dialect / dialect_version / decision policy / calendar / PIT mode.
    """
    fid = full_def.get("factor_id") or "?"
    catalog_expr = full_def.get("expression")
    if catalog_expr and str(catalog_expr).strip():
        actual_expr = str(getattr(factor, "source_expr", "") or "").strip()
        if production and not actual_expr:
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建后无 source_expr，无法证明 identity（P0-17 fail-closed）"
            )
        if actual_expr and actual_expr != str(catalog_expr).strip():
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建表达式与 catalog 不一致: "
                f"{actual_expr!r} != {catalog_expr!r}"
            )
    ast_hash = full_def.get("ast_hash")
    if ast_hash and str(ast_hash).strip():
        from storage.catalog import compute_ir_hash

        try:
            # R14 #4：rebuild 的 factor 是 Expr，materialize 落库的 ast_hash 来自
            # ``compute_ir_hash(analysis.ir)``（IRNode）。直接对 Expr 哈希会拿不到
            # ``.attrs`` 报错——必须先按与执行同源的路径 lower 成 IR 再哈希，否则
            # production 事件（full_def 含 ast_hash）永远无法证明 identity。
            from ir.analyzer import Analyzer

            ir_node = Analyzer(production=production).lower(factor.expr).ir
            actual_hash = compute_ir_hash(ir_node)
        except Exception as exc:  # Expr is not always a serializable IR node
            if production:
                raise FactorSemanticIdentityMismatch(
                    f"因子 {fid!r} 无法计算 IR hash，无法证明 identity（P0-17 "
                    f"fail-closed）: {type(exc).__name__}: {exc}"
                ) from exc
            actual_hash = None
        if (
            actual_hash is not None
            and str(actual_hash) != str(ast_hash).strip()
        ):
            if production:
                raise FactorSemanticIdentityMismatch(
                    f"因子 {fid!r} IR hash 与 catalog 不一致: "
                    f"{str(actual_hash)[:12]} != {str(ast_hash)[:12]}"
                )
            # research：记录但不 fail——research 允许覆盖/旧数据，只有 production 才
            # 是 identity 的权威门（R14 #4 修 Expr→IR hash 后，以前 compute 永远
            # 报错 → 这里实际从不触发；现在真能算，research 保持历史宽容）。
            logger.warning(
                "因子 %s 重建 IR hash 与 catalog 不一致（research 容忍）: %s != %s",
                fid,
                str(actual_hash)[:12],
                str(ast_hash)[:12],
            )
    for key, getter in (
        ("surface", lambda f: str(getattr(f, "surface", None) or "")),
        ("dialect", lambda f: str(getattr(f, "dialect", None) or "")),
        ("dialect_version", lambda f: str(getattr(f, "dialect_version", None) or "")),
        # R14 #4：decision_policy / market / universe / frequency / calendar 从
        # ``factor.semantic_identity``（rebuild 时恢复的 FactorExecutionScopeHint）
        # 优先读取，回退到因子级属性——``Factor`` 没有 decision_time_policy 字段，
        # 旧实现 getattr 恒为空，导致 declared=eod、rebuilt="" 静默通过（fail-open）。
        (
            "decision_policy",
            lambda f: _factor_scope_field(f, "decision_time_policy", "decision_time_policy"),
        ),
        ("market", lambda f: _factor_scope_field(f, "market", "market")),
        (
            "universe",
            lambda f: _factor_scope_field(f, "universe_id", "universe"),
        ),
        ("frequency", lambda f: _factor_scope_field(f, "frequency", "freq")),
        (
            "calendar",
            lambda f: _factor_scope_field(f, "calendar_id", "calendar_id"),
        ),
    ):
        declared = full_def.get(key)
        if not declared:
            continue
        actual = getter(factor)
        if not actual:
            # R14 #4 fail-closed：declared 有值 + 重建 actual 缺失 = 无法证明
            # identity（旧实现只检查「actual 非空时不同」，丢失的决策策略被放过）。
            if production:
                raise FactorSemanticIdentityMismatch(
                    f"因子 {fid!r} 重建 {key} 缺失（catalog 声明 {declared!r}，"
                    f"重建后为空）—— P0-17 fail-closed：无法证明 identity"
                )
            continue
        if actual != str(declared):
            raise FactorSemanticIdentityMismatch(
                f"因子 {fid!r} 重建 {key} 与 catalog 不一致: {actual!r} != {declared!r}"
            )


class DataEventStaleError(ValueError):
    """An event's ``snapshot_before`` does not match the ledger's last committed
    snapshot for its (dataset, field) chain — stale / out-of-order (P0-18)."""


class DataEventLedgerCorruptionError(ValueError):
    """Ledger 文件损坏且不是可恢复的尾部半行（崩溃残留）时抛出（R14 #5）。

    旧实现 ``_load`` 把整文件损坏静默跳过/清空、``_append`` 把 OSError 直接吞掉
    ——「顺序 / 幂等 / chain-head」保证会在损坏时悄悄失效。现在：
      * 文件无法读取 / 中段 JSON 损坏 → fail loud（本异常），绝不静默降级；
      * 仅最后一行是半行（崩溃 mid-append）→ 截断该行恢复（WAL 语义）。
    """


class PartialIncrementalFailureError(RuntimeError):
    """Production forbids partial-success events: some downstream factors failed,
    so the event must stay pending (P0-20).

    R14 #5：production 下已经成功物化的 factor **不会**回滚（跨 factor 的
    parquet 全量回滚不现实）；事件保持 ``rejected``/``pending``，重跑同一
    ``event_id`` 幂等收敛（成功过的因子由 checkpoint 跳过，失败的重试）——
    决不允许把 partial 结果当成功 publish。
    """


@contextlib.contextmanager
def _ledger_lock(lock_path: Path):
    """ledger 文件写锁（flock）：begin/commit/reject 与跨进程 reload 串行化。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


# ---------------------------------------------------------------------------
# R14 #5 P1：pending 预留的 lease（worker crash 后可由其他 worker takeover）
# ---------------------------------------------------------------------------

#: pending 预留的默认 lease 时长（秒）。worker 在 begin 后 crash、超过该时长
#: 未 commit/reject → 其他 worker 可 takeover（旧实现 ``pending`` 永久 → 事件
#: 永远 ``in_flight``，卡死）。可用 ``DATA_EVENT_LEDGER_LEASE_SECONDS`` 覆盖。
_LEDGER_LEASE_SECONDS = max(
    60, int(os.environ.get("DATA_EVENT_LEDGER_LEASE_SECONDS", "3600"))
)


def _ledger_now_iso() -> str:
    """UTC ISO 时间戳（含 tz，供 lease 过期判定）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ledger_reserved_at_seconds(reserved_at: Any) -> float | None:
    """解析 ``reserved_at`` 为 epoch 秒；非法返回 None（保守不 takeover）。"""
    from datetime import datetime, timezone

    if reserved_at is None:
        return None
    try:
        dt = datetime.fromisoformat(str(reserved_at))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _ledger_pending_is_stale(rec: dict[str, Any]) -> bool:
    """pending 记录是否 lease 过期。无 ``reserved_at`` 的 legacy 记录保守不 takeover。"""
    reserved = rec.get("reserved_at")
    if not reserved:
        return False
    ts = _ledger_reserved_at_seconds(reserved)
    if ts is None:
        return False
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).timestamp() - ts > _LEDGER_LEASE_SECONDS


class DataEventLedger:
    """P0-18: idempotent / ordered DataEvent processing ledger.

    Persists ``{event_id -> record}`` to ``<lake_root>/.event_ledger.jsonl``
    (empty when ``lake_root`` is None).  Enforcement is per event:

      * an already-committed ``event_id`` is a duplicate retry (skipped);
      * an in-flight ``pending`` record is a concurrent worker's reservation
        (skipped as ``"in_flight"``);
      * ``event.snapshot_before`` must equal the chain's **latest** committed
        ``snapshot_after``（不是第一个匹配记录），且 ``sequence`` 必须严格递增，
        否则事件 stale / out-of-order（R14 #5 修 chain-head 选择）；
      * only after ALL downstream factors succeed is ``snapshot_after``
        committed — a partial success leaves the event ``pending``/``failed``.

    R14 #5（transactional 语义）：
      * ``begin``/``commit``/``reject`` 都在 ``<path>.lock`` flock 内
        reload + append —— 并发 worker 同事件只有一个能 begin（``pending`` 即
        预留），append 串行化、crash 不吞 OSError；
      * 文件损坏 fail loud（``DataEventLedgerCorruptionError``），仅尾部半行
        （崩溃 mid-append）截断恢复；
      * ``sequence`` 参与 chain-head 选择与乱序校验。
    """

    def __init__(self, lake_root: str | Path | None = None) -> None:
        self._path: Path | None = None
        if lake_root is not None:
            self._path = Path(lake_root) / ".event_ledger.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = (
            Path(str(self._path) + ".lock") if self._path is not None else None
        )
        self._records: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._path is None or not self._path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        try:
            data = self._path.read_bytes()
        except Exception as exc:  # pragma: no cover - 不可读即 fail loud
            raise DataEventLedgerCorruptionError(
                f"ledger 无法读取 {self._path}: {type(exc).__name__}: {exc}"
            ) from exc
        offset = 0
        for i, line in enumerate(data.split(b"\n")):
            if not line.strip():
                offset += len(line) + 1
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 最后一条非空行 = 崩溃 mid-append 的尾部半行 → 截断恢复；
                # 中段损坏 → fail loud，绝不静默跳过（会把顺序/幂等伪装成内存态）。
                if i >= len(data.split(b"\n")) - 2:
                    with open(self._path, "r+b") as fh:
                        fh.truncate(offset)
                    break
                raise DataEventLedgerCorruptionError(
                    f"ledger 中段 JSON 损坏（offset={offset}）: "
                    f"{line[:80]!r} —— 顺序/幂等保证无法恢复，请人工修复或清除 "
                    f"ledger"
                ) from None
            eid = rec.get("event_id")
            if eid:
                records[str(eid)] = rec
            offset += len(line) + 1
        return records

    def _reload(self) -> None:
        self._records = self._load()

    def _append_unlocked(self, rec: dict[str, Any]) -> None:
        self._records[str(rec.get("event_id"))] = rec
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def is_committed(self, event_id: str) -> bool:
        return self._records.get(str(event_id), {}).get("status") == "committed"

    def _last_committed_unlocked(
        self, dataset: str, field: str
    ) -> tuple[Any | None, int | None]:
        """返回 (snapshot_after, sequence) 的**最新 chain head**（R14 #5）。

        旧实现遍历后返回第一个匹配的 committed record——事件链 e1→e2→e3 时会把
        e1 的 snapshot 当 chain head，把合法的 ``snapshot_before=s2`` 误判 stale。
        现在取 sequence 最大者；sequence 为空时取文件顺序最后者。
        """
        best_snap: Any | None = None
        best_seq: int | None = None
        best_order = -1
        order = 0
        for rec in self._records.values():
            if (
                rec.get("status") == "committed"
                and str(rec.get("dataset")) == str(dataset)
                and str(rec.get("field")) == str(field)
            ):
                seq = rec.get("sequence")
                seqk = int(seq) if seq is not None else -1
                if seqk > (best_seq if best_seq is not None else -1) or (
                    seqk == (best_seq if best_seq is not None else -1)
                    and order > best_order
                ):
                    best_seq = seqk if seq is not None else None
                    best_snap = rec.get("snapshot_after")
                    best_order = order
            order += 1
        return best_snap, best_seq

    def last_committed_snapshot(self, dataset: str, field: str) -> Any | None:
        snap, _seq = self._last_committed_unlocked(dataset, field)
        return snap

    def begin(self, event: DataEvent) -> str:
        """Validate ordering / idempotency; mark ``pending``.  Returns
        ``"duplicate"`` when already committed, ``"in_flight"`` when another
        worker holds a pending reservation, ``"ok"`` otherwise."""
        if not event.event_id:
            return "ok"
        if self._path is None:
            if self.is_committed(event.event_id):
                return "duplicate"
            self._append_unlocked(self._pending_record(event))
            return "ok"
        with _ledger_lock(self._lock_path):
            self._reload()
            rec = self._records.get(str(event.event_id)) or {}
            status = rec.get("status")
            if status == "committed":
                return "duplicate"
            if status == "pending":
                # R14 #5 P1 lease：pending 预留 lease 过期 → 其他 worker 可
                # takeover（worker 在 begin 后 crash 不再永久 ``in_flight``）。
                if _ledger_pending_is_stale(rec):
                    self._append_unlocked(self._pending_record(event))
                    return "ok"
                return "in_flight"
            if event.snapshot_before is not None:
                last, _seq = self._last_committed_unlocked(event.dataset, event.field)
                if last is not None and str(last) != str(event.snapshot_before):
                    raise DataEventStaleError(
                        f"event {event.event_id!r}: snapshot_before="
                        f"{event.snapshot_before!r} != ledger last committed "
                        f"snapshot {last!r} for ({event.dataset}, {event.field}) — "
                        f"stale / out-of-order"
                    )
            if event.sequence is not None:
                _last, last_seq = self._last_committed_unlocked(
                    event.dataset, event.field
                )
                if last_seq is not None and event.sequence <= last_seq:
                    raise DataEventStaleError(
                        f"event {event.event_id!r}: sequence={event.sequence} <= "
                        f"chain head sequence {last_seq} for "
                        f"({event.dataset}, {event.field}) — 乱序 / 重复事件"
                    )
            self._append_unlocked(self._pending_record(event))
            return "ok"

    @staticmethod
    def _pending_record(event: DataEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "field": event.field,
            "sequence": event.sequence,
            "status": "pending",
            "snapshot_before": event.snapshot_before,
            "snapshot_after": None,
            "received_at": event.updated_date,
            # R14 #5 P1 lease：owner / attempt_id / reserved_at 使 stale pending 可被
            # 其他 worker takeover（worker 在 begin 后 crash 不再永久卡死）。
            "owner": event.owner,
            "attempt_id": event.attempt_id,
            "reserved_at": _ledger_now_iso(),
        }

    def commit(self, event: DataEvent) -> None:
        if not event.event_id:
            return
        rec = {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "field": event.field,
            "sequence": event.sequence,
            "status": "committed",
            "snapshot_before": event.snapshot_before,
            "snapshot_after": event.snapshot_after,
            "received_at": event.updated_date,
        }
        if self._path is None:
            self._append_unlocked(rec)
            return
        with _ledger_lock(self._lock_path):
            self._reload()
            self._append_unlocked(rec)

    def reject(self, event: DataEvent, reason: str) -> None:
        if not event.event_id:
            return
        rec = {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "field": event.field,
            "sequence": event.sequence,
            "status": "rejected",
            "reason": reason,
            "received_at": event.updated_date,
        }
        if self._path is None:
            self._append_unlocked(rec)
            return
        with _ledger_lock(self._lock_path):
            self._reload()
            self._append_unlocked(rec)


def _is_production_run(event: DataEvent) -> bool:
    """Effective production mode for an event (P0-19/P0-20).

    An explicit ``event.run_mode == "production"`` (when a producer sets it)
    wins; otherwise the global production policy decides.  Default research,
    so existing callers keep their behavior.
    """
    mode = getattr(event, "run_mode", None)
    if mode and str(mode).strip().lower() in {"production", "prod"}:
        return True
    try:
        from runtime.production_policy import is_production_mode

        return bool(is_production_mode())
    except Exception:  # pragma: no cover - 策略解析失败按 research 处理
        return False


def execute_incremental_updates_from_event(
    engine: Any,
    event: DataEvent | dict[str, Any],
    *,
    lake_root: str | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    dry_run: bool = False,
    materialize_kwargs: dict[str, Any] | None = None,
    engine_factory: Any | None = None,
) -> dict[str, Any]:
    """DataEvent → 计划 → 逐因子物化（每个因子使用自己的重建引擎，R10 #51）。

    P0-18/P0-20: an event with an ``event_id`` passes through a ``DataEventLedger``
    (idempotent retries skipped, out-of-order ``snapshot_before`` rejected) and is
    committed only when ALL downstream factors succeed.  Production forbids
    partial-success events: any failure raises ``PartialIncrementalFailureError``
    and the event stays pending.
    """
    from storage.materializer import ParquetMaterializer
    from runtime.dependency_catalog import DependencyCatalog

    event = normalize_data_event(event)
    production = _is_production_run(event)
    ledger = DataEventLedger(lake_root=lake_root)
    ledger_status = ledger.begin(event) if event.event_id else "ok"
    if ledger_status in ("duplicate", "in_flight"):
        # Idempotent retry of an already-committed event, or another worker's
        # in-flight reservation (R14 #5): nothing to do.  Distinguish the two in
        # ``ledger_status`` — old code collapsed both to "duplicate", hiding the
        # concurrent-reservation state.
        return {
            "event": event.to_dict(),
            "plans": [],
            "factor_count": 0,
            "dry_run": dry_run,
            "materializations": {},
            "ledger_status": ledger_status,
            "succeeded": 0,
            "failed": 0,
        }
    catalog = ParquetMaterializer(lake_root=lake_root).catalog
    dep_catalog = DependencyCatalog(catalog)
    plans = plan_updates_from_data_event(
        dep_catalog,
        event,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
        production=production,
    )
    out: dict[str, Any] = {
        "event": event.to_dict(),
        "plans": [p.to_dict() for p in plans],
        "factor_count": len(plans),
        "dry_run": dry_run,
        "materializations": {},
        "ledger_status": ledger_status,
    }
    if dry_run or not plans:
        if event.event_id:
            ledger.reject(event, "dry_run_or_no_plans")
            out["ledger_status"] = "rejected"
        return out

    mat_kwargs = dict(materialize_kwargs or {})
    mat_kwargs.setdefault("lake_root", lake_root)
    mat_kwargs.setdefault("lookback_extra", lookback_extra)
    mat_kwargs.setdefault("market", market)
    # R14 #5a：production 事件 = 两阶段原子发布。先把**所有**因子写到 staging
    # （``materialize`` 对 production+local 直接拒绝；staging 只落暂存区、权威
    # 水位线 defer），**全部 stage 成功后才逐因子 publish**（staging→published +
    # 推进水位线）。任一 stage 失败 → 直接 reject，published 因子湖完全没动——
    # **绝无 mixed generation**。调用方显式传了 write_target 则尊重（如
    # staging_clickhouse）。
    if production:
        mat_kwargs.setdefault("write_target", "staging")

    failures: list[dict[str, str]] = []
    for plan in plans:
        info = dep_catalog.full_factor_definition(plan.factor_id)
        if info is None:
            failures.append(
                {
                    "factor_id": plan.factor_id,
                    "error": "catalog 中无 factor_registry 记录",
                }
            )
            continue
        try:
            factor = factor_from_catalog_info({**info, "factor_id": plan.factor_id})
        except ValueError as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})
            continue
        try:
            # R10 #51: rebuild a dedicated engine from the catalog's full spec.
            ds_config = info.get("data_source_config") or {}
            if ds_config:
                target_engine = _build_engine_for_factor(
                    info,
                    lake_root=lake_root,
                    market=market,
                    engine_factory=engine_factory,
                )
                _verify_factor_semantic_identity(
                    factor, info, production=production
                )
            else:
                # Legacy rows carry no source config — fall back to the caller's
                # engine so existing catalog content keeps materializing.
                target_engine = engine
            result = target_engine.materialize_incremental(
                factor,
                factor_id=plan.factor_id,
                since=plan.since,
                end_date=plan.end_date or end_date,
                expression=info.get("expression"),
                frequency=info.get("frequency"),
                description=info.get("description"),
                deleted_keys=plan.deleted_keys,
                **mat_kwargs,
            )
            out["materializations"][plan.factor_id] = result.get(
                "materialization", result
            )
        except Exception as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})

    if failures:
        out["failures"] = failures
        if event.event_id:
            ledger.reject(
                event, f"{len(failures)} downstream factor(s) failed"
            )
            out["ledger_status"] = "failed"
        if production:
            # P0-20: production events may NOT silently end partial — a partial
            # snapshot in the lake (some factors new, some old) is data corruption.
            raise PartialIncrementalFailureError(
                f"production: event {event.event_id or event.updated_date!r} had "
                f"{len(failures)} failed downstream factor(s): "
                + "; ".join(f"{f['factor_id']}: {f['error']}" for f in failures[:5])
            )
        out["succeeded"] = len(out["materializations"])
        out["failed"] = len(failures)
        return out

    # R14 #5a：production 事件 = 两阶段原子发布。所有因子 **stage 成功** 后才
    # 逐因子 publish（staging → published 因子湖 + 推进权威水位线）；任一 stage
    # 失败在上面直接 reject，published 湖完全没动 —— 绝无 mixed generation。
    # publish 本身失败同样 fail-closed：事件保持 rejected/pending，重跑同一
    # event_id 幂等收敛，最终只 commit 一次。
    published: list[str] = []
    if production and not dry_run:
        from storage.materialize.lake_publish import publish_factor_lake

        publish_failures: list[dict[str, str]] = []
        for plan in plans:
            try:
                publish_factor_lake(
                    factor_id=plan.factor_id,
                    lake_root=lake_root,
                    approve=True,
                    sync_from_local=False,
                    reconcile=False,
                )
                published.append(plan.factor_id)
            except Exception as exc:
                publish_failures.append(
                    {"factor_id": plan.factor_id, "error": str(exc)}
                )
        if publish_failures:
            out["failures"] = publish_failures
            out["published"] = published
            if event.event_id:
                ledger.reject(
                    event, f"{len(publish_failures)} factor(s) publish failed"
                )
                out["ledger_status"] = "failed"
            raise PartialIncrementalFailureError(
                f"production: event {event.event_id or event.updated_date!r} publish "
                f"failed for {len(publish_failures)} factor(s): "
                + "; ".join(
                    f"{f['factor_id']}: {f['error']}" for f in publish_failures[:5]
                )
            )
        out["published"] = published

    if event.event_id:
        ledger.commit(event)
        out["ledger_status"] = "committed"
    out["succeeded"] = len(out["materializations"])
    out["failed"] = len(failures)
    return out
