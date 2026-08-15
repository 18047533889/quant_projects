# -*- coding: utf-8 -*-
"""Correct point-in-time event reads and latest-period state selection."""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data_access.core.exceptions import (
    AvailabilityLatencyError,
    SemanticCatalogUnavailableError,
    ValidationError,
)
from data_access.read.temporal_join import availability_uses_calendar
from .cos_contract import COSDatasetContract, resolve_event_clock, validate_event_filters
from .cos_panel_runtime import compile_filters, quote
from data_access.read.predicate import ensure_sequence_arg

logger = logging.getLogger("data_access.cos_event_runtime")


def _required(contract: COSDatasetContract, clock: str) -> list[str]:
    return list(dict.fromkeys([contract.instrument_column, clock, contract.period_column, *contract.revision_columns, *contract.event_id_columns, *contract.required_event_filters]))


def _declared_schema(self: Any, dataset: str) -> list[str]:
    """数据集 registry 声明的完整 schema 列（用于 columns=None → 全列）。"""
    reg = getattr(self, "_registry", None)
    if reg is None:
        return []
    try:
        ds = reg.get(dataset)
        schema = dict(getattr(ds, "schema", None) or {})
        return list(dict.fromkeys(str(c) for c in schema))
    except Exception:
        return []


def _allocate_column_name(used: set[str], base: str) -> str:
    """从 ``used`` 之外分配输出列名（#P0-C11 统一冲突策略）。

    base 未被占用 → base；占用 → ``base_event``；再被占 → ``base_event_2`` ...
    decisions 已有 ``x`` 且事件也有 ``x`` 时事件列改名 ``x_event``；若 decisions
    同时已有 ``x_event``，则继续递增（旧实现只改一层，会静默覆盖 ``x_event``）。
    调用方每分配一个名字必须把结果加进 ``used``（事件列重命名也可能互相撞）。
    """
    if base not in used:
        return base
    candidate = f"{base}_event"
    n = 2
    while candidate in used:
        candidate = f"{base}_event_{n}"
        n += 1
    return candidate


def read_cos_events(self: Any, dataset: str, *, columns: Sequence[str] | None = None, start: Any | None = None, end: Any | None = None, instrument_filter: Sequence[str] | None = None, event_filters: Mapping[str, Any] | None = None, allow_effective_time: bool = False, **params: Any) -> pd.DataFrame:
    contract, clock = resolve_event_clock(dataset, allow_effective_time=allow_effective_time)
    filters = validate_event_filters(contract, event_filters)
    if instrument_filter is not None:
        ensure_sequence_arg(instrument_filter, name="instrument_filter")
    required = _required(contract, clock)
    selected = None if columns is None else list(dict.fromkeys([*columns, *required]))
    projection = "*" if selected is None else ", ".join(quote(c) for c in selected)
    clauses, bind = compile_filters(filters)
    if start is not None:
        clauses.append(f"{quote(clock)} >= ?")
        bind.append(start)
    if end is not None:
        clauses.append(f"{quote(clock)} <= ?")
        bind.append(end)
    # #P0-C3 [] = 空股票池（≠ None = 全市场）：[] 必须生成 `1 = 0`（WHERE FALSE），
    # 不能走 truthiness 跳过 → 否则静默读出全市场。
    if instrument_filter is not None:
        if not instrument_filter:
            clauses.append("1 = 0")
        else:
            clauses.append(f"{quote(contract.instrument_column)} IN ({', '.join('?' for _ in instrument_filter)})")
            bind.extend(str(v) for v in instrument_filter)
    query = f"SELECT {projection} FROM {{{{{dataset}}}}}"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    physical_range = (start, end) if contract.pit_policy == "strict" and (start is not None or end is not None) else None
    # #P0-26 官方 semantic helper 不能撞 production 的 view_columns 门禁：显式传
    # 视图所需列（含 PIT 必要内部列）。本 helper 在入口已做 contract 校验
    # （resolve_event_clock / validate_event_filters），是语义层自身，跳过
    # sql() 的用户级 semantic gate（_semantic_gate=False）。
    # #P0-33 columns=None → TEMP VIEW 必须是完整 declared schema（否则 SELECT *
    # 只看得见 PIT 必需列）。
    # #P0-34 event filter 引用的列必须进 view projection——WHERE 引用但 view 里
    # 没有 → DuckDB Binder Error。
    if selected is None:
        declared = _declared_schema(self, dataset)
        if not declared:
            raise ValidationError(
                f"数据集 {dataset!r} 未声明完整物理 schema；"
                "columns=None 不能声称返回完整事件 payload"
            )
        view_cols = list(dict.fromkeys([*declared, *required]))
    else:
        view_cols = list(dict.fromkeys([*selected, *required, *filters.keys()]))
    table = self.sql(
        query,
        read_datasets=[dataset],
        read_params={dataset: params} if params else None,
        read_time_ranges={dataset: physical_range},
        view_columns={dataset: view_cols},
        params=bind,
        _semantic_gate=False,
    )
    return table.to_pandas(split_blocks=True)


def _normalize(events: pd.DataFrame, contract: COSDatasetContract, clock: str) -> pd.DataFrame:
    required = {contract.instrument_column, clock}
    if contract.period_column:
        required.add(contract.period_column)
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValidationError(f"数据集 {contract.name!r} 缺少 PIT 必要列 {missing}")
    out = events.copy()
    out[contract.instrument_column] = out[contract.instrument_column].astype("string").str.strip()
    if out[contract.instrument_column].isna().any() or (out[contract.instrument_column] == "").any():
        raise ValidationError(f"数据集 {contract.name!r} 含空 instrument key")
    out[clock] = pd.to_datetime(out[clock], errors="coerce", utc=True)
    if out[clock].isna().any():
        raise ValidationError(f"数据集 {contract.name!r}.{clock} 含非法时间")
    if contract.period_column:
        period = contract.period_column
        out[period] = pd.to_datetime(out[period], errors="coerce", utc=True)
        if out[period].isna().any():
            raise ValidationError(f"数据集 {contract.name!r}.{period} 含非法时间")
        if contract.availability_must_follow_period and (out[clock] < out[period]).any():
            raise ValidationError(f"数据集 {contract.name!r} 存在可知时间早于报告期结束时间")
    out = out.drop_duplicates().copy()
    key = [contract.instrument_column, clock]
    if contract.period_column:
        key.append(contract.period_column)
    key.extend(c for c in contract.required_event_filters if c in out.columns)
    key.extend(c for c in contract.event_id_columns if c in out.columns)
    revisions = [c for c in contract.revision_columns if c in out.columns]
    duplicate_mask = out.duplicated(key, keep=False)
    if duplicate_mask.any():
        if not revisions:
            raise ValidationError(f"数据集 {contract.name!r} 同一 PIT vintage 冲突重复且无 revision 列")
        for revision in revisions:
            if out.loc[duplicate_mask, revision].isna().any():
                raise ValidationError(f"数据集 {contract.name!r} 的重复 vintage 含空 revision={revision!r}")
        out = out.sort_values([*key, *revisions], kind="mergesort").drop_duplicates(key, keep="last")
    return out


def _apply_latency_minutes(avail: Any, latency: int | None) -> Any:
    """Apply a declared minute latency, failing closed on unsupported values."""
    if not latency:
        return avail
    try:
        minutes = int(latency)
        if isinstance(avail, _dt.datetime):
            return avail + _dt.timedelta(minutes=minutes)
        if isinstance(avail, _dt.date):
            return _dt.datetime(avail.year, avail.month, avail.day) + _dt.timedelta(
                minutes=minutes
            )
        return avail + _dt.timedelta(minutes=minutes)
    except Exception as exc:
        raise AvailabilityLatencyError(
            f"availability_latency={latency!r} 无法应用到 available_from={avail!r}"
        ) from exc


def _select(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    contract: COSDatasetContract,
    decision_time: str,
    decision_instrument: str,
    clock: str,
    mode: str,
    *,
    availability: str = "same_day",
    calendar: Any = None,
    latency: int | None = None,
) -> list[pd.Series | None]:
    """#P0-12 PIT asof 选择，availability 语义与 read_joined 共用同一套。

    不再写死 ``event_clock <= decision``：先解析数据集 availability——
        - 需日历的种类（next_trading_day / next_session_open / session …
          next_bar / after_close_next_open）且有市场日历 → 用统一
          ``compile_available_from`` 把 knowledge 编译成 available_from
          （含 availability_latency 叠加），条件变 ``available_from <=
          decision``（与 read_joined 的 ``_session_avail_sql`` 同一语义）；
    Calendar-required availability is always compiled by the authoritative
    AvailabilityCompiler.  A missing/empty calendar is a typed hard failure;
    it is never approximated by comparing raw knowledge with decision time.
    """
    if mode not in {"latest_period", "latest_available"}:
        raise ValidationError("period_selection 只能是 latest_period 或 latest_available")
    requires_calendar = availability_uses_calendar(availability)
    cal_tz = getattr(calendar, "timezone", "UTC") if calendar is not None else "UTC"

    def _to_utc(avail: Any) -> Any:
        """``available_from`` 返回交易所本地 naive datetime/date → 统一 UTC aware。"""
        if isinstance(avail, _dt.datetime):
            if avail.tzinfo is None:
                return pd.Timestamp(avail).tz_localize(cal_tz).tz_convert("UTC")
            return avail
        if isinstance(avail, _dt.date):
            return pd.Timestamp(avail, tz=cal_tz).tz_convert("UTC")
        return avail

    def _visible(knowledge: Any, decision: Any) -> bool:
        if requires_calendar:
            from data_access.read.session_calendar import compile_available_from

            avail = compile_available_from(
                knowledge,
                availability,
                calendar=calendar,
                latency=None,
                strict=True,
            )
            avail = _apply_latency_minutes(avail, latency)
            return _to_utc(avail) <= decision
        avail = _apply_latency_minutes(knowledge, latency)
        return _to_utc(avail) <= decision

    chosen: list[pd.Series | None] = [None] * len(decisions)
    groups = {str(k): g.sort_values([clock, *([contract.period_column] if contract.period_column else [])], kind="mergesort") for k, g in events.groupby(contract.instrument_column, sort=False)}
    for instrument, left_group in decisions.groupby(decision_instrument, sort=False):
        right = groups.get(str(instrument))
        if right is None or right.empty:
            continue
        rows = list(right.iterrows())
        cursor = 0
        latest = None
        by_period = {}
        for _, decision in left_group.sort_values(decision_time, kind="mergesort").iterrows():
            while cursor < len(rows) and _visible(rows[cursor][1][clock], decision[decision_time]):
                latest = rows[cursor][1]
                if contract.period_column:
                    by_period[latest[contract.period_column]] = latest
                cursor += 1
            selected = by_period[max(by_period)] if mode == "latest_period" and contract.period_column and by_period else latest
            chosen[int(decision["__pit_position"])] = selected
    return chosen


# #P1-final closure 5 availability 严格度（research 冲突回退时取最严，绝不用
# same_day 静默降级）。same_day/effective 最松，next_trading_day 最严。
_AVAILABILITY_RANK = {
    "same_instant": 0,
    "same_day": 0,
    "effective_date_only": 0,
    "next_bar": 1,
    "session": 2,
    "next_session_open": 2,
    "after_close_next_open": 2,
    "next_trading_day": 3,
}


def _resolve_event_availability(
    self: Any, dataset: str
) -> tuple[str, int | None]:
    """解析事件数据集在 asof 里的 availability + availability_latency。

    与 read_joined 同源：字段级一致声明（financial 数据集标
    ``next_trading_day``）；无声明 → 契约默认 ``same_day``。

    #P1-final closure 5 fail-closed：字段级 availability **互相冲突时不再
    静默回退 same_day**——production/strict 抛 ``AmbiguousSemanticFieldError``；
    research 告警后取最严（最保守可见）的声明。latency 冲突取最大值
    （额外延迟取最大 = 最保守，绝不提前可见）。
    """
    try:
        from data_access.read.semantic_catalog import get_semantic_catalog

        catalog = get_semantic_catalog()
        storage = catalog._fields
        fields = [
            f
            for f in storage.values()
            if getattr(f, "dataset", None) == dataset
        ]
    except Exception as exc:
        if isinstance(exc, SemanticCatalogUnavailableError):
            raise
        raise SemanticCatalogUnavailableError(
            f"数据集 {dataset!r} 的权威 semantic catalog 无法加载或读取"
        ) from exc
    declared = [
        (str(f.availability), getattr(f, "availability_latency", None))
        for f in fields
        if getattr(f, "availability", None)
    ]
    if not declared:
        return "same_day", None
    avail_set = {a for a, _ in declared}
    if len(avail_set) > 1:
        from data_access.core.exceptions import AmbiguousSemanticFieldError
        from data_access.read.query_budget import is_strict_semantics

        msg = (
            f"数据集 {dataset!r} 的语义字段 availability 声明冲突"
            f"（{sorted(avail_set)}），无法确定事件可见性语义。"
            "请统一这些字段的 availability 声明。"
        )
        if is_strict_semantics():
            raise AmbiguousSemanticFieldError(msg)
        logger.warning("%s（research 取最严 availability）", msg)
        strictest = max(avail_set, key=lambda a: _AVAILABILITY_RANK.get(a, 0))
        avail = strictest
    else:
        avail = next(iter(avail_set))
    latencies = [lat for _, lat in declared if lat is not None]
    latency = max(latencies) if latencies else None
    return avail, latency


def read_cos_events_asof(self: Any, dataset: str, decisions: pd.DataFrame, *, decision_time: str = "decision_timestamp", decision_instrument: str = "instrument", columns: Sequence[str] | None = None, event_filters: Mapping[str, Any] | None = None, max_age_days: int | None = None, period_selection: str = "latest_period", allow_effective_time: bool = False, **params: Any) -> pd.DataFrame:
    contract, clock = resolve_event_clock(dataset, allow_effective_time=allow_effective_time)
    if contract.pit_policy != "strict":
        raise ValidationError(f"数据集 {dataset!r} 只有事件生效时间；一对一 as-of 会丢失多事件/累计语义，请先 read_cos_events 后显式聚合")
    # #P0-25 RAW_EVENT / one-to-many 禁止 generic latest-asof：新闻等不是 state
    # table，"挑一条 latest event"会静默退化成最后一条新闻，丢失多事件/累计语义。
    # 必须显式 window 聚合（count / sentiment mean / latest N / decay / embedding）。
    if contract.is_raw_event:
        raise ValidationError(
            f"数据集 {dataset!r} 是 RAW_EVENT（one_to_many 事件流，非 state table）；"
            "禁止 generic latest-asof。请先 read_cos_events 后显式做 window "
            "aggregation（count / sentiment mean / latest N / decay / event "
            "embedding aggregation），否则 ValidationError。"
        )
    if getattr(contract, "cardinality", None) == "one_to_many":
        raise ValidationError(
            f"数据集 {dataset!r} cardinality=one_to_many，不是 state table；"
            "generic latest-asof 会静默退化。请显式 window/aggregate 后使用。"
        )
    validate_event_filters(contract, event_filters)
    if decision_time not in decisions or decision_instrument not in decisions:
        raise ValidationError(f"decisions 必须包含 {decision_time!r} 和 {decision_instrument!r}")
    # #P2-80 max_age_days 严格整数：1.9 静默截断成 1 会悄悄放宽 staleness 阈值，
    # bool 也不该当 0/1。
    if max_age_days is not None:
        if isinstance(max_age_days, bool) or not isinstance(max_age_days, int):
            raise ValidationError(
                f"max_age_days 必须是非负整数或 None，收到 {max_age_days!r}"
            )
        if max_age_days < 0:
            raise ValidationError("max_age_days 必须为非负整数或 None")
    left = decisions.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="coerce", utc=True)
    left[decision_instrument] = left[decision_instrument].astype("string").str.strip()
    if left[decision_time].isna().any() or left[decision_instrument].isna().any() or (left[decision_instrument] == "").any():
        raise ValidationError("decisions 含空/非法 PIT key")
    left["__pit_position"] = np.arange(len(left), dtype=np.int64)
    if left.empty:
        output = left.drop(columns=["__pit_position"])
        # #P0-C11 空 decisions 分支同样走统一冲突策略：不静默覆盖 decisions 已有
        # 的 fundamental_staleness_days 原列。
        used = set(output.columns)
        staleness_name = _allocate_column_name(used, "fundamental_staleness_days")
        output[staleness_name] = pd.Series(dtype="float64")
        return output
    events = read_cos_events(self, dataset, columns=columns, end=left[decision_time].max(), instrument_filter=sorted(set(left[decision_instrument].astype(str))), event_filters=event_filters, allow_effective_time=allow_effective_time, **params)
    right = _normalize(events, contract, clock)
    # #P0-12 统一 availability：与 read_joined 共用同一套语义（financial 事件表
    # 字段级标 next_trading_day → 事件下一交易日起才可见；无日历按 strict 回退）。
    # #P1-final closure 5：``_resolve_event_availability`` 现在同时解析
    # availability_latency 并注入 ``compile_available_from``（next_bar/session 等
    # 粒度 + 额外延迟全部走统一 IR）。
    availability, latency = _resolve_event_availability(self, dataset)
    market = getattr(contract, "market", None)
    calendar = (
        self.get_calendar(market)
        if market and hasattr(self, "get_calendar")
        else None
    )
    selected = _select(
        left,
        right,
        contract,
        decision_time,
        decision_instrument,
        clock,
        period_selection,
        availability=availability,
        calendar=calendar,
        latency=latency,
    )
    output = left.copy()
    event_cols = [c for c in right.columns if c != contract.instrument_column]
    # #P0-C11 统一冲突策略：事件列名分配逐层查 used（decisions 已有 x 且已有
    # x_event 时 → x_event_2，不再静默覆盖）；事件列相互之间也不撞。
    used = set(output.columns)
    names: dict[str, str] = {}
    for c in event_cols:
        names[c] = _allocate_column_name(used, c)
        used.add(names[c])
    values = {target: [] for target in names.values()}
    ages: list[float] = []
    limit = int(max_age_days) if max_age_days is not None else None
    for pos, row in enumerate(selected):
        if row is None:
            for target in values:
                values[target].append(pd.NA)
            ages.append(float("nan"))
            continue
        age = (output.iloc[pos][decision_time] - row[clock]).total_seconds() / 86400
        stale = limit is not None and age > limit
        for source, target in names.items():
            values[target].append(pd.NA if stale else row[source])
        ages.append(age)
    for target, vals in values.items():
        output[target] = vals
    # #P2-81/#P0-C11 staleness 列与 decisions 已有列、事件重命名列统一去冲突。
    staleness_name = _allocate_column_name(used, "fundamental_staleness_days")
    output[staleness_name] = ages
    return output.sort_values("__pit_position", kind="mergesort").drop(columns=["__pit_position"])


__all__ = ["read_cos_events", "read_cos_events_asof"]
