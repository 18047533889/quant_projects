# -*- coding: utf-8 -*-
"""Correct point-in-time event reads and latest-period state selection."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data_access.core.exceptions import ValidationError
from data_access.read.temporal_join import (
    availability_strict_next,
    availability_uses_calendar,
)
from .cos_contract import COSDatasetContract, resolve_event_clock, validate_event_filters
from .cos_panel_runtime import compile_filters, quote
from data_access.read.predicate import ensure_sequence_arg


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
    if instrument_filter:
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
        view_cols = list(dict.fromkeys([*_declared_schema(self, dataset), *required]))
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
) -> list[pd.Series | None]:
    """#P0-12 PIT asof 选择，availability 语义与 read_joined 共用同一套。

    不再写死 ``event_clock <= decision``：先解析数据集 availability——
        - 需日历的种类（next_trading_day / next_session_open / session …）且
          有市场日历 → 用 ``MarketCalendar.available_from`` 把 knowledge 编译成
          available_from，条件变 ``available_from <= decision``（与 read_joined
          的 ``_session_avail_sql`` 同一语义）；
        - 无日历 → 按统一 availability 回退：严格下一交易日类用 ``<``，其余
          ``<=``（与 ``TemporalJoinSpec.comparison_operator`` 同一语义）。
    """
    if mode not in {"latest_period", "latest_available"}:
        raise ValidationError("period_selection 只能是 latest_period 或 latest_available")
    strict_next = availability_strict_next(availability)
    use_calendar = (
        availability_uses_calendar(availability)
        and calendar is not None
        and bool(getattr(calendar, "has_data", False))
    )
    cal_tz = getattr(calendar, "timezone", "UTC") if use_calendar else "UTC"

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
        if use_calendar:
            return _to_utc(calendar.available_from(knowledge, availability)) <= decision
        return (knowledge < decision) if strict_next else (knowledge <= decision)

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


def _resolve_event_availability(self: Any, dataset: str) -> str:
    """解析事件数据集在 asof 里的 availability（与 read_joined 同源）。

    优先字段级一致声明（financial 数据集标 ``next_trading_day``）；否则契约
    默认 ``same_day``——与 ``_effective_join_specs`` 的「字段语义 → COS 契约
    默认」合成顺序一致。字段级互相冲突时不猜测，回退 same_day。
    """
    try:
        from data_access.read.semantic_catalog import get_semantic_catalog

        avail = {
            str(f.availability)
            for f in get_semantic_catalog()._fields.values()
            if getattr(f, "dataset", None) == dataset
            and getattr(f, "availability", None)
        }
        if len(avail) == 1:
            return next(iter(avail))
    except Exception:
        pass
    return "same_day"


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
        output["fundamental_staleness_days"] = pd.Series(dtype="float64")
        return output
    events = read_cos_events(self, dataset, columns=columns, end=left[decision_time].max(), instrument_filter=sorted(set(left[decision_instrument].astype(str))), event_filters=event_filters, allow_effective_time=allow_effective_time, **params)
    right = _normalize(events, contract, clock)
    # #P0-12 统一 availability：与 read_joined 共用同一套语义（financial 事件表
    # 字段级标 next_trading_day → 事件下一交易日起才可见；无日历按 strict 回退）。
    availability = _resolve_event_availability(self, dataset)
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
    )
    output = left.copy()
    event_cols = [c for c in right.columns if c != contract.instrument_column]
    names = {c: c if c not in output.columns else f"{c}_event" for c in event_cols}
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
    # #P2-81 保留输出名冲突防护：decisions 自身已有 fundamental_staleness_days
    # 时不能覆盖——改名 _event 后缀。
    staleness_name = "fundamental_staleness_days"
    if staleness_name in output.columns:
        staleness_name = "fundamental_staleness_days_event"
    output[staleness_name] = ages
    return output.sort_values("__pit_position", kind="mergesort").drop(columns=["__pit_position"])


__all__ = ["read_cos_events", "read_cos_events_asof"]
