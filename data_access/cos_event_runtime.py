# -*- coding: utf-8 -*-
"""Correct point-in-time event reads and latest-period state selection."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from data_access.core.exceptions import ValidationError
from .cos_contract import COSDatasetContract, resolve_event_clock, validate_event_filters
from .cos_panel_runtime import compile_filters, quote


def _required(contract: COSDatasetContract, clock: str) -> list[str]:
    return list(dict.fromkeys([contract.instrument_column, clock, contract.period_column, *contract.revision_columns, *contract.event_id_columns, *contract.required_event_filters]))


def read_cos_events(self: Any, dataset: str, *, columns: Sequence[str] | None = None, start: Any | None = None, end: Any | None = None, instrument_filter: Sequence[str] | None = None, event_filters: Mapping[str, Any] | None = None, allow_effective_time: bool = False, **params: Any) -> pd.DataFrame:
    contract, clock = resolve_event_clock(dataset, allow_effective_time=allow_effective_time)
    filters = validate_event_filters(contract, event_filters)
    selected = None if columns is None else list(dict.fromkeys([*columns, *_required(contract, clock)]))
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
    table = self.sql(query, read_datasets=[dataset], read_params={dataset: params} if params else None, read_time_ranges={dataset: physical_range}, params=bind)
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


def _select(decisions: pd.DataFrame, events: pd.DataFrame, contract: COSDatasetContract, decision_time: str, decision_instrument: str, clock: str, mode: str) -> list[pd.Series | None]:
    if mode not in {"latest_period", "latest_available"}:
        raise ValidationError("period_selection 只能是 latest_period 或 latest_available")
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
            while cursor < len(rows) and rows[cursor][1][clock] <= decision[decision_time]:
                latest = rows[cursor][1]
                if contract.period_column:
                    by_period[latest[contract.period_column]] = latest
                cursor += 1
            selected = by_period[max(by_period)] if mode == "latest_period" and contract.period_column and by_period else latest
            chosen[int(decision["__pit_position"])] = selected
    return chosen


def read_cos_events_asof(self: Any, dataset: str, decisions: pd.DataFrame, *, decision_time: str = "decision_timestamp", decision_instrument: str = "instrument", columns: Sequence[str] | None = None, event_filters: Mapping[str, Any] | None = None, max_age_days: int | None = None, period_selection: str = "latest_period", allow_effective_time: bool = False, **params: Any) -> pd.DataFrame:
    contract, clock = resolve_event_clock(dataset, allow_effective_time=allow_effective_time)
    if contract.pit_policy != "strict":
        raise ValidationError(f"数据集 {dataset!r} 只有事件生效时间；一对一 as-of 会丢失多事件/累计语义，请先 read_cos_events 后显式聚合")
    validate_event_filters(contract, event_filters)
    if decision_time not in decisions or decision_instrument not in decisions:
        raise ValidationError(f"decisions 必须包含 {decision_time!r} 和 {decision_instrument!r}")
    if max_age_days is not None and int(max_age_days) < 0:
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
    selected = _select(left, right, contract, decision_time, decision_instrument, clock, period_selection)
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
    output["fundamental_staleness_days"] = ages
    return output.sort_values("__pit_position", kind="mergesort").drop(columns=["__pit_position"])


__all__ = ["read_cos_events", "read_cos_events_asof"]
