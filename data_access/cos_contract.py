# -*- coding: utf-8 -*-
"""Executable COS dataset semantics for DataAccess.

The physical registry answers *where* data lives.  This module answers *what one
row means* and which read/join operation is safe.  It is deliberately fail
closed for sparse placeholders and event tables so a daily factor pipeline
cannot silently turn an event feed into a false panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class COSDatasetContract:
    name: str
    market: str
    time_model: str
    join_policy: str
    instrument_column: str
    event_time_column: str | None = None
    period_column: str | None = None
    revision_columns: tuple[str, ...] = ()
    return_column: str | None = None
    return_scale: float = 1.0
    adjustment_column: str | None = None
    adjustment_convention: str | None = None
    required_filters: tuple[tuple[str, Any], ...] = ()
    panel_allowed: bool = True
    note: str = ""

    @property
    def is_event(self) -> bool:
        return self.time_model in {"E1", "E2"}

    @property
    def is_sparse(self) -> bool:
        return self.time_model == "X0"

    @property
    def is_empty(self) -> bool:
        return self.time_model == "EMPTY"


def _c(name: str, market: str, model: str, join: str, instrument: str, **kwargs: Any) -> COSDatasetContract:
    panel_allowed = bool(kwargs.pop("panel_allowed", model not in {"E1", "E2", "X0", "EMPTY", "STATIC"}))
    return COSDatasetContract(name=name, market=market, time_model=model, join_policy=join, instrument_column=instrument, panel_allowed=panel_allowed, **kwargs)


COS_DATASET_CONTRACTS: dict[str, COSDatasetContract] = {
    "ashare_calendar": _c("ashare_calendar", "ashare", "STATIC", "read_full", "TradeDate"),
    "ashare_stock_daily": _c("ashare_stock_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1.0 / 10000.0, adjustment_column="Factor", adjustment_convention="forward_vendor_factor"),
    "ashare_stock_minute": _c("ashare_stock_minute", "ashare", "MINUTE", "equi", "Symbol"),
    "ashare_stock_list": _c("ashare_stock_list", "ashare", "D1", "equi", "Symbol"),
    "ashare_stock_status": _c("ashare_stock_status", "ashare", "S1", "equi", "Symbol"),
    "ashare_stock_industry": _c("ashare_stock_industry", "ashare", "D1", "equi", "Symbol", required_filters=(("IndustrySource", "sw_l1"),)),
    "ashare_stock_valuation_daily": _c("ashare_stock_valuation_daily", "ashare", "D1", "equi", "Symbol"),
    "ashare_stock_capital_daily": _c("ashare_stock_capital_daily", "ashare", "S1", "equi", "Symbol"),
    "ashare_etf_daily": _c("ashare_etf_daily", "ashare", "D1", "equi", "Symbol"),
    "ashare_etf_list": _c("ashare_etf_list", "ashare", "D1", "equi", "Symbol"),
    "ashare_index_daily": _c("ashare_index_daily", "ashare", "D1", "equi", "Symbol", return_column="Return", return_scale=1.0 / 10000.0),
    "ashare_index_list": _c("ashare_index_list", "ashare", "D1", "equi", "Symbol"),
    "ashare_index_constituent": _c("ashare_index_constituent", "ashare", "D1", "equi", "Symbol"),
    "ashare_universe_daily": _c("ashare_universe_daily", "ashare", "D1", "equi", "Symbol"),
    "ashare_stock_topten_shareholder": _c("ashare_stock_topten_shareholder", "ashare", "S1", "equi", "Symbol"),
    "ashare_stock_topten_float_shareholder": _c("ashare_stock_topten_float_shareholder", "ashare", "S1", "equi", "Symbol"),
    "ashare_stock_balance": _c("ashare_stock_balance", "ashare", "E1", "asof", "Symbol", event_time_column="PubDate", period_column="ReportPeriodEndDate", revision_columns=("UpdateTime",)),
    "ashare_stock_income": _c("ashare_stock_income", "ashare", "E1", "asof", "Symbol", event_time_column="PubDate", period_column="ReportPeriodEndDate", revision_columns=("UpdateTime",)),
    "ashare_stock_cashflow": _c("ashare_stock_cashflow", "ashare", "E1", "asof", "Symbol", event_time_column="PubDate", period_column="ReportPeriodEndDate", revision_columns=("UpdateTime",)),
    "ashare_stock_indicator": _c("ashare_stock_indicator", "ashare", "E1", "asof", "Symbol", event_time_column="PubDate", period_column="ReportPeriodEndDate", revision_columns=("UpdateTime",)),
    "ashare_stock_dividend": _c("ashare_stock_dividend", "ashare", "E1", "asof", "Symbol", event_time_column="PubDate", period_column="PubDate", revision_columns=("UpdateTime",)),
    "us_calendar": _c("us_calendar", "us", "STATIC", "read_full", "trade_date"),
    "us_stock_daily": _c("us_stock_daily", "us", "D1", "equi", "Ticker", return_column="Ret", return_scale=1.0, adjustment_column="AdjFactor", adjustment_convention="backward_vendor_factor"),
    "us_stock_list": _c("us_stock_list", "us", "D1", "equi", "Symbol"),
    "us_etf_daily": _c("us_etf_daily", "us", "D1", "equi", "Ticker"),
    "us_etf_list": _c("us_etf_list", "us", "STATIC", "read_full", "ticker"),
    "us_security_master": _c("us_security_master", "us", "STATIC", "read_full", "ticker"),
    "us_ticker_map": _c("us_ticker_map", "us", "STATIC", "read_full", "ticker"),
    "us_stock_indices_components": _c("us_stock_indices_components", "us", "D1", "equi", "Symbol"),
    "us_universe_daily": _c("us_universe_daily", "us", "D1", "equi", "ticker"),
    "us_adj_factor": _c("us_adj_factor", "us", "D1", "equi", "ticker", adjustment_column="adj_factor", adjustment_convention="backward_clean_factor"),
    "us_adj_factor_clamped": _c("us_adj_factor_clamped", "us", "D1", "equi", "ticker"),
    "us_is_early_close": _c("us_is_early_close", "us", "STATIC", "read_full", "date"),
    "us_is_ticker_halt_minute": _c("us_is_ticker_halt_minute", "us", "MINUTE", "equi", "ticker"),
    "us_stocks_sip_day_aggs": _c("us_stocks_sip_day_aggs", "us", "D1", "equi", "ticker"),
    "us_stock_balance": _c("us_stock_balance", "us", "E2", "asof", "ticker", event_time_column="filing_date", period_column="period_end", revision_columns=("accepted_datetime", "UpdateTime")),
    "us_stock_income": _c("us_stock_income", "us", "E2", "asof", "ticker", event_time_column="filing_date", period_column="period_end", revision_columns=("accepted_datetime", "UpdateTime")),
    "us_stock_cashflow": _c("us_stock_cashflow", "us", "E2", "asof", "ticker", event_time_column="filing_date", period_column="period_end", revision_columns=("accepted_datetime", "UpdateTime")),
    "us_stock_dividend": _c("us_stock_dividend", "us", "E2", "asof", "ticker", event_time_column="TradeDate", period_column="ex_dividend_date", revision_columns=("id",), note="effective-date event feed; no announcement timestamp is present"),
    "us_stock_capital_daily": _c("us_stock_capital_daily", "us", "E2", "asof", "ticker", event_time_column="execution_date", period_column="execution_date", revision_columns=("id",), note="split-event feed; never interpret as a daily share-count snapshot"),
    "us_stock_indicator": _c("us_stock_indicator", "us", "X0", "refuse_panel", "ticker", panel_allowed=False, note="sparse placeholder; inspect row count before explicit raw use"),
    "us_stock_valuation_daily": _c("us_stock_valuation_daily", "us", "X0", "refuse_panel", "Ticker", panel_allowed=False, note="sparse placeholder; not a full daily cross-section"),
    "us_ticker_alias": _c("us_ticker_alias", "us", "EMPTY", "refuse", "ticker", panel_allowed=False),
    "us_stock_status": _c("us_stock_status", "us", "EMPTY", "refuse", "Symbol", panel_allowed=False),
    "us_stock_industry": _c("us_stock_industry", "us", "EMPTY", "refuse", "Symbol", panel_allowed=False),
}


def get_cos_contract(dataset: str) -> COSDatasetContract | None:
    return COS_DATASET_CONTRACTS.get(str(dataset))


def require_cos_contract(dataset: str) -> COSDatasetContract:
    contract = get_cos_contract(dataset)
    if contract is None:
        raise ValidationError(f"数据集 {dataset!r} 没有 COS 语义契约；不得推断其 panel/PIT 语义")
    return contract


def validate_panel_request(dataset: str, *, semantic_filters: Mapping[str, Any] | None = None, allow_sparse: bool = False) -> COSDatasetContract:
    contract = require_cos_contract(dataset)
    if contract.is_empty:
        raise ValidationError(f"数据集 {dataset!r} 是 EMPTY 占位表，禁止读取")
    if contract.is_event:
        raise ValidationError(f"数据集 {dataset!r} 是 {contract.time_model} 事件表；必须使用 read_cos_events_asof，不能当日频面板读取")
    if contract.is_sparse and not allow_sparse:
        raise ValidationError(f"数据集 {dataset!r} 是 X0 稀疏表，不是完整日频面板；显式诊断读取需 allow_sparse=True")
    if contract.time_model == "STATIC":
        raise ValidationError(f"数据集 {dataset!r} 是 STATIC 维表，不是 timestamp×instrument 面板")
    supplied = dict(semantic_filters or {})
    for key, expected in contract.required_filters:
        if supplied.get(key) != expected:
            raise ValidationError(f"数据集 {dataset!r} 读取前必须过滤 {key}={expected!r}；收到 {supplied.get(key)!r}")
    return contract


def normalize_return_values(values: Any, dataset: str) -> Any:
    contract = require_cos_contract(dataset)
    if contract.return_column is None:
        raise ValidationError(f"数据集 {dataset!r} 未声明收益字段")
    scale = float(contract.return_scale)
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        import pyarrow.compute as pc
        return pc.multiply(values, pa.scalar(scale, type=pa.float64()))
    return values.astype(float) * scale if hasattr(values, "astype") else np.asarray(values, dtype=float) * scale


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _read_cos_panel(self: Any, dataset: str, *, columns: Sequence[str] | None = None, time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, semantic_filters: Mapping[str, Any] | None = None, normalize_returns: bool = False, return_output_column: str = "return_decimal", allow_sparse: bool = False, **params: Any) -> pa.Table:
    contract = validate_panel_request(dataset, semantic_filters=semantic_filters, allow_sparse=allow_sparse)
    filters = dict(semantic_filters or {})
    selected = list(columns) if columns else None
    if normalize_returns:
        if contract.return_column is None:
            raise ValidationError(f"数据集 {dataset!r} 没有可归一化的收益字段")
        if selected is not None and contract.return_column not in selected:
            selected.append(contract.return_column)
    if filters or instrument_filter:
        projection = "*" if selected is None else ", ".join(_quote_identifier(c) for c in selected)
        clauses: list[str] = []
        bind: list[Any] = []
        for key, value in filters.items():
            clauses.append(f"{_quote_identifier(key)} = ?")
            bind.append(value)
        if instrument_filter:
            marks = ", ".join("?" for _ in instrument_filter)
            clauses.append(f"{_quote_identifier(contract.instrument_column)} IN ({marks})")
            bind.extend(str(x) for x in instrument_filter)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        table = self.sql(f"SELECT {projection} FROM {{{{{dataset}}}}}{where}", read_datasets=[dataset], read_time_ranges={dataset: time_range}, view_columns={dataset: selected} if selected is not None else None, params=bind)
    else:
        table = self.read_arrow(dataset, columns=selected, time_range=time_range, **params)
    if normalize_returns:
        normalized = normalize_return_values(table[contract.return_column], dataset)
        if return_output_column in table.column_names:
            table = table.drop([return_output_column])
        table = table.append_column(return_output_column, normalized)
    return table


def _deduplicate_event_vintages(events: pd.DataFrame, contract: COSDatasetContract, *, period_selection: str) -> pd.DataFrame:
    event_col = contract.event_time_column
    if event_col is None:
        raise ValidationError(f"{contract.name!r} 缺少 event_time_column 契约")
    required = {contract.instrument_column, event_col}
    if contract.period_column:
        required.add(contract.period_column)
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValidationError(f"{contract.name!r} 缺少 PIT 列: {missing}")
    out = events.copy()
    out[event_col] = pd.to_datetime(out[event_col], errors="coerce", utc=True)
    if out[event_col].isna().any():
        raise ValidationError(f"{contract.name!r}.{event_col} 含空值或非法时间")
    if contract.period_column:
        out[contract.period_column] = pd.to_datetime(out[contract.period_column], errors="coerce", utc=True)
        if out[contract.period_column].isna().any():
            raise ValidationError(f"{contract.name!r}.{contract.period_column} 含空值或非法时间")
        if (out[event_col] < out[contract.period_column]).any():
            raise ValidationError(f"{contract.name!r} 存在公告时间早于报告期结束时间，PIT风险")
    sort_cols = [contract.instrument_column, event_col]
    if contract.period_column:
        sort_cols.append(contract.period_column)
    revision_cols = [c for c in contract.revision_columns if c in out.columns]
    sort_cols.extend(revision_cols)
    out = out.sort_values(sort_cols, kind="mergesort")
    vintage_keys = [contract.instrument_column, event_col]
    if contract.period_column:
        vintage_keys.append(contract.period_column)
    if out.duplicated(vintage_keys, keep=False).any():
        if not revision_cols:
            raise ValidationError(f"{contract.name!r} 同一公告时点/报告期存在重复版本但没有 revision 列")
        out = out.drop_duplicates(vintage_keys, keep="last")
    if period_selection == "latest_period":
        out = out.sort_values(sort_cols, kind="mergesort").drop_duplicates([contract.instrument_column, event_col], keep="last")
    elif period_selection != "all":
        raise ValidationError("period_selection 只能是 'latest_period' 或 'all'")
    return out


def _read_cos_events_asof(self: Any, dataset: str, decisions: pd.DataFrame, *, decision_time: str = "decision_timestamp", decision_instrument: str = "instrument", columns: Sequence[str] | None = None, max_age_days: int | None = 180, period_selection: str = "latest_period", **params: Any) -> pd.DataFrame:
    contract = require_cos_contract(dataset)
    if not contract.is_event:
        raise ValidationError(f"数据集 {dataset!r} 的 time_model={contract.time_model}，不是 E1/E2 事件表")
    if decision_time not in decisions or decision_instrument not in decisions:
        raise ValidationError(f"decisions 必须包含 {decision_time!r} 和 {decision_instrument!r}")
    left = decisions.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    instruments = sorted({str(x) for x in left[decision_instrument].dropna().astype(str)})
    event_col = contract.event_time_column
    required_columns = [contract.instrument_column, event_col, contract.period_column, *contract.revision_columns]
    selected = list(dict.fromkeys([*(columns or []), *(c for c in required_columns if c)]))
    events = self.read_frame(dataset, columns=selected or None, time_range=(None, left[decision_time].max()), instrument_filter=instruments or None, **params)
    right = _deduplicate_event_vintages(events, contract, period_selection=period_selection)
    if period_selection == "all":
        raise ValidationError("period_selection='all' 返回多报告期事件，无法一对一 merge_asof；请直接读取事件表或使用 latest_period")
    left = left.rename(columns={decision_instrument: "__pit_instrument"})
    right = right.rename(columns={contract.instrument_column: "__pit_instrument"})
    left["__pit_instrument"] = left["__pit_instrument"].astype(str)
    right["__pit_instrument"] = right["__pit_instrument"].astype(str)
    left = left.sort_values(["__pit_instrument", decision_time], kind="mergesort")
    right = right.sort_values(["__pit_instrument", event_col], kind="mergesort")
    joined = pd.merge_asof(left, right, left_on=decision_time, right_on=event_col, by="__pit_instrument", direction="backward", allow_exact_matches=True, suffixes=("", "_event"))
    available = pd.to_datetime(joined[event_col], errors="coerce", utc=True)
    age = (joined[decision_time] - available).dt.total_seconds() / 86400.0
    joined["fundamental_staleness_days"] = age
    if max_age_days is not None:
        limit = int(max_age_days)
        if limit < 0:
            raise ValidationError("max_age_days 必须为非负整数或 None")
        stale = age > limit
        event_columns = [c for c in right.columns if c != "__pit_instrument"]
        joined.loc[stale, event_columns] = pd.NA
    return joined.rename(columns={"__pit_instrument": decision_instrument})


def install_cos_contract_methods(store: Any) -> Any:
    if getattr(store, "_cos_contract_methods_installed", False):
        return store
    store.read_cos_panel = MethodType(_read_cos_panel, store)
    store.read_cos_events_asof = MethodType(_read_cos_events_asof, store)
    store.get_cos_contract = MethodType(lambda self, dataset: require_cos_contract(dataset), store)
    store._cos_contract_methods_installed = True
    return store


__all__ = ["COSDatasetContract", "COS_DATASET_CONTRACTS", "get_cos_contract", "require_cos_contract", "validate_panel_request", "normalize_return_values", "install_cos_contract_methods"]
