# -*- coding: utf-8 -*-
"""COS panel reads, factor adapters and fail-closed legacy guards."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import pyarrow as pa

from data_access.core.exceptions import ValidationError
from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.read.key_policy import resolve_key_policy
from data_access.read.predicate import ensure_sequence_arg
from .cos_contract import get_cos_contract, normalize_return_values, validate_panel_request


def quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def compile_filters(filters: Mapping[str, Any] | None) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    bind: list[Any] = []
    for key, value in dict(filters or {}).items():
        if isinstance(value, (list, tuple, set, frozenset)):
            values = list(value)
            if not values:
                clauses.append("1 = 0")
            else:
                clauses.append(f"{quote(key)} IN ({', '.join('?' for _ in values)})")
                bind.extend(values)
        elif value is None:
            clauses.append(f"{quote(key)} IS NULL")
        else:
            clauses.append(f"{quote(key)} = ?")
            bind.append(value)
    return clauses, bind


def read_cos_panel(self: Any, dataset: str, *, columns: Sequence[str] | None = None, time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, semantic_filters: Mapping[str, Any] | None = None, normalize_returns: bool = False, return_output_column: str = "return_decimal", allow_sparse: bool = False, **params: Any) -> pa.Table:
    contract = validate_panel_request(dataset, semantic_filters=semantic_filters, allow_sparse=allow_sparse)
    if instrument_filter is not None:
        ensure_sequence_arg(instrument_filter, name="instrument_filter")
    selected = list(columns) if columns is not None else None
    if normalize_returns:
        if contract.return_column is None:
            raise ValidationError(f"数据集 {dataset!r} 没有声明可归一化收益字段")
        if selected is not None and contract.return_column not in selected:
            selected.append(contract.return_column)
    filters = dict(semantic_filters or {})
    if filters:
        projection = "*" if selected is None else ", ".join(quote(c) for c in selected)
        clauses, bind = compile_filters(filters)
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
        # #P0-26 官方 semantic helper 不撞 production view_columns 门禁 + 跳过
        # user-SQL semantic gate（本 helper 入口已 validate_panel_request）。
        ds_time = None
        reg = getattr(self, "_registry", None)
        if reg is not None:
            try:
                reg_ds = reg.get(dataset)
                ds_time = getattr(reg_ds, "time_column", None)
                declared = list(dict.fromkeys(str(c) for c in (getattr(reg_ds, "schema", {}) or {}).keys()))
            except Exception:
                ds_time = None
                declared = []
        else:
            declared = []
        base_cols = [contract.instrument_column, *( [ds_time] if ds_time else [])]
        # #P0-35 columns=None → TEMP VIEW 必须完整 schema（SELECT * 才看得到全列）。
        # #P0-36 normalize_returns + columns=None → return_column 必须进 view。
        # #P0-34 filter 引用的列必须进 view projection（否则 Binder Error）。
        view_source = (
            [*declared, *base_cols]
            if selected is None
            else [*selected, *base_cols]
        )
        if normalize_returns and contract.return_column is not None:
            view_source.append(contract.return_column)
        view_source.extend(filters.keys())
        view_cols = list(dict.fromkeys(view_source))
        table = self.sql(
            query,
            read_datasets=[dataset],
            read_params={dataset: params} if params else None,
            read_time_ranges={dataset: time_range},
            view_columns={dataset: view_cols},
            params=bind,
            _semantic_gate=False,
        )
    else:
        table = self.read_arrow(dataset, columns=selected, time_range=time_range, instrument_filter=instrument_filter, **params)
    if normalize_returns:
        values = normalize_return_values(table[contract.return_column], dataset)
        if return_output_column in table.column_names:
            table = table.drop([return_output_column])
        table = table.append_column(return_output_column, values)
    return table


def load_factor_columns(self: Any, dataset: str, *, columns: Sequence[str], time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, output_names: Mapping[str, str] | None = None, semantic_filters: Mapping[str, Any] | None = None, **params: Any) -> dict[str, Any]:
    contract = validate_panel_request(dataset, semantic_filters=semantic_filters)
    ds = self._registry.get(dataset)
    requested = list(columns)
    return_output = "__return_decimal"
    replace_return = contract.return_column in requested
    table = read_cos_panel(self, dataset, columns=list(dict.fromkeys([ds.time_column, ds.instrument_column, *requested])), time_range=time_range, instrument_filter=instrument_filter, semantic_filters=semantic_filters, normalize_returns=replace_return, return_output_column=return_output, **params)
    source_value = {c: return_output if replace_return and c == contract.return_column else c for c in requested}
    targets = {c: dict(output_names or {}).get(c, c) for c in requested}
    result = arrow_table_to_multiindex_columns(table, timestamp_column=ds.time_column, instrument_column=ds.instrument_column, value_columns=[source_value[c] for c in requested], output_names={source_value[c]: targets[c] for c in requested}, key_policy=resolve_key_policy())
    return {targets[c]: result[targets[c]] for c in requested}


def guarded_load_columns(original: Any, dataset: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    contract = get_cos_contract(dataset)
    if contract is None:
        return original(dataset, **kwargs)
    columns = list(kwargs.get("columns") or [])
    if contract.is_event:
        raise ValidationError(f"{dataset!r} 是 {contract.temporal_model} 事件表；load_columns 会制造伪日频面板")
    if contract.is_sparse or contract.is_empty or contract.is_static:
        raise ValidationError(f"{dataset!r} temporal_model={contract.temporal_model}，禁止作为因子面板加载")
    if contract.required_panel_filters:
        raise ValidationError(f"{dataset!r} 必须显式提供语义过滤；请使用 load_factor_columns")
    if contract.return_column in columns and contract.return_scale != 1.0:
        raise ValidationError(f"{dataset!r}.{contract.return_column} 不是小数收益；请使用 load_factor_columns")
    return original(dataset, **kwargs)


__all__ = ["read_cos_panel", "load_factor_columns", "guarded_load_columns", "quote", "compile_filters"]
