# -*- coding: utf-8 -*-
"""Runtime installation and fail-closed factor reads for COS contracts."""
from __future__ import annotations

from dataclasses import replace
from types import MethodType
from typing import Any, Mapping, Sequence

import pyarrow as pa

from data_access.core.exceptions import ValidationError
from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.read.key_policy import resolve_key_policy
from . import cos_contract as base


for _name in ("us_stock_balance", "us_stock_income", "us_stock_cashflow"):
    base.COS_DATASET_CONTRACTS[_name] = replace(
        base.COS_DATASET_CONTRACTS[_name], revision_columns=()
    )


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _read_cos_panel(self: Any, dataset: str, *, columns: Sequence[str] | None = None, time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, semantic_filters: Mapping[str, Any] | None = None, normalize_returns: bool = False, return_output_column: str = "return_decimal", allow_sparse: bool = False, **params: Any) -> pa.Table:
    contract = base.validate_panel_request(dataset, semantic_filters=semantic_filters, allow_sparse=allow_sparse)
    selected = list(columns) if columns else None
    if normalize_returns:
        if contract.return_column is None:
            raise ValidationError(f"dataset {dataset!r} has no declared return field")
        if selected is not None and contract.return_column not in selected:
            selected.append(contract.return_column)
    filters = dict(semantic_filters or {})
    if filters or instrument_filter:
        projection = "*" if selected is None else ", ".join(_quote(c) for c in selected)
        clauses, bind = [], []
        for key, value in filters.items():
            clauses.append(f"{_quote(key)} = ?")
            bind.append(value)
        if instrument_filter:
            clauses.append(f"{_quote(contract.instrument_column)} IN ({', '.join('?' for _ in instrument_filter)})")
            bind.extend(str(value) for value in instrument_filter)
        table = self.sql(
            f"SELECT {projection} FROM {{{{{dataset}}}}} WHERE {' AND '.join(clauses)}",
            read_datasets=[dataset],
            read_params={dataset: params} if params else None,
            read_time_ranges={dataset: time_range},
            view_columns=None,
            params=bind,
        )
    else:
        table = self.read_arrow(dataset, columns=selected, time_range=time_range, instrument_filter=instrument_filter, **params)
    if normalize_returns:
        values = base.normalize_return_values(table[contract.return_column], dataset)
        if return_output_column in table.column_names:
            table = table.drop([return_output_column])
        table = table.append_column(return_output_column, values)
    return table


def _load_factor_columns(self: Any, dataset: str, *, columns: Sequence[str], time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, output_names: Mapping[str, str] | None = None, semantic_filters: Mapping[str, Any] | None = None, normalize_returns: bool = True, **params: Any) -> dict[str, Any]:
    contract = base.validate_panel_request(dataset, semantic_filters=semantic_filters)
    ds = self._registry.get(dataset)
    requested = list(columns)
    physical = list(requested)
    return_output = "__return_decimal"
    replace_return = bool(normalize_returns and contract.return_column in requested)
    axes = [ds.time_column, ds.instrument_column]
    table = self.read_cos_panel(
        dataset,
        columns=list(dict.fromkeys([*axes, *physical])),
        time_range=time_range,
        instrument_filter=instrument_filter,
        semantic_filters=semantic_filters,
        normalize_returns=replace_return,
        return_output_column=return_output,
        **params,
    )
    value_columns = [return_output if replace_return and c == contract.return_column else c for c in requested]
    names = dict(output_names or {})
    if replace_return:
        names[return_output] = names.get(contract.return_column, contract.return_column)
    result = arrow_table_to_multiindex_columns(
        table,
        timestamp_column=ds.time_column,
        instrument_column=ds.instrument_column,
        value_columns=value_columns,
        output_names=names or None,
        key_policy=resolve_key_policy(),
    )
    return {
        names.get(return_output if replace_return and source == contract.return_column else source,
                  contract.return_column if replace_return and source == contract.return_column else source):
        result[names.get(return_output if replace_return and source == contract.return_column else source,
                         contract.return_column if replace_return and source == contract.return_column else source)]
        for source in requested
    }


def _guarded_load_columns(self: Any, original: Any, dataset: str, **kwargs: Any) -> dict[str, Any]:
    contract = base.get_cos_contract(dataset)
    if contract is None:
        return original(dataset, **kwargs)
    columns = list(kwargs.get("columns") or [])
    if contract.is_event:
        raise ValidationError(f"{dataset!r} is {contract.time_model}; load_columns would create a false daily panel. Use read_cos_events_asof.")
    if contract.is_sparse or contract.is_empty or contract.time_model == "STATIC":
        raise ValidationError(f"{dataset!r} time_model={contract.time_model} cannot be loaded as a factor panel")
    if contract.required_filters:
        raise ValidationError(f"{dataset!r} requires semantic filters; use load_factor_columns(..., semantic_filters=...)")
    if contract.return_column in columns and contract.return_scale != 1.0:
        raise ValidationError(f"{dataset!r}.{contract.return_column} is not decimal return; use load_factor_columns for mandatory unit normalization")
    return original(dataset, **kwargs)


def install_cos_contract_methods(store: Any) -> Any:
    if getattr(store, "_cos_runtime_installed", False):
        return store
    base.install_cos_contract_methods(store)
    original_load_columns = store.load_columns
    store.read_cos_panel = MethodType(_read_cos_panel, store)
    store.load_factor_columns = MethodType(_load_factor_columns, store)
    store.load_columns = MethodType(
        lambda self, dataset, **kwargs: _guarded_load_columns(self, original_load_columns, dataset, **kwargs),
        store,
    )
    store._cos_runtime_installed = True
    return store


__all__ = ["install_cos_contract_methods"]
