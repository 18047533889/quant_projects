# -*- coding: utf-8 -*-
"""Final runtime installation for COS contracts."""
from __future__ import annotations

from dataclasses import replace
from types import MethodType
from typing import Any, Mapping, Sequence

import pyarrow as pa

from data_access.core.exceptions import ValidationError
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
        table = self.read_arrow(dataset, columns=selected, time_range=time_range, **params)
    if normalize_returns:
        values = base.normalize_return_values(table[contract.return_column], dataset)
        if return_output_column in table.column_names:
            table = table.drop([return_output_column])
        table = table.append_column(return_output_column, values)
    return table


def install_cos_contract_methods(store: Any) -> Any:
    base.install_cos_contract_methods(store)
    store.read_cos_panel = MethodType(_read_cos_panel, store)
    return store


__all__ = ["install_cos_contract_methods"]
