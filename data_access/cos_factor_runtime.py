# -*- coding: utf-8 -*-
"""Factor-oriented COS read guards layered over raw COS runtime helpers."""
from __future__ import annotations

from types import MethodType
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError
from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.read.key_policy import resolve_key_policy
from . import cos_contract as base
from .cos_runtime import install_cos_contract_methods as install_base_runtime


def _load_factor_columns(self: Any, dataset: str, *, columns: Sequence[str], time_range: tuple[Any, Any] | None = None, instrument_filter: Sequence[str] | None = None, output_names: Mapping[str, str] | None = None, semantic_filters: Mapping[str, Any] | None = None, normalize_returns: bool = True, **params: Any) -> dict[str, Any]:
    contract = base.validate_panel_request(dataset, semantic_filters=semantic_filters)
    ds = self._registry.get(dataset)
    requested = list(columns)
    return_output = "__return_decimal"
    replace_return = bool(normalize_returns and contract.return_column in requested)
    table = self.read_cos_panel(
        dataset,
        columns=list(dict.fromkeys([ds.time_column, ds.instrument_column, *requested])),
        time_range=time_range,
        instrument_filter=instrument_filter,
        semantic_filters=semantic_filters,
        normalize_returns=replace_return,
        return_output_column=return_output,
        **params,
    )
    source_to_value = {
        source: return_output if replace_return and source == contract.return_column else source
        for source in requested
    }
    targets = {source: dict(output_names or {}).get(source, source) for source in requested}
    adapter_names = {source_to_value[source]: targets[source] for source in requested}
    result = arrow_table_to_multiindex_columns(
        table,
        timestamp_column=ds.time_column,
        instrument_column=ds.instrument_column,
        value_columns=[source_to_value[source] for source in requested],
        output_names=adapter_names,
        key_policy=resolve_key_policy(),
    )
    return {targets[source]: result[targets[source]] for source in requested}


def _guarded_load_columns(original: Any, dataset: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    contract = base.get_cos_contract(dataset)
    if contract is None:
        return original(dataset, **kwargs)
    columns = list(kwargs.get("columns") or [])
    if contract.is_event:
        raise ValidationError(
            f"{dataset!r} is {contract.time_model}; load_columns would create a false daily panel. "
            "Use read_cos_events_asof."
        )
    if contract.is_sparse or contract.is_empty or contract.time_model == "STATIC":
        raise ValidationError(
            f"{dataset!r} time_model={contract.time_model} cannot be loaded as a factor panel"
        )
    if contract.required_filters:
        raise ValidationError(
            f"{dataset!r} requires semantic filters; use load_factor_columns with semantic_filters"
        )
    if contract.return_column in columns and contract.return_scale != 1.0:
        raise ValidationError(
            f"{dataset!r}.{contract.return_column} is not decimal return; "
            "use load_factor_columns for mandatory unit normalization"
        )
    return original(dataset, **kwargs)


def install_cos_factor_runtime(store: Any) -> Any:
    store = install_base_runtime(store)
    if getattr(store, "_cos_factor_runtime_installed", False):
        return store
    if hasattr(store, "load_columns") and hasattr(store, "_registry"):
        original = store.load_columns
        store.load_factor_columns = MethodType(_load_factor_columns, store)
        store.load_columns = MethodType(
            lambda self, dataset, **kwargs: _guarded_load_columns(original, dataset, kwargs),
            store,
        )
    store._cos_factor_runtime_installed = True
    return store


__all__ = ["install_cos_factor_runtime"]
