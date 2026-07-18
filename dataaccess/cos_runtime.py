# -*- coding: utf-8 -*-
"""Install COS semantics on the public DataAccess store."""
from __future__ import annotations

from types import MethodType
from typing import Any

from data_access.core.exceptions import ValidationError
from .cos_contract import get_cos_contract, require_cos_contract
from .cos_event_runtime import read_cos_events, read_cos_events_asof
from .cos_panel_runtime import guarded_load_columns, load_factor_columns, read_cos_panel
from .cos_registry_runtime import patch_store_registry
from .cos_storage_runtime import install_cos_storage_runtime


def guarded_read_asof(original: Any, dataset: str, kwargs: dict[str, Any]) -> Any:
    """Prevent the legacy upper-bound reader from being mistaken for event PIT."""
    contract = get_cos_contract(dataset)
    if contract is not None and contract.is_event:
        raise ValidationError(
            f"{dataset!r} 是 {contract.temporal_model} 事件表；legacy read_asof 只做时间上界过滤，"
            "不执行报告期版本选择。请使用 read_cos_events_asof。"
        )
    if contract is not None and (contract.is_sparse or contract.is_empty):
        raise ValidationError(
            f"{dataset!r} temporal_model={contract.temporal_model}，不支持 legacy read_asof"
        )
    return original(dataset, **kwargs)


def install_cos_runtime(store: Any) -> Any:
    """Install raw helpers on lightweight stores and factor guards on full stores."""
    if getattr(store, "_cos_runtime_installed", False):
        return store
    if hasattr(store, "_registry"):
        install_cos_storage_runtime()
        patch_store_registry(store)
    store.read_cos_panel = MethodType(read_cos_panel, store)
    store.read_cos_events = MethodType(read_cos_events, store)
    store.read_cos_events_asof = MethodType(read_cos_events_asof, store)
    store.get_cos_contract = MethodType(
        lambda self, dataset: require_cos_contract(dataset), store
    )
    if hasattr(store, "_registry"):
        store.load_factor_columns = MethodType(load_factor_columns, store)
    if hasattr(store, "load_columns") and hasattr(store, "_registry"):
        original = store.load_columns
        store._cos_original_load_columns = original
        store.load_columns = MethodType(
            lambda self, dataset, **kwargs: guarded_load_columns(
                original, dataset, kwargs
            ),
            store,
        )
    if hasattr(store, "read_asof"):
        original_asof = store.read_asof
        store._cos_original_read_asof = original_asof
        store.read_asof = MethodType(
            lambda self, dataset, **kwargs: guarded_read_asof(
                original_asof, dataset, kwargs
            ),
            store,
        )
    store._cos_runtime_installed = True
    return store


install_cos_contract_methods = install_cos_runtime

_read_cos_panel = read_cos_panel
_read_cos_events = read_cos_events
_read_cos_events_asof = read_cos_events_asof
_load_factor_columns = load_factor_columns
_guarded_load_columns = guarded_load_columns
_guarded_read_asof = guarded_read_asof

__all__ = [
    "install_cos_runtime",
    "install_cos_contract_methods",
    "_read_cos_panel",
    "_read_cos_events",
    "_read_cos_events_asof",
    "_load_factor_columns",
    "_guarded_load_columns",
    "_guarded_read_asof",
]
