"""Team DataAccess API with executable COS panel and PIT contracts."""
from __future__ import annotations

from data_access.core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from data_access.core.exceptions import DataAccessError, DataError, EngineError, ValidationError
from data_access.read.key_policy import KeyPolicy
from data_access.read.query_budget import QueryBudget
from data_access.read.read_contract import DataSnapshot, ReadResult, SqlReadResult
from data_access.read.scan_handle import ScanHandle
from . import store as _store_module
from .store import DataAccessStore, get_store as _get_store, reset_store
from .cos_contract import (
    COSDatasetContract,
    COS_DATASET_CONTRACTS,
    get_cos_contract,
    normalize_return_values,
    require_cos_contract,
    resolve_event_clock,
    semantic_contract_fingerprint,
    validate_panel_request,
)
from .cos_runtime import install_cos_runtime


def get_store() -> DataAccessStore:
    """Return the process store with COS semantics installed exactly once."""
    return install_cos_runtime(_get_store())


# Importing data_access.store still initializes this package first; expose the
# same hardened factory there so callers cannot bypass COS contracts by import path.
_store_module.get_store = get_store


__all__ = [
    "get_store",
    "reset_store",
    "get_shared_engine",
    "reset_shared_engine",
    "DataAccessStore",
    "DuckDBEngine",
    "DataAccessError",
    "ValidationError",
    "DataError",
    "EngineError",
    "QueryBudget",
    "KeyPolicy",
    "DataSnapshot",
    "ReadResult",
    "SqlReadResult",
    "ScanHandle",
    "COSDatasetContract",
    "COS_DATASET_CONTRACTS",
    "get_cos_contract",
    "require_cos_contract",
    "validate_panel_request",
    "resolve_event_clock",
    "normalize_return_values",
    "semantic_contract_fingerprint",
]

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("data-access")
except PackageNotFoundError:
    __version__ = "0.3.1+local"
