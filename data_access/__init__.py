"""Public DataAccess API with executable COS semantic contracts."""
from __future__ import annotations

from typing import Any

from data_access.core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from data_access.core.exceptions import DataAccessError, DataError, EngineError, ValidationError
from data_access.read.key_policy import KeyPolicy
from data_access.read.query_budget import QueryBudget
from data_access.read.read_contract import DataSnapshot, ReadResult, SqlReadResult
from data_access.read.scan_handle import ScanHandle
from .store import DataAccessStore, get_store as _get_store, reset_store
from .cos_contract import (
    COSDatasetContract,
    COS_DATASET_CONTRACTS,
    get_cos_contract,
    install_cos_contract_methods,
    normalize_return_values,
    require_cos_contract,
    validate_panel_request,
)


def get_store(*args: Any, **kwargs: Any) -> DataAccessStore:
    """Return the shared store with strict COS panel/PIT helpers installed."""
    return install_cos_contract_methods(_get_store(*args, **kwargs))


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
    "normalize_return_values",
]

__version__ = "0.2.0"
