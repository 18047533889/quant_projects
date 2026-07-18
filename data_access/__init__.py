"""Canonical ``data_access`` package backed by the renamed ``dataaccess`` tree.

The repository directory was renamed to ``dataaccess`` while the public Python
package and all internal imports remained ``data_access``.  Python resolves
submodules through ``__path__``; point that path at the physical implementation
tree so the public import contract remains stable without duplicating code.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path

# The implementation files live in the sibling ``dataaccess`` directory.
__path__ = [str(Path(__file__).resolve().parents[1] / "dataaccess")]

from .core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from .core.exceptions import DataAccessError, DataError, EngineError, ValidationError
from .read.key_policy import KeyPolicy
from .read.query_budget import QueryBudget
from .read.read_contract import DataSnapshot, ReadResult, SqlReadResult
from .read.scan_handle import ScanHandle
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


# Importing data_access.store initializes this package first.  Expose the same
# hardened factory there so callers cannot bypass COS contracts by import path.
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

try:
    __version__ = _package_version("data-access")
except PackageNotFoundError:
    __version__ = "0.3.0+local"

# Top-level ``import dataaccess`` remains supported.  Its compatibility module
# replaces itself with this canonical module when it is imported first.
sys.modules.setdefault("dataaccess", sys.modules[__name__])
