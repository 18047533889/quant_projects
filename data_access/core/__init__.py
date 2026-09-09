"""核心基础设施：引擎、能力、存储、异常、命名空间、审计。"""
from .engine import (
    DuckDBEngine,
    StorageKind,
    StorageRequirement,
    get_shared_engine,
    reset_shared_engine,
)
from .duckdb_capabilities import (
    DuckDBCapabilities,
    detect_duckdb_capabilities,
    get_duckdb_capabilities,
    reset_duckdb_capabilities,
)
from .storage import (
    StorageBackend,
    StorageSpec,
    authorize_storage_path,
    is_remote_storage,
    parse_storage_spec,
    resolve_storage_for_dataset,
    storage_description,
    to_s3_uri,
)
from .exceptions import DataAccessError, DataError, EngineError, ValidationError
from .namespace import is_namespace_explicit, resolve_namespace, resolve_operator
from .retry import retry_io
from .missingness import MissingReason, MissingReasonPlane

__all__ = [
    "DuckDBEngine",
    "StorageKind",
    "StorageRequirement",
    "get_shared_engine",
    "reset_shared_engine",
    "DuckDBCapabilities",
    "detect_duckdb_capabilities",
    "get_duckdb_capabilities",
    "reset_duckdb_capabilities",
    "StorageBackend",
    "StorageSpec",
    "parse_storage_spec",
    "resolve_storage_for_dataset",
    "is_remote_storage",
    "authorize_storage_path",
    "storage_description",
    "to_s3_uri",
    "DataAccessError",
    "ValidationError",
    "DataError",
    "EngineError",
    "resolve_namespace",
    "resolve_operator",
    "is_namespace_explicit",
    "retry_io",
    "MissingReason",
    "MissingReasonPlane",
]
