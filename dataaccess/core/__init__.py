"""核心基础设施：引擎、异常、命名空间、审计。"""
from .engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from .exceptions import DataAccessError, DataError, EngineError, ValidationError
from .namespace import is_namespace_explicit, resolve_namespace, resolve_operator
from .retry import retry_io

__all__ = [
    "DuckDBEngine",
    "get_shared_engine",
    "reset_shared_engine",
    "DataAccessError",
    "ValidationError",
    "DataError",
    "EngineError",
    "resolve_namespace",
    "resolve_operator",
    "is_namespace_explicit",
    "retry_io",
]
