# -*- coding: utf-8
"""Backend runtime 标签解析。"""
from __future__ import annotations

from typing import Any


def resolve_runtime_backend_label(backend: Any | None) -> str:
    """从 backend 实例解析 telemetry 用 canonical 名称。"""
    if backend is None:
        return "unknown"
    explicit = getattr(backend, "runtime_backend_label", None)
    if explicit:
        return str(explicit)
    cls = type(backend).__name__
    mapping = {
        "HybridLongBackend": "hybrid_long",
        "PolarsLongBackend": "polars_long",
        "SqlBackend": "duckdb_sql",
        "PolarsBackend": "polars",
        "PandasBackend": "pandas",
        "HybridBackend": "hybrid",
    }
    return mapping.get(cls, cls)
