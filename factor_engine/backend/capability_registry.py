# -*- coding: utf-8 -*-
"""DEPRECATED: Use backend.operator_capability instead.

This module was a facade wrapper that delegated to operator_capability.
All functionality has been merged into backend.operator_capability which is
now the single source of truth for backend capability queries.

For backward compatibility, this module re-exports the main classes and
functions from operator_capability.
"""
from __future__ import annotations

import warnings

# Re-export everything from the unified authority
from factor_engine.backend.operator_capability import (
    CAPABILITY_REGISTRY_VERSION,
    BackendCapability,
    BackendCapabilityRecord,
    BackendCapabilityRegistry,
    BackendKind,
    BackendName,
    CapabilityLevel,
    CapabilityQueryResult,
    CapabilityStatus,
    ExecutionKind,
    supports_pandas,
    supports_polars,
    supports_sql,
)

warnings.warn(
    "backend.capability_registry is deprecated. "
    "Import from backend.operator_capability instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CAPABILITY_REGISTRY_VERSION",
    "BackendCapability",
    "BackendCapabilityRecord",
    "BackendCapabilityRegistry",
    "BackendKind",
    "BackendName",
    "CapabilityLevel",
    "CapabilityQueryResult",
    "CapabilityStatus",
    "ExecutionKind",
    "supports_pandas",
    "supports_polars",
    "supports_sql",
]
