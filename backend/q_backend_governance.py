# -*- coding: utf-8 -*-
"""q/K backend availability governance and adapter (MB-P2-006, MB-P2-007, MB-P2-008).

This module handles:
- MB-P2-006: q process/session/version/license availability explicit governance
- MB-P2-007: PyKX zero-copy only as optimization, not correctness requirement
- MB-P2-008: q adapter versioning for null/symbol/date/timestamp semantics
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QAvailabilityStatus(str, Enum):
    """q/K runtime availability status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LICENSE_EXPIRED = "license_expired"
    VERSION_INCOMPATIBLE = "version_incompatible"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True)
class QRuntimeInfo:
    """q/K runtime availability information (MB-P2-006)."""

    status: QAvailabilityStatus
    version: str = ""
    process_id: int = 0
    session_id: str = ""
    license_valid: bool = False
    pykx_available: bool = False
    pykx_version: str = ""
    zero_copy_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "license_valid": self.license_valid,
            "pykx_available": self.pykx_available,
            "pykx_version": self.pykx_version,
            "zero_copy_available": self.zero_copy_available,
        }


def probe_q_availability() -> QRuntimeInfo:
    """Probe q/K runtime availability (MB-P2-006).

    Returns runtime info without assuming q is available. This must be called
    before any q backend routing decision.
    """
    try:
        import pykx as kx
    except ImportError:
        return QRuntimeInfo(status=QAvailabilityStatus.NOT_CONFIGURED)

    try:
        # Check if licensed q is available
        if not kx.licensed:
            return QRuntimeInfo(
                status=QAvailabilityStatus.LICENSE_EXPIRED,
                pykx_available=True,
                pykx_version=str(getattr(kx, "__version__", "")),
            )

        # Get version
        version_result = kx.q("string .z.K")
        version = str(version_result) if version_result else ""

        # Check zero-copy support (optimization only, not required)
        zero_copy = hasattr(kx, "toq") and hasattr(kx, "q")

        return QRuntimeInfo(
            status=QAvailabilityStatus.AVAILABLE,
            version=version,
            process_id=int(kx.q(".z.i")) if kx.q else 0,
            session_id=str(kx.q(".z.u")) if kx.q else "",
            license_valid=True,
            pykx_available=True,
            pykx_version=str(getattr(kx, "__version__", "")),
            zero_copy_available=zero_copy,
        )
    except Exception as e:
        return QRuntimeInfo(
            status=QAvailabilityStatus.UNAVAILABLE,
            pykx_available=True,
            pykx_version=str(getattr(kx, "__version__", "")) if 'kx' in locals() else "",
        )


# MB-P2-008: q adapter version for null/symbol/date/timestamp semantics

Q_ADAPTER_VERSION = "v1"

@dataclass(frozen=True)
class QTypeAdapter:
    """Versioned q type adapter (MB-P2-008).

    Defines canonical mapping between FactorEngine types and q types.
    Version changes when semantics change (null handling, timestamp precision, etc).
    """

    version: str = Q_ADAPTER_VERSION

    # Null semantics
    float_null_to_nan: bool = True  # q null float → Python NaN
    int_null_to_none: bool = True   # q null int → Python None

    # Timestamp semantics
    timestamp_unit: str = "ns"  # nanosecond precision
    date_epoch: str = "2000-01-01"  # q date epoch

    # Symbol semantics
    symbol_to_str: bool = True  # q symbol → Python str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "float_null_to_nan": self.float_null_to_nan,
            "int_null_to_none": self.int_null_to_none,
            "timestamp_unit": self.timestamp_unit,
            "date_epoch": self.date_epoch,
            "symbol_to_str": self.symbol_to_str,
        }

    def adapter_hash(self) -> str:
        """Stable hash for adapter version."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# Singleton adapter instance
DEFAULT_Q_ADAPTER = QTypeAdapter()


def pandas_to_q_safe(df: Any, *, zero_copy: bool = False) -> Any:
    """Convert Pandas DataFrame to q table (MB-P2-007).

    Args:
        df: Pandas DataFrame
        zero_copy: Whether to attempt PyKX zero-copy optimization.
                   This is ONLY an optimization - correctness does not depend on it.

    Returns:
        q table

    Raises:
        ImportError: If PyKX not available
        RuntimeError: If conversion fails
    """
    try:
        import pykx as kx
    except ImportError:
        raise ImportError("PyKX required for q backend") from None

    # MB-P2-007: zero_copy is optimization only, never a correctness requirement
    if zero_copy and hasattr(kx, "toq"):
        try:
            return kx.toq(df, handle_nulls=True)
        except Exception:
            # Fall back to safe conversion if zero-copy fails
            pass

    # Safe conversion path (always works)
    return kx.q.qsql.fromtable(df, handle_nulls=True)


def q_to_pandas_safe(qtable: Any, *, zero_copy: bool = False) -> Any:
    """Convert q table to Pandas DataFrame (MB-P2-007).

    Args:
        qtable: q table
        zero_copy: Whether to attempt PyKX zero-copy optimization.
                   This is ONLY an optimization - correctness does not depend on it.

    Returns:
        Pandas DataFrame

    Raises:
        ImportError: If PyKX not available
        RuntimeError: If conversion fails
    """
    try:
        import pykx as kx
    except ImportError:
        raise ImportError("PyKX required for q backend") from None

    # MB-P2-007: zero_copy is optimization only, never a correctness requirement
    if zero_copy and hasattr(qtable, "pd"):
        try:
            return qtable.pd()
        except Exception:
            # Fall back to safe conversion if zero-copy fails
            pass

    # Safe conversion path (always works)
    return kx.q.qsql.totable(qtable).pd()


class QBackendUnavailableError(RuntimeError):
    """Raised when q backend is required but unavailable (MB-P2-006)."""

    def __init__(self, info: QRuntimeInfo):
        self.info = info
        super().__init__(f"q backend unavailable: {info.status.value}")


def require_q_available() -> QRuntimeInfo:
    """Require q backend to be available, raise if not (MB-P2-006).

    This must be called during planning if q backend is selected.
    Production must fail-fast at planning time, not silently fall back.
    """
    info = probe_q_availability()
    if info.status != QAvailabilityStatus.AVAILABLE:
        raise QBackendUnavailableError(info)
    return info
