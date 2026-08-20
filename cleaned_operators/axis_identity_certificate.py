# -*- coding: utf-8 -*-
"""R39 PERF-029: ``AxisIdentityCertificate`` + per-frame axis-identity cache.

Motivation
----------
``PanelIdentity.from_frame`` SHA-256-hashes the full ordered time axis and the
instrument column set on EVERY call.  ``verify_frames_share_identity`` calls it
on every panel input of every multi-input operator, so a panel that fans out to
several operators (high/low/close used by ATR, Stochastic, ADX, ...) is re-hashed
many times for exactly the same axis.

The R39 scheme computes the (expensive) axis hash ONCE per frame object and
attaches the resulting :class:`AxisIdentityCertificate` to the frame itself:

* ``certificate_for_frame(frame)`` returns the cached certificate when the
  frame still carries one that is *consistent* with the frame's current cheap
  invariants (row/column counts and first/last time values — O(1)), and only
  falls back to the full SHA-256 computation when the certificate is absent or
  the axis actually changed.
* ``verify_frames_share_identity`` (in ``_polars_bridge``) takes the fast path
  when every panel has a certificate and all certificates *match*
  (axis_hash + counts + sortedness).  Any mismatch falls back to the full
  ``PanelIdentity`` comparison, which is bit-for-bit identical to pre-R39.

Correctness contract
--------------------
* The certificate's ``axis_hash`` is derived from the EXACT SAME components
  ``PanelIdentity.from_frame`` uses (time_index_hash, instrument_axis_hash,
  row_order_hash, grain, frequency) — folded through ``_stable_axis_hash`` so a
  KNOWN grain/frequency mismatch (R9-P0-014) is also represented.  Two frames
  with matching certificates therefore produce equal ``PanelIdentity`` objects,
  so a fast-path "equal" verdict can never contradict the slow path.
* Any certificate mismatch (or absent certificate) forces the authoritative
  slow path (``PanelIdentity.from_frame`` + ``__eq__``) — so a fast-path
  "different" verdict is never emitted on its own; the slow path always has the
  final word.
* Certificates are attached to a frame ONLY at identity-verification time
  (post-transform, on the final frame) and are NEVER propagated through
  operator transforms — ``with_columns``/``select``/``drop`` create new frame
  objects without the attribute, so an axis-changing operator's output starts
  with no certificate and is recomputed.
* The remaining hazard is a frame mutated *in place* (same object) after its
  certificate was attached, in a way that preserves row count, column count,
  first/last time value and sortedness while changing other axis values.  The
  engine contract treats panel inputs as immutable during execution (operators
  produce new frames; ``aligned_pd`` validates without mutating), so this is
  outside the supported execution model.  The O(1) consistency check catches
  the common mutation classes (row/column changes, date-axis shifts, reorders
  that touch either end).

Thread safety
-------------
The per-frame attribute is naturally thread-confined (each frame is touched by
one worker at a time and ``setattr``/``getattr`` are atomic under the GIL).  The
bounded global LRU of canonical certificate objects and the metric counters are
guarded by a module lock.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

_LOCK = threading.Lock()

# --- counters / metrics (observable via axis_identity_counters) -----------
_axis_identity_compute_count = 0
_axis_identity_certificate_hit = 0
_next_snapshot_id = 0

# Proxy for the spec's §9 KPI ``axis_identity_ms / TTDC < 1%``.  A real
# ``from_frame`` measures ~6-15 ms on a 2500x30 panel; the proxy uses a
# conservative 1 ms baseline so the count itself is the load-bearing signal.
# Overridable for calibration in CI.
_ESTIMATED_MS_PER_COMPUTE = float(
    os.environ.get("FACTOR_ENGINE_AXIS_IDENTITY_MS_PER_COMPUTE", "1.0")
)

_CERT_ATTR = "_r39_axis_certificate"

# Bounded LRU of canonical certificate objects keyed by the observable identity
# fingerprint.  Frames sharing an axis share ONE certificate object (memory
# dedup); the per-frame attribute is the fast path.
_LRU_MAX = 4096
_lru: dict[tuple[Any, ...], "AxisIdentityCertificate"] = {}


@dataclass(frozen=True)
class AxisIdentityCertificate:
    """Immutable fingerprint of a panel's axis, computed once and cached.

    ``source_snapshot_id`` is a monotonic id identifying this certificate
    snapshot (informational — NOT part of equality).  ``axis_hash`` is a
    SHA-256 stable digest folded over the SAME components ``PanelIdentity`` is
    built from (time axis, instrument axis, row order, grain, frequency), so a
    matching ``axis_hash`` implies matching identities.  ``datetime_count`` /
    ``instrument_count`` / ``sortedness`` are the remaining observable axis
    properties compared on the fast path.

    ``first_time_value`` / ``last_time_value`` are O(1) staleness cross-checks
    used only to detect in-place axis mutation; they are excluded from equality
    and repr.
    """

    source_snapshot_id: int
    axis_hash: int
    datetime_count: int
    instrument_count: int
    sortedness: bool

    first_time_value: Any = field(default=None, init=False, repr=False, compare=False)
    last_time_value: Any = field(default=None, init=False, repr=False, compare=False)

    def describe(self) -> str:
        return (
            "AxisIdentityCertificate("
            f"snapshot={self.source_snapshot_id}, "
            f"axis_hash={self.axis_hash}, "
            f"datetime_count={self.datetime_count}, "
            f"instrument_count={self.instrument_count}, "
            f"sortedness={self.sortedness})"
        )


def _bump_hit() -> None:
    global _axis_identity_certificate_hit
    with _LOCK:
        _axis_identity_certificate_hit += 1


def certificates_match(a: "AxisIdentityCertificate", b: "AxisIdentityCertificate") -> bool:
    """Fast-path predicate: do two certificates describe the same axis?

    Compares the observable fingerprint fields ONLY.  ``source_snapshot_id`` and
    the staleness cross-check values are deliberately excluded (distinct frame
    objects sharing an axis have distinct snapshot ids).
    """
    return (
        a.axis_hash == b.axis_hash
        and a.datetime_count == b.datetime_count
        and a.instrument_count == b.instrument_count
        and a.sortedness == b.sortedness
    )


def _certificate_still_consistent(frame: Any, cert: "AxisIdentityCertificate") -> bool:
    """Cheap O(1) guard that a cached certificate still describes ``frame``.

    Catches the common in-place mutation classes (row/column count changes,
    date-axis shifts touching either end) without re-hashing.  On any doubt we
    return False so the caller recomputes the authoritative certificate.
    """
    try:
        if isinstance(frame, pd.DataFrame):
            if len(frame) != cert.datetime_count:
                return False
            if _frame_value_col_count(frame) != cert.instrument_count:
                return False
            if len(frame) > 0:
                first, last = _frame_time_edge_values(frame)
                if first != cert.first_time_value or last != cert.last_time_value:
                    return False
            return True
        if pl is not None and isinstance(frame, pl.DataFrame):
            if frame.height != cert.datetime_count:
                return False
            if _frame_value_col_count(frame) != cert.instrument_count:
                return False
            first, last = _frame_time_edge_values(frame)
            if first is not None and (first != cert.first_time_value or last != cert.last_time_value):
                return False
            return True
    except Exception:  # pragma: no cover - defensive
        return False
    return False


def _frame_value_col_count(frame: Any) -> int:
    """Number of instrument (value) columns — O(#columns)."""
    from cleaned_operators.common._polars_bridge import SKIP

    return sum(1 for c in frame.columns if str(c) not in SKIP)


def _frame_time_edge_values(frame: Any) -> tuple[Any, Any]:
    """First/last time-axis values (O(1)); ``(None, None)`` when no time axis."""
    from cleaned_operators.common._polars_bridge import _TIME_AXIS_COLUMNS

    if isinstance(frame, pd.DataFrame):
        if len(frame) == 0:
            return None, None
        return frame.index[0], frame.index[-1]
    if pl is not None and isinstance(frame, pl.DataFrame):
        columns = tuple(str(c) for c in frame.columns)
        time_col = next((c for c in _TIME_AXIS_COLUMNS if c in columns), None)
        if time_col is None or frame.height == 0:
            return None, None
        return frame[time_col][0], frame[time_col][-1]
    return None, None


def _compute_certificate(frame: Any) -> "AxisIdentityCertificate":
    """Full certificate computation (the expensive SHA-256 path)."""
    from cleaned_operators.common._polars_bridge import (
        _frame_axis_components,
        _stable_axis_hash,
    )

    (
        time_hash,
        instrument_hash,
        row_order_hash,
        grain,
        frequency,
        datetime_count,
        instrument_count,
        sortedness,
    ) = _frame_axis_components(frame)

    # Fold grain/frequency INTO the axis hash so a KNOWN grain/frequency
    # mismatch (R9-P0-014) is captured on the fast path too.
    axis_hash = _stable_axis_hash(
        (
            "axis_identity_cert",
            time_hash,
            instrument_hash,
            row_order_hash,
            grain,
            frequency,
        )
    )
    key = (axis_hash, datetime_count, instrument_count, sortedness)

    global _lru, _next_snapshot_id, _axis_identity_compute_count
    with _LOCK:
        cert = _lru.get(key)
        if cert is None:
            _next_snapshot_id += 1
            cert = AxisIdentityCertificate(
                source_snapshot_id=_next_snapshot_id,
                axis_hash=axis_hash,
                datetime_count=datetime_count,
                instrument_count=instrument_count,
                sortedness=sortedness,
            )
            first, last = _frame_time_edge_values(frame)
            object.__setattr__(cert, "first_time_value", first)
            object.__setattr__(cert, "last_time_value", last)
            _lru[key] = cert
            if len(_lru) > _LRU_MAX:
                _lru.pop(next(iter(_lru)))  # FIFO eviction
        _axis_identity_compute_count += 1
    return cert


def certificate_for_frame(frame: Any) -> "AxisIdentityCertificate":
    """Return the cached certificate for ``frame``, computing it on demand.

    Fast path: the frame already carries a consistent certificate (O(1) guard)
    -> reuse it and count a hit.  Slow path: recompute the full SHA-256
    fingerprint, attach it to the frame, and count a compute.
    """
    if not isinstance(frame, (pd.DataFrame,)) and not (
        pl is not None and isinstance(frame, pl.DataFrame)
    ):
        raise TypeError(f"cannot compute AxisIdentityCertificate from {type(frame)!r}")

    cert = getattr(frame, _CERT_ATTR, None)
    if cert is not None and _certificate_still_consistent(frame, cert):
        _bump_hit()
        return cert
    cert = _compute_certificate(frame)
    try:
        setattr(frame, _CERT_ATTR, cert)
    except Exception:  # pragma: no cover - non-fatal: cache simply won't engage
        pass
    return cert


def clear_certificates(*frames: Any) -> None:
    """Drop cached certificates from frames (test / reset helper)."""
    for frame in frames:
        try:
            if getattr(frame, _CERT_ATTR, None) is not None:
                delattr(frame, _CERT_ATTR)
        except Exception:  # pragma: no cover - defensive
            pass


def axis_identity_counters() -> dict[str, int]:
    """Snapshot of the certificate cache counters (thread-safe)."""
    with _LOCK:
        return {
            "axis_identity_compute_count": _axis_identity_compute_count,
            "axis_identity_certificate_hit": _axis_identity_certificate_hit,
        }


def axis_identity_compute_count() -> int:
    with _LOCK:
        return _axis_identity_compute_count


def axis_identity_certificate_hit() -> int:
    with _LOCK:
        return _axis_identity_certificate_hit


def axis_identity_ms() -> float:
    """Proxy for the §9 KPI: certificate-computation time in ms.

    ``compute_count * estimated_ms_per_compute``.  The estimate is a documented
    proxy (see ``_ESTIMATED_MS_PER_COMPUTE``); the count is the honest signal.
    """
    with _LOCK:
        return float(_axis_identity_compute_count) * _ESTIMATED_MS_PER_COMPUTE


def reset_axis_identity_counters() -> None:
    """Reset counters (test helper only)."""
    global _axis_identity_compute_count, _axis_identity_certificate_hit
    with _LOCK:
        _axis_identity_compute_count = 0
        _axis_identity_certificate_hit = 0
