# -*- coding: utf-8 -*-
"""PERF-2 telemetry counters (GO_PROMPT §6.3).

Lazy, import-safe counters for vector-bound kernel health and scalar-fallback
frequency.  These are bumped by a 1-2 line lazy-import hook in the scalar
branches of ``_core.daily_agg{,_two,_three}`` (and may also be wired into the
dispatch hooks that record a *bind failure* — see ``_vec_daily_agg*``).

Counters are plain module globals so they need no singleton plumbing and are
safe to read from both the equivalence harness and the vector-coverage report.
"""
from __future__ import annotations

import typing as t

# --- module-level counters (mutable singletons; process-wide) ---------------
_intraday_vector_kernels_bound = 0
_intraday_vector_bind_failures = 0
_intraday_scalar_fallback_count = 0


def increment_scalar_fallback() -> None:
    """Record one scalar-path fallback (one per scalar (inst, day) aggregate)."""
    global _intraday_scalar_fallback_count
    _intraday_scalar_fallback_count += 1


def set_vector_kernels_bound(n: int) -> None:
    """Synchronise the bound-kernel count seen by the report/harness."""
    global _intraday_vector_kernels_bound
    _intraday_vector_kernels_bound = int(n)


def increment_bind_failure() -> None:
    """Record a vector bind whose ``__vec__`` was missing at dispatch."""
    global _intraday_vector_bind_failures
    _intraday_vector_bind_failures += 1


def get_telemetry_snapshot() -> t.Dict[str, int]:
    """Return the §6.3 telemetry counters as a dict."""
    return {
        "intraday_vector_kernels_bound": _intraday_vector_kernels_bound,
        "intraday_vector_bind_failures": _intraday_vector_bind_failures,
        "intraday_scalar_fallback_count": _intraday_scalar_fallback_count,
    }
