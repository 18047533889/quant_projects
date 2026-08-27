# -*- coding: utf-8
"""Polars fallback coverage for every canonical that has a pandas reference.

Registered late in :func:`cleaned_operators.load_all` (after every operator
module and the governance layers), so the canonical names and the
``pandas_numpy`` reference are the *final* post-dedupe identities.

For each canonical that still only has a ``pandas_numpy`` backend, a polars
backend is added via :func:`cleaned_operators.rolling_pack.register_polars_udf`
— an exact-parity delegation to the certified pandas reference.  This is the
same mechanism ``tail_systemic`` / ``feature_geometry`` / ``advanced_intraday``
already use for kernels with no native polars expression.

Why this matters:

* the ``auto`` router and the DuckDB/ClickHouse hybrid backends no longer fall
  through to the slow pandas path for operators that do have a working
  reference — their non-pushdown fallback becomes polars;
* every operator on the registry gains a real ``polars`` backend slot, so
  capability reports no longer mislabel them "pandas-only".

The reference backend is untouched: production admission still reads
``pandas_numpy``; the UDF is an additional accelerated/delegation slot.

.. important::

   **Topology honesty (R13 P1-64).**  The cleanup pass
   (:mod:`cleaned_operators.overhaul.cleanup`) removes *native polars bridges*
   (``source="pandas_bridge"``) that the overhaul replaces with native
   implementations.  This module runs *after* that cleanup and deliberately
   adds back a **labeled delegation layer** — every slot it registers is a
   ``polars_udf`` / pandas-delegating UDF, classified by
   :func:`backend.polars_backend_kind.polars_backend_kind` as
   ``polars_udf_pandas_delegate``.  It is **not** a native polars slot and must
   never be reported as one.  The "bridge cleanup" final state therefore
   reflects the true topology: only *native* bridges were removed; the
   delegation layer is an explicit, labeled policy decision on top.

Wave1-E backend-coverage audit note: the following canonicals are **deliberately
delegate** even though a "matching-name" polars class exists elsewhere — the
matching class is the same per-column pandas-EWM-delgate family, so wiring it
would change nothing:
  ``ts_ewm_corr`` / ``ts_ewm_cov`` (EXCEPTION R20-P0-EWM-PAIRWISE,
  ``polars_ts_rolling.TSEwmCorrNative/TSEwmCovNative`` declare
  ``ExecutionKind.POLARS_PANDAS_DELEGATE`` by design and land on
  backend='polars' only).  They are native-in-shape but delegate-in-execution;
  under the Wave1-E "no fake native kernels" rule we keep them classified as
  delegates (measured: ``canonical_polars_kind`` ==
  ``polars_udf_pandas_delegate``, count == 2 exactly).
"""
from __future__ import annotations

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.rolling_pack import register_polars_udf

# Deliberately skipped: ``arg`` sits on the ``unsafe`` surface (its
# argmax-index semantics are flagged non-production and it has never had a
# polars/SQL backend).  Registering a fallback would only widen the reach of an
# unsafe path, so it stays pandas-only.
_SKIP_UNSAFE = frozenset({"arg"})


def _stamp_delegate_meta(canonical: str) -> None:
    """Stamp the polars slot's backend_meta with the explicit delegate kind.

    R13 P0-63 / P1-84: a pandas-delegating polars slot is **never** a native
    polars implementation.  ``_attach_explicit_polars_contracts`` (the cleanup
    layer that stamps native ``execution_kind``) runs *before* this module, so
    delegate slots would otherwise report ``execution_kind="expression_native"``
    / ``"unsupported"`` to capability/coverage consumers.  Write the truthful
    physical flags so any report that reads ``backend_meta`` reports a delegate.
    """
    entry = OperatorRegistry._catalog.get(canonical)
    if entry is None:
        return
    backend_meta = dict(entry.get("backend_meta") or {})
    polars_meta = dict(backend_meta.get("polars") or {})
    polars_meta.update({
        "execution_kind": "polars_udf_pandas_delegate",
        "supports_lazy": False,
        "supports_streaming": False,
        "materializes_full_panel": True,
        "supports_nulls": True,
        "supports_nan": True,
        "supports_inf": True,
        "supports_scalar_broadcast": True,
        "supports_group": False,
        "supports_window": False,
        "supports_min_periods": False,
    })
    backend_meta["polars"] = polars_meta
    entry["backend_meta"] = backend_meta


def _reconcile_delegate_metadata() -> None:
    """Re-stamp the delegate kind on EVERY polars slot the classifier identifies
    as a pandas delegate (R13 P1-84).

    The overhaul cleanup labels every polars slot ``expression_native`` /
    ``polars_eager_native`` by scope, which mislabels pandas-delegating slots
    (``polars_geometry_math`` / ``polars_misc_utils`` / ``*_polars`` numpy-UDF
    sources) as native.  This pass runs after cleanup and after gap coverage, so
    the registry's ``backend_meta`` — the source of truth for capability reports —
    reflects the true topology: delegate slots are labelled delegate, native
    slots stay native.
    """
    from factor_engine.backend.polars_backend_kind import (
        PolarsImplementationKind,
        canonical_polars_kind,
    )

    for canonical in list(OperatorRegistry._catalog):
        if "polars" not in OperatorRegistry.backends_for(canonical):
            continue
        if canonical_polars_kind(canonical) == PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE:
            _stamp_delegate_meta(canonical)


def register_polars_gap_coverage() -> int:
    """Add a ``polars_udf`` backend to every remaining pandas-only canonical.

    Returns the number of backends **actually registered** (R13 P2-83).
    ``register_polars_udf`` silently no-ops when the polars runtime is missing;
    the count is taken from the registry post-condition so a no-op is never
    counted as a success.
    """
    count = 0
    for canonical in sorted(OperatorRegistry.list_canonical()):
        if canonical in _SKIP_UNSAFE:
            continue
        backends = set(OperatorRegistry.backends_for(canonical))
        if "pandas_numpy" not in backends or "polars" in backends:
            continue
        register_polars_udf(canonical)
        # Only a real registration (polars runtime present) counts.
        if "polars" in OperatorRegistry.backends_for(canonical):
            count += 1
    # Reconcile delegate metadata for ALL delegate-classified polars slots (both
    # the ones registered here and pre-existing *_polars numpy-UDF sources that
    # the cleanup layer mislabelled as native).
    _reconcile_delegate_metadata()
    return count


__all__ = ["register_polars_gap_coverage"]