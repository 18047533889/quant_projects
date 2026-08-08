# -*- coding: utf-8 -*-
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
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.rolling_pack import register_polars_udf

# Deliberately skipped: ``arg`` sits on the ``unsafe`` surface (its
# argmax-index semantics are flagged non-production and it has never had a
# polars/SQL backend).  Registering a fallback would only widen the reach of an
# unsafe path, so it stays pandas-only.
_SKIP_UNSAFE = frozenset({"arg"})


def register_polars_gap_coverage() -> int:
    """Add a ``polars_udf`` backend to every remaining pandas-only canonical.

    Returns the number of backends registered (0 when polars is unavailable —
    ``register_polars_udf`` silently no-ops without the polars runtime).
    """
    count = 0
    for canonical in sorted(OperatorRegistry.list_canonical()):
        if canonical in _SKIP_UNSAFE:
            continue
        backends = set(OperatorRegistry.backends_for(canonical))
        if "pandas_numpy" not in backends or "polars" in backends:
            continue
        register_polars_udf(canonical)
        count += 1
    return count
