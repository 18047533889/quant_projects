# -*- coding: utf-8 -*-
"""SameAxis contract helpers (Master Spec Part BM-259/260).

Every multi-input operator must assert that its panel inputs are aligned on the
SAME index (row / date axis) and SAME columns (instrument axis) BEFORE any
``.to_numpy()`` / ``np.stack(...)`` / positional alignment.  Two panels with
the same length but different date ORDER, or columns in a different order, are
NOT the same panel — silently aligning them by position is how cross-sectional
alpha and relation/group/regression statistics get silently wrong.

These helpers are drop-in guards: they raise :class:`SameAxisError` (a
subclass of ``ValueError`` so existing ``except ValueError`` callers stay
correct) when the contract is violated, and expose boolean predicates for the
closure audit.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


class SameAxisError(ValueError):
    """Raised when multi-input operators receive mis-aligned panel axes."""


def assert_unique_index(*frames: pd.DataFrame, context: str = "operator") -> None:
    """Every frame must have a unique (non-duplicated) row index."""
    for f in frames:
        if f.index.duplicated().any():
            raise SameAxisError(
                f"{context}: input panel has duplicated index rows "
                f"({int(f.index.duplicated().sum())} duplicated); duplicate "
                "timestamps make a trailing window ill-defined"
            )


def assert_unique_columns(*frames: pd.DataFrame, context: str = "operator") -> None:
    """Every frame must have unique (non-duplicated) instrument columns."""
    for f in frames:
        if f.columns.duplicated().any():
            raise SameAxisError(
                f"{context}: input panel has duplicated instrument columns "
                f"({int(f.columns.duplicated().sum())} duplicated)"
            )


def assert_same_index(*frames: pd.DataFrame, context: str = "operator") -> None:
    """All frames must share the EXACT same row index (values AND order).

    Length alone is not enough: ``[A,B,C]`` vs ``[A,C,B]`` is a different
    panel and must fail loudly instead of being positionally mis-aligned.
    """
    if len(frames) < 2:
        return
    ref = frames[0].index
    for f in frames[1:]:
        idx = f.index
        if len(idx) != len(ref) or not (idx == ref).all():
            raise SameAxisError(
                f"{context}: input panels are not aligned on the same row index "
                "(values or order differ)"
            )


def assert_same_columns(*frames: pd.DataFrame, context: str = "operator") -> None:
    """All frames must share the EXACT same instrument columns (values AND order).

    Column ORDER matters: rank/regression statistics index columns positionally;
    shuffling a column can silently move a stock's factor to another stock.
    """
    if len(frames) < 2:
        return
    ref = frames[0].columns
    for f in frames[1:]:
        cols = f.columns
        if len(cols) != len(ref) or not (cols == ref).all():
            raise SameAxisError(
                f"{context}: input panels are not aligned on the same instrument "
                "columns (values or order differ)"
            )


def assert_same_axes(*frames: pd.DataFrame, context: str = "operator") -> None:
    """Full SameAxis contract: unique index, unique columns, identical row axis
    and identical column axis across every frame."""
    assert_unique_index(*frames, context=context)
    assert_unique_columns(*frames, context=context)
    assert_same_index(*frames, context=context)
    assert_same_columns(*frames, context=context)


def same_index(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Boolean SameAxis predicate (for audits / tests, non-raising)."""
    if len(a.index) != len(b.index):
        return False
    return bool((a.index == b.index).all())


def same_columns(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if len(a.columns) != len(b.columns):
        return False
    return bool((a.columns == b.columns).all())


# ---------------------------------------------------------------------------
# Per-canonical axis-contract declaration (for the closure audit)
# ---------------------------------------------------------------------------

# Canonical -> tuple of required axis contracts.  Possible entries:
#   "same_index"    all panel inputs share the exact row index
#   "same_columns"  all panel inputs share the exact instrument columns
#   "unique_index"  every panel input has a unique row index
#   "unique_columns" every panel input has unique instrument columns
_AXIS_CONTRACTS: dict[str, frozenset[str]] = {}

_VALID_CONTRACTS = frozenset(
    {"same_index", "same_columns", "unique_index", "unique_columns"}
)


def declare_axis_contract(canonical: str, contract: str | Iterable[str]) -> None:
    """Declare the SameAxis contract a multi-input canonical is required to
    enforce (Master Spec Part BM-259).  ``contract`` is one or more of
    ``{"same_index","same_columns","unique_index","unique_columns"}``."""
    if isinstance(contract, str):
        items = [contract]
    else:
        items = list(contract)
    unknown = [i for i in items if i not in _VALID_CONTRACTS]
    if unknown:
        raise ValueError(
            f"unknown axis contract {unknown}; choose from "
            f"{sorted(_VALID_CONTRACTS)}"
        )
    if not items:
        raise ValueError(f"declare_axis_contract({canonical}): empty contract")
    _AXIS_CONTRACTS[str(canonical)] = frozenset(items)


def axis_contract_for(canonical: str) -> Optional[frozenset[str]]:
    return _AXIS_CONTRACTS.get(str(canonical))


def has_axis_contract(canonical: str) -> bool:
    return str(canonical) in _AXIS_CONTRACTS


def clear_axis_contracts() -> None:
    _AXIS_CONTRACTS.clear()
