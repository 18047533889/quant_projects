# -*- coding: utf-8 -*-
"""Fundamental strict-parameter contract (NEW-006).

Historical design: ``transforms_v2._pos_int`` was a truncating ``int(value)``
validator, and this module monkey-patched ``transforms_v2._pos_int`` AFTER
``expectation_v2``/``flow_semantics_v2`` had already done
``from transforms_v2 import _pos_int`` — so the same process ran BOTH a strict
and a truncating validator depending on which module had been imported first.
The patch also silently coerced ``periods=3.7 -> 3`` and ``True -> 1``.

Fix: the monkey patch is removed entirely.  ``transforms_v2._pos_int`` now
delegates to ``common.strict_params`` (the single authority), so every module
that imports it gets the SAME strict function object regardless of import order.
This module remains importable (it is listed in the loader) but performs no
mutation: it re-exports the strict gates for callers that historically imported
from here.
"""
from __future__ import annotations

from cleaned_operators.common.strict_params import (
    strict_bool,
    strict_condition_bool,
    strict_enum,
    strict_event_bool,
    strict_float,
    strict_int,
    strict_nonnegative_int,
    strict_positive_int,
    strict_probability,
    strict_signed_event,
    strict_universe_bool,
)

# Retained for backward compatibility with callers that imported ``_pos_int``
# from this module: it is now the STRICT positive-int gate (``3.7`` rejected,
# never truncated to ``3``).
_pos_int = strict_positive_int

__all__ = [
    "strict_int",
    "strict_nonnegative_int",
    "strict_positive_int",
    "strict_float",
    "strict_probability",
    "strict_bool",
    "strict_enum",
    "strict_condition_bool",
    "strict_event_bool",
    "strict_signed_event",
    "strict_universe_bool",
    "_pos_int",
]
