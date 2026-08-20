"""Strict scalar parameter parsing shared by Pandas and Polars operators.

R19-004: this module is a PURE re-export of the single strict scalar-parameter
authority (``cleaned_operators.common.strict_params``).  It must never contain
independent validation logic — the historical ``strict_integer`` /
``strict_finite_scalar`` were a second implementation whose acceptance domain
did NOT recognize ``np.integer`` / ``np.floating`` (an np scalar that the
authoritative ``strict_int`` / ``strict_float`` accept was rejected here),
creating planning/runtime drift.  They are now the authoritative gates under
their historical names so existing importers (``common/time_series.py``,
``common/polars_ops.py``, ``common/polars_daily_native.py``,
``common/elementwise.py``, …) keep working without a second copy.
"""
from __future__ import annotations

from cleaned_operators.common.strict_params import (
    strict_float as strict_finite_scalar,
)
from cleaned_operators.common.strict_params import (
    strict_int as strict_integer,
)

__all__ = ["strict_integer", "strict_finite_scalar"]
