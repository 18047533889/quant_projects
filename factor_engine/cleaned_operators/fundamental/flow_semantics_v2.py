# -*- coding: utf-8 -*-
"""Explicit reporting-flow semantics for PIT fundamental panels.

``fin_ttm_quarterly`` accepts one-period flow values.  ``fin_ttm_cumulative``
accepts fiscal-year-to-date cumulative values and requires a daily-aligned fiscal
quarter panel.  Both advance by ``period_id`` rather than by trading rows.
"""
from __future__ import annotations

from typing import Iterable

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.fiscal_strict import (
    pd_quarter_from_cumulative,
    pd_ttm_from_quarterly,
)
from factor_engine.cleaned_operators.fundamental.transforms_v2 import _pos_int


def _aligned(*frames):
    """Strict multi-panel alignment (round-6 P0-25): fail closed, never reindex.

    A period-id panel that is not on the exact same date/instrument grid as the
    value panel is a caller bug — silently reindexing would pair values with
    the wrong fiscal period.
    """
    if not frames:
        return ()
    from factor_engine.cleaned_operators.alignment import align_panel_inputs

    return align_panel_inputs(*frames, strict_axes=True)


def fin_ttm_quarterly(x, period_id, periods_per_year=4):
    """TTM over the latest consecutive single-period flow observations.

    Delegates to the strict fiscal kernel: the sequence is walked by fiscal
    ordinal (year*4+quarter) and a skipped/missing quarter fails closed
    (require_consecutive=True) instead of summing non-adjacent periods.
    """
    periods = _pos_int(periods_per_year, "periods_per_year")
    x, period_id = _aligned(x, period_id)
    return pd_ttm_from_quarterly(x, period_id, periods=periods, require_consecutive=True)


def fin_quarter_from_cumulative(x, period_id, fiscal_quarter):
    """Convert fiscal YTD cumulative values to one-quarter flow values.

    A non-Q1 period is emitted only when the immediately preceding visible report
    period carries the preceding fiscal ordinal (Q1 of YYYY cannot be subtracted
    from Q2 of a different year).  Missing/out-of-order reports fail closed instead
    of treating a cumulative value as a quarter.
    """
    x, period_id, fiscal_quarter = _aligned(x, period_id, fiscal_quarter)
    return pd_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=fiscal_quarter,
        revision_policy="latest_available",
    )


def fin_ttm_cumulative(x, period_id, fiscal_quarter, periods_per_year=4):
    """Convert fiscal YTD cumulative values to quarters, then calculate TTM."""
    periods = _pos_int(periods_per_year, "periods_per_year")
    x, period_id, fiscal_quarter = _aligned(x, period_id, fiscal_quarter)
    quarterly = pd_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=fiscal_quarter,
        revision_policy="latest_available",
    )
    return pd_ttm_from_quarterly(
        quarterly,
        period_id,
        periods=periods,
        require_consecutive=True,
        revision_policy="latest_available",
    )


def _register(name: str, params: Iterable[str], function, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "explicit_flow_semantics",
            "production_extension",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    operator = type(
        f"FundamentalFlowV2_{name}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": _calculate_series,
            "__module__": __name__,
        },
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental_flow_semantics_v2",
        backend="pandas_numpy",
        status="production",
    )(operator)


_register(
    "fin_ttm_quarterly",
    ("x", "period_id", "periods_per_year"),
    fin_ttm_quarterly,
    "TTM sum for single-report-period flow values.",
)
_register(
    "fin_quarter_from_cumulative",
    ("x", "period_id", "fiscal_quarter"),
    fin_quarter_from_cumulative,
    "Convert fiscal YTD cumulative values to one-quarter flows.",
)
_register(
    "fin_ttm_cumulative",
    ("x", "period_id", "fiscal_quarter", "periods_per_year"),
    fin_ttm_cumulative,
    "TTM for fiscal YTD cumulative values via explicit quarter conversion.",
)

# This module is imported before production-policy finalisation.  Extend the
# reviewed surface deterministically and mark the old ambiguous spelling as a
# compatibility-only semantic alias for the quarterly-flow interpretation.
from factor_engine.cleaned_operators import operator_surface as _surface
from factor_engine.cleaned_operators.registry import OperatorRegistry as _registry

_NEW = frozenset({
    "fin_ttm_quarterly",
    "fin_quarter_from_cumulative",
    "fin_ttm_cumulative",
})
_surface._FUNDAMENTAL_V2_CANONICALS = frozenset(
    set(_surface._FUNDAMENTAL_V2_CANONICALS) | set(_NEW)
)
_surface.extend_extended_only(set(_NEW))
_legacy = _registry._catalog.get("fin_ttm")
if _legacy is not None:
    _legacy["compatibility_only"] = True
    _legacy["semantic_note"] = (
        "fin_ttm assumes single-period flows; new formulas must use "
        "fin_ttm_quarterly or fin_ttm_cumulative explicitly"
    )
    _legacy["preferred_replacements"] = [
        "fin_ttm_quarterly",
        "fin_ttm_cumulative",
    ]
