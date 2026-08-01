# -*- coding: utf-8 -*-
"""Explicit reporting-flow semantics for PIT fundamental panels.

``fin_ttm_quarterly`` accepts one-period flow values.  ``fin_ttm_cumulative``
accepts fiscal-year-to-date cumulative values and requires a daily-aligned fiscal
quarter panel.  Both advance by ``period_id`` rather than by trading rows.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import (
    _period_key,
    _pos_int,
    _values,
    _walk_periods,
)


def _quarter_number(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        if text.startswith("Q"):
            text = text[1:]
        try:
            value = int(text)
        except ValueError:
            return None
    try:
        quarter = int(value)
    except (TypeError, ValueError):
        return None
    return quarter if quarter in {1, 2, 3, 4} else None


def fin_ttm_quarterly(x, period_id, periods_per_year=4):
    """Sum the latest complete set of single-period flow observations."""
    periods = _pos_int(periods_per_year, "periods_per_year")

    def calculate(order, visible, current):
        values = _values(order, visible, current, periods)
        return float(np.sum(values)) if len(values) == periods else np.nan

    return _walk_periods(x, period_id, calculate)


def fin_quarter_from_cumulative(x, period_id, fiscal_quarter):
    """Convert fiscal YTD cumulative values to one-quarter flow values.

    A non-Q1 period is emitted only when the immediately preceding visible report
    period carries the preceding fiscal-quarter number.  Missing/out-of-order
    reports fail closed instead of treating a cumulative value as a quarter.
    """
    period_id = period_id.reindex(index=x.index, columns=x.columns)
    fiscal_quarter = fiscal_quarter.reindex(index=x.index, columns=x.columns)
    output = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)

    for column in x.columns:
        order: list[object] = []
        visible: OrderedDict[object, float] = OrderedDict()
        quarters: OrderedDict[object, int] = OrderedDict()
        values = pd.to_numeric(x[column], errors="coerce").to_numpy(dtype=float)
        periods = period_id[column].to_numpy()
        quarter_values = fiscal_quarter[column].to_numpy()
        result = np.full(len(x), np.nan, dtype=float)

        for index, (value, raw_period, raw_quarter) in enumerate(
            zip(values, periods, quarter_values)
        ):
            key = _period_key(raw_period)
            quarter = _quarter_number(raw_quarter)
            if key is not None and np.isfinite(value) and quarter is not None:
                if key not in visible:
                    order.append(key)
                visible[key] = float(value)
                quarters[key] = quarter
            if key is None or key not in visible or key not in quarters:
                continue

            current_quarter = quarters[key]
            if current_quarter == 1:
                result[index] = visible[key]
                continue

            position = order.index(key)
            if position <= 0:
                continue
            previous_key = order[position - 1]
            previous_quarter = quarters.get(previous_key)
            previous_value = visible.get(previous_key, np.nan)
            if (
                previous_quarter == current_quarter - 1
                and np.isfinite(previous_value)
            ):
                result[index] = float(visible[key] - previous_value)

        output[column] = result
    return output


def fin_ttm_cumulative(x, period_id, fiscal_quarter, periods_per_year=4):
    """Convert fiscal YTD cumulative values to quarters, then calculate TTM."""
    quarterly = fin_quarter_from_cumulative(x, period_id, fiscal_quarter)
    return fin_ttm_quarterly(quarterly, period_id, periods_per_year)


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
from cleaned_operators import operator_surface as _surface
from cleaned_operators.registry import OperatorRegistry as _registry

_NEW = frozenset({
    "fin_ttm_quarterly",
    "fin_quarter_from_cumulative",
    "fin_ttm_cumulative",
})
_surface._FUNDAMENTAL_V2_CANONICALS = frozenset(
    set(_surface._FUNDAMENTAL_V2_CANONICALS) | set(_NEW)
)
_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW)
)
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
