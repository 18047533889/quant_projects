# -*- coding: utf-8 -*-
"""Generic PIT-aware actual/expectation and estimate-revision operators."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import (
    _pos_int,
    _value_streak_span,
    _values,
    _walk_periods,
)

_EPS = 1e-12


def _safe(numerator, denominator):
    output = numerator / denominator.replace(0, np.nan)
    return output.replace([np.inf, -np.inf], np.nan)


def fin_surprise(actual, expected, scale_base):
    return _safe(actual - expected, scale_base.abs())


def fin_surprise_zscore(actual, expected, scale_base, window_days=252):
    """Compatibility daily-observation z-score.

    For forward-filled realized actuals, prefer ``fin_surprise_event_zscore`` so
    each report period contributes once.
    """
    window = _pos_int(window_days, "window_days", 2)
    surprise = fin_surprise(actual, expected, scale_base)
    mean = surprise.shift(1).rolling(window, min_periods=window).mean()
    std = surprise.shift(1).rolling(window, min_periods=window).std()
    return _safe(surprise - mean, std)


def fin_surprise_event_zscore(
    actual,
    expected,
    scale_base,
    period_id,
    periods=8,
):
    count = _pos_int(periods, "periods", 3)
    surprise = fin_surprise(actual, expected, scale_base)

    def calculate(order, visible, current):
        values = np.asarray(_values(order, visible, current, count), dtype=float)
        if len(values) != count:
            return np.nan
        history = values[:-1]
        std = float(np.std(history, ddof=1))
        if not np.isfinite(std) or std <= _EPS:
            return np.nan
        return float((values[-1] - np.mean(history)) / std)

    return _walk_periods(surprise, period_id, calculate)


def fin_surprise_event_percentile(
    actual,
    expected,
    scale_base,
    period_id,
    periods=8,
):
    count = _pos_int(periods, "periods", 2)
    surprise = fin_surprise(actual, expected, scale_base)

    def calculate(order, visible, current):
        values = np.asarray(_values(order, visible, current, count), dtype=float)
        if len(values) != count:
            return np.nan
        current_value = values[-1]
        return float(
            (
                np.sum(values < current_value)
                + 0.5 * np.sum(values == current_value)
            )
            / len(values)
        )

    return _walk_periods(surprise, period_id, calculate)


def _same_target(expected, target_period_id):
    period = target_period_id.reindex(index=expected.index, columns=expected.columns)
    return period.eq(period.shift(1)) & period.notna()


def fin_expectation_revision(expected, target_period_id):
    return (expected - expected.shift(1)).where(
        _same_target(expected, target_period_id), 0.0
    )


def fin_expectation_revision_pct(expected, target_period_id):
    revision = _safe(expected, expected.shift(1)) - 1.0
    return revision.where(_same_target(expected, target_period_id), 0.0)


def fin_expectation_revision_speed(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    return fin_expectation_revision_pct(expected, target_period_id).rolling(
        window, min_periods=1
    ).sum()


def _revision_event(expected, target_period_id):
    revision = fin_expectation_revision(expected, target_period_id)
    return revision.ne(0) & revision.notna()


def fin_expectation_revision_count(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    return _revision_event(expected, target_period_id).astype(float).rolling(
        window, min_periods=1
    ).sum()


def fin_expectation_revision_magnitude(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    return fin_expectation_revision_pct(expected, target_period_id).abs().rolling(
        window, min_periods=1
    ).sum()


def fin_days_since_expectation_revision(
    expected,
    target_period_id,
    max_days=252,
):
    cap = _pos_int(max_days, "max_days")
    events = _revision_event(expected, target_period_id)
    output = pd.DataFrame(np.nan, index=expected.index, columns=expected.columns)
    for column in expected.columns:
        age = cap
        values: list[float] = []
        for event in events[column].fillna(False).to_numpy(dtype=bool):
            age = 0 if event else min(cap, age + 1)
            values.append(float(age))
        output[column] = values
    return output


def fin_expectation_dispersion(expected_std, expected_mean):
    return _safe(expected_std.abs(), expected_mean.abs())


def fin_actual_expectation_divergence(actual, expected, scale_base):
    return fin_surprise(actual, expected, scale_base)


def _beat_miss_streak(
    actual,
    expected,
    period_id,
    max_periods,
    *,
    beat: bool,
):
    count = _pos_int(max_periods, "max_periods", 2)
    difference = actual - expected

    def calculate(order, visible, current):
        values = _values(order, visible, current, count)
        streak = 0
        for value in values[::-1]:
            condition = value > 0 if beat else value < 0
            if not condition:
                break
            streak += 1
        return float(streak)

    return _walk_periods(difference, period_id, calculate)


def fin_beat_streak(actual, expected, period_id, max_periods=8):
    return _beat_miss_streak(
        actual, expected, period_id, max_periods, beat=True
    )


def fin_miss_streak(actual, expected, period_id, max_periods=8):
    return _beat_miss_streak(
        actual, expected, period_id, max_periods, beat=False
    )


def _register(
    name: str,
    params: Iterable[str],
    function,
    description: str,
) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=[
            "fundamental",
            "expectation",
            "pit_safe",
            "causal",
            "production_extension",
        ],
    )

    def calculate(self, *args, **kwargs):
        return function(*args, **kwargs)

    operator = type(
        f"ExpectationV2_{name}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": calculate,
            "__module__": __name__,
        },
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental_expectation_v2",
        backend="pandas_numpy",
        status="production",
    )(operator)


_SPECS = (
    (
        "fin_surprise",
        ("actual", "expected", "scale_base"),
        fin_surprise,
        "Scaled actual-minus-expectation surprise.",
    ),
    (
        "fin_surprise_zscore",
        ("actual", "expected", "scale_base", "window_days"),
        fin_surprise_zscore,
        "Daily-observation prior-window z-score of realized surprise.",
    ),
    (
        "fin_surprise_event_zscore",
        ("actual", "expected", "scale_base", "period_id", "periods"),
        fin_surprise_event_zscore,
        "Report-event surprise z-score over distinct visible periods.",
    ),
    (
        "fin_surprise_event_percentile",
        ("actual", "expected", "scale_base", "period_id", "periods"),
        fin_surprise_event_percentile,
        "Report-event surprise percentile over distinct visible periods.",
    ),
    (
        "fin_expectation_revision",
        ("expected", "target_period_id"),
        fin_expectation_revision,
        "Same-target-period change in analyst expectation.",
    ),
    (
        "fin_expectation_revision_pct",
        ("expected", "target_period_id"),
        fin_expectation_revision_pct,
        "Same-target-period percent expectation revision.",
    ),
    (
        "fin_expectation_revision_speed",
        ("expected", "target_period_id", "window_days"),
        fin_expectation_revision_speed,
        "Bounded cumulative expectation revision speed.",
    ),
    (
        "fin_expectation_revision_count",
        ("expected", "target_period_id", "window_days"),
        fin_expectation_revision_count,
        "Count of same-target estimate changes in a bounded daily window.",
    ),
    (
        "fin_expectation_revision_magnitude",
        ("expected", "target_period_id", "window_days"),
        fin_expectation_revision_magnitude,
        "Absolute expectation revision accumulated in a bounded daily window.",
    ),
    (
        "fin_days_since_expectation_revision",
        ("expected", "target_period_id", "max_days"),
        fin_days_since_expectation_revision,
        "Bounded trading days since the latest same-target estimate revision.",
    ),
    (
        "fin_expectation_dispersion",
        ("expected_std", "expected_mean"),
        fin_expectation_dispersion,
        "Consensus dispersion scaled by absolute consensus mean.",
    ),
    (
        "fin_actual_expectation_divergence",
        ("actual", "expected", "scale_base"),
        fin_actual_expectation_divergence,
        "Generic actual/expectation divergence.",
    ),
    (
        "fin_beat_streak",
        ("actual", "expected", "period_id", "max_periods"),
        fin_beat_streak,
        "Bounded consecutive positive surprise streak.",
    ),
    (
        "fin_miss_streak",
        ("actual", "expected", "period_id", "max_periods"),
        fin_miss_streak,
        "Bounded consecutive negative surprise streak.",
    ),
)

_NAMES: list[str] = []
for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
    _NAMES.append(_name)

from cleaned_operators import operator_surface as _surface
from cleaned_operators.registry import OperatorRegistry as _registry

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NAMES)
)
_surface._FUNDAMENTAL_V2_CANONICALS = frozenset(
    set(_surface._FUNDAMENTAL_V2_CANONICALS) | set(_NAMES)
)
_legacy = _registry._catalog.get("fin_surprise_zscore")
if _legacy is not None:
    _legacy["semantic_note"] = (
        "daily-observation z-score; use fin_surprise_event_zscore for "
        "forward-filled realized report data"
    )
    _legacy["preferred_replacement"] = "fin_surprise_event_zscore"
