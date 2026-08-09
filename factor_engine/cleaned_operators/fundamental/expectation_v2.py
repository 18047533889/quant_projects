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


def _expectation_masks(expected, target_period_id):
    """Complete-data masks for same-target expectation revision.

    A revision can only be asserted when the current expectation, prior
    expectation, current target period and prior target period are ALL present.
    Any missing input (or a data gap in the prior row) makes the row
    undetermined -> NaN, never a guessed 0 (review R4-26).  The first row is a
    valid baseline (no prior estimate to revise -> 0), matching the parity
    contract.
    """
    period = target_period_id.reindex(index=expected.index, columns=expected.columns)
    prev_exp = expected.shift(1)
    prev_period = period.shift(1)
    first_row = pd.DataFrame(False, index=expected.index, columns=expected.columns)
    first_row.iloc[0] = True
    prev_ok = prev_exp.notna() & prev_period.notna()
    complete = expected.notna() & period.notna() & (prev_ok | first_row)
    same = period.eq(prev_period) & prev_ok
    delta = expected - prev_exp
    changed = delta.abs().gt(0) & prev_ok
    return complete, same, changed, delta


def _expectation_complete(expected, target_period_id):
    period = target_period_id.reindex(index=expected.index, columns=expected.columns)
    prev_ok = expected.shift(1).notna() & period.shift(1).notna()
    first_row = pd.DataFrame(False, index=expected.index, columns=expected.columns)
    first_row.iloc[0] = True
    return expected.notna() & period.notna() & (prev_ok | first_row)


def fin_expectation_revision(expected, target_period_id):
    complete, same, changed, delta = _expectation_masks(expected, target_period_id)
    revision = complete & same & changed
    # 0 only when data is complete and a same-target revision is confirmed
    # absent; missing inputs -> NaN (review R4-26).
    out = delta.where(revision, 0.0)
    return out.where(complete, np.nan)


def fin_expectation_revision_pct(expected, target_period_id):
    complete, same, changed, _ = _expectation_masks(expected, target_period_id)
    revision = complete & same & changed
    prev_exp = expected.shift(1)
    denom = prev_exp.where(prev_exp.ne(0))
    pct = (expected / denom - 1.0).replace([np.inf, -np.inf], np.nan)
    out = pct.where(revision, 0.0)
    return out.where(complete, np.nan)


def fin_expectation_revision_speed(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    speed = fin_expectation_revision_pct(expected, target_period_id).rolling(
        window, min_periods=1
    ).sum()
    return speed.where(_expectation_complete(expected, target_period_id), np.nan)


def _revision_event(expected, target_period_id):
    revision = fin_expectation_revision(expected, target_period_id)
    return revision.ne(0) & revision.notna()


def fin_expectation_revision_count(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    count = _revision_event(expected, target_period_id).astype(float).rolling(
        window, min_periods=1
    ).sum()
    return count.where(_expectation_complete(expected, target_period_id), np.nan)


def fin_expectation_revision_magnitude(expected, target_period_id, window_days=60):
    window = _pos_int(window_days, "window_days", 2)
    magnitude = fin_expectation_revision_pct(expected, target_period_id).abs().rolling(
        window, min_periods=1
    ).sum()
    return magnitude.where(_expectation_complete(expected, target_period_id), np.nan)


def fin_days_since_expectation_revision(
    expected,
    target_period_id,
    max_days=252,
):
    cap = _pos_int(max_days, "max_days")
    events = _revision_event(expected, target_period_id)
    period = target_period_id.reindex(index=expected.index, columns=expected.columns)
    output = pd.DataFrame(np.nan, index=expected.index, columns=expected.columns)
    for column in expected.columns:
        ev = expected[column].to_numpy(dtype=float)
        pv = period[column].to_numpy()
        event = events[column].fillna(False).to_numpy(dtype=bool)
        age = cap
        values: list[float] = []
        for i in range(len(expected)):
            complete = bool(np.isfinite(ev[i]) and not pd.isna(pv[i]))
            if not complete:
                # R4-27: cannot observe an update today -> NaN, and never age
                # blindly across an unobservable gap.
                values.append(np.nan)
                age = None
                continue
            if event[i]:
                values.append(0.0)
                age = 0
            elif age is not None:
                age = min(cap, age + 1)
                values.append(float(age))
            else:
                # P1-06: after a gap (age=None) the revision age is censored —
                # output NaN, NOT 0 (a 0 would read as "revision happened today").
                # The age only resumes at a NEW revision event (the ``event[i]``
                # branch above emits 0 and restarts the clock).
                values.append(np.nan)
                age = None
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
    # A beat/miss streak must only bridge *adjacent* fiscal periods: a skipped
    # report's surprise sign is unknown, so it ends the streak (review R4-23).
    return _walk_periods(
        difference,
        period_id,
        lambda o, v, c: _value_streak_span(o, v, c, positive=beat, max_periods=count),
    )


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

_surface.extend_extended_only(set(_NAMES))
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
