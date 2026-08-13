# -*- coding: utf-8 -*-
"""Fiscal TRUE_GAP operators: combined batch 1 + batch 2 (10 operators total).

This module implements 10 fiscal fundamental operators using strict fiscal event
semantics via FiscalEventView. All operators work on distinct fiscal events
rather than forward-filled daily panels.

Batch 1 (5 operators):
- fiscal_acceleration: second difference of fiscal values
- fiscal_pct_change: percentage change over specified lag
- fiscal_rolling_std: rolling standard deviation over fiscal events
- fiscal_accrual_quality: negative std of accruals (Dechow-Dichev 2002)
- fiscal_direction_consistency: fraction of changes in same direction

Batch 2 (5 operators):
- fiscal_reversal_ratio: magnitude-weighted sign reversal ratio
- fiscal_standardized_surprise: seasonal surprise standardized by prior volatility
- fiscal_pair_direction_agreement: sign agreement between two fiscal signals
- fiscal_asymmetric_elasticity: down-minus-up elasticity (sticky cost)
- fiscal_logit_score: logit-transformed z-score for [0,1] bounded metrics
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_event_ops import FiscalEventView, _align, _finite, _policy, _positive

_EPS = 1e-12


# ============================================================================
# BATCH 1: Core fiscal event operators
# ============================================================================

def pd_fiscal_acceleration(
    value,
    period_id,
    lag=1,
    periods=8,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal acceleration: second difference of fiscal values.

    Measures the rate of change of the rate of change:
        acceleration_t = (value_t - value_{t-lag}) - (value_{t-lag} - value_{t-2*lag})
                       = value_t - 2*value_{t-lag} + value_{t-2*lag}

    Args:
        value: Fiscal metric panel (e.g., earnings, revenue)
        period_id: Fiscal period identifier (e.g., "2024Q3")
        lag: Number of fiscal periods for differencing (default 1)
        periods: Number of recent fiscal events to consider (default 8)
        min_periods: Minimum fiscal events required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal acceleration values

    Notes:
        - Requires at least 2*lag + 1 fiscal events
        - TRUE_GAP: works on distinct fiscal events, not daily forward-fill
        - Extended surface only
    """
    value, period_id = _align(value, period_id)
    lag = _positive(lag, "lag")
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < 2 * lag + 1:
        raise ValueError(f"min_periods must be at least {2 * lag + 1} for lag={lag}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) < min_periods:
                continue

            by_ordinal = dict(history[-periods:])
            if not history:
                continue

            current_ord, current_val = history[-1]
            lag1_val = by_ordinal.get(current_ord - lag)
            lag2_val = by_ordinal.get(current_ord - 2 * lag)

            if lag1_val is None or lag2_val is None:
                continue
            if not _finite(current_val) or not _finite(lag1_val) or not _finite(lag2_val):
                continue

            # acceleration = value_t - 2*value_{t-lag} + value_{t-2*lag}
            out[row, col] = float(current_val - 2 * lag1_val + lag2_val)

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_pct_change(
    value,
    period_id,
    lag=1,
    periods=8,
    min_periods=2,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal percentage change over specified lag.

    Computes: (value_t - value_{t-lag}) / |value_{t-lag}|

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        lag: Number of fiscal periods for differencing (default 1)
        periods: Number of recent fiscal events to consider (default 8)
        min_periods: Minimum fiscal events required (default 2)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal percentage changes

    Notes:
        - Denominator uses absolute value to handle sign changes
        - Returns NaN when denominator is near zero
        - TRUE_GAP: works on distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    lag = _positive(lag, "lag")
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < lag + 1:
        raise ValueError(f"min_periods must be at least {lag + 1} for lag={lag}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) < min_periods:
                continue

            by_ordinal = dict(history[-periods:])
            if not history:
                continue

            current_ord, current_val = history[-1]
            lag_val = by_ordinal.get(current_ord - lag)

            if lag_val is None:
                continue
            if not _finite(current_val) or not _finite(lag_val):
                continue

            denom = abs(lag_val)
            if denom <= _EPS:
                continue

            out[row, col] = (float((current_val - lag_val)) / denom) if denom) != 0 else np.nan

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_rolling_std(
    value,
    period_id,
    periods=8,
    min_periods=3,
    ddof=1,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal rolling standard deviation over recent fiscal events.

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 3)
        ddof: Delta degrees of freedom (default 1)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal rolling standard deviations

    Notes:
        - Uses sample standard deviation (ddof=1) by default
        - TRUE_GAP: computed over distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if min_periods <= ddof:
        raise ValueError(f"min_periods must be greater than ddof={ddof}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            values = [v for _, v in window if _finite(v)]
            if len(values) < min_periods:
                continue

            if len(values) <= ddof:
                continue

            std_val = float(np.std(values, ddof=ddof))
            if _finite(std_val):
                out[row, col] = std_val

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_accrual_quality(
    accruals,
    period_id,
    periods=8,
    min_periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal accrual quality: negative standard deviation of accruals.

    Measures accrual volatility as a proxy for earnings quality. Lower volatility
    (higher score, closer to 0) indicates more stable accruals.

    Returns: -std(accruals) over recent fiscal events

    Args:
        accruals: Accruals panel (e.g., net income - operating cash flow)
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 4)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of accrual quality scores (negative std; closer to 0 is better)

    Notes:
        - Based on Dechow-Dichev (2002) accrual quality framework
        - Inputs should be preprocessed (typically scaled by assets)
        - TRUE_GAP: computed over distinct fiscal events
        - Extended surface only
    """
    accruals, period_id = _align(accruals, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(accruals, period_id, revision_policy=policy)
    out = np.full(accruals.shape, np.nan, dtype=float)

    for row in range(accruals.shape[0]):
        for col in range(accruals.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            values = [v for _, v in window if _finite(v)]
            if len(values) < min_periods:
                continue

            if len(values) <= 1:
                continue

            std_val = float(np.std(values, ddof=1))
            if _finite(std_val):
                out[row, col] = -std_val  # Negative: lower volatility is better

    return pd.DataFrame(out, index=accruals.index, columns=accruals.columns)


def pd_fiscal_direction_consistency(
    value,
    period_id,
    periods=8,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal direction consistency: fraction of fiscal changes in same direction.

    Measures the consistency of fiscal metric growth/decline direction over time.
    Returns the fraction of consecutive changes that agree with the most recent change.

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of direction consistency ratios [0, 1]

    Notes:
        - 1.0 indicates all changes in same direction as current
        - 0.0 indicates all changes in opposite direction
        - Only considers non-zero changes
        - TRUE_GAP: computed over distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            if len(window) < min_periods:
                continue

            # Compute consecutive changes
            changes = []
            for i in range(1, len(window)):
                prev_val = window[i - 1][1]
                curr_val = window[i][1]
                if _finite(prev_val) and _finite(curr_val):
                    change = curr_val - prev_val
                    if abs(change) > _EPS:  # Non-zero change
                        changes.append(change)

            if len(changes) < min_periods - 1:
                continue

            # Get current direction and compute agreement
            current_direction = np.sign(changes[-1])
            agreement = sum(np.sign(c) == current_direction for c in changes)
            out[row, col] = (float(agreement) / len(changes)) if len(changes) != 0 else np.nan

    return pd.DataFrame(out, index=value.index, columns=value.columns)


# ============================================================================
# BATCH 2: Advanced fiscal event operators
# ============================================================================

def pd_fiscal_reversal_ratio(
    x,
    period_id,
    periods=8,
    min_pairs=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Magnitude-weighted fiscal sign reversal ratio.

    Measures the fraction of fiscal period magnitude that reverses sign, weighted
    by the absolute value of preceding periods. Captures the severity of directional
    instability in fiscal signals.

    Formula:
        numerator = sum(min(|current|, |prev|)) where sign(current) != sign(prev)
        denominator = sum(|prev|) across all consecutive pairs
        ratio = numerator / denominator

    Args:
        x: Input fiscal metric panel
        period_id: Fiscal period identifier (e.g., "2024Q3")
        periods: Number of recent fiscal events to include (default 8)
        min_pairs: Minimum consecutive pairs required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of reversal ratios in [0, 1], where higher values indicate more
        sign instability weighted by magnitude

    Notes:
        - Extended surface only (research operator)
        - Uses strict fiscal event semantics via FiscalEventView
        - Dual backend: pandas_numpy + polars
    """
    x, period_id = _align(x, period_id)
    periods = _positive(periods, "periods")
    min_pairs = _positive(min_pairs, "min_pairs")
    policy = _policy(revision_policy)

    view = FiscalEventView.from_panel(x, period_id, revision_policy=policy)
    out = np.full(x.shape, np.nan, dtype=float)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            history = history[-periods:]
            if len(history) < min_pairs + 1:
                continue

            numerator = denominator = 0.0
            pairs = 0
            for (_, prev), (_, current) in zip(history, history[1:]):
                if not _finite(prev) or not _finite(current) or prev == 0 or current == 0:
                    continue
                pairs += 1
                denominator += abs(prev)
                if np.sign(prev) != np.sign(current):
                    numerator += min(abs(current), abs(prev))

            if pairs >= min_pairs and denominator > _EPS:
                out[row, col] = (numerator) / denominator if denominator != 0 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_standardized_surprise(
    x,
    period_id,
    seasonal_lag=4,
    lookback_periods=8,
    min_history=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Causal seasonal surprise standardized by prior fiscal events.

    Computes the surprise (current - seasonal_lag_prior) standardized by the
    volatility of prior seasonal surprises over lookback_periods.

    Formula:
        surprise_t = x_t - x_{t-seasonal_lag}
        std = std([surprise_{t-1}, surprise_{t-2}, ..., surprise_{t-lookback_periods}])
        standardized_surprise_t = surprise_t / std

    Args:
        x: Input fiscal metric panel
        period_id: Fiscal period identifier (e.g., "2024Q3")
        seasonal_lag: Fiscal periods to lag for seasonal comparison (default 4)
        lookback_periods: Number of prior surprises for std estimation (default 8)
        min_history: Minimum prior surprises required (default 4)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of standardized seasonal surprises (z-score scale)

    Notes:
        - Extended surface only (research operator)
        - Uses strict fiscal event semantics via FiscalEventView
        - Dual backend: pandas_numpy + polars
    """
    x, period_id = _align(x, period_id)
    seasonal_lag = _positive(seasonal_lag, "seasonal_lag")
    lookback_periods = _positive(lookback_periods, "lookback_periods")
    min_history = _positive(min_history, "min_history")
    policy = _policy(revision_policy)

    view = FiscalEventView.from_panel(x, period_id, revision_policy=policy)
    out = np.full(x.shape, np.nan, dtype=float)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) <= seasonal_lag:
                continue

            current_ord, current = history[-1]
            by_ord = dict(history)
            previous = by_ord.get(current_ord - seasonal_lag)
            if previous is None:
                continue

            # Collect prior surprises
            surprises = []
            for ordinal, value in history[:-1]:
                prior = by_ord.get(ordinal - seasonal_lag)
                if prior is not None and _finite(value) and _finite(prior):
                    surprises.append(value - prior)

            surprises = surprises[-lookback_periods:]
            if len(surprises) < min_history:
                continue

            std = float(np.std(np.asarray(surprises), ddof=1)) if len(surprises) > 1 else np.nan
            if np.isfinite(std) and std > _EPS:
                out[row, col] = ((current - previous)) / std if std != 0 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_pair_direction_agreement(
    x,
    y,
    period_id,
    periods=8,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Agreement of two precomputed fiscal signals (sign matching ratio).

    Computes the fraction of fiscal events where x and y have the same sign
    (both positive, both negative, or both zero). Measures signal coherence.

    Formula:
        agreement = mean([sign(x_t) == sign(y_t) for t in fiscal events])

    Args:
        x: First fiscal signal panel
        y: Second fiscal signal panel
        period_id: Fiscal period identifier (e.g., "2024Q3")
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum events with non-zero values required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of agreement ratios in [0, 1], where 1.0 indicates perfect sign
        agreement and 0.0 indicates perfect disagreement

    Notes:
        - Extended surface only (research operator)
        - Uses strict fiscal event semantics via FiscalEventView
        - Dual backend: pandas_numpy + polars
        - Filters out zero values from both signals
    """
    x, y, period_id = _align(x, y, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")
    policy = _policy(revision_policy)

    viewx = FiscalEventView.from_panel(x, period_id, revision_policy=policy)
    viewy = FiscalEventView.from_panel(y, period_id, revision_policy=policy)
    out = np.full(x.shape, np.nan, dtype=float)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            hx = dict(viewx.history(row, col, require_consecutive=bool(require_consecutive)))
            hy = dict(viewy.history(row, col, require_consecutive=bool(require_consecutive)))
            common = sorted(set(hx) & set(hy))[-periods:]

            pairs = [
                (hx[o], hy[o])
                for o in common
                if _finite(hx[o]) and _finite(hy[o]) and hx[o] != 0 and hy[o] != 0
            ]

            if len(pairs) >= min_periods:
                out[row, col] = float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_asymmetric_elasticity(
    cost,
    activity,
    period_id,
    periods=12,
    mode="down_minus_up",
    add_intercept=True,
    min_obs_per_regime=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Down-minus-up fiscal activity elasticity (sticky cost measure).

    Measures asymmetric cost response to activity changes: β_down - β_up.
    Positive values indicate cost stickiness (costs rise faster than they fall).

    Algorithm:
        1. Compute fiscal-to-fiscal changes: Δcost_t, Δactivity_t
        2. Partition into up regime (Δactivity > 0) and down regime (Δactivity < 0)
        3. Fit: Δcost = α + β * Δactivity separately for each regime
        4. Return: β_down - β_up

    Args:
        cost: Cost metric panel (e.g., SG&A, COGS)
        activity: Activity metric panel (e.g., revenue, production)
        period_id: Fiscal period identifier (e.g., "2024Q3")
        periods: Number of recent fiscal changes to include (default 12)
        mode: Elasticity mode, must be "down_minus_up" (default)
        add_intercept: Whether to include intercept in regime regressions (default True)
        min_obs_per_regime: Minimum changes required per regime (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of asymmetric elasticity measures (β_down - β_up), where positive
        values indicate sticky costs

    Notes:
        - Extended surface only (research operator)
        - Uses strict fiscal event semantics via FiscalEventView
        - Dual backend: pandas_numpy + polars
        - Classic Anderson et al. (2003) sticky cost model
    """
    cost, activity, period_id = _align(cost, activity, period_id)
    periods = _positive(periods, "periods")
    minimum = _positive(min_obs_per_regime, "min_obs_per_regime")
    mode = str(mode).lower()
    if mode != "down_minus_up":
        raise ValueError("mode must be 'down_minus_up'")
    policy = _policy(revision_policy)

    cost_view = FiscalEventView.from_panel(cost, period_id, revision_policy=policy)
    activity_view = FiscalEventView.from_panel(activity, period_id, revision_policy=policy)
    out = np.full(cost.shape, np.nan, dtype=float)

    for row in range(cost.shape[0]):
        for col in range(cost.shape[1]):
            c_hist = dict(cost_view.history(row, col, require_consecutive=bool(require_consecutive)))
            a_hist = dict(activity_view.history(row, col, require_consecutive=bool(require_consecutive)))
            common = sorted(set(c_hist) & set(a_hist))[-periods:]

            changes = [
                (c_hist[o] - c_hist[p], a_hist[o] - a_hist[p])
                for p, o in zip(common, common[1:])
                if _finite(c_hist[o]) and _finite(c_hist[p]) and _finite(a_hist[o]) and _finite(a_hist[p])
            ]

            up = [(dc, da) for dc, da in changes if da > 0]
            down = [(dc, da) for dc, da in changes if da < 0]

            if len(up) < minimum or len(down) < minimum:
                continue

            def slope(samples):
                yy = np.asarray([pair[0] for pair in samples])
                xx = np.asarray([pair[1] for pair in samples])
                design = np.column_stack([np.ones(len(xx)), xx]) if add_intercept else xx[:, None]
                if np.linalg.matrix_rank(design) != design.shape[1]:
                    return np.nan
                return float(np.linalg.lstsq(design, yy, rcond=None)[0][-1])

            beta_up = slope(up)
            beta_down = slope(down)
            if _finite(beta_up) and _finite(beta_down):
                out[row, col] = beta_down - beta_up

    return pd.DataFrame(out, index=cost.index, columns=cost.columns)


def pd_fiscal_logit_score(
    x,
    period_id,
    periods=8,
    min_periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal logit transformation score for bounded [0, 1] metrics.

    Applies logit transformation to fiscal metrics that are naturally bounded
    in [0, 1] (e.g., margins, ratios), then computes causal z-score over
    recent fiscal events.

    Formula:
        logit(p) = log(p / (1 - p))
        score_t = (logit(x_t) - mean(logit(x_{t-1..t-periods}))) / std(logit(...))

    Args:
        x: Input fiscal metric panel, values in (0, 1) exclusive
        period_id: Fiscal period identifier (e.g., "2024Q3")
        periods: Number of recent fiscal events for normalization (default 8)
        min_periods: Minimum fiscal events required (default 4)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of logit-transformed z-scores

    Notes:
        - Extended surface only (research operator)
        - Uses strict fiscal event semantics via FiscalEventView
        - Dual backend: pandas_numpy + polars
        - Values outside (0, 1) are clamped to (0.001, 0.999) for numerical stability
        - Useful for margin, efficiency, and ratio metrics
    """
    x, period_id = _align(x, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")
    policy = _policy(revision_policy)

    view = FiscalEventView.from_panel(x, period_id, revision_policy=policy)
    out = np.full(x.shape, np.nan, dtype=float)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            history = history[-periods:]
            if len(history) < min_periods:
                continue

            # Clamp values to (0.001, 0.999) and apply logit
            logit_values = []
            for _, value in history:
                if not _finite(value):
                    continue
                # Clamp to avoid log(0) or log(negative)
                clamped = max(0.001, min(0.999, value))
                logit_val = np.where((1.0 - clamped) != 0, (np.log(clamped) / ((1.0 - clamped))), np.nan)
                if _finite(logit_val):
                    logit_values.append(logit_val)

            if len(logit_values) < min_periods:
                continue

            # Current is the last value
            current_logit = logit_values[-1]
            # Prior is everything except current
            prior_logits = logit_values[:-1]

            if len(prior_logits) < min_periods - 1:
                continue

            mean_prior = float(np.mean(prior_logits))
            std_prior = float(np.std(prior_logits, ddof=1)) if len(prior_logits) > 1 else np.nan

            if np.isfinite(std_prior) and std_prior > _EPS:
                out[row, col] = ((current_logit - mean_prior)) / std_prior if std_prior != 0 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ============================================================================
# Polars backend implementations
# ============================================================================

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False


if HAS_POLARS:

    def pl_fiscal_acceleration(
        value,
        period_id,
        lag=1,
        periods=8,
        min_periods=3,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_acceleration."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_acceleration(
            value_pd,
            period_id_pd,
            lag=lag,
            periods=periods,
            min_periods=min_periods,
            require_consecutive=require_consecutive,
            revision_policy=revision_policy,
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_pct_change(
        value,
        period_id,
        lag=1,
        periods=8,
        min_periods=2,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_pct_change."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_pct_change(
            value_pd,
            period_id_pd,
            lag=lag,
            periods=periods,
            min_periods=min_periods,
            require_consecutive=require_consecutive,
            revision_policy=revision_policy,
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_rolling_std(
        value,
        period_id,
        periods=8,
        min_periods=3,
        ddof=1,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_rolling_std."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_rolling_std(
            value_pd,
            period_id_pd,
            periods=periods,
            min_periods=min_periods,
            ddof=ddof,
            require_consecutive=require_consecutive,
            revision_policy=revision_policy,
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_accrual_quality(
        accruals,
        period_id,
        periods=8,
        min_periods=4,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_accrual_quality."""
        if not isinstance(accruals, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        accruals_pd = accruals.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_accrual_quality(
            accruals_pd,
            period_id_pd,
            periods=periods,
            min_periods=min_periods,
            require_consecutive=require_consecutive,
            revision_policy=revision_policy,
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_direction_consistency(
        value,
        period_id,
        periods=8,
        min_periods=3,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_direction_consistency."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_direction_consistency(
            value_pd,
            period_id_pd,
            periods=periods,
            min_periods=min_periods,
            require_consecutive=require_consecutive,
            revision_policy=revision_policy,
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_reversal_ratio(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_reversal_ratio requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_reversal_ratio(*args_pd, **kwargs)
        return pl.from_pandas(result)

    def pl_fiscal_standardized_surprise(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_standardized_surprise requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_standardized_surprise(*args_pd, **kwargs)
        return pl.from_pandas(result)

    def pl_fiscal_pair_direction_agreement(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_pair_direction_agreement requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_pair_direction_agreement(*args_pd, **kwargs)
        return pl.from_pandas(result)

    def pl_fiscal_asymmetric_elasticity(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_asymmetric_elasticity requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_asymmetric_elasticity(*args_pd, **kwargs)
        return pl.from_pandas(result)

    def pl_fiscal_logit_score(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_logit_score requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_logit_score(*args_pd, **kwargs)
        return pl.from_pandas(result)


# ============================================================================
# Operator class wrappers
# ============================================================================

# BATCH 1 classes
class _FiscalAcceleration(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_acceleration",
        category="fundamental_period",
        description="Second difference of fiscal values: value_t - 2*value_{t-lag} + value_{t-2*lag}. "
        "Measures acceleration of fiscal metric changes over distinct fiscal events.",
        param_names=[
            "value",
            "period_id",
            "lag",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_acceleration(*args, **kwargs)


class _FiscalPctChange(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_pct_change",
        category="fundamental_period",
        description="Fiscal percentage change over specified lag: (value_t - value_{t-lag}) / |value_{t-lag}|. "
        "Works on distinct fiscal events.",
        param_names=[
            "value",
            "period_id",
            "lag",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_pct_change(*args, **kwargs)


class _FiscalRollingStd(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_rolling_std",
        category="fundamental_period",
        description="Rolling standard deviation over recent fiscal events. "
        "Measures fiscal metric volatility over distinct fiscal periods.",
        param_names=[
            "value",
            "period_id",
            "periods",
            "min_periods",
            "ddof",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_rolling_std(*args, **kwargs)


class _FiscalAccrualQuality(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_accrual_quality",
        category="fundamental_period",
        description="Negative standard deviation of accruals over recent fiscal events. "
        "Higher (closer to 0) indicates more stable accruals. Based on Dechow-Dichev 2002.",
        param_names=[
            "accruals",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
            "research",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_accrual_quality(*args, **kwargs)


class _FiscalDirectionConsistency(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_direction_consistency",
        category="fundamental_period",
        description="Fraction of fiscal changes in same direction as most recent change. "
        "Returns ratio [0, 1] measuring growth/decline consistency over fiscal events.",
        param_names=[
            "value",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_direction_consistency(*args, **kwargs)


# BATCH 2 classes
class _FiscalReversalRatio(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_reversal_ratio",
        category="fundamental_period",
        description="Magnitude-weighted fiscal sign reversal ratio. Higher values indicate "
        "more directional instability. Returns numerator/denominator where numerator is "
        "sum(min(|current|, |prev|)) for sign reversals and denominator is sum(|prev|).",
        param_names=[
            "x",
            "period_id",
            "periods",
            "min_pairs",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_reversal_ratio(*args, **kwargs)


class _FiscalStandardizedSurprise(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_standardized_surprise",
        category="fundamental_period",
        description="Causal seasonal surprise (current - seasonal_lag_prior) standardized "
        "by prior surprise volatility over lookback_periods. Returns z-score scale.",
        param_names=[
            "x",
            "period_id",
            "seasonal_lag",
            "lookback_periods",
            "min_history",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_standardized_surprise(*args, **kwargs)


class _FiscalPairDirectionAgreement(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_pair_direction_agreement",
        category="fundamental_period",
        description="Agreement of two precomputed fiscal signals (fraction of events with "
        "matching signs). Returns ratio in [0, 1] where 1.0 indicates perfect agreement.",
        param_names=[
            "x",
            "y",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_pair_direction_agreement(*args, **kwargs)


class _FiscalAsymmetricElasticity(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_asymmetric_elasticity",
        category="fundamental_period",
        description="Down-minus-up fiscal activity elasticity (Anderson 2003 sticky cost). "
        "Positive values indicate costs rise faster than they fall (β_down - β_up).",
        param_names=[
            "cost",
            "activity",
            "period_id",
            "periods",
            "mode",
            "add_intercept",
            "min_obs_per_regime",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_asymmetric_elasticity(*args, **kwargs)


class _FiscalLogitScore(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_logit_score",
        category="fundamental_period",
        description="Fiscal logit transformation score for [0, 1] bounded metrics. "
        "Applies logit(p) = log(p/(1-p)) then computes causal z-score over fiscal events. "
        "Values clamped to (0.001, 0.999) for stability.",
        param_names=[
            "x",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_logit_score(*args, **kwargs)


# Polars wrapper classes
if HAS_POLARS:

    class _PolarsFiscalAcceleration(SeriesOperator):
        metadata = _FiscalAcceleration.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_acceleration(*args, **kwargs)

    class _PolarsFiscalPctChange(SeriesOperator):
        metadata = _FiscalPctChange.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_pct_change(*args, **kwargs)

    class _PolarsFiscalRollingStd(SeriesOperator):
        metadata = _FiscalRollingStd.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_rolling_std(*args, **kwargs)

    class _PolarsFiscalAccrualQuality(SeriesOperator):
        metadata = _FiscalAccrualQuality.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_accrual_quality(*args, **kwargs)

    class _PolarsFiscalDirectionConsistency(SeriesOperator):
        metadata = _FiscalDirectionConsistency.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_direction_consistency(*args, **kwargs)

    class _PolarsFiscalReversalRatio(SeriesOperator):
        metadata = _FiscalReversalRatio.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_reversal_ratio(*args, **kwargs)

    class _PolarsFiscalStandardizedSurprise(SeriesOperator):
        metadata = _FiscalStandardizedSurprise.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_standardized_surprise(*args, **kwargs)

    class _PolarsFiscalPairDirectionAgreement(SeriesOperator):
        metadata = _FiscalPairDirectionAgreement.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_pair_direction_agreement(*args, **kwargs)

    class _PolarsFiscalAsymmetricElasticity(SeriesOperator):
        metadata = _FiscalAsymmetricElasticity.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_asymmetric_elasticity(*args, **kwargs)

    class _PolarsFiscalLogitScore(SeriesOperator):
        metadata = _FiscalLogitScore.metadata

        def _calculate_series(self, *args, **kwargs):
            return pl_fiscal_logit_score(*args, **kwargs)


# ============================================================================
# Registration
# ============================================================================

def register() -> None:
    """Register fiscal batch1_batch2 operators."""
    from cleaned_operators.registry import OperatorRegistry

    # Register pandas_numpy backend
    operators = [
        ("fiscal_acceleration", _FiscalAcceleration),
        ("fiscal_pct_change", _FiscalPctChange),
        ("fiscal_rolling_std", _FiscalRollingStd),
        ("fiscal_accrual_quality", _FiscalAccrualQuality),
        ("fiscal_direction_consistency", _FiscalDirectionConsistency),
        ("fiscal_reversal_ratio", _FiscalReversalRatio),
        ("fiscal_standardized_surprise", _FiscalStandardizedSurprise),
        ("fiscal_pair_direction_agreement", _FiscalPairDirectionAgreement),
        ("fiscal_asymmetric_elasticity", _FiscalAsymmetricElasticity),
        ("fiscal_logit_score", _FiscalLogitScore),
    ]

    for name, cls in operators:
        # Register pandas_numpy backend (skip if already registered)
        existing_pandas = OperatorRegistry.get(name, "pandas_numpy")
        if not existing_pandas:
            register_operator(
                name=name,
                category="fundamental_period",
                business_category="fundamental",
                canonical=name,
                source="fundamental.fiscal_batch1_batch2",
                backend="pandas_numpy",
                status="production_ready",
            )(cls)

        # Register polars backend if available (skip if already registered)
        if HAS_POLARS:
            existing_polars = OperatorRegistry.get(name, "polars")
            if not existing_polars:
                polars_cls = globals().get(f"_Polars{cls.__name__[1:]}")
                if polars_cls:
                    register_operator(
                        name=name,
                        category="fundamental_period",
                        business_category="fundamental",
                        canonical=name,
                        source="fundamental.fiscal_batch1_batch2",
                        backend="polars",
                        status="production_ready",
                    )(polars_cls)

    # Add to extended surface
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(
        {
            "fiscal_acceleration",
            "fiscal_pct_change",
            "fiscal_rolling_std",
            "fiscal_accrual_quality",
            "fiscal_direction_consistency",
            "fiscal_reversal_ratio",
            "fiscal_standardized_surprise",
            "fiscal_pair_direction_agreement",
            "fiscal_asymmetric_elasticity",
            "fiscal_logit_score",
        }
    )


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "fiscal_acceleration": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_pct_change": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 2,
    },
    "fiscal_rolling_std": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_accrual_quality": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 4,
    },
    "fiscal_direction_consistency": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_reversal_ratio": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_pairs": 3,
    },
    "fiscal_standardized_surprise": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_history": 4,
    },
    "fiscal_pair_direction_agreement": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_asymmetric_elasticity": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_obs_per_regime": 3,
    },
    "fiscal_logit_score": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 4,
    },
}


register()
