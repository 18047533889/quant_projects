# -*- coding: utf-8 -*-
"""Fiscal TRUE_GAP batch 2: advanced fiscal event operators.

This module implements 5 fiscal fundamental operators using strict fiscal event
semantics via FiscalEventView. All operators work on distinct fiscal events
rather than forward-filled daily panels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.fiscal_event_ops import FiscalEventView

_EPS = 1e-12


def _positive(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal-event operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def _policy(value: Any) -> str:
    result = str(value).lower()
    if result not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return result


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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
                out[row, col] = numerator / denominator

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
                out[row, col] = (current - previous) / std

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
                logit_val = np.log(clamped / (1.0 - clamped))
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
                out[row, col] = (current_logit - mean_prior) / std_prior

    return pd.DataFrame(out, index=x.index, columns=x.columns)


# Polars implementations (convert to pandas, compute, convert back)
try:
    import polars as pl

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

except ImportError:
    pl = None


# Operator wrapper classes
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
if pl is not None:

    class _PolarsFiscalReversalRatio:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalReversalRatio.metadata

        def calculate(self, *args, **kwargs):
            from factor_engine.cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_reversal_ratio(*processed_args, **processed_kwargs)

    class _PolarsFiscalStandardizedSurprise:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalStandardizedSurprise.metadata

        def calculate(self, *args, **kwargs):
            from factor_engine.cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_standardized_surprise(*processed_args, **processed_kwargs)

    class _PolarsFiscalPairDirectionAgreement:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalPairDirectionAgreement.metadata

        def calculate(self, *args, **kwargs):
            from factor_engine.cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_pair_direction_agreement(*processed_args, **processed_kwargs)

    class _PolarsFiscalAsymmetricElasticity:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalAsymmetricElasticity.metadata

        def calculate(self, *args, **kwargs):
            from factor_engine.cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_asymmetric_elasticity(*processed_args, **processed_kwargs)

    class _PolarsFiscalLogitScore:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalLogitScore.metadata

        def calculate(self, *args, **kwargs):
            from factor_engine.cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_logit_score(*processed_args, **processed_kwargs)


def register() -> None:
    """Register fiscal batch 2 operators."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # Check if already registered - if so, skip
    if "fiscal_logit_score" in OperatorRegistry._operators:
        return

    # Register pandas_numpy backend
    for name, cls in [
        ("fiscal_reversal_ratio", _FiscalReversalRatio),
        ("fiscal_standardized_surprise", _FiscalStandardizedSurprise),
        ("fiscal_pair_direction_agreement", _FiscalPairDirectionAgreement),
        ("fiscal_asymmetric_elasticity", _FiscalAsymmetricElasticity),
        ("fiscal_logit_score", _FiscalLogitScore),
    ]:
        register_operator(
            name=name,
            category="fundamental_period",
            business_category="fundamental",
            canonical=name,
            source="fundamental.fiscal_batch2",
            backend="pandas_numpy",
            status="experimental",
        )(cls)

    # Register polars backend
    if pl is not None:
        for name, cls in [
            ("fiscal_reversal_ratio", _PolarsFiscalReversalRatio),
            ("fiscal_standardized_surprise", _PolarsFiscalStandardizedSurprise),
            ("fiscal_pair_direction_agreement", _PolarsFiscalPairDirectionAgreement),
            ("fiscal_asymmetric_elasticity", _PolarsFiscalAsymmetricElasticity),
            ("fiscal_logit_score", _PolarsFiscalLogitScore),
        ]:
            register_operator(
                name=name,
                category="fundamental_period",
                business_category="fundamental",
                canonical=name,
                source="fundamental.fiscal_batch2",
                backend="polars",
                status="experimental",
            )(cls)

    # Add to extended surface
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(
        {
            "fiscal_reversal_ratio",
            "fiscal_standardized_surprise",
            "fiscal_pair_direction_agreement",
            "fiscal_asymmetric_elasticity",
            "fiscal_logit_score",
        }
    )


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
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
