# -*- coding: utf-8 -*-
"""Fiscal logit score operator - TRUE_GAP batch 2 extension.

This module adds the fiscal_logit_score operator to complement the existing
fiscal event operators in fiscal_event_ops.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_event_ops import FiscalEventView

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
                logit_val = np.where((1.0 - clamped)) != 0, np.log(clamped / (1.0 - clamped)), np.nan)
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
                out[row, col] = (current_logit - mean_prior) / std_prior if std_prior > 1e-10 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


# Polars implementation
try:
    import polars as pl

    def pl_fiscal_logit_score(*args, **kwargs) -> "pl.DataFrame":
        if not args or not isinstance(args[0], pl.DataFrame):
            raise TypeError("pl_fiscal_logit_score requires Polars panels")
        args_pd = tuple(arg.to_pandas() if isinstance(arg, pl.DataFrame) else arg for arg in args)
        result = pd_fiscal_logit_score(*args_pd, **kwargs)
        return pl.from_pandas(result)

except ImportError:
    pl = None


# Operator wrapper classes
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


# Polars wrapper class
if pl is not None:

    class _PolarsFiscalLogitScore:
        _HANDLES_CALL_CONTRACT = True
        metadata = _FiscalLogitScore.metadata

        def calculate(self, *args, **kwargs):
            from cleaned_operators.base import validate_operator_call

            processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
            return pl_fiscal_logit_score(*processed_args, **processed_kwargs)


def register() -> None:
    """Register fiscal_logit_score operator."""
    from cleaned_operators.registry import OperatorRegistry

    if "fiscal_logit_score" in OperatorRegistry._operators:
        return

    # Register pandas_numpy backend
    register_operator(
        name="fiscal_logit_score",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_logit_score",
        source="fundamental.fiscal_logit_score_op",
        backend="pandas_numpy",
        status="experimental",
    )(_FiscalLogitScore)

    # Register polars backend
    if pl is not None:
        register_operator(
            name="fiscal_logit_score",
            category="fundamental_period",
            business_category="fundamental",
            canonical="fiscal_logit_score",
            source="fundamental.fiscal_logit_score_op",
            backend="polars",
            status="experimental",
        )(_PolarsFiscalLogitScore)

    # Add to extended surface
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"fiscal_logit_score"})


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "fiscal_logit_score": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 4,
    },
}


register()
