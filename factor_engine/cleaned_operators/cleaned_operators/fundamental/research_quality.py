# -*- coding: utf-8 -*-
"""Research-grade fundamental quality operators (P2).

These operators implement advanced accounting quality measures from academic
literature. They are marked as experimental and require careful preprocessing
of inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.fiscal_event_ops import FiscalEventView
from factor_engine.cleaned_operators.fiscal_strict import period_ordinal

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


def pd_accounting_comparability_score(
    scaled_earnings,
    report_return,
    industry,
    period_id,
    periods=16,
    min_periods=12,
    min_peers=5,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Accounting comparability score (De Franco et al. 2011).

    Measures how similarly a firm's accounting maps economic events (returns)
    to earnings relative to industry peers. Higher (closer to 0) indicates
    greater comparability.

    Algorithm:
    1. Within each industry × fiscal period, regress: earnings_i = a + b * return_i
    2. Extract firm i's coefficient b_i and all peer coefficients b_j
    3. Comparability = -mean(|b_i * return_j - earnings_j|) over peers j

    Args:
        scaled_earnings: Scaled earnings (e.g., earnings / total assets)
        report_return: Stock return over the reporting period
        industry: Industry classification (categorical, e.g., industry code)
        period_id: Fiscal period identifier (e.g., "2024Q1")
        periods: Number of historical periods for regression (default 16)
        min_periods: Minimum periods required for regression (default 12)
        min_peers: Minimum peer firms required (default 5)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of comparability scores (negative; closer to 0 is better)

    Notes:
        - Inputs must be preprocessed: scaled_earnings should be normalized
          (e.g., earnings/assets), report_return should be cumulative return
          over the fiscal period
        - Requires sufficient industry peers with overlapping fiscal periods
        - Extended surface only (complex research operator)
    """
    scaled_earnings, report_return, industry, period_id = _align(
        scaled_earnings, report_return, industry, period_id
    )
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    min_peers = _positive(min_peers, "min_peers")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")

    policy = _policy(revision_policy)

    # Convert period_id to ordinals
    ordinal_array = np.full(period_id.shape, np.nan, dtype=float)
    raw_periods = period_id.to_numpy(dtype=object)
    for row in range(period_id.shape[0]):
        for col in range(period_id.shape[1]):
            ordinal = period_ordinal(raw_periods[row, col])
            if ordinal is not None:
                ordinal_array[row, col] = ordinal

    # Build fiscal event history per instrument
    earnings_arr = scaled_earnings.to_numpy(dtype=float)
    return_arr = report_return.to_numpy(dtype=float)
    industry_arr = industry.to_numpy(dtype=object)

    # Track visible events per instrument (revision-aware)
    state_earnings = [dict() for _ in range(scaled_earnings.shape[1])]
    state_returns = [dict() for _ in range(scaled_earnings.shape[1])]
    state_industry = [dict() for _ in range(scaled_earnings.shape[1])]
    first_seen = [set() for _ in range(scaled_earnings.shape[1])]

    out = np.full(scaled_earnings.shape, np.nan, dtype=float)

    for row in range(scaled_earnings.shape[0]):
        # Update state with new events at this row
        for col in range(scaled_earnings.shape[1]):
            ordinal = ordinal_array[row, col]
            earnings_val = earnings_arr[row, col]
            return_val = return_arr[row, col]
            ind_val = industry_arr[row, col]

            if not np.isfinite(ordinal):
                continue
            if not _finite(earnings_val) or not _finite(return_val):
                continue

            key = int(ordinal)
            if policy == "latest_available" or key not in first_seen[col]:
                state_earnings[col][key] = float(earnings_val)
                state_returns[col][key] = float(return_val)
                state_industry[col][key] = ind_val
            first_seen[col].add(key)

        # Compute comparability for each instrument at this row
        for col in range(scaled_earnings.shape[1]):
            current_ordinal = ordinal_array[row, col]
            if not np.isfinite(current_ordinal):
                continue

            # Get history for this instrument
            hist_ordinals = [
                k for k in state_earnings[col].keys()
                if k <= int(current_ordinal)
            ]
            hist_ordinals.sort()
            hist_ordinals = hist_ordinals[-periods:]

            if len(hist_ordinals) < min_periods:
                continue

            # Extract this firm's data
            firm_earnings = np.array([state_earnings[col][k] for k in hist_ordinals])
            firm_returns = np.array([state_returns[col][k] for k in hist_ordinals])
            firm_industry = state_industry[col].get(int(current_ordinal))

            if firm_industry is None or pd.isna(firm_industry):
                continue

            # Fit firm-specific regression: earnings = a + b * returns
            valid = np.isfinite(firm_earnings) & np.isfinite(firm_returns)
            if valid.sum() < min_periods:
                continue

            X = firm_returns[valid]
            y = firm_earnings[valid]

            if np.std(X) <= _EPS:
                continue

            design = np.column_stack([np.ones(len(X)), X])
            if np.linalg.matrix_rank(design) != design.shape[1]:
                continue

            coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
            firm_b = coeffs[1]

            # Collect peer data from same industry with overlapping periods
            peer_comparisons = []

            for peer_col in range(scaled_earnings.shape[1]):
                if peer_col == col:
                    continue

                # Check if peer has same industry at current decision point
                peer_industry = state_industry[peer_col].get(int(current_ordinal))
                if peer_industry != firm_industry:
                    continue

                # Get peer's overlapping fiscal periods
                peer_ordinals = [
                    k for k in state_earnings[peer_col].keys()
                    if k <= int(current_ordinal) and k in hist_ordinals
                ]

                if len(peer_ordinals) < min_periods:
                    continue

                # Extract peer data for overlapping periods
                peer_earnings = np.array([state_earnings[peer_col][k] for k in peer_ordinals])
                peer_returns = np.array([state_returns[peer_col][k] for k in peer_ordinals])

                peer_valid = np.isfinite(peer_earnings) & np.isfinite(peer_returns)
                if peer_valid.sum() < min_periods:
                    continue

                # Compute |b_i * return_j - earnings_j| for each peer observation
                predicted = firm_b * peer_returns[peer_valid]
                actual = peer_earnings[peer_valid]
                errors = np.abs(predicted - actual)
                peer_comparisons.extend(errors.tolist())

            if len(peer_comparisons) >= min_peers:
                # Comparability = -mean(|b_i * return_j - earnings_j|)
                # (negative; closer to 0 is better)
                out[row, col] = -float(np.mean(peer_comparisons))

    return pd.DataFrame(out, index=scaled_earnings.index, columns=scaled_earnings.columns)


def pd_fiscal_asymmetric_timeliness(
    scaled_earnings,
    report_return,
    period_id,
    periods=16,
    min_periods=12,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Basu (1997) asymmetric timeliness: bad-news interaction coefficient.

    Measures differential earnings timeliness for bad news vs good news via
    piecewise regression over distinct fiscal events:
        earnings_t = α + β·return_t + γ·bad_t + δ·(return_t × bad_t)
    where bad_t = 1 if return_t < 0 else 0.

    Returns δ (incremental bad-news sensitivity).

    Args:
        scaled_earnings: Preprocessed scaled earnings (e.g., earnings / lagged price)
        report_return: Preprocessed return over report period (e.g., report window return)
        period_id: Fiscal period identifier (e.g., "2024Q3")
        periods: Number of recent fiscal events to include (default 16)
        min_periods: Minimum fiscal events required for regression (default 12)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of delta coefficients (bad-news timeliness increment)

    Notes:
        - **Research operator** on extended surface
        - Inputs must be **preprocessed**: scaled_earnings and report_return should
          already be normalized to appropriate scales for regression
        - Uses pandas_numpy backend only (no Polars/SQL)
        - Requires consecutive fiscal periods by default for valid timeliness measurement
        - Positive δ indicates earnings recognize bad news faster than good news
          (conservative accounting)
    """
    scaled_earnings, report_return, period_id = _align(
        scaled_earnings, report_return, period_id
    )
    n = _positive(periods, "periods")
    min_n = _positive(min_periods, "min_periods")
    if min_n > n:
        raise ValueError("min_periods cannot exceed periods")

    policy = _policy(revision_policy)

    earnings_view = FiscalEventView.from_panel(
        scaled_earnings, period_id, revision_policy=policy
    )
    return_view = FiscalEventView.from_panel(
        report_return, period_id, revision_policy=policy
    )

    out = np.full(scaled_earnings.shape, np.nan, dtype=float)

    for row in range(scaled_earnings.shape[0]):
        for col in range(scaled_earnings.shape[1]):
            # Retrieve fiscal-event histories (consecutive by default)
            e_hist = dict(
                earnings_view.history(row, col, require_consecutive=True)
            )
            r_hist = dict(
                return_view.history(row, col, require_consecutive=True)
            )
            # Intersect on common fiscal ordinals
            common = sorted(set(e_hist) & set(r_hist))[-n:]
            if len(common) < min_n:
                continue

            # Build regression data: earnings ~ return + bad + return*bad
            pairs = [
                (float(e_hist[k]), float(r_hist[k]))
                for k in common
                if _finite(e_hist[k]) and _finite(r_hist[k])
            ]
            if len(pairs) < min_n:
                continue

            y = np.asarray([p[0] for p in pairs], dtype=float)
            ret = np.asarray([p[1] for p in pairs], dtype=float)
            bad = (ret < 0).astype(float)
            interaction = ret * bad

            # Design matrix: [1, ret, bad, ret*bad]
            X = np.column_stack([np.ones(len(ret)), ret, bad, interaction])

            # Check rank
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue

            # OLS: beta = (X'X)^-1 X'y
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                delta = float(coef[3])  # interaction coefficient
                if _finite(delta):
                    out[row, col] = delta
            except (np.linalg.LinAlgError, ValueError):
                continue

    return pd.DataFrame(out, index=scaled_earnings.index, columns=scaled_earnings.columns)


class _AccountingComparabilityScore(SeriesOperator):
    metadata = OperatorMetadata(
        name="accounting_comparability_score",
        category="fundamental_period",
        description="De Franco 2011 accounting comparability: negative mean absolute "
        "prediction error of firm earnings mapping vs industry peers. Requires "
        "preprocessed scaled_earnings and report_return. Closer to 0 is better.",
        param_names=[
            "scaled_earnings",
            "report_return",
            "industry",
            "period_id",
            "periods",
            "min_periods",
            "min_peers",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
            "cross_sectional_dependency",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_accounting_comparability_score(*args, **kwargs)


class _FiscalAsymmetricTimeliness(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_asymmetric_timeliness",
        category="fundamental_period",
        description="Basu 1997 asymmetric timeliness: delta from earnings ~ return + bad + "
        "return*bad interaction regression. Requires preprocessed scaled_earnings and "
        "report_return. Positive delta indicates conservative recognition of bad news.",
        param_names=[
            "scaled_earnings",
            "report_return",
            "period_id",
            "periods",
            "min_periods",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_asymmetric_timeliness(*args, **kwargs)


def register() -> None:
    """Register accounting comparability operator."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if "accounting_comparability_score" in OperatorRegistry._operators:
        return

    register_operator(
        name="accounting_comparability_score",
        category="fundamental_period",
        business_category="fundamental",
        canonical="accounting_comparability_score",
        source="fundamental.research_quality",
        backend="pandas_numpy",
        status="experimental",
    )(_AccountingComparabilityScore)

    register_operator(
        name="fiscal_asymmetric_timeliness",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_asymmetric_timeliness",
        source="fundamental.research_quality",
        backend="pandas_numpy",
        status="experimental",
    )(_FiscalAsymmetricTimeliness)

    # Add to extended surface
    import factor_engine.cleaned_operators.operator_surface as _surface
    _surface.extend_extended_only({"accounting_comparability_score", "fiscal_asymmetric_timeliness"})


# Explicit policy declaration
_EXPLICIT_POLICIES = {
    "accounting_comparability_score": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 12,
    },
    "fiscal_asymmetric_timeliness": {
        "scope": "fundamental_period",
        "pit_safe": True,
    },
}


register()
