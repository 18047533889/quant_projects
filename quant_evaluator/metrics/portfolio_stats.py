"""
Portfolio statistics for long/short backtesting.

Provides Sharpe ratio, drawdown analysis, and portfolio return metrics.
"""

from typing import Tuple, Optional
import numpy as np

# QE-METRIC P0-10: canonical missing-return policy strings shared by the
# return-computation paths. Semantics:
#   "zero_fill" — NaN returns are treated as 0 (flat period). Back-compat
#       default; documented, NOT silent.
#   "drop"      — periods with any missing return are excluded (NaN output
#       for that period) instead of being filled.
#   "fail"      — raise ValueError on any non-finite return.
_MISSING_RETURN_POLICIES = ("zero_fill", "drop", "fail")


def compute_compound_annualized_return(returns, periods_per_year=252, *, min_periods=2,
                                      missing_return_policy="drop"):
    """CAGR authority, with absorbing total loss and explicit observation policy.

    Periods are equally spaced observations, not inferred calendar days.
    `drop` annualizes observed periods; callers must separately audit coverage.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0 or min_periods < 1:
        raise ValueError("annualization frequency and minimum periods must be positive")
    _validate_missing_return_policy(missing_return_policy)
    finite = np.isfinite(values)
    if missing_return_policy == "fail" and not finite.all():
        raise ValueError("returns contains non-finite values")
    values = values[finite] if missing_return_policy == "drop" else np.where(finite, values, 0.)
    if len(values) < min_periods:
        return float("nan")
    if np.any(values <= -1):
        return -1.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(np.log1p(values).sum() * periods_per_year / len(values))
    return float(result) if np.isfinite(result) else float("nan")


def compute_wealth_curve(
    returns: np.ndarray,
    missing_return_policy: str = "drop",
) -> np.ndarray:
    """Compute a canonical wealth curve from periodic returns.

    ``drop`` removes unobserved periods instead of manufacturing flat days;
    ``zero_fill`` preserves the original time axis; ``fail`` rejects missing
    observations.  This function is the report/chart authority for NAV and
    cumulative return.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got {values.ndim}D")
    _validate_missing_return_policy(missing_return_policy)
    if missing_return_policy == "fail" and np.any(~np.isfinite(values)):
        raise ValueError("returns contains non-finite values and missing_return_policy='fail'")
    if missing_return_policy == "drop":
        values = values[np.isfinite(values)]
    else:
        values = np.where(np.isfinite(values), values, 0.0)
    wealth = np.cumprod(1.0 + values)
    # Equity cannot recover after exhaustion without an explicit capital injection.
    return np.where(np.maximum.accumulate(wealth <= 0.0), 0.0, wealth)


def compute_aligned_wealth_curve(returns: np.ndarray) -> np.ndarray:
    """Return a time-aligned wealth curve with gaps on unobserved periods.

    Missing returns never become zero-return observations.  The last observed
    wealth is retained internally so the next valid period can continue, while
    the missing position itself remains NaN for charts and coverage checks.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got {values.ndim}D")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    wealth = 1.0
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        wealth *= 1.0 + float(value)
        wealth = max(0.0, wealth)
        out[index] = wealth
    return out


def equal_gross_weights(long_members, short_members):
    """Equal absolute weight per selected stock; 100% gross including full short margin.

    Masks must be signal-time decisions. No forward-return filter is used.
    Both legs are required; otherwise the portfolio remains in cash.
    """
    long_members, short_members = np.asarray(long_members, bool), np.asarray(short_members, bool)
    if long_members.shape != short_members.shape or long_members.ndim != 2:
        raise ValueError("membership masks must have matching (time, asset) shapes")
    if np.any(long_members & short_members):
        raise ValueError("an asset cannot be both long and short")
    total = (long_members.sum(axis=1) + short_members.sum(axis=1))[:, None]
    active = (long_members.any(axis=1) & short_members.any(axis=1))[:, None]
    return np.divide(long_members.astype(float)-short_members, total,
                     out=np.zeros(long_members.shape, float), where=active & (total > 0))


def equal_gross_long_short_returns(long_members, short_members, forward_returns, *, cost_rate=0.0,
                                  missing_return_policy="drop"):
    """100% gross target-weight portfolio; full-notional turnover including entry.

    A missing selected return invalidates the day under 'drop', not membership.
    'zero_fill' is an explicit flat-mark assumption, never a reweighting rule.
    """
    weights = equal_gross_weights(long_members, short_members)
    returns = np.asarray(forward_returns, float)
    if returns.shape != weights.shape or not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("invalid returns shape or commission")
    _validate_missing_return_policy(missing_return_policy)
    missing = ((weights != 0) & ~np.isfinite(returns)).any(axis=1)
    if missing_return_policy == "fail" and missing.any():
        raise ValueError("missing return on a selected position")
    pnl = (weights * np.where(np.isfinite(returns), returns, 0.)).sum(axis=1)
    turnover = np.abs(np.diff(np.vstack([np.zeros((1, weights.shape[1])), weights]), axis=0)).sum(axis=1)
    pnl -= cost_rate * turnover
    if missing_return_policy == "drop":
        pnl[missing] = np.nan
    return pnl


def apply_long_short_costs(
    long_returns: np.ndarray,
    short_returns: np.ndarray,
    *,
    long_turnover: np.ndarray,
    short_turnover: np.ndarray,
    cost_rate: float,
) -> np.ndarray:
    """Compute net long-short returns and charge turnover on both legs."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (long_returns, short_returns, long_turnover, short_turnover)
    ]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("returns and turnover arrays must have matching shapes")
    if cost_rate < 0 or not np.isfinite(cost_rate):
        raise ValueError("cost_rate must be a finite non-negative fraction")
    long_ret, short_ret, long_to, short_to = arrays
    # Aggregate leg API assumes equal capital per leg (50% + 50%). For
    # unequal membership counts use equal_gross_long_short_returns instead.
    return .5 * (long_ret - short_ret - cost_rate * (long_to + short_to))


def _validate_missing_return_policy(policy: str) -> str:
    if policy not in _MISSING_RETURN_POLICIES:
        raise ValueError(
            f"missing_return_policy must be one of "
            f"{_MISSING_RETURN_POLICIES}, got {policy!r}"
        )
    return policy


def _apply_missing_return_policy(
    values: np.ndarray, policy: str, context: str
) -> np.ndarray:
    """Apply the missing-return policy to a (T,) or (T, F) return series.

    Returns the series to use downstream. For "drop", non-finite entries
    stay NaN (callers must propagate NaN); for "zero_fill" they become 0;
    for "fail" a ValueError is raised.
    """
    _validate_missing_return_policy(policy)
    non_finite = ~np.isfinite(values)
    if policy == "fail" and np.any(non_finite):
        n_missing = int(np.sum(non_finite))
        raise ValueError(
            f"{context} contains {n_missing} non-finite value(s) and "
            "missing_return_policy='fail'"
        )
    if policy == "zero_fill":
        return np.where(non_finite, 0.0, values)
    return values  # "drop": keep NaN, callers propagate


def compute_long_short_returns(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
    missing_return_policy: str = "zero_fill",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long/short portfolio returns based on factor quantiles.

    Missing-return policy (QE-METRIC P0-10):

    - ``"zero_fill"`` (default, back-compat): forward returns that are NaN
      are treated as 0 for the assets selected into the long/short buckets.
      This affects bucket means (a NaN-return asset contributes 0 instead
      of being excluded) and is documented here precisely because it can
      bias portfolio returns toward 0 in sparse universes.
    - ``"drop"``: assets with non-finite forward returns are excluded from
      the bucket means; if a bucket ends up empty, that period's return is
      NaN (never 0).
    - ``"fail"``: raise ValueError if any forward return is non-finite.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        long_threshold: Quantile threshold for long positions (default 0.8 = top 20%)
        short_threshold: Quantile threshold for short positions (default 0.2 = bottom 20%)
        validity_mask: Optional boolean mask (T, N) or (T, N, F)
        missing_return_policy: "zero_fill" | "drop" | "fail" (see above)

    Returns:
        (long_returns, short_returns, long_short_returns)
        Each shape (T,) or (T, F) for time series of portfolio returns
    """
    _validate_missing_return_policy(missing_return_policy)
    if missing_return_policy == "fail" and np.any(~np.isfinite(forward_returns)):
        n_missing = int(np.sum(~np.isfinite(forward_returns)))
        raise ValueError(
            f"forward_returns contains {n_missing} non-finite value(s) and "
            "missing_return_policy='fail'"
        )

    factor_values, forward_returns = np.asarray(factor_values), np.asarray(forward_returns)
    if factor_values.ndim not in (2, 3) or forward_returns.shape != factor_values.shape[:2]:
        raise ValueError("factor and label axes must match (T,N[,F]) and (T,N)")
    if not 0 <= short_threshold < long_threshold <= 1:
        raise ValueError("require 0 <= short_threshold < long_threshold <= 1")
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask)
        if validity_mask.dtype != np.bool_ or validity_mask.shape not in (factor_values.shape[:2], factor_values.shape):
            raise ValueError("validity_mask must be boolean (T,N) or match factor axes")
    # Handle 3D factor values
    if factor_values.ndim == 3:
        T, N, F = factor_values.shape
        long_rets = np.full((T, F), np.nan)
        short_rets = np.full((T, F), np.nan)
        ls_rets = np.full((T, F), np.nan)

        for f in range(F):
            fv = factor_values[:, :, f]
            vm = (validity_mask if validity_mask.ndim == 2 else validity_mask[:, :, f]) if validity_mask is not None else None
            long_rets[:, f], short_rets[:, f], ls_rets[:, f] = compute_long_short_returns(
                fv, forward_returns, long_threshold, short_threshold, vm,
                missing_return_policy=missing_return_policy,
            )
        return long_rets, short_rets, ls_rets

    # 2D case
    T, N = factor_values.shape
    long_returns = np.full(T, np.nan)
    short_returns = np.full(T, np.nan)
    long_short_returns = np.full(T, np.nan)

    for t in range(T):
        factor_t = factor_values[t, :]
        ret_t = forward_returns[t, :]

        # Apply validity mask
        if validity_mask is not None:
            valid = validity_mask[t, :]
            factor_t = np.where(valid, factor_t, np.nan)

        # Filter finite factor values. Missing-return policy:
        # - "zero_fill": buckets are formed on finite factors only; NaN
        #   forward returns contribute 0 to the bucket mean (documented).
        # - "drop": assets with non-finite returns are excluded entirely.
        finite_mask = np.isfinite(factor_t)
        if missing_return_policy == "zero_fill":
            ret_t = np.where(np.isfinite(ret_t), ret_t, 0.0)

        if np.sum(finite_mask) < 2:
            # QE-R2 (P0-FA-015 hardening): a long/short bucket needs at least
            # two valid cross-sectional observations (a single asset cannot
            # form a top-20%/bottom-20% bucket pair).  Leave the period NaN —
            # an empty bucket is never a fabricated 0.
            continue

        factor_valid = factor_t[finite_mask]
        ret_valid = ret_t[finite_mask]

        # Compute quantiles
        long_cutoff = np.quantile(factor_valid, long_threshold)
        short_cutoff = np.quantile(factor_valid, short_threshold)

        # Select long/short positions
        long_mask = factor_valid >= long_cutoff
        short_mask = factor_valid <= short_cutoff
        if np.any(long_mask & short_mask):
            continue

        if np.sum(long_mask) > 0:
            long_returns[t] = np.mean(ret_valid[long_mask])

        if np.sum(short_mask) > 0:
            short_returns[t] = np.mean(ret_valid[short_mask])

        if np.sum(long_mask) > 0 and np.sum(short_mask) > 0:
            long_short_returns[t] = equal_gross_long_short_returns(
                long_mask[None, :], short_mask[None, :], ret_valid[None, :],
                missing_return_policy=missing_return_policy)[0]

    return long_returns, short_returns, long_short_returns


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        min_periods: Minimum periods required

    Returns:
        Sharpe ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    if T == 0:
        return float("nan") if squeeze else np.full(F, np.nan, dtype=np.float64)
    sharpe = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Compute excess returns
        rf_per_period = risk_free_rate / periods_per_year
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)
        std_excess = np.std(excess_ret, ddof=1)

        if not np.isfinite(std_excess) or std_excess <= 1e-10:
            continue

        # Annualize
        sharpe[f] = mean_excess / std_excess * np.sqrt(periods_per_year)

    return sharpe[0] if squeeze else sharpe


def compute_maximum_drawdown(
    returns: np.ndarray,
    missing_return_policy: str = "zero_fill",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute maximum drawdown from return series.

    This is the drawdown authority alongside
    ``metrics/risk/drawdown_analysis.py``; both share the same wealth<=0
    wipeout guard semantics.

    CAVEAT (silent zero fill): by default (``missing_return_policy=
    "zero_fill"``) NaN returns are treated as flat days (0 return) for the
    wealth curve. Pass ``missing_return_policy="fail"`` to raise instead
    when any NaN is present.

    Args:
        returns: Return series (T,) or (T, F)
        missing_return_policy: "zero_fill" (default, back-compat) or "fail"
            (raise ValueError on any NaN return)

    Returns:
        (max_drawdown, drawdown_series, peak_indices)
        max_drawdown: Maximum drawdown magnitude (positive), shape () or (F,)
        drawdown_series: Drawdown at each time step, shape (T,) or (T, F);
            -1 from the first nonpositive wealth onward (absorbing total loss)
        peak_indices: Index of the PEAK (last index where the running
            maximum is attained at or before the maximum-drawdown trough),
            shape () or (F,); -1 denotes initial capital before the first return.
    """
    if missing_return_policy not in ("zero_fill", "fail"):
        raise ValueError(
            f"missing_return_policy must be 'zero_fill' or 'fail', "
            f"got {missing_return_policy!r}"
        )

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    if missing_return_policy == "fail" and np.any(~np.isfinite(returns)):
        n_missing = int(np.sum(~np.isfinite(returns)))
        raise ValueError(
            f"returns contains {n_missing} non-finite value(s) and "
            "missing_return_policy='fail'"
        )

    # Replace NaN with 0 for cumulative product (documented caveat above).
    returns_filled = np.where(np.isfinite(returns), returns, 0.0)

    # Compute cumulative returns (wealth curve)
    cum_returns = np.cumprod(1.0 + returns_filled, axis=0)

    # Include initial capital so an initial loss is part of the drawdown.
    running_max = np.maximum(1.0, np.maximum.accumulate(cum_returns, axis=0))

    # Drawdown series with total-wipeout guard: from the first nonpositive
    # wealth onward, drawdown is -1 (total loss) — a negative wealth
    # times (1 + r) can flip positive again and fabricate a fake recovery.
    invalid = np.maximum.accumulate(cum_returns <= 0, axis=0)
    drawdown_series = np.full(cum_returns.shape, np.nan)
    np.divide(
        cum_returns - running_max,
        running_max,
        out=drawdown_series,
        where=~invalid,
    )
    drawdown_series[invalid] = -1.0

    # Maximum drawdown per factor (most negative, converted to positive).
    with np.errstate(invalid="ignore"):
        max_dd = -np.nanmin(drawdown_series, axis=0)
    max_dd = np.where(np.isfinite(max_dd), max_dd, np.nan)

    # Trough index per factor: first occurrence of the minimum drawdown,
    # NaN-safe (an all-NaN column has no
    # defined trough; np.nanargmin would raise on it).
    trough_indices = np.empty(F, dtype=np.int64)
    for f in range(F):
        col = drawdown_series[:, f]
        finite_idx = np.nonzero(np.isfinite(col))[0]
        if finite_idx.size == 0:
            trough_indices[f] = 0
            continue
        vals = col[finite_idx]
        min_val = np.min(vals)
        trough_indices[f] = int(finite_idx[np.nonzero(vals == min_val)[0][0]])

    # Peak index: last index at or before the trough where the wealth curve
    # attains its running maximum (i.e. cum_returns == running_max). This is
    # the true peak of the maximum drawdown episode, not the trough.
    peak_indices = np.empty(F, dtype=np.int64)
    for f in range(F):
        trough = int(trough_indices[f])
        col = drawdown_series[: trough + 1, f]
        finite_idx = np.nonzero(np.isfinite(col))[0]
        if finite_idx.size == 0:
            # Entire prefix nonfinite: peak undefined,
            # use index 0.
            peak_indices[f] = 0
            continue
        trough_eff = int(finite_idx[-1])
        at_max = np.isclose(
            cum_returns[: trough_eff + 1, f], running_max[trough_eff, f]
        )
        if not np.any(at_max):
            peak_indices[f] = -1
            continue
        # Last index where wealth equals the running max at the trough.
        peak_indices[f] = int(np.nonzero(at_max)[0][-1])

    if squeeze:
        return max_dd[0], drawdown_series[:, 0], int(peak_indices[0])
    else:
        return max_dd, drawdown_series, peak_indices


def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Calmar ratio (annualized return / maximum drawdown).

    Args:
        returns: Return series (T,) or (T, F)
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Calmar ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    calmar = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Annualized return
        ann_ret = compute_compound_annualized_return(ret_valid, periods_per_year, min_periods=min_periods)

        # Maximum drawdown
        max_dd, _, _ = compute_maximum_drawdown(ret_valid)

        if not np.isfinite(max_dd) or max_dd <= 1e-12:
            continue

        calmar[f] = ann_ret / max_dd

    return calmar[0] if squeeze else calmar


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Sortino ratio (excess return / downside deviation).

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Sortino ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    sortino = np.full(F, np.nan)

    rf_per_period = risk_free_rate / periods_per_year

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)

        # Downside deviation (only negative excess returns)
        downside_ret = excess_ret[excess_ret < 0]
        if len(downside_ret) == 0:
            continue

        downside_std = np.sqrt(np.mean(downside_ret ** 2))

        if not np.isfinite(downside_std) or downside_std <= 1e-12:
            continue

        # Annualize
        sortino[f] = mean_excess / downside_std * np.sqrt(periods_per_year)

    return sortino[0] if squeeze else sortino


def compute_win_rate(
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute win rate (fraction of positive returns).

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        Win rate in [0, 1], scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    valid = np.isfinite(returns)
    n_valid = np.sum(valid, axis=0)

    wins = np.sum((returns > 0) & valid, axis=0)
    win_rate = wins / np.maximum(n_valid, 1)

    # Set to NaN if no valid observations
    win_rate = np.where(n_valid > 0, win_rate, np.nan)

    return win_rate[0] if squeeze else win_rate
