"""
Structural break tests for time series.

Structural breaks occur when relationships or parameters change over time due to
regime shifts, policy changes, or market disruptions. Detection is critical for
model stability and out-of-sample forecasting.

References:
    Chow, G.C. (1960). "Tests of Equality Between Sets of Coefficients in Two
    Linear Regressions". Econometrica. 28 (3): 591-605. doi:10.2307/1910133

    Brown, R.L.; Durbin, J.; Evans, J.M. (1975). "Techniques for Testing the
    Constancy of Regression Relationships over Time". Journal of the Royal
    Statistical Society, Series B. 37 (2): 149-192.

    Andrews, D.W.K. (1993). "Tests for Parameter Instability and Structural
    Change with Unknown Change Point". Econometrica. 61 (4): 821-856.
    doi:10.2307/2951764

    Bai, J.; Perron, P. (2003). "Computation and Analysis of Multiple Structural
    Change Models". Journal of Applied Econometrics. 18 (1): 1-22.
    doi:10.1002/jae.659
"""

from typing import Tuple, Optional, List
import numpy as np
from scipy import stats


def chow_test(
    y: np.ndarray,
    X: np.ndarray,
    breakpoint: int,
    min_obs: int = 10,
) -> Tuple[float, float, int, int, int]:
    """
    Chow test for structural break at known breakpoint.

    Tests null hypothesis that coefficients are equal in two subsamples.
    F-statistic = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n - 2k))

    Args:
        y: Dependent variable (T,)
        X: Independent variables (T, K) or (T,) for univariate
        breakpoint: Index of breakpoint (must be in [min_obs, T-min_obs])
        min_obs: Minimum observations per subsample

    Returns:
        (f_stat, p_value, df1, df2, k)
        f_stat: Chow F-statistic
        p_value: p-value from F(k, n-2k) distribution
        df1: numerator degrees of freedom (k)
        df2: denominator degrees of freedom (n - 2k)
        k: number of parameters

    Raises:
        ValueError: If invalid inputs or insufficient data
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    T = len(y)

    # Handle univariate X
    if X.ndim == 1:
        X = X[:, np.newaxis]

    if X.ndim != 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Chow requires a complete finite joint y/X calendar")
    if isinstance(breakpoint, (bool, np.bool_)) or not isinstance(breakpoint, (int, np.integer)):
        raise ValueError("breakpoint must be an integer")

    if X.shape[0] != T:
        raise ValueError(f"X and y must have same length: {X.shape[0]} vs {T}")

    K = X.shape[1]

    if breakpoint < min_obs or breakpoint > T - min_obs:
        raise ValueError(
            f"Breakpoint {breakpoint} must be in [{min_obs}, {T - min_obs}]"
        )

    if K >= min_obs:
        raise ValueError(f"Number of parameters {K} must be less than min_obs {min_obs}")

    # Add intercept
    X_aug = np.column_stack([np.ones(T), X])
    k = K + 1  # Total parameters including intercept

    # Split at breakpoint
    y1, y2 = y[:breakpoint], y[breakpoint:]
    X1, X2 = X_aug[:breakpoint], X_aug[breakpoint:]

    n1, n2 = len(y1), len(y2)
    n = n1 + n2

    if n1 <= k or n2 <= k:
        raise ValueError(f"Insufficient observations in subsamples: {n1}, {n2} < {k}")
    if any(np.linalg.matrix_rank(design) != k for design in (X_aug, X1, X2)):
        raise ValueError("Chow requires full-rank pooled and segment designs")

    # Fit pooled model
    beta_pooled = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    residuals_pooled = y - X_aug @ beta_pooled
    rss_pooled = np.sum(residuals_pooled**2)

    # Fit subsample models
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    residuals1 = y1 - X1 @ beta1
    rss1 = np.sum(residuals1**2)

    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    residuals2 = y2 - X2 @ beta2
    rss2 = np.sum(residuals2**2)

    # Chow F-statistic
    numerator = (rss_pooled - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n - 2 * k)

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan, np.nan, k, n - 2 * k, k

    f_stat = numerator / denominator

    # p-value from F distribution
    df1 = k
    df2 = n - 2 * k
    p_value = stats.f.sf(f_stat, df1, df2)

    return f_stat, p_value, df1, df2, k


def cusum_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
    *, residual_kind: str = "recursive",
) -> Tuple[np.ndarray, float, float, bool]:
    """
    CUSUM (Cumulative Sum) test for parameter stability.

    Explicit OLS-residual CUSUM, using statsmodels' asymptotic boundary.
    The legacy recursive mode is unqualified and fails closed.

    Args:
        residuals: OLS residuals from the declared regression (T,)
        alpha: Significance level (0.01, 0.05, or 0.10)
        residual_kind: Must explicitly be 'ols'; recursive is unsupported.

    Returns:
        (cusum_stats, boundary, max_stat, is_stable)
        cusum_stats: Cumulative sum statistic at each time (T,)
        boundary: Critical boundary value
        max_stat: Maximum absolute CUSUM statistic
        is_stable: True if no break detected (max_stat < boundary)

    Note:
        Accepts pre-computed residuals; this does not certify their fit lineage.
    """
    if alpha not in (.01, .05, .10):
        raise ValueError("unsupported CUSUM alpha")
    if residual_kind != "ols":
        raise ValueError("recursive-residual CUSUM is not qualified; request residual_kind='ols' with OLS residuals")
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 1 or len(residuals) < 20 or not np.isfinite(residuals).all():
        raise ValueError("CUSUM requires a complete finite residual calendar of at least 20 periods")
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    if np.sum(residuals**2) <= 0:
        raise ValueError("zero residual energy")
    statistic, _, critical = breaks_cusumolsresid(residuals, ddof=0)
    boundary = dict(critical)[int(round(100*alpha))]
    path = np.cumsum(residuals)/np.sqrt(np.sum(residuals**2))
    return path, float(boundary), float(statistic), bool(statistic < boundary)


def cusum_of_squares_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
    *, research_only: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], float, bool]:
    """
    CUSUM of Squares test for variance stability.

    Tests whether cumulative sum of squared residuals stays within bounds.
    More sensitive to variance changes than CUSUM.

    Args:
        residuals: Residuals from regression (T,)
        alpha: Significance level

    Returns:
        (cusumsq_stats, bounds, max_deviation, is_stable)
        cusumsq_stats: Cumulative sum of squares at each time (T,)
        bounds: (lower, upper) critical bounds
        max_deviation: Maximum deviation from bounds
        is_stable: True if stays within bounds

    Reference:
        Brown, Durbin, Evans (1975), Section 4.2
    """
    if research_only is not True:
        raise ValueError("CUSUM squares boundary is HEURISTIC; explicit research_only=True required")
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError("Residuals must be 1-dimensional")

    # Remove NaN
    mask = np.isfinite(residuals)
    residuals = residuals[mask]
    T = len(residuals)

    if T < 20:
        raise ValueError(f"Insufficient data: T={T} < 20")

    # Squared residuals
    sq_residuals = residuals**2
    sum_sq = np.sum(sq_residuals)

    if sum_sq <= 0:
        raise ValueError("Sum of squared residuals is zero")

    # Cumulative sum of squares (normalized)
    cusum_sq = np.cumsum(sq_residuals) / sum_sq

    # Critical bounds: approximately c_α ± (t/T) where c_α depends on alpha
    # Asymptotic bounds under null of stability
    if alpha not in (0.01, 0.05, 0.10):
        raise ValueError("unsupported CUSUM-squares alpha")
    c_alpha = {
        0.01: 1.63,
        0.05: 1.36,
        0.10: 1.22,
    }[alpha]

    t_normalized = np.arange(1, T + 1) / T
    lower_bound = t_normalized - c_alpha * np.sqrt(t_normalized * (1 - t_normalized))
    upper_bound = t_normalized + c_alpha * np.sqrt(t_normalized * (1 - t_normalized))

    # Clip bounds to [0, 1]
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.minimum(upper_bound, 1)

    # Check if CUSUM-sq stays within bounds
    below_lower = np.sum(cusum_sq < lower_bound)
    above_upper = np.sum(cusum_sq > upper_bound)

    is_stable = (below_lower == 0) and (above_upper == 0)

    # Maximum deviation from bounds
    dev_lower = np.max(np.maximum(lower_bound - cusum_sq, 0))
    dev_upper = np.max(np.maximum(cusum_sq - upper_bound, 0))
    max_deviation = max(dev_lower, dev_upper)

    return cusum_sq, (lower_bound, upper_bound), max_deviation, is_stable


def sup_wald_test(
    y: np.ndarray,
    X: np.ndarray,
    trim: float = 0.15,
    min_obs: int = 10,
) -> Tuple[float, int, float]:
    """
    Supremum Wald test for structural break at unknown location.

    Tests for a single break at unknown location by computing Chow test over
    all potential breakpoints and taking the maximum (supremum).

    Args:
        y: Dependent variable (T,)
        X: Independent variables (T, K) or (T,)
        trim: Fraction to trim from start/end (default 0.15)
        min_obs: Minimum observations per subsample

    Returns:
        (sup_stat, breakpoint, unavailable_p_value)
        sup_stat: Supremum Wald statistic (max over all breakpoints)
        breakpoint: Estimated breakpoint location
        unavailable_p_value: NaN; descriptive scan is not calibrated inference.

    Reference:
        Andrews (1993), Section 2
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    T = len(y)

    if X.ndim == 1:
        X = X[:, np.newaxis]

    if X.shape[0] != T:
        raise ValueError(f"X and y must have same length: {X.shape[0]} vs {T}")

    # Determine search range for breakpoint
    start = max(min_obs, int(T * trim))
    end = min(T - min_obs, int(T * (1 - trim)))

    if start >= end:
        raise ValueError(f"Insufficient data for trimming: start={start}, end={end}")

    # Compute Chow F-statistic over all candidate breakpoints
    f_stats = []
    breakpoints = []

    for bp in range(start, end + 1):
        try:
            f_stat, _, _, _, _ = chow_test(y, X, bp, min_obs=min_obs)
            if np.isfinite(f_stat):
                f_stats.append(f_stat)
                breakpoints.append(bp)
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(f_stats) == 0:
        raise ValueError("No valid breakpoints found")

    f_stats = np.array(f_stats)
    breakpoints = np.array(breakpoints)

    # Supremum statistic
    sup_idx = np.argmax(f_stats)
    sup_stat = f_stats[sup_idx]
    breakpoint = breakpoints[sup_idx]

    # Coarse critical-value bands are not calibrated p-values. Preserve the
    # descriptive supremum/location; missing inference cannot enter FDR.
    return sup_stat, breakpoint, float("nan")


def bai_perron_test(
    y: np.ndarray,
    X: np.ndarray,
    max_breaks: int = 5,
    trim: float = 0.15,
    min_obs: int = 10,
) -> Tuple[int, List[int], float]:
    """
    Bai-Perron test for multiple structural breaks.

    Sequential procedure to estimate number and location of multiple breaks
    by minimizing sum of squared residuals.

    Args:
        y: Dependent variable (T,)
        X: Independent variables (T, K) or (T,)
        max_breaks: Maximum number of breaks to search
        trim: Minimum segment length as fraction of T
        min_obs: Minimum observations per segment

    Returns:
        (n_breaks, breakpoints, bic)
        n_breaks: Estimated number of breaks
        breakpoints: List of breakpoint locations
        bic: Bayesian Information Criterion for selected model

    Note:
        This is a simplified implementation. Full Bai-Perron involves dynamic
        programming for global minimization. Here we use sequential addition.

    Reference:
        Bai & Perron (2003), Section 2
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    T = len(y)

    if X.ndim == 1:
        X = X[:, np.newaxis]

    K = X.shape[1]
    X_aug = np.column_stack([np.ones(T), X])
    k = K + 1

    min_segment = max(min_obs, int(T * trim))
    if isinstance(max_breaks, (bool, np.bool_)) or not isinstance(max_breaks, (int, np.integer)) or max_breaks < 0:
        raise ValueError("max_breaks must be a nonnegative integer")
    if not 0 < trim < .5 or min_obs < 1 or T < min_segment or min_segment <= k:
        raise ValueError("invalid or infeasible segment policy")
    if X.shape[0] != T or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("sequential breaks require a complete finite joint calendar")

    # Sequential search for breaks
    breakpoints = []
    prev_bic = np.inf
    n_breaks = 0
    accepted_breakpoints = []

    for m in range(max_breaks + 1):
        if m == 0:
            # No breaks: single regression
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            residuals = y - X_aug @ beta
            rss = np.sum(residuals**2)
            n_params = k
        else:
            # m breaks: find best additional breakpoint
            candidate_breaks = []
            candidate_rss = []

            # Try adding a break in each feasible location
            for bp in range(min_segment, T - min_segment + 1):
                # Check if bp creates valid segments
                all_bp = sorted(breakpoints + [bp])
                segments = [0] + all_bp + [T]

                valid = True
                for i in range(len(segments) - 1):
                    seg_len = segments[i + 1] - segments[i]
                    if seg_len < min_segment:
                        valid = False
                        break

                if not valid:
                    continue

                # Fit model with all breaks
                rss_total = 0
                for i in range(len(segments) - 1):
                    start, end = segments[i], segments[i + 1]
                    y_seg = y[start:end]
                    X_seg = X_aug[start:end]

                    if len(y_seg) < k:
                        valid = False
                        break

                    try:
                        beta_seg = np.linalg.lstsq(X_seg, y_seg, rcond=None)[0]
                        resid_seg = y_seg - X_seg @ beta_seg
                        rss_total += np.sum(resid_seg**2)
                    except np.linalg.LinAlgError:
                        valid = False
                        break

                if valid:
                    candidate_breaks.append(bp)
                    candidate_rss.append(rss_total)

            if len(candidate_breaks) == 0:
                break  # No more valid breaks

            # Select break with minimum RSS
            best_idx = np.argmin(candidate_rss)
            breakpoints.append(candidate_breaks[best_idx])
            breakpoints.sort()
            rss = candidate_rss[best_idx]
            n_params = k * (m + 1)

        # Compute BIC: n*log(RSS/n) + k*log(n)
        if rss > 0:
            bic = T * np.log(rss / T) + n_params * np.log(T)
        else:
            bic = -np.inf

        # Stop if BIC increases (overfitting)
        if bic > prev_bic:
            breakpoints = accepted_breakpoints.copy()
            n_breaks = len(breakpoints)
            bic = prev_bic
            break

        prev_bic = bic
        accepted_breakpoints = breakpoints.copy()
        n_breaks = len(breakpoints)

        if m == max_breaks:
            n_breaks = max_breaks
            bic = prev_bic

    return n_breaks, accepted_breakpoints, prev_bic
