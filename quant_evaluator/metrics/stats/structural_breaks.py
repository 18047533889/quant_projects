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

    if n1 < k or n2 < k:
        raise ValueError(f"Insufficient observations in subsamples: {n1}, {n2} < {k}")

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
    p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)

    return f_stat, p_value, df1, df2, k


def cusum_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    CUSUM (Cumulative Sum) test for parameter stability.

    Tests whether cumulative sum of recursive residuals stays within confidence
    bounds. Detects gradual parameter drift.

    Args:
        residuals: Recursive residuals from rolling regression (T,)
        alpha: Significance level (0.01, 0.05, or 0.10)

    Returns:
        (cusum_stats, boundary, max_stat, is_stable)
        cusum_stats: Cumulative sum statistic at each time (T,)
        boundary: Critical boundary value
        max_stat: Maximum absolute CUSUM statistic
        is_stable: True if no break detected (max_stat < boundary)

    Note:
        Input should be recursive residuals, not OLS residuals.
        For simplicity, this accepts pre-computed residuals.
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError("Residuals must be 1-dimensional")

    T = len(residuals)

    # Remove NaN
    mask = np.isfinite(residuals)
    residuals = residuals[mask]
    T = len(residuals)

    if T < 20:
        raise ValueError(f"Insufficient data: T={T} < 20")

    # Standardize residuals
    sigma = np.std(residuals, ddof=1)
    if sigma <= 0:
        raise ValueError("Residuals have zero variance")

    standardized = residuals / sigma

    # Cumulative sum
    cusum_stats = np.cumsum(standardized)

    # Critical boundary: c * sqrt(T) where c depends on alpha
    # Approximate values from Brown, Durbin, Evans (1975)
    critical_values = {
        0.01: 1.143,
        0.05: 0.948,
        0.10: 0.850,
    }

    c = critical_values.get(alpha, 0.948)
    boundary = c * np.sqrt(T)

    # Test statistic: max |CUSUM_t|
    max_stat = np.max(np.abs(cusum_stats))

    is_stable = max_stat < boundary

    return cusum_stats, boundary, max_stat, is_stable


def cusum_of_squares_test(
    residuals: np.ndarray,
    alpha: float = 0.05,
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
    c_alpha = {
        0.01: 1.63,
        0.05: 1.36,
        0.10: 1.22,
    }.get(alpha, 1.36)

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
        (sup_stat, breakpoint, p_value_approx)
        sup_stat: Supremum Wald statistic (max over all breakpoints)
        breakpoint: Estimated breakpoint location
        p_value_approx: Approximate p-value

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

    # Approximate p-value (very rough, based on Andrews 1993 Table I)
    # For 1 regressor, trim=0.15, critical values at 10%/5%/1% ≈ 7.04/8.68/11.79
    if sup_stat > 11.79:
        p_value_approx = 0.01
    elif sup_stat > 8.68:
        p_value_approx = 0.05
    elif sup_stat > 7.04:
        p_value_approx = 0.10
    else:
        p_value_approx = 0.20

    return sup_stat, breakpoint, p_value_approx


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

    # Sequential search for breaks
    breakpoints = []
    prev_bic = np.inf

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
            for bp in range(min_segment, T - min_segment):
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
            breakpoints = breakpoints[:-1]  # Remove last break
            n_breaks = m - 1
            bic = prev_bic
            break

        prev_bic = bic

        if m == max_breaks:
            n_breaks = max_breaks
            bic = prev_bic

    return n_breaks, breakpoints, bic
