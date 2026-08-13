"""
Test assertion helpers for integration testing.

Provides specialized assertions for validating factor data, timing consistency,
and evaluation results.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
import warnings


def assert_panel_shape(
    data: np.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "data",
):
    """
    Assert that panel data has expected shape.

    Args:
        data: Panel array
        expected_shape: Expected (T, N, F) or (T, N) shape
        name: Name for error messages
    """
    if data.shape != expected_shape:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, got {data.shape}"
        )


def assert_factor_properties(
    factor_values: np.ndarray,
    check_finite: bool = True,
    check_range: Optional[Tuple[float, float]] = None,
    max_missing_rate: Optional[float] = None,
    name: str = "factor",
):
    """
    Assert factor data satisfies basic quality properties.

    Args:
        factor_values: Factor panel (T, N, F) or (T, N)
        check_finite: Verify no infinite values
        check_range: Optional (min, max) valid range
        max_missing_rate: Maximum allowed fraction of NaN values
        name: Factor name for error messages
    """
    if check_finite:
        inf_mask = np.isinf(factor_values)
        if np.any(inf_mask):
            n_inf = np.sum(inf_mask)
            raise AssertionError(
                f"{name} contains {n_inf} infinite values"
            )

    if check_range is not None:
        min_val, max_val = check_range
        valid_data = factor_values[~np.isnan(factor_values)]
        if len(valid_data) > 0:
            actual_min = np.min(valid_data)
            actual_max = np.max(valid_data)
            if actual_min < min_val or actual_max > max_val:
                raise AssertionError(
                    f"{name} values outside expected range [{min_val}, {max_val}]: "
                    f"actual range [{actual_min:.4f}, {actual_max:.4f}]"
                )

    if max_missing_rate is not None:
        missing_rate = np.sum(np.isnan(factor_values)) / factor_values.size
        if missing_rate > max_missing_rate:
            raise AssertionError(
                f"{name} missing rate {missing_rate:.2%} exceeds threshold "
                f"{max_missing_rate:.2%}"
            )


def assert_timing_consistency(
    decision_time: Tuple[str, ...],
    label_start_time: Tuple[str, ...],
    label_end_time: Tuple[str, ...],
    min_horizon: int = 0,
):
    """
    Assert timing fields are consistent and properly ordered.

    Args:
        decision_time: Tuple of decision timestamps
        label_start_time: Tuple of label start timestamps
        label_end_time: Tuple of label end timestamps
        min_horizon: Minimum required horizon between decision and label
    """
    T = len(decision_time)

    # Check lengths match
    if len(label_start_time) != T:
        raise AssertionError(
            f"label_start_time length {len(label_start_time)} != decision_time length {T}"
        )
    if len(label_end_time) != T:
        raise AssertionError(
            f"label_end_time length {len(label_end_time)} != decision_time length {T}"
        )

    # Check temporal ordering
    for i in range(T):
        dt = datetime.fromisoformat(decision_time[i])
        ls = datetime.fromisoformat(label_start_time[i])
        le = datetime.fromisoformat(label_end_time[i])

        # Decision must be before or at label start
        if dt > ls:
            raise AssertionError(
                f"At index {i}: decision_time {decision_time[i]} is after "
                f"label_start_time {label_start_time[i]}"
            )

        # Label start must be before or at label end
        if ls > le:
            raise AssertionError(
                f"At index {i}: label_start_time {label_start_time[i]} is after "
                f"label_end_time {label_end_time[i]}"
            )

        # Check minimum horizon
        if min_horizon > 0:
            horizon_days = (ls - dt).days
            if horizon_days < min_horizon:
                raise AssertionError(
                    f"At index {i}: horizon {horizon_days} days is less than "
                    f"minimum {min_horizon} days"
                )


def assert_numeric_close(
    actual: float,
    expected: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "value",
):
    """
    Assert numeric values are close within tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Value name for error messages
    """
    if np.isnan(actual) and np.isnan(expected):
        return

    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        diff = abs(actual - expected)
        rel_diff = diff / (abs(expected) + 1e-10)
        raise AssertionError(
            f"{name} mismatch: expected {expected:.6f}, got {actual:.6f} "
            f"(abs diff: {diff:.6e}, rel diff: {rel_diff:.6e})"
        )


def assert_no_lookahead(
    factor_values: np.ndarray,
    label_values: np.ndarray,
    decision_time: Tuple[str, ...],
    label_start_time: Tuple[str, ...],
    min_lag: int = 1,
):
    """
    Assert no lookahead bias between factors and labels.

    Checks that factor values at time t do not depend on label values
    from time t+min_lag or later.

    Args:
        factor_values: (T, N, F) factor panel
        label_values: (T, N) label panel
        decision_time: Decision timestamps
        label_start_time: Label start timestamps
        min_lag: Minimum required lag in periods
    """
    T = len(decision_time)

    if factor_values.shape[0] != T:
        raise AssertionError(
            f"factor_values time dimension {factor_values.shape[0]} != "
            f"decision_time length {T}"
        )

    if label_values.shape[0] != T:
        raise AssertionError(
            f"label_values time dimension {label_values.shape[0]} != "
            f"decision_time length {T}"
        )

    # Check timing structure ensures no lookahead
    for i in range(T - min_lag):
        dt = datetime.fromisoformat(decision_time[i])
        future_label_start = datetime.fromisoformat(label_start_time[i + min_lag])

        if dt >= future_label_start:
            warnings.warn(
                f"Potential lookahead at index {i}: decision_time {decision_time[i]} "
                f"is at or after label_start_time at {i+min_lag} ({label_start_time[i+min_lag]})",
                UserWarning
            )


def compare_evaluation_results(
    result1: Dict[str, Any],
    result2: Dict[str, Any],
    metric_names: Optional[List[str]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, bool]:
    """
    Compare two evaluation result dictionaries.

    Args:
        result1: First evaluation result
        result2: Second evaluation result
        metric_names: Optional list of metrics to compare (default: all common)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary mapping metric names to match status (True/False)
    """
    if metric_names is None:
        metric_names = list(set(result1.keys()) & set(result2.keys()))

    comparison = {}

    for metric_name in metric_names:
        if metric_name not in result1:
            warnings.warn(f"Metric {metric_name} not in result1")
            comparison[metric_name] = False
            continue

        if metric_name not in result2:
            warnings.warn(f"Metric {metric_name} not in result2")
            comparison[metric_name] = False
            continue

        val1 = result1[metric_name]
        val2 = result2[metric_name]

        # Handle nested dictionaries (metric results often have 'value', 'std_error', etc.)
        if isinstance(val1, dict) and isinstance(val2, dict):
            if 'value' in val1 and 'value' in val2:
                val1 = val1['value']
                val2 = val2['value']
            else:
                # Compare all common keys
                all_match = True
                for key in set(val1.keys()) & set(val2.keys()):
                    if not np.isclose(val1[key], val2[key], rtol=rtol, atol=atol, equal_nan=True):
                        all_match = False
                        break
                comparison[metric_name] = all_match
                continue

        # Scalar comparison
        try:
            match = np.isclose(val1, val2, rtol=rtol, atol=atol, equal_nan=True)
            comparison[metric_name] = bool(match)
        except (TypeError, ValueError):
            # Non-numeric comparison
            comparison[metric_name] = (val1 == val2)

    return comparison


def assert_correlation_structure(
    factor_panel: np.ndarray,
    expected_correlation: np.ndarray,
    rtol: float = 0.1,
):
    """
    Assert factor panel has expected correlation structure.

    Args:
        factor_panel: (T, N, F) factor panel
        expected_correlation: (F, F) expected correlation matrix
        rtol: Relative tolerance for correlation comparison
    """
    T, N, F = factor_panel.shape

    if expected_correlation.shape != (F, F):
        raise ValueError(
            f"expected_correlation must be ({F}, {F}), got {expected_correlation.shape}"
        )

    # Compute empirical correlation
    reshaped = factor_panel.reshape(-1, F)
    valid_mask = ~np.isnan(reshaped).any(axis=1)
    valid_data = reshaped[valid_mask]

    if len(valid_data) < F:
        raise AssertionError(
            f"Insufficient valid observations ({len(valid_data)}) to compute "
            f"correlation for {F} factors"
        )

    empirical_corr = np.corrcoef(valid_data, rowvar=False)

    # Compare
    max_diff = np.max(np.abs(empirical_corr - expected_correlation))
    if max_diff > rtol:
        raise AssertionError(
            f"Correlation structure mismatch: max absolute difference {max_diff:.4f} "
            f"exceeds tolerance {rtol:.4f}"
        )


def assert_cross_sectional_neutrality(
    factor_values: np.ndarray,
    exposure_matrix: np.ndarray,
    rtol: float = 0.05,
):
    """
    Assert factor is approximately cross-sectionally neutral to exposures.

    Args:
        factor_values: (T, N) or (T, N, 1) factor panel
        exposure_matrix: (T, N, K) exposure matrix
        rtol: Relative tolerance for neutrality check
    """
    if factor_values.ndim == 3:
        factor_values = factor_values.squeeze(-1)

    T, N = factor_values.shape
    K = exposure_matrix.shape[2]

    # Check each time period
    max_correlation = 0.0

    for t in range(T):
        f_t = factor_values[t]
        e_t = exposure_matrix[t]

        valid_mask = ~np.isnan(f_t)
        if np.sum(valid_mask) < K + 10:
            continue

        f_valid = f_t[valid_mask]
        e_valid = e_t[valid_mask]

        # Compute correlation with each exposure
        for k in range(K):
            corr = np.corrcoef(f_valid, e_valid[:, k])[0, 1]
            max_correlation = max(max_correlation, abs(corr))

    if max_correlation > rtol:
        raise AssertionError(
            f"Factor not neutral: max exposure correlation {max_correlation:.4f} "
            f"exceeds tolerance {rtol:.4f}"
        )


def assert_panel_aligned(
    panel1: np.ndarray,
    panel2: np.ndarray,
    axis: int = 0,
    name1: str = "panel1",
    name2: str = "panel2",
):
    """
    Assert two panels are aligned along specified axis.

    Args:
        panel1: First panel
        panel2: Second panel
        axis: Axis to check (0=time, 1=asset, 2=factor)
        name1: Name of first panel
        name2: Name of second panel
    """
    if panel1.shape[axis] != panel2.shape[axis]:
        raise AssertionError(
            f"{name1} and {name2} not aligned on axis {axis}: "
            f"{panel1.shape[axis]} != {panel2.shape[axis]}"
        )


def assert_monotonic_increasing(
    values: np.ndarray,
    name: str = "values",
    strict: bool = False,
):
    """
    Assert array is monotonically increasing.

    Args:
        values: 1D array to check
        name: Name for error messages
        strict: If True, require strictly increasing (no equal consecutive values)
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {values.shape}")

    diffs = np.diff(values)

    if strict:
        if not np.all(diffs > 0):
            first_violation = np.argmax(diffs <= 0)
            raise AssertionError(
                f"{name} not strictly monotonic increasing at index {first_violation}: "
                f"{values[first_violation]} >= {values[first_violation + 1]}"
            )
    else:
        if not np.all(diffs >= 0):
            first_violation = np.argmax(diffs < 0)
            raise AssertionError(
                f"{name} not monotonic increasing at index {first_violation}: "
                f"{values[first_violation]} > {values[first_violation + 1]}"
            )


def assert_stationary(
    values: np.ndarray,
    max_autocorr: float = 0.95,
    name: str = "values",
):
    """
    Assert time series appears stationary (low autocorrelation).

    Args:
        values: 1D or 2D array (if 2D, checks each column)
        max_autocorr: Maximum allowed lag-1 autocorrelation
        name: Name for error messages
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    T, N = values.shape

    for n in range(N):
        series = values[:, n]
        valid_mask = ~np.isnan(series)

        if np.sum(valid_mask) < 10:
            continue

        valid_series = series[valid_mask]
        lag1_corr = np.corrcoef(valid_series[:-1], valid_series[1:])[0, 1]

        if abs(lag1_corr) > max_autocorr:
            raise AssertionError(
                f"{name} column {n} not stationary: lag-1 autocorr {lag1_corr:.4f} "
                f"exceeds threshold {max_autocorr:.4f}"
            )


def summarize_panel(data: np.ndarray, name: str = "panel") -> Dict[str, Any]:
    """
    Generate summary statistics for panel data.

    Args:
        data: Panel array
        name: Panel name

    Returns:
        Dictionary of summary statistics
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return {
            "name": name,
            "shape": data.shape,
            "all_missing": True,
        }

    return {
        "name": name,
        "shape": data.shape,
        "all_missing": False,
        "missing_rate": np.sum(np.isnan(data)) / data.size,
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "q25": float(np.percentile(valid_data, 25)),
        "q50": float(np.percentile(valid_data, 50)),
        "q75": float(np.percentile(valid_data, 75)),
        "n_inf": int(np.sum(np.isinf(data))),
    }
