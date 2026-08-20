"""Data quality checks for factor computation pipelines.

Provides statistical validation and quality gates:
- NaN rate monitoring
- Outlier detection (IQR, z-score, MAD)
- Distribution checks (skewness, kurtosis, normality)
- Cross-sectional and time-series consistency
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


__all__ = [
    "DataQualityReport",
    "OutlierStats",
    "check_data_quality",
    "check_nan_rates",
    "check_outliers",
    "check_distribution",
]


@dataclass(frozen=True)
class OutlierStats:
    """Outlier detection statistics."""

    method: str
    n_outliers: int
    outlier_fraction: float
    lower_bound: float
    upper_bound: float
    outlier_indices: tuple[int, ...] = field(repr=False)

    def summary(self) -> str:
        return (
            f"{self.method}: {self.n_outliers} outliers ({self.outlier_fraction:.2%}), "
            f"bounds=[{self.lower_bound:.3f}, {self.upper_bound:.3f}]"
        )


@dataclass(frozen=True)
class DataQualityReport:
    """Comprehensive data quality report."""

    n_rows: int
    n_cols: int
    nan_fraction: float
    inf_fraction: float
    zero_fraction: float
    outlier_stats: OutlierStats | None
    distribution: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            f"Data Quality Report: {self.n_rows} rows × {self.n_cols} cols",
            f"  NaN: {self.nan_fraction:.2%}",
            f"  Inf: {self.inf_fraction:.2%}",
            f"  Zero: {self.zero_fraction:.2%}",
        ]
        if self.outlier_stats:
            lines.append(f"  {self.outlier_stats.summary()}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append(f"  Status: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def check_nan_rates(
    values: Any,
    *,
    max_total_nan: float = 0.5,
    max_row_nan: float = 0.8,
    max_col_nan: float = 0.8,
) -> tuple[bool, dict[str, float], list[str]]:
    """Check NaN rates across multiple dimensions.

    Parameters
    ----------
    values : array-like
        Data to check (1D or 2D)
    max_total_nan : float
        Maximum allowed total NaN fraction
    max_row_nan : float
        Maximum allowed NaN fraction per row (2D only)
    max_col_nan : float
        Maximum allowed NaN fraction per column (2D only)

    Returns
    -------
    passed : bool
        Whether all checks passed
    stats : dict
        NaN statistics
    warnings : list[str]
        Warning messages for failures
    """
    arr = np.asarray(values, dtype=float)
    warnings = []
    stats = {}

    # Total NaN rate
    total_nan = np.isnan(arr).sum() / arr.size
    stats["total_nan"] = total_nan
    if total_nan > max_total_nan:
        warnings.append(f"Total NaN rate {total_nan:.2%} exceeds {max_total_nan:.2%}")

    # Per-dimension checks for 2D arrays
    if arr.ndim == 2:
        # Row-wise (time-series: each row is a timestamp)
        row_nan = np.isnan(arr).sum(axis=1) / arr.shape[1]
        max_row_nan_actual = row_nan.max()
        stats["max_row_nan"] = max_row_nan_actual
        stats["n_bad_rows"] = (row_nan > max_row_nan).sum()
        if max_row_nan_actual > max_row_nan:
            warnings.append(
                f"Max row NaN rate {max_row_nan_actual:.2%} exceeds {max_row_nan:.2%} "
                f"({stats['n_bad_rows']} rows affected)"
            )

        # Column-wise (cross-section: each column is an instrument)
        col_nan = np.isnan(arr).sum(axis=0) / arr.shape[0]
        max_col_nan_actual = col_nan.max()
        stats["max_col_nan"] = max_col_nan_actual
        stats["n_bad_cols"] = (col_nan > max_col_nan).sum()
        if max_col_nan_actual > max_col_nan:
            warnings.append(
                f"Max col NaN rate {max_col_nan_actual:.2%} exceeds {max_col_nan:.2%} "
                f"({stats['n_bad_cols']} columns affected)"
            )

    passed = len(warnings) == 0
    return passed, stats, warnings


def check_outliers(
    values: Any,
    *,
    method: str = "iqr",
    threshold: float = 3.0,
    max_outlier_fraction: float = 0.05,
) -> OutlierStats:
    """Detect outliers using various methods.

    Parameters
    ----------
    values : array-like
        Data to check (flattened for analysis)
    method : str
        Detection method: "iqr", "zscore", "mad"
    threshold : float
        Threshold parameter (IQR multiplier, z-score, or MAD multiplier)
    max_outlier_fraction : float
        Maximum allowed outlier fraction (for warnings)

    Returns
    -------
    OutlierStats
        Outlier detection results
    """
    arr = np.asarray(values, dtype=float).ravel()
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    if len(finite_vals) == 0:
        return OutlierStats(
            method=method,
            n_outliers=0,
            outlier_fraction=0.0,
            lower_bound=float("nan"),
            upper_bound=float("nan"),
            outlier_indices=(),
        )

    if method == "iqr":
        q25, q75 = np.percentile(finite_vals, [25, 75])
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
    elif method == "zscore":
        mean = np.mean(finite_vals)
        std = np.std(finite_vals)
        lower = mean - threshold * std
        upper = mean + threshold * std
    elif method == "mad":
        # Median absolute deviation
        median = np.median(finite_vals)
        mad = np.median(np.abs(finite_vals - median))
        # Scale MAD to approximate std (1.4826 for normal distribution)
        lower = median - threshold * 1.4826 * mad
        upper = median + threshold * 1.4826 * mad
    else:
        raise ValueError(f"unknown outlier method {method!r}")

    # Find outliers in original array
    outlier_mask = (arr < lower) | (arr > upper)
    outlier_indices = tuple(np.where(outlier_mask)[0].tolist())
    n_outliers = len(outlier_indices)
    outlier_fraction = n_outliers / len(arr)

    return OutlierStats(
        method=method,
        n_outliers=n_outliers,
        outlier_fraction=outlier_fraction,
        lower_bound=float(lower),
        upper_bound=float(upper),
        outlier_indices=outlier_indices,
    )


def check_distribution(
    values: Any,
    *,
    check_skewness: bool = True,
    check_kurtosis: bool = True,
    max_skewness: float = 10.0,
    max_kurtosis: float = 100.0,
) -> tuple[dict[str, float], list[str]]:
    """Check distribution properties of data.

    Parameters
    ----------
    values : array-like
        Data to analyze
    check_skewness : bool
        Whether to compute skewness
    check_kurtosis : bool
        Whether to compute kurtosis
    max_skewness : float
        Maximum allowed absolute skewness
    max_kurtosis : float
        Maximum allowed excess kurtosis

    Returns
    -------
    stats : dict
        Distribution statistics
    warnings : list[str]
        Warning messages
    """
    arr = np.asarray(values, dtype=float).ravel()
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    stats = {}
    warnings = []

    if len(finite_vals) == 0:
        warnings.append("No finite values to analyze")
        return stats, warnings

    # Basic statistics
    stats["mean"] = float(np.mean(finite_vals))
    stats["std"] = float(np.std(finite_vals))
    stats["median"] = float(np.median(finite_vals))
    stats["min"] = float(np.min(finite_vals))
    stats["max"] = float(np.max(finite_vals))
    stats["q25"] = float(np.percentile(finite_vals, 25))
    stats["q75"] = float(np.percentile(finite_vals, 75))

    # Skewness (asymmetry)
    if check_skewness and len(finite_vals) >= 3:
        mean = stats["mean"]
        std = stats["std"]
        if std > 0:
            skewness = np.mean(((finite_vals - mean) / std) ** 3)
            stats["skewness"] = float(skewness)
            if abs(skewness) > max_skewness:
                warnings.append(
                    f"Absolute skewness {abs(skewness):.2f} exceeds {max_skewness}"
                )

    # Kurtosis (tail heaviness, excess kurtosis = kurtosis - 3)
    if check_kurtosis and len(finite_vals) >= 4:
        mean = stats["mean"]
        std = stats["std"]
        if std > 0:
            kurtosis = np.mean(((finite_vals - mean) / std) ** 4)
            excess_kurtosis = kurtosis - 3.0
            stats["excess_kurtosis"] = float(excess_kurtosis)
            if abs(excess_kurtosis) > max_kurtosis:
                warnings.append(
                    f"Absolute excess kurtosis {abs(excess_kurtosis):.2f} "
                    f"exceeds {max_kurtosis}"
                )

    return stats, warnings


def check_data_quality(
    values: Any,
    *,
    max_nan_fraction: float = 0.5,
    max_inf_fraction: float = 0.0,
    max_zero_fraction: float = 0.99,
    outlier_method: str | None = "iqr",
    outlier_threshold: float = 3.0,
    max_outlier_fraction: float = 0.05,
    check_distribution_stats: bool = True,
) -> DataQualityReport:
    """Comprehensive data quality check.

    Parameters
    ----------
    values : array-like
        Data to validate
    max_nan_fraction : float
        Maximum allowed NaN fraction
    max_inf_fraction : float
        Maximum allowed inf fraction
    max_zero_fraction : float
        Maximum allowed zero fraction
    outlier_method : str or None
        Outlier detection method ("iqr", "zscore", "mad", None to skip)
    outlier_threshold : float
        Outlier detection threshold
    max_outlier_fraction : float
        Maximum allowed outlier fraction
    check_distribution_stats : bool
        Whether to compute distribution statistics

    Returns
    -------
    DataQualityReport
        Comprehensive quality report
    """
    arr = np.asarray(values, dtype=float)
    warnings = []

    # Shape
    if arr.ndim == 1:
        n_rows, n_cols = len(arr), 1
    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
    else:
        n_rows, n_cols = arr.size, 1
        arr = arr.ravel()

    # NaN check
    nan_mask = np.isnan(arr)
    nan_fraction = nan_mask.sum() / arr.size
    if nan_fraction > max_nan_fraction:
        warnings.append(f"NaN fraction {nan_fraction:.2%} exceeds {max_nan_fraction:.2%}")

    # Inf check
    inf_mask = np.isinf(arr)
    inf_fraction = inf_mask.sum() / arr.size
    if inf_fraction > max_inf_fraction:
        warnings.append(f"Inf fraction {inf_fraction:.2%} exceeds {max_inf_fraction:.2%}")

    # Zero check
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        finite_vals = arr[finite_mask]
        zero_mask = finite_vals == 0
        zero_fraction = zero_mask.sum() / len(finite_vals)
        if zero_fraction > max_zero_fraction:
            warnings.append(
                f"Zero fraction {zero_fraction:.2%} exceeds {max_zero_fraction:.2%}"
            )
    else:
        zero_fraction = 0.0

    # Outlier check
    outlier_stats = None
    if outlier_method is not None:
        outlier_stats = check_outliers(
            arr, method=outlier_method, threshold=outlier_threshold
        )
        if outlier_stats.outlier_fraction > max_outlier_fraction:
            warnings.append(
                f"Outlier fraction {outlier_stats.outlier_fraction:.2%} "
                f"exceeds {max_outlier_fraction:.2%}"
            )

    # Distribution statistics
    distribution = {}
    if check_distribution_stats:
        dist_stats, dist_warnings = check_distribution(arr)
        distribution = dist_stats
        warnings.extend(dist_warnings)

    passed = len(warnings) == 0

    return DataQualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        nan_fraction=nan_fraction,
        inf_fraction=inf_fraction,
        zero_fraction=zero_fraction,
        outlier_stats=outlier_stats,
        distribution=distribution,
        warnings=warnings,
        passed=passed,
    )
