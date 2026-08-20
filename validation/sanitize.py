"""Input sanitization and normalization for factor computation.

Provides defensive data cleaning:
- Numeric array sanitization (inf/nan handling)
- DataFrame sanitization (column types, index validation)
- Factor input normalization (clipping, winsorization)
- Safe type coercion with validation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


__all__ = [
    "SanitizationResult",
    "sanitize_numeric_array",
    "sanitize_dataframe",
    "sanitize_factor_inputs",
]


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitization operation."""

    sanitized: Any
    n_changes: int
    changes: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Sanitization: {self.n_changes} changes"]
        for name, count in self.changes.items():
            lines.append(f"  {name}: {count}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  Warning: {w}")
        return "\n".join(lines)


def sanitize_numeric_array(
    values: Any,
    *,
    replace_inf: float | str = "nan",
    replace_nan: float | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    winsorize_quantiles: tuple[float, float] | None = None,
    force_finite: bool = False,
) -> SanitizationResult:
    """Sanitize numeric array with configurable policies.

    Parameters
    ----------
    values : array-like
        Input data to sanitize
    replace_inf : float or "nan"
        Value to replace inf with (or "nan" for NaN)
    replace_nan : float or None
        Value to replace NaN with (None to keep NaN)
    clip_min : float or None
        Minimum value for clipping
    clip_max : float or None
        Maximum value for clipping
    winsorize_quantiles : tuple[float, float] or None
        Quantiles for winsorization (e.g., (0.01, 0.99))
    force_finite : bool
        If True, raise error if any non-finite values remain

    Returns
    -------
    SanitizationResult
        Sanitized array and change statistics
    """
    arr = np.asarray(values, dtype=float).copy()
    changes = {}
    warnings = []
    n_changes = 0

    # Replace inf
    inf_mask = np.isinf(arr)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        if replace_inf == "nan":
            arr[inf_mask] = np.nan
        else:
            arr[inf_mask] = float(replace_inf)
        changes["inf_replaced"] = n_inf
        n_changes += n_inf

    # Replace NaN
    if replace_nan is not None:
        nan_mask = np.isnan(arr)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            arr[nan_mask] = replace_nan
            changes["nan_replaced"] = n_nan
            n_changes += n_nan

    # Winsorization (before clipping)
    if winsorize_quantiles is not None:
        lower_q, upper_q = winsorize_quantiles
        if not (0 <= lower_q < upper_q <= 1):
            raise ValueError("winsorize_quantiles must be (lower, upper) in [0, 1]")

        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            finite_vals = arr[finite_mask]
            lower_val = np.percentile(finite_vals, lower_q * 100)
            upper_val = np.percentile(finite_vals, upper_q * 100)

            lower_clip_mask = (arr < lower_val) & finite_mask
            upper_clip_mask = (arr > upper_val) & finite_mask
            n_winsorized = lower_clip_mask.sum() + upper_clip_mask.sum()

            if n_winsorized > 0:
                arr[lower_clip_mask] = lower_val
                arr[upper_clip_mask] = upper_val
                changes["winsorized"] = n_winsorized
                n_changes += n_winsorized

    # Clipping
    if clip_min is not None or clip_max is not None:
        before = arr.copy()
        arr = np.clip(arr, clip_min, clip_max)
        n_clipped = (arr != before).sum()
        if n_clipped > 0:
            changes["clipped"] = n_clipped
            n_changes += n_clipped

    # Force finite check
    if force_finite and not np.isfinite(arr).all():
        n_nonfinite = (~np.isfinite(arr)).sum()
        warnings.append(f"force_finite=True but {n_nonfinite} non-finite values remain")

    return SanitizationResult(
        sanitized=arr,
        n_changes=n_changes,
        changes=changes,
        warnings=warnings,
    )


def sanitize_dataframe(
    df: Any,
    *,
    numeric_cols: list[str] | None = None,
    replace_inf: float | str = "nan",
    replace_nan: float | None = None,
    check_sorted_index: bool = True,
    check_duplicated_index: bool = True,
    coerce_dtypes: bool = False,
) -> SanitizationResult:
    """Sanitize pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sanitize
    numeric_cols : list[str] or None
        Columns to sanitize (None for all numeric columns)
    replace_inf : float or "nan"
        Value to replace inf with
    replace_nan : float or None
        Value to replace NaN with
    check_sorted_index : bool
        Check that index is sorted
    check_duplicated_index : bool
        Check for duplicated index values
    coerce_dtypes : bool
        Attempt to coerce object columns to numeric

    Returns
    -------
    SanitizationResult
        Sanitized DataFrame and change statistics
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for sanitize_dataframe")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected DataFrame, got {type(df)}")

    df_clean = df.copy()
    changes = {}
    warnings = []
    n_changes = 0

    # Index validation
    if check_sorted_index:
        if not df_clean.index.is_monotonic_increasing:
            warnings.append("Index is not sorted")

    if check_duplicated_index:
        n_dup = df_clean.index.duplicated().sum()
        if n_dup > 0:
            warnings.append(f"Index has {n_dup} duplicated values")

    # Determine numeric columns
    if numeric_cols is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # Coerce dtypes
    if coerce_dtypes:
        for col in df_clean.columns:
            if col not in numeric_cols and df_clean[col].dtype == object:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                    n_coerced = df_clean[col].notna().sum()
                    if n_coerced > 0:
                        changes[f"coerced_{col}"] = n_coerced
                        n_changes += n_coerced
                        numeric_cols.append(col)
                except Exception:
                    pass

    # Sanitize numeric columns
    for col in numeric_cols:
        if col not in df_clean.columns:
            warnings.append(f"Column {col!r} not found")
            continue

        col_data = df_clean[col].values
        result = sanitize_numeric_array(
            col_data,
            replace_inf=replace_inf,
            replace_nan=replace_nan,
        )

        if result.n_changes > 0:
            df_clean[col] = result.sanitized
            for change_type, count in result.changes.items():
                key = f"{col}_{change_type}"
                changes[key] = count
                n_changes += count

        warnings.extend(result.warnings)

    return SanitizationResult(
        sanitized=df_clean,
        n_changes=n_changes,
        changes=changes,
        warnings=warnings,
    )


def sanitize_factor_inputs(
    values: Any,
    *,
    strict: bool = False,
    max_abs_value: float = 1e15,
    replace_inf: str = "nan",
    winsorize: tuple[float, float] | None = None,
    normalize: bool = False,
    normalize_method: str = "zscore",
) -> SanitizationResult:
    """Sanitize factor inputs with quantitative finance policies.

    Parameters
    ----------
    values : array-like
        Factor input data
    strict : bool
        Strict mode: force finite values
    max_abs_value : float
        Maximum absolute value (clips beyond this)
    replace_inf : str
        How to handle inf: "nan", "clip", "zero"
    winsorize : tuple[float, float] or None
        Winsorization quantiles (e.g., (0.01, 0.99))
    normalize : bool
        Whether to normalize after sanitization
    normalize_method : str
        Normalization method: "zscore", "minmax", "robust"

    Returns
    -------
    SanitizationResult
        Sanitized factor inputs
    """
    arr = np.asarray(values, dtype=float)
    changes = {}
    warnings = []
    n_changes = 0

    # Replace inf
    inf_mask = np.isinf(arr)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        if replace_inf == "nan":
            arr = arr.copy()
            arr[inf_mask] = np.nan
        elif replace_inf == "clip":
            arr = arr.copy()
            arr[inf_mask & (arr > 0)] = max_abs_value
            arr[inf_mask & (arr < 0)] = -max_abs_value
        elif replace_inf == "zero":
            arr = arr.copy()
            arr[inf_mask] = 0.0
        else:
            raise ValueError(f"unknown replace_inf policy {replace_inf!r}")
        changes["inf_replaced"] = n_inf
        n_changes += n_inf

    # Clip extreme values
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        extreme_mask = (np.abs(arr) > max_abs_value) & finite_mask
        n_extreme = extreme_mask.sum()
        if n_extreme > 0:
            arr = arr.copy()
            arr[extreme_mask] = np.clip(arr[extreme_mask], -max_abs_value, max_abs_value)
            changes["extreme_clipped"] = n_extreme
            n_changes += n_extreme

    # Winsorization
    if winsorize is not None:
        result = sanitize_numeric_array(
            arr, winsorize_quantiles=winsorize, force_finite=False
        )
        arr = result.sanitized
        if result.n_changes > 0:
            changes.update(result.changes)
            n_changes += result.n_changes

    # Normalization
    if normalize:
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            finite_vals = arr[finite_mask]

            if normalize_method == "zscore":
                mean = np.mean(finite_vals)
                std = np.std(finite_vals)
                if std > 0:
                    arr = arr.copy()
                    arr[finite_mask] = (finite_vals - mean) / std
                    changes["normalized_zscore"] = finite_mask.sum()
                else:
                    warnings.append("Cannot zscore normalize: std=0")

            elif normalize_method == "minmax":
                vmin = np.min(finite_vals)
                vmax = np.max(finite_vals)
                if vmax > vmin:
                    arr = arr.copy()
                    arr[finite_mask] = (finite_vals - vmin) / (vmax - vmin)
                    changes["normalized_minmax"] = finite_mask.sum()
                else:
                    warnings.append("Cannot minmax normalize: range=0")

            elif normalize_method == "robust":
                # Robust scaling using median and IQR
                median = np.median(finite_vals)
                q25, q75 = np.percentile(finite_vals, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    arr = arr.copy()
                    arr[finite_mask] = (finite_vals - median) / iqr
                    changes["normalized_robust"] = finite_mask.sum()
                else:
                    warnings.append("Cannot robust normalize: IQR=0")

            else:
                raise ValueError(f"unknown normalize_method {normalize_method!r}")

    # Strict mode check
    if strict:
        n_nonfinite = (~np.isfinite(arr)).sum()
        if n_nonfinite > 0:
            raise ValueError(
                f"strict=True but {n_nonfinite} non-finite values remain after sanitization"
            )

    return SanitizationResult(
        sanitized=arr,
        n_changes=n_changes,
        changes=changes,
        warnings=warnings,
    )
