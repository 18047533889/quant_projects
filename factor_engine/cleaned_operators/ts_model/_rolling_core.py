# -*- coding: utf-8 -*-
"""Shared rolling regression kernels for time-series model operators."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata

_INTEGER_PARAMS = frozenset(
    {"window", "min_periods", "order", "lag", "coefficient_index", "max_q", "q"}
)


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int = 5,
    domain: str = "price_volume",
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_regression",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_regression", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def aligned(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    base = frames[0]
    result = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        result.append(frame)
    return tuple(result)


def ols_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """OLS coefficients, None if design is rank deficient / degenerate."""
    if design.shape[0] < design.shape[1]:
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta


def huber_fit(design: np.ndarray, y: np.ndarray, *, delta: float = 1.345, iterations: int = 5) -> np.ndarray | None:
    beta = ols_fit(design, y)
    if beta is None:
        return None
    for _ in range(iterations):
        resid = y - design @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 0.0:
            scale = float(np.std(resid))
        if scale <= 0.0:
            break
        z = resid / scale
        abs_z = np.abs(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(abs_z <= delta, 1.0, delta / abs_z)
        beta = ols_fit(design * weight[:, None], y * weight)
        if beta is None:
            return None
    return beta


def ridge_fit(design: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray | None:
    """Ridge OLS with L2 penalty on non-intercept coefficients."""
    if design.shape[0] < design.shape[1]:
        return None
    penalty = float(alpha) * np.eye(design.shape[1])
    if design.shape[1] >= 1:
        penalty[0, 0] = 0.0  # do not penalise the intercept
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta


def quantile_fit(design: np.ndarray, y: np.ndarray, q: float, iterations: int = 8) -> np.ndarray | None:
    beta = ols_fit(design, y)
    if beta is None:
        return None
    for _ in range(iterations):
        resid = y - design @ beta
        weight = np.where(resid > 0, q, 1.0 - q)
        weight = np.clip(weight, 1e-6, None)
        beta = ols_fit(design * weight[:, None], y * weight)
        if beta is None:
            return None
    return beta


def build_design(features: list[np.ndarray], add_intercept: bool) -> np.ndarray:
    cols: list[np.ndarray] = []
    if add_intercept:
        cols.append(np.ones(features[0].shape[0], dtype=float))
    cols.extend(features)
    return np.column_stack(cols)


def rolling_xy(
    y: np.ndarray, xs: list[np.ndarray], window: int, min_periods: int
) -> np.ndarray:
    """Return per-row beta / intercept-free helper --- not used directly.

    Kept for symmetry with ``microstructure``; see ``rolling_fit`` below.
    """
    raise NotImplementedError("use rolling_fit")


def rolling_fit(
    y: np.ndarray,
    xs: list[np.ndarray],
    window: int,
    min_periods: int,
    *,
    fit_fn: Any = ols_fit,
    add_intercept: bool = True,
    extra: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling regression over a 1-D series.

    Returns ``(beta, resid, resid_std)`` arrays aligned to ``y``.  ``beta`` has
    shape (len(y), n_coeffs).  Each window uses the last ``window`` rows ending
    at the current row.  If fewer than ``min_periods`` finite rows, NaN.
    """
    n = len(y)
    n_coeffs = len(xs) + (1 if add_intercept else 0)
    beta = np.full((n, n_coeffs), np.nan, dtype=float)
    resid = np.full(n, np.nan, dtype=float)
    resid_std = np.full(n, np.nan, dtype=float)
    if n_coeffs <= 0:
        return beta, resid, resid_std
    w = int(window)
    mp = max(int(min_periods), n_coeffs + 1)
    for row in range(n):
        start = max(0, row - w + 1)
        seg_y = y[start : row + 1]
        seg_xs = [x[start : row + 1] for x in xs]
        valid = np.isfinite(seg_y)
        for x in seg_xs:
            valid &= np.isfinite(x)
        if valid.sum() < mp:
            continue
        vy = seg_y[valid]
        vxs = [x[valid] for x in seg_xs]
        if any(np.std(vx) <= 0.0 for vx in vxs):
            continue
        design = build_design(vxs, add_intercept)
        b = fit_fn(design, vy) if extra is None else fit_fn(design, vy, extra)
        if b is None:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            pred = design @ b
            e = vy - pred
            ddof = max(design.shape[1], 1)
            sd = float(np.sqrt(np.sum(e * e) / max(len(e) - ddof, 1))) if len(e) > ddof else np.nan
        beta[row] = b
        resid_std[row] = sd
        # Residual at the current row uses current x values, only if current
        # y is finite (current x is implicitly finite because row is finite).
        if np.isfinite(seg_y[-1]):
            cur_xs = [x[-1] for x in seg_xs]
            pred_cur = current_prediction(cur_xs, b, add_intercept)
            resid[row] = float(seg_y[-1] - pred_cur)
    return beta, resid, resid_std


def current_prediction(seg_xs: list[float], b: np.ndarray, add_intercept: bool) -> float:
    terms: list[float] = []
    if add_intercept:
        terms.append(1.0)
    terms.extend(seg_xs)
    return float(np.dot(terms, b))
