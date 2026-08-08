# -*- coding: utf-8 -*-
"""Shared rolling regression kernels for time-series model operators."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata

try:  # HiGHS LP for true pinball-loss quantile regression.
    from scipy.optimize import linprog as _linprog
except Exception:  # pragma: no cover
    _linprog = None

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
    input_units: dict[str, str] | None = None,
    output_unit: str | None = None,
) -> OperatorMetadata:
    # P1-89: the typed-v2 surface carries explicit input_units / output_unit
    # field semantics.  Regression kernels that previously wrote a generic
    # ``unit`` tag now also declare the concrete unit relationships (e.g. beta
    # -> unit(y)/unit(x), residual -> unit(y)); ``unit`` remains in the tags for
    # backward compatibility with legacy consumers.
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
        input_units=dict(input_units) if input_units else {},
        output_unit=output_unit,
    )


def frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def trailing_contiguous_finite(vals: np.ndarray) -> np.ndarray:
    """Most recent suffix of consecutive finite values ending at the last row.

    A missing observation *breaks* the segment: rows on either side of a NaN
    are never treated as adjacent observations (no time-axis compression for
    suspensions / provider gaps).  A NaN at the last row yields an empty block
    so the caller emits NaN instead of re-using the last valid value.
    """
    n = len(vals)
    if n == 0 or not np.isfinite(vals[-1]):
        return np.empty(0, dtype=float)
    i = n - 1
    while i >= 0 and np.isfinite(vals[i]):
        i -= 1
    return vals[i + 1 :]


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


def huber_fit(
    design: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = 1.345,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray | None:
    """Huber M-estimator via IRLS.

    Standard weighted least squares multiplies design and y by ``sqrt(weight)``;
    multiplying by ``weight`` itself would square the influence weights.  The
    iteration is monotone; if it does not converge within ``iterations`` steps
    the fit is treated as degenerate and ``None`` is returned (callers then emit
    NaN rather than a half-converged coefficient).
    """
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
        sqrt_w = np.sqrt(weight)
        beta_new = ols_fit(design * sqrt_w[:, None], y * sqrt_w)
        if beta_new is None:
            return None
        if np.max(np.abs(beta_new - beta)) < tolerance:
            return beta_new
        beta = beta_new
    return beta


def ridge_fit(design: np.ndarray, y: np.ndarray, alpha: float, *, has_intercept: bool = True) -> np.ndarray | None:
    """Ridge OLS with L2 penalty on non-intercept coefficients.

    The intercept column is only exempt from the penalty when the design
    actually starts with an intercept (``has_intercept=True``).  Penalising the
    first *feature* column when no intercept is present would wrongly shrink a
    real regressor.
    """
    if design.shape[0] < design.shape[1]:
        return None
    penalty = float(alpha) * np.eye(design.shape[1])
    if has_intercept and design.shape[1] >= 1:
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
    """IRLS asymmetric-weighted least squares.

    This minimises an asymmetric *squared*-error objective, i.e. it is an
    **expectile** regression (Neyman--Pearson asymmetric loss), not a
    pinball-loss quantile regression.  It is kept under the historic
    ``quantile_*`` operator names for backward compatibility, with the honest
    ``expectile_*`` names exposed alongside; both must be read as expectiles.
    """
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


# The IRLS asymmetric-weighted fit above is an expectile fit; expose an
# explicitly-named alias so factor authors can call the honest name.
expectile_fit = quantile_fit


def pinball_quantile_fit(design: np.ndarray, y: np.ndarray, q: float) -> np.ndarray | None:
    """True pinball-loss quantile regression (Koenker & Bassett, 1978) via LP.

    Minimises ``q * sum(u) + (1-q) * sum(v)`` subject to
    ``X beta + u - v = y, u >= 0, v >= 0`` — the classic LP form.  This is the
    *quantile* (not the expectile): the loss is linear in the residual, so the
    solution is a conditional quantile line.  The LP solver (HiGHS) is
    deterministic for a given input, keeping the operator reproducible.
    Returns ``None`` when the LP is infeasible / unbounded or the sample is
    too small, so the caller emits NaN rather than a spurious coefficient.
    """
    if _linprog is None:
        return None
    n, p = design.shape
    if n < p + 2:
        return None
    c = np.concatenate([np.zeros(p), q * np.ones(n), (1.0 - q) * np.ones(n)])
    a_eq = np.column_stack([design, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    try:
        res = _linprog(c, A_eq=a_eq, b_eq=y.astype(float), bounds=bounds, method="highs")
    except Exception:  # pragma: no cover
        return None
    if not getattr(res, "success", False):
        return None
    beta = np.asarray(res.x[:p], dtype=float)
    if not np.all(np.isfinite(beta)):
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
    fit_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling regression over a 1-D series.

    Returns ``(beta, resid, resid_std)`` arrays aligned to ``y``.  ``beta`` has
    shape (len(y), n_coeffs).  Each window uses the last ``window`` rows ending
    at ``row - fit_lag`` (``fit_lag=0`` is the legacy in-sample fit whose
    training set includes the current row; ``fit_lag>=1`` fits only on rows
    strictly before the current one and reports the out-of-sample residual at
    the current row).  If fewer than ``min_periods`` finite rows, NaN.
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
    lag = max(0, int(fit_lag))
    for row in range(n):
        fit_end = row - lag
        if fit_end < 0:
            continue
        start = max(0, fit_end - w + 1)
        seg_y = y[start : fit_end + 1]
        seg_xs = [x[start : fit_end + 1] for x in xs]
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
        if np.isfinite(y[row]):
            cur_xs = [x[row] for x in xs]
            pred_cur = current_prediction(cur_xs, b, add_intercept)
            resid[row] = float(y[row] - pred_cur)
    return beta, resid, resid_std


def current_prediction(seg_xs: list[float], b: np.ndarray, add_intercept: bool) -> float:
    terms: list[float] = []
    if add_intercept:
        terms.append(1.0)
    terms.extend(seg_xs)
    return float(np.dot(terms, b))
