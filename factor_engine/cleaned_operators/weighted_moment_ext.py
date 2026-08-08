# -*- coding: utf-8 -*-
"""Weighted-moment / conditional-covariance / robust-residual primitives
(2026-08-08 Gemini round).

* ``ts_weighted_standardized_moment`` — general non-negative-weight standardized
  central moment; ``order`` is restricted to {3, 4} (weighted skew/kurtosis) so
  the parameter surface stays fixed.
* ``ts_cov_if`` — conditional rolling covariance, completing the existing
  ``ts_*_if`` conditional family (min/max/quantile/corr/beta/regression_resid).
* ``cs_multi_robust_resid`` — cross-sectional multi-regressor residual using
  standardized design + SVD least squares with a fixed mild ridge (stable under
  collinear exposures; NOT order-dependent recursive Gram-Schmidt).

All operators are strict-PIT, deterministic and NaN fail-closed.  ``weight``
inputs must be non-negative; non-finite weights fail closed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry

_EPS = 1e-12


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        out.append(frame)
    return tuple(out)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# ts_weighted_standardized_moment(x, weight, window, order)
# ---------------------------------------------------------------------------
def _ts_weighted_standardized_moment(
    x: pd.DataFrame,
    weight: pd.DataFrame,
    window: int = 20,
    order: int = 3,
) -> pd.DataFrame:
    x, weight = _align(x, weight)
    w = int(window)
    if w < 2:
        raise ValueError("ts_weighted_standardized_moment requires window >= 2")
    p = int(order)
    if p not in (3, 4):
        raise ValueError("order must be 3 (weighted skew) or 4 (weighted kurtosis)")
    min_samples = 4 if p == 4 else 3
    xv = x.to_numpy(dtype=float)
    wv = weight.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            xs = xv[start : r + 1, c]
            ws = wv[start : r + 1, c]
            # Negative weights are an invalid state (P1-007): a window that
            # contains one is fail-closed, never silently dropped from the mean.
            if np.any(ws < 0.0):
                continue
            valid = np.isfinite(xs) & np.isfinite(ws)
            n = int(valid.sum())
            if n < min_samples:
                continue
            xa = xs[valid].astype(float)
            wa = ws[valid].astype(float)
            sw = float(wa.sum())
            if sw <= _EPS:
                continue
            # Effective sample size N_eff = (Σw)²/Σw²: a window where ~all weight
            # sits on one observation (N_eff ≈ 1) must not masquerade as an
            # n-sample moment estimate.
            n_eff = (sw * sw) / (float(np.sum(wa * wa)) + _EPS)
            if n_eff < float(min_samples) - 1e-9:
                continue
            mu = float((wa * xa).sum() / sw)
            dev = xa - mu
            var_w = float((wa * dev * dev).sum() / sw)
            if var_w <= _EPS:
                continue
            sigma = np.sqrt(var_w)
            m_p = float((wa * dev**p).sum() / sw) / (sigma**p + _EPS)
            out[r, c] = m_p
    return _frame_like(x, out)


# ---------------------------------------------------------------------------
# ts_cov_if(x, y, condition, window, min_periods)
# ---------------------------------------------------------------------------
def _ts_cov_if(
    x: pd.DataFrame,
    y: pd.DataFrame,
    condition: pd.DataFrame,
    window: int = 20,
    min_periods: int = 2,
) -> pd.DataFrame:
    x, y, condition = _align(x, y, condition)
    w = int(window)
    if w < 2:
        raise ValueError("ts_cov_if requires window >= 2")
    mp = max(2, int(min_periods))
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    cv = condition.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            xs = xv[start : r + 1, col]
            ys = yv[start : r + 1, col]
            cs = cv[start : r + 1, col]
            mask = (
                np.isfinite(xs)
                & np.isfinite(ys)
                & np.isfinite(cs)
                & (cs != 0.0)
            )
            xa = xs[mask].astype(float)
            ya = ys[mask].astype(float)
            if xa.size < mp:
                continue
            mx = float(xa.mean())
            my = float(ya.mean())
            if xa.size < 2:
                continue
            cov = float(np.sum((xa - mx) * (ya - my)) / (xa.size - 1.0))
            out[r, col] = cov
    return _frame_like(x, out)


# ---------------------------------------------------------------------------
# cs_multi_robust_resid(y, x1, x2, x3, add_intercept)
# ---------------------------------------------------------------------------
def _cs_multi_robust_resid(
    y: pd.DataFrame,
    x1: pd.DataFrame,
    x2: pd.DataFrame | None = None,
    x3: pd.DataFrame | None = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    features = [x1]
    if x2 is not None:
        features.append(x2)
    if x3 is not None:
        features.append(x3)
    aligned = _align(y, *[f for f in features if f is not None])
    y = aligned[0]
    features = aligned[1:]
    yv = y.to_numpy(dtype=float)
    fvs = [f.to_numpy(dtype=float) for f in features]
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    ridge = 1e-3
    for r in range(rows):
        mask = np.isfinite(yv[r])
        for fv in fvs:
            mask &= np.isfinite(fv[r])
        n = int(mask.sum())
        if n < 2:
            continue
        # Standardize each exposure (mean 0 / std 1).  A constant exposure is
        # DROPPED entirely (it carries no information and only invites rank loss),
        # instead of being kept as a zero column.
        std_cols: list[np.ndarray] = []
        for fv in fvs:
            col = fv[r][mask].astype(float)
            sd = float(np.std(col))
            if sd <= _EPS:
                continue
            std_cols.append((col - float(np.mean(col))) / sd)
        k = len(std_cols) + (1 if add_intercept else 0)
        if n <= k:
            continue
        design = np.column_stack(std_cols) if std_cols else np.empty((n, 0))
        if add_intercept:
            design = np.column_stack((np.ones(n), design))
        # Ridge-normalized SVD/pinv least squares (fixed mild ridge, deterministic).
        # The intercept is NOT penalized (standard ridge practice) so the level of
        # y is absorbed and only the exposure slopes are shrunk.  Collinearity
        # never drops the whole section: pinv of the ridge-augmented normal matrix
        # is always defined (condition number stays a diagnostic, not a gate).
        reg = np.zeros((k, k))
        start = 1 if add_intercept else 0
        for _j in range(start, k):
            reg[_j, _j] = ridge
        target = yv[r][mask]
        xtx = design.T @ design + reg
        try:
            beta = np.linalg.pinv(xtx) @ (design.T @ target)
        except np.linalg.LinAlgError:
            continue
        out[r, mask] = target - design @ beta
    return _frame_like(y, out)


# ---------------------------------------------------------------------------
# Registration (pandas + polars)
# ---------------------------------------------------------------------------
_DAILY_CANONICALS: tuple[str, ...] = (
    "ts_weighted_standardized_moment",
    "ts_cov_if",
    "cs_multi_robust_resid",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_weighted_standardized_moment": _ts_weighted_standardized_moment,
    "ts_cov_if": _ts_cov_if,
    "cs_multi_robust_resid": _cs_multi_robust_resid,
}

_PARAMS: dict[str, list[str]] = {
    "ts_weighted_standardized_moment": ["x", "weight", "window", "order"],
    "ts_cov_if": ["x", "y", "condition", "window", "min_periods"],
    "cs_multi_robust_resid": ["y", "x1", "x2", "x3", "add_intercept"],
}

_CATEGORIES: dict[str, str] = {
    "ts_weighted_standardized_moment": "time_series_risk",
    "ts_cov_if": "time_series_condition",
    "cs_multi_robust_resid": "cross_sectional_regression",
}

# Output unit per operator (P1-27): only ``same_as:target`` for a plain mean.
# A standardized moment is dimensionless, a covariance carries unit(x)*unit(y),
# and a regression residual keeps unit(y) — never ``same_as:target``.
_UNITS: dict[str, str] = {
    "ts_weighted_standardized_moment": "dimensionless",
    "ts_cov_if": "unit(x)*unit(y)",
    "cs_multi_robust_resid": "unit(y)",
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]

        class _PandasOp(PandasOperator):
            metadata = PandasMetadata(
                name=canonical,
                category=category,
                description=canonical,
                examples=[],
                param_names=params,
                return_type="series",
                tags=["daily", "panel", "pit_safe", "causal", "deterministic",
                      f"signature:{','.join(params)}->series",
                      "domain:statistics", f"unit:{_UNITS[canonical]}", "cost:4"],
            )

            def calculate(self, *args, _fn=fn, **kwargs):
                return _fn(*args, **kwargs)

        OperatorRegistry.register(
            _PandasOp(), canonical=canonical, backend="pandas_numpy",
            source="weighted_moment_ext", backend_explicit=True,
        )

        class _PolarsOp(PolarsSeriesOperator):
            metadata = PolarsMetadata(name=canonical, category="weighted_moment_ext", param_names=[])

            def _calculate_series(self, *frames, _fn=fn, **params):
                import polars as pl  # noqa: F401
                pdfs = [f.select([c for c in f.columns if c not in _SKIP]).to_pandas() for f in frames]
                out = _fn(*pdfs, **params)
                base = frames[0]
                cols = [c for c in base.columns if c not in _SKIP]
                return base.with_columns(
                    [pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols]
                )

        OperatorRegistry.register(
            _PolarsOp(), canonical=canonical, backend="polars",
            source="weighted_moment_ext_polars", backend_explicit=True,
        )

    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_DAILY_CANONICALS)
    )


_register()
