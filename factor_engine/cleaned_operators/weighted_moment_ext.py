# -*- coding: utf-8 -*-
"""Weighted-moment / conditional-covariance / robust-residual primitives
(2026-08-08 Gemini round).

* ``ts_weighted_standardized_moment`` — general non-negative-weight standardized
  central moment; ``order`` is restricted to {3, 4} (weighted skew/kurtosis) so
  the parameter surface stays fixed.
* ``ts_cov_if`` — conditional rolling covariance, completing the existing
  ``ts_*_if`` conditional family (min/max/quantile/corr/beta/regression_resid).
* ``cs_multi_ridge_resid`` — cross-sectional multi-regressor residual via
  standardized ridge least squares (fixed ridge=1e-3, versioned implementation
  constant); ``cs_multi_robust_resid`` is a deprecated alias.

All operators are strict-PIT, deterministic and NaN fail-closed.  ``weight``
inputs must be non-negative; non-finite weights fail closed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamSpec
from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from factor_engine.cleaned_operators.registry import OperatorRegistry

_EPS = 1e-12

# R11 #140: the ridge penalty is a fixed, VERSIONED implementation constant —
# deliberately NOT a searchable parameter.  It is excluded from param_names /
# param_specs so the search grammar cannot sweep it; the value is part of the
# operator's versioned implementation contract (documented in the metadata
# description).  Changing it is a semantic change, not a knob to tune.
_RIDGE = 1e-3
# R11 #141: residual regression DOF margin.  ``N == K+1`` leaves zero residual
# DOF (the fit is an interpolation).  Require at least _MIN_DOF_MARGIN_N
# observations AND at least _DOF_MARGIN_RATIO observations per estimated
# parameter; below the gate the operator fails closed (NaN).
_MIN_DOF_MARGIN_N = 20
_DOF_MARGIN_RATIO = 5.0


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            # R11 #138: silent reindex is inconsistent with sibling modules.  A
            # genuinely misaligned panel (shifted index / different instrument
            # columns) pairs x_t with score_{t'} or target with a shifted peer —
            # a silent cross-sectional/PIT corruption.  Fail loudly instead of
            # reindexing.
            raise ValueError(
                "multi-panel inputs must share identical index/columns; "
                "reindexing misaligned panels is not allowed"
            )
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
            # A NaN/Inf weight invalidates the WHOLE window (P1-03) — the window's
            # weight state is unknown, so pairwise omission would silently re-weight
            # the moment.  This matches the module contract ("non-finite weights
            # fail closed"); only non-finite x observations are skipped as missing
            # data (finite-observation trailing window).
            if np.any(~np.isfinite(ws)):
                continue
            valid = np.isfinite(xs) & (ws > 0.0)
            n = int(valid.sum())
            if n < min_samples:
                continue
            xa = xs[valid].astype(float)
            wa = ws[valid].astype(float)
            # Positive rescaling of weights cannot change support or moments.
            wa /= float(np.max(wa))
            wa /= float(wa.sum())
            # Effective sample size N_eff = (Σw)²/Σw²: a window where ~all weight
            # sits on one observation (N_eff ≈ 1) must not masquerade as an
            # n-sample moment estimate.
            n_eff = 1.0 / float(np.dot(wa, wa))
            if n_eff < float(min_samples) - 1e-9:
                continue
            # Prefer subtraction before scaling to retain representable small
            # differences around a large common offset. Normalize first only
            # when opposite-sign extremes would overflow the subtraction.
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = xa - xa[0]
            if not np.all(np.isfinite(normalized)):
                normalized = xa / float(np.max(np.abs(xa)))
                normalized -= normalized[0]
            magnitude = float(np.max(np.abs(normalized)))
            if magnitude == 0.0:
                continue
            normalized /= magnitude
            dev = normalized - float(np.dot(wa, normalized))
            spread = float(np.max(np.abs(dev)))
            if spread == 0.0:
                continue
            dev /= spread
            var_w = float(np.dot(wa, dev * dev))
            if var_w <= 0.0:
                continue
            m_p = float(np.dot(wa, dev**p)) / var_w**(p / 2.0)
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
    # R11 #144: condition must be a ConditionBool ({0, 1} / NaN missing), never
    # an arbitrary non-zero numeric.
    _assert_condition_bool(condition, name="condition")
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
                & (cs == 1.0)
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
# cs_multi_ridge_resid(y, x1, x2, x3, add_intercept)
# ---------------------------------------------------------------------------
def _assert_condition_bool(condition: pd.DataFrame, name: str = "condition") -> None:
    """R11 #144: the condition input must be a ConditionBool.

    Accepted values: {0, 1} (or boolean True/False); NaN = missing and is
    excluded from the selection.  Any other finite numeric value (e.g. 5.0,
    -3.0) is neither a probability nor a boolean — silently treating it as
    "truthy" is a hidden semantic the operator contract forbids.  Fail the
    call (raise) instead of guessing.
    """
    cv = condition.to_numpy()
    finite = np.isfinite(cv)
    bad = finite & (cv != 0.0) & (cv != 1.0)
    if np.any(bad):
        raise ValueError(
            f"{name} must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )


def _cs_multi_ridge_resid(
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
    ridge = _RIDGE  # versioned implementation constant (R11 #140)
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
        # R11 #141: residual regression DOF margin.  N == K+1 leaves zero
        # residual DOF (interpolation); require a real margin so the residuals
        # describe noise, not in-sample fit.  Fail closed below the gate.
        if n < max(_MIN_DOF_MARGIN_N, int(_DOF_MARGIN_RATIO * k)):
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
    "cs_multi_ridge_resid",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_weighted_standardized_moment": _ts_weighted_standardized_moment,
    "ts_cov_if": _ts_cov_if,
    "cs_multi_ridge_resid": _cs_multi_ridge_resid,
}

_PARAMS: dict[str, list[str]] = {
    "ts_weighted_standardized_moment": ["x", "weight", "window", "order"],
    "ts_cov_if": ["x", "y", "condition", "window", "min_periods"],
    "cs_multi_ridge_resid": ["y", "x1", "x2", "x3", "add_intercept"],
}

_CATEGORIES: dict[str, str] = {
    "ts_weighted_standardized_moment": "time_series_risk",
    "ts_cov_if": "time_series_condition",
    "cs_multi_ridge_resid": "cross_sectional_regression",
}

# R11 #139/#140: honest canonical naming.  ``cs_multi_ridge_resid`` is a
# STANDARDIZED RIDGE regression (not robust regression); the old
# ``cs_multi_robust_resid`` name is a deprecated alias and the fixed ridge=1e-3
# is a versioned implementation constant, NOT a searchable parameter.
_DESCRIPTIONS: dict[str, str] = {
    "ts_weighted_standardized_moment": "weighted standardized central moment (order 3=skew / 4=kurtosis)",
    "ts_cov_if": "conditional rolling covariance (condition is a ConditionBool)",
    "cs_multi_ridge_resid": (
        "cross-sectional multi-regressor residual via standardized ridge least "
        "squares (fixed ridge=1e-3, versioned implementation constant, NOT a "
        "searchable parameter); cs_multi_robust_resid is a deprecated alias"
    ),
}

# Output unit per operator (P1-27): only ``same_as:target`` for a plain mean.
# A standardized moment is dimensionless, a covariance carries unit(x)*unit(y),
# and a regression residual keeps unit(y) — never ``same_as:target``.
_UNITS: dict[str, str] = {
    "ts_weighted_standardized_moment": "dimensionless",
    "ts_cov_if": "unit(x)*unit(y)",
    "cs_multi_ridge_resid": "unit(y)",
}

# R4-97: EnumSpec declarations.  ``order`` is a genuine enum (weighted skew=3 /
# weighted kurtosis=4) and ``add_intercept`` a bool switch; declaring choices
# makes ``validate_operator_call`` reject out-of-set values instead of silently
# accepting them (the kernels also raise, but the central validator now reports
# the contract error uniformly at dispatch time).
_PARAM_SPECS: dict[str, dict[str, ParamSpec]] = {
    "ts_weighted_standardized_moment": {
        "order": ParamSpec(dtype=int, choices=(3, 4)),
    },
    "ts_cov_if": {},
    "cs_multi_ridge_resid": {
        "add_intercept": ParamSpec(dtype=bool, choices=(True, False)),
    },
}

# R4-95: trailing-window semantics.  Both rolling operators estimate from the
# finite observations inside the trailing window (gaps are skipped, negative
# weights/zero-condition invalidate); they are not contiguous-window kernels.
_WINDOW_SEMANTICS: dict[str, str] = {
    "ts_weighted_standardized_moment": "finite_observations",
    "ts_cov_if": "finite_observations",
    "cs_multi_ridge_resid": None,
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from factor_engine.cleaned_operators.base import Operator as PandasOperator
    from factor_engine.cleaned_operators.base import OperatorMetadata as PandasMetadata
    from factor_engine.cleaned_operators.base import validate_operator_call

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]

        class _PandasOp(PandasOperator):
            # R11 #139/#142: propagate algebraic output units to the metadata
            # so the catalog / typed search see the real dimension.
            _ou = _UNITS[canonical]
            output_unit = _ou if (_ou.startswith("same_as:") or _ou.startswith("unit(") or _ou == "dimensionless") else None
            metadata = PandasMetadata(
                name=canonical,
                category=category,
                description=_DESCRIPTIONS[canonical],
                examples=[],
                param_names=params,
                return_type="series",
                tags=["daily", "panel", "pit_safe", "causal", "deterministic",
                      f"signature:{','.join(params)}->series",
                      "domain:statistics", f"unit:{_UNITS[canonical]}", "cost:4"],
                param_specs=_PARAM_SPECS[canonical],
                window_semantics=_WINDOW_SEMANTICS[canonical],
                output_unit=output_unit,
            )

            _HANDLES_CALL_CONTRACT = True  # R5-02: routes through validate_operator_call

            def calculate(self, *args, _fn=fn, **kwargs):
                # R4-02: route the direct-``calculate`` kernel through the central
                # logical-call validator (integer / panel-axis / param checks).
                processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
                return _fn(*processed_args, **processed_kwargs)

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

    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_DAILY_CANONICALS))


_register()


def _register_ridge_deprecated_alias() -> None:
    """R11 #139/#140: canonical-honesty rename.

    The operator is a STANDARDIZED RIDGE regression (fixed ridge=1e-3), not a
    robust regression, so the canonical is ``cs_multi_ridge_resid``.  The old
    ``cs_multi_robust_resid`` name stays resolvable as a DEPRECATED alias to the
    same implementation.  Guarded so a double-load / concurrent re-import cannot
    raise (if the alias already resolves to the canonical it is a no-op).
    """
    if OperatorRegistry.resolve_canonical("cs_multi_robust_resid") == "cs_multi_ridge_resid":
        return
    try:
        OperatorRegistry.register_alias("cs_multi_robust_resid", "cs_multi_ridge_resid")
    except (KeyError, ValueError):
        # A concurrent session may have registered the alias already; only an
        # unexpected target is an error worth propagating.
        if OperatorRegistry.resolve_canonical("cs_multi_robust_resid") != "cs_multi_ridge_resid":
            raise


_register_ridge_deprecated_alias()
