# -*- coding: utf-8 -*-
"""Native Polars backends for cross-sectional group ex-self and robust
regression-residual operators.

These operate per row (a cross-section across panel columns) using the same
NumPy kernels as the pandas references; results are wrapped into
``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})

# R11 #137: cross-section minimum breadth (mirrors group_ext._MIN_BREADTH /
# _BREADTH_PARAM_RATIO so the pandas and polars twins share the same gate).
_MIN_BREADTH = 10
_BREADTH_PARAM_RATIO = 5.0


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _row_arrays(*frames: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.stack([f[c].to_numpy() for f in frames for c in cols], axis=1) if len(frames) == 1 else None


def group_ex_self_mean(x, group):
    cols = _cols(x, group)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        finite = np.isfinite(xv[t])
        labels = np.unique(gv[t])
        for label in labels:
            idx = (gv[t] == label) & finite
            count = int(np.sum(idx))
            if count <= 1:
                continue
            total = float(np.sum(xv[t][idx]))
            for j in np.flatnonzero(gv[t] == label):
                if not finite[j]:
                    continue
                out[t, j] = np.where((count - 1.0) != 0, (total - xv[t][j]) / (count - 1.0), np.nan)
    return _make(x, cols, out)


def group_ex_self_weighted_mean(x, weight, group):
    cols = _cols(x, weight, group)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    wv = np.stack([weight[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        g_row = gv[t]
        for label in np.unique(g_row):
            idx = g_row == label
            # R5 P1-37(a) parity with the pandas twin: numerator and denominator
            # MUST use the same mask `finite(x) & finite(w) & w >= 0`.  The old
            # polars mask counted missing-x peers in the denominator but silently
            # dropped them from the numerator.
            mask = idx & np.isfinite(wv[t]) & np.isfinite(xv[t]) & (wv[t] >= 0)
            total_w = float(np.sum(wv[t][mask]))
            if not np.isfinite(total_w) or total_w <= 0.0:
                continue
            weighted = float(np.sum(wv[t][mask] * xv[t][mask]))
            for j in np.flatnonzero(idx):
                if not mask[j]:
                    continue
                denom = total_w - wv[t][j]
                if denom <= 0.0:
                    continue
                out[t, j] = (weighted - wv[t][j] * xv[t][j]) / denom if denom != 0 else np.nan
    return _make(x, cols, out)


def _valid_membership_label(label: object) -> bool:
    """True when ``label`` is a concrete group/subgroup membership label.

    Missing/unknown labels (NaN, +/-inf, ``None``, empty string) are NOT real
    memberships: a cell carrying one must keep its residual NaN and must never
    join a composite ``(group, subgroup)`` demeaning key.  This is the same
    membership notion the group kernels use elsewhere in this file (a NaN group
    label never matches ``==`` in ``group_ex_self_mean``; here we make the
    exclusion explicit for every missing sentinel).
    """
    if label is None:
        return False
    if isinstance(label, str):
        return label != ""
    if isinstance(label, bool):
        return True
    try:
        # Numeric labels (incl. numpy scalars): NaN / +/-inf are non-membership.
        return bool(np.isfinite(float(label)))
    except (TypeError, ValueError):
        # Opaque object label (e.g. a non-empty tuple / arbitrary tag): treat as
        # a concrete membership value.
        return True


def _demean_composite_row(values: np.ndarray, group_labels: np.ndarray, subgroup_labels: np.ndarray) -> np.ndarray:
    """Demean by the composite ``(group, subgroup)`` key (R11 #134).

    Mirrors ``group_ext.HierarchicalGroupNeutralize._demean_composite_row``: a
    bare subgroup label that repeats across parent groups must NOT be merged;
    the composite key keeps subgroup means within their own parent group.

    R11 (honesty): a cell whose group OR subgroup label is missing/unknown
    (NaN, +/-inf, ``None``, empty string) must keep its residual NaN.  It must
    never participate in any composite key — a lone unknown-group cell would
    otherwise be manufactured into a perfectly-neutralized 0 (mean == self).
    Only cells with BOTH a valid group label and a valid subgroup label join
    the ``(group, subgroup)`` key.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    n = len(values)
    keys = [None] * n
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if _valid_membership_label(group_labels[i]) and _valid_membership_label(subgroup_labels[i]):
            keys[i] = (group_labels[i], subgroup_labels[i])
            valid[i] = True
    seen: list[tuple] = []
    for i in range(n):
        if not valid[i]:
            continue
        key = keys[i]
        if key not in seen:
            seen.append(key)
    for key in seen:
        idx = np.flatnonzero(
            np.fromiter((valid[i] and keys[i] == key for i in range(n)), dtype=bool, count=n)
        )
        group_values = values[idx]
        finite = group_values[np.isfinite(group_values)]
        if finite.size == 0:
            continue
        mean = float(np.mean(finite))
        out[idx] = values[idx] - mean
    return out


def hierarchical_group_neutralize(x, group, subgroup):
    cols = _cols(x, group, subgroup)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    sv = np.stack([subgroup[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        out[t] = _demean_composite_row(xv[t].copy(), gv[t], sv[t])
    return _make(x, cols, out)


def cs_trimmed_ols_resid(y, x, trim_ratio=0.1, add_intercept=True):
    """Trimmed cross-sectional OLS residual.

    Note (R5 P1-37c): this trims the most extreme ``x`` observations and then
    fits an ordinary least-squares line.  It is *trimmed OLS*, not a true robust
    regression (no Huber / LAD / Theil-Sen weighting).  The canonical name is
    ``cs_trimmed_ols_resid`` (R11 honesty); ``cs_robust_resid`` is kept as a
    deprecated alias for backward compatibility with the pandas twin
    (``group_ext.CsRobustResid``).
    R11 #137: a cross-section minimum-breadth gate matches the pandas twin —
    3 stocks must not drive the regression.
    """
    trim = _pf(trim_ratio, "trim_ratio")
    if not (0.0 <= trim < 0.5):
        raise ValueError("cs_trimmed_ols_resid requires 0 <= trim_ratio < 0.5")
    cols = _cols(y, x)
    rows = y.height
    yv = np.stack([y[c].to_numpy() for c in cols], axis=1)
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    # R11 #137: cross-section minimum breadth (mirror of group_ext).
    min_breadth = max(_MIN_BREADTH, int(_BREADTH_PARAM_RATIO * (2 if add_intercept else 1)))
    for t in range(rows):
        valid = np.isfinite(xv[t]) & np.isfinite(yv[t])
        if valid.sum() < min_breadth:
            continue
        xs = xv[t][valid]
        ys = yv[t][valid]
        cut = int(np.floor(trim * xs.size))
        if cut > 0:
            keep = np.ones(xs.size, dtype=bool)
            keep[np.argsort(xs)[:cut]] = False
            keep[np.argsort(xs)[-cut:]] = False
            xs = xs[keep]
            ys = ys[keep]
        if xs.size < (2 if add_intercept else 1) + 1 or np.std(xs) == 0:
            continue
        if add_intercept:
            coeffs = np.polyfit(xs, ys, 1)
            fitted = np.polyval(coeffs, xv[t])
        else:
            slope = np.where(np.sum(xs * xs)) != 0, float(np.sum(xs * ys) / np.sum(xs * xs)), np.nan)
            fitted = slope * xv[t]
        out[t] = yv[t] - fitted
    return _make(y, cols, out)


_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("group_ex_self_mean", ("x", "group"), group_ex_self_mean, "Group mean excluding self."),
    ("group_ex_self_weighted_mean", ("x", "weight", "group"), group_ex_self_weighted_mean, "Group weighted mean excluding self."),
    ("hierarchical_group_neutralize", ("x", "group", "subgroup"), hierarchical_group_neutralize, "Subgroup then group demean."),
    # R11 honesty: the implementation is genuinely trimmed-OLS (trim extreme x,
    # then ordinary least squares), NOT a robust regression.  The canonical name
    # is cs_trimmed_ols_resid; cs_robust_resid remains a deprecated alias.
    ("cs_trimmed_ols_resid", ("y", "x", "trim_ratio", "add_intercept"), cs_trimmed_ols_resid, "Trimmed-OLS cross-sectional residual (canonical cs_trimmed_ols_resid; cs_robust_resid is a deprecated alias — not a true robust regression)."),
)

# Backward-compatible module-level name for code that imported the function by
# its historical spelling.  The canonical registry name is cs_trimmed_ols_resid.
cs_robust_resid = cs_trimmed_ols_resid


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="group_neutralization",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsGroup_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="group_neutralization",
        business_category="group_neutralization",
        canonical=name,
        source="polars_group",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)


def _register_trimmed_ols_honest_rename() -> None:
    """R11: keep the historical ``cs_robust_resid`` resolvable as a deprecated
    alias of the honest canonical ``cs_trimmed_ols_resid``.

    The implementation is genuinely trimmed-OLS (trim extreme ``x``, then
    ordinary least squares) — not a robust regression (no Huber/LAD/Theil-Sen) —
    so the canonical name must be honest.  ``rename_canonical``/``register_alias``
    preserve the old DSL name.  The static surface partition is updated in sync
    (new canonical -> extended; retired spelling -> removed) so
    ``layer_governance``'s exact-classification check still holds at finalize.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "cs_robust_resid" in OperatorRegistry._operators:
        # A backend registered the historical spelling directly; migrate it.
        OperatorRegistry.rename_canonical("cs_robust_resid", "cs_trimmed_ols_resid")
    elif "cs_trimmed_ols_resid" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("cs_robust_resid", "cs_trimmed_ols_resid")
    try:
        from cleaned_operators.operator_surface import extend_extended_only, retract_extended_only

        extend_extended_only(["cs_trimmed_ols_resid"])
        retract_extended_only(["cs_robust_resid"])
    except ImportError:  # pragma: no cover - surface module always present in-tree
        pass


_register_trimmed_ols_honest_rename()
