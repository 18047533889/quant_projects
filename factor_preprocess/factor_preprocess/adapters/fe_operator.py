"""
FE operator adapter (R61-FI-041, plan §26 F2/F3).

The FactorEngine operator registry is the *stateless-math authority* for
transforms whose math it implements.  For every FP transform that FI-040
classified as an ``exact`` and ``stateless`` FE duplicate, parity is proven in
``tests/test_fe_operator_parity.py`` and the FP registry metadata carries:

    implementation_origin = "FE_OPERATOR"
    fe_operator_id        = <canonical FE operator id>
    fit_kind              = "stateless"

Execution routing
-----------------
``TransformRegistry.get_execution(name)`` asks this module for an executor.
FE is NOT a hard dependency of factor_preprocess (dependency isolation —
factor_engine lives at the monorepo root and is not a declared FP wheel
dependency), so this adapter never imports FE at module scope.  The FE
registry is loaded lazily through the bridge and a ``(canonical_id, kwargs)
-> panel`` executor is built on first use; when FE is unavailable, the
executor is ``None`` and the registry falls back to the retained FP-native
kernel (deprecated fallback — plan §26 F3 migration period; never deleted in
one shot).

Calling convention
------------------
Every current FP transform consumes a LONG-format ``pd.DataFrame``
(``asset_id``/``date``/``value`` columns, optionally sorted per asset) and
returns a ``pd.Series`` aligned to the input index, while FE operators
consume a WIDE panel (``index=timestamp, columns=instrument``).  The adapter
converts: long frame -> per-``value_col`` wide panel (index = sorted unique
``time_col``, columns = sorted unique ``asset_col``), runs the FE operator,
then stacks back to a Series aligned on the input row order.  The set of
value columns is derived from the FE operator's arity (its metadata
``param_names`` minus non-panel params); the first panel column is the
primary signal.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Non-panel (scalar/flag) parameters that must NOT be interpreted as value
#: columns when mapping a long frame onto an FE operator call.
_NON_PANEL_PARAMS = frozenset({
    "x", "y", "window", "min_periods", "min_obs", "ddof", "span", "halflife",
    "alpha", "lower", "upper", "max_gap", "max_periods", "limit", "p", "q",
    "n", "period", "value", "method", "group", "weight", "add_intercept",
    "include", "null_policy", "nan_policy", "includes_current_bar",
    "zero_std_policy", "lineage", "fill_value", "constant", "com", "adjust",
    "target_std", "target_vol", "annualization_factor", "threshold", "axis",
})

#: FE params that are panel-valued (secondary inputs beyond the primary).
_PANEL_PARAMS = frozenset({"exposures", "features", "x1", "x2", "x3"})


def _fe_operator_registry():
    """Lazy-load the FE operator registry (never at module scope).

    Returns ``(registry, ensure_loaded)`` or raises ImportError when FE is not
    importable (e.g. the FP-only wheel environment).
    """
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    return OperatorRegistry


def _long_to_wide(
    values: pd.DataFrame,
    value_col: str,
    time_col: str,
    asset_col: str,
) -> pd.DataFrame:
    """Pivot a long factor frame to the wide (timestamp x instrument) panel.

    Row order is irrelevant for the pivot; the caller re-aligns the stacked
    result back onto the input index afterwards.
    """
    if values[time_col].dtype.kind == "M":
        times = pd.DatetimeIndex(sorted(values[time_col].unique()))
    else:
        times = pd.Index(sorted(values[time_col].unique()))
    assets = pd.Index(sorted(values[asset_col].astype(str).unique()))
    pivot = values.pivot_table(
        index=time_col,
        columns=asset_col,
        values=value_col,
        aggfunc="first",
        sort=False,
    )
    pivot = pivot.reindex(index=times, columns=assets)
    return pivot


def _stack_back(panel: pd.DataFrame, template: pd.DataFrame, time_col, asset_col, value_col) -> pd.Series:
    """Re-align a wide FE result onto the template's long row order."""
    wide = panel
    # Build a lookup from (time, asset-as-str) -> value for every template row.
    lookup = {}
    times = list(wide.index)
    for col in wide.columns:
        series = wide[col]
        for t in times:
            v = series.get(t)
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            lookup[(t, str(col))] = v
    result = np.full(len(template), np.nan, dtype=float)
    t_vals = template[time_col].tolist()
    a_vals = template[asset_col].astype(str).tolist()
    for i in range(len(template)):
        result[i] = lookup.get((t_vals[i], a_vals[i]), np.nan)
    return pd.Series(result, index=template.index, name=value_col)


class FeOperatorExecutor:
    """Callable executor for one FE canonical operator.

    ``fallback`` is the retained FP-native kernel used only when FE is
    unavailable at call time (never deleted — plan §26 F3).
    """

    def __init__(self, canonical: str, fallback: Optional[Callable] = None):
        self.canonical = canonical
        self.fallback = fallback
        self._registry = None
        self._op = None

    def _ensure(self):
        if self._registry is None:
            self._registry = _fe_operator_registry()
        if self._op is None:
            self._op = self._registry.get(self.canonical, backend="pandas_numpy")
            if self._op is None:
                raise KeyError(
                    f"FE operator {self.canonical!r} unavailable on pandas_numpy"
                )

    def __call__(self, *args, **kwargs):
        """Route a long-format FP call through the FE operator."""
        try:
            self._ensure()
        except Exception:
            if self.fallback is not None:
                return self.fallback(*args, **kwargs)
            raise
        values = kwargs.get("values", args[0] if args else None)
        if values is None:
            raise TypeError(
                f"FE-backed {self.canonical}: expected a long-format values "
                "DataFrame as first positional/keyword 'values' argument"
            )
        value_col = kwargs.get("value_col", "value")
        time_col = kwargs.get("time_col", "date")
        asset_col = kwargs.get("asset_col", "asset_id")

        # Extract scalar params to forward (winzoring bounds / OLS flags / etc).
        scalar_kw = {}
        for k in ("lower", "upper", "add_intercept", "min_obs", "ddof",
                  "target_std", "max_gap", "max_periods", "p", "min_periods"):
            if k in kwargs and k not in ("time_col", "asset_col", "value_col", "values"):
                scalar_kw[k] = kwargs[k]
        # expose frame: kwargs['exposures'] is the exposure long frame.
        exposures = kwargs.get("exposures")

        panel = _long_to_wide(values, value_col, time_col, asset_col)
        if exposures is not None:
            # Exposures come as a long frame with the SAME long schema but
            # multiple value columns (e1, e2, ...). Build one panel per
            # exposure column so FE receives exposure panels in column order.
            exp_value_cols = [
                c for c in exposures.columns
                if c not in (time_col, asset_col)
            ]
            exp_panels = [
                _long_to_wide(exposures, c, time_col, asset_col)
                for c in exp_value_cols
            ]
            out = self._op.calculate(panel, *exp_panels, **scalar_kw)
        else:
            out = self._op.calculate(panel, **scalar_kw)
        return _stack_back(out, values, time_col, asset_col, value_col)


def get_fe_executor(canonical: str, fallback: Optional[Callable] = None) -> Optional[FeOperatorExecutor]:
    """Return a lazily-FE-backed executor for ``canonical``.

    Returns ``None`` when FE cannot be imported (so the registry falls back to
    the retained FP-native kernel).  Raises only on an *unexpected* registry
    error; a missing operator surfaces at first call.
    """
    try:
        _fe_operator_registry()  # import + load probe
    except Exception:
        return None
    return FeOperatorExecutor(canonical, fallback=fallback)


__all__ = [
    "FeOperatorExecutor",
    "get_fe_executor",
]
