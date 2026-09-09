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
import inspect

import numpy as np
import pandas as pd
import inspect

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
    required = {time_col, asset_col, value_col}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"long factor frame missing columns: {sorted(missing)}")
    if values[[time_col, asset_col]].isna().any().any():
        raise ValueError("time and asset identity columns cannot contain nulls")
    duplicate = values.duplicated([time_col, asset_col], keep=False)
    if duplicate.any():
        keys = values.loc[duplicate, [time_col, asset_col]].drop_duplicates()
        raise ValueError(
            "conflicting duplicate (time, asset) identities are ambiguous: "
            f"{list(keys.itertuples(index=False, name=None))[:5]}"
        )
    if values[time_col].dtype.kind == "M":
        times = pd.DatetimeIndex(sorted(values[time_col].unique()))
    else:
        times = pd.Index(sorted(values[time_col].unique()))
    assets = pd.Index(sorted(values[asset_col].unique()))
    pivot = values.pivot(
        index=time_col,
        columns=asset_col,
        values=value_col,
    )
    pivot = pivot.reindex(index=times, columns=assets)
    return pivot


def _stack_back(panel: pd.DataFrame, template: pd.DataFrame, time_col, asset_col, value_col) -> pd.Series:
    """Re-align a wide FE result onto the template's long row order."""
    wide = panel
    # Preserve identity types: only the security catalog may equate 1 and "1".
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
            lookup[(t, col)] = v
    result = np.full(len(template), np.nan, dtype=float)
    t_vals = template[time_col].tolist()
    a_vals = template[asset_col].tolist()
    for i in range(len(template)):
        result[i] = lookup.get((t_vals[i], a_vals[i]), np.nan)
    return pd.Series(result, index=template.index, name=value_col)


class FeOperatorExecutor:
    """Callable executor for one FE canonical operator.

    ``fallback`` is the retained FP-native kernel used only when FE is
    unavailable at call time (never deleted — plan §26 F3).
    """

    _PARAM_ALIASES = {"max_lag": "max_periods"}

    def __init__(self, canonical: str, fallback: Optional[Callable] = None, *, allow_research_fallback: bool = False):
        self.canonical = canonical
        self.fallback = fallback
        self._registry = None
        self._op = None
        self.allow_research_fallback = allow_research_fallback
        self.effective_parameters: Dict[str, Any] = {}

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
        except (ImportError, ModuleNotFoundError):
            if self.fallback is not None and self.allow_research_fallback:
                return self.fallback(*args, **kwargs)
            raise
        call_args = dict(kwargs)
        if args:
            if self.fallback is None:
                if len(args) > 1:
                    raise TypeError("only values may be positional without a declared FP signature")
                positional = {"values": args[0]}
            else:
                positional = dict(inspect.signature(self.fallback).bind_partial(*args).arguments)
            overlap = set(positional).intersection(call_args)
            if overlap:
                raise TypeError(f"parameters supplied positionally and by keyword: {sorted(overlap)}")
            call_args.update(positional)
        if "date_col" in call_args:
            if "time_col" in call_args:
                raise TypeError("date_col and time_col cannot both be supplied")
            call_args["time_col"] = call_args.pop("date_col")
        values = call_args.get("values")
        if values is None:
            raise TypeError(
                f"FE-backed {self.canonical}: expected a long-format values "
                "DataFrame as first positional/keyword 'values' argument"
            )
        value_col = call_args.get("value_col", "value")
        time_col = call_args.get("time_col", "date")
        asset_col = call_args.get("asset_col", "asset_id")

        # Extract scalar params to forward (winzoring bounds / OLS flags / etc).
        adapter_only = {"values", "time_col", "asset_col", "value_col", "exposures", "exposure_cols"}
        scalar_kw = {}
        for key, value in call_args.items():
            if key in adapter_only:
                continue
            effective = self._PARAM_ALIASES.get(key, key)
            if effective in scalar_kw:
                raise TypeError(f"parameter {effective!r} supplied more than once")
            scalar_kw[effective] = value
        self.effective_parameters = dict(scalar_kw)
        # expose frame: kwargs['exposures'] is the exposure long frame.
        exposures = call_args.get("exposures")

        panel = _long_to_wide(values, value_col, time_col, asset_col)
        if exposures is not None:
            # Exposures come as a long frame with the SAME long schema but
            # multiple value columns (e1, e2, ...). Build one panel per
            # exposure column so FE receives exposure panels in column order.
            exp_value_cols = call_args.get("exposure_cols")
            if not exp_value_cols:
                raise ValueError("exposure_cols must explicitly identify exposure columns")
            unknown = set(exp_value_cols).difference(exposures.columns)
            if unknown:
                raise ValueError(f"unknown exposure columns: {sorted(unknown)}")
            exp_panels = [
                _long_to_wide(exposures, c, time_col, asset_col)
                for c in exp_value_cols
            ]
            if any(
                not item.index.equals(panel.index) or not item.columns.equals(panel.columns)
                for item in exp_panels
            ):
                raise ValueError("exposures must exactly match the factor time/asset axes")
            inspect.signature(self._op.calculate).bind(panel, *exp_panels, **scalar_kw)
            out = self._op.calculate(panel, *exp_panels, **scalar_kw)
        else:
            inspect.signature(self._op.calculate).bind(panel, **scalar_kw)
            out = self._op.calculate(panel, **scalar_kw)
        return _stack_back(out, values, time_col, asset_col, value_col)


class FeRecipeExecutor:
    """Execute an all-FE stateless TreatmentRecipe with one panel boundary."""

    def __init__(self, recipe, registry):
        self.recipe = recipe
        self.registry = registry
        self.runtime_stats: Dict[str, Any] = {}

    def __call__(self, values, *, value_col="value", time_col="date", asset_col="asset_id"):
        panel = _long_to_wide(values, value_col, time_col, asset_col)
        steps = []
        for step in self.recipe.ordered_steps:
            meta = self.registry.get(step.implementation_ref)
            if meta is None or self.registry.resolve_origin(step.implementation_ref) != "FE_OPERATOR":
                raise ValueError(
                    f"recipe step {step.implementation_ref!r} is not an FE stateless operator"
                )
            params = {
                FeOperatorExecutor._PARAM_ALIASES.get(k, k): v
                for k, v in dict(step.parameters).items()
            }
            # FE ``rank`` has fixed pct=True semantics and therefore declares
            # no scalar pct parameter. FP's domain gate has already rejected
            # pct=False; omit the identity-valued spelling from the physical call.
            if meta.fe_operator_id == "rank" and params.get("pct") is True:
                params.pop("pct")
            steps.append((meta.fe_operator_id, params))
        from factor_engine.backend.cleaned_bridge import execute_operator_recipe
        out = execute_operator_recipe(panel, tuple(steps), runtime_stats=self.runtime_stats)
        return _stack_back(out, values, time_col, asset_col, value_col)


def get_fe_executor(canonical: str, fallback: Optional[Callable] = None, *, allow_research_fallback: bool = False) -> Optional[FeOperatorExecutor]:
    """Return a lazily-FE-backed executor for ``canonical``.

    Returns ``None`` when FE cannot be imported (so the registry falls back to
    the retained FP-native kernel).  Raises only on an *unexpected* registry
    error; a missing operator surfaces at first call.
    """
    try:
        _fe_operator_registry()  # import + load probe
    except (ImportError, ModuleNotFoundError):
        return None
    return FeOperatorExecutor(canonical, fallback=fallback, allow_research_fallback=allow_research_fallback)


__all__ = [
    "FeOperatorExecutor",
    "FeRecipeExecutor",
    "get_fe_executor",
]
