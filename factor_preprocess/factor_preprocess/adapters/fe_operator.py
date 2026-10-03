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
executor is ``None``. Production registry execution fails closed; only an
explicit ``allow_research=True`` call may fall back to the retained FP-native
kernel (plan §26 F3 migration path).

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
from collections.abc import Mapping
import hashlib
import json

import numpy as np
import pandas as pd

from factor_preprocess.errors import GovernanceError

_EXECUTION_IDENTITY_SCHEMA = "factor-preprocess-execution-identity/v1"
_BACKEND = "pandas_numpy"
_IDENTITY_COVERAGE = (
    "Selected FE pandas_numpy operator implementation and declared operator "
    "contract, plus the FP long/panel adapter code and call mapping; does not "
    "cover the full transitive runtime, native libraries, or data dependencies."
)
_ADAPTER_ONLY_PARAMS = frozenset({
    "values", "time_col", "asset_col", "value_col", "exposures", "exposure_cols",
})


def _adapter_implementation_hash() -> str:
    """Hash only the adapter code participating in FE_OPERATOR execution."""
    from factor_engine.backend.evidence_provenance import compute_implementation_hash

    sources = (
        inspect.getsource(_long_to_wide),
        inspect.getsource(_stack_back),
        inspect.getsource(FeOperatorExecutor._ensure),
        inspect.getsource(FeOperatorExecutor.__call__),
    )
    return compute_implementation_hash("\n\0".join(sources))


def _identity_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
    asset_types = {type(value) for value in values[asset_col]}
    if len(asset_types) > 1:
        raise ValueError(
            "mixed asset identity types require canonicalization by the "
            "security catalog before FP/FE execution"
        )
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
    result = np.full(len(template), np.nan, dtype=float)
    # Object indexes preserve dictionary-key semantics, including 1 != "1"
    # and Timestamp != its string representation. DatetimeIndex.get_indexer
    # would otherwise coerce parseable strings to timestamps.
    rows = pd.Index(panel.index, dtype=object).get_indexer(
        pd.Index(template[time_col], dtype=object))
    cols = pd.Index(panel.columns, dtype=object).get_indexer(
        pd.Index(template[asset_col], dtype=object))
    present = (rows >= 0) & (cols >= 0)
    result[present] = panel.to_numpy()[rows[present], cols[present]]
    return pd.Series(result, index=template.index, name=value_col)


class FeOperatorExecutor:
    """Callable executor for one FE canonical operator.

    ``fallback`` is the retained FP-native kernel. It is used only when FE is
    unavailable and the executor was explicitly created with
    ``allow_research_fallback=True`` (never deleted — plan §26 F3).
    """

    _PARAM_ALIASES = {"max_lag": "max_periods", "min_observations": "min_obs"}

    def __init__(self, canonical: str, fallback: Optional[Callable] = None, *, allow_research_fallback: bool = False):
        self.canonical = canonical
        self.fallback = fallback
        self._registry = None
        self._op = None
        self._resolved_canonical = None
        self.allow_research_fallback = allow_research_fallback
        self.effective_parameters: Dict[str, Any] = {}

    def _ensure(self):
        if self._registry is None:
            self._registry = _fe_operator_registry()
        canonical = self._registry.resolve_canonical(self.canonical)
        op = self._registry.get(canonical, backend=_BACKEND)
        if op is None:
            raise KeyError(f"FE operator {canonical!r} unavailable on {_BACKEND}")
        # Refresh on every execution/identity lookup. A cached operator must
        # never be paired with metadata from a replacement registry binding.
        self._resolved_canonical = canonical
        self._op = op

    @property
    def execution_identity(self) -> dict:
        """Return a live identity for the exact backend binding this adapter uses."""
        try:
            self._ensure()
            from factor_engine.cleaned_operators.registry import (
                _contract_hash,
                _impl_source_hash,
            )

            _, _, catalog = self._registry._read_state()
            canonical = self._resolved_canonical
            entry = catalog.get(canonical)
            if not isinstance(entry, Mapping):
                raise GovernanceError(f"FE catalog entry missing for {canonical!r}")
            backend_meta = (entry.get("backend_meta") or {}).get(_BACKEND) or {}
            source = str(backend_meta.get("source") or "").strip()
            semantic_version = str(entry.get("semantic_version") or "").strip()
            if not source or not semantic_version:
                raise GovernanceError(
                    f"FE identity metadata is incomplete for {canonical!r}/{_BACKEND}"
                )
            implementation_hash = _impl_source_hash(self._op)
            contract_hash = _contract_hash(self._op)
            adapter_hash = _adapter_implementation_hash()
            if not all((implementation_hash, contract_hash, adapter_hash)):
                raise GovernanceError(
                    f"FE operator identity is not certifiable for {canonical!r}"
                )
        except GovernanceError:
            raise
        except Exception as exc:
            raise GovernanceError(
                f"FE operator identity is unavailable for {self.canonical!r}"
            ) from exc

        payload = {
            "schema": _EXECUTION_IDENTITY_SCHEMA,
            "status": "bound",
            "execution_origin": "FE_OPERATOR",
            "identity_kind": "FE_OPERATOR_SELECTED_BACKEND",
            "fe_canonical_id": canonical,
            "backend": _BACKEND,
            "backend_source": source,
            "semantic_version": semantic_version,
            "fe_implementation_hash": implementation_hash,
            "fe_contract_hash": contract_hash,
            "adapter_implementation_hash": adapter_hash,
            "adapter_call_contract": {
                "input_layout": "pandas long frame -> pandas wide panel -> FE -> aligned pandas Series",
                "identity_columns": ["time_col", "asset_col"],
                "value_column": "value_col",
                "parameter_aliases": dict(sorted(self._PARAM_ALIASES.items())),
                "adapter_only_parameters": sorted(_ADAPTER_ONLY_PARAMS),
                "runtime_arguments": "caller-provided parameters are alias-normalized and validated against the FE contract",
                "fe_parameter_contract_hash": contract_hash,
            },
            "runtime_versions": {
                "numpy": str(np.__version__),
                "pandas": str(pd.__version__),
            },
            "coverage_marker": _IDENTITY_COVERAGE,
        }
        return {**payload, "digest": _identity_digest(payload)}

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
        adapter_only = _ADAPTER_ONLY_PARAMS
        scalar_kw = {}
        for key, value in call_args.items():
            if key in adapter_only:
                continue
            effective = self._PARAM_ALIASES.get(key, key)
            if effective in scalar_kw:
                raise TypeError(f"parameter {effective!r} supplied more than once")
            scalar_kw[effective] = value
        # expose frame: kwargs['exposures'] is the exposure long frame.
        exposures = call_args.get("exposures")

        if self.canonical == "cs_neutralize" and self.fallback is not None:
            # Preserve the declared FP default even when the caller omits it.
            support = inspect.signature(self.fallback).parameters.get("min_observations")
            if support is not None:
                scalar_kw.setdefault("min_obs", support.default)
        self.effective_parameters = dict(scalar_kw)

        panel = _long_to_wide(values, value_col, time_col, asset_col)
        if exposures is not None:
            # Exposures come as a long frame with the SAME long schema but
            # multiple value columns (e1, e2, ...). Build one panel per
            # exposure column so FE receives exposure panels in column order.
            exp_value_cols = call_args.get("exposure_cols")
            if exp_value_cols is None and self.canonical == "cs_neutralize" and self.fallback is not None:
                # Public FP OLS accepts an exposure frame, not exposure_cols.
                # Match its declared convention: every non-identity column.
                exp_value_cols = [c for c in exposures.columns if c not in {time_col, asset_col}]
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
            if self.canonical == "cs_neutralize":
                # FE's canonical call is (y, exposures, group, weight, ...).
                # Splatting panels makes the second exposure become a group
                # and can create singleton regressions / all-NaN residuals.
                inspect.signature(self._op.calculate).bind(panel, exposures=tuple(exp_panels), **scalar_kw)
                out = self._op.calculate(panel, exposures=tuple(exp_panels), **scalar_kw)
            else:
                inspect.signature(self._op.calculate).bind(panel, *exp_panels, **scalar_kw)
                out = self._op.calculate(panel, *exp_panels, **scalar_kw)
        else:
            inspect.signature(self._op.calculate).bind(panel, **scalar_kw)
            out = self._op.calculate(panel, **scalar_kw)
        return _stack_back(out, values, time_col, asset_col, value_col)


class FeRecipeExecutor:
    """Execute an all-FE stateless TreatmentRecipe with one panel boundary."""

    def __init__(self, recipe, registry, *, execution_context=None, backend=None, allow_research=False):
        self.recipe = recipe
        self.registry = registry
        self.execution_context = execution_context
        self.backend = backend
        self.allow_research = bool(allow_research)
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
        out = execute_operator_recipe(
            panel, tuple(steps), runtime_stats=self.runtime_stats,
            execution_context=self.execution_context, backend=self.backend,
            allow_research=self.allow_research,
        )
        return _stack_back(out, values, time_col, asset_col, value_col)


def get_fe_executor(canonical: str, fallback: Optional[Callable] = None, *, allow_research_fallback: bool = False) -> Optional[FeOperatorExecutor]:
    """Return a lazily-FE-backed executor for ``canonical``.

    Returns ``None`` when FE cannot be imported. The registry fails closed for
    production execution and may use the retained FP-native kernel only when
    research fallback was explicitly requested. Raises unexpected registry
    errors; a missing operator surfaces at first call.
    """
    try:
        _fe_operator_registry()  # import + load probe
    except (ImportError, ModuleNotFoundError):
        return None
    return FeOperatorExecutor(canonical, fallback=fallback, allow_research_fallback=allow_research_fallback)


def get_fe_operator_identity(executor, transform_name: str) -> dict:
    """Bind an FP transform name to its executor's live FE identity."""
    identity = getattr(executor, "execution_identity", None)
    if not isinstance(identity, dict) or identity.get("status") != "bound":
        raise GovernanceError(
            f"FE operator execution identity is unavailable for {transform_name!r}"
        )
    payload = dict(identity)
    payload["transform_name"] = transform_name
    payload["fe_operator_id"] = payload.get("fe_canonical_id")
    payload.pop("digest", None)
    return {**payload, "digest": _identity_digest(payload)}


__all__ = [
    "FeOperatorExecutor",
    "FeRecipeExecutor",
    "get_fe_executor",
    "get_fe_operator_identity",
]
