"""FE native Polars elementwise adapter for small stateless predicates.

The adapter validates the same long-panel identity contract as ``fe_operator``
but passes only the value vector to a registered FE Polars expression. Inputs
whose dtype is not supported by this native path use the existing FE pandas
operator and expose that selected backend in their live execution identity.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
from typing import Any

import numpy as np
import pandas as pd

from factor_preprocess.errors import GovernanceError

_SCHEMA = "factor-preprocess-execution-identity/v1"
_COVERAGE = (
    "Selected FE is_null operator implementation and declared contract, plus "
    "this value-vector adapter and call mapping; excludes transitive runtime "
    "and data dependencies."
)


def _identity_digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_long_identity(values: pd.DataFrame, time_col: str, asset_col: str,
                            value_col: str) -> None:
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


def _adapter_hash() -> str:
    from factor_engine.backend.evidence_provenance import compute_implementation_hash

    sources = (
        inspect.getsource(_validate_long_identity),
        inspect.getsource(FeNativeElementwiseExecutor._native_operator),
        inspect.getsource(FeNativeElementwiseExecutor._native_call),
        inspect.getsource(FeNativeElementwiseExecutor._pandas),
        inspect.getsource(FeNativeElementwiseExecutor.__call__),
        inspect.getsource(get_fe_native_elementwise_executor),
    )
    return compute_implementation_hash("\n\0".join(sources))


class FeNativeElementwiseExecutor:
    """Run an FE Polars predicate over a long input's value vector."""

    def __init__(self, canonical: str = "is_null", *, fallback: Any = None,
                 allow_research_fallback: bool = False):
        if canonical != "is_null":
            raise ValueError("native elementwise adapter currently supports is_null only")
        self.canonical = canonical
        self.fallback = fallback
        self.allow_research_fallback = bool(allow_research_fallback)
        self._registry = None
        self._polars_op = None
        self._polars_entry = None
        self._last_binding = None
        self._pandas_executor = None
        self._last_backend: str | None = None
        self._last_value_col = "value"

    def _ensure_registry(self):
        if self._registry is None:
            from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
            from factor_engine.cleaned_operators.registry import OperatorRegistry

            ensure_cleaned_loaded()
            self._registry = OperatorRegistry
        return self._registry

    def _native_operator(self):
        registry = self._ensure_registry()
        from factor_engine.backend.polars_backend_kind import (
            PolarsImplementationKind,
            polars_backend_kind,
        )
        from factor_engine.backend.contracts import ExecutionKind

        operators, _, catalog = registry._read_state()
        op = operators.get(self.canonical, {}).get("polars")
        entry = catalog.get(self.canonical)
        spec = getattr(op, "_physical_spec", None)
        if op is None or spec is None or not isinstance(entry, Mapping):
            self._polars_op = None
            self._polars_entry = None
            return None
        if getattr(spec, "execution_kind", None) != ExecutionKind.POLARS_NATIVE_EXPR:
            self._polars_op = None
            self._polars_entry = None
            return None
        if polars_backend_kind(op, production_mode=True) != PolarsImplementationKind.POLARS_NATIVE:
            self._polars_op = None
            self._polars_entry = None
            return None
        self._polars_op = op
        self._polars_entry = entry
        return op

    def _native_call(self, values: pd.Series) -> np.ndarray:
        import polars as pl

        op = self._native_operator()
        if op is None:
            raise GovernanceError("FE is_null has no registered native Polars implementation")
        selected_entry = self._polars_entry
        # Polars' missing expression accepts numeric scalars only. Keep strings,
        # object, temporal, categorical, and other unsupported dtypes on the
        # existing FE pandas implementation instead of coercing their values.
        if not (pd.api.types.is_integer_dtype(values.dtype)
                or pd.api.types.is_float_dtype(values.dtype)):
            raise TypeError("value dtype is outside the native Polars is_null contract")
        dtype = getattr(values.dtype, "numpy_dtype", values.dtype)
        if np.dtype(dtype).kind == "f" and np.dtype(dtype).itemsize > 8:
            raise TypeError("value dtype is outside the native Polars is_null contract")
        source = pl.from_pandas(values.reset_index(drop=True), include_index=False)
        frame = pl.DataFrame({"value": source.rename("value")})
        result = op.calculate(frame)
        if not isinstance(result, pl.DataFrame) or "value" not in result.columns:
            raise GovernanceError("FE native is_null returned an invalid Polars result")
        self._last_binding = (op, selected_entry)
        return np.asarray(result["value"].to_numpy(), dtype=np.float64)

    def _pandas(self, values: pd.DataFrame, *, time_col: str, asset_col: str,
                value_col: str) -> pd.Series:
        if self._pandas_executor is None:
            from factor_preprocess.adapters.fe_operator import get_fe_executor

            executor = get_fe_executor(self.canonical, fallback=self.fallback,
                                       allow_research_fallback=self.allow_research_fallback)
            if executor is None:
                raise GovernanceError("FE pandas_numpy is unavailable for is_null fallback")
            self._pandas_executor = executor
        # pandas pivot/unstack has no extended-float or complex kernel.
        # Boxing preserves original scalars (including infinities and tiny finite
        # values); narrowing to float64 would change missingness semantics.
        source_dtype = getattr(values[value_col].dtype, "numpy_dtype", values[value_col].dtype)
        prepared = values
        if (pd.api.types.is_complex_dtype(source_dtype)
                or (pd.api.types.is_float_dtype(source_dtype)
                    and np.dtype(source_dtype).itemsize > 8)):
            prepared = values.copy(deep=False)
            prepared[value_col] = values[value_col].astype(object)
        result = self._pandas_executor(prepared, time_col=time_col,
                                       asset_col=asset_col, value_col=value_col)
        self._last_backend = "pandas_numpy"
        self._last_value_col = value_col
        return result

    def __call__(self, values: pd.DataFrame, *, value_col: str = "value",
                 time_col: str = "date", asset_col: str = "asset_id") -> pd.Series:
        _validate_long_identity(values, time_col, asset_col, value_col)
        source = values[value_col]
        try:
            result = self._native_call(source)
        except TypeError as exc:
            if "outside the native Polars" not in str(exc):
                raise
            return self._pandas(values, time_col=time_col, asset_col=asset_col,
                                value_col=value_col)
        if len(result) != len(values):
            raise GovernanceError("FE native is_null changed the input row count")
        self._last_backend = "polars"
        self._last_value_col = value_col
        return pd.Series(result, index=values.index, name=value_col)

    @property
    def execution_identity(self) -> dict:
        if self._last_backend == "pandas_numpy":
            from factor_preprocess.adapters.fe_operator import get_fe_operator_identity

            base = get_fe_operator_identity(self._pandas_executor, "missing_indicator")
            payload = dict(base)
            payload.pop("digest", None)
            from factor_engine.backend.evidence_provenance import compute_implementation_hash
            payload["adapter_implementation_hash"] = compute_implementation_hash(
                str(base["adapter_implementation_hash"]) + "\0" + _adapter_hash()
            )
            payload["adapter_call_contract"] = dict(payload["adapter_call_contract"])
            payload["adapter_call_contract"]["backend_dispatch"] = (
                "unsupported value dtype uses selected FE pandas_numpy operator"
            )
            import polars as pl
            payload["runtime_versions"] = dict(payload["runtime_versions"])
            payload["runtime_versions"]["polars"] = str(pl.__version__)
            payload["coverage_marker"] = _COVERAGE
            payload["binding_state"] = "current_selected_binding"
            return {**payload, "digest": _identity_digest(payload)}
        if self._last_backend == "polars":
            op, entry = self._last_binding
        else:
            op = self._native_operator()
            entry = self._polars_entry
        if op is None or not isinstance(entry, Mapping):
            raise GovernanceError("FE native Polars is_null identity is unavailable")
        from factor_engine.cleaned_operators.registry import _contract_hash, _impl_source_hash
        meta = (entry.get("backend_meta") or {}).get("polars") or {}
        source = str(meta.get("source") or "").strip()
        version = str(entry.get("semantic_version") or "").strip()
        if not source or not version:
            raise GovernanceError("FE native Polars identity metadata is incomplete")
        try:
            import polars as pl
        except ImportError as exc:
            raise GovernanceError("Polars runtime is unavailable") from exc
        payload = {
            "schema": _SCHEMA,
            "status": "bound",
            "execution_origin": "FE_OPERATOR",
            "identity_kind": "FE_OPERATOR_SELECTED_BACKEND",
            "binding_state": "last_execution" if self._last_backend == "polars" else "planned_native_candidate",
            "fe_canonical_id": self.canonical,
            "fe_operator_id": self.canonical,
            "transform_name": "missing_indicator",
            "backend": "polars",
            "backend_source": source,
            "semantic_version": version,
            "fe_implementation_hash": _impl_source_hash(op),
            "fe_contract_hash": _contract_hash(op),
            "adapter_implementation_hash": _adapter_hash(),
            "adapter_call_contract": {
                "input_layout": "pandas long frame -> single value vector -> FE Polars -> aligned pandas Series",
                "identity_columns": ["time_col", "asset_col"],
                "value_column": self._last_value_col,
                "ordering": "input row order and index preserved positionally",
                "identity_validation": "non-null time/asset and unique (time, asset)",
            },
            "runtime_versions": {
                "numpy": str(np.__version__),
                "pandas": str(pd.__version__),
                "polars": str(pl.__version__),
            },
            "coverage_marker": _COVERAGE,
        }
        return {**payload, "digest": _identity_digest(payload)}


def get_fe_native_elementwise_executor(canonical: str = "is_null", *, fallback=None,
                                       allow_research_fallback: bool = False):
    """Return the lazy FE native elementwise executor."""
    try:
        import polars  # noqa: F401
        executor = FeNativeElementwiseExecutor(
            canonical, fallback=fallback,
            allow_research_fallback=allow_research_fallback,
        )
        if executor._native_operator() is None:
            return None
        return executor
    except (ImportError, ModuleNotFoundError):
        return None


__all__ = ["FeNativeElementwiseExecutor", "get_fe_native_elementwise_executor"]
