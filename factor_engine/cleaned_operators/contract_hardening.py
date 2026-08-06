# -*- coding: utf-8 -*-
"""Final operator-contract convergence.

This layer does not register numerical kernels.  It normalises runtime
validation and derives one machine-readable contract from the final registry,
policy and evidence state after all backends have been registered.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cleaned_operators.registry import OperatorRegistry

_APPLIED = False

# The final contract layer may derive lookback / parameter / shape contracts but
# must never alter these evidence-converged certification fields (review §2.7).
IMMUTABLE_CERTIFICATION_FIELDS = frozenset(
    {
        "production_certified",
        "operator_certification",
        "pit_safe",
        "status",
        "lifecycle_status",
        "semantic_pit_review_passed",
        "implementation_certified",
        "semantic_certified",
        "temporal_certified",
        "source_contract_certified",
        "edge_case_passed",
        "backend_passed",
    }
)

_STATEFUL_CANONICALS = frozenset(
    {
        "ts_ema",
        "ts_ewm_std",
        "ts_ewm_var",
        "ts_ewm_cov",
        "ts_ewm_corr",
        "RSI_WILDER",
        "ATR_WILDER",
        "ADX",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "ts_sma_cn",
    }
)

# ``pit_safe`` is deliberately NOT corrected here: the final contract layer may
# derive lookback / parameter / shape contracts but must never overwrite the
# evidence-converged certification fields (review §2.7).
_POLICY_CORRECTIONS = {
    "ADX": {"scope": "ts", "min_periods": 2},
    "MACD_line": {"scope": "ts", "min_periods": 1},
    "MACD_signal": {"scope": "ts", "min_periods": 1},
    "MACD_hist": {"scope": "ts", "min_periods": 1},
    "lqtp_historical_cvar": {"scope": "ts", "min_periods": 1},
    "ts_sma_cn": {"scope": "ts", "min_periods": 1},
}

_MIN_PERIODS_FLOORS = {
    "ts_std": 2,
    "ts_var": 2,
    "ts_cov": 2,
    "ts_corr": 2,
    "ts_beta": 2,
    "ts_autocorr": 2,
    "ts_skew": 3,
    "ts_kurt": 4,
    "ts_regression_slope": 3,
    "ts_regression_intercept": 3,
    "ts_regression_resid": 3,
    "ts_regression_r2": 3,
    "ts_regression_tstat": 3,
    "ts_partial_corr": 3,
    "ts_topk_std": 2,
    "ts_bottomk_std": 2,
}

_LOOKBACK_PARAM_PRIORITY = (
    "window",
    "periods",
    "d",
    "lag",
    "n",
    "max_lookback",
    "slow_period",
    "fast_period",
    "signal_period",
)


def _install_polars_validation() -> None:
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return

    from cleaned_operators import base_polars
    from cleaned_operators.base import _normalise_call, _validate_common_integer_relations

    def prepare(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        processed_args, processed_kwargs = _normalise_call(self.metadata, args, kwargs)
        frames = [
            value
            for value in (*processed_args, *processed_kwargs.values())
            if isinstance(value, pl.DataFrame)
        ]
        if frames:
            base = frames[0]
            if len(set(base.columns)) != len(base.columns):
                raise ValueError(f"{self.metadata.name}: primary Polars panel has duplicate columns")
            if "allow_panel_broadcast" not in set(self.metadata.tags or []):
                for position, frame in enumerate(frames[1:], start=1):
                    if frame.height != base.height:
                        raise ValueError(
                            f"{self.metadata.name}: Polars panel {position} height is misaligned"
                        )
                    if frame.columns != base.columns:
                        raise ValueError(
                            f"{self.metadata.name}: Polars panel {position} columns are misaligned"
                        )
        _validate_common_integer_relations(
            self.metadata, processed_args, processed_kwargs
        )
        valid = self.validate_params(*processed_args, **processed_kwargs)
        if valid is False:
            raise ValueError(f"{self.metadata.name}: parameter validation failed")
        return processed_args, processed_kwargs

    def series_calculate(self, *args, **kwargs):
        processed_args, processed_kwargs = self._prepare_call(tuple(args), dict(kwargs))
        return self._calculate_series(*processed_args, **processed_kwargs)

    def scalar_calculate(self, *args, **kwargs):
        processed_args, processed_kwargs = self._prepare_call(tuple(args), dict(kwargs))
        return self._calculate_scalar(*processed_args, **processed_kwargs)

    def transform_calculate(self, x, **kwargs):
        processed_args, processed_kwargs = self._prepare_call((x,), dict(kwargs))
        return self._calculate_series(processed_args[0], **processed_kwargs)

    def two_var_calculate(self, x, y, **kwargs):
        processed_args, processed_kwargs = self._prepare_call((x, y), dict(kwargs))
        return self._calculate_series(
            processed_args[0], processed_args[1], **processed_kwargs
        )

    base_polars.Operator._prepare_call = prepare
    base_polars.SeriesOperator.calculate = series_calculate
    base_polars.ScalarOperator.calculate = scalar_calculate
    base_polars.TransformOperator.calculate = transform_calculate
    base_polars.TwoVarOperator.calculate = two_var_calculate


def _correct_policy_metadata() -> None:
    from cleaned_operators import operator_policy

    explicit = operator_policy._EXPLICIT_POLICIES
    for canonical, correction in _POLICY_CORRECTIONS.items():
        current = dict(explicit.get(canonical) or {})
        current.update(correction)
        explicit[canonical] = current
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        catalog["scope"] = {
            "ts": "time_series",
            "cs": "cross_sectional",
        }.get(correction["scope"], correction["scope"])
        catalog["min_periods"] = correction.get("min_periods")

    for canonical, minimum in _MIN_PERIODS_FLOORS.items():
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        previous = catalog.get("min_periods")
        catalog["min_periods"] = max(
            minimum,
            int(previous) if isinstance(previous, (int, np.integer)) else minimum,
        )


def _lookback_contract(canonical: str, catalog: dict[str, Any]) -> dict[str, Any]:
    params = tuple(catalog.get("param_names") or ())
    scope = str(catalog.get("scope") or "unknown")
    if scope == "fundamental_period":
        return {
            "kind": "fiscal_period",
            "parameter": "periods" if "periods" in params else None,
            "requires_period_id": True,
            "revision_aware": True,
        }
    if canonical in _STATEFUL_CANONICALS:
        controlling = [name for name in _LOOKBACK_PARAM_PRIORITY if name in params]
        return {
            "kind": "recursive_state",
            "parameters": controlling,
            "requires_checkpoint_for_segments": True,
            "finite_warmup_is_approximation": True,
        }
    if scope in {"time_series", "ts", "session_intraday"}:
        controlling = [name for name in _LOOKBACK_PARAM_PRIORITY if name in params]
        if canonical == "ts_days_since":
            return {
                "kind": "parameterized_rows",
                "parameters": ["max_lookback"],
                "inclusive_max_distance": True,
                "unbounded_when_null": True,
            }
        return {
            "kind": "parameterized_rows" if controlling else "causal_unbounded",
            "parameters": controlling,
            "includes_current_bar": True,
        }
    return {"kind": "zero", "rows": 0}


def _parameter_constraints(canonical: str) -> dict[str, Any]:
    if canonical in {"ts_topk_std", "ts_bottomk_std"}:
        return {"k": {"type": "integer", "minimum": 2}, "ddof": 1}
    if canonical == "ts_days_since":
        return {"max_lookback": {"type": "positive_integer_or_null", "inclusive": True}}
    if canonical in {
        "period_lag",
        "period_change",
        "period_average",
        "period_cagr",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
    }:
        return {
            "revision_policy": ["first_available", "latest_available"],
            "require_consecutive": [False, True],
        }
    return {}


def _derive_contracts() -> None:
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_spec import infer_production_policy
    from backend.primitive_evidence import POLARS_NO_FALLBACK_VERIFIED

    scope_map = {"ts": "time_series", "cs": "cross_sectional"}
    for canonical, implementations in OperatorRegistry._operators.items():
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        preferred = implementations.get("pandas_numpy") or next(iter(implementations.values()))
        policy = infer_operator_policy(preferred, canonical=canonical)
        policy_scope = scope_map.get(policy.scope, policy.scope)
        if policy_scope != "unknown":
            catalog["scope"] = policy_scope
        # Certification fields are immutable here: never overwrite the value
        # converged by ``reconcile_operator_certification`` (review §2.7).
        catalog.setdefault("pit_safe", bool(policy.pit_safe))
        catalog["stateful"] = bool(catalog.get("stateful")) or canonical in _STATEFUL_CANONICALS
        lookback = _lookback_contract(canonical, catalog)
        catalog["lookback_contract"] = lookback
        catalog["lookback_param_names"] = list(lookback.get("parameters") or ())
        catalog["parameter_constraints"] = _parameter_constraints(canonical)
        catalog["contract_version"] = "2.1"

        backend_meta = dict(catalog.get("backend_meta") or {})
        signatures = {}
        for backend, operator in implementations.items():
            params = list(getattr(getattr(operator, "metadata", None), "param_names", None) or ())
            signatures[backend] = params
            meta = dict(backend_meta.get(backend) or {})
            meta["param_names"] = params
            if backend == "polars":
                meta["no_fallback_runtime_verified"] = canonical in POLARS_NO_FALLBACK_VERIFIED
                meta["production_native_source_of_truth"] = "verified_runtime_evidence"
            backend_meta[backend] = meta
        catalog["backend_meta"] = backend_meta
        catalog["backend_signatures"] = signatures

        production_policy = infer_production_policy(canonical)
        contract = {
            "version": "2.1",
            "canonical": canonical,
            "surface": catalog.get("surface", "unknown"),
            "lifecycle_status": catalog.get("lifecycle_status", catalog.get("status", "research")),
            "production_policy": production_policy,
            "scope": catalog.get("scope", policy_scope),
            "pit_safe": bool(catalog.get("pit_safe")),
            "stateful": bool(catalog.get("stateful")),
            "shape_preserving": bool(getattr(policy, "shape_preserving", True)),
            "param_names": list(catalog.get("param_names") or ()),
            "backend_signatures": signatures,
            "backends": sorted(implementations),
            "lookback": lookback,
            "parameter_constraints": catalog["parameter_constraints"],
        }
        catalog["contract"] = contract
        catalog["production_policy"] = production_policy


def _validate_final_contracts() -> None:
    errors: list[str] = []
    for canonical, catalog in OperatorRegistry._catalog.items():
        if canonical not in OperatorRegistry._operators:
            continue
        contract = catalog.get("contract") or {}
        if not contract:
            errors.append(f"{canonical}: missing unified contract")
            continue
        if contract.get("scope") == "unknown" and contract.get("surface") == "daily":
            errors.append(f"{canonical}: daily operator has unknown scope")
        if not contract.get("param_names") and canonical not in {"constant"}:
            errors.append(f"{canonical}: missing parameter contract")
        if not contract.get("lookback"):
            errors.append(f"{canonical}: missing lookback contract")
    if errors:
        raise RuntimeError("operator contract convergence failed: " + "; ".join(errors[:20]))


def _snapshot_certification_state() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for canonical, catalog in OperatorRegistry._catalog.items():
        preserved = {
            field: catalog.get(field)
            for field in IMMUTABLE_CERTIFICATION_FIELDS
            if field in catalog
        }
        if preserved:
            snapshot[canonical] = preserved
    return snapshot


def _restore_certification_state(snapshot: dict[str, dict[str, Any]]) -> None:
    for canonical, preserved in snapshot.items():
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        for field, value in preserved.items():
            catalog[field] = value


def apply_final_contract_hardening() -> None:
    global _APPLIED
    if _APPLIED:
        return
    # Write-protect the evidence-converged certification fields across every
    # contract-derivation code path (review §2.7).
    snapshot = _snapshot_certification_state()
    _install_polars_validation()
    _correct_policy_metadata()
    _derive_contracts()
    _validate_final_contracts()
    _restore_certification_state(snapshot)
    _APPLIED = True
