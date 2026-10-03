"""TRAIN-context-bound deduplication of executable repair proposals."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from enum import Enum
import json
import math


@dataclass(frozen=True)
class DeduplicatedProposal:
    proposal: tuple
    aliases: tuple[dict, ...]


def _json_value(value):
    """Return stable JSON-compatible values, rejecting non-finite floats."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("execution parameters must be finite")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("execution binding keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported execution binding value: {type(value).__qualname__}")


def _fe_rank_execution_binding():
    """Bind the adapter's actual FE rank implementation, not a route label.

    This adapter calls FE's canonical ``rank`` through its ``pandas_numpy``
    registration. It does not select the Polars registration.
    """
    import numpy as np
    import pandas as pd
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_optimizer.adapters.fe_execution_identity import build_fe_operator_identity
    from factor_optimizer.adapters.repair_execution import _execute_fe_cs_rank

    canonical = OperatorRegistry.resolve_canonical_strict("rank")
    operator = OperatorRegistry.get(canonical, backend="pandas_numpy", mode="any")
    if operator is None:
        raise RuntimeError("FactorEngine canonical 'rank' has no pandas_numpy backend")
    entry = OperatorRegistry.catalog_entry(canonical)
    return {
        "route": "factor_engine.operator_registry",
        "binding": build_fe_operator_identity(
            canonical=canonical, backend="pandas_numpy", mode="any",
            operator=operator, adapter=_execute_fe_cs_rank,
            input_contract="pandas.DataFrame[asset_id,date,value]->float64",
            semantic_version=entry.get("semantic_version"),
            versions={"numpy_version": np.__version__,
                      "pandas_version": pd.__version__}),
    }


def _fe_neg_execution_binding():
    """Bind the FE neg backend selected by the SIGN adapter at runtime."""
    import importlib.metadata
    import numpy as np
    import pandas as pd
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_optimizer.adapters.fe_execution_identity import build_fe_operator_identity
    from factor_optimizer.adapters.repair_execution import _execute_fe_neg

    canonical = OperatorRegistry.resolve_canonical_strict("neg")
    # Keep lookup order and mode identical to _execute_fe_neg: Polars first,
    # then pandas_numpy only when the Polars registration is absent.
    operator = OperatorRegistry.get("neg", backend="polars", mode="any")
    if operator is not None:
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError(
                "FactorEngine neg selected Polars but Polars cannot be imported"
            ) from exc
        backend = "polars"
        polars_version = getattr(pl, "__version__", None)
        if not isinstance(polars_version, str) or not polars_version:
            raise RuntimeError("cannot certify the selected Polars version")
    else:
        operator = OperatorRegistry.get(
            "neg", backend="pandas_numpy", mode="any")
        if operator is None:
            raise RuntimeError("FactorEngine canonical 'neg' has no executable backend")
        backend = "pandas_numpy"
        try:
            polars_version = importlib.metadata.version("polars")
        except importlib.metadata.PackageNotFoundError:
            polars_version = "not-installed"

    entry = OperatorRegistry.catalog_entry(canonical)
    return {
        "route": "factor_engine.operator_registry",
        "binding": build_fe_operator_identity(
            canonical=canonical, backend=backend, mode="any", operator=operator,
            adapter=_execute_fe_neg,
            input_contract="pandas.Series->float64; FE single-column x",
            semantic_version=entry.get("semantic_version"),
            versions={"numpy_version": np.__version__,
                      "pandas_version": pd.__version__,
                      "polars_version": polars_version}),
    }


def _fe_tail_saturation_execution_binding():
    """Bind the exact FE winsorize implementation used by the adapter."""
    import platform
    import numpy as np
    import pandas as pd
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_optimizer.adapters.fe_execution_identity import build_fe_operator_identity
    from factor_optimizer.adapters.fe_tail_saturation import execute_fe_tail_saturation

    canonical = OperatorRegistry.resolve_canonical_strict("winsorize")
    operator = OperatorRegistry.get(
        "winsorize", backend="pandas_numpy", mode="any")
    if operator is None:
        raise RuntimeError(
            "FactorEngine canonical 'winsorize' has no pandas_numpy backend")
    entry = OperatorRegistry.catalog_entry(canonical)
    return {
        "route": "factor_engine.operator_registry",
        "binding": build_fe_operator_identity(
            canonical=canonical, backend="pandas_numpy", mode="any",
            operator=operator, adapter=execute_fe_tail_saturation,
            input_contract="pandas.DataFrame[date,value]->float64; row-wise by date",
            semantic_version=entry.get("semantic_version"),
            versions={"python_version": platform.python_version(),
                      "numpy_version": np.__version__,
                      "pandas_version": pd.__version__}),
    }


def _execution_binding(plan):
    """Describe only execution paths whose semantic authority is known."""
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan
    from factor_optimizer.adapters.preprocessing import SmoothingRepairPlan

    if type(plan) is ValueRepairPlan:
        direct_routes = {
            "raw": "identity.v1",
            "sign": "factor_optimizer.adapters.repair_execution._execute_fe_neg",
            "rank_shape": "factor_preprocess.transforms.repair_shapes.rank_shape",
            "fp_cs_rank_min": "factor_preprocess.transforms.repair_shapes.cross_sectional_rank.min",
            "ts_rank_history": "factor_preprocess.transforms.temporal_representation.time_series_rank",
            "ts_zscore_history": "factor_preprocess.transforms.temporal_representation.capped_time_series_zscore",
            "trailing_sma": "factor_optimizer.adapters.fe_smoothing.execute_lagged_sma",
            "capped_zscore": "factor_preprocess.transforms.repair_shapes.capped_zscore",
            "tail_hinge": "factor_preprocess.transforms.repair_shapes.tail_hinge",
            "tail_saturation": "factor_optimizer.adapters.fe_tail_saturation.execute_fe_tail_saturation",
            "robust_scale": "factor_preprocess.transforms.repair_shapes.robust_scale",
        }
        route = direct_routes.get(plan.transform)
        if plan.transform == "sign":
            multiplier = dict(plan.parameters).get("multiplier")
            if type(multiplier) is not float or multiplier not in {-1.0, 1.0}:
                raise ValueError("uncertified SIGN_ORIENTATION multiplier")
            if multiplier == -1.0:
                return _fe_neg_execution_binding()
            route = "pandas.multiply.v1"
        elif plan.transform == "cs_rank":
            if dict(plan.parameters).get("method", "average") == "average":
                return _fe_rank_execution_binding()
            from factor_optimizer.adapters.repair_execution_identity import (
                build_direct_fp_execution_identity,
            )
            return {
                "route": "factor_preprocess.transforms.repair_shapes.cross_sectional_rank",
                "mapping_version": plan.mapping_version,
                "binding": build_direct_fp_execution_identity(plan),
            }
        if plan.transform == "tail_saturation":
            return _fe_tail_saturation_execution_binding()
        if plan.transform == "trailing_sma":
            return _fe_smoothing_execution_binding()
        if route is not None:
            if plan.transform in {
                "rank_shape", "fp_cs_rank_min", "ts_rank_history",
                "ts_zscore_history", "capped_zscore", "tail_hinge", "robust_scale",
            }:
                from factor_optimizer.adapters.repair_execution_identity import (
                    build_direct_fp_execution_identity,
                )
                return {
                    "route": route, "mapping_version": plan.mapping_version,
                    "binding": build_direct_fp_execution_identity(plan),
                }
            return {"route": route, "mapping_version": plan.mapping_version}
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
    if type(plan) is LayeredDecayPlan:
        from factor_optimizer.adapters.layered_decay_identity import (
            build_layered_decay_execution_identity,
        )
        return {"route": "factor_optimizer.adapters.layered_decay_long._execute_sparse_layered_decay_validated",
                "mapping_version": "layered-decay.v1",
                "binding": build_layered_decay_execution_identity()}
    if type(plan) is not SmoothingRepairPlan and type(plan) is not ValueRepairPlan:
        raise TypeError("unknown plan execution authority; keep proposal distinct")
    from factor_preprocess.registry.transforms import get_default_registry
    registry = get_default_registry()
    metadata = registry.get(plan.transform)
    executor = registry.get_execution(plan.transform)
    fields = (
        "name", "version", "semantic_id", "signature_hash", "implementation_hash",
        "numeric_policy_hash", "implementation_origin", "fe_operator_id", "fit_kind",
        "numeric_policy", "fe_equivalent_semantics", "parameter_domain",
    )
    binding = {field: getattr(metadata, field, None) for field in fields}
    binding["resolved_origin"] = registry.resolve_origin(plan.transform)
    identity = getattr(executor, "execution_identity", None)
    if identity is not None:
        binding["execution_identity"] = identity
    target = getattr(executor, "executor", executor)
    binding["executor"] = {"module": getattr(target, "__module__", None),
                           "qualname": getattr(target, "__qualname__", None)}
    return {"route": "fp_registry", "binding": _json_value(binding)}


def _fe_smoothing_execution_binding():
    """Bind the selected direct FE lagged-SMA function and adapter."""
    from factor_optimizer.adapters.fe_smoothing_identity import (
        build_fe_smoothing_identity,
    )

    return {
        "route": "factor_engine.backend.long_smoothing.lagged_mean",
        "binding": _json_value(build_fe_smoothing_identity()),
    }


def _execution_signature(plan, orientation, baseline_context):
    params = dict(plan.parameters)
    binding = _execution_binding(plan)
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
    scale = None if type(plan) is LayeredDecayPlan else plan.natural_time_scale
    payload = {
        "version": "research-execution.v1",
        "transform": plan.transform,
        "parameters": _json_value(params),
        "execution_binding": binding,
        "baseline_context": baseline_context,
        "training_context_ref": plan.training_context_ref,
        "natural_time_scale": scale,
        "orientation": orientation,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def deduplicate_proposals(proposals, *, compile_plan, final_identity=None, baseline_context=None):
    """Group only plans with the same resolved execution and TRAIN context.

    A compile failure remains its own proposal for the normal ineligible
    reporting path. Representative selection is the smallest final oriented
    plan identity, preserving the optimizer's established deterministic tie.
    """
    groups = {}
    passthrough = []
    for ordinal, proposal in enumerate(proposals):
        family, parameters, precompiled, orientation = proposal
        try:
            plan = precompiled or compile_plan(family, parameters)
            signature = _execution_signature(plan, orientation, baseline_context)
        except Exception:
            passthrough.append((ordinal, DeduplicatedProposal(proposal, ())))
            continue
        identity = (final_identity(plan, orientation) if final_identity is not None
                    else plan.identity if orientation == 1
                    else f"oriented:{plan.identity}:-1")
        alias = {"family": family, "parameters": dict(parameters),
                 "orientation": orientation, "plan_identity": identity,
                 "base_plan_identity": plan.identity}
        resolved_proposal = (family, dict(parameters), plan, orientation)
        groups.setdefault(signature, []).append((ordinal, resolved_proposal, alias, identity))

    retained = list(passthrough)
    for members in groups.values():
        members.sort(key=lambda item: (item[3], item[0]))
        _, representative, _, _ = members[0]
        aliases = tuple(item[2] for item in sorted(members, key=lambda item: item[0]))
        retained.append((min(item[0] for item in members),
                         DeduplicatedProposal(representative, aliases)))
    return tuple(item for _, item in sorted(retained, key=lambda item: item[0]))


__all__ = ["DeduplicatedProposal", "deduplicate_proposals"]
