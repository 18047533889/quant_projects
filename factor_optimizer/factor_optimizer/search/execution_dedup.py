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


def _execution_binding(plan):
    """Describe only execution paths whose semantic authority is known."""
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan
    from factor_optimizer.adapters.preprocessing import SmoothingRepairPlan

    if type(plan) is ValueRepairPlan:
        direct_routes = {
            "raw": "identity.v1",
            "sign": "factor_optimizer.adapters.repair_execution._execute_fe_neg",
            "rank_shape": "factor_preprocess.transforms.repair_shapes.rank_shape",
            "cs_rank": "factor_engine.adapters.cs_rank",
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
            route = ("pandas.multiply.v1" if dict(plan.parameters).get("multiplier") == 1.0
                     else direct_routes["sign"])
        elif plan.transform == "cs_rank":
            route = ("factor_engine.adapters.cs_rank" if dict(plan.parameters).get(
                "method", "average") == "average" else
                "factor_preprocess.transforms.repair_shapes.cross_sectional_rank")
        if route is not None:
            return {"route": route, "mapping_version": plan.mapping_version}
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
    if type(plan) is LayeredDecayPlan:
        return {"route": "factor_optimizer.adapters.layered_decay.LayeredDecayPlan.execute",
                "mapping_version": "layered-decay.v1"}
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
