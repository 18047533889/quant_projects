"""Versioned metric variants; runtime scenario inputs are deliberately not serialized."""
from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping, Optional

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.metric_artifacts import FrozenMapping


@dataclass(frozen=True)
class MetricInstance:
    metric_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    scenario_id: str = "default"
    horizon: Optional[int] = None
    price_convention: Optional[str] = None
    usage_profile: str = "research"
    cost_profile: str = "gross"
    portfolio_profile: str = "default"

    leg: str = "unspecified"
    output_mode: str = "FULL_DIAGNOSTIC"
    quantile_builder_parameters: Mapping[str, int] = field(default_factory=dict)
    metric_version: Optional[str] = None

    def __post_init__(self):
        from quant_evaluator.registry.metrics import CANONICAL_METRIC_ALIASES, get_metric
        canonical = CANONICAL_METRIC_ALIASES.get(self.metric_id, self.metric_id)
        spec = get_metric(canonical)
        if self.metric_version is not None and self.metric_version != spec.metric_version:
            raise ValueError("MetricInstance implementation version is unavailable")
        if self.output_mode not in {"SUMMARY_ONLY", "SERIES", "FULL_DIAGNOSTIC"}:
            raise ValueError("Unknown metric instance output_mode")
        if not all(isinstance(x, str) and x for x in (
                self.scenario_id, self.usage_profile, self.cost_profile, self.portfolio_profile, self.leg)):
            raise ValueError("Metric instance scenario identities must be nonempty strings")
        if self.horizon is not None and (type(self.horizon) is not int or self.horizon < 1):
            raise ValueError("Metric instance horizon must be a positive integer")
        signature = inspect.signature(spec.compute_fn)
        runtime = {"factor_batch", "label_bundle", "computed_metrics", "metadata", "factor_values",
                   "forward_returns", "returns", "validity_mask", "calendar_snapshot", "time_index", "factor_ids"}
        invalid = set(self.parameters) - set(signature.parameters) | (set(self.parameters) & runtime)
        if invalid:
            raise ValueError(f"Invalid metric instance parameters: {sorted(invalid)}")
        defaults = {k: p.default for k, p in signature.parameters.items()
                    if k not in runtime and p.default is not inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        defaults.update(self.parameters)
        object.__setattr__(self, "metric_id", canonical)
        object.__setattr__(self, "metric_version", spec.metric_version)
        object.__setattr__(self, "parameters", FrozenMapping(defaults))
        object.__setattr__(self, "quantile_builder_parameters", FrozenMapping(self.quantile_builder_parameters))
        # Force unsupported parameter identities to fail before allocating a backend.
        self.instance_id

    def to_dict(self):
        from dataclasses import fields
        from quant_evaluator.contracts._ndarray_codec import encode_value
        return {f.name: encode_value(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, value):
        from quant_evaluator.contracts._ndarray_codec import decode_value
        return cls(**{k: decode_value(v) for k, v in value.items()})

    @property
    def instance_id(self):
        return "mi_" + stable_content_hex(tag="MetricInstance.v1", fields=self.to_dict())


@dataclass(frozen=True)
class EvaluationScenario:
    """Explicit runtime binding; changing a string cannot change the supplied data."""
    label_bundle: Any
    portfolio_returns: Any = None
    holding_returns: Any = None
    portfolio_spec: Any = None
    trade_eligibility: Any = None
    calendar_snapshot: Any = None
    exposure_panel: Any = None
    cost_profile: str = "gross"
    portfolio_profile: str = "default"
    trajectory: Any = None

    def __post_init__(self):
        from quant_evaluator.contracts.label_bundle import LabelBundle
        if not isinstance(self.label_bundle, LabelBundle):
            raise TypeError("EvaluationScenario requires a LabelBundle")
        if self.trajectory is not None:
            if self.portfolio_returns is not None or self.holding_returns is not None:
                raise ValueError("Bind trajectory or portfolio/holding returns, not both")
            from collections.abc import Mapping
            if isinstance(self.trajectory, Mapping):
                from types import MappingProxyType
                object.__setattr__(self, "trajectory", MappingProxyType(dict(self.trajectory)))
            probe = self.probe_for("unspecified")
            object.__setattr__(self, "portfolio_returns", probe)
        if self.cost_profile != "gross" and self.trajectory is None:
            raise ValueError("Net scenarios require an actual portfolio trajectory")

    def validate(self, instance):
        from collections.abc import Mapping
        if self.trajectory is not None:
            trajectories = self.trajectory.values() if isinstance(self.trajectory, Mapping) else (self.trajectory,)
            if any(item.scenario_id != instance.scenario_id for item in trajectories):
                raise ValueError("Metric instance scenario disagrees with execution trajectory")
        self.probe_for(instance.leg)
        if instance.horizon is not None and instance.horizon != self.label_bundle.horizon:
            raise ValueError("Metric instance horizon disagrees with scenario labels")
        if instance.price_convention is not None and instance.price_convention != self.label_bundle.price_convention:
            raise ValueError("Metric instance price convention disagrees with scenario labels")
        if (instance.cost_profile, instance.portfolio_profile) != (self.cost_profile, self.portfolio_profile):
            raise ValueError("Metric instance profiles disagree with bound scenario")

    def probe_for(self, leg, factor_ids=None):
        if self.trajectory is None:
            if leg not in {"unspecified", "total"}:
                raise ValueError("Named portfolio legs require an execution trajectory")
            return self.portfolio_returns
        from collections.abc import Mapping
        from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact, trajectories_to_probe_artifact
        kwargs = dict(expected_portfolio_profile=self.portfolio_profile,
                      expected_cost_profile=self.cost_profile, expected_leg=leg)
        if isinstance(self.trajectory, Mapping):
            return trajectories_to_probe_artifact(self.trajectory,
                factor_ids=tuple(self.trajectory) if factor_ids is None else tuple(factor_ids), **kwargs)
        return trajectory_to_probe_artifact(self.trajectory, factor_ids=factor_ids, **kwargs)
