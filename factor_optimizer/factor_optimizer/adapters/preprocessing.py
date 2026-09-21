"""Versioned research bridge from smoothing proposals to real FP kernels.

This adapter never calculates evaluator scores or fits a time scale. Callers
must freeze that scale from TRAIN and explicitly opt into research execution.
It does not implement production recipe admission or the other repair families.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from typing import Mapping, Tuple

from factor_optimizer.search.conditional_search import build_conditional_tree


class IneligibleSmoothingRepair(ValueError):
    """A proposed repair cannot bind to the current FP execution authority."""


@dataclass(frozen=True)
class SmoothingRepairPlan:
    """Immutable resolved kernel arguments and training-context provenance."""

    family: str
    transform: str
    parameters: Tuple[Tuple[str, object], ...]
    training_context_ref: str
    natural_time_scale: float
    mapping_version: str = "smoothing-repair.v2"

    @property
    def identity(self) -> str:
        payload = {"family": self.family, "transform": self.transform,
                   "parameters": self.parameters, "training_context_ref": self.training_context_ref,
                   "natural_time_scale": self.natural_time_scale, "version": self.mapping_version}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def execute(self, values, *, allow_research: bool = False):
        """Delegate a long asset_id/date/value panel to FP; retain its lag/NaNs.

        No data loading, scoring, publication, or production admission occurs.
        Provide historical warmup before the scoring slice, not future data.
        """
        if allow_research is not True:
            raise ValueError("this adapter is research-only; explicit allow_research=True required")
        from factor_preprocess.registry.transforms import get_default_registry

        return get_default_registry().get_execution(self.transform)(values, **dict(self.parameters))


def compile_smoothing_repair(family: str, parameters: Mapping[str, object], *,
                             natural_time_scale: float,
                             training_context_ref: str) -> SmoothingRepairPlan:
    """Resolve a supported conditional candidate to FP parameters.

    CAUSAL_SMOOTHING uses h=natural_time_scale*relative. SMA has ceil(h)
    observations and full-window warmup. KAMA uses ER=ceil(h), fast=2,
    slow=max(3,ceil(3*h)). EWMA/IIR share half-life h; Kalman has unit
    measurement noise and q=alpha**2/(1-alpha), matching steady-state gain.

    DECAY_REFINEMENT(relative=True) uses h=scale*decay. Otherwise decay is
    previous-state retention, so alpha=1-decay. Zero decay always means a
    one-bar lag, never an invalid zero half-life or an unlagged raw signal.
    The FP registry may forbid that gain/window; reject it before evaluation,
    rather than clamp it silently or execute outside the approved domain.
    This explicit v1 mapping must not be attached to older ambiguous recipes.
    """
    if family not in {"CAUSAL_SMOOTHING", "DECAY_REFINEMENT"}:
        raise ValueError("repair family has no smoothing compiler")
    if (isinstance(natural_time_scale, bool) or not isinstance(natural_time_scale, numbers.Real)
            or not math.isfinite(natural_time_scale) or not 1 <= natural_time_scale <= 10000):
        raise ValueError("TRAIN natural_time_scale must be finite and within [1, 10000] bars")
    if not isinstance(training_context_ref, str) or not training_context_ref.strip():
        raise ValueError("training_context_ref is required")
    params = dict(parameters)
    build_conditional_tree([family])[0].validate_params(params)
    if family == "DECAY_REFINEMENT":
        if type(params["half_life_relative"]) is not bool:
            raise ValueError("half_life_relative must be a strict bool")
        decay = float(params["decay"])
        if not params["half_life_relative"] or decay == 0:
            transform, kwargs = "one_sided_iir_lowpass", {"alpha": 1. - decay}
        else:
            transform, kwargs = "ewma", {"halflife": natural_time_scale * decay, "min_periods": 1}
    else:
        h = float(natural_time_scale) * float(params["natural_time_scale_relative"])
        alpha = -math.expm1(-math.log(2.) / h)
        method = params["method"]
        if method == "SMA":
            window = max(1, math.ceil(h))
            transform, kwargs = "trailing_sma", {"window": window, "min_periods": window}
        elif method == "EWMA":
            transform, kwargs = "ewma", {"halflife": h, "min_periods": 1}
        elif method == "IIR":
            transform, kwargs = "one_sided_iir_lowpass", {"alpha": alpha}
        elif method == "KAMA":
            transform, kwargs = "kama", {"period_er": max(1, math.ceil(h)), "period_fast": 2,
                                          "period_slow": max(3, math.ceil(3*h)), "use_current": False}
        else:  # Kalman is the only remaining validated choice.
            transform, kwargs = "kalman_local_level", {
                "process_noise": alpha**2 / (1.-alpha), "measurement_noise": 1.}
    from factor_preprocess.registry.transforms import get_default_registry
    metadata = get_default_registry().get(transform)
    try:
        metadata.bind_parameters(kwargs)
    except ValueError as exc:
        raise IneligibleSmoothingRepair(f"{family} cannot execute {transform}: {exc}") from exc
    return SmoothingRepairPlan(family, transform, tuple(sorted(kwargs.items())),
                               training_context_ref, float(natural_time_scale))


def compile_admissible_smoothing_grid(
        *, natural_time_scale: float, training_context_ref: str,
        target_time_scales=(3.0, 5.0, 8.0, 10.0, 13.0, 20.0, 30.0, 60.0),
) -> Tuple[SmoothingRepairPlan, ...]:
    """Compile a small method-balanced grid inside both FO and FP domains.

    Target time scales are desired half-lives/windows in bars. The compiler
    converts each target to FO's relative parameter, validates the declared
    FO domain, then binds resolved kernel arguments against FP admissibility.
    A target legal for one method can therefore be absent for another; no
    value is clamped or run out of domain.

    The deterministic result contains immutable evidence-bound plans and is
    de-duplicated after integer-valued SMA/KAMA mappings. It is intended for
    research batch enumeration, not production admission.
    """
    if (isinstance(natural_time_scale, bool)
            or not isinstance(natural_time_scale, numbers.Real)
            or not math.isfinite(natural_time_scale)
            or not 1 <= natural_time_scale <= 10000):
        raise ValueError("TRAIN natural_time_scale must be finite and within [1, 10000] bars")
    if isinstance(target_time_scales, (str, bytes)):
        raise ValueError("target_time_scales must be an iterable of positive finite bars")

    targets = []
    for target in target_time_scales:
        if (isinstance(target, bool) or not isinstance(target, numbers.Real)
                or not math.isfinite(target) or target <= 0):
            raise ValueError("each target_time_scale must be a positive finite number")
        targets.append(float(target))

    plans = []
    seen = set()
    for method in ("SMA", "EWMA", "IIR", "KAMA", "Kalman"):
        for target in targets:
            relative = target / float(natural_time_scale)
            if not 0.1 <= relative <= 2.0:
                continue
            try:
                plan = compile_smoothing_repair(
                    "CAUSAL_SMOOTHING",
                    {"method": method, "natural_time_scale_relative": relative},
                    natural_time_scale=natural_time_scale,
                    training_context_ref=training_context_ref,
                )
            except IneligibleSmoothingRepair:
                continue
            if plan.identity not in seen:
                seen.add(plan.identity)
                plans.append(plan)
    return tuple(plans)
