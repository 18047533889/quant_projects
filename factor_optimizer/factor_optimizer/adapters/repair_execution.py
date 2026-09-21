"""Versioned, research-only execution plans for value-level repair families."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from typing import Mapping, Tuple

import pandas as pd

from factor_optimizer.search.conditional_search import build_conditional_tree


class IneligibleValueRepair(ValueError):
    """A valid registry candidate has no exact executable value primitive."""


def _validate_frame(values) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values must be a pandas DataFrame")
    missing = {"asset_id", "date", "value"} - set(values.columns)
    if missing:
        raise ValueError(f"long factor frame missing columns: {sorted(missing)}")
    if values[["asset_id", "date"]].isna().any().any():
        raise ValueError("asset_id/date identity columns cannot contain nulls")
    if values.duplicated(["date", "asset_id"]).any():
        raise ValueError("duplicate (date, asset_id) identities are ambiguous")
    return values


@dataclass(frozen=True)
class ValueRepairPlan:
    family: str
    transform: str
    parameters: Tuple[Tuple[str, object], ...]
    training_context_ref: str
    natural_time_scale: float
    mapping_version: str = "value-repair.v3"

    @property
    def identity(self) -> str:
        payload = {
            "family": self.family, "transform": self.transform,
            "parameters": self.parameters,
            "training_context_ref": self.training_context_ref,
            "natural_time_scale": self.natural_time_scale,
            "version": self.mapping_version,
        }
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def execute(self, values, *, allow_research: bool = False) -> pd.Series:
        if allow_research is not True:
            raise ValueError("this adapter is research-only; explicit allow_research=True required")
        frame = _validate_frame(values)
        kwargs = dict(self.parameters)
        if self.transform == "raw":
            return frame["value"].copy()
        if self.transform == "sign":
            return (frame["value"] * kwargs["multiplier"]).rename("value")
        if self.transform == "rank_shape":
            from factor_preprocess.transforms.repair_shapes import rank_shape
            return rank_shape(frame, **kwargs)
        if self.transform == "cs_rank":
            from factor_preprocess.transforms.repair_shapes import cross_sectional_rank
            return cross_sectional_rank(frame, **kwargs)
        if self.transform in {"ts_rank_history", "ts_zscore_history"}:
            from factor_preprocess.transforms.temporal_representation import (
                time_series_rank, capped_time_series_zscore,
            )
            function = time_series_rank if self.transform == "ts_rank_history" else capped_time_series_zscore
            return function(frame, **kwargs)
        if self.transform == "capped_zscore":
            from factor_preprocess.transforms.repair_shapes import capped_zscore
            return capped_zscore(frame, **kwargs)
        if self.transform == "tail_hinge":
            from factor_preprocess.transforms.repair_shapes import tail_hinge
            return tail_hinge(frame, **kwargs)
        if self.transform == "tail_saturation":
            from factor_preprocess.transforms.repair_shapes import tail_saturation
            return tail_saturation(frame, **kwargs)
        if self.transform == "robust_scale":
            from factor_preprocess.transforms.repair_shapes import robust_scale
            return robust_scale(frame, **kwargs)
        from factor_preprocess.registry.transforms import get_default_registry
        return get_default_registry().get_execution(self.transform)(frame, **kwargs)


def _plan(family, transform, kwargs, scale, context):
    return ValueRepairPlan(family, transform, tuple(sorted(kwargs.items())), context, float(scale))


def compile_value_repair(family: str, parameters: Mapping[str, object], *,
                         natural_time_scale: float,
                         training_context_ref: str) -> ValueRepairPlan:
    if (isinstance(natural_time_scale, bool)
            or not isinstance(natural_time_scale, numbers.Real)
            or not math.isfinite(natural_time_scale)
            or not 1 <= natural_time_scale <= 10000):
        raise ValueError("TRAIN natural_time_scale must be finite and within [1, 10000] bars")
    if not isinstance(training_context_ref, str) or not training_context_ref.strip():
        raise ValueError("training_context_ref is required")
    params = dict(parameters)
    for name, value in params.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    tree = build_conditional_tree([family])[0]
    if family in {"REPRESENTATION_RANK", "REPRESENTATION_ZSCORE"} and "window" not in params:
        axis = params.get("rank_axis", params.get("zscore_axis"))
        if axis == "ts":
            raise IneligibleValueRepair("time-series representation requires an explicit window")
        # Preserve existing CS calls: this field is inactive and not included
        # in the resolved stateless plan identity.
        params["window"] = int(next(p.prior for p in tree.parameters if p.name == "window"))
    tree.validate_params(params)

    if family == "NO_OP_RAW":
        if params["keep_raw"] is not True:
            raise ValueError("NO_OP_RAW requires keep_raw=True")
        return _plan(family, "raw", {}, natural_time_scale, training_context_ref)
    if family == "SIGN_ORIENTATION":
        multiplier = -1.0 if params["direction"] == "flip" else 1.0
        return _plan(family, "sign", {"multiplier": multiplier}, natural_time_scale, training_context_ref)
    if family in {"CAUSAL_SMOOTHING", "DECAY_REFINEMENT"}:
        from factor_optimizer.adapters.preprocessing import (
            IneligibleSmoothingRepair, compile_smoothing_repair,
        )
        try:
            smoothing = compile_smoothing_repair(
                family, params, natural_time_scale=natural_time_scale,
                training_context_ref=training_context_ref,
            )
        except IneligibleSmoothingRepair as exc:
            raise IneligibleValueRepair(str(exc)) from exc
        return _plan(family, smoothing.transform, dict(smoothing.parameters),
                     natural_time_scale, training_context_ref)
    if family in {"U_SHAPE_REPAIR", "INVERTED_U_REPAIR"}:
        return _plan(family, "rank_shape", {
            "center": float(params["center"]), "power": float(params["power"]),
            "inverted": family == "INVERTED_U_REPAIR",
            "asymmetric": params["asymmetry"],
        }, natural_time_scale, training_context_ref)
    if family == "ROBUST_OUTLIER":
        if params["lower_quantile"] >= params["upper_quantile"]:
            raise ValueError("lower_quantile must be less than upper_quantile")
        return _plan(family, "cs_winsor", {
            "lower": float(params["lower_quantile"]),
            "upper": float(params["upper_quantile"]),
        }, natural_time_scale, training_context_ref)
    if family == "MISSINGNESS_FRESHNESS":
        mode = params["mode"]
        if mode == "fill":
            return _plan(family, "forward_fill", {"max_lag": int(params["freshness_window"])},
                         natural_time_scale, training_context_ref)
        if mode == "flag":
            return _plan(family, "missing_indicator", {}, natural_time_scale, training_context_ref)
        raise IneligibleValueRepair("drop mode is row-selection, not a value Series primitive")
    if family == "REPRESENTATION_RANK":
        if params["rank_axis"] == "ts":
            return _plan(family, "ts_rank_history",
                         {"method": params["tie_method"], "window": int(params["window"])},
                         natural_time_scale, training_context_ref)
        return _plan(family, "cs_rank", {"method": params["tie_method"]},
                     natural_time_scale, training_context_ref)
    if family == "REPRESENTATION_ZSCORE":
        if params["zscore_axis"] == "ts":
            return _plan(family, "ts_zscore_history",
                         {"cap": float(params["cap"]), "window": int(params["window"])},
                         natural_time_scale, training_context_ref)
        return _plan(family, "capped_zscore", {"cap": float(params["cap"])},
                     natural_time_scale, training_context_ref)
    if family == "TAIL_SATURATION":
        q = float(params["saturation_quantile"]); side = params["saturate"]
        return _plan(family, "tail_saturation", {"quantile": q, "side": side},
                     natural_time_scale, training_context_ref)
    if family == "TAIL_HINGE":
        return _plan(family, "tail_hinge", {
            "hinge": params["hinge"], "hinge_value": float(params["hinge_value"]),
        }, natural_time_scale, training_context_ref)
    if family == "ROBUST_SCALE":
        return _plan(family, "robust_scale", params, natural_time_scale, training_context_ref)
    if family in {"INDUSTRY_NEUTRALIZATION", "SIZE_NEUTRALIZATION", "STYLE_NEUTRALIZATION"}:
        raise IneligibleValueRepair(f"{family} requires an external exposure panel")
    if family in {"WINDOW_REFINEMENT", "OPERATOR_SWAP", "LOW_DOF_INTERACTION"}:
        raise IneligibleValueRepair(f"{family} requires FactorEngine DSL recompilation")
    raise IneligibleValueRepair(f"{family} has no executable value-level mapping")
