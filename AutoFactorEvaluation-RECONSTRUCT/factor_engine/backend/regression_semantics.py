# -*- coding: utf-8
"""Regression 数值稳定性与 production 分层契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OlsAlgorithm = Literal["mean_formula", "centered_two_pass", "online_stable"]
ProductionTier = Literal["certified_low_risk", "deferred_unstable_kernel"]


@dataclass(frozen=True)
class RegressionSpec:
    algorithm: OlsAlgorithm = "mean_formula"
    pairwise_missing: bool = True
    zero_variance_x_is_null: bool = True
    production_tier: ProductionTier = "deferred_unstable_kernel"


REGRESSION_SPECS: dict[str, RegressionSpec] = {
    "cs_resid": RegressionSpec(production_tier="certified_low_risk"),
    "cs_regression": RegressionSpec(production_tier="certified_low_risk"),
    "ts_regression": RegressionSpec(production_tier="deferred_unstable_kernel"),
    "Slope": RegressionSpec(production_tier="deferred_unstable_kernel"),
}


def regression_spec_for(canon: str) -> RegressionSpec:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return REGRESSION_SPECS.get(name, RegressionSpec())


def ts_regression_deferred() -> bool:
    return regression_spec_for("ts_regression").production_tier == "deferred_unstable_kernel"
