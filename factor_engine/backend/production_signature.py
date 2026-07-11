# -*- coding: utf-8
"""Production 白名单按调用签名（非仅算子名）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SignatureStatus = Literal["production", "pending", "forbidden"]


@dataclass(frozen=True)
class ParamConstraint:
    name: str
    constraint: str
    status: SignatureStatus = "production"


@dataclass(frozen=True)
class OperatorProductionSignature:
    canonical: str
    params: tuple[ParamConstraint, ...] = ()
    default_status: SignatureStatus = "pending"


# 第一阶段已锁定的签名子集（完整六证仍须 primitive_verified 交集）
PRODUCTION_SIGNATURES: dict[str, OperatorProductionSignature] = {
    "ts_mean": OperatorProductionSignature(
        "ts_mean",
        (
            ParamConstraint("window", "positive_integer"),
            ParamConstraint("min_periods", "1..window"),
        ),
        default_status="production",
    ),
    "scale": OperatorProductionSignature(
        "scale",
        (ParamConstraint("to", "finite_scalar", status="production"),),
        default_status="pending",
    ),
    "fillna_const": OperatorProductionSignature(
        "fillna_const",
        (ParamConstraint("value", "finite_scalar"),),
        default_status="production",
    ),
    "bfill": OperatorProductionSignature(
        "bfill",
        default_status="forbidden",
    ),
    "cs_regression": OperatorProductionSignature(
        "cs_regression",
        (ParamConstraint("mode", "0|1|2"),),
        default_status="production",
    ),
    "ffill": OperatorProductionSignature(
        "ffill",
        default_status="pending",  # unlimited ffill research-only until limit=N
    ),
}


def signature_for(canon: str) -> OperatorProductionSignature | None:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PRODUCTION_SIGNATURES.get(name)


def param_allowed(canon: str, param: str, *, value: Any = None) -> SignatureStatus:
    sig = signature_for(canon)
    if sig is None:
        return "pending"
    for pc in sig.params:
        if pc.name == param:
            if pc.constraint == "finite_scalar" and value is not None:
                try:
                    v = float(value)
                    if v != v or v in (float("inf"), float("-inf")):
                        return "forbidden"
                except (TypeError, ValueError):
                    return "forbidden"
            return pc.status
    return sig.default_status
