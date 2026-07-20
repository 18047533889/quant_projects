# -*- coding: utf-8
"""Production 白名单按调用签名（非仅算子名）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from planner.logical_plan import PlanNode

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


# 第一阶段已锁定的签名子集（完整六证仍须 primitive_verified artifact）
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
    "digital_count": OperatorProductionSignature(
        "digital_count",
        (
            ParamConstraint("d", "positive_integer"),
            ParamConstraint("threshold", "finite_nonnegative_scalar"),
            ParamConstraint("run", "positive_integer"),
        ),
        default_status="production",
    ),
    "price_spread_deviation": OperatorProductionSignature(
        "price_spread_deviation",
        (ParamConstraint("d", "positive_integer"),),
        default_status="production",
    ),
    "saturate": OperatorProductionSignature(
        "saturate",
        (
            ParamConstraint("lower", "finite_scalar"),
            ParamConstraint("upper", "finite_scalar"),
        ),
        default_status="production",
    ),
    "signed_power": OperatorProductionSignature(
        "signed_power",
        (ParamConstraint("c", "finite_scalar"),),
        default_status="production",
    ),
    "ts_decay_exp_window": OperatorProductionSignature(
        "ts_decay_exp_window",
        (
            ParamConstraint("window", "positive_integer"),
            ParamConstraint("alpha", "finite_0_1"),
        ),
        default_status="production",
    ),
    "ts_sum_decay": OperatorProductionSignature(
        "ts_sum_decay",
        (ParamConstraint("window", "positive_integer"),),
        default_status="production",
    ),
    "ts_moment": OperatorProductionSignature(
        "ts_moment",
        (
            ParamConstraint("d", "positive_integer"),
            ParamConstraint("k", "positive_integer"),
        ),
        default_status="production",
    ),
    "ts_ratio": OperatorProductionSignature(
        "ts_ratio",
        default_status="production",
    ),
    "ts_max_buildup": OperatorProductionSignature(
        "ts_max_buildup",
        (ParamConstraint("d", "positive_integer"),),
        default_status="production",
    ),
    "trade_when": OperatorProductionSignature(
        "trade_when",
        default_status="pending",
    ),
}


def signature_for(canon: str) -> OperatorProductionSignature | None:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PRODUCTION_SIGNATURES.get(name)


def has_production_signature(canon: str) -> bool:
    return signature_for(canon) is not None


def param_allowed(canon: str, param: str, *, value: Any = None) -> SignatureStatus:
    sig = signature_for(canon)
    if sig is None:
        return "pending"
    for pc in sig.params:
        if pc.name == param:
            if pc.constraint == "finite_0_1" and value is not None:
                try:
                    v = float(value)
                    if not 0.0 <= v <= 1.0:
                        return "forbidden"
                except (TypeError, ValueError):
                    return "forbidden"
            if pc.constraint == "finite_nonnegative_scalar" and value is not None:
                try:
                    v = float(value)
                    if v < 0 or v != v or v in (float("inf"), float("-inf")):
                        return "forbidden"
                except (TypeError, ValueError):
                    return "forbidden"
            if pc.constraint == "positive_integer" and value is not None:
                try:
                    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                        return "forbidden"
                except (TypeError, ValueError):
                    return "forbidden"
            if pc.constraint == "finite_scalar" and value is not None:
                try:
                    v = float(value)
                    if v != v or v in (float("inf"), float("-inf")):
                        return "forbidden"
                except (TypeError, ValueError):
                    return "forbidden"
            return pc.status
    return sig.default_status


def _literal_at(node: PlanNode, index: int) -> Any:
    if index >= len(node.inputs):
        return None
    child = node.inputs[index]
    if child.op != "literal":
        return None
    return child.attrs.get("value")


def _ffill_has_limit(node: PlanNode) -> bool:
    attrs = node.attrs or {}
    limit = attrs.get("limit")
    max_age = attrs.get("max_age")
    if limit is None:
        limit = _literal_at(node, 1)
    if max_age is None:
        max_age = attrs.get("max_age_days") or attrs.get("max_age_bars")
    if limit is not None:
        try:
            return int(limit) > 0
        except (TypeError, ValueError):
            return False
    return max_age is not None


def verify_production_signature(canon: str, node: PlanNode | None, *, production: bool) -> tuple[bool, str]:
    """按 production 签名表校验调用（fail-closed）。"""
    sig = signature_for(canon)
    if sig is None:
        return True, ""
    if not production:
        return True, ""
    if sig.default_status == "forbidden":
        return False, f"{canon}: production forbidden"
    if sig.default_status == "pending":
        if canon == "ffill":
            if node is None:
                return False, "ffill: 需要 plan 节点解析 limit/max_age"
            if _ffill_has_limit(node):
                return True, ""
            return False, "ffill(x) unlimited forbidden in production; use limit=N or max_age"
        if canon == "scale":
            if node is None:
                return False, "scale: pending except scale(to=1)"
            to_val = _literal_at(node, 1)
            if to_val is None:
                to_val = (node.attrs or {}).get("to", 1.0)
            try:
                if float(to_val) == 1.0:
                    return True, ""
            except (TypeError, ValueError):
                pass
            return False, f"{canon}: pending production signature"
        if node is None:
            return False, f"{canon}: pending production signature"
        return False, f"{canon}: pending production signature"
    if node is None:
        return True, ""
    if canon == "scale":
        to_val = _literal_at(node, 1)
        if to_val is None:
            to_val = (node.attrs or {}).get("to", 1.0)
        try:
            to_f = float(to_val)
        except (TypeError, ValueError):
            return False, f"scale(to={to_val!r}) 非法"
        if to_f != 1.0 and param_allowed("scale", "to", value=to_f) != "production":
            return False, f"scale(to={to_f}) 未认证"
    return True, ""


def operational_production_allowed(canon: str, node: PlanNode | None = None) -> bool:
    """算子级 operational production（不含 backend-specific 检查）。"""
    from cleaned_operators.registry import OperatorRegistry

    canonical = OperatorRegistry.resolve_canonical_optional(canon)
    if canonical not in OperatorRegistry._operators:
        return False
    ok, _ = verify_production_signature(canon, node, production=True)
    return ok
