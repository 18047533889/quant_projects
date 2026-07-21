# -*- coding: utf-8
"""Production 白名单按调用签名（非仅算子名）。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from planner.logical_plan import PlanNode

SignatureStatus = Literal["production", "pending", "forbidden"]


@dataclass(frozen=True)
class ParamConstraint:
    name: str
    constraint: str
    status: SignatureStatus = "production"
    input_index: int | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class OperatorProductionSignature:
    canonical: str
    params: tuple[ParamConstraint, ...] = ()
    default_status: SignatureStatus = "pending"


def _c(
    name: str,
    constraint: str,
    input_index: int | None = None,
    *,
    choices: tuple[Any, ...] = (),
) -> ParamConstraint:
    return ParamConstraint(name, constraint, input_index=input_index, choices=choices)


_TS_BASE = (
    _c("min_periods", "1..window"),
    _c("null_policy", "enum", choices=("propagate", "ignore")),
    _c("nan_policy", "enum", choices=("propagate", "ignore")),
    _c("includes_current_bar", "enum", choices=(True,)),
)
_TS_WINDOW_UNARY = frozenset({
    "ts_max", "ts_mean", "ts_median", "ts_min", "ts_rank", "ts_std",
    "ts_sum", "ts_var", "ts_zscore",
})
_NO_PARAM_DAILY = frozenset({
    "abs", "add", "and_", "ceil", "coalesce", "cs_count", "cs_demean",
    "cs_mad", "cs_mad_zscore", "cs_mean", "cs_pct_rank", "cs_std",
    "cs_sum", "divide", "eq", "exp", "floor", "ge", "group_count",
    "group_max", "group_mean", "group_min", "group_neutralize",
    "group_normalize", "group_rank", "group_std", "group_sum", "group_zscore",
    "gt", "inverse", "is_finite", "is_infinite", "is_not_null", "is_null",
    "le", "log", "log_abs", "lt", "maximum", "minimum", "multiply", "ne",
    "neg", "normalize", "not_", "or_", "power", "quarter_from_cumulative",
    "rank", "sign", "signed_log", "signed_sqrt", "sqrt", "subtract", "tanh",
    "ttm_from_cumulative", "ttm_from_quarterly", "where", "yoy_by_period",
    "zscore",
})


def _daily_signatures() -> dict[str, OperatorProductionSignature]:
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    signatures = {
        name: OperatorProductionSignature(name, (), default_status="production")
        for name in _NO_PARAM_DAILY
    }
    for name in _TS_WINDOW_UNARY:
        extra = (_c("ddof", "enum", choices=(0, 1)),) if name in {
            "ts_std", "ts_var", "ts_zscore"
        } else ()
        signatures[name] = OperatorProductionSignature(
            name,
            (_c("window", "positive_integer", 1), *_TS_BASE, *extra),
            default_status="production",
        )
    signatures.update({
        "clip": OperatorProductionSignature("clip", (
            _c("lo", "finite_scalar", 1), _c("hi", "finite_scalar", 2),
        ), default_status="production"),
        "fillna_const": OperatorProductionSignature("fillna_const", (
            _c("value", "finite_scalar", 1),
        ), default_status="production"),
        "group_winsorize": OperatorProductionSignature("group_winsorize", (
            _c("a", "probability", 2),
            _c("interpolation", "enum", choices=("linear",)),
        ), default_status="production"),
        "period_average": OperatorProductionSignature("period_average", (
            _c("periods", "positive_integer", 2), _c("require_consecutive", "boolean", 3),
        ), default_status="production"),
        "period_change": OperatorProductionSignature("period_change", (
            _c("periods", "positive_integer", 2),
            _c("mode", "enum", 3, choices=("absolute", "ratio", "log")),
            _c("require_consecutive", "boolean", 4),
        ), default_status="production"),
        "period_cagr": OperatorProductionSignature("period_cagr", (
            _c("periods", "positive_integer", 2),
            _c("periods_per_year", "positive_integer", 3),
            _c("sign_policy", "enum", 4, choices=("strict", "absolute")),
            _c("require_consecutive", "boolean", 5),
        ), default_status="production"),
        "safe_div_null": OperatorProductionSignature("safe_div_null", (
            _c("epsilon", "finite_nonnegative_scalar", 2),
        ), default_status="production"),
        "ts_autocorr": OperatorProductionSignature("ts_autocorr", (
            _c("window", "positive_integer", 1), _c("lag", "positive_integer", 2),
            *_TS_BASE, _c("ddof", "enum", choices=(0, 1)),
        ), default_status="production"),
        "ts_beta": OperatorProductionSignature("ts_beta", (
            _c("window", "positive_integer", 2), *_TS_BASE,
            _c("ddof", "enum", choices=(0, 1)),
        ), default_status="production"),
        "ts_corr": OperatorProductionSignature("ts_corr", (
            _c("window", "positive_integer", 2), *_TS_BASE,
            _c("ddof", "enum", choices=(0, 1)),
        ), default_status="production"),
        "ts_cov": OperatorProductionSignature("ts_cov", (
            _c("window", "positive_integer", 2), *_TS_BASE,
            _c("ddof", "enum", choices=(0, 1)),
        ), default_status="production"),
        "ts_delay": OperatorProductionSignature("ts_delay", (
            _c("n", "nonnegative_integer", 1),
        ), default_status="production"),
        "ts_delta": OperatorProductionSignature("ts_delta", (
            _c("n", "positive_integer", 1),
        ), default_status="production"),
        "ts_log_return": OperatorProductionSignature("ts_log_return", (
            _c("d", "positive_integer", 1),
        ), default_status="production"),
        "ts_pct": OperatorProductionSignature("ts_pct", (
            _c("d", "positive_integer", 1),
        ), default_status="production"),
        "ts_sharpe": OperatorProductionSignature("ts_sharpe", (
            _c("window", "positive_integer", 1),
            _c("ann_factor", "finite_nonnegative_scalar", 2),
            *_TS_BASE, _c("ddof", "enum", choices=(0, 1)),
        ), default_status="production"),
        "winsorize": OperatorProductionSignature("winsorize", (
            _c("lower", "probability", 1), _c("upper", "probability", 2),
            _c("interpolation", "enum", choices=("linear",)),
        ), default_status="production"),
    })
    missing = set(DAILY_CANONICALS).difference(signatures)
    extra = set(signatures).difference(DAILY_CANONICALS)
    if missing or extra:
        raise RuntimeError(f"daily production signature drift: missing={sorted(missing)} extra={sorted(extra)}")
    return signatures


# Exact production authoring surface.  Extended/research compatibility gates
# live separately and cannot grant production membership.
PRODUCTION_SIGNATURES: dict[str, OperatorProductionSignature] = _daily_signatures()

_COMPATIBILITY_SIGNATURES: dict[str, OperatorProductionSignature] = {
    # Kept only so archived plans receive the precise bounded-fill diagnostic;
    # ffill is not part of the active production authoring surface.
    "ffill": OperatorProductionSignature(
        "ffill",
        (
            _c("limit", "positive_integer", 1),
            _c("max_age", "positive_integer"),
        ),
        default_status="pending",
    ),
    "scale": OperatorProductionSignature(
        "scale",
        (ParamConstraint("to", "finite_scalar", status="production"),),
        default_status="pending",
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
}


def signature_for(canon: str) -> OperatorProductionSignature | None:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PRODUCTION_SIGNATURES.get(name) or _COMPATIBILITY_SIGNATURES.get(name)


def has_production_signature(canon: str) -> bool:
    return signature_for(canon) is not None


def param_allowed(canon: str, param: str, *, value: Any = None) -> SignatureStatus:
    sig = signature_for(canon)
    if sig is None:
        return "pending"
    for pc in sig.params:
        if pc.name == param:
            if value is not None and not _constraint_valid(pc, value):
                return "forbidden"
            return pc.status
    return sig.default_status


def _constraint_valid(pc: ParamConstraint, value: Any) -> bool:
    if pc.constraint in {"positive_integer", "nonnegative_integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)) or int(value) != value:
            return False
        return int(value) > 0 if pc.constraint == "positive_integer" else int(value) >= 0
    if pc.constraint in {"finite_scalar", "finite_nonnegative_scalar", "probability", "finite_0_1"}:
        if isinstance(value, bool):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
        if pc.constraint == "finite_nonnegative_scalar":
            return number >= 0
        if pc.constraint in {"probability", "finite_0_1"}:
            return 0 <= number <= 1
        return True
    if pc.constraint == "boolean":
        return isinstance(value, bool)
    if pc.constraint == "enum":
        return value in pc.choices
    if pc.constraint == "1..window":
        return _constraint_valid(_c(pc.name, "positive_integer"), value)
    return False


def _literal_at(node: PlanNode, index: int) -> Any:
    if index >= len(node.inputs):
        return None
    child = node.inputs[index]
    if child.op != "literal":
        return None
    return child.attrs.get("value")


def _constraint_value(node: PlanNode, constraint: ParamConstraint) -> Any:
    if constraint.name in (node.attrs or {}):
        return node.attrs[constraint.name]
    if constraint.input_index is not None:
        return _literal_at(node, constraint.input_index)
    return None


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
    if not production:
        return True, ""
    sig = signature_for(canon)
    if sig is None:
        return False, f"{canon}: missing production signature"
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
    allowed_attrs = {constraint.name for constraint in sig.params}
    unexpected_attrs = sorted(set(node.attrs or {}).difference(allowed_attrs))
    if unexpected_attrs:
        return False, f"{canon}: undeclared production parameters {unexpected_attrs}"
    values: dict[str, Any] = {}
    for constraint in sig.params:
        value = _constraint_value(node, constraint)
        if value is None:
            continue
        values[constraint.name] = value
        if not _constraint_valid(constraint, value):
            return False, (
                f"{canon}: invalid {constraint.name}={value!r} "
                f"for {constraint.constraint}"
            )
    window = values.get("window")
    min_periods = values.get("min_periods")
    if canon in {"ts_sharpe", "ts_autocorr"} and window is not None and int(window) < 2:
        return False, f"{canon}: window must be >= 2"
    if window is not None and min_periods is not None and int(min_periods) > int(window):
        return False, f"{canon}: min_periods must satisfy 1 <= min_periods <= window"
    if canon == "ts_autocorr":
        lag = values.get("lag")
        if window is not None and lag is not None and int(lag) >= int(window):
            return False, f"{canon}: lag must satisfy 1 <= lag < window"
    if canon in {"clip", "winsorize"}:
        lower_name, upper_name = ("lo", "hi") if canon == "clip" else ("lower", "upper")
        lower, upper = values.get(lower_name), values.get(upper_name)
        if lower is not None and upper is not None and float(lower) > float(upper):
            return False, f"{canon}: {lower_name} must be <= {upper_name}"
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

    try:
        canonical = OperatorRegistry.resolve_canonical_strict(canon)
    except KeyError:
        return False
    if canonical not in OperatorRegistry._operators:
        return False
    ok, _ = verify_production_signature(canon, node, production=True)
    return ok
