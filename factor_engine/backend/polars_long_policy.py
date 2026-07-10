# -*- coding: utf-8
"""Polars long-table 能力分层与 strict fallback 策略。"""
from __future__ import annotations

import os

from planner.logical_plan import PlanNode

from .polars_registry_bridge import registry_op_long_capable

# 纯 Polars Expr（无 Python map_groups / pandas kernel）
POLARS_LONG_NATIVE: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "materialized_series",
        "plan_ref",
        "add",
        "subtract",
        "multiply",
        "divide",
        "neg",
        "abs",
        "sign",
        "log",
        "exp",
        "sqrt",
        "clip",
        "power",
        "floor",
        "ceil",
        "protected_div",
        "protected_log",
        "protected_sqrt",
        "inverse",
        "maximum",
        "minimum",
        "ts_delay",
        "ts_delta",
        "ts_pct",
        "ts_mean",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_std",
        "ts_var",
        "ts_median",
        "ts_zscore",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "ts_ema",
        "ts_rank",
        "ts_decay_linear",
        "ts_sharpe",
        "ts_autocorr",
        "rank",
        "rank_pct",
        "zscore",
        "cs_demean",
        "scale",
        "normalize",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "vwap",
        "volatility",
        "log_returns",
        "ffill",
        "cum_sum",
        "cum_max",
        "cum_min",
        "cum_prod",
        "ewm_mean",
        "ewm_std",
        "ewm_var",
        "coalesce",
        "where",
        "if_else",
        "fillna_const",
        "fillna",
        "nan_to_num",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "not_",
        "is_finite",
        "is_nan",
        "winsorize",
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_std",
        "group_normalize",
        "group_percentile",
        "group_winsorize",
        "cs_mad",
        "cs_mad_zscore",
        "cs_resid",
        "cs_regression",
        "RSI_WILDER",
        "ATR_WILDER",
        "rolling_beta",
        "ts_argmax",
        "ts_argmin",
        "WMA",
        "ts_mad",
        "ts_quantile",
        "ts_product",
        "ts_skew",
        "ts_regression",
        "Slope",
        "ts_ratio",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "group_decay_linear",
        "cum_delta",
        "expanding_mean",
        "expanding_std",
        "expanding_sum",
        "count",
    }
)

# map_groups / rolling_map(pandas) / 截面 Python 分箱等
POLARS_LONG_MAP_GROUPS: frozenset[str] = frozenset(
    {
        "ewm_corr",
        "ewm_cov",
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "expanding_rank",
        "fillna_interpolate",
        "quantile",
    }
)

# 因果透传（与 SQL 一致：不引用未来值，long path 直接 passthrough inner）
POLARS_LONG_PASSTHROUGH: frozenset[str] = frozenset({"bfill", "causal_bfill"})

POLARS_LONG_COMPATIBLE: frozenset[str] = (
    POLARS_LONG_NATIVE | POLARS_LONG_MAP_GROUPS | POLARS_LONG_PASSTHROUGH
)

# 向后兼容旧名
POLARS_EXPR_CAPABLE = POLARS_LONG_COMPATIBLE

_POLARS_LONG_CAPABLE_CACHE: frozenset[str] | None = None


def get_polars_long_capable() -> frozenset[str]:
    """``POLARS_LONG_COMPATIBLE`` ∪ Registry polars bridge。"""
    global _POLARS_LONG_CAPABLE_CACHE
    if _POLARS_LONG_CAPABLE_CACHE is not None:
        return _POLARS_LONG_CAPABLE_CACHE
    from cleaned_operators import load_all

    load_all()
    from .polars_registry_bridge import polars_registry_long_capable

    _POLARS_LONG_CAPABLE_CACHE = frozenset(
        POLARS_LONG_COMPATIBLE | polars_registry_long_capable()
    )
    return _POLARS_LONG_CAPABLE_CACHE


POLARS_LONG_CAPABLE: frozenset[str] = POLARS_LONG_COMPATIBLE


def _resolve(op: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def classify_plan_op(op: str) -> str:
    """返回 ``native`` | ``map_groups`` | ``passthrough`` | ``registry`` | ``other`` | ``meta``。"""
    from backend.production_fastpath_tiers import resolve_polars_native_canonical

    canon = resolve_polars_native_canonical(_resolve(op))
    if canon in {"column", "literal", "materialized_series", "plan_ref"}:
        return "meta"
    if canon in POLARS_LONG_PASSTHROUGH:
        return "passthrough"
    if canon in POLARS_LONG_NATIVE:
        return "native"
    if canon in POLARS_LONG_MAP_GROUPS:
        return "map_groups"
    if registry_op_long_capable(canon):
        return "registry"
    return "other"


def infer_polars_long_tier(op: str) -> str:
    """算子级 long-table 能力：``native`` | ``map_groups`` | ``passthrough`` | ``registry`` | ``unsupported``。"""
    kind = classify_plan_op(op)
    if kind == "meta":
        return "meta"
    if kind in {"native", "map_groups", "passthrough", "registry"}:
        return kind
    return "unsupported"


def collect_plan_op_stats(plan: PlanNode) -> dict[str, list[str]]:
    native: set[str] = set()
    map_groups: set[str] = set()
    passthrough: set[str] = set()
    registry: set[str] = set()
    other: set[str] = set()

    def _walk(node: PlanNode) -> None:
        canon = _resolve(node.op)
        kind = classify_plan_op(canon)
        if kind == "native" and canon not in {"column", "literal", "materialized_series", "plan_ref"}:
            native.add(canon)
        elif kind == "map_groups":
            map_groups.add(canon)
        elif kind == "passthrough":
            passthrough.add(canon)
        elif kind == "registry":
            registry.add(canon)
        elif kind == "other" and canon not in {
            "column",
            "literal",
            "materialized_series",
            "plan_ref",
        }:
            other.add(canon)
        for child in node.inputs:
            _walk(child)

    _walk(plan)
    return {
        "polars_long_native_ops": sorted(native),
        "polars_long_map_group_ops": sorted(map_groups),
        "polars_long_passthrough_ops": sorted(passthrough),
        "polars_long_registry_ops": sorted(registry),
        "polars_long_other_ops": sorted(other),
    }


def long_path_telemetry_flags(op_stats: dict[str, list[str]]) -> dict[str, bool]:
    """从 ``collect_plan_op_stats`` 推断 runtime 布尔标记。"""
    has_map = bool(op_stats.get("polars_long_map_group_ops"))
    has_registry = bool(op_stats.get("polars_long_registry_ops"))
    has_other = bool(op_stats.get("polars_long_other_ops"))
    has_passthrough = bool(op_stats.get("polars_long_passthrough_ops"))
    has_native = bool(op_stats.get("polars_long_native_ops"))
    fast_native = has_native and not (has_map or has_registry or has_other or has_passthrough)
    # 仅 native / meta 节点（无 map_groups/registry/passthrough/other 算子）
    if not has_native and not has_map and not has_registry and not has_other and not has_passthrough:
        fast_native = True
    return {
        "used_polars_long_path": True,
        "used_polars_long_native": fast_native,
        "used_polars_long_map_groups": has_map,
        "used_polars_long_registry": has_registry,
        "used_polars_long_passthrough": has_passthrough,
    }


def strict_polars_long_fallback(ctx=None) -> bool:
    """production 或 ``FACTOR_ENGINE_STRICT_POLARS_LONG=1`` 时禁止 silent fallback。"""
    raw = os.environ.get("FACTOR_ENGINE_STRICT_POLARS_LONG", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    runtime = getattr(ctx, "runtime_stats", None) if ctx is not None else None
    if isinstance(runtime, dict):
        mode = runtime.get("run_mode")
        if mode is not None:
            return str(mode).lower() == "production"
    from runtime.production_policy import is_production_mode

    return is_production_mode()


class PolarsLongStrictError(RuntimeError):
    """Strict 模式下 long native 失败，禁止 fallback。"""


class PolarsLongNativeRequiredError(PolarsLongStrictError):
    """``FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE=1`` 时 plan 含非 native 算子。"""


def require_polars_long_native_only(ctx=None) -> bool:
    """仅允许 ``POLARS_LONG_NATIVE`` + meta；禁止 map_groups / registry / passthrough。"""
    raw = os.environ.get("FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def assert_native_only_plan(op_stats: dict[str, list[str]], ctx=None) -> None:
    """REQUIRE_NATIVE 模式下，plan 不得含 map_groups / registry / passthrough。"""
    if not require_polars_long_native_only(ctx):
        return
    blocked: list[str] = []
    for key, label in (
        ("polars_long_map_group_ops", "map_groups"),
        ("polars_long_registry_ops", "registry"),
        ("polars_long_passthrough_ops", "passthrough"),
        ("polars_long_other_ops", "other"),
    ):
        ops = op_stats.get(key) or []
        if ops:
            blocked.append(f"{label}: {','.join(ops)}")
    if blocked:
        raise PolarsLongNativeRequiredError(
            "polars_long require_native: plan uses non-native ops — " + "; ".join(blocked)
        )
