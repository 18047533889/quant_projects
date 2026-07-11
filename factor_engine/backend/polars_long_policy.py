# -*- coding: utf-8
"""Polars long-table 能力分层与 strict fallback 策略。"""
from __future__ import annotations

import os

from planner.logical_plan import PlanNode

from .polars_registry_bridge import registry_op_long_capable

# 纯 Polars Expr（无 Python rolling_map / map_groups / pandas kernel）
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
        "safe_div_null",
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
        "div_or_default",
        "log_fill_invalid",
        "div_or_null",
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
        "is_infinite",
        "is_nan",
        "is_null",
        "is_not_null",
        "winsorize",
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
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
        "ts_ratio",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "group_decay_linear",
        "cum_delta",
        "expanding_mean",
        "expanding_sum",
        "count",
    }
)

POLARS_LONG_STATEFUL: frozenset[str] = frozenset(
    {
        "RSI_WILDER",
        "ATR_WILDER",
    }
)

# 算法语义未冻结 / 非标准实现：仍可走 long path，但不得标为 native production
POLARS_LONG_NONSTANDARD_ALG: frozenset[str] = frozenset(
    {
        "ts_mad",
        "ts_regression",
    }
)

# rolling_map + NumPy/pandas callback（LazyFrame 内仍含 Python UDF）
POLARS_LONG_PYTHON_ROLLING: frozenset[str] = frozenset(
    {
        "expanding_std",
        "ts_decay_linear",
        "WMA",
        "Slope",
        "ts_skew",
        "ts_quantile",
        "ts_argmax",
        "ts_argmin",
        "ts_product",
        "ts_median_abs_deviation",
        "ts_mean_abs_deviation",
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

# 因果填充算子：禁止 polars_long / SQL fast path silent no-op（见 ``UnsupportedCausalOperatorError``）
POLARS_LONG_BLOCKED_CAUSAL: frozenset[str] = frozenset({"bfill", "causal_bfill"})

POLARS_LONG_PASSTHROUGH: frozenset[str] = frozenset()

POLARS_LONG_COMPATIBLE: frozenset[str] = (
    POLARS_LONG_NATIVE
    | POLARS_LONG_NONSTANDARD_ALG
    | POLARS_LONG_STATEFUL
    | POLARS_LONG_PYTHON_ROLLING
    | POLARS_LONG_MAP_GROUPS
    | POLARS_LONG_PASSTHROUGH
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
    """将算子别名解析为 canonical 名称。"""
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def classify_plan_op(op: str) -> str:
    """返回 ``native`` | ``python_rolling`` | ``map_groups`` | ``passthrough`` | ``registry`` | ``other`` | ``meta``。"""
    from backend.production_fastpath_tiers import resolve_polars_native_canonical

    canon = resolve_polars_native_canonical(_resolve(op))
    if canon in {"column", "literal", "materialized_series", "plan_ref"}:
        return "meta"
    if canon in POLARS_LONG_BLOCKED_CAUSAL:
        return "blocked_causal"
    if canon in POLARS_LONG_NONSTANDARD_ALG:
        return "nonstandard_alg"
    if canon in POLARS_LONG_STATEFUL:
        return "stateful"
    if canon in POLARS_LONG_PASSTHROUGH:
        return "passthrough"
    if canon in POLARS_LONG_NATIVE:
        return "native"
    if canon in POLARS_LONG_PYTHON_ROLLING:
        return "python_rolling"
    if canon in POLARS_LONG_MAP_GROUPS:
        return "map_groups"
    if registry_op_long_capable(canon):
        return "registry"
    return "other"


def infer_polars_long_tier(op: str) -> str:
    """算子级 long-table 能力 tier。"""
    kind = classify_plan_op(op)
    if kind == "meta":
        return "meta"
    if kind == "blocked_causal":
        return "blocked_causal"
    if kind in {"native", "nonstandard_alg", "stateful", "python_rolling", "map_groups", "passthrough", "registry"}:
        return kind
    return "unsupported"


def collect_plan_op_stats(plan: PlanNode) -> dict[str, list[str]]:
    """遍历逻辑计划，按 PolarsLong tier 分类算子。

    参数:
        plan: 逻辑计划根节点。

    返回:
        含 ``polars_long_native_ops`` 等各 tier 算子列表的字典。
    """
    native: set[str] = set()
    nonstandard: set[str] = set()
    stateful: set[str] = set()
    python_rolling: set[str] = set()
    map_groups: set[str] = set()
    passthrough: set[str] = set()
    blocked_causal: set[str] = set()
    registry: set[str] = set()
    other: set[str] = set()

    def _walk(node: PlanNode) -> None:
        """递归遍历计划并按 tier 分类算子。"""
        canon = _resolve(node.op)
        kind = classify_plan_op(canon)
        if kind == "native" and canon not in {"column", "literal", "materialized_series", "plan_ref"}:
            native.add(canon)
        elif kind == "nonstandard_alg":
            nonstandard.add(canon)
        elif kind == "stateful":
            stateful.add(canon)
        elif kind == "python_rolling":
            python_rolling.add(canon)
        elif kind == "map_groups":
            map_groups.add(canon)
        elif kind == "passthrough":
            passthrough.add(canon)
        elif kind == "blocked_causal":
            blocked_causal.add(canon)
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
        "polars_long_nonstandard_alg_ops": sorted(nonstandard),
        "polars_long_stateful_ops": sorted(stateful),
        "polars_long_python_rolling_ops": sorted(python_rolling),
        "polars_long_map_group_ops": sorted(map_groups),
        "polars_long_passthrough_ops": sorted(passthrough),
        "polars_long_blocked_causal_ops": sorted(blocked_causal),
        "polars_long_registry_ops": sorted(registry),
        "polars_long_other_ops": sorted(other),
    }


def long_path_telemetry_flags(op_stats: dict[str, list[str]]) -> dict[str, bool]:
    """从 ``collect_plan_op_stats`` 推断 runtime 布尔标记。"""
    has_map = bool(op_stats.get("polars_long_map_group_ops"))
    has_registry = bool(op_stats.get("polars_long_registry_ops"))
    has_other = bool(op_stats.get("polars_long_other_ops"))
    has_passthrough = bool(op_stats.get("polars_long_passthrough_ops"))
    has_python_rolling = bool(op_stats.get("polars_long_python_rolling_ops"))
    has_native = bool(op_stats.get("polars_long_native_ops"))
    fast_native = has_native and not (
        has_map or has_registry or has_other or has_passthrough or has_python_rolling
    )
    return {
        "used_polars_long_path": True,
        "used_polars_long_native": fast_native,
        "used_polars_long_python_rolling": has_python_rolling,
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


class UnsupportedCausalOperatorError(PolarsLongStrictError):
    """bfill / causal_bfill 等在 fast path 禁止 silent no-op。"""


class PolarsLongNativeRequiredError(PolarsLongStrictError):
    """``FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE=1`` 时 plan 含非 native 算子。"""


def require_polars_long_native_only(ctx=None) -> bool:
    """仅允许 ``POLARS_LONG_NATIVE`` + meta；禁止 map_groups / registry / passthrough。"""
    raw = os.environ.get("FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def plan_contains_blocked_causal(plan) -> bool:
    """计划树是否含 ``bfill`` / ``causal_bfill``。"""
    from planner.logical_plan import PlanNode

    if not isinstance(plan, PlanNode):
        return False
    canon = _resolve(str(plan.op))
    if canon in POLARS_LONG_BLOCKED_CAUSAL:
        return True
    return any(plan_contains_blocked_causal(c) for c in plan.inputs)


def assert_no_blocked_causal_plan(plan, *, backend: str) -> None:
    """fast path 遇 blocked causal 算子时立即失败。"""
    if plan_contains_blocked_causal(plan):
        raise UnsupportedCausalOperatorError(
            f"{backend} 不支持 bfill/causal_bfill（因果占位，非传统 backward fill）；请改用 ffill 或 pandas 研究路径"
        )


def assert_native_only_plan(op_stats: dict[str, list[str]], ctx=None) -> None:
    """REQUIRE_NATIVE 模式下，plan 不得含 map_groups / registry / passthrough。"""
    if not require_polars_long_native_only(ctx):
        return
    blocked: list[str] = []
    for key, label in (
        ("polars_long_map_group_ops", "map_groups"),
        ("polars_long_python_rolling_ops", "python_rolling"),
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
