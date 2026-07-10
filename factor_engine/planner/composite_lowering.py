# -*- coding: utf-8
"""高级算子 composite lowering：展开为基础 PlanNode DAG。"""
from __future__ import annotations

from collections.abc import Callable

from planner.logical_plan import PlanNode

LoweringFn = Callable[[PlanNode], PlanNode]

_COMPOSITE_LOWERINGS: dict[str, LoweringFn] = {}
_LOWERINGS_LOADED = False

ExecutionKind = str  # primitive | composite | stateful | external_kernel | forbidden

# 需 external kernel / 非 SQL-Polars 可内联的算子（research 或专用 runtime）
EXTERNAL_KERNEL_CANONICALS: frozenset[str] = frozenset(
    {
        "fft",
        "ifft",
        "wavelet",
        "convolve",
        "correlate",
        "mat_inverse",
        "eig",
        "svd",
        "pca",
        "granger_causality",
        "adf",
        "kpss_test",
        "cointegration",
        "rolling_beta_to_market",
        "downside_beta",
        "tail_beta",
        "residual_momentum_capm",
        "coskewness_to_market",
        "idio_vol",
        "idio_skew",
        "rank_corr",
        "ts_poly2_coeff",
        "ts_poly2_resid",
        "digital_count",
    }
)

# 有状态 / Wilder / EWM 递推；暂不进 dual-backend production fastpath
STATEFUL_DEFERRED_CANONICALS: frozenset[str] = frozenset(
    {
        "MACD",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "KAMA",
        "RSI_WILDER",
        "ATR_WILDER",
        "ts_ema",
        "ewm_mean",
        "ewm_std",
        "ewm_var",
        "ewm_cov",
        "ewm_corr",
        "ts_decay_linear",
    }
)


def register_lowering(canonical: str):
    """注册 canonical 算子的 lowering 函数。"""

    def decorator(fn: LoweringFn) -> LoweringFn:
        _COMPOSITE_LOWERINGS[str(canonical)] = fn
        return fn

    return decorator


def _ensure_lowerings_loaded() -> None:
    global _LOWERINGS_LOADED
    if _LOWERINGS_LOADED:
        return
    from planner.lowerings import (  # noqa: F401
        fundamental,
        microstructure,
        technical,
    )

    _ = fundamental, microstructure, technical
    _LOWERINGS_LOADED = True


def list_composite_lowerings() -> frozenset[str]:
    """已注册 composite lowering 的 canonical 集合。"""
    _ensure_lowerings_loaded()
    return frozenset(_COMPOSITE_LOWERINGS)


def has_composite_lowering(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    resolved = OperatorRegistry._aliases.get(canon, canon)
    _ensure_lowerings_loaded()
    return resolved in _COMPOSITE_LOWERINGS


def infer_execution_kind(canon: str) -> str:
    """推断算子 execution_kind（coverage / manifest 用）。"""
    from cleaned_operators.operator_spec import is_production_denied

    resolved = canon
    try:
        from cleaned_operators.registry import OperatorRegistry

        resolved = OperatorRegistry._aliases.get(canon, canon)
    except Exception:
        pass

    if is_production_denied(resolved):
        return "forbidden"
    if has_composite_lowering(resolved):
        return "composite"
    if resolved in EXTERNAL_KERNEL_CANONICALS:
        return "external_kernel"
    if resolved in STATEFUL_DEFERRED_CANONICALS:
        return "stateful"
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(resolved)
        meta = getattr(op, "metadata", None) if op else None
        tags = {str(t).lower() for t in (getattr(meta, "tags", None) or [])}
        if "session_aware" in tags or "period_aware" in tags:
            return "stateful"
    except Exception:
        pass
    return "primitive"


def lower_composite_operators(node: PlanNode) -> PlanNode:
    """自底向上递归，将已注册的高级算子展开为基础 DAG。"""
    _ensure_lowerings_loaded()
    from cleaned_operators.registry import OperatorRegistry

    children = [lower_composite_operators(child) for child in node.inputs]
    current = PlanNode(
        op=node.op,
        inputs=children,
        attrs=dict(node.attrs),
        node_id=node.node_id,
    )
    canon = OperatorRegistry._aliases.get(current.op, current.op)
    lowering = _COMPOSITE_LOWERINGS.get(canon)
    if lowering is None:
        return current
    lowered = lowering(current)
    if lowered.op == current.op and lowered.inputs == current.inputs:
        return current
    return lower_composite_operators(lowered)


def collect_plan_ops(node: PlanNode) -> list[str]:
    """深度优先收集 plan 中的 canonical 算子（去重保序）。"""
    from cleaned_operators.registry import OperatorRegistry

    seen: set[str] = set()
    ops: list[str] = []

    def walk(n: PlanNode) -> None:
        op = str(n.op or "")
        if not op:
            return
        canon = OperatorRegistry._aliases.get(op, op)
        if canon not in seen and canon not in {"column", "literal", "materialized_series", "plan_ref"}:
            seen.add(canon)
            ops.append(canon)
        for child in n.inputs:
            walk(child)

    walk(node)
    return ops


def composite_dual_backend_capable(canon: str) -> bool:
    """composite 算子展开后，全部 primitive 均在 dual-backend production-safe 集合内。"""
    prims = lowered_primitives(canon)
    if not prims:
        return False
    from backend.production_fastpath_tiers import dual_backend_production_safe

    safe = dual_backend_production_safe()
    return all(p in safe for p in prims)


def lowered_primitives(canon: str) -> tuple[str, ...] | None:
    """若存在 composite lowering，返回展开后的 primitive 集合（静态探测）。"""
    from cleaned_operators.registry import OperatorRegistry

    _ensure_lowerings_loaded()
    resolved = OperatorRegistry._aliases.get(canon, canon)
    lowering = _COMPOSITE_LOWERINGS.get(resolved)
    if lowering is None:
        return None
    probe_inputs = [
        PlanNode(op="column", attrs={"name": f"__probe_{i}__"}, inputs=[])
        for i in range(3)
    ]
    stub = PlanNode(
        op=resolved,
        inputs=probe_inputs,
        attrs={"window": 3, "d": 3, "std_dev": 2.0},
    )
    lowered = lower_composite_operators(stub)
    return tuple(collect_plan_ops(lowered))
