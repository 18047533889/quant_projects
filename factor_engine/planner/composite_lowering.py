# -*- coding: utf-8
"""高级算子 composite lowering：展开为基础 PlanNode DAG。"""
from __future__ import annotations

from collections.abc import Callable

from planner.logical_plan import PlanNode

LoweringFn = Callable[[PlanNode], PlanNode]

_COMPOSITE_LOWERINGS: dict[str, LoweringFn] = {}
_LOWERINGS_LOADED = False

ExecutionKind = str  # primitive | composite | stateful | external_kernel

MAX_COMPOSITE_LOWERING_DEPTH = 32


class CompositeLoweringCycleError(RuntimeError):
    """Composite lowering 检测到循环。"""


class CompositeLoweringDepthError(RuntimeError):
    """Composite lowering 超过最大深度。"""

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
    """推断算子 execution_kind（不受 production deny 影响）。"""
    resolved = canon
    try:
        from cleaned_operators.registry import OperatorRegistry

        resolved = OperatorRegistry._aliases.get(canon, canon)
    except Exception:
        pass

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


def build_lowering_trace(before: PlanNode, after: PlanNode) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """对比 lowering 前后 plan，返回 ``(source_canonical, lowered_primitives...)`` 轨迹。"""
    from cleaned_operators.registry import OperatorRegistry

    before_ops = set(collect_plan_ops(before))
    trace: list[tuple[str, tuple[str, ...]]] = []
    for src in before_ops:
        if not has_composite_lowering(src):
            continue
        prims = lowered_primitives(src)
        if prims:
            trace.append((src, prims))
    _ = OperatorRegistry  # registry import warms aliases for collect_plan_ops
    after_ops = tuple(collect_plan_ops(after))
    if not trace and before_ops - set(after_ops):
        for removed in sorted(before_ops - set(after_ops)):
            if has_composite_lowering(removed):
                prims = tuple(op for op in after_ops if op not in before_ops) or after_ops
                trace.append((removed, prims))
    return tuple(trace)


def lower_composite_operators(
    node: PlanNode,
    *,
    _stack: tuple[str, ...] = (),
) -> PlanNode:
    """自底向上递归，将已注册的高级算子展开为基础 DAG。"""
    _ensure_lowerings_loaded()
    from cleaned_operators.registry import OperatorRegistry

    children = [lower_composite_operators(child, _stack=_stack) for child in node.inputs]
    current = PlanNode(
        op=node.op,
        inputs=children,
        attrs=dict(node.attrs),
        node_id=node.node_id,
    )
    canon = OperatorRegistry._aliases.get(current.op, current.op)
    if canon in _stack:
        chain = " -> ".join(_stack + (canon,))
        raise CompositeLoweringCycleError(f"CompositeLoweringCycleError: {chain}")
    if len(_stack) >= MAX_COMPOSITE_LOWERING_DEPTH:
        raise CompositeLoweringDepthError(
            f"Composite lowering exceeded max depth {MAX_COMPOSITE_LOWERING_DEPTH}: {' -> '.join(_stack)}"
        )
    lowering = _COMPOSITE_LOWERINGS.get(canon)
    if lowering is None:
        return current
    lowered = lowering(current)
    if lowered.op == current.op and lowered.inputs == current.inputs:
        return current
    return lower_composite_operators(lowered, _stack=_stack + (canon,))


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
    """composite 算子展开后，全部 primitive 均在双后端 structural 候选内。"""
    prims = lowered_primitives(canon)
    if not prims:
        return False
    from backend.production_fastpath_tiers import dual_backend_structural_candidates

    safe = dual_backend_structural_candidates()
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
    lowered = lower_composite_operators(stub, _stack=())
    return tuple(collect_plan_ops(lowered))
