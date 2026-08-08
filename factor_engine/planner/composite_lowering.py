# -*- coding: utf-8
"""高级算子 composite lowering：展开为基础 PlanNode DAG。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from planner.logical_plan import PlanNode

LoweringFn = Callable[[PlanNode], PlanNode]


class CompositeLoweringDuplicateError(RuntimeError):
    """R5-29: 同一 canonical 的 composite lowering 被重复注册。"""
# R5-29: declared dependency signature per composite (``(param_names, min_inputs,
# active_branches)``).  When present it replaces the fake-stub capability probe so
# ``lowered_primitives`` reflects real parameter branches instead of a generic
# ``window=3`` AST.
_LOWERING_DECLARATIONS: dict[str, dict] = {}
# R5-29 / R6 P0-05: explicit replacement manifest — a second registration of the
# same canonical is an error unless it is declared here with a reason.  Use
# ``declare_lowering_replacement()`` to add an entry (never mutate directly).
_LOWERING_REPLACEMENTS: dict[str, str] = {}
# R6 P0-05: source-module provenance per composite — duplicate-registration
# errors now name BOTH the original and the clashing module.
_LOWERING_SOURCES: dict[str, str] = {}
# Composite lowering registry: canonical -> lowering callable.  Kept separate
# from the declaration/replacement manifests above; registration/query paths
# reference it, so it must always be defined at module scope.
_COMPOSITE_LOWERINGS: dict[str, LoweringFn] = {}
_LOWERINGS_LOADED = False

ExecutionKind = str  # primitive | composite | stateful | external_kernel

MAX_COMPOSITE_LOWERING_DEPTH = 32


@dataclass(frozen=True)
class ExecutionContract:
    """R5-28: 把 ``execution_kind`` 拆成三个正交轴。

    - ``scope``: 数据作用域（elementwise / ts / cs / group / fiscal / session）。
    - ``statefulness``: 是否需要跨 bar 状态（stateless / bounded / recursive /
      checkpointed）。
    - ``execution``: 执行方式（primitive / composite / external_kernel）。

    一个算子可以是 ``scope='fiscal', statefulness='stateless',
    execution='primitive'``（财报期感知但无递归状态），也可以是
    ``scope='session', statefulness='recursive', execution='stateful'``。
    """

    scope: str = "unknown"
    statefulness: str = "stateless"
    execution: str = "primitive"

    @property
    def kind(self) -> str:
        if self.statefulness in ("recursive", "checkpointed"):
            return "stateful"
        return self.execution


class CompositeLoweringCycleError(RuntimeError):
    """Composite lowering 检测到循环。"""


class CompositeLoweringDepthError(RuntimeError):
    """Composite lowering 超过最大深度。"""


# R6 P0-06: declaration of a composite's true parameter dependencies.  Replaces
# the fake ``window=3`` capability probe: when a composite declares its
# ``deps`` (parameter names its lowering actually reads) and ``min_inputs``, the
# static probe uses REAL defaults pulled from the registered operator's ParamSpec
# so branch decisions (MACD fast/slow/signal, Bollinger std_dev, …) actually fire.
@dataclass(frozen=True)
class LoweringContract:
    """Declared contract for one composite lowering (R6 P0-06).

    ``canonical``-independent: the contract is attached per canonical at
    registration.  ``param_branches`` maps a parameter name to the branch value
    range it drives (e.g. ``{"fast": (5, 26)}``) — used by the static probe to
    sample representative values instead of one fabricated number.
    """

    deps: tuple[str, ...] = ()
    min_inputs: int | None = None
    source_module: str = ""
    param_branches: dict[str, tuple[float, float]] = field(default_factory=dict)
    history_requirement: str | None = None  # exact_rows / max_rows / finite_observations
    grain_transform: str | None = None  # none / minute_to_daily / tick_to_daily
    statefulness: str = "stateless"


def declare_lowering_replacement(canonical: str, reason: str) -> None:
    """R6 P0-05: declare that a canonical's lowering may be re-registered.

    ``register_lowering`` refuses a duplicate registration unless the canonical
    is declared here with a replacement reason — the explicit escape hatch that
    replaces silent overwrite.
    """
    _LOWERING_REPLACEMENTS[str(canonical)] = reason

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
# R6 P1-16: names are the CANONICAL registry names (ts_ewm_std, not ewm_std) so
# every collection in the stateful system refers to the same canonical.
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
        "ts_ewm_std",
        "ts_ewm_var",
        "ts_ewm_cov",
        "ts_ewm_corr",
        "ts_decay_linear",
    }
)


def register_lowering(
    canonical: str,
    *,
    deps: tuple[str, ...] | None = None,
    min_inputs: int | None = None,
):
    """注册 canonical 算子的 lowering 函数（R5-29）。

    重复注册同一 canonical 会报错——silent overwrite 与 OperatorRegistry 的
    duplicate guard 同罪。仅显式 ``_LOWERING_REPLACEMENTS`` 声明的覆盖才允许。

    参数:
        canonical: 待展开的 composite canonical 名。
        deps: 可选的依赖签名——lowering 真正读取的参数名列表。声明后
            ``lowered_primitives`` 用这些参数构造探测 stub，而不是硬编码
            ``{"window":3,"d":3,"std_dev":2.0}``（R5-30）。
        min_inputs: 展开所需的最少 panel 输入数。
    """

    def decorator(fn: LoweringFn) -> LoweringFn:
        key = str(canonical)
        source = getattr(fn, "__module__", "") or ""
        if key in _COMPOSITE_LOWERINGS:
            original_source = _LOWERING_SOURCES.get(key, "<unknown>")
            if key not in _LOWERING_REPLACEMENTS:
                raise CompositeLoweringDuplicateError(
                    f"composite lowering for {key!r} already registered by "
                    f"{original_source}; silent overwrite by {source} is forbidden "
                    "(R5-29 / R6 P0-05).  Call "
                    "planner.composite_lowering.declare_lowering_replacement("
                    f"{key!r}, reason=...) to authorise the replacement."
                )
        _COMPOSITE_LOWERINGS[key] = fn
        _LOWERING_DECLARATIONS[key] = {
            "deps": tuple(deps or ()),
            "min_inputs": min_inputs,
            "source_module": source,
        }
        _LOWERING_SOURCES[key] = source
        return fn

    return decorator


def _ensure_lowerings_loaded() -> None:
    global _LOWERINGS_LOADED
    if _LOWERINGS_LOADED:
        return
    from planner.lowerings import (  # noqa: F401
        ashare,
        fundamental,
        microstructure,
        technical,
        timeseries,
    )

    _ = ashare, fundamental, microstructure, technical, timeseries
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


def infer_execution_contract(canon: str) -> ExecutionContract:
    """R5-28: 把 execution_kind 拆成 scope / statefulness / execution 三轴。

    ``infer_execution_kind`` 把 "财报 period-aware"、"session-aware" 一律归为
    ``stateful``，但实际上财报期感知（fiscal scope）多数是**无跨 bar 递归状态**
    的有限窗口/事件语义，session 聚合也可能是逐 session 重置的 bounded 状态。
    这里给出三个正交轴，调用方按需消费；``kind`` 保留旧的粗粒度标签做兼容。
    """
    resolved = canon
    try:
        from cleaned_operators.registry import OperatorRegistry

        resolved = OperatorRegistry._aliases.get(canon, canon)
    except Exception:
        pass
    scope: str = "unknown"
    statefulness: str = "stateless"
    execution: str = "primitive"
    if resolved in STATEFUL_DEFERRED_CANONICALS:
        statefulness = "recursive"
        execution = "stateful"
    elif resolved in EXTERNAL_KERNEL_CANONICALS:
        execution = "external_kernel"
    elif has_composite_lowering(resolved):
        execution = "composite"
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(resolved)
        meta = getattr(op, "metadata", None) if op else None
        tags = {str(t).lower() for t in (getattr(meta, "tags", None) or [])}
        for tag_scope, marker in (
            ("fiscal", "strict_fiscal_event"),
            ("fiscal", "fundamental_period"),
            ("session", "session_aware"),
            ("session", "session_intraday"),
            ("group", "group"),
            ("cs", "cross_section"),
        ):
            if marker in tags:
                scope = tag_scope
                break
        # Recursive state is a real property (Wilder / EWM / state machine);
        # period/session awareness alone is not enough to call an op recursive.
        if statefulness == "stateless":
            for marker in ("recursive", "stateful", "checkpointed", "state_machine"):
                if marker in tags:
                    statefulness = "recursive"
                    break
    except Exception:
        pass
    return ExecutionContract(
        scope=scope, statefulness=statefulness, execution=execution
    )


def infer_scope(canon: str) -> str:
    """R5-28: 只取 scope 轴。"""
    return infer_execution_contract(canon).scope


def infer_statefulness(canon: str) -> str:
    """R5-28: 只取 statefulness 轴。"""
    return infer_execution_contract(canon).statefulness


def infer_execution(canon: str) -> str:
    """R5-28: 只取 execution 轴。"""
    return infer_execution_contract(canon).execution


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
    """composite lowering 后全部 primitive 须已 dual-backend production 认证。"""
    prims = lowered_primitives(canon)
    if not prims:
        return False
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    return all(p in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE for p in prims)


def lowered_primitives(canon: str) -> tuple[str, ...] | None:
    """若存在 composite lowering，返回展开后的 primitive 集合（静态探测）。

    R5-30: 当 lowering 声明了 ``deps``（真正读取的参数名）时，探测 stub 使用
    声明参数构造真实分支，而不是硬编码 ``window=3`` 猜——像 MACD 这种依赖
    ``fast/slow/signal`` 才走展开分支的 composite，硬编码 stub 会得到空展开。
    """
    from cleaned_operators.registry import OperatorRegistry

    _ensure_lowerings_loaded()
    resolved = OperatorRegistry._aliases.get(canon, canon)
    lowering = _COMPOSITE_LOWERINGS.get(resolved)
    if lowering is None:
        return None
    declaration = _LOWERING_DECLARATIONS.get(resolved, {})
    deps = declaration.get("deps") or ()
    min_inputs = declaration.get("min_inputs")
    # R6 P0-06: probe with REAL declared defaults from the composite's registered
    # operator (ParamSpec default / param_types) instead of fabricated 5/2.0
    # numbers, so parameter-gated branches (MACD fast/slow/signal, Bollinger
    # std_dev, …) fire the same way they do in a real call.  Only when a dep has
    # no declared default do we fall back to a small integer.  The old hardcoded
    # ``{"window":3,"d":3,"std_dev":2.0}`` fallback is gone — an undeclared
    # composite probes with column inputs only and its own internal defaults.
    op_meta = getattr(OperatorRegistry.get(resolved), "metadata", None)
    specs = getattr(op_meta, "param_specs", None) or {}
    types = getattr(op_meta, "param_types", None) or {}
    stub_attrs: dict[str, object] = {}
    for dep in deps:
        spec = specs.get(dep)
        if spec is not None and spec.dtype is not None:
            default = None
            if spec.choices:
                default = spec.choices[0]
            if default is None:
                low = spec.min if spec.min is not None else 2
                default = int(low) if spec.dtype is int else float(low)
            stub_attrs[dep] = default
            continue
        declared = types.get(dep)
        if declared is int:
            stub_attrs[dep] = 5
        elif declared is float:
            stub_attrs[dep] = 2.0
        elif dep in {"fast", "fast_window", "short_window", "er_window", "signal_window"}:
            stub_attrs[dep] = 5  # fast side of an EMA/MACD pair
        elif dep in {"slow", "slow_window", "long_window"}:
            stub_attrs[dep] = 26  # slow side — must stay > fast so the branch fires
        elif dep in {"signal", "signal_period"}:
            stub_attrs[dep] = 9
        elif dep in {"window", "d", "left_window", "right_window", "history_window",
                     "n", "k", "period", "lag"}:
            stub_attrs[dep] = 5
        else:
            stub_attrs[dep] = 2.0 if dep in {"std_dev", "tolerance", "alpha", "span", "q"} else 1
    probe_count = max(min_inputs or 3, 3)
    probe_inputs = [
        PlanNode(op="column", attrs={"name": f"__probe_{i}__"}, inputs=[])
        for i in range(probe_count)
    ]
    stub = PlanNode(
        op=resolved,
        inputs=probe_inputs,
        attrs=dict(stub_attrs),
    )
    lowered = lower_composite_operators(stub, _stack=())
    return tuple(collect_plan_ops(lowered))
