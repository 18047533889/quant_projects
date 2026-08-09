# -*- coding: utf-8
"""高级算子 composite lowering：展开为基础 PlanNode DAG。"""
from __future__ import annotations

import hashlib
import inspect
import itertools
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
# R5-29 / R6 P0-05 / WS-D #261: explicit replacement manifest — a second
# registration of the same canonical is an error unless it is declared here with
# a reason AND (when provided) a hash of the lowering it expects to replace.
# Use ``declare_lowering_replacement()`` to add an entry (never mutate directly).
_LOWERING_REPLACEMENTS: dict[str, dict] = {}
# R6 P0-05: source-module provenance per composite — duplicate-registration
# errors now name BOTH the original and the clashing module.
_LOWERING_SOURCES: dict[str, str] = {}
# Composite lowering registry: canonical -> lowering callable.  Kept separate
# from the declaration/replacement manifests above; registration/query paths
# reference it, so it must always be defined at module scope.
_COMPOSITE_LOWERINGS: dict[str, LoweringFn] = {}
# WS-D #260: the FULL LoweringContract stored per canonical (deps / min_inputs /
# source_module / param_branches / history_requirement / grain_transform /
# statefulness).  Lookups read this — never a partial declaration dict.
_LOWERING_CONTRACTS: dict[str, "LoweringContract"] = {}
# WS-D #261: hash of the currently-registered lowering fn per canonical — the
# "old identity" a declared replacement must pin to.
_LOWERING_OLD_HASH: dict[str, str] = {}
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


def _lowering_hash(fn: LoweringFn) -> str:
    """Deterministic identity hash for one lowering function (WS-D #261).

    Combines the module path, the qualified name and the compiled bytecode so a
    source-level change to the lowering body invalidates the hash and a declared
    replacement pinned to the old hash is refused.
    """
    code = getattr(fn, "__code__", None)
    body = (code.co_code if code is not None else None) or repr(fn)
    payload = (
        f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}:"
        f"{body}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def declare_lowering_replacement(
    canonical: str,
    reason: str,
    *,
    expected_old_hash: str | None = None,
    new_hash: str | None = None,
    semantic_version: str | None = None,
) -> None:
    """R6 P0-05 / WS-D #261: declare that a canonical's lowering may be
    re-registered.

    ``register_lowering`` refuses a duplicate registration unless the canonical
    is declared here with a replacement reason — the explicit escape hatch that
    replaces silent overwrite.  When ``expected_old_hash`` is given, the
    re-registration is ALSO refused unless the currently-registered lowering
    hashes to exactly that value (identity-pinned replacement).
    """
    _LOWERING_REPLACEMENTS[str(canonical)] = {
        "reason": str(reason),
        "expected_old_hash": expected_old_hash,
        "new_hash": new_hash,
        "semantic_version": semantic_version,
    }

# WS-D #262: known multi-branch composites whose ``param_branches`` are declared
# here (the stored LoweringContract is enriched at registration).  ``fast`` /
# ``slow`` / ``signal`` carry distinct representative values (5/26/9 plus the
# canonical defaults) so every branch — including MACD's ``fast < slow`` gate —
# is exercised by ``certified_for_all_branches``.
_KNOWN_PARAM_BRANCHES: dict[str, dict[str, tuple[float, ...]]] = {
    "MACD_line": {"fast": (5.0, 12.0), "slow": (26.0,), "signal": (9.0,)},
    "MACD_signal": {"fast": (5.0, 12.0), "slow": (26.0,), "signal": (9.0,)},
    "MACD_hist": {"fast": (5.0, 12.0), "slow": (26.0,), "signal": (9.0,)},
    "BollingerUpper": {"window": (10.0, 20.0), "std_dev": (1.5, 2.0, 2.5)},
    "BollingerLower": {"window": (10.0, 20.0), "std_dev": (1.5, 2.0, 2.5)},
    "BollingerBands": {"window": (10.0, 20.0), "std_dev": (1.5, 2.0, 2.5)},
}

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
    contract: LoweringContract | None = None,
):
    """注册 canonical 算子的 lowering 函数（R5-29）。

    重复注册同一 canonical 会报错——silent overwrite 与 OperatorRegistry 的
    duplicate guard 同罪。仅显式 ``_LOWERING_REPLACEMENTS`` 声明的覆盖才允许；
    当该声明携带 ``expected_old_hash`` 时，还会校验当前已注册 lowering 的
    identity hash 与声明一致（WS-D #261）。

    参数:
        canonical: 待展开的 composite canonical 名。
        deps: 可选的依赖签名——lowering 真正读取的参数名列表。声明后
            ``lowered_primitives`` 用这些参数构造探测 stub，而不是硬编码
            ``{"window":3,"d":3,"std_dev":2.0}``（R5-30）。
        min_inputs: 展开所需的最少 panel 输入数。
        contract: WS-D #260 — 完整的 ``LoweringContract``（param_branches /
            history_requirement / grain_transform / statefulness）。传入后
            替代 ``deps``/``min_inputs`` 成为 lookups 的唯一契约来源。
    """

    def decorator(fn: LoweringFn) -> LoweringFn:
        key = str(canonical)
        source = getattr(fn, "__module__", "") or ""
        if key in _COMPOSITE_LOWERINGS:
            original_source = _LOWERING_SOURCES.get(key, "<unknown>")
            replacement = _LOWERING_REPLACEMENTS.get(key)
            if replacement is None:
                raise CompositeLoweringDuplicateError(
                    f"composite lowering for {key!r} already registered by "
                    f"{original_source}; silent overwrite by {source} is forbidden "
                    "(R5-29 / R6 P0-05).  Call "
                    "planner.composite_lowering.declare_lowering_replacement("
                    f"{key!r}, reason=...) to authorise the replacement."
                )
            expected_old_hash = replacement.get("expected_old_hash")
            if expected_old_hash:
                current_hash = _LOWERING_OLD_HASH.get(key) or _lowering_hash(
                    _COMPOSITE_LOWERINGS[key]
                )
                if current_hash != expected_old_hash:
                    raise CompositeLoweringDuplicateError(
                        f"declared replacement for {key!r} pins expected old "
                        f"lowering hash {expected_old_hash}, but the currently "
                        f"registered lowering hashes to {current_hash}; refusing "
                        f"replacement (WS-D #261)."
                    )
        declared_branches = dict(contract.param_branches or {}) if contract is not None else {}
        if not declared_branches and key in _KNOWN_PARAM_BRANCHES:
            declared_branches = dict(_KNOWN_PARAM_BRANCHES[key])
        stored = LoweringContract(
            deps=tuple(contract.deps or deps or ()) if contract is not None else tuple(deps or ()),
            min_inputs=contract.min_inputs if contract is not None else min_inputs,
            source_module=(
                (contract.source_module or source) if contract is not None else source
            ),
            param_branches=declared_branches,
            history_requirement=contract.history_requirement if contract is not None else None,
            grain_transform=contract.grain_transform if contract is not None else None,
            statefulness=contract.statefulness if contract is not None else None,
        )
        _COMPOSITE_LOWERINGS[key] = fn
        _LOWERING_CONTRACTS[key] = stored
        _LOWERING_DECLARATIONS[key] = {
            "deps": stored.deps,
            "min_inputs": stored.min_inputs,
            "source_module": stored.source_module,
        }
        _LOWERING_SOURCES[key] = source
        _LOWERING_OLD_HASH[key] = _lowering_hash(fn)
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
        semantic_attrs=dict(node.semantic_attrs),
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
    """composite lowering 后全部 primitive 须已 dual-backend production 认证。

    Reads the stored LoweringContract (default branch).  For the strict
    every-branch certification use :func:`certified_for_all_branches`.
    """
    prims = lowered_primitives(canon)
    if not prims:
        return False
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    return all(p in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE for p in prims)


def _resolve_composite(canon: str) -> str | None:
    """Resolve an alias/canonical to a registered composite, else None."""
    from cleaned_operators.registry import OperatorRegistry

    _ensure_lowerings_loaded()
    resolved = OperatorRegistry._aliases.get(canon, canon)
    if resolved not in _COMPOSITE_LOWERINGS:
        return None
    return resolved


def _kernel_panel_count(resolved: str) -> int | None:
    """Honest panel-input arity from the operator's kernel signature.

    Counts positional parameters (after ``self``) with no default — the panel
    inputs.  Falls back to ``None`` when the signature cannot be introspected.
    """
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(resolved)
    except Exception:
        return None
    fn = getattr(op, "_calculate_series", None) or getattr(op, "calculate", None)
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    count = 0
    for parameter in sig.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            break
        if parameter.default is inspect.Parameter.empty:
            count += 1
        else:
            break
    return count if count > 0 else None


def _probe_panel_count(resolved: str, contract: LoweringContract | None) -> int:
    """Derive the number of panel probe inputs for a composite (WS-D #264).

    Resolution order: the operator kernel signature (honest panel arity) ->
    the declared contract (``min_inputs`` minus scalar deps the probe passes as
    attrs) -> a conservative floor of 2 (single-series lowerings ignore the
    extra probe input; ratio/family lowerings need at least two series).
    """
    kernel_count = _kernel_panel_count(resolved)
    if kernel_count is not None:
        return kernel_count
    if contract is not None and contract.min_inputs is not None:
        return max(1, contract.min_inputs - len(contract.deps))
    return 2


def _probe_attrs(resolved: str, deps: tuple[str, ...]) -> dict[str, object]:
    """Default-branch probe attrs from the composite's declared parameter
    branches (or the legacy ParamSpec / name fallback).

    ``param_branches`` is the single source for the probe when available; the
    first value of each branch is the default branch.  Fast/slow/signal use
    DISTINCT representative values (5/26/9) so MACD's ``fast < slow`` branch
    actually fires.
    """
    branches = composite_param_branches(resolved)
    if branches:
        return {name: values[0] for name, values in branches.items()}
    from cleaned_operators.registry import OperatorRegistry

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
    return stub_attrs


def _probe_primitives(
    resolved: str,
    contract: LoweringContract | None,
    attrs: dict[str, object],
) -> tuple[str, ...] | None:
    """Lower one static probe stub and collect the resulting primitive ops."""
    panel_count = _probe_panel_count(resolved, contract)
    probe_inputs = [
        PlanNode(op="column", attrs={"name": f"__probe_{i}__"}, inputs=[])
        for i in range(panel_count)
    ]
    stub = PlanNode(
        op=resolved,
        inputs=probe_inputs,
        attrs=dict(attrs),
    )
    lowered = lower_composite_operators(stub, _stack=())
    return tuple(collect_plan_ops(lowered))


def lowered_primitives(canon: str) -> tuple[str, ...] | None:
    """若存在 composite lowering，返回展开后的 primitive 集合（静态探测）。

    R5-30 / WS-D #263-#264: 探测 stub 使用存储的 ``LoweringContract``（真实
    param_branches / ParamSpec 默认值）和真实 panel 输入构造，而不是硬编码
    ``window=3`` 或 ``max(min_inputs, 3)`` 个伪造输入。MACD fast/slow/signal
    取不同的代表性值（5/26/9），让 ``fast < slow`` 分支真实触发。
    """
    resolved = _resolve_composite(canon)
    if resolved is None:
        return None
    contract = _LOWERING_CONTRACTS.get(resolved)
    attrs = _probe_attrs(resolved, tuple(contract.deps) if contract else ())
    return _probe_primitives(resolved, contract, attrs)


def composite_param_branches(canon: str) -> dict[str, tuple[float, ...]]:
    """WS-D #262: enumerate the reachable parameter branch values of a composite.

    Prefers the stored ``LoweringContract.param_branches``; when a composite was
    registered without explicit branches, derives representative values from its
    declared deps + operator defaults (fast/slow/signal stay distinct and within
    a firing regime).  Returns ``{}`` when the canonical has no composite
    lowering or no branchable parameters.
    """
    resolved = _resolve_composite(canon)
    if resolved is None:
        return {}
    contract = _LOWERING_CONTRACTS.get(resolved)
    if contract is not None and contract.param_branches:
        return {name: tuple(values) for name, values in contract.param_branches.items()}
    if contract is not None and resolved in _KNOWN_PARAM_BRANCHES:
        return {name: tuple(values) for name, values in _KNOWN_PARAM_BRANCHES[resolved].items()}
    if contract is None:
        return {}
    deps = tuple(contract.deps)
    if not deps:
        return {}
    from cleaned_operators.registry import OperatorRegistry

    op_meta = getattr(OperatorRegistry.get(resolved), "metadata", None)
    specs = getattr(op_meta, "param_specs", None) or {}
    branches: dict[str, list[float]] = {}
    for dep in deps:
        values: list[float] = []
        spec = specs.get(dep)
        if spec is not None:
            if spec.choices:
                values.append(float(spec.choices[0]))
            default = getattr(spec, "default", None)
            if default is not None:
                values.append(float(default))
            if spec.min is not None:
                values.append(float(spec.min))
        if dep in {"fast", "fast_window", "short_window", "er_window", "signal_window"}:
            values.append(5.0)
        elif dep in {"slow", "slow_window", "long_window"}:
            values.append(26.0)
        elif dep in {"signal", "signal_period", "signal_span"}:
            values.append(9.0)
        elif dep in {"window", "d", "left_window", "right_window", "history_window",
                     "n", "k", "period", "lag", "ema_window", "atr_window",
                     "er_window", "short_window", "long_window"}:
            values.append(5.0)
        elif dep in {"std_dev", "tolerance", "alpha", "span", "q", "std"}:
            values.append(2.0)
        deduped = sorted({float(v) for v in values if v is not None})
        if deduped:
            branches[dep] = deduped
    return {name: tuple(values) for name, values in branches.items()}


def certified_for_all_branches(canon: str) -> bool:
    """WS-D #262: certify a composite over EVERY reachable parameter branch.

    Unlike :func:`composite_dual_backend_capable` (default branch only), this
    lowers the probe for every combination of ``composite_param_branches``
    values and requires every lowered primitive to be dual-backend production
    certified on every branch.  Returns ``False`` when any branch fails to lower
    or yields an uncertified primitive.
    """
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    resolved = _resolve_composite(canon)
    if resolved is None:
        return False
    contract = _LOWERING_CONTRACTS.get(resolved)
    branches = composite_param_branches(resolved)
    if not branches:
        prims = lowered_primitives(resolved)
        return bool(prims) and all(
            p in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE for p in prims
        )
    keys = list(branches)
    for combo in itertools.product(*(branches[key] for key in keys)):
        attrs = dict(zip(keys, combo))
        prims = _probe_primitives(resolved, contract, attrs)
        if not prims:
            return False
        if not all(p in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE for p in prims):
            return False
    return True


def composite_history_requirement(canon: str) -> str | None:
    """WS-D #260: the stored contract's ``history_requirement`` string."""
    resolved = _resolve_composite(canon)
    if resolved is None:
        return None
    contract = _LOWERING_CONTRACTS.get(resolved)
    return contract.history_requirement if contract is not None else None


def composite_grain_transform(canon: str) -> str | None:
    """WS-D #260: the stored contract's ``grain_transform`` string."""
    resolved = _resolve_composite(canon)
    if resolved is None:
        return None
    contract = _LOWERING_CONTRACTS.get(resolved)
    return contract.grain_transform if contract is not None else None


def composite_statefulness(canon: str) -> str:
    """WS-D #260: the stored contract's ``statefulness`` (default: inferred)."""
    resolved = _resolve_composite(canon)
    if resolved is None:
        return "stateless"
    contract = _LOWERING_CONTRACTS.get(resolved)
    if contract is not None and contract.statefulness:
        return contract.statefulness
    return infer_execution_contract(resolved).statefulness


def registered_lowering_contract(canon: str) -> LoweringContract | None:
    """Expose the stored ``LoweringContract`` for a composite (WS-D #260)."""
    resolved = _resolve_composite(canon)
    if resolved is None:
        return None
    return _LOWERING_CONTRACTS.get(resolved)
