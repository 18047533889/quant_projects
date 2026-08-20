# -*- coding: utf-8
"""Registry Polars 算子 → long-table 编译桥。

背景
----
Polars long 后端（``polars_long_backend``）把因子计划编译成 **长表 LazyFrame**，
列固定为 ``(ts, inst, _v)``，避免整棵 plan 在宽表 panel 与 long 表之间来回转换。

本模块职责
----------
对 ``OperatorRegistry`` 里已有 ``backend=polars``、但 **未** 在 ``polars_expr_emitter``
手写 expr 的 canonical，在 ``_compile_polars`` 末尾 **fallback** 到此桥：

1. 递归编译子节点得到多路 LazyFrame；
2. 按 ``inst``（时序算子）或 ``ts``（截面算子）``group_by + map_groups``；
3. 每组内构造 mini panel，调用算子原生 ``calculate()``；
4. 写回 ``_v`` 列。

非职责
------
- ``group_*`` 算子（需宽表分组列，long 桥不支持）→ 返回 None 走其他后端
- 已移除的前视/随机算子不会进入 Registry，因此不会被 long bridge 接入
"""
from __future__ import annotations

from typing import Any, Callable

from planner.logical_plan import PlanNode

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

# long 表标准列名：时间戳 / 标的 / 值
_TS = "ts"
_INST = "inst"
_VAL = "_v"

# 不参与 registry long-bridge 的 canonical（语义或安全原因）
_SKIP_REGISTRY_LONG: frozenset[str] = frozenset()

# ``polars_registry_long_capable()`` 的进程内缓存，避免每次扫 Registry
_REGISTRY_LONG_CACHE: frozenset[str] | None = None


def _resolve(op: str) -> str:
    """Strictly resolve a DSL alias to its registered canonical name."""
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.resolve_canonical_strict(op)


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    """从 Polars 算子 ``calculate`` 返回值里找出数值结果列（排除轴列）。"""
    skip = frozenset({"date", "stock_code", _TS, _INST})
    return [c for c in df.columns if c not in skip]


def polars_registry_long_capable(*, exclude_native: frozenset[str] | None = None) -> frozenset[str]:
    """返回 Registry 中可走 long-bridge 的 canonical 集合（惰性缓存）。

    条件：``backend=polars`` 且不在 ``INTENTIONALLY_PANDAS_ONLY`` / ``_SKIP_REGISTRY_LONG`` 中。

    参数：
        exclude_native: 已在 ``polars_expr_emitter`` 原生实现的算子，从集合中排除。
    """
    global _REGISTRY_LONG_CACHE
    if _REGISTRY_LONG_CACHE is not None:
        reg_set = _REGISTRY_LONG_CACHE
    else:
        from cleaned_operators import load_all
        from cleaned_operators.operator_policy import INTENTIONALLY_PANDAS_ONLY, polars_implemented_canonicals

        load_all()
        reg = polars_implemented_canonicals()
        skip = INTENTIONALLY_PANDAS_ONLY | _SKIP_REGISTRY_LONG | {"column", "literal"}
        _REGISTRY_LONG_CACHE = frozenset(c for c in reg if c not in skip)
        reg_set = _REGISTRY_LONG_CACHE
    if exclude_native:
        return reg_set - exclude_native
    return reg_set


def registry_op_long_capable(op: str) -> bool:
    """单个算子名（可含别名）是否可由 long-bridge 编译。"""
    try:
        canonical = _resolve(op)
    except KeyError:
        # Capability discovery is intentionally non-throwing. Execution uses
        # ``compile_registry_op`` below and therefore retains strict resolution.
        return False
    return canonical in polars_registry_long_capable()


def plan_registry_long_capable(plan: PlanNode) -> bool:
    """整棵 PlanNode 子树是否 **全部** 可由 long-bridge 编译（递归检查）。"""
    try:
        op = _resolve(plan.op)
    except KeyError:
        return False
    if op in {"column", "literal"}:
        return True
    if not registry_op_long_capable(op):
        return False
    return all(plan_registry_long_capable(c) for c in plan.inputs)


def _op_scope(canonical: str) -> str:
    """Read the reviewed execution scope; scope guessing is forbidden."""
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    policy = _EXPLICIT_POLICIES.get(canonical)
    if policy is None or not policy.get("scope"):
        raise RuntimeError(f"{canonical}: missing explicit long-bridge scope policy")
    return str(policy["scope"])


def _call_polars_operator(
    canonical: str,
    operator: Any,
    call_args: list[Any],
    kw: dict[str, Any],
) -> pl.DataFrame:
    """Call a native Polars operator with Analyzer-normalized parameters."""
    from backend.parameter_aliases import reject_runtime_parameter_aliases

    reject_runtime_parameter_aliases(canonical, kw)
    return operator.calculate(*call_args, **kw)


def _extract_output_values(result: pl.DataFrame, n: int) -> list[float]:
    """从 ``calculate`` 返回的 DataFrame 抽出 ``n`` 个标量，对齐当前分组行数。

    兼容多种算子返回形态：单列 n 行、1 行 n 列、单元素等。
    """
    cols = _numeric_cols(result)
    if not cols:
        return [float("nan")] * n
    if len(cols) == 1 and result.height == n:
        return result[cols[0]].to_numpy().tolist()
    if result.height == 1 and len(cols) >= n:
        row = result.select(cols[:n]).to_numpy().flatten()
        return row.tolist()
    if result.height == 1 and len(cols) == 1 and n == 1:
        return [float(result[cols[0]][0])]
    if len(cols) == 1:
        arr = result[cols[0]].to_numpy()
        if arr.size == n:
            return arr.tolist()
    raise ValueError(f"registry bridge: unexpected polars result shape {result.shape} for n={n}")


def _build_call_args(
    g: pl.DataFrame,
    *,
    arg_specs: list[tuple[str, Any]],
    series_cols: list[str],
    cs: bool,
) -> list[Any]:
    """把 long 表分组 ``g`` 里的序列列转成算子 ``calculate`` 期望的 panel 参数。

    时序算子（``cs=False``）：单列 ``pl.DataFrame({"x": ndarray})``。
    截面算子（``cs=True``）：每标的一列的宽表 ``pl.DataFrame({c0: [...], ...})``。
    """
    call_args: list[Any] = []
    for kind, payload in arg_specs:
        if kind == "lit":
            call_args.append(payload)
            continue
        col = series_cols[payload]
        if cs:
            vals = g[col].to_list()
            call_args.append(pl.DataFrame({f"c{i}": [v] for i, v in enumerate(vals)}))
        else:
            call_args.append(pl.DataFrame({"x": g[col].to_numpy()}))
    return call_args


def _join_series_frames(frames: list[pl.LazyFrame], col_names: list[str]) -> pl.LazyFrame:
    """按 ``(ts, inst)`` anchor LEFT JOIN 多路子节点 LazyFrame。"""
    out = frames[0].rename({_VAL: col_names[0]})
    for lf, name in zip(frames[1:], col_names[1:]):
        out = out.join(lf.rename({_VAL: name}), on=[_TS, _INST], how="left")
    return out


def _parse_inputs(
    node: PlanNode,
    base: pl.LazyFrame,
    compile_fn: Callable[[PlanNode, pl.LazyFrame], pl.LazyFrame | None],
) -> tuple[list[tuple[str, Any]], list[pl.LazyFrame], dict[str, Any]] | None:
    """解析 PlanNode 输入：字面量 → ``("lit", value)``；子计划 → 编译后的 LazyFrame。

    返回 ``(arg_specs, series_frames, kwargs)``；任一子节点无法编译则返回 None。
    """
    arg_specs: list[tuple[str, Any]] = []
    series_frames: list[pl.LazyFrame] = []
    for child in node.inputs:
        if child.op == "literal":
            arg_specs.append(("lit", child.attrs.get("value")))
            continue
        compiled = compile_fn(child, base)
        if compiled is None:
            return None
        arg_specs.append(("ser", len(series_frames)))
        series_frames.append(compiled)
    if not series_frames:
        return None
    return arg_specs, series_frames, dict(node.attrs)


def _registry_map_inst(
    joined: pl.LazyFrame,
    *,
    canonical: str,
    arg_specs: list[tuple[str, Any]],
    series_cols: list[str],
    operator: Any,
    kw: dict[str, Any],
) -> pl.LazyFrame:
    """时序算子路径：按 ``inst`` 分组，每组内调用 ``calculate``，输出 ``(ts, inst, _v)``。"""
    schema = joined.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        call_args = _build_call_args(g, arg_specs=arg_specs, series_cols=series_cols, cs=False)
        result = _call_polars_operator(canonical, operator, call_args, kw)
        out = _extract_output_values(result, g.height)
        return g.select(
            pl.col(_TS),
            pl.col(_INST),
            pl.Series(_VAL, out),
        )

    return joined.group_by(_INST, maintain_order=True).map_groups(
        _apply,
        schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
    )


def _registry_map_cs(
    joined: pl.LazyFrame,
    *,
    canonical: str,
    arg_specs: list[tuple[str, Any]],
    series_cols: list[str],
    operator: Any,
    kw: dict[str, Any],
) -> pl.LazyFrame:
    """截面算子路径：按 ``ts`` 分组，每组内对所有标的做截面 ``calculate``。"""
    schema = joined.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        call_args = _build_call_args(g, arg_specs=arg_specs, series_cols=series_cols, cs=True)
        result = _call_polars_operator(canonical, operator, call_args, kw)
        out = _extract_output_values(result, g.height)
        return g.select(
            pl.col(_TS),
            pl.col(_INST),
            pl.Series(_VAL, out),
        )

    return joined.group_by(_TS, maintain_order=True).map_groups(
        _apply,
        schema={_TS: schema[_TS], _INST: schema[_INST], _VAL: pl.Float64},
    )


def compile_registry_op(
    node: PlanNode,
    base: pl.LazyFrame,
    compile_fn: Callable[[PlanNode, pl.LazyFrame], pl.LazyFrame | None],
) -> pl.LazyFrame | None:
    """Fallback：把单个 Registry Polars 算子编译为 long-table LazyFrame。

    参数：
        node: 当前 PlanNode（算子名 + 子输入 + attrs/kwargs）
        base: 基础 long LazyFrame（含 ts/inst 轴，供 column 子节点挂接）
        compile_fn: 递归编译子节点的回调（通常即 ``_compile_polars`` 自身）

    返回：
        成功 → ``LazyFrame(ts, inst, _v)``；不支持或失败 → ``None``（上层走其他后端）。
    """
    if pl is None:
        return None
    canonical = _resolve(node.op)
    if not registry_op_long_capable(canonical):
        return None
    parsed = _parse_inputs(node, base, compile_fn)
    if parsed is None:
        return None
    arg_specs, series_frames, kw = parsed
    from cleaned_operators.registry import OperatorRegistry

    try:
        operator = OperatorRegistry.get(canonical, backend="polars")
    except Exception:
        return None

    scope = _op_scope(canonical)
    if scope == "group":
        return None

    series_cols = [f"_a{i}" for i in range(len(series_frames))]
    joined = _join_series_frames(series_frames, series_cols)

    if scope == "cs":
        return _registry_map_cs(
            joined,
            canonical=canonical,
            arg_specs=arg_specs,
            series_cols=series_cols,
            operator=operator,
            kw=kw,
        )
    return _registry_map_inst(
        joined,
        canonical=canonical,
        arg_specs=arg_specs,
        series_cols=series_cols,
        operator=operator,
        kw=kw,
    )
