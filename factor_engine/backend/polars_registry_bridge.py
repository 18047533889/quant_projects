# -*- coding: utf-8
"""Registry Polars 算子 → long-table 编译桥（按 inst / ts 分组，无整 plan panel 往返）。

对 ``OperatorRegistry`` 中已有 ``backend=polars``、但未手写 expr 的 canonical，
在 ``_compile_polars`` 末尾 fallback 到此模块。
"""
from __future__ import annotations

from typing import Any, Callable

from planner.logical_plan import PlanNode

from .pandas_compat import pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_TS = "ts"
_INST = "inst"
_VAL = "_v"

_SKIP_REGISTRY_LONG: frozenset[str] = frozenset(
    {
        "Lead",
        "next",
        "shuffle",
    }
)

_REGISTRY_LONG_CACHE: frozenset[str] | None = None


def _resolve(op: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    skip = frozenset({"date", "stock_code", _TS, _INST})
    return [c for c in df.columns if c not in skip]


def polars_registry_long_capable(*, exclude_native: frozenset[str] | None = None) -> frozenset[str]:
    """Registry 有 polars backend、可 long-bridge 的 canonical（惰性缓存）。"""
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
    return _resolve(op) in polars_registry_long_capable()


def plan_registry_long_capable(plan: PlanNode) -> bool:
    op = _resolve(plan.op)
    if op in {"column", "literal"}:
        return True
    if not registry_op_long_capable(op):
        return False
    return all(plan_registry_long_capable(c) for c in plan.inputs)


def _op_scope(canonical: str) -> str:
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    if canonical in _EXPLICIT_POLICIES:
        return str(_EXPLICIT_POLICIES[canonical].get("scope", "ts"))
    try:
        from cleaned_operators.registry import OperatorRegistry

        meta = OperatorRegistry.get(canonical, backend="polars").metadata
        cat = getattr(meta, "category", "") or ""
        if cat == "cross_sectional" or canonical.startswith(("cs_", "c_")):
            return "cs"
        if cat in {"group_neutralization", "group"} or canonical.startswith("group_"):
            return "group"
    except Exception:
        pass
    return "ts"


def _remap_d_to_window(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if "d" not in kwargs or "window" in kwargs:
        return None
    remapped = {k: v for k, v in kwargs.items() if k != "d"}
    remapped["window"] = kwargs["d"]
    return remapped


def _call_polars_operator(operator: Any, call_args: list[Any], kw: dict[str, Any]) -> pl.DataFrame:
    try:
        return operator.calculate(*call_args, **kw)
    except TypeError as exc:
        remapped = _remap_d_to_window(kw)
        if remapped is None:
            raise
        msg = str(exc).lower()
        if "unexpected keyword" not in msg and "got an unexpected" not in msg:
            raise
        try:
            return operator.calculate(*call_args, **remapped)
        except TypeError:
            raise exc from None


def _extract_output_values(result: pl.DataFrame, n: int) -> list[float]:
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
    out = frames[0].rename({_VAL: col_names[0]})
    for lf, name in zip(frames[1:], col_names[1:]):
        out = out.join(lf.rename({_VAL: name}), on=[_TS, _INST], how="inner")
    return out


def _parse_inputs(
    node: PlanNode,
    base: pl.LazyFrame,
    compile_fn: Callable[[PlanNode, pl.LazyFrame], pl.LazyFrame | None],
) -> tuple[list[tuple[str, Any]], list[pl.LazyFrame], dict[str, Any]] | None:
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
    arg_specs: list[tuple[str, Any]],
    series_cols: list[str],
    operator: Any,
    kw: dict[str, Any],
) -> pl.LazyFrame:
    schema = joined.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        call_args = _build_call_args(g, arg_specs=arg_specs, series_cols=series_cols, cs=False)
        result = _call_polars_operator(operator, call_args, kw)
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
    arg_specs: list[tuple[str, Any]],
    series_cols: list[str],
    operator: Any,
    kw: dict[str, Any],
) -> pl.LazyFrame:
    schema = joined.collect_schema()

    def _apply(g: pl.DataFrame) -> pl.DataFrame:
        call_args = _build_call_args(g, arg_specs=arg_specs, series_cols=series_cols, cs=True)
        result = _call_polars_operator(operator, call_args, kw)
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
    """Fallback：调用 Registry polars 算子（按 inst 或 ts 分组）。"""
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
            arg_specs=arg_specs,
            series_cols=series_cols,
            operator=operator,
            kw=kw,
        )
    return _registry_map_inst(
        joined,
        arg_specs=arg_specs,
        series_cols=series_cols,
        operator=operator,
        kw=kw,
    )
