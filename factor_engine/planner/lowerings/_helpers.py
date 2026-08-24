# -*- coding: utf-8
"""Composite lowering 计划构造 helper。"""
from __future__ import annotations

import math

from factor_engine.planner.logical_plan import PlanNode


def strict_int(value: object, name: str) -> int:
    """R5-08: a window-like/lag-like parameter must already be a whole number by
    the time composite lowering runs.

    Pipeline contract is ``Parse → canonical parameter validation → typed IR →
    lowering → backend``.  A lowering that does ``int(5.9) -> 5`` re-legalizes a
    fractional parameter the runtime validator would reject, so a composite like
    ``BollingerUpper(x, window=5.9)`` would silently compile to ``ts_mean(x,5)``.
    These helpers therefore refuse to truncate: any non-integral value is a
    planning-time error, not a silent floor.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}: window-like parameters cannot be {value!r} (R5-08)")
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError as exc:
            raise ValueError(f"{name}={value!r} is not numeric (R5-08)") from exc
    else:
        f = float(value)
    if not math.isfinite(f) or f != int(f):
        raise ValueError(
            f"{name}={value!r} is not an integer; a composite lowering must not "
            "truncate a fractional parameter that runtime validation rejects (R5-08)"
        )
    return int(f)


def strict_float(value: object, name: str) -> float:
    """R6 P0-04: a float-like parameter must be a finite non-boolean scalar by the
    time composite lowering runs — the pipeline contract is ``Parse → canonical
    parameter validation → typed IR → lowering → backend``.

    ``float_attr``/``literal`` used bare ``float()`` coercion, which silently
    accepted booleans (``float(True)==1.0``), ``inf`` and ``NaN`` — re-legalizing
    values the runtime validator rejects and baking them into the lowered DAG as
    literal constants.  Any non-finite / boolean / None value is a planning-time
    error, not a silent conversion (mirrors ``strict_int``).
    """
    if value is None:
        raise ValueError(f"{name}: scalar parameters cannot be None (R6 P0-04)")
    if isinstance(value, bool):
        raise ValueError(f"{name}: scalar parameters cannot be bool {value!r} (R6 P0-04)")
    # numpy bool_ has dtype.kind == 'b' — catch it without importing numpy here.
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", "")
    if dtype_kind == "b":
        raise ValueError(f"{name}: scalar parameters cannot be numpy-bool (R6 P0-04)")
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError as exc:
            raise ValueError(f"{name}={value!r} is not numeric (R6 P0-04)") from exc
    else:
        f = float(value)
    if not math.isfinite(f):
        raise ValueError(
            f"{name}={value!r} is not finite; a composite lowering must not "
            "re-legalize NaN/Inf that runtime validation rejects (R6 P0-04)"
        )
    return f


def literal(value: float) -> PlanNode:
    # A *structural* literal may legitimately be NaN (e.g. a ``where(cond, x,
    # NaN)`` else-branch, or fillna-to-NaN).  ``strict_float`` rejects NaN for
    # user-supplied parameters, but an internally-built NaN constant is a valid
    # formula literal — rejecting it here breaks every ``where``-to-NaN lowering.
    # bool / None / Inf remain planning errors either way.
    if value is None or isinstance(value, bool):
        raise ValueError(f"literal: scalar literal cannot be {value!r}")
    f = float(value)
    if not math.isfinite(f) and not math.isnan(f):
        raise ValueError(f"literal={value!r} is infinite; a composite lowering "
                         "must not bake Inf into the DAG")
    return PlanNode(op="literal", attrs={"value": f}, inputs=[])


def _literal_input(node: PlanNode, index: int) -> float | None:
    if len(node.inputs) <= index:
        return None
    child = node.inputs[index]
    if child.op == "literal" and "value" in child.attrs:
        return float(child.attrs["value"])
    return None


def window_int(node: PlanNode, *, default: int = 1, input_index: int = 1) -> int:
    w = node.attrs.get("window")
    if w is not None:
        return strict_int(w, "window")
    d = node.attrs.get("d")
    if d is not None:
        return strict_int(d, "d")
    lit = _literal_input(node, input_index)
    if lit is not None:
        return strict_int(lit, "window")
    return strict_int(default, "window")


def window_attrs(node: PlanNode, *, default: int = 1, input_index: int = 1) -> dict[str, int]:
    w = window_int(node, default=default, input_index=input_index)
    return {"d": w, "window": w}


def float_attr(
    node: PlanNode,
    *keys: str,
    default: float,
    input_index: int | None = None,
) -> float:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return strict_float(node.attrs[key], key)
    if input_index is not None:
        lit = _literal_input(node, input_index)
        if lit is not None:
            return strict_float(lit, keys[0] if keys else "float")
    return strict_float(default, keys[0] if keys else "float")


def delay(x: PlanNode, window: int) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_delay", inputs=[x, literal(float(w))], attrs={})


def ts_mean(x: PlanNode, window: int) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_mean", inputs=[x, literal(float(w))], attrs={})


def ts_std(x: PlanNode, window: int) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_std", inputs=[x, literal(float(w))], attrs={})


def ts_min(x: PlanNode, window: int) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_min", inputs=[x, literal(float(w))], attrs={})


def ts_max(x: PlanNode, window: int) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_max", inputs=[x, literal(float(w))], attrs={})


def ts_delta(x: PlanNode, window: int = 1) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_delta", inputs=[x, literal(float(w))], attrs={})


def ts_ema(x: PlanNode, window: int) -> PlanNode:
    """构造与 registry ``ts_ema`` 一致的时序 EMA 节点。"""
    w = strict_int(window, "window")
    return PlanNode(op="ts_ema", inputs=[x, literal(float(w))], attrs={})


def ts_pct(x: PlanNode, window: int = 1) -> PlanNode:
    w = strict_int(window, "window")
    return PlanNode(op="ts_pct", inputs=[x, literal(float(w))], attrs={})


def binop(op: str, left: PlanNode, right: PlanNode) -> PlanNode:
    return PlanNode(op=op, inputs=[left, right], attrs={})


def unary(op: str, inner: PlanNode) -> PlanNode:
    return PlanNode(op=op, inputs=[inner], attrs={})


def cum_sum(x: PlanNode) -> PlanNode:
    return PlanNode(op="cum_sum", inputs=[x], attrs={})


def fillna_const(x: PlanNode, value: float) -> PlanNode:
    return PlanNode(op="fillna_const", inputs=[x, literal(float(value))], attrs={})


def protected_div(left: PlanNode, right: PlanNode) -> PlanNode:
    return binop("protected_div", left, right)


def safe_div(left: PlanNode, right: PlanNode) -> PlanNode:
    return binop("safe_div_null", left, right)


def coalesce(*values: PlanNode) -> PlanNode:
    return PlanNode(op="coalesce", inputs=list(values), attrs={})
