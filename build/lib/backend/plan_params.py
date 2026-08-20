# -*- coding: utf-8
"""Plan 参数解析与校验（跨 PolarsLong / DuckDB 统一语义）。"""
from __future__ import annotations

from planner.logical_plan import PlanNode


class PlanParamError(ValueError):
    """非法 plan 参数（window / mode 等）。"""


def parse_positive_int_literal(raw: object, *, label: str = "window") -> int:
    """解析正整数 literal；拒绝非整数浮点、NaN、Inf、非正数。"""
    if raw is None:
        raise PlanParamError(f"{label} 未提供")
    if isinstance(raw, bool):
        raise PlanParamError(f"{label} 不能为 boolean")
    if isinstance(raw, float):
        if raw != raw or raw in (float("inf"), float("-inf")):
            raise PlanParamError(f"{label} 不能为 NaN/Inf")
        if raw != int(raw):
            raise PlanParamError(f"{label} 必须为整数 literal，收到 {raw!r}")
        val = int(raw)
    elif isinstance(raw, int):
        val = raw
    else:
        raise PlanParamError(f"{label} 必须为整数 literal，收到 {type(raw).__name__}")
    if val <= 0:
        raise PlanParamError(f"{label} 必须 > 0，收到 {val}")
    return val


def parse_finite_float(
    raw: object,
    *,
    label: str = "value",
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
) -> float:
    """解析有限浮点 literal；拒绝 NaN/Inf/非数值。"""
    if raw is None:
        raise PlanParamError(f"{label} 未提供")
    if isinstance(raw, bool):
        raise PlanParamError(f"{label} 不能为 boolean")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise PlanParamError(f"{label} 须为数值 literal，收到 {raw!r}") from exc
    if val != val or val in (float("inf"), float("-inf")):
        raise PlanParamError(f"{label} 须为有限值，收到 {raw!r}")
    if gt is not None and not val > gt:
        raise PlanParamError(f"{label} 必须 > {gt}，收到 {val}")
    if ge is not None and not val >= ge:
        raise PlanParamError(f"{label} 必须 >= {ge}，收到 {val}")
    if le is not None and not val <= le:
        raise PlanParamError(f"{label} 必须 <= {le}，收到 {val}")
    return val


def parse_unit_interval(raw: object, *, label: str = "p") -> float:
    """解析 [0, 1] 区间内的概率/分位参数。"""
    return parse_finite_float(raw, label=label, ge=0.0, le=1.0)


def window_from_plan_node(node: PlanNode, *, default: int = 3) -> int:
    """从 attrs 或 positional literal 解析滚动窗口（Polars/DuckDB 共用）。"""
    from backend.window_spec import WindowSpec

    return WindowSpec.from_plan_node(node, default_size=default).size


def window_spec_from_plan_node(node: PlanNode, *, default: int = 3):
    """返回完整 ``WindowSpec``（含 min_periods / ddof / null_policy）。"""
    from backend.window_spec import WindowSpec

    return WindowSpec.from_plan_node(node, default_size=default)


def int_mode_from_plan_node(
    node: PlanNode,
    *,
    input_index: int,
    default: int = 0,
    label: str = "mode",
) -> int:
    """读取 positional literal 整数 mode（``input_index`` 为 inputs 下标）。"""
    if input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            raw = child.attrs.get("value")
            if raw is not None:
                if isinstance(raw, bool):
                    return default
                if isinstance(raw, (int, float)) and raw == int(raw):
                    return int(raw)
    for key in ("mode", label):
        if key in (node.attrs or {}) and node.attrs[key] is not None:
            return int(node.attrs[key])
    return default


def parse_winsorize_quantiles(
    node: PlanNode,
    *,
    default_a: float = 0.05,
    default_upper: float = 0.95,
) -> tuple[float, float]:
    """解析 winsorize 上下分位。

    * ``attrs['a']`` / ``attrs['p']`` → 对称 ``(a, 1-a)``
    * ``winsorize(x, lo, hi)`` → 显式上下界
    * ``winsorize(x, lo)`` → ``(lo, default_upper)``（Pandas 兼容）
    """
    lo_p = hi_p = None
    pos_literals: list[float] = []
    for child in node.inputs[1:]:
        if child.op == "literal":
            raw = child.attrs.get("value")
            if raw is not None:
                pos_literals.append(parse_unit_interval(raw, label="winsorize"))
    attrs = node.attrs or {}
    if "a" in attrs or "p" in attrs:
        a = parse_unit_interval(attrs.get("a", attrs.get("p")), label="winsorize_a")
        return a, 1.0 - a
    if len(pos_literals) >= 2:
        lo_p, hi_p = pos_literals[0], pos_literals[1]
    elif len(pos_literals) == 1:
        lo_p, hi_p = pos_literals[0], default_upper
    elif "lower" in attrs or "min_pct" in attrs:
        lo_p = parse_unit_interval(attrs.get("lower", attrs.get("min_pct")), label="lower")
        hi_p = parse_unit_interval(
            attrs.get("upper", attrs.get("max_pct", default_upper)),
            label="upper",
        )
    else:
        lo_p, hi_p = default_a, 1.0 - default_a
    if lo_p > hi_p:
        raise PlanParamError(f"winsorize lower({lo_p}) 不能大于 upper({hi_p})")
    if lo_p == hi_p:
        raise PlanParamError(f"winsorize lower/upper 不能相等: {lo_p}")
    return lo_p, hi_p
