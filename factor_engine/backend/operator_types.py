# -*- coding: utf-8
"""算子 schema / 类型系统（plan 编译前静态校验）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from planner.logical_plan import PlanNode


class TypeKind(str, Enum):
    SERIES_FLOAT = "Series[Float]"
    SERIES_BOOL = "Series[Bool]"
    SERIES_STRING = "Series[String]"
    SERIES_DATETIME = "Series[Datetime]"
    SCALAR_INT = "Scalar[Int]"
    SCALAR_FLOAT = "Scalar[Float]"
    GROUP_KEY = "GroupKey"
    WINDOW = "Window"
    ANY = "Any"


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type_kind: TypeKind
    required: bool = True
    allow_scalar_broadcast: bool = False


@dataclass(frozen=True)
class OperatorSignature:
    canonical: str
    inputs: tuple[ArgSpec, ...]
    output: TypeKind = TypeKind.SERIES_FLOAT
    allow_dynamic_window: bool = False
    allow_string_group: bool = True

    def validate_node(self, node: PlanNode) -> list[str]:
        """校验 plan 节点输入类型/形状（literal vs series）。"""
        violations: list[str] = []
        attrs = node.attrs or {}
        has_window_attr = any(
            k in attrs and attrs[k] is not None for k in ("d", "window", "n", "periods", "span")
        )
        required_inputs = sum(1 for a in self.inputs if a.required)
        # window 可在 attrs 中，不必占用 positional 输入
        if has_window_attr:
            required_inputs = sum(
                1 for a in self.inputs if a.required and a.type_kind != TypeKind.WINDOW
            )
        if len(node.inputs) < required_inputs:
            violations.append(
                f"{self.canonical}: 需要 {required_inputs} 个输入，收到 {len(node.inputs)}"
            )
            return violations

        for idx, arg in enumerate(self.inputs):
            if arg.type_kind == TypeKind.WINDOW and has_window_attr:
                continue
            if idx >= len(node.inputs):
                if arg.required:
                    violations.append(f"{self.canonical}: 缺少参数 {arg.name}")
                continue
            child = node.inputs[idx]
            err = _check_input(child, arg, canonical=self.canonical, arg_name=arg.name)
            if err:
                violations.append(err)

        if not self.allow_dynamic_window:
            for idx, child in enumerate(node.inputs):
                if child.op in {"column", "literal", "materialized_series", "plan_ref"}:
                    continue
                arg = self.inputs[idx] if idx < len(self.inputs) else None
                if arg and arg.type_kind == TypeKind.WINDOW:
                    violations.append(
                        f"{self.canonical}: 参数 {arg.name} 须为整数 literal Window，"
                        f"收到动态表达式 {child.op!r}"
                    )
        return violations


def _check_input(
    child: PlanNode,
    arg: ArgSpec,
    *,
    canonical: str,
    arg_name: str,
) -> str | None:
    if arg.type_kind == TypeKind.WINDOW:
        if child.op != "literal":
            return f"{canonical}: {arg_name} 须为整数 literal 窗口"
        raw = child.attrs.get("value")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return f"{canonical}: {arg_name} 须为整数 literal 窗口"
        if isinstance(raw, float) and raw != int(raw):
            return f"{canonical}: {arg_name} 不能为非整数浮点 {raw!r}"
        if int(raw) <= 0:
            return f"{canonical}: {arg_name} 必须 > 0"
        return None

    if arg.type_kind in {
        TypeKind.SERIES_FLOAT,
        TypeKind.SERIES_BOOL,
        TypeKind.SERIES_STRING,
        TypeKind.SERIES_DATETIME,
        TypeKind.GROUP_KEY,
    }:
        if child.op == "literal" and not arg.allow_scalar_broadcast:
            raw = child.attrs.get("value")
            if arg.type_kind == TypeKind.GROUP_KEY:
                return f"{canonical}: {arg_name} 须为 group Series，收到 literal {raw!r}"
            if arg.type_kind == TypeKind.SERIES_BOOL and not isinstance(raw, (bool, int, float)):
                return f"{canonical}: {arg_name} 须为 bool Series"
        if child.op not in {"column", "literal", "materialized_series", "plan_ref"} and child.op:
            return None
        return None

    if arg.type_kind in {TypeKind.SCALAR_INT, TypeKind.SCALAR_FLOAT}:
        if child.op != "literal":
            return f"{canonical}: {arg_name} 须为 scalar literal"
    return None


# 第一阶段核心签名（逐步扩展，batch-2 rolling 复用 WindowSpec）
OPERATOR_SIGNATURES: dict[str, OperatorSignature] = {
    "ts_mean": OperatorSignature(
        "ts_mean",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("window", TypeKind.WINDOW),
        ),
    ),
    "ts_std": OperatorSignature(
        "ts_std",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("window", TypeKind.WINDOW),
        ),
    ),
    "group_mean": OperatorSignature(
        "group_mean",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("group", TypeKind.GROUP_KEY),
        ),
    ),
    "where": OperatorSignature(
        "where",
        (
            ArgSpec("cond", TypeKind.SERIES_BOOL),
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "rank": OperatorSignature(
        "rank",
        (ArgSpec("x", TypeKind.SERIES_FLOAT),),
    ),
    "cs_regression": OperatorSignature(
        "cs_regression",
        (
            ArgSpec("y", TypeKind.SERIES_FLOAT),
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("mode", TypeKind.SCALAR_INT, allow_scalar_broadcast=True),
        ),
    ),
    "ts_argmax": OperatorSignature(
        "ts_argmax",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_argmin": OperatorSignature(
        "ts_argmin",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "add": OperatorSignature(
        "add",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "subtract": OperatorSignature(
        "subtract",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "multiply": OperatorSignature(
        "multiply",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "divide": OperatorSignature(
        "divide",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "ts_delay": OperatorSignature(
        "ts_delay",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_delta": OperatorSignature(
        "ts_delta",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_pct": OperatorSignature(
        "ts_pct",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_sum": OperatorSignature(
        "ts_sum",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_max": OperatorSignature(
        "ts_max",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "ts_min": OperatorSignature(
        "ts_min",
        (ArgSpec("x", TypeKind.SERIES_FLOAT), ArgSpec("window", TypeKind.WINDOW)),
    ),
    "coalesce": OperatorSignature(
        "coalesce",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "fillna_const": OperatorSignature(
        "fillna_const",
        (
            ArgSpec("x", TypeKind.SERIES_FLOAT),
            ArgSpec("value", TypeKind.SCALAR_FLOAT, allow_scalar_broadcast=True),
        ),
    ),
    "and_": OperatorSignature(
        "and_",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "or_": OperatorSignature(
        "or_",
        (
            ArgSpec("a", TypeKind.SERIES_FLOAT),
            ArgSpec("b", TypeKind.SERIES_FLOAT),
        ),
    ),
    "not_": OperatorSignature("not_", (ArgSpec("x", TypeKind.SERIES_FLOAT),)),
}


def check_operator_types(node: PlanNode, *, canonical: str | None = None) -> list[str]:
    """对 plan 节点做类型/schema 校验。"""
    from cleaned_operators.registry import OperatorRegistry

    canon = canonical or OperatorRegistry._aliases.get(str(node.op or ""), str(node.op or ""))
    sig = OPERATOR_SIGNATURES.get(canon)
    if sig is None:
        return []
    return sig.validate_node(node)


def check_plan_types(plan: Any) -> list[str]:
    """深度遍历 plan 收集类型违规。"""
    _SKIP = frozenset({"column", "literal", "materialized_series", "plan_ref"})
    violations: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, PlanNode):
            return
        op = str(node.op or "")
        if not op or op in _SKIP:
            for child in node.inputs or []:
                walk(child)
            return
        from cleaned_operators.registry import OperatorRegistry

        canon = OperatorRegistry._aliases.get(op, op)
        violations.extend(check_operator_types(node, canonical=canon))
        for child in node.inputs or []:
            walk(child)

    walk(plan)
    return violations
