# -*- coding: utf-8 -*-
"""Bridge compiled factor recipes into the main FactorEngine logical planner.

The recipe compiler owns safe AST expansion and batch CSE. This bridge preserves
that DAG identity while translating its nodes to ``planner.PlanNode`` so the
existing optimizer and PolarsLong/DuckDB capability routing can be reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from factor_recipes.compiler import CompiledRecipeBatch, PlanNode as RecipeNode, RecipeExpansionError
from planner.logical_plan import PlanNode

_BINOPS = {
    "Add": "add",
    "Sub": "subtract",
    "Mult": "multiply",
    "Div": "divide",
    "Pow": "power",
}
_UNARY = {"USub": "neg", "UAdd": "identity", "Not": "not_"}
_COMPARE = {
    "Eq": "eq",
    "NotEq": "ne",
    "Lt": "lt",
    "LtE": "le",
    "Gt": "gt",
    "GtE": "ge",
}


@dataclass(frozen=True)
class LogicalRecipeBatch:
    plans: Mapping[str, PlanNode]
    expanded_formulas: Mapping[str, str]
    node_count: int
    shared_node_count: int


def _fold_binary(op: str, nodes: list[PlanNode]) -> PlanNode:
    if not nodes:
        raise RecipeExpansionError(f"{op} requires at least one input")
    result = nodes[0]
    for child in nodes[1:]:
        result = PlanNode(op=op, inputs=[result, child])
    return result


def to_logical_plans(batch: CompiledRecipeBatch) -> LogicalRecipeBatch:
    """Translate a compiled recipe batch into shared main-planner nodes."""
    memo: dict[str, PlanNode] = {}

    def convert(node_id: str) -> PlanNode:
        existing = memo.get(node_id)
        if existing is not None:
            return existing
        node: RecipeNode = batch.nodes[node_id]
        if node.kind == "literal":
            result = PlanNode(op="literal", attrs={"value": node.value}, node_id=node_id)
        elif node.kind == "symbol":
            result = PlanNode(op="column", attrs={"name": str(node.value)}, node_id=node_id)
        elif node.kind == "call":
            inputs = [convert(child) for child in node.args]
            attrs: dict[str, object] = {}
            for key, child_id in node.kwargs:
                child = convert(child_id)
                if child.op != "literal":
                    raise RecipeExpansionError(
                        f"main planner requires literal recipe keyword {key!r}; got {child.op!r}"
                    )
                attrs[key] = child.attrs.get("value")
            result = PlanNode(op=node.op or "", inputs=inputs, attrs=attrs, node_id=node_id)
        elif node.kind == "binop":
            op = _BINOPS.get(node.op or "")
            if op is None:
                raise RecipeExpansionError(f"unsupported planner binop: {node.op}")
            result = PlanNode(op=op, inputs=[convert(child) for child in node.args], node_id=node_id)
        elif node.kind == "unary":
            op = _UNARY.get(node.op or "")
            if op is None:
                raise RecipeExpansionError(f"unsupported planner unary op: {node.op}")
            result = PlanNode(op=op, inputs=[convert(node.args[0])], node_id=node_id)
        elif node.kind == "bool":
            op = "and_" if node.op == "And" else "or_"
            result = _fold_binary(op, [convert(child) for child in node.args])
            result.node_id = node_id
        elif node.kind == "compare":
            operands = [convert(child) for child in node.args]
            comparisons = []
            for index, raw_op in enumerate(node.value):
                op = _COMPARE.get(raw_op)
                if op is None:
                    raise RecipeExpansionError(f"unsupported planner comparison: {raw_op}")
                comparisons.append(PlanNode(op=op, inputs=[operands[index], operands[index + 1]]))
            result = _fold_binary("and_", comparisons)
            result.node_id = node_id
        elif node.kind == "if":
            condition, truthy, falsy = [convert(child) for child in node.args]
            result = PlanNode(op="where", inputs=[condition, truthy, falsy], node_id=node_id)
        else:
            raise RecipeExpansionError(f"unsupported recipe DAG node kind: {node.kind}")
        memo[node_id] = result
        return result

    plans = {name: convert(root) for name, root in batch.roots.items()}
    references: dict[str, int] = {node_id: 0 for node_id in memo}
    for recipe_node in batch.nodes.values():
        for child in recipe_node.args:
            references[child] = references.get(child, 0) + 1
        for _, child in recipe_node.kwargs:
            references[child] = references.get(child, 0) + 1
    for root in batch.roots.values():
        references[root] = references.get(root, 0) + 1
    return LogicalRecipeBatch(
        plans=plans,
        expanded_formulas=dict(batch.expanded_formulas),
        node_count=len(memo),
        shared_node_count=sum(count > 1 for count in references.values()),
    )


def compile_recipe_plans(requests, *, allowed_statuses=("production",)) -> LogicalRecipeBatch:
    """Compile recipes once, preserve CSE, and return main logical plans."""
    from factor_recipes.registry import FactorRecipeRegistry

    return to_logical_plans(
        FactorRecipeRegistry.compile_batch(requests, allowed_statuses=allowed_statuses)
    )


__all__ = ["LogicalRecipeBatch", "to_logical_plans", "compile_recipe_plans"]
