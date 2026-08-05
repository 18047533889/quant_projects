# -*- coding: utf-8 -*-
"""Safe recipe expansion, common-subexpression elimination and execution.

Recipes are parsed as a restricted expression language, recursively expanded to
production primitive operators, interned into a deterministic DAG, and executed
with per-node memoization.  The same rolling/EMA/regression node is therefore
computed only once even when several factor recipes reference it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import operator as py_operator
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from factor_recipes.registry import FactorRecipeRegistry


class RecipeExpansionError(ValueError):
    pass


_ALLOWED_NODES = (
    ast.Expression,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _validate_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RecipeExpansionError(f"unsupported recipe syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise RecipeExpansionError("only direct function calls are allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RecipeExpansionError("dunder names are forbidden")


def _to_ast(value: Any) -> ast.AST:
    if isinstance(value, ast.AST):
        return value
    if isinstance(value, str):
        tree = ast.parse(value, mode="eval")
        _validate_tree(tree)
        return tree.body
    if value is None or isinstance(value, (bool, int, float)):
        return ast.Constant(value=value)
    raise RecipeExpansionError(f"unsupported recipe binding type: {type(value).__name__}")


class _NameBinder(ast.NodeTransformer):
    def __init__(self, bindings: Mapping[str, ast.AST]):
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.bindings.get(node.id)
        if replacement is None:
            return node
        return ast.fix_missing_locations(ast.parse(ast.unparse(replacement), mode="eval").body)


class RecipeCompiler:
    def __init__(self, *, allowed_statuses: Iterable[str] = ("production",)):
        self.allowed_statuses = frozenset(allowed_statuses)

    def expand_ast(
        self,
        recipe_name: str,
        bindings: Mapping[str, Any],
        *,
        _stack: tuple[str, ...] = (),
    ) -> ast.AST:
        recipe = FactorRecipeRegistry.get(recipe_name)
        if recipe is None:
            raise RecipeExpansionError(f"unknown factor recipe: {recipe_name}")
        if recipe.status not in self.allowed_statuses:
            raise RecipeExpansionError(
                f"recipe {recipe_name!r} has status {recipe.status!r}; "
                f"allowed={sorted(self.allowed_statuses)}"
            )
        if (
            recipe.status == "production"
            and os.getenv("FACTOR_ENGINE_CERTIFY_RECIPE_EVIDENCE") != "1"
            and os.getenv("FACTOR_ENGINE_EXPAND_RECIPE_USAGE") != "1"
        ):
            from backend.recipe_evidence import recipe_execution_verified

            if not recipe_execution_verified(recipe_name):
                raise RecipeExpansionError(
                    f"recipe {recipe_name!r} lacks valid three-backend production evidence"
                )
        if recipe_name in _stack:
            chain = " -> ".join((*_stack, recipe_name))
            raise RecipeExpansionError(f"cyclic factor recipe dependency: {chain}")
        unknown = sorted(set(bindings) - set(recipe.parameters))
        missing = sorted(set(recipe.parameters) - set(bindings))
        if unknown:
            raise RecipeExpansionError(f"unknown parameters for {recipe_name}: {unknown}")
        if missing:
            raise RecipeExpansionError(f"missing parameters for {recipe_name}: {missing}")

        parsed = ast.parse(recipe.expression, mode="eval")
        _validate_tree(parsed)
        bound = _NameBinder({key: _to_ast(value) for key, value in bindings.items()}).visit(parsed.body)
        bound = ast.fix_missing_locations(bound)
        return self._expand_nested(bound, (*_stack, recipe_name))

    def _expand_nested(self, node: ast.AST, stack: tuple[str, ...]) -> ast.AST:
        class NestedExpander(ast.NodeTransformer):
            def __init__(self, outer: RecipeCompiler):
                self.outer = outer

            def visit_Call(self, call: ast.Call) -> ast.AST:
                call = self.generic_visit(call)
                assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                nested = FactorRecipeRegistry.get(call.func.id)
                if nested is None:
                    return call
                if call.keywords and any(keyword.arg is None for keyword in call.keywords):
                    raise RecipeExpansionError("**kwargs are forbidden in recipes")
                if len(call.args) > len(nested.parameters):
                    raise RecipeExpansionError(f"too many arguments for nested recipe {nested.name}")
                values: dict[str, ast.AST] = {}
                for parameter, value in zip(nested.parameters, call.args):
                    values[parameter] = value
                for keyword in call.keywords:
                    assert keyword.arg is not None
                    if keyword.arg in values:
                        raise RecipeExpansionError(f"duplicate argument {keyword.arg!r}")
                    values[keyword.arg] = keyword.value
                return self.outer.expand_ast(nested.name, values, _stack=stack)

        expanded = NestedExpander(self).visit(node)
        return ast.fix_missing_locations(expanded)

    def expand(self, recipe_name: str, bindings: Mapping[str, Any]) -> str:
        return ast.unparse(self.expand_ast(recipe_name, bindings))

    def compile_batch(self, requests: Mapping[str, tuple[str, Mapping[str, Any]]]) -> "CompiledRecipeBatch":
        builder = DAGBuilder()
        roots: dict[str, str] = {}
        formulas: dict[str, str] = {}
        for output_name, (recipe_name, bindings) in sorted(requests.items()):
            tree = self.expand_ast(recipe_name, bindings)
            roots[output_name] = builder.intern(tree)
            formulas[output_name] = ast.unparse(tree)
        return CompiledRecipeBatch(nodes=builder.nodes, roots=roots, expanded_formulas=formulas)


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    kind: str
    value: Any = None
    op: str | None = None
    args: tuple[str, ...] = ()
    kwargs: tuple[tuple[str, str], ...] = ()


class DAGBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, PlanNode] = {}
        self._keys: dict[str, str] = {}

    @staticmethod
    def _digest(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _store(self, key: Any, node_factory: Callable[[str], PlanNode]) -> str:
        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        existing = self._keys.get(canonical)
        if existing is not None:
            return existing
        node_id = self._digest(key)
        if node_id in self.nodes and self.nodes[node_id] != node_factory(node_id):
            node_id = hashlib.sha256((canonical + "#collision").encode()).hexdigest()
        node = node_factory(node_id)
        self.nodes[node_id] = node
        self._keys[canonical] = node_id
        return node_id

    def intern(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            key = ("literal", type(node.value).__name__, node.value)
            return self._store(key, lambda node_id: PlanNode(node_id, "literal", value=node.value))
        if isinstance(node, ast.Name):
            key = ("symbol", node.id)
            return self._store(key, lambda node_id: PlanNode(node_id, "symbol", value=node.id))
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            args = tuple(self.intern(arg) for arg in node.args)
            kwargs = tuple(sorted((keyword.arg or "", self.intern(keyword.value)) for keyword in node.keywords))
            key = ("call", node.func.id, args, kwargs)
            return self._store(key, lambda node_id: PlanNode(node_id, "call", op=node.func.id, args=args, kwargs=kwargs))
        if isinstance(node, ast.BinOp):
            args = (self.intern(node.left), self.intern(node.right))
            op = type(node.op).__name__
            key = ("binop", op, args)
            return self._store(key, lambda node_id: PlanNode(node_id, "binop", op=op, args=args))
        if isinstance(node, ast.UnaryOp):
            args = (self.intern(node.operand),)
            op = type(node.op).__name__
            key = ("unary", op, args)
            return self._store(key, lambda node_id: PlanNode(node_id, "unary", op=op, args=args))
        if isinstance(node, ast.BoolOp):
            args = tuple(self.intern(value) for value in node.values)
            op = type(node.op).__name__
            key = ("bool", op, args)
            return self._store(key, lambda node_id: PlanNode(node_id, "bool", op=op, args=args))
        if isinstance(node, ast.Compare):
            args = (self.intern(node.left), *(self.intern(value) for value in node.comparators))
            operators = tuple(type(op).__name__ for op in node.ops)
            key = ("compare", operators, args)
            return self._store(key, lambda node_id: PlanNode(node_id, "compare", value=operators, args=args))
        if isinstance(node, ast.IfExp):
            args = (self.intern(node.test), self.intern(node.body), self.intern(node.orelse))
            key = ("if", args)
            return self._store(key, lambda node_id: PlanNode(node_id, "if", args=args))
        raise RecipeExpansionError(f"cannot compile AST node: {type(node).__name__}")


_BINOPS = {
    "Add": py_operator.add,
    "Sub": py_operator.sub,
    "Mult": py_operator.mul,
    "Div": py_operator.truediv,
    "Pow": py_operator.pow,
    "Mod": py_operator.mod,
}
_UNARY = {"USub": py_operator.neg, "UAdd": py_operator.pos, "Not": py_operator.not_}
_COMPARE = {
    "Eq": py_operator.eq,
    "NotEq": py_operator.ne,
    "Lt": py_operator.lt,
    "LtE": py_operator.le,
    "Gt": py_operator.gt,
    "GtE": py_operator.ge,
}


@dataclass(frozen=True)
class CompiledRecipeBatch:
    nodes: Mapping[str, PlanNode]
    roots: Mapping[str, str]
    expanded_formulas: Mapping[str, str]

    @property
    def shared_node_count(self) -> int:
        references: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for child in node.args:
                references[child] = references.get(child, 0) + 1
            for _, child in node.kwargs:
                references[child] = references.get(child, 0) + 1
        for root in self.roots.values():
            references[root] = references.get(root, 0) + 1
        return sum(count > 1 for count in references.values())

    def execute(
        self,
        inputs: Mapping[str, Any],
        *,
        backend: str = "pandas_numpy",
        resolver: Callable[[str, str], Any] | None = None,
    ) -> dict[str, Any]:
        if resolver is None:
            from cleaned_operators.registry import OperatorRegistry

            resolver = lambda name, selected_backend: OperatorRegistry.get(name, backend=selected_backend)
        memo: dict[str, Any] = {}

        def evaluate(node_id: str) -> Any:
            if node_id in memo:
                return memo[node_id]
            node = self.nodes[node_id]
            args = [evaluate(child) for child in node.args]
            kwargs = {key: evaluate(child) for key, child in node.kwargs}
            if node.kind == "literal":
                result = node.value
            elif node.kind == "symbol":
                if node.value not in inputs:
                    raise KeyError(f"missing recipe input: {node.value}")
                result = inputs[node.value]
            elif node.kind == "call":
                target = resolver(node.op or "", backend)
                if target is None:
                    raise RecipeExpansionError(f"operator unavailable: {node.op}/{backend}")
                result = target.calculate(*args, **kwargs) if hasattr(target, "calculate") else target(*args, **kwargs)
            elif node.kind == "binop":
                result = _BINOPS[node.op or ""](*args)
            elif node.kind == "unary":
                result = _UNARY[node.op or ""](*args)
            elif node.kind == "bool":
                reducer = py_operator.and_ if node.op == "And" else py_operator.or_
                result = args[0]
                for value in args[1:]:
                    result = reducer(result, value)
            elif node.kind == "compare":
                comparisons = [_COMPARE[op](args[index], args[index + 1]) for index, op in enumerate(node.value)]
                result = comparisons[0]
                for value in comparisons[1:]:
                    result = py_operator.and_(result, value)
            elif node.kind == "if":
                condition, truthy, falsy = args
                try:
                    result = truthy.where(condition, falsy)
                except AttributeError:
                    result = truthy if condition else falsy
            else:
                raise RecipeExpansionError(f"unknown plan node kind: {node.kind}")
            memo[node_id] = result
            return result

        return {name: evaluate(root) for name, root in self.roots.items()}


def compile_recipes(
    requests: Mapping[str, tuple[str, Mapping[str, Any]]],
    *,
    allowed_statuses: Sequence[str] = ("production",),
) -> CompiledRecipeBatch:
    return RecipeCompiler(allowed_statuses=allowed_statuses).compile_batch(requests)
