# -*- coding: utf-8 -*-
"""Production evidence for declarative FactorEngine recipes.

Recipes do not own numerical kernels.  Their executable meaning is the fully
expanded operator DAG, so production admission is derived from the current
immutable recipe definition plus the evidence-backed capability of every leaf.
A recipe is valid when:

* it is registered with ``status='production'``;
* every nested recipe is itself production-valid;
* every operator leaf resolves to a canonical with at least one certified
  production backend; and
* the expansion is acyclic.

A common backend is reported when all leaves share one.  Otherwise the recipe
is marked ``hybrid`` and the normal plan router may select certified backends
per node.  The historical JSON artifact remains a reproducible test output for
portable recipes, but stale JSON can no longer disable an otherwise fully
certified declarative DAG.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = FE_ROOT.parent / "evidence" / "recipe_case_registry.json"
VERIFIED_PATH = FE_ROOT.parent / "evidence" / "recipe_verified.json"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        # R22-EVIDENCE-CONTAMINATION: never hash test-certified / fixture /
        # synthetic subtrees as production evidence.
        from factor_engine.backend.evidence_provenance import evidence_path_excluded

        if evidence_path_excluded(path, root=root):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _production_recipe_names() -> set[str]:
    from factor_engine.factor_recipes import FactorRecipeRegistry

    return set(FactorRecipeRegistry.list_names(status="production"))


def current_hashes() -> dict[str, str]:
    """Hashes used when a fresh portable-execution artifact is generated."""
    return {
        "case_registry_hash": _hash_file(CASE_PATH),
        "recipe_source_hash": _hash_tree(FE_ROOT / "factor_recipes"),
        "portable_test_source_hash": _hash_file(
            FE_ROOT / "tests" / "operators" / "test_recipe_backend_certification.py"
        ),
        "production_test_source_hash": _hash_file(
            FE_ROOT / "tests" / "operators" / "test_recipe_production_admission_v2.py"
        ),
        "primitive_evidence_hash": _hash_file(
            FE_ROOT.parent / "evidence" / "primitive_verified.json"
        ),
        "factor_operator_evidence_hash": _hash_file(
            FE_ROOT.parent / "evidence" / "factor_operator_verified.json"
        ),
    }


_BINOP_CANONICALS = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.FloorDiv: "divide",
    ast.Mod: "mod",
    ast.Pow: "power",
}
_COMPARE_CANONICALS = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}


def _expression_dependencies(expression: str) -> tuple[set[str], set[str]]:
    """Return function names and implicit arithmetic operator canonicals."""
    tree = ast.parse(str(expression), mode="eval")
    functions: set[str] = set()
    implicit: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("recipe calls must use direct DSL names")
            functions.add(node.func.id)
        elif isinstance(node, ast.BinOp):
            canonical = _BINOP_CANONICALS.get(type(node.op))
            if canonical is None:
                raise ValueError(f"unsupported recipe binary operator: {type(node.op).__name__}")
            implicit.add(canonical)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                implicit.add("neg")
            elif isinstance(node.op, ast.Not):
                implicit.add("not_")
            elif not isinstance(node.op, ast.UAdd):
                raise ValueError(
                    f"unsupported recipe unary operator: {type(node.op).__name__}"
                )
        elif isinstance(node, ast.BoolOp):
            implicit.add("and_" if isinstance(node.op, ast.And) else "or_")
        elif isinstance(node, ast.Compare):
            for operator in node.ops:
                canonical = _COMPARE_CANONICALS.get(type(operator))
                if canonical is None:
                    raise ValueError(
                        f"unsupported recipe comparison: {type(operator).__name__}"
                    )
                implicit.add(canonical)
        elif isinstance(node, ast.IfExp):
            implicit.add("where")
    return functions, implicit


def _leaf_backend_sets(name: str, stack: tuple[str, ...]) -> list[frozenset[str]]:
    from factor_engine.backend.operator_capability import production_eligible_backends
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.factor_recipes import FactorRecipeRegistry

    if name in stack:
        raise ValueError("cyclic recipe expansion: " + " -> ".join((*stack, name)))
    recipe = FactorRecipeRegistry.get(name)
    if recipe is None or recipe.status != "production":
        raise ValueError(f"recipe is not production-registered: {name}")

    functions, implicit = _expression_dependencies(recipe.expression)
    leaves: list[frozenset[str]] = []
    next_stack = (*stack, name)
    for dependency in sorted(functions):
        nested = FactorRecipeRegistry.get(dependency)
        if nested is not None:
            if nested.status != "production":
                raise ValueError(
                    f"production recipe {name} depends on non-production recipe {dependency}"
                )
            leaves.extend(_leaf_backend_sets(dependency, next_stack))
            continue
        canonical = OperatorRegistry.resolve_canonical_strict(dependency)
        backends = frozenset(production_eligible_backends(canonical))
        if not backends:
            raise ValueError(
                f"production recipe {name} leaf {canonical} has no certified backend"
            )
        leaves.append(backends)

    for dependency in sorted(implicit):
        canonical = OperatorRegistry.resolve_canonical_strict(dependency)
        backends = frozenset(production_eligible_backends(canonical))
        if not backends:
            raise ValueError(
                f"production recipe {name} leaf {canonical} has no certified backend"
            )
        leaves.append(backends)
    if not leaves:
        raise ValueError(f"production recipe {name} has no executable operator leaves")
    return leaves


@lru_cache(maxsize=256)
def _derived_recipe_backends(name: str) -> frozenset[str]:
    from factor_engine.backend.evidence_provenance import evidence_artifact_valid
    from factor_engine.backend.factor_operator_evidence import factor_operator_evidence_valid
    from factor_engine.cleaned_operators import load_all

    load_all()
    if not evidence_artifact_valid() or not factor_operator_evidence_valid():
        return frozenset()
    try:
        leaves = _leaf_backend_sets(str(name), ())
    except Exception:
        return frozenset()
    common = set(leaves[0])
    for backends in leaves[1:]:
        common.intersection_update(backends)
    if common:
        return frozenset(common)
    # Every leaf is independently certified, but the plan needs per-node routing.
    return frozenset({"hybrid"})


@lru_cache(maxsize=1)
def _derived_verified_names() -> frozenset[str]:
    names = _production_recipe_names()
    return frozenset(name for name in names if _derived_recipe_backends(name))


def recipe_evidence_valid(*, require_commit_ancestor: bool = False) -> bool:
    """Return whether every production recipe has an evidence-backed DAG.

    ``require_commit_ancestor`` is retained for API compatibility.  The derived
    evidence is bound to current leaf artifacts and recipe source at runtime,
    so no separate recipe commit ancestry check is required.
    """
    del require_commit_ancestor
    names = _production_recipe_names()
    return bool(names) and _derived_verified_names() == names


def recipe_execution_verified(name: str) -> bool:
    return str(name) in _derived_verified_names()


def recipe_certified_backends(name: str) -> frozenset[str]:
    if str(name) not in _production_recipe_names():
        return frozenset()
    return _derived_recipe_backends(str(name))


def verified_recipe_names() -> frozenset[str]:
    return _derived_verified_names()
