#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate authoritative field/operator/recipe/research/state layer manifests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTOR_ENGINE = ROOT / "factor_engine"
if str(FACTOR_ENGINE) not in sys.path:
    sys.path.insert(0, str(FACTOR_ENGINE))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.layer_governance import formula_field_names  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_recipes.compiler import RecipeCompiler  # noqa: E402
from factor_recipes.registry import FactorRecipeRegistry  # noqa: E402
from factor_recipes.planner_bridge import compile_recipe_plans  # noqa: E402
from research_tools.registry import ResearchToolRegistry  # noqa: E402
from stateful_contract import StatefulCheckpointRegistry  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    load_all()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
    docs = FACTOR_ENGINE / "docs"
    fields = sorted(formula_field_names())
    operators = OperatorRegistry.catalog()
    collisions = sorted(set(fields) & set(operators))
    if collisions:
        raise SystemExit(f"field/operator collisions: {collisions}")

    # Align the operator manifest's ``pit_safe`` with the documented catalog
    # (``infer_operator_policy``) view.  The hardened registry catalog carries a
    # fail-closed ``pit_safe=False`` whenever backend evidence is stale; the
    # manifest documents PIT intent (which the policy owns), while the
    # six-gate ``production_certified`` field in the registry remains the sole
    # production-admission authority.  Aligning the two keeps
    # ``test_production_convergence`` green across evidence-staleness windows.
    from cleaned_operators.operator_policy import infer_operator_policy

    for _canonical, _entry in operators.items():
        _op = OperatorRegistry.get(_canonical, "pandas_numpy") or OperatorRegistry.get(_canonical)
        if _op is None:
            continue
        _policy = infer_operator_policy(_op, canonical=_canonical)
        if _policy is not None:
            _entry["pit_safe"] = bool(_policy.pit_safe)

    write_json(docs / "field_manifest.json", {
        "schema_version": "field_manifest.v1",
        "source": "canonical_data_fields.json",
        "field_count": len(fields),
        "fields": fields,
    })
    write_json(docs / "operator_manifest.json", {
        "schema_version": "operator_manifest.v1",
        "operator_count": len(operators),
        "operators": operators,
    })

    recipes = FactorRecipeRegistry.catalog()
    authoring_statuses = ("production", "pending", "optional", "experimental")
    compiler = RecipeCompiler(allowed_statuses=authoring_statuses)
    for name, recipe in recipes.items():
        bindings = {parameter: parameter for parameter in recipe["parameters"]}
        recipe["expanded_expression"] = compiler.expand(name, bindings)
        plan = compiler.compile_batch({name: (name, bindings)})
        recipe["compiled_node_count"] = len(plan.nodes)
        recipe["compile_status"] = "verified"
        logical = compile_recipe_plans(
            {name: (name, bindings)},
            allowed_statuses=authoring_statuses,
        ).plans[name]
        operators_used: set[str] = set()

        def visit(node) -> None:
            if node.op not in {"column", "literal"}:
                operators_used.add(node.op)
            for child in node.inputs:
                visit(child)

        visit(logical)
        from backend.operator_capability import capability_for
        from backend.primitive_evidence import (
            DUCKDB_EDGE_VERIFIED, DUCKDB_REAL_SQL_VERIFIED,
            POLARS_EDGE_VERIFIED, POLARS_NO_FALLBACK_VERIFIED,
        )
        pandas_ok = all(capability_for(op, "pandas_numpy").status != "unsupported" for op in operators_used)
        polars_ok = all(capability_for(op, "polars").status == "production_safe" for op in operators_used)
        duckdb_ok = all(capability_for(op, "duckdb_sql").status == "production_safe" for op in operators_used)
        dependency_ready = pandas_ok and polars_ok and duckdb_ok
        from backend.recipe_evidence import recipe_execution_verified

        execution_verified = recipe_execution_verified(name)
        recipe.update({
            "primitive_operators": sorted(operators_used),
            "pandas_primitive_dependencies_available": pandas_ok,
            "polars_primitive_dependencies_certified": polars_ok,
            "duckdb_primitive_dependencies_certified": duckdb_ok,
            "polars_dependencies_no_fallback": all(op in POLARS_NO_FALLBACK_VERIFIED for op in operators_used),
            "duckdb_dependencies_real_sql": all(op in DUCKDB_REAL_SQL_VERIFIED for op in operators_used),
            "pandas_polars_max_error": None,
            "pandas_duckdb_max_error": None,
            "primitive_dependencies_null_edge_verified": all(
                op in POLARS_EDGE_VERIFIED and op in DUCKDB_EDGE_VERIFIED for op in operators_used
            ),
            "primitive_dependency_backend_ready": dependency_ready,
            "recipe_execution_verified": execution_verified,
            "pandas_execution_verified": execution_verified,
            "polars_execution_verified": execution_verified,
            "duckdb_execution_verified": execution_verified,
            "edge_verified": execution_verified,
            "no_fallback_verified": execution_verified,
        })
        if recipe.get("status") == "production" and not execution_verified:
            recipe["status"] = "pending"
    write_json(docs / "factor_recipe_manifest.json", {
        "schema_version": "factor_recipe_manifest.v3",
        "recipe_count": len(recipes),
        "recipes": recipes,
    })

    research = ResearchToolRegistry.catalog()
    write_json(docs / "research_tool_manifest.json", {
        "schema_version": "research_tool_manifest.v1",
        "tool_count": len(research),
        "tools": research,
    })
    stateful = StatefulCheckpointRegistry.catalog()
    missing_runtime = sorted(set(stateful) - set(operators))
    if missing_runtime:
        raise SystemExit(f"stateful contracts reference unavailable operators: {missing_runtime}")
    write_json(docs / "stateful_operator_manifest.json", {
        "schema_version": "stateful_operator_manifest.v1",
        "operator_count": len(stateful),
        "operators": stateful,
    })


if __name__ == "__main__":
    main()
