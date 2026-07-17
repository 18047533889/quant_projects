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
from factor_recipes.verification import verify_recipe  # noqa: E402
from research_tools.registry import ResearchToolRegistry  # noqa: E402
from stateful_contract import StatefulCheckpointRegistry  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    load_all()
    docs = FACTOR_ENGINE / "docs"
    fields = sorted(formula_field_names())
    operators = OperatorRegistry.catalog()
    collisions = sorted(set(fields) & set(operators))
    if collisions:
        raise SystemExit(f"field/operator collisions: {collisions}")

    write_json(docs / "field_manifest.json", {
        "schema_version": "field_manifest.v1",
        "source": "canonical_data_fields.json",
        "field_count": len(fields),
        "fields": fields,
    })
    write_json(docs / "operator_manifest.json", {
        "schema_version": "operator_manifest.v2",
        "operator_count": len(operators),
        "operators": operators,
    })

    recipes = FactorRecipeRegistry.catalog()
    compiler = RecipeCompiler(allowed_statuses=("production", "optional", "experimental"))
    execution_failures: list[str] = []
    for name, recipe in recipes.items():
        bindings = {parameter: parameter for parameter in recipe["parameters"]}
        recipe["expanded_expression"] = compiler.expand(name, bindings)
        plan = compiler.compile_batch({name: (name, bindings)})
        recipe["compiled_node_count"] = len(plan.nodes)
        recipe["compile_status"] = "verified"
        if recipe["status"] == "production":
            verification = verify_recipe(name).to_dict()
            recipe["backend_verification"] = verification
            if not verification["pandas_execution_verified"]:
                execution_failures.append(f"{name}: {verification['error']}")
        else:
            recipe["backend_verification"] = {
                "pandas_execution_verified": False,
                "polars_plan_capable": False,
                "duckdb_plan_capable": False,
                "missing_polars_operators": [],
                "missing_duckdb_operators": [],
                "error": "not certified: non-production recipe",
            }
    if execution_failures:
        raise SystemExit("production recipe execution failures:\n- " + "\n- ".join(execution_failures))
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
        "schema_version": "stateful_operator_manifest.v2",
        "operator_count": len(stateful),
        "operators": stateful,
    })


if __name__ == "__main__":
    main()
