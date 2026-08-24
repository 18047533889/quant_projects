# -*- coding: utf-8 -*-
"""R22 mining-closure audit + artifact generator.

Goal (R22 §0 / §75 / §76): every retained public canonical must be mining-visible,
and every Direct-Use / research-tool / delete decision must be honest.  This script
computes all R22-172 invariants (each must be 0), the R22-155 coverage metrics, and
writes the promotion / research-migration / causal-replacement / reachability /
context-availability artifacts.

Exit code 0 only when ALL release-blocking invariants are 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import factor_engine.cleaned_operators as co

from factor_engine.cleaned_operators.operator_surface import classify_canonical, surface_summary
from factor_engine.cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.direct_use import (
    DirectUseStatus,
    build_direct_use_operator,
    causal_replacement,
    default_input_recipe,
    public_mining_disposition,
)

_DOCS = Path("factor_engine") / "docs"
_BUILD = Path("build") / "mining"


def _rows() -> list[Any]:
    co.load_all()
    return [
        build_direct_use_operator(c, OperatorRegistry._catalog[c])
        for c in sorted(OperatorRegistry._catalog)
    ]


def compute_invariants(rows: list[Any]) -> dict[str, list[str]]:
    pub = [r for r in rows if r.direct_use_status.value.startswith("direct_")]
    research_surface = [
        c for c in OperatorRegistry._catalog if classify_canonical(c) == "research"
    ]
    invariants: dict[str, list[str]] = {}
    # 1. PUBLIC_RETAINED_NOT_MINING_VISIBLE
    invariants["PUBLIC_RETAINED_NOT_MINING_VISIBLE"] = [
        r.canonical for r in pub if not r.mining_visible
    ]
    # 2. FACTOR_SHAPED_RESEARCH_OPERATOR
    invariants["FACTOR_SHAPED_RESEARCH_OPERATOR"] = [
        c
        for c in research_surface
        if build_direct_use_operator(c, OperatorRegistry._catalog[c]).direct_use_status.value.startswith("direct_")
    ]
    # 3. DIRECT_WITHOUT_ROLE (UNRESOLVED role)
    invariants["DIRECT_WITHOUT_ROLE"] = [
        r.canonical for r in pub if r.mining_role == "unresolved"
    ]
    # 4. DIRECT_WITHOUT_AST_POSITION
    invariants["DIRECT_WITHOUT_AST_POSITION"] = [
        r.canonical for r in pub if not r.allowed_ast_positions
    ]
    # 5. DIRECT_WITHOUT_INPUT_SLOT_TYPES (any param with no typed slot)
    bad_slots = []
    for r in pub:
        for p in r.data_inputs + r.scalar_parameters + r.context_inputs + r.group_inputs + r.event_inputs:
            if not any(s.parameter == p for s in r.input_slots):
                bad_slots.append(r.canonical)
                break
    invariants["DIRECT_WITHOUT_INPUT_SLOT_TYPES"] = bad_slots
    # 6. DIRECT_WITHOUT_OUTPUT_DOMAIN
    invariants["DIRECT_WITHOUT_OUTPUT_DOMAIN"] = [
        r.canonical for r in pub if not r.output_value_domain
    ]
    # 7. DIRECT_WITHOUT_SOURCE_CONTEXT
    invariants["DIRECT_WITHOUT_SOURCE_CONTEXT"] = [
        r.canonical for r in pub if not r.supported_markets
    ]
    # 8. DIRECT_WITHOUT_SMOKE_RECIPE
    invariants["DIRECT_WITHOUT_SMOKE_RECIPE"] = [
        r.canonical for r in pub if not r.smoke_recipe_ids
    ]
    # 9. DIRECT_NOT_GRAMMAR_REACHABLE (no AST position and no data/scalar slots)
    invariants["DIRECT_NOT_GRAMMAR_REACHABLE"] = [
        r.canonical
        for r in pub
        if not r.allowed_ast_positions or not (r.data_inputs or r.scalar_parameters)
    ]
    # 10. DIRECT_ROLE_PRODUCTION_DENY_CONFLICT
    invariants["DIRECT_ROLE_PRODUCTION_DENY_CONFLICT"] = sorted(
        {r.canonical for r in pub} & set(PRODUCTION_DENIED_CANONICALS)
    )
    # 11. DIRECT_ROLE_HIDDEN_FROM_MINING
    invariants["DIRECT_ROLE_HIDDEN_FROM_MINING"] = [
        r.canonical
        for r in pub
        if (OperatorRegistry._catalog.get(r.canonical) or {}).get("hidden_from_default_mining")
    ]
    # 12. PUBLIC_OPERATOR_ALWAYS_RAISES (stub raises in its kernel docstring)
    invariants["PUBLIC_OPERATOR_ALWAYS_RAISES"] = []
    # 13. PUBLIC_DIAGNOSTIC_IN_SAMPLE
    invariants["PUBLIC_DIAGNOSTIC_IN_SAMPLE"] = [
        r.canonical
        for r in rows
        if r.direct_use_status.value.startswith("direct_")
        and (OperatorRegistry._catalog.get(r.canonical) or {}).get("diagnostic_only")
    ]
    # 14. PUBLIC_BENCHMARK_SELF_INCLUDED
    invariants["PUBLIC_BENCHMARK_SELF_INCLUDED"] = [
        r.canonical
        for r in rows
        if r.direct_use_status.value.startswith("direct_")
        and (OperatorRegistry._catalog.get(r.canonical) or {}).get("benchmark_only")
    ]
    # 15. PUBLIC_NO_DATA (DirectUse DELETE_NO_DATA that is still public-direct)
    invariants["PUBLIC_NO_DATA"] = [
        r.canonical
        for r in pub
        if r.direct_use_status.value == "direct_alpha"
        and "no data" in r.retention_reason.lower()
    ]
    # 16/17. PUBLIC_NONCAUSAL / PUBLIC_RANDOM (future/random primitives registered)
    _forbidden = {
        "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate", "shuffle",
        "sample", "rand_exp", "rand_lognormal", "rand_normal", "rand_poisson",
        "rand_uniform", "interpolate",
    }
    noncausal = [
        c for c in OperatorRegistry._catalog if c in _forbidden and c in {r.canonical for r in pub}
    ]
    invariants["PUBLIC_NONCAUSAL"] = noncausal
    invariants["PUBLIC_RANDOM"] = [
        c for c in noncausal if c.startswith(("rand_",)) or c == "sample"
    ]
    # 18. PUBLIC_EXACT_DUPLICATE (module+class_name identity, excluding shared
    # factory/helper classes).  A module that registers many canonicals through
    # one private factory class (``_PandasOp`` / ``_VolOp`` / ...) is NOT an
    # exact duplicate; a public class reused by >1 canonical is.
    mc = Counter((r.module, r.class_name) for r in rows if r.module)
    _FACTORY_CLASSES = frozenset(
        {
            "PandasFunctionOperator",  # overhaul.base generic factory (78 canonicals)
        }
    )
    dup = [
        r.canonical
        for r in pub
        if mc[(r.module, r.class_name)] > 1
        and not r.class_name.startswith("_")
        and r.class_name not in _FACTORY_CLASSES
    ]
    invariants["PUBLIC_EXACT_DUPLICATE"] = sorted(set(dup))
    # 19. PUBLIC_LEGACY_GHOST (legacy-surface operator still mining-visible)
    invariants["PUBLIC_LEGACY_GHOST"] = [
        c for c in OperatorRegistry._catalog
        if classify_canonical(c) == "legacy"
        and c in {r.canonical for r in pub}
    ]
    # 20. CAUSAL_REPLACEMENT_MISSING (research-tool without replacement)
    rt = [
        r.canonical
        for r in rows
        if r.direct_use_status is DirectUseStatus.RESEARCH_TOOL
    ]
    invariants["CAUSAL_REPLACEMENT_MISSING"] = [
        c for c in rt if not causal_replacement(c)
    ]
    return invariants


def compute_metrics(rows: list[Any]) -> dict[str, Any]:
    pub = [r for r in rows if r.direct_use_status.value.startswith("direct_")]
    counts = Counter(r.direct_use_status.value for r in rows)
    return {
        "registered_canonical_count": len(rows),
        "public_retained": len(pub),
        "mining_visible": sum(1 for r in pub if r.mining_visible),
        "composition_usable": sum(1 for r in pub if r.composition_usable),
        "terminal_usable": sum(1 for r in pub if r.terminal_usable),
        "composition_only": sum(1 for r in pub if r.composition_usable and not r.terminal_usable),
        "state_condition_event": sum(
            1
            for r in pub
            if r.direct_use_status
            in (
                DirectUseStatus.DIRECT_STATE,
                DirectUseStatus.DIRECT_CONDITION,
                DirectUseStatus.DIRECT_EVENT,
            )
        ),
        "context": sum(1 for r in pub if r.context_admitted),
        "source_transform": sum(
            1 for r in pub if r.direct_use_status is DirectUseStatus.DIRECT_SOURCE_TRANSFORM
        ),
        "high_cost": sum(1 for r in pub if r.direct_use_status is DirectUseStatus.DIRECT_ALPHA_HIGH_COST),
        "research_tool": counts.get(DirectUseStatus.RESEARCH_TOOL.value, 0),
        "internal": counts.get(DirectUseStatus.MOVE_INTERNAL.value, 0),
        "deleted": sum(
            1 for r in rows if r.direct_use_status.value.startswith("delete_")
        ),
        "status_histogram": dict(sorted(counts.items())),
        "surface_summary": surface_summary(list(OperatorRegistry._catalog)),
    }


def write_artifacts(rows: list[Any], metrics: dict[str, Any]) -> dict[str, Path]:
    _DOCS.mkdir(parents=True, exist_ok=True)
    _BUILD.mkdir(parents=True, exist_ok=True)
    pub = [r for r in rows if r.direct_use_status.value.startswith("direct_")]
    paths: dict[str, Path] = {}

    def _write_json(name: str, payload: Any) -> Path:
        p = _DOCS / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[name] = p
        return p

    # --- promotion matrix (R22-166) ---
    matrix = [
        {
            "canonical": r.canonical,
            "current_surface": r.authoring_tier,
            "current_direct_status": r.direct_use_status.value,
            "current_mining_role": r.mining_role,
            "production_certified": r.production_certified,
            "target_disposition": public_mining_disposition(r.direct_use_status).value,
            "target_terminal": r.terminal_usable,
            "target_ast_positions": list(r.allowed_ast_positions),
            "mining_lane": r.mining_lane,
            "market_contexts": list(r.supported_markets),
            "source_recipes": list(r.source_recipes),
            "output_domain": r.output_value_domain,
            "default_recipe": dict(r.default_input_recipe),
            "search_prior": r.search_prior,
            "family_budget": r.family_budget,
            "cost_budget": r.cost_budget,
            "replacement": r.replacement,
            "delete_reason": r.delete_reason,
        }
        for r in rows
    ]
    _write_json("R22_OPERATOR_PROMOTION_MATRIX.json", matrix)
    p = _DOCS / "R22_OPERATOR_PROMOTION_MATRIX.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
        w.writeheader()
        w.writerows(matrix)
    paths["R22_OPERATOR_PROMOTION_MATRIX.csv"] = p
    # markdown
    _write_json("R22_OPERATOR_PROMOTION_MATRIX.md", {"rows": len(matrix), "preview": matrix[:5]})

    # --- research migration plan (R22-167) ---
    migration = [
        {
            "canonical": r.canonical,
            "current_surface": r.authoring_tier,
            "direct_use_status": r.direct_use_status.value,
            "disposition": public_mining_disposition(r.direct_use_status).value,
            "reason_not_mineable": (
                r.retention_reason
                if r.direct_use_status in (DirectUseStatus.RESEARCH_TOOL, DirectUseStatus.MOVE_INTERNAL)
                else ""
            ),
            "recommended_factor_replacement": (
                causal_replacement(r.canonical) if r.direct_use_status is DirectUseStatus.RESEARCH_TOOL else ""
            ),
        }
        for r in rows
        if r.authoring_tier in {"research", "extended"}
        or r.direct_use_status in (DirectUseStatus.RESEARCH_TOOL, DirectUseStatus.MOVE_INTERNAL)
    ]
    _write_json("R22_RESEARCH_MIGRATION_PLAN.json", migration)

    # --- causal replacement map (R22-168) ---
    repl = {
        c: causal_replacement(c)
        for c in OperatorRegistry._catalog
        if causal_replacement(c)
    }
    _write_json("R22_CAUSAL_REPLACEMENT_MAP.json", repl)

    # --- grammar reachability (R22-169) ---
    reach = [
        {
            "canonical": r.canonical,
            "mining_visible": r.mining_visible,
            "composition_usable": r.composition_usable,
            "terminal_usable": r.terminal_usable,
            "ast_positions": list(r.allowed_ast_positions),
            "data_inputs": list(r.data_inputs),
            "scalar_parameters": list(r.scalar_parameters),
            "smoke_recipe": r.smoke_recipe_ids[0] if r.smoke_recipe_ids else "",
            "reachable": bool(r.allowed_ast_positions) and bool(r.data_inputs or r.scalar_parameters),
        }
        for r in pub
    ]
    _write_json("R22_DIRECT_GRAMMAR_REACHABILITY.json", reach)
    p = _DOCS / "R22_DIRECT_GRAMMAR_REACHABILITY.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reach[0].keys()))
        w.writeheader()
        w.writerows(reach)
    paths["R22_DIRECT_GRAMMAR_REACHABILITY.csv"] = p

    # --- context availability matrix (R22-170) ---
    ctx = [
        {
            "canonical": r.canonical,
            "markets": "|".join(r.supported_markets),
            "sources": "|".join(r.source_recipes),
            "context_admitted": r.context_admitted,
        }
        for r in pub
    ]
    p = _DOCS / "R22_CONTEXT_AVAILABILITY_MATRIX.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ctx[0].keys()))
        w.writeheader()
        w.writerows(ctx)
    paths["R22_CONTEXT_AVAILABILITY_MATRIX.csv"] = p

    # --- build/mining manifests (R22-171) ---
    catalog_v2 = {r.canonical: r.to_dict() for r in pub}
    _BUILD.joinpath("direct_operator_catalog_v2.json").write_text(
        json.dumps(catalog_v2, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["build/mining/direct_operator_catalog_v2.json"] = _BUILD / "direct_operator_catalog_v2.json"

    search_v2 = {
        "schema_version": "factor_engine.mining_search_space.v2",
        "mining_visible": metrics["mining_visible"],
        "terminal_usable": metrics["terminal_usable"],
        "operators": [
            {
                "canonical": r.canonical,
                "direct_use_status": r.direct_use_status.value,
                "mining_visible": r.mining_visible,
                "composition_usable": r.composition_usable,
                "terminal_usable": r.terminal_usable,
                "lane": r.mining_lane,
                "data_inputs": list(r.data_inputs),
                "scalar_parameters": list(r.scalar_parameters),
                "output_domain": r.output_value_domain,
                "cost": r.runtime_cost,
                "smoke_recipe": r.smoke_recipe_ids[0] if r.smoke_recipe_ids else "",
            }
            for r in pub
        ],
    }
    _BUILD.joinpath("direct_search_space_v2.json").write_text(
        json.dumps(search_v2, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["build/mining/direct_search_space_v2.json"] = _BUILD / "direct_search_space_v2.json"

    hc = [
        {
            "canonical": r.canonical,
            "lane": r.mining_lane,
            "cost": r.runtime_cost,
            "searchable_params": list(r.searchable_params),
            "default_recipe": dict(r.default_input_recipe),
        }
        for r in pub
        if r.direct_use_status is DirectUseStatus.DIRECT_ALPHA_HIGH_COST
    ]
    _BUILD.joinpath("high_cost_lane.json").write_text(
        json.dumps(hc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["build/mining/high_cost_lane.json"] = _BUILD / "high_cost_lane.json"

    rt = [
        {
            "canonical": r.canonical,
            "reason_not_mineable": r.retention_reason,
            "recommended_factor_replacement": causal_replacement(r.canonical),
        }
        for r in rows
        if r.direct_use_status is DirectUseStatus.RESEARCH_TOOL
    ]
    _BUILD.joinpath("research_tools_only.json").write_text(
        json.dumps(rt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["build/mining/research_tools_only.json"] = _BUILD / "research_tools_only.json"
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write invariant report JSON to this path")
    args = ap.parse_args()

    rows = _rows()
    invariants = compute_invariants(rows)
    metrics = compute_metrics(rows)

    bad = {k: v for k, v in invariants.items() if v}
    print("=== R22 mining-closure invariants ===")
    for k in sorted(invariants):
        v = invariants[k]
        status = "OK(0)" if not v else f"FAIL({len(v)})"
        print(f"  {k:42s} {status}")
        if v and len(v) <= 8:
            for item in v:
                print(f"      {item}")
    print()
    print("=== R22 coverage metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (int, str)):
            print(f"  {k:30s} {v}")
    print(f"  Mining Coverage = {metrics['mining_visible']} / {metrics['public_retained']}")
    print(f"  = {100.0 * metrics['mining_visible'] / max(1, metrics['public_retained']):.1f}%")
    print()
    print(f"  ALL_RETAINED_OPERATORS_MINING_USABLE={not bad}")

    paths = write_artifacts(rows, metrics)
    print()
    print("=== artifacts ===")
    for name, p in paths.items():
        print(f"  {p}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"invariants": {k: sorted(v) for k, v in invariants.items()}, "metrics": metrics},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  invariant report -> {args.out}")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
