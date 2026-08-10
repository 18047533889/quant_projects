#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R18 direct-use catalog / manifest / matrix exporter.

R18-096/098/110/129: produces the artifacts AlphaProbe / AlphaMiner / cold-start
actually consume — one source module (``mining.direct_use``) drives every
artifact so they can never drift from the code.

Outputs (``--out`` default ``build/mining``, plus ``--docs`` default ``docs``):
  * build/mining/direct_mining_catalog.json     — every DIRECT_* row (R18-110)
  * build/mining/direct_mining_manifest.json    — eligible DIRECT_* only
  * build/mining/research_operator_manifest.json— RESEARCH_TOOL rows
  * build/mining/remediation_backlog_manifest.json — DELETE_*/MOVE_* rows
  * docs/R18_DIRECT_USE_MATRIX.json / .csv / .md — every registered canonical
  * docs/R18_OPERATOR_SMOKE_RECIPES.json        — real smoke recipe per direct op
  * docs/R18_DELETE_PLAN.json / .md             — the deletion plan
  * docs/operator_migration_map.json            — alias/duplicate -> canonical

Every row carries a non-empty ``direct_use_status`` (R18-124: no UNRESOLVED /
UNKNOWN / PENDING may survive).  ``--strict`` fails the run on any unresolved row
or any direct row with an empty status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mining.direct_use import (  # noqa: E402
    DirectUseOperator,
    DirectUseStatus,
    build_direct_use_operator,
    direct_use_matrix_rows,
    direct_use_status_counts,
    retained_direct_rows,
)


def _fingerprint() -> dict[str, str]:
    head = ""
    dirty = ""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip()
    except Exception:
        pass
    return {"head": head, "dirty": bool(dirty)}


def _smoke_recipe_for(row: DirectUseOperator) -> dict[str, object]:
    """One honest smoke recipe per direct row (R18-071)."""
    recipe: dict[str, object] = {
        "canonical": row.canonical,
        "market": "ashare" if "ashare" in row.supported_markets else ("us" if row.supported_markets else ""),
        "role": row.mining_role,
        "direct_use_status": row.direct_use_status.value,
        "terminal_allowed": row.terminal_allowed,
        "input_bindings": row.default_input_recipe or {
            p: p for p in row.data_inputs
        },
        "parameter_values": {
            p: None for p in row.scalar_parameters[:4]
        },
        "required_context": {"sources": list(row.source_recipes)},
        "expected_output_kind": row.output_semantic_kind,
        "expected_grain": "daily",
        "expected_cardinality": row.output_cardinality,
        "minimum_coverage": 0.5,
        "expected_value_domain": "",
    }
    return recipe


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_all(*, out_dir: Path, docs_dir: Path) -> dict[str, int]:
    """Build every R18 artifact.  Returns a summary dict."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    rows = direct_use_matrix_rows()
    direct = retained_direct_rows(rows)

    unresolved = [
        r for r in rows if r.direct_use_status.value in ("unresolved", "unknown", "pending")
    ]
    no_status = [r for r in rows if not r.direct_use_status.value]

    fp = _fingerprint()
    status_counts = direct_use_status_counts(rows)

    # ---- direct_mining_catalog.json (every DIRECT_* row) --------------------
    catalog_rows = [r.to_dict() for r in direct]
    for r, d in zip(direct, catalog_rows):
        d["smoke_recipe"] = _smoke_recipe_for(r)
    write_json(
        out_dir / "direct_mining_catalog.json",
        {
            "schema_version": "factor_engine.r18.direct_mining_catalog.v1",
            "fingerprint": fp,
            "count": len(catalog_rows),
            "direct_use_status_counts": status_counts,
            "operators": catalog_rows,
        },
    )

    # ---- direct_mining_manifest.json (eligible DIRECT_* only) ---------------
    eligible = [r for r in direct if r.directly_usable and r.production_certified]
    write_json(
        out_dir / "direct_mining_manifest.json",
        {
            "schema_version": "factor_engine.r18.direct_mining_manifest.v1",
            "fingerprint": fp,
            "count": len(eligible),
            "pending_count": 0,  # R18-005: the DEFAULT manifest is never pending
            "research_count": 0,
            "operators": [r.canonical for r in eligible],
        },
    )

    # ---- research_operator_manifest.json ------------------------------------
    research = [r for r in rows if r.direct_use_status is DirectUseStatus.RESEARCH_TOOL]
    write_json(
        out_dir / "research_operator_manifest.json",
        {
            "schema_version": "factor_engine.r18.research_operator_manifest.v1",
            "fingerprint": fp,
            "count": len(research),
            "operators": sorted(r.canonical for r in research),
        },
    )

    # ---- remediation_backlog_manifest.json (DELETE_*/MOVE_*) ----------------
    backlog = [r for r in rows if r.direct_use_status.value.startswith(("delete_", "move_"))]
    write_json(
        out_dir / "remediation_backlog_manifest.json",
        {
            "schema_version": "factor_engine.r18.remediation_backlog_manifest.v1",
            "fingerprint": fp,
            "count": len(backlog),
            "operators": [
                {
                    "canonical": r.canonical,
                    "direct_use_status": r.direct_use_status.value,
                    "delete_reason": r.delete_reason,
                    "replacement": r.replacement,
                }
                for r in backlog
            ],
        },
    )

    # ---- docs/R18_DIRECT_USE_MATRIX.json ------------------------------------
    write_json(
        docs_dir / "R18_DIRECT_USE_MATRIX.json",
        {
            "schema_version": "factor_engine.r18.direct_use_matrix.v1",
            "fingerprint": fp,
            "count": len(rows),
            "direct_use_status_counts": status_counts,
            "rows": [r.to_dict() for r in rows],
        },
    )

    # ---- docs/R18_DIRECT_USE_MATRIX.csv -------------------------------------
    cols = [
        "canonical", "direct_use_status", "mining_role", "terminal_allowed",
        "allowed_ast_positions", "mining_lane", "economic_effect_family",
        "semantic_redundancy_group", "data_inputs", "scalar_parameters",
        "context_inputs", "group_inputs", "event_inputs", "output_semantic_kind",
        "output_unit", "supported_markets", "source_recipes", "runtime_cost",
        "production_certified", "directly_usable", "retention_reason",
        "delete_reason", "replacement",
    ]
    with (docs_dir / "R18_DIRECT_USE_MATRIX.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [
                    r.canonical,
                    r.direct_use_status.value,
                    r.mining_role,
                    r.terminal_allowed,
                    "|".join(r.allowed_ast_positions),
                    r.mining_lane,
                    r.economic_effect_family,
                    r.semantic_redundancy_group,
                    "|".join(r.data_inputs),
                    "|".join(r.scalar_parameters),
                    "|".join(r.context_inputs),
                    "|".join(r.group_inputs),
                    "|".join(r.event_inputs),
                    r.output_semantic_kind,
                    r.output_unit or "",
                    "|".join(r.supported_markets),
                    "|".join(r.source_recipes),
                    r.runtime_cost,
                    r.production_certified,
                    r.directly_usable,
                    r.retention_reason,
                    r.delete_reason,
                    r.replacement,
                ]
            )

    # ---- docs/R18_DIRECT_USE_MATRIX.md --------------------------------------
    md_lines = [
        "# R18 Direct-Use Matrix",
        "",
        f"Fingerprint: `{fp['head']}` (dirty={fp['dirty']})",
        "",
        f"Registered canonicals: **{len(rows)}**  |  retained DIRECT_*: **{len(direct)}**",
        "",
        "## DirectUseStatus histogram",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for k, v in status_counts.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines.append("")
    md_lines.append("## Delete / remediation plan")
    md_lines.append("")
    md_lines.append("| canonical | status | replacement | reason |")
    md_lines.append("|---|---|---|---|")
    for r in backlog:
        md_lines.append(
            f"| {r.canonical} | {r.direct_use_status.value} | "
            f"{r.replacement or '-'} | {r.delete_reason or r.retention_reason} |"
        )
    (docs_dir / "R18_DIRECT_USE_MATRIX.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # ---- docs/R18_OPERATOR_SMOKE_RECIPES.json -------------------------------
    write_json(
        docs_dir / "R18_OPERATOR_SMOKE_RECIPES.json",
        {
            "schema_version": "factor_engine.r18.operator_smoke_recipes.v1",
            "fingerprint": fp,
            "count": len(direct),
            "recipes": [_smoke_recipe_for(r) for r in direct],
        },
    )

    # ---- docs/R18_DELETE_PLAN.json / .md ------------------------------------
    write_json(
        docs_dir / "R18_DELETE_PLAN.json",
        {
            "schema_version": "factor_engine.r18.delete_plan.v1",
            "fingerprint": fp,
            "count": len(backlog),
            "deletions": [
                {
                    "canonical": r.canonical,
                    "direct_use_status": r.direct_use_status.value,
                    "delete_reason": r.delete_reason,
                    "replacement": r.replacement,
                    "retention_reason": r.retention_reason,
                }
                for r in backlog
            ],
        },
    )
    md_lines = [
        "# R18 Delete / remediation plan",
        "",
        f"Fingerprint: `{fp['head']}` (dirty={fp['dirty']})",
        "",
        f"**{len(backlog)}** canonicals to delete / move.",
        "",
        "| canonical | status | replacement | delete reason |",
        "|---|---|---|---|",
    ]
    for r in backlog:
        md_lines.append(
            f"| {r.canonical} | {r.direct_use_status.value} | "
            f"{r.replacement or '-'} | {r.delete_reason or r.retention_reason} |"
        )
    (docs_dir / "R18_DELETE_PLAN.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # ---- docs/operator_migration_map.json -----------------------------------
    aliases: dict[str, str] = {}
    try:
        from cleaned_operators.registry import OperatorRegistry

        aliases = dict(OperatorRegistry._aliases)
    except Exception:
        pass
    migration: dict[str, dict[str, str]] = {}
    for r in backlog:
        if r.replacement:
            migration[r.canonical] = {"replacement": r.replacement, "kind": r.direct_use_status.value}
    for alias, target in aliases.items():
        migration.setdefault(alias, {"replacement": target, "kind": "alias"})
    write_json(
        docs_dir / "operator_migration_map.json",
        {
            "schema_version": "factor_engine.operator_migration_map.v1",
            "fingerprint": fp,
            "count": len(migration),
            "map": migration,
        },
    )

    # ---- docs/R18_DELETE_PLAN.md summary + totals ----------------------------
    return {
        "registered": len(rows),
        "direct_retained": len(direct),
        "eligible": len(eligible),
        "research": len(research),
        "remediation_backlog": len(backlog),
        "unresolved": len(unresolved),
        "no_status": len(no_status),
        **status_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build" / "mining"))
    ap.add_argument("--docs", default=str(REPO / "docs"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    summary = build_all(
        out_dir=Path(args.out),
        docs_dir=Path(args.docs),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and (summary["unresolved"] or summary["no_status"]):
        print(
            f"STRICT FAIL: unresolved={summary['unresolved']} "
            f"no_status={summary['no_status']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
