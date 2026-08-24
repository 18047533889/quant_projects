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

from factor_engine.mining.direct_use import (  # noqa: E402
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


def _current_head() -> str:
    """Return the exact commit the checked artifacts must be bound to."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _load_json_object(path: Path, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON artifact {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"artifact root must be an object: {path}")
        return None
    return payload


def _canonical_set(
    payload: dict[str, object],
    *,
    path: Path,
    collection_key: str,
    object_rows: bool,
    errors: list[str],
) -> set[str] | None:
    rows = payload.get(collection_key)
    if not isinstance(rows, list):
        errors.append(f"{path}: {collection_key} must be a list")
        return None

    names: list[str] = []
    for index, row in enumerate(rows):
        value = row.get("canonical") if object_rows and isinstance(row, dict) else row
        if not isinstance(value, str) or not value:
            errors.append(f"{path}: invalid canonical at {collection_key}[{index}]")
            return None
        names.append(value)

    declared_count = payload.get("count")
    if type(declared_count) is not int or declared_count != len(rows):
        errors.append(
            f"{path}: count={declared_count!r} does not match "
            f"{collection_key} length={len(rows)}"
        )
    canonical_set = set(names)
    if len(canonical_set) != len(names):
        errors.append(f"{path}: duplicate canonicals in {collection_key}")
    return canonical_set


def check_direct_artifacts(*, out_dir: Path, docs_dir: Path) -> list[str]:
    """Validate persisted direct-use artifacts without loading the registry."""
    artifacts = {
        "catalog": (
            out_dir / "direct_mining_catalog.json",
            "factor_engine.r18.direct_mining_catalog.v1",
            "operators",
            True,
        ),
        "manifest": (
            out_dir / "direct_mining_manifest.json",
            "factor_engine.r18.direct_mining_manifest.v1",
            "operators",
            False,
        ),
        "matrix": (
            docs_dir / "R18_DIRECT_USE_MATRIX.json",
            "factor_engine.r18.direct_use_matrix.v1",
            "rows",
            True,
        ),
        "recipes": (
            docs_dir / "R18_OPERATOR_SMOKE_RECIPES.json",
            "factor_engine.r18.operator_smoke_recipes.v1",
            "recipes",
            True,
        ),
    }
    errors: list[str] = []
    try:
        head = _current_head()
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"unable to resolve current HEAD: {exc}"]
    if not head:
        return ["unable to resolve current HEAD: empty git revision"]

    sets: dict[str, set[str]] = {}
    payloads: dict[str, dict[str, object]] = {}
    for name, (path, schema_version, collection_key, object_rows) in artifacts.items():
        payload = _load_json_object(path, errors)
        if payload is None:
            continue
        payloads[name] = payload
        if payload.get("schema_version") != schema_version:
            errors.append(
                f"{path}: schema_version={payload.get('schema_version')!r} "
                f"does not match expected {schema_version!r}"
            )
        fingerprint = payload.get("fingerprint")
        artifact_head = fingerprint.get("head") if isinstance(fingerprint, dict) else None
        if artifact_head != head:
            errors.append(
                f"{path}: fingerprint.head={artifact_head!r} does not match HEAD={head!r}"
            )
        artifact_dirty = fingerprint.get("dirty") if isinstance(fingerprint, dict) else None
        if artifact_dirty is not False:
            errors.append(f"{path}: fingerprint.dirty must be exactly false")
        canonical_set = _canonical_set(
            payload,
            path=path,
            collection_key=collection_key,
            object_rows=object_rows,
            errors=errors,
        )
        if canonical_set is not None:
            sets[name] = canonical_set

    if "catalog" in sets and "matrix" in sets:
        matrix_direct_rows = {
            row["canonical"]: row
            for row in payloads["matrix"]["rows"]
            if isinstance(row, dict)
            and isinstance(row.get("canonical"), str)
            and isinstance(row.get("direct_use_status"), str)
            and row["direct_use_status"].startswith("direct_")
        }
        if sets["catalog"] != set(matrix_direct_rows):
            errors.append("catalog canonical set does not match DIRECT_* matrix rows")
        catalog_rows = {
            row["canonical"]: row
            for row in payloads["catalog"]["operators"]
            if isinstance(row, dict) and isinstance(row.get("canonical"), str)
        }
        metadata_fields = (
            "direct_use_status",
            "directly_usable",
            "production_certified",
        )
        for canonical in sorted(sets["catalog"] & set(matrix_direct_rows)):
            for field in metadata_fields:
                catalog_value = catalog_rows[canonical].get(field)
                matrix_value = matrix_direct_rows[canonical].get(field)
                if catalog_value != matrix_value:
                    errors.append(
                        f"catalog/matrix metadata mismatch for {canonical}.{field}: "
                        f"catalog={catalog_value!r} matrix={matrix_value!r}"
                    )
    if "catalog" in sets and "recipes" in sets and sets["catalog"] != sets["recipes"]:
        errors.append("smoke recipe canonical set does not match catalog")
    if "catalog" in sets and "manifest" in sets:
        catalog_rows = payloads["catalog"]["operators"]
        eligible = {
            row["canonical"]
            for row in catalog_rows
            if isinstance(row, dict)
            and row.get("directly_usable") is True
            and row.get("production_certified") is True
        }
        if sets["manifest"] != eligible:
            errors.append("manifest canonical set does not match eligible catalog rows")
    return errors


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
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

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
        from factor_engine.cleaned_operators.registry import OperatorRegistry

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
    ap.add_argument(
        "--check",
        action="store_true",
        help="validate required direct artifacts without loading the registry",
    )
    args = ap.parse_args()

    if args.check:
        errors = check_direct_artifacts(
            out_dir=Path(args.out),
            docs_dir=Path(args.docs),
        )
        if errors:
            for error in errors:
                print(f"CHECK FAIL: {error}", file=sys.stderr)
            return 1
        print("direct mining artifacts are current and consistent")
        return 0

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
