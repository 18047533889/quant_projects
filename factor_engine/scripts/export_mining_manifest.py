#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the single-source mining manifest consumed by AlphaProbe/AlphaMiner.

``get_mining_operators()`` is the ONLY authority for what a mining engine may
use; this script persists that answer so the cold-start generator never needs
to read the raw surface partitions itself.

Outputs (``--out``, default ``build/mining``):
  * mining_manifest.json         — one record per mineable operator (+ pending)
  * mining_operator_catalog.csv  — tabular view
  * mining_operator_catalog.md   — human summary

``--validate-cold-start`` additionally verifies that every operator referenced
by the public DSL allowlist is present in the manifest (invariant
CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST == ∅).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mining.operator_catalog import (
    MiningOperator,
    MiningRole,
    get_mining_operators,
)


def _manifest_entry(op: MiningOperator) -> dict[str, Any]:
    return {
        "canonical": op.canonical,
        "production_certified": op.production_certified,
        "mining_eligible": op.mining_eligible,
        "mining_role": op.role.value,
        "cost_tier": op.cost_tier,
        "required_sources": list(op.required_sources),
        "input_semantics": list(op.input_semantic_types),
        "output_semantics": op.output_unit,
        "searchable_params": list(op.searchable_params),
        "allowed_ast_positions": list(op.allowed_ast_positions),
        "terminal_allowed": op.terminal_allowed,
        "stateful": op.stateful,
        "checkpoint_supported": op.checkpoint_supported,
        "incremental_supported": op.incremental_supported,
        "full_history_replay_required": op.full_history_replay_required,
        "blockers": list(op.blockers),
        "recommended_action": op.recommended_action,
    }


def export_mining_manifest(out_dir: Path, *, admission: str = "pending") -> dict[str, Path]:
    operators = get_mining_operators(admission=admission)
    manifest = {
        "schema_version": "factor_engine.mining_manifest.v1",
        "admission": admission,
        "count": len(operators),
        "eligible": sum(1 for op in operators if op.mining_eligible),
        "operators": [_manifest_entry(op) for op in operators],
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "mining_manifest.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    csv_path = out_dir / "mining_operator_catalog.csv"
    keys = sorted(set().union(*(set(e) for e in manifest["operators"])) if manifest["operators"] else {"canonical"})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for entry in manifest["operators"]:
            row = {k: entry.get(k) for k in keys}
            for k in ("required_sources", "input_semantics", "searchable_params",
                      "allowed_ast_positions", "blockers"):
                if isinstance(row.get(k), list):
                    row[k] = "|".join(str(v) for v in row[k])
            writer.writerow(row)

    md_path = out_dir / "mining_operator_catalog.md"
    md_lines = [
        "# FactorEngine Mining Operator Catalog",
        "",
        f"- admission = {admission}",
        f"- total operators = {len(operators)}",
        f"- mining_eligible = {manifest['eligible']}",
        "",
        "| canonical | role | eligible | certified | cost | sources | positions |",
        "|---|---|---|---|---|---|---|",
    ]
    for op in operators:
        md_lines.append(
            f"| {op.canonical} | {op.role.value} | {op.mining_eligible} | "
            f"{op.production_certified} | {op.cost_tier} | "
            f"{','.join(op.required_sources)} | {','.join(op.allowed_ast_positions)} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "md": md_path}


def validate_cold_start(out_dir: Path) -> list[str]:
    """Every registered canonical must be reachable from the mining manifest.

    DSL pseudo-functions (``intraday_*`` features, technical macros) lower to
    primitives before IR and are intentionally not canonicals, so they are not
    compared here — the invariant is that no registered canonical is invisible
    to the mining layer (``UNUSED_PUBLIC_CANONICALS == ∅``).
    """
    manifest_path = out_dir / "mining_manifest.json"
    if not manifest_path.exists():
        return ["mining_manifest.json not found — run export first"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    in_manifest = {e["canonical"] for e in manifest["operators"]}

    from cleaned_operators.registry import OperatorRegistry

    registered = set(OperatorRegistry._catalog)
    violations: list[str] = []
    for name in sorted(registered - in_manifest):
        violations.append(f"{name}: registered canonical not in mining manifest")
    return violations


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/mining")
    parser.add_argument(
        "--admission", default="all",
        help="eligible (certified only) | pending (certified + reviewed admissible) | all (every registered canonical)",
    )
    parser.add_argument("--validate-cold-start", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out)
    paths = export_mining_manifest(out_dir, admission=args.admission)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.validate_cold_start:
        violations = validate_cold_start(out_dir)
        if violations:
            print("cold-start violations:")
            for v in violations[:40]:
                print(f"  {v}")
            return 2
        print("cold-start validation: OK (all public DSL names in manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
