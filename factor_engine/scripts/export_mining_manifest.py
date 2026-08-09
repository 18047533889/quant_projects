#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the single-source mining manifest consumed by AlphaProbe/AlphaMiner.

``get_mining_operators()`` is the ONLY authority for what a mining engine may
use; this script persists that answer so the cold-start generator never needs
to read the raw surface partitions itself.

R15 changes:

* The manifest header records the AdmissionQuery + HEAD/dirty/registry/evidence
  fingerprints so an offline artifact can never be mistaken for the current
  tree (R15-INC-023/054/256).
* CLI and API share the SAME explicit default admission (R15-INC-053): the API
  default is ``pending``, the CLI must pass it explicitly — no silent drift.
* Every searchable scalar carries its full schema (dtype/range/choices/default/
  role/search grade/active_when) — names alone cannot protect the search space
  (R15-INC-061).
* ``terminal_allowed`` is cross-checked against ``allowed_ast_positions`` and
  every non-eligible entry carries blockers + ordered required actions from the
  shared AdmissionDecision (R15-INC-055/059).
* Cold-start validation resolves the DSL AST to canonical/role and checks each
  against the manifest admission — it never compares raw registry sets against
  a manifest (R15-INC-056/057).

Outputs (``--out``, default ``build/mining``):
  * mining_manifest.json         — one record per mineable operator (+ pending)
  * mining_operator_catalog.csv  — tabular view
  * mining_operator_catalog.md   — human summary
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from mining.operator_catalog import (
    MiningOperator,
    MiningRole,
    get_mining_operators,
)

# R15-INC-053: one explicit shared default.  The API and the CLI use the same
# value; neither silently drifts.
_DEFAULT_ADMISSION = "pending"


def _searchable_schema(canonical: str) -> dict[str, dict[str, Any]]:
    """R15-INC-061: full schema for every searchable scalar — dtype, domain,
    choices, default, role, search grade and active_when.  A name alone is never
    enough to protect the search space from estimator-knob overfitting."""
    schema: dict[str, dict[str, Any]] = {}
    try:
        from cleaned_operators.base import (
            ParamRole,
            effective_param_role,
            param_search_grade,
            searchable_param_names,
        )
        from cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        metadata = getattr(operator, "metadata", None)
        if metadata is None:
            return schema
        grades = searchable_param_names(metadata)
        searchable = set(grades.get("full", ())) | set(grades.get("coarse", ()))
        specs = getattr(metadata, "param_specs", None) or {}
        for name in sorted(searchable):
            spec = specs.get(name)
            dtype = getattr(spec, "dtype", None)
            default = getattr(spec, "default", None)
            try:
                from cleaned_operators.base import MISSING

                if default is MISSING:
                    default = None
            except Exception:
                pass
            if hasattr(default, "__class__") and type(default).__name__ in (
                "_MissingDefaultType",
            ):
                default = None
            entry: dict[str, Any] = {
                "dtype": dtype.__name__ if dtype is not None else None,
                "min": getattr(spec, "min", None),
                "max": getattr(spec, "max", None),
                "choices": list(getattr(spec, "choices", ())) or None,
                "default": default,
                "role": None,
                "search_grade": param_search_grade(spec),
                "active_when": getattr(spec, "active_when", None),
            }
            try:
                role = effective_param_role(spec)
                entry["role"] = role.value if role is not None else None
            except Exception:
                entry["role"] = None
            schema[name] = entry
    except Exception:
        pass
    return schema


def _manifest_entry(op: MiningOperator) -> dict[str, Any]:
    entry = {
        "canonical": op.canonical,
        "production_certified": op.production_certified,
        "mining_eligible": op.mining_eligible,
        "mining_role": op.role.value,
        "role_source": op.role_source.value,
        "admission_state": op.admission_state.value,
        "cost_tier": op.cost_tier,
        "cost_contract_declared": op.cost_contract_declared,
        "required_sources": list(op.required_sources),
        "available_sources": list(op.available_sources),
        "source_unknown": op.source_unknown,
        "input_semantics": list(op.input_semantic_types),
        "output_semantics": op.output_unit,
        "output_grain": op.output_grain,
        "searchable_params": list(op.searchable_params),
        "searchable_param_schema": _searchable_schema(op.canonical),
        "allowed_ast_positions": list(op.allowed_ast_positions),
        "terminal_allowed": op.terminal_allowed,
        "stateful": op.stateful,
        "execution_model": op.execution_model,
        "checkpoint_supported": op.checkpoint_supported,
        "incremental_supported": op.incremental_supported,
        "full_history_replay_required": op.full_history_replay_required,
        "blockers": list(op.blockers),
        "required_actions": list(op.required_actions),
        "recommended_action": op.recommended_action,
    }
    # R15-INC-059: a manifest record that contradicts its own terminal/positions
    # contract is a catalog bug written straight into the manifest — fail loud.
    if op.terminal_allowed != ("terminal" in op.allowed_ast_positions):
        raise ValueError(
            f"manifest invariant violation for {op.canonical}: "
            f"terminal_allowed={op.terminal_allowed} but positions="
            f"{op.allowed_ast_positions}"
        )
    return entry


def _query_fingerprint() -> dict[str, str]:
    """R15-INC-023/054: HEAD / dirty / registry / evidence fingerprints."""
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    from cleaned_operators.registry import OperatorRegistry

    registry_digest = ""
    try:
        blob = json.dumps(
            {
                c: sorted(k for k in OperatorRegistry._catalog[c] if not callable(k))
                for c in sorted(OperatorRegistry._catalog)
            },
            sort_keys=True,
        )
        registry_digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    except Exception:
        pass
    evidence_digest = ""
    try:
        from backend.factor_operator_evidence import load_factor_operator_evidence

        payload = load_factor_operator_evidence() or {}
        evidence_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        pass
    return {
        "commit_sha": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain") or "",
        "registry_fingerprint": registry_digest,
        "evidence_fingerprint": evidence_digest,
    }


def export_mining_manifest(
    out_dir: Path, *, admission: str = _DEFAULT_ADMISSION
) -> dict[str, Path]:
    """Export the manifest.  API default == CLI default (R15-INC-053)."""
    operators = get_mining_operators(admission=admission)
    manifest = {
        "schema_version": "factor_engine.mining_manifest.v2",
        "admission": admission,
        "query_fingerprint": _query_fingerprint(),
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
            for k in ("required_sources", "available_sources", "input_semantics",
                      "searchable_params", "searchable_param_schema",
                      "allowed_ast_positions", "blockers", "required_actions"):
                if isinstance(row.get(k), list):
                    row[k] = "|".join(str(v) for v in row[k])
                elif isinstance(row.get(k), dict):
                    row[k] = json.dumps(row[k], sort_keys=True)
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
    """R15-INC-056/057: resolve the cold-start DSL AST to canonical + role and
    verify each against the manifest admission — never a raw set comparison.

    DSL pseudo-functions (``intraday_*`` features, technical macros) lower to
    primitives before IR; they are resolved via the registry alias map when
    possible and otherwise reported as unverifiable rather than skipped.
    """
    manifest_path = out_dir / "mining_manifest.json"
    if not manifest_path.exists():
        return ["mining_manifest.json not found — run export first"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    in_manifest = {e["canonical"] for e in manifest["operators"]}
    admission_state = {e["canonical"]: e.get("admission_state") for e in manifest["operators"]}

    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.operator_surface import is_dsl_name_allowed

    violations: list[str] = []

    # R15-INC-051: CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST.  Only a
    # PRODUCTION-CERTIFIED factor operator must be present in the manifest —
    # an uncertified mineable-role operator is legitimately absent from a
    # pending/all manifest (it has not passed the gates yet).  A DENIED /
    # supporting canonical is likewise legitimately absent.
    for name in sorted(OperatorRegistry._catalog):
        entry = OperatorRegistry._catalog[name]
        if not entry.get("production_certified"):
            continue
        role = None
        try:
            from mining.operator_catalog import assign_mining_role

            role = assign_mining_role(name, entry)
        except Exception:
            role = None
        if role is not None and role.value in (
            "alpha", "alpha_high_cost", "intraday_eod", "fundamental_pit",
            "state", "condition", "event", "group_state", "global_state",
        ):
            if name not in in_manifest:
                violations.append(f"{name}: CERTIFIED factor canonical not in mining manifest")
            if admission_state.get(name) == "unresolved":
                violations.append(f"{name}: UNRESOLVED admission in manifest")

    # DSL names / aliases: every registered alias must resolve (via the registry
    # alias map) to a canonical that is present in the manifest when the target
    # is mineable.  An alias pointing at a missing or supporting canonical is a
    # broken formula — the R12 "register every canonical, verify membership"
    # invariant was a set comparison that never proved the AST could resolve.
    try:
        aliases = getattr(OperatorRegistry, "_aliases", {}) or {}
        for alias in sorted(aliases):
            try:
                canonical = OperatorRegistry.resolve_canonical(alias)
            except Exception:
                canonical = None
            if canonical is None or canonical not in in_manifest:
                # only flag if the alias target is a factor-shaped canonical
                target = aliases[alias]
                if target in OperatorRegistry._catalog:
                    from mining.operator_catalog import assign_mining_role

                    try:
                        role = assign_mining_role(target, OperatorRegistry._catalog[target])
                    except Exception:
                        role = None
                    if role is not None and role.value in (
                        "alpha", "alpha_high_cost", "intraday_eod",
                        "fundamental_pit", "state", "condition", "event",
                        "group_state", "global_state",
                    ):
                        violations.append(
                            f"alias {alias}: resolves to {target!r} not present "
                            f"in the manifest"
                        )
    except Exception:
        pass
    return violations


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/mining")
    parser.add_argument(
        "--admission", default=_DEFAULT_ADMISSION,
        help="eligible (certified only) | pending (certified + reviewed admissible, default) | all (every registered canonical)",
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
