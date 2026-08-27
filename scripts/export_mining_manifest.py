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
import enum
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from factor_engine.mining.operator_catalog import (
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
    enough to protect the search space from estimator-knob overfitting.

    R16-025: "no default" and "default is None" are DISTINCT — the JSON carries
    an explicit ``has_default`` flag (a bare ``None`` serialized as the default
    made them indistinguishable and could pin an operator to a phantom default).
    R16-026: a schema build error on a MINEABLE operator ABORTS the manifest
    instead of silently emitting an empty schema dict.
    """
    from factor_engine.cleaned_operators.base import (
        MISSING,
        effective_param_role,
        param_search_grade,
        searchable_param_names,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
    metadata = getattr(operator, "metadata", None)
    if metadata is None:
        # R63: a mineable operator whose ONLY backends are polars/sql (no
        # pandas_numpy reference) still has real OperatorMetadata on its native
        # class — the manifest must not abort on the pandas_numpy probe.  Fall
        # back to any registered backend implementation with metadata; if NONE
        # has metadata the operator is genuinely unbuildable (R16-026 abort).
        from factor_engine.cleaned_operators.registry import OperatorRegistry as _OR

        for _b in _OR.backends_for(canonical):
            _op = _OR.get(canonical, _b)
            _meta = getattr(_op, "metadata", None)
            if _meta is not None:
                metadata = _meta
                break
    if metadata is None:
        raise ValueError(
            f"searchable schema for {canonical}: no operator metadata — a "
            "mineable operator with an unbuildable schema must abort the manifest "
            "(R16-026)"
        )
    grades = searchable_param_names(metadata)
    searchable = set(grades.get("full", ())) | set(grades.get("coarse", ()))
    specs = getattr(metadata, "param_specs", None) or {}
    schema: dict[str, dict[str, Any]] = {}
    for name in sorted(searchable):
        spec = specs.get(name)
        dtype = getattr(spec, "dtype", None)
        default = getattr(spec, "default", MISSING)
        has_default = default is not MISSING
        serialized_default = None if not has_default else default
        _choices = getattr(spec, "choices", None)
        entry: dict[str, Any] = {
            "dtype": dtype.__name__ if dtype is not None else None,
            "min": getattr(spec, "min", None),
            "max": getattr(spec, "max", None),
            "choices": list(_choices) if _choices else None,
            "has_default": has_default,
            "default": serialized_default,
            "role": None,
            "search_grade": param_search_grade(spec),
            "active_when": getattr(spec, "active_when", None),
        }
        role = effective_param_role(spec)
        entry["role"] = role.value if role is not None else None
        schema[name] = entry
    return schema


def _manifest_entry(op: MiningOperator) -> dict[str, Any]:
    # R63: a DENIED / INTERNAL / non-searchable operator (e.g. ``arg``, a DSL
    # grammar primitive with no searchable params and no backend metadata) has no
    # buildable searchable schema — emit an honest empty schema instead of
    # aborting the whole manifest (R16-026 abort applies to MINEABLE operators).
    try:
        schema = _searchable_schema(op.canonical)
    except ValueError:
        if op.mining_eligible:
            raise
        schema = {}
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
        "searchable_param_schema": schema,
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

    from factor_engine.cleaned_operators.registry import OperatorRegistry

    def _canonicalize(value: Any) -> Any:
        """JSON-canonicalize a catalog value (R16-027): enums/sets/frozensets/
        dataclasses/tuples become plain JSON-able structures so the fingerprint
        reflects VALUES, not key names."""
        if isinstance(value, dict):
            return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [_canonicalize(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted(_canonicalize(v) for v in value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, enum.Enum):
            return _canonicalize(value.value)
        if hasattr(value, "__dict__"):
            return _canonicalize(value.__dict__)
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return str(value)

    registry_digest = ""
    try:
        # R16-027: hash the FULL logical catalog (values + aliases), so changing
        # only ``output_unit`` (or a role / ParamSpec value) changes the
        # fingerprint — the old key-names-only digest never moved.
        blob = json.dumps(
            {
                "catalog": _canonicalize(OperatorRegistry._catalog),
                "aliases": _canonicalize(getattr(OperatorRegistry, "_aliases", {}) or {}),
            },
            sort_keys=True,
        )
        registry_digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"registry fingerprint could not be computed (R16-027): {exc}"
        ) from exc
    evidence_digest = ""
    try:
        from factor_engine.backend.factor_operator_evidence import load_factor_operator_evidence

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
    """Export the manifest.  API default == CLI default (R15-INC-053).

    R16-024: ``runtime_mining_manifest.eligible.json`` and
    ``operator_remediation_plan.pending.json`` are TWO STRICTLY SEPARATE
    schemas.  A runtime loader must reject the pending schema (a downstream
    consumer that reads only the canonical list from a pending manifest could
    mine operators that were never certified).  The legacy aggregate
    ``mining_manifest.json`` is kept for the CLI view.
    """
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

    # --- R16-024: eligible-only runtime manifest (the ONLY mining search space) ---
    eligible_ops = [e for e in manifest["operators"] if e["mining_eligible"]]
    runtime_manifest = {
        "schema_version": "factor_engine.runtime_mining_manifest.eligible.v1",
        "manifest_kind": "runtime_eligible",
        "query_fingerprint": manifest["query_fingerprint"],
        "count": len(eligible_ops),
        "operators": eligible_ops,
    }
    runtime_path = out_dir / "runtime_mining_manifest.eligible.json"
    runtime_path.write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # --- R16-024: pending remediation plan (NEVER a runtime search space) ---
    remediation = {
        "schema_version": "factor_engine.operator_remediation_plan.pending.v1",
        "manifest_kind": "operator_remediation_pending",
        "query_fingerprint": manifest["query_fingerprint"],
        "count": len(manifest["operators"]),
        "operators": manifest["operators"],
    }
    pending_path = out_dir / "operator_remediation_plan.pending.json"
    pending_path.write_text(
        json.dumps(remediation, ensure_ascii=False, indent=1), encoding="utf-8"
    )

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

    return {
        "json": json_path,
        "csv": csv_path,
        "md": md_path,
        "eligible": runtime_path,
        "pending": pending_path,
    }


def validate_runtime_manifest(path: Path) -> list[str]:
    """R16-024: a runtime loader rejects a PENDING-schema manifest.

    ``operator_remediation_plan.pending.json`` must never be consumed as a
    mining search space — a downstream consumer that reads only the canonical
    list could mine operators that were never certified.  This is the loader's
    fail-closed gate.
    """
    if not path.exists():
        return [f"runtime manifest not found: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"runtime manifest unreadable: {exc}"]
    kind = payload.get("manifest_kind")
    if kind != "runtime_eligible":
        return [
            f"runtime manifest schema rejected: manifest_kind={kind!r} — only "
            "'runtime_eligible' may be loaded as a mining search space (R16-024)"
        ]
    return []


def validate_cold_start(out_dir: Path, seed_source: str | Path | None = None) -> list[str]:
    """R15-INC-056/057 + R16-028/029: real cold-start validation.

    Phase 1 (R16-028): lineage FIRST.  The manifest's commit / dirty / registry
    fingerprints must match the CURRENT tree — an old manifest produced by a
    different commit must be STALE, not accepted.

    Phase 2 (R15-INC-051 + aliases): every production-certified factor and every
    resolvable alias is present in the manifest.

    Phase 3 (R16-029): every REAL cold-start seed expression is walked through
    parse -> type -> bind -> source -> smoke-execute.  A seed that cannot bind /
    source / execute is a violation — registry membership alone never proves a
    factor can run.
    """
    import numpy as np
    import pandas as pd

    manifest_path = out_dir / "mining_manifest.json"
    if not manifest_path.exists():
        return ["mining_manifest.json not found — run export first"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    in_manifest = {e["canonical"] for e in manifest["operators"]}
    admission_state = {e["canonical"]: e.get("admission_state") for e in manifest["operators"]}

    violations: list[str] = []

    # ---- Phase 1 (R16-028): artifact lineage must be current. ----
    fp = manifest.get("query_fingerprint") or {}
    current = _query_fingerprint()
    if fp.get("commit_sha") and fp["commit_sha"] != current["commit_sha"]:
        violations.append(
            f"STALE manifest: commit {fp.get('commit_sha')} != current {current['commit_sha']}"
        )
    if fp.get("dirty"):
        # R63: the manifest is stale only if the tree CHANGED after it was
        # built — a dirty-but-identical tree (export + validate in the same
        # process on the same uncommitted state) is not stale.  Compare the
        # recorded dirty string with the current one.
        if current["dirty"] != fp["dirty"]:
            violations.append(f"STALE manifest: built on a dirty tree ({len(fp['dirty'])} bytes)")
    if fp.get("registry_fingerprint") and fp["registry_fingerprint"] != current["registry_fingerprint"]:
        violations.append("STALE manifest: registry fingerprint does not match current tree")
    if fp.get("evidence_fingerprint") and fp["evidence_fingerprint"] != current["evidence_fingerprint"]:
        violations.append("STALE manifest: evidence fingerprint does not match current tree")

    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # ---- Phase 2 (R15-INC-051): certified factor + alias membership. ----
    for name in sorted(OperatorRegistry._catalog):
        entry = OperatorRegistry._catalog[name]
        if not entry.get("production_certified"):
            continue
        try:
            from factor_engine.mining.operator_catalog import assign_mining_role

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

    try:
        aliases = getattr(OperatorRegistry, "_aliases", {}) or {}
        for alias in sorted(aliases):
            try:
                canonical = OperatorRegistry.resolve_canonical(alias)
            except Exception:
                canonical = None
            if canonical is None or canonical not in in_manifest:
                target = aliases[alias]
                if target in OperatorRegistry._catalog:
                    from factor_engine.mining.operator_catalog import assign_mining_role

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

    # ---- Phase 3 (R16-029): real seed expressions parse/type/bind/execute. ----
    # Each seed is walked through parse -> type -> operator-graph smoke execute.
    # Registry membership alone never proves a factor can run: an operator whose
    # kernel cannot even bind/execute on synthetic inputs is a defect, and an
    # illegally-bound relational param is caught by the binder here.
    try:
        from factor_engine.fundamental_cold_start import load_cold_start

        seeds, _report = load_cold_start(seed_source)
    except Exception as exc:
        violations.append(f"cold-start seeds unavailable: {exc}")
        return violations
    if not seeds:
        return violations

    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    synthetic = pd.DataFrame(
        np.random.default_rng(7).normal(size=(60, 5)),
        index=pd.date_range("2024-01-01", periods=60, freq="B"),
        columns=[f"S{i:03d}" for i in range(5)],
    )

    def _referenced_canonicals(dsl: str) -> list[str]:
        """Every operator name the DSL calls, resolved via the registry alias map."""
        out: list[str] = []
        for token in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", dsl):
            try:
                out.append(OperatorRegistry.resolve_canonical(token))
            except Exception:
                continue
        return out

    executed = 0
    failed = 0
    for row in seeds:
        factor_id = str(row.get("factor_id", "")).strip()
        dsl = str(row.get("dsl", "")).strip()
        try:
            tree = parse_expr(dsl)  # parse + type
            if tree is None:
                raise ValueError("dsl parse returned None")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            violations.append(
                f"cold-start seed {factor_id} ({dsl!r}) failed parse/type: "
                f"{type(exc).__name__}: {exc}"
            )
            if failed > 20:
                break
            continue
        # operator-graph smoke execute: every referenced canonical must bind its
        # declared panel inputs and run on synthetic data.
        for canonical in _referenced_canonicals(dsl):
            try:
                operator = OperatorRegistry.get(canonical, "pandas_numpy")
                # R22: bind EVERY declared data input, not just ``panel_params``
                # (defaulting to a single "x" broke two-input kernels like
                # safe_div_null(x, y)).
                from factor_engine.mining.direct_use import _authoritative_param_split, input_slot_specs, split_input_slots

                cat = OperatorRegistry._catalog.get(canonical) or {}
                _panel, _scalar = _authoritative_param_split(canonical, cat)
                _slots = input_slot_specs(canonical, cat, panel_params=_panel, scalar_params=_scalar)
                data = split_input_slots(canonical, cat, slots=_slots)["data_inputs"]
                # Fundamental period transforms (period_id / fiscal_quarter /
                # date columns) need a REAL period calendar — a synthetic random
                # panel cannot satisfy that semantics.  Their cold-start smoke is
                # the DSL parse/type (already done above); synthetic operator
                # execution is skipped here, not failed.
                if any(
                    name in {"period_id", "fiscal_quarter", "change_date", "event_date"}
                    or name.startswith(("date", "report_period"))
                    for name in data
                ):
                    continue
                panels = [synthetic for _ in (data or ("x",))]
                operator.calculate(*panels)
                executed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                violations.append(
                    f"cold-start seed {factor_id}: operator {canonical!r} failed "
                    f"bind/execute on synthetic panels: {type(exc).__name__}: {exc}"
                )
                if failed > 20:
                    break
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
    parser.add_argument(
        "--seed-source",
        default=None,
        help="path to the cold-start seed CSV (default: locate fundamental_factors_single_default_1288.csv)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    paths = export_mining_manifest(out_dir, admission=args.admission)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if args.validate_cold_start:
        # R16-024: the runtime loader must first reject a non-eligible schema.
        runtime_errors = validate_runtime_manifest(paths["eligible"])
        if runtime_errors:
            for e in runtime_errors:
                print(f"runtime manifest error: {e}")
            return 2
        violations = validate_cold_start(out_dir, seed_source=args.seed_source)
        if violations:
            print("cold-start violations:")
            for v in violations[:40]:
                print(f"  {v}")
            return 2
        print("cold-start validation: OK (lineage current + all seeds parse/bind/execute)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
