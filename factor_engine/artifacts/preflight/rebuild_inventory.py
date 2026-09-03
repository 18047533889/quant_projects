# -*- coding: utf-8 -*-
"""P0#1 fact-rebuild: authoritative inventory for CURRENT HEAD (100k GO §1/§35/§110.1).

Read-only audit generator.  Loads the live registry exactly once with the SAME
authoritative load the known facts used — `load_all(include_research=False)`
= 1673 canonical (production surface; default include_research=True would add 67
research-module canonicals = 1740, a DIFFERENT fact plane than the GO baseline).

Outputs under ../../artifacts/...  (repo-root artifacts dir):
  operator_inventory.json          — every canonical with the FULL catalog row
  alias_inventory.json             — canonical -> aliases / compat_aliases
  surface_inventory.json           — surface/lifecycle/execution_kind/scope/status dists
  direct_use_matrix.json           — DirectUseOperator rows (R18 authority module)
  backend_matrix.json              — per-canonical backend_* facts + backends list
  field_registry.json              — input_fields / input_units aggregation
  source_contract_matrix.json      — source_contract_* / certification fields
  operator_matrix.csv              — §4 overview (flat)
  inventory_manifest.json          — fingerprint + per-file counts
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import os

_REPO_ROOT = Path("/home/sunhaiwei/quant_projects/factor_engine")
_ART_ROOT = _REPO_ROOT / "artifacts"
_PREFLIGHT = _ART_ROOT / "preflight"
_AUDIT_DIR = _ART_ROOT / "operator_audit" / "current_head"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


def _artifact_fingerprint(registry_digest: str) -> dict:
    return {
        "schema_version": "factor_engine.current_head.inventory.v1",
        "generated_at_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip(),
        "load_mode": "load_all(include_research=False)",
        "commit_sha": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "dirty_files": _git("status", "--porcelain").splitlines() if _git("status", "--porcelain") else [],
        "registry_fingerprint": registry_digest,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    import time

    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    # ---- authoritative load: production surface (1673), matching known facts.
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.operator_surface import classify_canonical, surface_summary
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.mining.direct_use import build_direct_use_operator

    load_all(include_research=False)

    catalog: dict = OperatorRegistry._catalog
    aliases: dict = OperatorRegistry._aliases
    operators: dict = OperatorRegistry._operators
    canonicals = sorted(catalog)
    n = len(canonicals)

    # registry fingerprint over the frozen read state (same digest style as the
    # admission-matrix _artifact_fingerprint).
    registry_digest = hashlib.sha256(
        json.dumps(
            {
                c: sorted(k for k in catalog[c] if not callable(k))
                for c in canonicals
            },
            sort_keys=True, default=str,
        ).encode("utf-8"),
    ).hexdigest()[:16]
    fp = _artifact_fingerprint(registry_digest)

    # ------------------------------------------------------------------ 1
    operator_inventory = {
        "header": fp,
        "canonical_count": n,
        "operators": {c: dict(catalog[c]) for c in canonicals},
    }
    _write_json(_PREFLIGHT / "operator_inventory.json", operator_inventory)

    # ------------------------------------------------------------------ 2
    alias_rows = {}
    reverse = Counter(aliases.values())
    for c in canonicals:
        cat = catalog[c]
        compat = dict(cat.get("compat_aliases") or {})
        alias_rows[c] = {
            "canonical": c,
            "aliases": sorted(str(a) for a in (cat.get("aliases") or ())),
            "compat_aliases": sorted(compat.keys()),
            "compat_alias_count": len(compat),
            "dsl_aliases_pointing_here": sorted(
                a for a, t in aliases.items() if t == c
            ),
            "inbound_alias_total": int(reverse[c]),
        }
    _write_json(_PREFLIGHT / "alias_inventory.json", {
        "header": fp,
        "canonical_count": n,
        "total_aliases": len(aliases),
        "aliases": alias_rows,
    })

    # ------------------------------------------------------------------ 3
    lifecycle_c = Counter(cat.get("lifecycle_status") for cat in catalog.values())
    surface_c = Counter(cat.get("surface") for cat in catalog.values())
    classify_c = Counter(classify_canonical(c) for c in canonicals)
    scope_c = Counter((cat.get("scope") or "undeclared") for cat in catalog.values())
    status_c = Counter((cat.get("status") or "undeclared") for cat in catalog.values())
    exec_kind = Counter()
    for cat in catalog.values():
        kinds = set()
        meta = cat.get("backend_meta") or {}
        for b, bm in meta.items():
            ek = bm.get("execution_kind")
            if ek:
                kinds.add(f"{b}:{ek}")
        if not kinds:
            kinds.add("undeclared")
        exec_kind["|".join(sorted(kinds))] += 1

    _write_json(_PREFLIGHT / "surface_inventory.json", {
        "header": fp,
        "canonical_count": n,
        "lifecycle": dict(lifecycle_c),
        "surface_field": dict(surface_c),
        "authoring_tier_classify": dict(classify_c),
        "scope": dict(scope_c),
        "status": dict(status_c),
        "backend_meta_execution_kind_bucket": dict(exec_kind),
    })

    # ------------------------------------------------------------------ 4 DirectUse matrix (R18 authority: build_direct_use_operator)
    rows = [build_direct_use_operator(c, catalog[c]) for c in canonicals]
    du_rows = [r.to_dict() for r in rows]
    _write_json(_PREFLIGHT / "direct_use_matrix.json", {
        "header": fp,
        "canonical_count": n,
        "direct_use_status_counts": dict(Counter(r["direct_use_status"] for r in du_rows)),
        "rows": du_rows,
    })

    # ------------------------------------------------------------------ 5 backend matrix
    backend_matrix = {}
    backend_status_c = Counter()
    backend_passed_c = Counter()
    signature_cons_c = Counter()
    for c in canonicals:
        cat = catalog[c]
        backends = list(cat.get("backends") or [])
        meta = cat.get("backend_meta") or {}
        per_backend = {
            b: dict(meta.get(b) or {})
            for b in backends
        }
        impl_names = sorted((operators.get(c) or {}).keys())
        backend_matrix[c] = {
            "canonical": c,
            "backends": backends,
            "backend_count": len(backends),
            "runtime_impl_backends": impl_names,
            "backend_status": cat.get("backend_status"),
            "backend_passed": cat.get("backend_passed"),
            "backend_signature_consistent": cat.get("backend_signature_consistent"),
            "backend_signatures": cat.get("backend_signatures"),
            "backend_meta": per_backend,
            "runtime_execution_verified": cat.get("runtime_execution_verified"),
            "determinism_verified": cat.get("determinism_verified"),
            "production_backend_policy": cat.get("production_backend_policy"),
        }
        backend_status_c[cat.get("backend_status")] += 1
        backend_passed_c[cat.get("backend_passed")] += 1
        signature_cons_c[cat.get("backend_signature_consistent")] += 1
    _write_json(_PREFLIGHT / "backend_matrix.json", {
        "header": fp,
        "canonical_count": n,
        "backend_status_dist": dict(backend_status_c),
        "backend_passed_dist": dict(backend_passed_c),
        "backend_signature_consistent_dist": dict(signature_cons_c),
        "rows": backend_matrix,
    })

    # ------------------------------------------------------------------ 6 field registry
    field_usage = Counter()
    field_unit_c = Counter()
    input_arity_c = Counter()
    for c in canonicals:
        cat = catalog[c]
        fields = cat.get("input_fields") or ()
        arity = cat.get("input_arity")
        input_arity_c[arity if arity is not None else len(fields)] += 1
        if not fields:
            field_usage["(no input_fields declared)"] += 1
            continue
        for f in fields:
            field_usage[f] += 1
        units = cat.get("input_units") or {}
        for f in fields:
            u = units.get(f)
            if u:
                field_unit_c[f"{f}->{u}"] += 1
    _write_json(_PREFLIGHT / "field_registry.json", {
        "header": fp,
        "canonical_count": n,
        "input_arity_dist": dict(input_arity_c),
        "input_field_usage": dict(field_usage.most_common()),
        "input_field_unit_map": dict(field_unit_c),
    })

    # ------------------------------------------------------------------ 7 source contract matrix
    src_keys = [
        "source_contract_certified", "source_contract_verified",
        "semantic_pit_review_passed", "pit_safe", "prefix_kernel_causality_verified",
        "temporal_certified", "temporal_prefix_verified",
        "semantic_certified", "semantic_golden_verified",
        "implementation_certified", "edge_case_passed",
        "shape_verified", "determinism_verified", "runtime_execution_verified",
        "production_certified", "required_cutoff", "availability", "calendar",
        "universe", "lookback_contract", "lookback_param_names",
    ]
    source_contract_matrix = {}
    dists = {k: Counter() for k in src_keys}
    for c in canonicals:
        cat = catalog[c]
        row = {"canonical": c}
        for k in src_keys:
            v = cat.get(k)
            row[k] = v
            dists[k][repr(v) if not isinstance(v, bool) else v] += 1
        source_contract_matrix[c] = row
    _write_json(_PREFLIGHT / "source_contract_matrix.json", {
        "header": fp,
        "canonical_count": n,
        "distributions": {k: dict(v) for k, v in dists.items()},
        "rows": source_contract_matrix,
    })

    # ------------------------------------------------------------------ 8 §4 operator_matrix.csv flat overview
    csv_cols = [
        "canonical", "aliases", "surface", "lifecycle_status", "scope",
        "execution_kind", "terminal_or_intermediate", "direct_use_status",
        "mining_role", "agent_visible", "production_certified",
        "production_admitted", "directly_usable", "mining_visible",
        "composition_usable", "terminal_usable", "context_admitted",
        "backend_passed", "backends", "backend_status",
        "backend_signature_consistent", "pandas_impl", "polars_panel_impl",
        "polars_long_impl", "duckdb_impl", "math_oracle_pass", "oracle_pass",
        "prefix_causality_pass", "pit_safe", "lookback_contract",
        "implementation_certified", "semantic_certified", "temporal_certified",
        "source_contract_certified", "edge_case_passed", "determinism_verified",
        "runtime_execution_verified", "stateful", "execution_model",
        "full_history_replay_required", "cost_tier", "factor_role",
    ]

    def _join(v) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple, set)):
            return "|".join(str(x) for x in v)
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False, default=str)
        return str(v)

    csv_rows = []
    by_status = Counter()
    terminal_n = 0
    du_by_canon = {r["canonical"]: r for r in du_rows}
    for c in canonicals:
        cat = catalog[c]
        du = du_by_canon.get(c) or {}
        by_status[du.get("direct_use_status", "")] += 1
        du_terminal = bool(du.get("terminal_allowed"))
        terminal_n += 1 if du_terminal else 0
        meta = cat.get("backend_meta") or {}
        backend_kinds = []
        for b in ("pandas_numpy", "polars", "duckdb_sql", "sql", "clickhouse_sql"):
            bm = meta.get(b) or {}
            ek = bm.get("execution_kind") or ""
            backend_kinds.append(f"{b}={ek}")
        csv_rows.append({
            "canonical": c,
            "aliases": _join(cat.get("aliases")),
            "surface": _join(cat.get("surface")),
            "lifecycle_status": _join(cat.get("lifecycle_status")),
            "scope": _join(cat.get("scope")),
            "execution_kind": "|".join(backend_kinds),
            "terminal_or_intermediate": "terminal" if du_terminal else (
                "intermediate" if du.get("retention_usage", {}).get("intermediate") else "supporting"
            ),
            "direct_use_status": du.get("direct_use_status", ""),
            "mining_role": du.get("mining_role", ""),
            "agent_visible": _join(bool(du.get("mining_visible")) and bool(du.get("production_admitted"))),
            "production_certified": _join(cat.get("production_certified") is True),
            "production_admitted": _join(bool(du.get("production_admitted"))),
            "directly_usable": _join(bool(du.get("directly_usable"))),
            "mining_visible": _join(bool(du.get("mining_visible"))),
            "composition_usable": _join(bool(du.get("composition_usable"))),
            "terminal_usable": _join(bool(du.get("terminal_usable"))),
            "context_admitted": _join(bool(du.get("context_admitted"))),
            "backend_passed": _join(cat.get("backend_passed")),
            "backends": _join(cat.get("backends")),
            "backend_status": _join(cat.get("backend_status")),
            "backend_signature_consistent": _join(cat.get("backend_signature_consistent")),
            "pandas_impl": _join("pandas_numpy" in (operators.get(c) or {})),
            "polars_panel_impl": _join("polars" in (operators.get(c) or {})),
            "polars_long_impl": _join("polars_long" in (operators.get(c) or {})),
            "duckdb_impl": _join(bool({"duckdb_sql", "sql"} & set(cat.get("backends") or ()))),
            "math_oracle_pass": du.get("r64_production_gates", {}).get("MATH_ORACLE_PASS", "NOT_RUN"),
            "oracle_pass": du.get("r64_production_gates", {}).get("MATH_ORACLE_PASS", "NOT_RUN"),
            "prefix_causality_pass": du.get("r64_causality", "unknown"),
            "pit_safe": _join(cat.get("pit_safe")),
            "lookback_contract": _join(cat.get("lookback_contract")),
            "implementation_certified": _join(cat.get("implementation_certified")),
            "semantic_certified": _join(cat.get("semantic_certified")),
            "temporal_certified": _join(cat.get("temporal_certified")),
            "source_contract_certified": _join(cat.get("source_contract_certified")),
            "edge_case_passed": _join(cat.get("edge_case_passed")),
            "determinism_verified": _join(cat.get("determinism_verified")),
            "runtime_execution_verified": _join(cat.get("runtime_execution_verified")),
            "stateful": _join(cat.get("stateful")),
            "execution_model": du.get("execution_model", ""),
            "full_history_replay_required": _join(du.get("full_history_replay_allowed") is False),
            "cost_tier": _join(du.get("runtime_cost")),
            "factor_role": du.get("mining_role", ""),
        })
    _write_csv(_AUDIT_DIR / "operator_matrix.csv", csv_cols, csv_rows)

    # ------------------------------------------------------------------ manifest
    manifest = {
        "header": fp,
        "canonical_count": n,
        "files": {
            "operator_inventory.json": _PREFLIGHT / "operator_inventory.json",
            "alias_inventory.json": _PREFLIGHT / "alias_inventory.json",
            "surface_inventory.json": _PREFLIGHT / "surface_inventory.json",
            "direct_use_matrix.json": _PREFLIGHT / "direct_use_matrix.json",
            "backend_matrix.json": _PREFLIGHT / "backend_matrix.json",
            "field_registry.json": _PREFLIGHT / "field_registry.json",
            "source_contract_matrix.json": _PREFLIGHT / "source_contract_matrix.json",
            "operator_matrix.csv": _AUDIT_DIR / "operator_matrix.csv",
        },
        "summary": {
            "lifecycle": dict(lifecycle_c),
            "surface_field": dict(surface_c),
            "authoring_tier": dict(classify_c),
            "production_certified": int(sum(1 for cat in catalog.values() if cat.get("production_certified") is True)),
            "direct_use_status_counts": dict(by_status),
            "terminal_allowed": terminal_n,
            "production_admitted": int(sum(1 for r in du_rows if r.get("production_admitted"))),
            "directly_usable": int(sum(1 for r in du_rows if r.get("directly_usable"))),
            "mining_visible": int(sum(1 for r in du_rows if r.get("mining_visible"))),
            "agent_visible_candidate": int(sum(
                1 for r in du_rows
                if r.get("mining_visible") and r.get("production_admitted")
            )),
        },
        "timing_seconds": round(time.time() - t0, 1),
    }
    # normalize file paths to str + record counts
    for name, p in list(manifest["files"].items()):
        txt = Path(p).read_text(encoding="utf-8")
        manifest["files"][name] = {
            "path": str(p),
            "bytes": len(txt.encode("utf-8")),
            "lines": txt.count("\n") + 1,
        }
    _write_json(_PREFLIGHT / "inventory_manifest.json", manifest)

    print(f"canonical: {n}")
    print(f"lifecycle: {dict(lifecycle_c)}")
    print(f"surface: {dict(surface_c)}")
    print(f"authoring_tier: {dict(classify_c)}")
    print(f"direct_use_status: {dict(by_status)}")
    print(f"production_certified: {manifest['summary']['production_certified']}")
    print(f"terminal_allowed: {terminal_n}")
    print(f"production_admitted: {manifest['summary']['production_admitted']}")
    print(f"directly_usable: {manifest['summary']['directly_usable']}")
    print(f"mining_visible: {manifest['summary']['mining_visible']}")
    print(f"agent_visible_candidate: {manifest['summary']['agent_visible_candidate']}")
    print(f"manifest: {_PREFLIGHT / 'inventory_manifest.json'}")
    print(f"timing_seconds: {manifest['timing_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
