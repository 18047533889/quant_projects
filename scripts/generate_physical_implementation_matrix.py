#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R21-P0-MATRIX-TRUTH: machine-generated backend implementation matrix.

Single source of truth for backend coverage counts.  Manual counts in
``backend/README.md`` / ``docs/sql_pushdown_coverage.md`` /
``evidence/r2/BACKEND_GAP_MATRIX.yaml`` are NOT authoritative — this generator
walks the LIVE registry (``cleaned_operators.load_all`` +
``backend.sql_pushdown.sql_registry.register_sql_backends``) and the live
evidence authorities:

* rows: ``backend.operator_capability.enumerate_physical_inventory`` — one row
  per (canonical, registry slot, physical backend), each with its explicit
  ``PhysicalImplementationSpec`` binding / fail-closed admission truth.
* oracle evidence: ``evidence/factor_operator_verified.json`` (runtime
  execution / shape / determinism / temporal prefix).  Missing record ⇒
  ``NOT_RUN`` — never guessed.
* polars parity/edge: ``cleaned_operators.operator_policy.POLARS_PARITY_VERIFIED``
  / ``POLARS_PRODUCTION_SAFE`` (derived from ``backend.primitive_evidence``).
* duckdb parity/production-safe: ``backend.sql_tiers.SQL_PARITY_VERIFIED_CANONICALS``
  / ``SQL_PRODUCTION_SAFE_CANONICALS`` — these currently fail closed to
  ``{column, literal}`` because ``evidence/primitive_verified.json`` provenance
  is stale; the generator reports that state honestly.
* direct-use fields: ``mining.direct_use.build_direct_use_operator`` (called,
  never duplicated).

Outputs (all generated artifacts):
* ``docs/PHYSICAL_IMPLEMENTATION_MATRIX.md`` — per-(canonical, backend) rows.
* ``docs/BACKEND_COVERAGE.md`` — counts summary derived from the matrix.

Run pinned: ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
POLARS_MAX_THREADS=1 python3 scripts/generate_physical_implementation_matrix.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]

# Pin threads defensively when executed rather than imported by the smoke test
# (the smoke test pins before importing this module).
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_var, "1")

MATRIX_DOC = FE_ROOT / "docs" / "PHYSICAL_IMPLEMENTATION_MATRIX.md"
COVERAGE_DOC = FE_ROOT / "docs" / "BACKEND_COVERAGE.md"
ORACLE_JSON = FE_ROOT / "evidence" / "factor_operator_verified.json"


def _bootstrap() -> None:
    """Initialize sys.path and load the live registry + SQL markers."""
    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _current_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(FE_ROOT), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _oracle_records() -> dict[str, dict[str, Any]]:
    if not ORACLE_JSON.is_file():
        return {}
    data = json.loads(ORACLE_JSON.read_text(encoding="utf-8"))
    operators = data.get("operators")
    return dict(operators) if isinstance(operators, dict) else {}


def _oracle_passed(canonical: str, records: dict[str, dict[str, Any]]) -> str:
    """Return ``yes`` / ``no`` / ``NOT_RUN`` from the evidence artifact only."""
    record = records.get(canonical)
    if record is None:
        return "NOT_RUN"
    return "yes" if record.get("runtime_execution_verified") is True else "no"


def build_matrix() -> list[dict[str, Any]]:
    """Walk the live registry and return one row per physical backend slot."""
    _bootstrap()

    from factor_engine.backend.operator_capability import enumerate_physical_inventory
    from factor_engine.backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_NAN_EDGE_VERIFIED,
    )
    from factor_engine.backend.sql_tiers import (
        SQL_PARITY_VERIFIED_CANONICALS,
        SQL_PRODUCTION_SAFE_CANONICALS,
    )
    from factor_engine.backend import evidence_provenance
    from factor_engine.cleaned_operators.operator_policy import (
        POLARS_PARITY_VERIFIED,
        POLARS_PRODUCTION_SAFE,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    head_sha = _current_head()
    oracle = _oracle_records()
    duckdb_edge = set(DUCKDB_EDGE_VERIFIED) | set(DUCKDB_NAN_EDGE_VERIFIED)

    rows: list[dict[str, Any]] = []
    provenance_cache: dict[str, dict[str, str]] = {}
    domain_hash_cache: dict[str, str] = {}
    for record in enumerate_physical_inventory():
        canonical = record.canonical
        backend = record.backend
        slot = record.registry_slot
        spec = record.spec
        if canonical not in provenance_cache:
            try:
                provenance_cache[canonical] = dict(
                    evidence_provenance.implementation_hashes_for(canonical)
                )
            except Exception:
                provenance_cache[canonical] = {}
        impl_hashes = provenance_cache[canonical]

        # implementation_id: one per (canonical, backend, execution_kind,
        # emitter identity) — bound by the spec itself (R21 §28); null when
        # the spec is missing/invalid, and then spec_complete is false.
        implementation_id = str(record.implementation_id) if record.implementation_id else ""

        execution_kind = (
            spec.execution_kind.value if spec is not None else "undeclared"
        )
        implemented = True  # enumerate only yields slots with a registration
        try:
            selectable = (
                OperatorRegistry.get(canonical, slot, mode="production") is not None
            )
        except Exception:
            selectable = False

        if backend == "polars":
            parity_passed = canonical in POLARS_PARITY_VERIFIED
            edge_key = "polars_edge"
            edge_passed = canonical in POLARS_PRODUCTION_SAFE  # parity∩edge∩no-fallback
            production_evidence_passed = canonical in POLARS_PRODUCTION_SAFE
            source_hash = (
                spec.implementation_source_hash
                if spec is not None and spec.implementation_source_hash
                else impl_hashes.get("implementation_hash_polars", "")
            )
        elif backend == "duckdb_sql":
            parity_passed = canonical in SQL_PARITY_VERIFIED_CANONICALS
            edge_passed = canonical in duckdb_edge
            production_evidence_passed = canonical in SQL_PRODUCTION_SAFE_CANONICALS
            source_hash = (
                spec.implementation_source_hash
                if spec is not None and spec.implementation_source_hash
                else impl_hashes.get("implementation_hash_duckdb", "")
            )
        elif backend == "clickhouse_sql":
            parity_passed = False  # no ClickHouse certification shares DuckDB's
            edge_passed = False
            production_evidence_passed = False
            source_hash = (
                spec.implementation_source_hash
                if spec is not None and spec.implementation_source_hash
                else ""
            )
        else:  # pandas_numpy — reference backend
            parity_passed = True  # reference semantics by definition
            edge_key = "pandas_edge"
            edge_passed = False  # pandas edge suite: NOT_RUN (no artifact)
            production_evidence_passed = False  # certification NOT_RUN
            source_hash = (
                spec.implementation_source_hash
                if spec is not None and spec.implementation_source_hash
                else impl_hashes.get("implementation_hash_pandas", "")
            )

        semantic_hash = spec.semantic_contract_hash if spec is not None else ""
        if not semantic_hash:
            semantic_hash = impl_hashes.get("semantic_contract_hash", "")
        parameter_domain_hash = spec.parameter_domain_hash if spec is not None else ""
        if not parameter_domain_hash:
            if canonical not in domain_hash_cache:
                try:
                    domain_hash_cache[canonical] = (
                        evidence_provenance.parameter_domain_hash_for(canonical) or ""
                    )
                except Exception:
                    domain_hash_cache[canonical] = ""
            parameter_domain_hash = domain_hash_cache[canonical]

        rows.append(
            {
                "canonical": canonical,
                "registry_slot": slot,
                "physical_backend": backend,
                "implementation_id": implementation_id,
                "execution_kind": execution_kind,
                "implemented": implemented,
                "selectable": selectable,
                "spec_complete": bool(record.admission.spec_complete),
                "oracle_passed": _oracle_passed(canonical, oracle),
                "edge_passed": bool(edge_passed),
                "parity_passed": bool(parity_passed),
                "production_evidence_passed": bool(production_evidence_passed),
                "production_admitted": bool(record.admission.admitted),
                "source_hash": source_hash,
                "semantic_hash": semantic_hash,
                "parameter_domain_hash": parameter_domain_hash,
                "current_head_sha": head_sha,
            }
        )
    return rows


def build_direct_use_rows() -> list[dict[str, Any]]:
    """DIRECT_USE_MATRIX fields per canonical — delegated to mining.direct_use."""
    from factor_engine.mining.direct_use import build_direct_use_operator
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    out: list[dict[str, Any]] = []
    catalog = OperatorRegistry._catalog
    for canonical in sorted(catalog):
        row = build_direct_use_operator(canonical, dict(catalog[canonical]))
        out.append(
            {
                "canonical": canonical,
                "direct_use_status": str(row.direct_use_status.value),
                "mining_visible": bool(row.mining_visible),
                "composition_usable": bool(row.composition_usable),
                "terminal_usable": bool(row.terminal_usable),
                "production_admitted": bool(row.production_admitted),
                "directly_usable": bool(row.directly_usable),
            }
        )
    return out


def summarize(
    rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Derive every count from the generated matrix rows only."""
    backends: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = backends.setdefault(
            row["physical_backend"],
            {
                "implemented": 0,
                "selectable": 0,
                "spec_complete": 0,
                "implementation_id_bound": 0,
                "oracle_passed": 0,
                "oracle_no_evidence": 0,
                "edge_passed": 0,
                "parity_passed": 0,
                "production_evidence_passed": 0,
                "production_admitted": 0,
            },
        )
        bucket["implemented"] += int(bool(row["implemented"]))
        bucket["selectable"] += int(bool(row["selectable"]))
        bucket["spec_complete"] += int(bool(row["spec_complete"]))
        bucket["implementation_id_bound"] += int(bool(row["implementation_id"]))
        bucket["oracle_passed"] += int(row["oracle_passed"] == "yes")
        bucket["oracle_no_evidence"] += int(row["oracle_passed"] == "NOT_RUN")
        bucket["edge_passed"] += int(bool(row["edge_passed"]))
        bucket["parity_passed"] += int(bool(row["parity_passed"]))
        bucket["production_evidence_passed"] += int(
            bool(row["production_evidence_passed"])
        )
        bucket["production_admitted"] += int(bool(row["production_admitted"]))
    direct = {
        "total_canonicals": len(direct_rows),
        "mining_visible": sum(r["mining_visible"] for r in direct_rows),
        "composition_usable": sum(r["composition_usable"] for r in direct_rows),
        "terminal_usable": sum(r["terminal_usable"] for r in direct_rows),
        "production_admitted": sum(r["production_admitted"] for r in direct_rows),
        "directly_usable": sum(r["directly_usable"] for r in direct_rows),
    }
    return {
        "total_rows": len(rows),
        "total_canonicals": len({r["canonical"] for r in rows}),
        "backends": backends,
        "direct_use": direct,
    }


ROW_FIELDS = [
    "canonical",
    "registry_slot",
    "physical_backend",
    "implementation_id",
    "execution_kind",
    "implemented",
    "selectable",
    "spec_complete",
    "oracle_passed",
    "edge_passed",
    "parity_passed",
    "production_evidence_passed",
    "production_admitted",
    "source_hash",
    "semantic_hash",
    "parameter_domain_hash",
    "current_head_sha",
]


def _cell(value: Any) -> str:
    """Render one table cell, keeping the markdown table well-formed.

    R21-MT-P2a: a raw ``|`` inside a cell value (e.g. a parameter_domain_hash
    like ``{a|b}``) splits the row into extra columns; newlines break the row
    onto multiple lines.  Every data-driven cell in the emitted tables goes
    through this single emitter, which escapes ``|`` as ``\\|`` and flattens
    newlines/tabs to single spaces.  Backslashes are left untouched (cell
    values are hashes/identifiers; double-escaping would corrupt them).
    """
    if value is True:
        text = "yes"
    elif value is False:
        text = "no"
    elif value is None or value == "":
        text = "-"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").replace("\t", " ")


def write_matrix_doc(
    rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    head_sha: str,
) -> None:
    lines = [
        "# Physical implementation matrix (generated)",
        "",
        "> MACHINE-GENERATED — do not edit by hand. Regenerate with",
        "> `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 python3 scripts/generate_physical_implementation_matrix.py`.",
        f"> Generated (UTC): {datetime.now(timezone.utc).isoformat()} | HEAD: `{head_sha}` (working tree is truth; uncommitted modifications may be present).",
        f"> Canonicals: **{summary['total_canonicals']}** | physical backend rows: **{summary['total_rows']}**.",
        "> oracle_passed is read from `evidence/factor_operator_verified.json`; a missing record is `NOT_RUN` (never guessed).",
        "> DuckDB parity/production-safe come from `backend.sql_tiers` and currently FAIL CLOSED to `{column, literal}` because `evidence/primitive_verified.json` provenance is stale (regeneration pipeline running separately).",
        "",
        "## Per-(canonical, physical backend) rows",
        "",
        "| " + " | ".join(ROW_FIELDS) + " |",
        "|" + "|".join(["---"] * len(ROW_FIELDS)) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(row[field]) for field in ROW_FIELDS) + " |"
        )
    lines += [
        "",
        "## DIRECT_USE_MATRIX (per canonical; delegated to mining.direct_use)",
        "",
        "| canonical | direct_use_status | mining_visible | composition_usable | terminal_usable | production_admitted | directly_usable |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in direct_rows:
        lines.append(
            "| {canonical} | {status} | {mv} | {cu} | {tu} | {pa} | {du} |".format(
                canonical=_cell(row["canonical"]),
                status=_cell(row["direct_use_status"]),
                mv=_cell(row["mining_visible"]),
                cu=_cell(row["composition_usable"]),
                tu=_cell(row["terminal_usable"]),
                pa=_cell(row["production_admitted"]),
                du=_cell(row["directly_usable"]),
            )
        )
    lines.append("")
    MATRIX_DOC.write_text("\n".join(lines), encoding="utf-8")


def write_coverage_doc(summary: dict[str, Any], head_sha: str) -> None:
    lines = [
        "# Backend coverage (generated counts summary)",
        "",
        "> MACHINE-GENERATED — do not edit by hand. Counts derive ONLY from",
        "> `docs/PHYSICAL_IMPLEMENTATION_MATRIX.md` (same run). Manual counts elsewhere are not authoritative.",
        f"> Generated (UTC): {datetime.now(timezone.utc).isoformat()} | HEAD: `{head_sha}`.",
        "",
        f"- total canonicals: **{summary['total_canonicals']}**",
        f"- physical backend rows: **{summary['total_rows']}**",
        "",
        "| physical_backend | implemented | selectable(prod) | spec_complete | impl_id_bound | oracle_passed | oracle_NOT_RUN | edge_passed | parity_passed | prod_evidence | prod_admitted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, counts in sorted(summary["backends"].items()):
        lines.append(
            f"| {backend} | {counts['implemented']} | {counts['selectable']} | "
            f"{counts['spec_complete']} | {counts['implementation_id_bound']} | "
            f"{counts['oracle_passed']} | {counts['oracle_no_evidence']} | "
            f"{counts['edge_passed']} | {counts['parity_passed']} | "
            f"{counts['production_evidence_passed']} | {counts['production_admitted']} |"
        )
    direct = summary["direct_use"]
    lines += [
        "",
        "## Direct-use split (per canonical)",
        "",
        f"- total canonicals: **{direct['total_canonicals']}**",
        f"- mining_visible: **{direct['mining_visible']}**",
        f"- composition_usable: **{direct['composition_usable']}**",
        f"- terminal_usable: **{direct['terminal_usable']}**",
        f"- production_admitted: **{direct['production_admitted']}**",
        f"- directly_usable: **{direct['directly_usable']}**",
        "",
        "## Honest caveats",
        "",
        "- DuckDB parity/production-safe sets fail closed to `{column, literal}` — `evidence/primitive_verified.json` provenance is stale (regeneration running separately). `parity_passed=no` below that set is the fail-closed state, not a measured failure.",
        "- pandas edge / production certification: NOT_RUN (no artifact consulted by this generator).",
        "- production certification gates: NOT_RUN for this slice.",
        "",
    ]
    COVERAGE_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    head_sha = _current_head()
    rows = build_matrix()
    direct_rows = build_direct_use_rows()
    summary = summarize(rows, direct_rows)
    write_matrix_doc(rows, direct_rows, summary, head_sha)
    write_coverage_doc(summary, head_sha)
    print(f"HEAD: {head_sha}")
    print(f"total canonicals: {summary['total_canonicals']}")
    print(f"total physical backend rows: {summary['total_rows']}")
    for backend, counts in sorted(summary["backends"].items()):
        print(
            f"{backend}: implemented={counts['implemented']} "
            f"selectable={counts['selectable']} spec_complete={counts['spec_complete']} "
            f"impl_id_bound={counts['implementation_id_bound']} "
            f"oracle_passed={counts['oracle_passed']} oracle_NOT_RUN={counts['oracle_no_evidence']} "
            f"edge={counts['edge_passed']} parity={counts['parity_passed']} "
            f"prod_evidence={counts['production_evidence_passed']} "
            f"prod_admitted={counts['production_admitted']}"
        )
    direct = summary["direct_use"]
    print(
        "direct_use: mining_visible={mining_visible} composition_usable={composition_usable} "
        "terminal_usable={terminal_usable} production_admitted={production_admitted} "
        "directly_usable={directly_usable}".format(**direct)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
