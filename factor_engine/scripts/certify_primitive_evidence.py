#!/usr/bin/env python3
"""Certify primitive backend evidence from tests and semantic content hashes.

Git commit SHA is retained as audit metadata but is not a correctness identity:
squash/rebase merges legitimately change commit ancestry while leaving every
certified semantic input unchanged. Hard validity is bound to operator source,
emitter, semantic contract, parameter-domain, golden/test and case-registry
hashes validated by ``evidence_provenance``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

CERTIFICATION_STAGES: list[tuple[str, list[str]]] = [
    ("polars_reference_parity", [
        "tests/backend_parity/test_production_core_triple_parity.py",
        "tests/backend_parity/test_production_safe_bulk_parity.py",
    ]),
    ("polars_edge_verified", ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"]),
    ("duckdb_reference_parity", [
        "tests/backend_parity/test_production_core_triple_parity.py",
        "tests/backend_parity/test_production_safe_bulk_parity.py",
    ]),
    ("duckdb_real_sql_verified", [
        "tests/backend_parity/test_production_core_triple_parity.py",
        "tests/backend_parity/test_production_safe_bulk_parity.py",
    ]),
    ("duckdb_edge_verified", ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"]),
    ("no_fallback_verified", ["tests/backend_parity/test_polars_long_no_pandas_path.py"]),
    ("duckdb_nan_edge_verified", [
        "tests/backend_parity/test_p0_semantic_edges.py",
        "tests/backend_parity/test_rank_null_singleton_boundaries.py",
    ]),
    ("duckdb_inf_edge_verified", [
        "tests/backend_parity/test_p0_semantic_edges.py",
        "tests/backend_parity/test_rank_null_singleton_boundaries.py",
    ]),
    ("composite_reference_parity", ["tests/backend_parity/test_composite_reference_semantics.py"]),
    ("composite_triple_parity", [
        "tests/backend_parity/test_composite_lowered_triple_parity.py",
        "tests/backend_parity/test_composite_edge_triple_parity.py",
    ]),
    ("polars_native_static_audit", ["tests/backend/test_polars_long_static_audit.py"]),
    ("parameter_signature_gate", [
        "tests/backend/test_operator_evidence_schema.py",
        "tests/backend/test_operator_call_policy.py",
        "tests/backend/test_production_checklist.py",
    ]),
    ("plan_variant_parity", ["tests/backend_parity/test_plan_variant_parity.py"]),
    ("chunk_invariance", ["tests/backend_parity/test_chunk_invariance.py"]),
    ("manifest_and_evidence_drift", [
        "tests/backend/test_primitive_evidence_contract.py",
        "tests/backend/test_operator_manifest_freshness.py",
        "tests/operator_contracts/test_manifest_sync.py",
    ]),
    ("strict_period_triple_parity_and_no_fallback", ["tests/operators/test_production_convergence.py"]),
]
POST_ARTIFACT_STAGES = frozenset({"manifest_and_evidence_drift"})


def _bootstrap() -> None:
    root, fe = str(FE_ROOT.parent), str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends
    load_all()
    register_sql_backends()


def _env() -> dict[str, str]:
    import os
    root, fe = str(FE_ROOT.parent), str(FE_ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{fe}"
    return env


def _run_pytest(files: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", *files, "--tb=line"]
    proc = subprocess.run(cmd, cwd=str(FE_ROOT), env=_env(), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()[-800:]


def _sync_case_registry_check() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(FE_ROOT / "scripts" / "sync_primitive_evidence.py"), "--check"],
        cwd=str(FE_ROOT), env=_env(), capture_output=True, text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def _refresh_operator_manifest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(FE_ROOT / "scripts" / "export_operator_manifest.py")],
        cwd=str(FE_ROOT), env=_env(), capture_output=True, text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def _build_verified_payload(*, stages_passed: list[str]) -> dict:
    from backend.evidence_provenance import (
        build_provenance, compute_case_registry_hash, enrich_operator_metadata, load_case_registry,
    )
    from tests.backend_parity.evidence_case_registry import six_way_certified_names
    from cleaned_operators.registry import OperatorRegistry

    registry = load_case_registry()
    polars_ref = frozenset(registry.get("polars_reference_parity") or [])
    polars_ed = frozenset(registry.get("polars_edge_verified") or [])
    duck_ref = frozenset(registry.get("duckdb_reference_parity") or [])
    duck_ed = frozenset(registry.get("duckdb_edge_verified") or [])
    duck_nan = frozenset(registry.get("duckdb_nan_edge_verified") or [])
    no_fb = frozenset(registry.get("no_fallback_verified") or [])
    dual = six_way_certified_names(
        polars_reference=polars_ref,
        polars_edge=polars_ed,
        duckdb_reference=duck_ref,
        duckdb_real_sql=frozenset(registry.get("duckdb_real_sql_verified") or []),
        duckdb_edge=duck_ed | duck_nan,
        no_fallback=no_fb,
    )
    dual = frozenset(
        name for name in dual
        if {"pandas_numpy", "polars", "sql"}.issubset(set(OperatorRegistry.backends_for(name)))
    )
    verified_path = FE_ROOT / "evidence" / "primitive_verified.json"
    existing_ops: dict[str, dict] = {}
    if verified_path.is_file():
        existing_ops = json.loads(verified_path.read_text(encoding="utf-8")).get("operators") or {}
    if not dual:
        raise RuntimeError("six-way evidence intersection is empty")

    edge_names = frozenset().union(*(
        frozenset(registry.get(key) or [])
        for key in ("duckdb_null_edge_verified", "duckdb_nan_edge_verified", "duckdb_inf_edge_verified")
    ))
    operator_names = dual | {name for name in edge_names if OperatorRegistry.backends_for(name)}
    payload = {
        "schema_version": 3,
        "description": "Primitive production evidence bound to semantic/test/source hashes",
        "polars_reference_parity": sorted(polars_ref & dual),
        "polars_edge_verified": sorted(polars_ed & dual),
        "duckdb_reference_parity": sorted(duck_ref & dual),
        "duckdb_real_sql_verified": sorted(frozenset(registry.get("duckdb_real_sql_verified") or []) & dual),
        "duckdb_edge_verified": sorted(duck_ed & dual),
        "duckdb_null_edge_verified": sorted(registry.get("duckdb_null_edge_verified") or []),
        "duckdb_nan_edge_verified": sorted(registry.get("duckdb_nan_edge_verified") or []),
        "duckdb_inf_edge_verified": sorted(registry.get("duckdb_inf_edge_verified") or []),
        "no_fallback_verified": sorted(no_fb & dual),
        "operators": enrich_operator_metadata(certified=operator_names, existing=existing_ops),
        "provenance": build_provenance(
            case_registry_hash=compute_case_registry_hash(registry),
            test_passed=True,
            stages_passed=stages_passed,
        ),
    }
    payload["_six_way_count"] = len(dual)
    return payload


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    restore = path.with_suffix(path.suffix + ".restore")
    restore.write_bytes(previous)
    restore.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true", help="仅输出未认证预览，不写 verified artifact")
    parser.add_argument("--check", action="store_true", help="校验 artifact 的语义/实现/测试哈希")
    args = parser.parse_args()
    _bootstrap()
    out = FE_ROOT / "evidence" / "primitive_verified.json"

    if args.check:
        from backend.evidence_provenance import evidence_artifact_validation_errors, load_verified_artifact
        # Squash/rebase-safe: semantic and implementation hashes are the hard
        # identity. commit_sha is retained only for audit/debug provenance.
        validation_errors = evidence_artifact_validation_errors(require_commit_match=False)
        if validation_errors:
            print("primitive_verified.json 无效:", file=sys.stderr)
            for error in validation_errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        data = load_verified_artifact()
        dual = set(data.get("polars_reference_parity") or [])
        for key in (
            "polars_edge_verified", "duckdb_reference_parity", "duckdb_real_sql_verified",
            "duckdb_edge_verified", "no_fallback_verified",
        ):
            dual &= set(data.get(key) or [])
        print(f"evidence artifact valid ({len(dual)} six-way test-certified; squash-safe hashes)")
        return 0

    ok_registry, registry_out = _sync_case_registry_check()
    if not ok_registry:
        print("case registry 未同步，先运行 sync_primitive_evidence.py", file=sys.stderr)
        print(registry_out, file=sys.stderr)
        return 1
    if args.skip_pytest:
        print("unverified_preview: --skip-pytest cannot write verified evidence")
        return 0

    stages_passed: list[str] = ["case_registry_sync"]
    for stage_name, files in CERTIFICATION_STAGES:
        if stage_name in POST_ARTIFACT_STAGES:
            continue
        ok, detail = _run_pytest(files)
        if not ok:
            print(f"stage failed: {stage_name}", file=sys.stderr)
            print(detail, file=sys.stderr)
            return 1
        stages_passed.append(stage_name)

    payload = _build_verified_payload(stages_passed=[
        *stages_passed,
        *(name for name, _ in CERTIFICATION_STAGES if name in POST_ARTIFACT_STAGES),
    ])
    count = payload.pop("_six_way_count", 0)
    previous = out.read_bytes() if out.is_file() else None
    manifest = FE_ROOT / "benchmarks" / "operator_manifest.json"
    previous_manifest = manifest.read_bytes() if manifest.is_file() else None
    candidate = out.with_suffix(".json.tmp")
    candidate.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate.replace(out)

    manifest_ok, manifest_detail = _refresh_operator_manifest()
    if not manifest_ok:
        _restore(out, previous)
        print("stage failed: refresh_operator_manifest", file=sys.stderr)
        print(manifest_detail, file=sys.stderr)
        return 1
    for stage_name, files in CERTIFICATION_STAGES:
        if stage_name not in POST_ARTIFACT_STAGES:
            continue
        ok, detail = _run_pytest(files)
        if not ok:
            _restore(out, previous)
            _restore(manifest, previous_manifest)
            print(f"stage failed: {stage_name}", file=sys.stderr)
            print(detail, file=sys.stderr)
            return 1
        stages_passed.append(stage_name)
    print(f"wrote {out} (test-certified six-way: {count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
