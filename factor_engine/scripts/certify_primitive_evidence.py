#!/usr/bin/env python3
"""Primitive evidence certification with a short-lived bootstrap candidate.

Parity tests must route the candidate backends before a fresh verified artifact
exists. The script creates an owner-only, nonce-bound temporary candidate,
runs parity stages in isolated subprocesses, writes real evidence only after
they pass, then reruns artifact/drift gates without the candidate context.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

PRE_CERTIFICATION_STAGES: list[tuple[str, list[str]]] = [
    ("polars_reference_parity", ["tests/backend_parity/test_production_core_triple_parity.py", "tests/backend_parity/test_production_safe_bulk_parity.py"]),
    ("polars_edge_verified", ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"]),
    ("duckdb_reference_parity", ["tests/backend_parity/test_production_core_triple_parity.py", "tests/backend_parity/test_production_safe_bulk_parity.py"]),
    ("duckdb_real_sql_verified", ["tests/backend_parity/test_production_core_triple_parity.py", "tests/backend_parity/test_production_safe_bulk_parity.py"]),
    ("duckdb_edge_verified", ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"]),
    ("no_fallback_verified", ["tests/backend_parity/test_polars_long_no_pandas_path.py"]),
    ("duckdb_nan_edge_verified", ["tests/backend_parity/test_p0_semantic_edges.py", "tests/backend_parity/test_rank_null_singleton_boundaries.py"]),
    ("duckdb_inf_edge_verified", ["tests/backend_parity/test_p0_semantic_edges.py", "tests/backend_parity/test_rank_null_singleton_boundaries.py"]),
    ("composite_reference_parity", ["tests/backend_parity/test_composite_reference_semantics.py"]),
    ("composite_triple_parity", ["tests/backend_parity/test_composite_lowered_triple_parity.py", "tests/backend_parity/test_composite_edge_triple_parity.py"]),
    ("polars_native_static_audit", ["tests/backend/test_polars_long_static_audit.py"]),
    ("parameter_signature_gate", ["tests/backend/test_operator_evidence_schema.py", "tests/backend/test_operator_call_policy.py", "tests/backend/test_production_checklist.py"]),
    ("plan_variant_parity", ["tests/backend_parity/test_plan_variant_parity.py"]),
    ("chunk_invariance", ["tests/backend_parity/test_chunk_invariance.py"]),
]

POST_WRITE_STAGES: list[tuple[str, list[str]]] = [
    ("manifest_and_evidence_drift", ["tests/backend/test_primitive_evidence_contract.py", "tests/backend/test_operator_manifest_freshness.py", "tests/operator_contracts/test_manifest_sync.py"]),
    ("strict_period_triple_parity_and_no_fallback", ["tests/operators/test_production_convergence.py"]),
]


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for path in (root, fe):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _env(candidate: tuple[Path, str] | None = None) -> dict[str, str]:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{fe}"
    if candidate is None:
        env.pop("FACTOR_ENGINE_CERTIFICATION_CANDIDATE", None)
        env.pop("FACTOR_ENGINE_CERTIFICATION_NONCE", None)
    else:
        path, nonce = candidate
        env["FACTOR_ENGINE_CERTIFICATION_CANDIDATE"] = str(path)
        env["FACTOR_ENGINE_CERTIFICATION_NONCE"] = nonce
    return env


def _run_pytest(files: list[str], *, candidate: tuple[Path, str] | None = None) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *files, "--tb=line"],
        cwd=str(FE_ROOT),
        env=_env(candidate),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()[-4000:]


def _sync_case_registry_check() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(FE_ROOT / "scripts" / "sync_primitive_evidence.py"), "--check"],
        cwd=str(FE_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def _existing_operator_metadata() -> dict:
    path = FE_ROOT / "evidence" / "primitive_verified.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("operators")
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate_artifact(registry: dict) -> dict:
    keys = (
        "polars_reference_parity", "polars_edge_verified",
        "duckdb_reference_parity", "duckdb_real_sql_verified",
        "duckdb_edge_verified", "duckdb_null_edge_verified",
        "duckdb_nan_edge_verified", "duckdb_inf_edge_verified",
        "no_fallback_verified",
    )
    artifact = {
        "schema_version": 3,
        "description": "short-lived certification candidate",
        "operators": _existing_operator_metadata(),
        "provenance": {"artifact_kind": "certification_candidate"},
    }
    for key in keys:
        artifact[key] = sorted(str(value) for value in (registry.get(key) or []))
    return artifact


def _create_candidate() -> tuple[Path, str]:
    from backend.evidence_provenance import compute_case_registry_hash, emitter_hashes, load_case_registry

    registry = load_case_registry()
    nonce = secrets.token_urlsafe(32)
    wrapper = {
        "artifact_kind": "certification_candidate",
        "nonce": nonce,
        "expires_at": time.time() + 3600,
        "case_registry_hash": compute_case_registry_hash(registry),
        "implementation_hashes": emitter_hashes(),
        "artifact": _candidate_artifact(registry),
    }
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="factorengine-certification-", suffix=".json", delete=False)
    try:
        os.chmod(handle.name, 0o600)
        json.dump(wrapper, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    finally:
        handle.close()
    return Path(handle.name), nonce


def _build_verified_payload(*, stages_passed: list[str]) -> dict:
    from backend.evidence_provenance import build_provenance, compute_case_registry_hash, enrich_operator_metadata, load_case_registry
    from tests.backend_parity.evidence_case_registry import six_way_certified_names

    registry = load_case_registry()
    polars_ref = frozenset(registry.get("polars_reference_parity") or [])
    polars_edge = frozenset(registry.get("polars_edge_verified") or [])
    duck_ref = frozenset(registry.get("duckdb_reference_parity") or [])
    duck_real = frozenset(registry.get("duckdb_real_sql_verified") or [])
    duck_edge = frozenset(registry.get("duckdb_edge_verified") or [])
    duck_nan = frozenset(registry.get("duckdb_nan_edge_verified") or [])
    no_fallback = frozenset(registry.get("no_fallback_verified") or [])
    dual = six_way_certified_names(
        polars_reference=polars_ref,
        polars_edge=polars_edge,
        duckdb_reference=duck_ref,
        duckdb_real_sql=duck_real,
        duckdb_edge=duck_edge | duck_nan,
        no_fallback=no_fallback,
    )
    if not dual:
        raise RuntimeError("six-way evidence intersection is empty")
    payload = {
        "schema_version": 3,
        "description": "Primitive production evidence（须 pytest 通过后由 certify_primitive_evidence.py 写入）",
        "polars_reference_parity": sorted(polars_ref & dual),
        "polars_edge_verified": sorted(polars_edge & dual),
        "duckdb_reference_parity": sorted(duck_ref & dual),
        "duckdb_real_sql_verified": sorted(duck_real & dual),
        "duckdb_edge_verified": sorted(duck_edge & dual),
        "duckdb_null_edge_verified": sorted(registry.get("duckdb_null_edge_verified") or []),
        "duckdb_nan_edge_verified": sorted(registry.get("duckdb_nan_edge_verified") or []),
        "duckdb_inf_edge_verified": sorted(registry.get("duckdb_inf_edge_verified") or []),
        "no_fallback_verified": sorted(no_fallback & dual),
        "operators": enrich_operator_metadata(certified=dual, existing=_existing_operator_metadata()),
        "provenance": build_provenance(
            case_registry_hash=compute_case_registry_hash(registry),
            test_passed=True,
            stages_passed=stages_passed,
        ),
    }
    payload["_six_way_count"] = len(dual)
    return payload


def _generate_governance_artifacts() -> tuple[bool, str]:
    commands = [
        [sys.executable, str(FE_ROOT / "scripts" / "generate_operators_catalog.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "export_dsl_allowlist.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "generate_layer_manifests.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "generate_operator_usage_report.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "sync_backend_docs.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "generate_backend_evidence_manifest.py")],
        [sys.executable, str(FE_ROOT / "scripts" / "export_operator_manifest.py"), "--out", str(FE_ROOT / "benchmarks" / "operator_manifest.json")],
    ]
    output: list[str] = []
    for command in commands:
        proc = subprocess.run(command, cwd=str(FE_ROOT), env=_env(), capture_output=True, text=True)
        output.append((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            return False, "".join(output)[-4000:]
    return True, "".join(output)[-4000:]


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true", help="仅输出未认证预览，不写入 primitive_verified.json")
    parser.add_argument("--check", action="store_true", help="校验 verified artifact 与当前 commit/实现一致")
    args = parser.parse_args()
    _bootstrap()
    output_path = FE_ROOT / "evidence" / "primitive_verified.json"

    if args.check:
        from backend.evidence_provenance import evidence_artifact_valid, load_verified_artifact

        if not evidence_artifact_valid(require_commit_match=True):
            print("primitive_verified.json 无效或 commit/实现不匹配", file=sys.stderr)
            return 1
        data = load_verified_artifact()
        dual = set(data.get("polars_reference_parity") or [])
        for key in ("polars_edge_verified", "duckdb_reference_parity", "duckdb_real_sql_verified", "duckdb_edge_verified", "no_fallback_verified"):
            dual &= set(data.get(key) or [])
        print(f"evidence artifact valid ({len(dual)} six-way test-certified)")
        return 0

    registry_ok, registry_detail = _sync_case_registry_check()
    if not registry_ok:
        print("case registry 未同步，先运行 sync_primitive_evidence.py", file=sys.stderr)
        print(registry_detail, file=sys.stderr)
        return 1
    if args.skip_pytest:
        print("unverified_preview: --skip-pytest cannot write verified evidence")
        return 0

    candidate = _create_candidate()
    previous = output_path.read_bytes() if output_path.is_file() else None
    stages_passed: list[str] = ["case_registry_sync"]
    try:
        for stage_name, files in PRE_CERTIFICATION_STAGES:
            print(f"running certification stage: {stage_name}", flush=True)
            ok, detail = _run_pytest(files, candidate=candidate)
            if not ok:
                print(f"stage failed: {stage_name}", file=sys.stderr)
                print(detail, file=sys.stderr)
                return 1
            stages_passed.append(stage_name)

        payload = _build_verified_payload(stages_passed=stages_passed)
        count = payload.pop("_six_way_count", 0)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        generated_ok, generated_detail = _generate_governance_artifacts()
        if not generated_ok:
            _restore(output_path, previous)
            print("governance artifact generation failed", file=sys.stderr)
            print(generated_detail, file=sys.stderr)
            return 1
        stages_passed.append("governance_artifact_generation")

        for stage_name, files in POST_WRITE_STAGES:
            print(f"running post-write stage: {stage_name}", flush=True)
            ok, detail = _run_pytest(files, candidate=None)
            if not ok:
                _restore(output_path, previous)
                print(f"post-write stage failed: {stage_name}", file=sys.stderr)
                print(detail, file=sys.stderr)
                return 1
            stages_passed.append(stage_name)

        payload = _build_verified_payload(stages_passed=stages_passed)
        count = payload.pop("_six_way_count", count)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {output_path} (test-certified six-way: {count})")
        return 0
    finally:
        candidate[0].unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
