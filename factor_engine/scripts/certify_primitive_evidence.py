#!/usr/bin/env python3
"""Primitive evidence 认证：先跑 pytest，再写入 ``evidence/primitive_verified.json``。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

# CI required checks → pytest 模块（须全部通过才授予 verified）
CERTIFICATION_STAGES: list[tuple[str, list[str]]] = [
    (
        "polars_reference_parity",
        [
            "tests/backend_parity/test_production_core_triple_parity.py",
            "tests/backend_parity/test_production_safe_bulk_parity.py",
        ],
    ),
    (
        "polars_edge_verified",
        ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"],
    ),
    (
        "duckdb_reference_parity",
        [
            "tests/backend_parity/test_production_core_triple_parity.py",
            "tests/backend_parity/test_production_safe_bulk_parity.py",
        ],
    ),
    (
        "duckdb_real_sql_verified",
        [
            "tests/backend_parity/test_production_core_triple_parity.py",
            "tests/backend_parity/test_production_safe_bulk_parity.py",
        ],
    ),
    (
        "duckdb_edge_verified",
        ["tests/backend_parity/test_p0_edge_cases_triple_parity.py"],
    ),
    (
        "no_fallback_verified",
        ["tests/backend_parity/test_polars_long_no_pandas_path.py"],
    ),
    (
        "duckdb_nan_edge_verified",
        [
            "tests/backend_parity/test_p0_semantic_edges.py",
            "tests/backend_parity/test_rank_null_singleton_boundaries.py",
        ],
    ),
    (
        "duckdb_inf_edge_verified",
        [
            "tests/backend_parity/test_p0_semantic_edges.py",
            "tests/backend_parity/test_rank_null_singleton_boundaries.py",
        ],
    ),
    (
        "composite_reference_parity",
        ["tests/backend_parity/test_composite_reference_semantics.py"],
    ),
    (
        "composite_triple_parity",
        [
            "tests/backend_parity/test_composite_lowered_triple_parity.py",
            "tests/backend_parity/test_composite_edge_triple_parity.py",
        ],
    ),
    (
        "polars_native_static_audit",
        ["tests/backend/test_polars_long_static_audit.py"],
    ),
    (
        "parameter_signature_gate",
        [
            "tests/backend/test_operator_evidence_schema.py",
            "tests/backend/test_operator_call_policy.py",
            "tests/backend/test_production_checklist.py",
        ],
    ),
    (
        "plan_variant_parity",
        ["tests/backend_parity/test_plan_variant_parity.py"],
    ),
    (
        "chunk_invariance",
        ["tests/backend_parity/test_chunk_invariance.py"],
    ),
    (
        "manifest_and_evidence_drift",
        [
            "tests/backend/test_primitive_evidence_contract.py",
            "tests/backend/test_operator_manifest_freshness.py",
            "tests/operator_contracts/test_manifest_sync.py",
        ],
    ),
    (
        "strict_period_triple_parity_and_no_fallback",
        ["tests/operators/test_production_convergence.py"],
    ),
]

# These checks consume the artifact being certified.  Running them before the
# candidate is visible creates a deadlock whenever the evidence schema/hash
# changes: capability fails closed, which makes the generated manifest appear
# stale, so a new valid artifact can never be produced.  They run transactionally
# after the candidate write and restore the previous artifact on failure.
POST_ARTIFACT_STAGES = frozenset({"manifest_and_evidence_drift"})


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _env() -> dict[str, str]:
    import os

    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
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
        cwd=str(FE_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def _build_verified_payload(*, stages_passed: list[str]) -> dict:
    from backend.evidence_provenance import (
        build_provenance,
        compute_case_registry_hash,
        enrich_operator_metadata,
        load_case_registry,
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
    existing_ops: dict[str, dict] = {}
    verified_path = FE_ROOT / "evidence" / "primitive_verified.json"
    if verified_path.is_file():
        existing_ops = json.loads(verified_path.read_text(encoding="utf-8")).get("operators") or {}

    if not dual:
        raise RuntimeError("six-way evidence intersection is empty")

    # Edge lists may include compiler-internal names (e.g. protected_div) that are
    # not six-way dual-certified.  Keep them in the edge sets but also record them
    # under ``operators`` so evidence_artifact_valid's subset check passes.
    edge_names = frozenset().union(
        *(
            frozenset(registry.get(key) or [])
            for key in (
                "duckdb_null_edge_verified",
                "duckdb_nan_edge_verified",
                "duckdb_inf_edge_verified",
            )
        )
    )
    operator_names = dual | {
        name for name in edge_names
        if OperatorRegistry.backends_for(name)
    }

    payload = {
        "schema_version": 3,
        "description": (
            "Primitive production evidence（须 pytest 通过后由 certify_primitive_evidence.py 写入）"
        ),
        "polars_reference_parity": sorted(polars_ref & dual),
        "polars_edge_verified": sorted(polars_ed & dual),
        "duckdb_reference_parity": sorted(duck_ref & dual),
        "duckdb_real_sql_verified": sorted(frozenset(registry.get("duckdb_real_sql_verified") or []) & dual),
        "duckdb_edge_verified": sorted(duck_ed & dual),
        "duckdb_null_edge_verified": sorted(registry.get("duckdb_null_edge_verified") or []),
        "duckdb_nan_edge_verified": sorted(registry.get("duckdb_nan_edge_verified") or []),
        "duckdb_inf_edge_verified": sorted(registry.get("duckdb_inf_edge_verified") or []),
        "no_fallback_verified": sorted(no_fb & dual) if dual else sorted(no_fb),
        "operators": enrich_operator_metadata(certified=operator_names, existing=existing_ops),
        "provenance": build_provenance(
            case_registry_hash=compute_case_registry_hash(registry),
            test_passed=True,
            stages_passed=stages_passed,
        ),
    }
    payload["_six_way_count"] = len(dual)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="仅输出未认证预览，不写入 primitive_verified.json",
    )
    parser.add_argument("--check", action="store_true", help="校验 verified artifact 与当前 commit 一致")
    args = parser.parse_args()
    _bootstrap()

    out = FE_ROOT / "evidence" / "primitive_verified.json"

    if args.check:
        from backend.evidence_provenance import (
            evidence_artifact_validation_errors,
            load_verified_artifact,
        )

        validation_errors = evidence_artifact_validation_errors(require_commit_match=True)
        if validation_errors:
            print("primitive_verified.json 无效:", file=sys.stderr)
            for error in validation_errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        data = load_verified_artifact()
        dual = set(data.get("polars_reference_parity") or [])
        for key in (
            "polars_edge_verified",
            "duckdb_reference_parity",
            "duckdb_real_sql_verified",
            "duckdb_edge_verified",
            "no_fallback_verified",
        ):
            dual &= set(data.get(key) or [])
        count = len(dual)
        print(f"evidence artifact valid ({count} six-way test-certified)")
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

    payload = _build_verified_payload(
        stages_passed=[
            *stages_passed,
            *(name for name, _ in CERTIFICATION_STAGES if name in POST_ARTIFACT_STAGES),
        ]
    )
    count = payload.pop("_six_way_count", 0)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    previous = out.read_bytes() if out.is_file() else None
    candidate = out.with_suffix(".json.tmp")
    candidate.write_text(text, encoding="utf-8")
    candidate.replace(out)
    for stage_name, files in CERTIFICATION_STAGES:
        if stage_name not in POST_ARTIFACT_STAGES:
            continue
        ok, detail = _run_pytest(files)
        if not ok:
            if previous is None:
                out.unlink(missing_ok=True)
            else:
                restore = out.with_suffix(".json.restore")
                restore.write_bytes(previous)
                restore.replace(out)
            print(f"stage failed: {stage_name}", file=sys.stderr)
            print(detail, file=sys.stderr)
            return 1
        stages_passed.append(stage_name)
    print(f"wrote {out} (test-certified six-way: {count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
