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
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

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
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
    load_all()
    register_sql_backends()


def _env() -> dict[str, str]:
    import os
    root, fe = str(FE_ROOT.parent), str(FE_ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{fe}"
    return env


def _run_pytest(files: list[str], junitxml: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable, "-m", "pytest", "-q", *files, "--tb=line",
        f"--junitxml={junitxml}",
    ]
    proc = subprocess.run(cmd, cwd=str(FE_ROOT), env=_env(), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()[-800:]


# ---------------------------------------------------------------------------
# Audit #382/#383: per-test-case executed+passed recording (JUnit XML).
#
# A pytest file exiting 0 is NOT sufficient evidence: ``pytest.importorskip``
# makes an entire file report SKIP while still exiting 0, and a too-small parity
# fixture can pass with both sides all-NaN.  We therefore parse the JUnit XML
# pytest emits, require failed==0 AND skipped==0, and derive the verified set
# from the test cases that actually executed and passed.
# ---------------------------------------------------------------------------

# Optional per-canonical expected-fixture file map for the #384 finite-coverage
# gate.  In-memory parity fixtures are not materialised on disk, so the default
# is empty; a backend that writes expected outputs can register them here and
# the certifier will downgrade any case whose expected output is NaN-dominated.
_FIXTURE_PATHS: dict[str, str] = {}

# Optional per-canonical parameter-domain declarations (audit #385).  A case that
# declares lower/typical/upper/active_when/missing_pattern is certified for
# exactly that domain; anything else is certified for the default parameters
# only, recorded as ``{"bounds": ["default"]}``.
_PARAM_DOMAINS: dict[str, dict] = {}


def _resolve_canonical(name: str) -> str:
    """Normalise a parametrized case name through the operator alias table."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return str(OperatorRegistry._aliases.get(name, name))


def _extract_canonical(tc_name: str, candidates: frozenset[str]) -> str | None:
    """Extract the canonical operator name from a JUnit testcase ``name``.

    Handles parametrized names of the form ``test_xxx[canon-<lambda>0]`` /
    ``test_xxx[canon-1]`` / ``test_xxx[canon]``.  The first bracket parameter is
    the operator/canonical name; canonical names never contain ``-``, so the
    first ``-`` split is reliable.  Falls back to prefix matching so aliases that
    normalise into the candidate set still match.
    """
    match = re.search(r"\[(.*)\]$", str(tc_name))
    if not match:
        return None
    inner = match.group(1)
    first = inner.split("-", 1)[0].strip()
    if not first:
        return None
    resolved = _resolve_canonical(first)
    if resolved in candidates:
        return resolved
    for canon in sorted(candidates, key=len, reverse=True):
        if inner == canon or inner.startswith(canon + "-") or inner.startswith(canon + "["):
            return canon
    return None


def _classname_to_file(classname: str) -> str:
    """``tests.backend_parity.test_foo`` -> ``tests/backend_parity/test_foo.py``."""
    return classname.replace(".", "/") + ".py"


def _backend_for_stage(stage_name: str) -> str:
    if "duckdb" in stage_name:
        return "duckdb"
    if "polars" in stage_name or stage_name == "no_fallback_verified":
        return "polars"
    return "pandas"


def _parse_junit_records(
    junit_path: Path,
    files: list[str],
    candidates: frozenset[str],
    stage_name: str,
) -> tuple[bool, list[str], list[dict]]:
    """Parse one stage's JUnit XML.

    Returns ``(ok, no_evidence_files, records)``.  ``ok`` is False when a file
    supplied NO evidence — every testcase it emitted was skipped (e.g. a
    whole-file ``pytest.importorskip``), meaning the dependency needed to prove
    its cases was absent (audit #382 evidence-execution gate).

    A per-case skip is different: it is a legitimate surface exclusion (e.g. an
    operator not on the production daily surface, guarded by ``pytest.skip``
    inside the test).  Such a case certifies nothing (skip != pass — it is not
    added to ``records``) but must not void the file's other executed+passed
    evidence.  ``records`` contains one entry per executed+passed per-canonical
    test case.
    """
    if not junit_path.is_file():
        return False, ["<missing-junit-xml>"], []
    no_evidence_files: list[str] = []
    per_file: dict[str, dict] = {}
    records: list[dict] = []
    try:
        tree = ET.parse(junit_path)
    except ET.ParseError as exc:
        return False, [f"<invalid-junit-xml: {exc}>"], []
    root = tree.getroot()
    for testsuite in root.iter("testsuite"):
        for tc in testsuite.iter("testcase"):
            classname = str(tc.get("classname") or "")
            file_rel = _classname_to_file(classname)
            if file_rel not in files:
                continue
            entry = per_file.setdefault(classname, {"executed": 0, "skipped": 0})
            if tc.find("skipped") is not None:
                entry["skipped"] += 1
                continue
            entry["executed"] += 1
            if tc.find("failure") is not None or tc.find("error") is not None:
                # A failed/errored case also ran; the stage already failed on the
                # pytest exit code before we get here.
                continue
            tc_name = str(tc.get("name") or "")
            canon = _extract_canonical(tc_name, candidates)
            if canon is None:
                # Non per-canonical helper test (e.g. a fixture/schema check) —
                # it executed and passed but does not certify a single operator.
                continue
            records.append(
                {
                    "canonical": canon,
                    "backend": _backend_for_stage(stage_name),
                    "case_id": f"{classname}::{tc_name}",
                    "status": "passed",
                    "params": (tc_name.split("[", 1)[1].rstrip("]") if "[" in tc_name else "n/a"),
                    "fixture_hash": "n/a",
                    "output_hash": "n/a",
                    "stage": stage_name,
                }
            )
    for classname, entry in per_file.items():
        if entry["executed"] == 0:
            no_evidence_files.append(classname)
    return (not no_evidence_files), no_evidence_files, records


def _executed_passed_sets(executed_records: list[dict]) -> dict[str, set[str]]:
    """Group executed+passed canonicals by stage."""
    sets: dict[str, set[str]] = {}
    for rec in executed_records:
        if rec.get("status") != "passed":
            continue
        stage = rec.get("stage")
        if not stage:
            continue
        sets.setdefault(stage, set()).add(str(rec["canonical"]))
    return sets


def _load_fixture_array(path: Path):
    """Load a numeric array from a .csv/.parquet/.npy/.json/.npz fixture."""
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".csv.gz"}:
            import pandas as pd

            return pd.read_csv(path).to_numpy(dtype=float)
        if suffix == ".parquet":
            import pandas as pd

            return pd.read_parquet(path).to_numpy(dtype=float)
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".json":
            import pandas as pd

            return pd.read_json(path).to_numpy(dtype=float)
    except Exception:
        return None
    return None


def _finite_coverage_ok(fixture_path: str) -> tuple[bool, float]:
    """Audit #384: gate a parity case on the finite coverage of its expected
    fixture output.

    Returns ``(ok, finite_ratio)``.  A case whose expected output is more than
    half NaN/Inf (``finite_ratio < 0.5``) — and is not an explicit all-null edge
    contract — cannot certify numeric equality, so it is downgraded.  In-memory
    parity fixtures are not on disk, so a non-existent path is vacuously OK
    (``(True, 1.0)``); the gate is authoritative only where a backend actually
    materialises the expected fixture.
    """
    path = Path(fixture_path)
    if not path.exists():
        return True, 1.0
    arr = _load_fixture_array(path)
    if arr is None or arr.size == 0:
        return False, 0.0
    try:
        finite = float(np.isfinite(arr.astype(float)).mean())
    except (TypeError, ValueError):
        return False, 0.0
    return finite >= 0.5, finite


def _certified_parameter_domain(canonical: str) -> dict:
    """Audit #385: the parameter domain a certified case actually proves."""
    domain = _PARAM_DOMAINS.get(canonical)
    if domain is not None:
        return dict(domain)
    # Default: the certified case ran the operator's default parameters, so the
    # certified domain is exactly those defaults — nothing else is claimed.
    return {"bounds": ["default"]}


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


def _build_verified_payload(
    *,
    stages_passed: list[str],
    executed_records: list[dict],
    downgraded: dict[str, str] | None = None,
) -> dict:
    from factor_engine.backend.evidence_provenance import (
        build_provenance, compute_case_registry_hash, enrich_operator_metadata, load_case_registry,
    )
    from tests.backend_parity.evidence_case_registry import six_way_certified_names
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    registry = load_case_registry()
    # Audit #382/#383: the verified set is derived from executed+passed test
    # cases (parsed from JUnit), intersected with the declared case registry so a
    # registry name that was skipped / unexecuted / unparseable is NEVER
    # certified.
    passed = _executed_passed_sets(executed_records)

    # R38 evidence-regen blocker: route the ``strict_period_*`` stage's executed
    # records into the six evidence fields by test-function granularity.  The
    # stage runs ``tests/operators/test_production_convergence.py``:
    #   - ``test_strict_period_polars_*``   -> polars_reference_parity (pandas<->polars parity)
    #   - ``test_strict_period_duckdb_*``   -> duckdb_real_sql_verified (real SQL execution)
    # Without this, the 7 strict-period primitives are dropped from every field
    # (they have no dedicated case-registry key) even though their parity tests
    # pass — the certification pipeline never attributes them.
    strict_stage = "strict_period_triple_parity_and_no_fallback"
    for rec in executed_records:
        if rec.get("stage") != strict_stage or rec.get("status") != "passed":
            continue
        case_id = str(rec.get("case_id") or "")
        canon = str(rec["canonical"])
        if "test_strict_period_polars" in case_id:
            passed.setdefault("polars_reference_parity", set()).add(canon)
        elif "test_strict_period_duckdb" in case_id:
            passed.setdefault("duckdb_real_sql_verified", set()).add(canon)

    def stage(key: str) -> frozenset[str]:
        static = frozenset(registry.get(key) or [])
        return static & frozenset(passed.get(key) or [])

    polars_ref = stage("polars_reference_parity")
    polars_ed = stage("polars_edge_verified")
    duck_ref = stage("duckdb_reference_parity")
    duck_real = stage("duckdb_real_sql_verified")
    duck_ed = stage("duckdb_edge_verified")
    duck_nan = stage("duckdb_nan_edge_verified")
    no_fb = stage("no_fallback_verified")
    dual = six_way_certified_names(
        polars_reference=polars_ref,
        polars_edge=polars_ed,
        duckdb_reference=duck_ref,
        duckdb_real_sql=duck_real,
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
        frozenset(passed.get(key) or [])
        for key in ("duckdb_null_edge_verified", "duckdb_nan_edge_verified", "duckdb_inf_edge_verified")
    ))
    operator_names = dual | {name for name in edge_names if OperatorRegistry.backends_for(name)}

    # Audit #384: downgrade cases whose expected fixture output is NaN-dominated.
    downgraded = dict(downgraded or {})
    kept: list[str] = []
    for name in sorted(operator_names):
        fixture = _FIXTURE_PATHS.get(name, name)
        ok, ratio = _finite_coverage_ok(fixture)
        if ok:
            kept.append(name)
        else:
            downgraded[name] = f"finite_coverage < threshold (ratio={ratio:.3f})"
    operator_names = set(kept)

    operators = enrich_operator_metadata(certified=frozenset(operator_names), existing=existing_ops)
    # Audit #385: bind each certified operator to the parameter domain its cases
    # actually proved (default-only unless a case declares otherwise).
    for name in operator_names:
        rec = operators.get(name)
        if isinstance(rec, dict):
            rec["certified_parameter_domain"] = _certified_parameter_domain(name)
    payload = {
        "schema_version": 3,
        "description": "Primitive production evidence bound to semantic/test/source hashes",
        "polars_reference_parity": sorted(polars_ref & dual),
        "polars_edge_verified": sorted(polars_ed & dual),
        "duckdb_reference_parity": sorted(duck_ref & dual),
        "duckdb_real_sql_verified": sorted(duck_real & dual),
        "duckdb_edge_verified": sorted(duck_ed & dual),
        "duckdb_null_edge_verified": sorted(stage("duckdb_null_edge_verified")),
        "duckdb_nan_edge_verified": sorted(duck_nan),
        "duckdb_inf_edge_verified": sorted(stage("duckdb_inf_edge_verified")),
        "no_fallback_verified": sorted(no_fb & dual),
        "operators": operators,
        "executed_records": executed_records,
        "downgraded_cases": dict(sorted(downgraded.items())),
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
    parser.add_argument(
        "--junitxml-out",
        default="",
        help="可选：导出逐 test-case executed+passed 记录 JSON 的路径",
    )
    args = parser.parse_args()
    _bootstrap()
    out = FE_ROOT / "evidence" / "primitive_verified.json"

    if args.check:
        from factor_engine.backend.evidence_provenance import evidence_artifact_validation_errors, load_verified_artifact
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

    from factor_engine.backend.evidence_provenance import load_case_registry

    registry = load_case_registry()
    stages_passed: list[str] = ["case_registry_sync"]
    executed_records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="fe-certify-primitive-") as tmpdir:
        for stage_name, files in CERTIFICATION_STAGES:
            if stage_name in POST_ARTIFACT_STAGES:
                continue
            junit_path = Path(tmpdir) / f"{stage_name}.xml"
            ok, detail = _run_pytest(files, junit_path)
            if not ok:
                print(f"stage failed: {stage_name}", file=sys.stderr)
                print(detail, file=sys.stderr)
                return 1
            candidates = frozenset(registry.get(stage_name) or [])
            if not candidates:
                # R38 evidence-regen blocker: the ``strict_period_triple_parity_and_no_fallback``
                # stage has no dedicated case-registry key (its cases are declared under the
                # six evidence fields), so ``_extract_canonical`` could never match and the
                # 7 strict-period primitives were silently dropped from every certified set.
                # Feed it the union of the six declared evidence fields as candidates so the
                # executed+passed JUnit cases can be attributed to their canonicals.
                candidates = frozenset().union(
                    *(frozenset(registry.get(k) or []) for k in (
                        "polars_reference_parity", "polars_edge_verified",
                        "duckdb_reference_parity", "duckdb_real_sql_verified",
                        "duckdb_edge_verified", "no_fallback_verified",
                    ))
                )
            stage_ok, skipped_files, records = _parse_junit_records(
                junit_path, files, candidates, stage_name
            )
            if not stage_ok:
                print(
                    f"stage failed: {stage_name}: no executed test cases in "
                    f"{', '.join(sorted(skipped_files))}",
                    file=sys.stderr,
                )
                print("required dependency missing -> certification FAIL", file=sys.stderr)
                return 1
            executed_records.extend(records)
            stages_passed.append(stage_name)

        downgraded: dict[str, str] = {}
        payload = _build_verified_payload(
            stages_passed=[
                *stages_passed,
                *(name for name, _ in CERTIFICATION_STAGES if name in POST_ARTIFACT_STAGES),
            ],
            executed_records=executed_records,
            downgraded=downgraded,
        )
        if args.junitxml_out:
            Path(args.junitxml_out).write_text(
                json.dumps(executed_records, indent=2, ensure_ascii=False), encoding="utf-8"
            )
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
            junit_path = Path(tmpdir) / f"{stage_name}.xml"
            ok, detail = _run_pytest(files, junit_path)
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
