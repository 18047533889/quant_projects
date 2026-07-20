# -*- coding: utf-8
"""Evidence artifact 溯源：绑定 commit、依赖版本、case hash、implementation hash。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import platform
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FE_ROOT = Path(__file__).resolve().parents[1]
CASE_REGISTRY_JSON = FE_ROOT / "evidence" / "primitive_case_registry.json"
VERIFIED_JSON = FE_ROOT / "evidence" / "primitive_verified.json"


def current_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(FE_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _commit_is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return whether certified code is reachable from the current commit.

    Exact HEAD equality is impossible to preserve for a tracked evidence file:
    committing the freshly certified artifact necessarily creates a new HEAD.
    Reachability plus exact case-registry and emitter hashes gives stable,
    fail-closed provenance without that self-referential SHA cycle.
    """
    if not ancestor or not descendant:
        return False
    try:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(FE_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except Exception:
        return False


def collect_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    for mod, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("polars", "polars"),
        ("duckdb", "duckdb"),
        ("pyarrow", "pyarrow"),
    ):
        try:
            m = __import__(mod)
            versions[key] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            versions[key] = "missing"
    versions["os"] = platform.system()
    versions["architecture"] = platform.machine()
    return versions


def compute_payload_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_case_registry_hash(registry: Mapping[str, Any]) -> str:
    keys = (
        "polars_reference_parity",
        "polars_edge_verified",
        "duckdb_reference_parity",
        "duckdb_real_sql_verified",
        "duckdb_edge_verified",
        "duckdb_null_edge_verified",
        "duckdb_nan_edge_verified",
        "duckdb_inf_edge_verified",
        "no_fallback_verified",
    )
    subset = {k: sorted(registry.get(k) or []) for k in keys}
    return compute_payload_hash(subset)


def compute_implementation_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def emitter_hashes() -> dict[str, str]:
    polars_path = FE_ROOT / "backend" / "polars_expr_emitter.py"
    duck_path = FE_ROOT / "backend" / "sql_pushdown" / "emitter.py"
    out: dict[str, str] = {}
    if polars_path.is_file():
        out["implementation_hash_polars_emitter"] = compute_implementation_hash(
            polars_path.read_text(encoding="utf-8")
        )
    if duck_path.is_file():
        out["implementation_hash_duckdb_emitter"] = compute_implementation_hash(
            duck_path.read_text(encoding="utf-8")
        )
    return out


def build_provenance(
    *,
    case_registry_hash: str,
    test_passed: bool,
    stages_passed: list[str],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "artifact_kind": "test_passed",
        "commit_sha": current_commit_sha(),
        "passed_at": now if test_passed else "",
        "runtime_versions": collect_runtime_versions(),
        "case_registry_hash": case_registry_hash,
        "stages_passed": stages_passed,
        "emitter_hashes": emitter_hashes(),
    }


def load_case_registry() -> dict[str, Any]:
    if not CASE_REGISTRY_JSON.is_file():
        raise FileNotFoundError(f"missing case registry: {CASE_REGISTRY_JSON}")
    return json.loads(CASE_REGISTRY_JSON.read_text(encoding="utf-8"))


def load_verified_artifact() -> dict[str, Any]:
    if not VERIFIED_JSON.is_file():
        raise FileNotFoundError(f"missing verified artifact: {VERIFIED_JSON}")
    return json.loads(VERIFIED_JSON.read_text(encoding="utf-8"))


def evidence_artifact_validation_errors(
    *, require_commit_match: bool = False
) -> list[str]:
    """Return every reason the committed primitive evidence is invalid.

    Keep this function uncached so CI diagnostics always describe the files on
    disk at the time of the check.  Production callers should use the cached
    boolean wrapper below.
    """
    errors: list[str] = []
    try:
        data = load_verified_artifact()
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as exc:
        return [f"artifact_load: {type(exc).__name__}: {exc}"]
    prov = data.get("provenance") or {}
    if prov.get("artifact_kind") != "test_passed":
        errors.append(
            f"artifact_kind: expected='test_passed' actual={prov.get('artifact_kind')!r}"
        )
    if not prov.get("passed_at") or not prov.get("commit_sha"):
        errors.append("provenance: passed_at and commit_sha are required")
    if require_commit_match and not _commit_is_ancestor(
        str(prov.get("commit_sha") or ""), current_commit_sha()
    ):
        errors.append(
            "commit_ancestry: certified commit "
            f"{prov.get('commit_sha')!r} is not an ancestor of {current_commit_sha()!r}"
        )
    try:
        registry = load_case_registry()
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError) as exc:
        return errors + [f"case_registry_load: {type(exc).__name__}: {exc}"]
    if not isinstance(registry, dict):
        return errors + [f"case_registry_type: expected=dict actual={type(registry).__name__}"]
    expected_case_hash = compute_case_registry_hash(registry)
    if prov.get("case_registry_hash") != expected_case_hash:
        errors.append(
            "case_registry_hash: "
            f"expected={expected_case_hash!r} actual={prov.get('case_registry_hash')!r}"
        )
    expected_emitters = emitter_hashes()
    recorded_emitters = dict(prov.get("emitter_hashes") or {})
    if recorded_emitters != expected_emitters:
        errors.append(
            f"emitter_hashes: expected={expected_emitters!r} actual={recorded_emitters!r}"
        )
    recorded_versions = dict(prov.get("runtime_versions") or {})
    current_versions = collect_runtime_versions()
    operators = data.get("operators") or {}
    if not isinstance(operators, dict):
        return errors + [f"operators_type: expected=dict actual={type(operators).__name__}"]
    for canonical, record in operators.items():
        if not isinstance(record, dict):
            errors.append(
                f"operator[{canonical}]: expected=dict actual={type(record).__name__}"
            )
            continue
        for backend_key in ("pandas", "polars", "duckdb"):
            hash_key = f"implementation_hash_{backend_key}"
            source_key = f"implementation_source_{backend_key}"
            recorded_hash = str(record.get(hash_key) or "")
            recorded_source = str(record.get(source_key) or "")
            if not recorded_hash:
                continue
            if not recorded_source:
                errors.append(f"operator[{canonical}].{source_key}: missing")
                continue
            source_path = (FE_ROOT / recorded_source).resolve()
            try:
                source_path.relative_to(FE_ROOT.resolve())
            except ValueError:
                errors.append(
                    f"operator[{canonical}].{source_key}: escapes factor_engine root"
                )
                continue
            expected_hash = _source_hash(source_path)
            if expected_hash != recorded_hash:
                errors.append(
                    f"operator[{canonical}].{hash_key}: expected={expected_hash!r} "
                    f"actual={recorded_hash!r} source={recorded_source!r}"
                )
        try:
            semantic_expected = semantic_hashes_for(str(canonical))
        except (ImportError, AttributeError, RuntimeError) as exc:
            # During registry bootstrap operator_policy intentionally imports
            # primitive_evidence.  A partially initialized semantic module must
            # fail closed, never break package import or grant capability.
            errors.append(
                f"operator[{canonical}].semantic_hashes: {type(exc).__name__}: {exc}"
            )
            continue
        for key, value in semantic_expected.items():
            if record.get(key) != value:
                errors.append(
                    f"operator[{canonical}].{key}: expected={value!r} "
                    f"actual={record.get(key)!r}"
                )
        recorded_domain_hash = str(record.get("parameter_domain_hash") or "")
        if recorded_domain_hash:
            try:
                expected_domain_hash = parameter_domain_hash_for(str(canonical))
            except (ImportError, AttributeError, RuntimeError) as exc:
                errors.append(
                    f"operator[{canonical}].parameter_domain_hash: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if recorded_domain_hash != expected_domain_hash:
                    errors.append(
                        f"operator[{canonical}].parameter_domain_hash: "
                        f"expected={expected_domain_hash!r} actual={recorded_domain_hash!r}"
                    )
    verified_names = set()
    for key in (
        "polars_reference_parity", "polars_edge_verified", "duckdb_reference_parity",
        "duckdb_real_sql_verified", "duckdb_edge_verified", "duckdb_null_edge_verified",
        "duckdb_nan_edge_verified", "duckdb_inf_edge_verified", "no_fallback_verified",
    ):
        verified_names.update(str(name) for name in (data.get(key) or []))
    if not verified_names.issubset(set(operators)):
        errors.append(
            "verified_operator_records: missing="
            f"{sorted(verified_names.difference(set(operators)))!r}"
        )
    # Platform/architecture remain provenance metadata, but do not invalidate
    # portable semantic evidence.  Python minor AST/runtime differences are
    # covered by source/golden hashes; require the Python major and library
    # major.minor ABI/semantic families.
    for key in ("python", "numpy", "pandas", "polars", "duckdb", "pyarrow"):
        recorded = str(recorded_versions.get(key, ""))
        current = str(current_versions.get(key, ""))
        if key == "python":
            recorded = recorded.split(".")[0]
            current = current.split(".")[0]
        else:
            recorded = ".".join(recorded.split(".")[:2])
            current = ".".join(current.split(".")[:2])
        if not recorded or recorded != current:
            errors.append(
                f"runtime_versions.{key}: expected_family={current!r} "
                f"actual_family={recorded!r}"
            )
    return errors


@lru_cache(maxsize=2)
def evidence_artifact_valid(*, require_commit_match: bool = False) -> bool:
    """验证 evidence 是否仍绑定当前代码、case registry 与 emitter 实现。"""
    return not evidence_artifact_validation_errors(
        require_commit_match=require_commit_match
    )


@lru_cache(maxsize=256)
def _source_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return compute_implementation_hash(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def _tree_hash(root: Path, patterns: tuple[str, ...] = ("*.py", "*.json", "*.csv")) -> str:
    """Hash a deterministic source/data tree while excluding generated caches."""
    if not root.exists():
        return ""
    files: set[Path] = set()
    for pattern in patterns:
        files.update(p for p in root.rglob(pattern) if "__pycache__" not in p.parts)
    payload = hashlib.sha256()
    for path in sorted(files):
        payload.update(str(path.relative_to(root)).encode("utf-8"))
        payload.update(b"\0")
        payload.update(path.read_bytes())
        payload.update(b"\0")
    return payload.hexdigest()[:16]


def semantic_hashes_for(canonical: str) -> dict[str, str]:
    """Return all mutable semantic inputs that invalidate operator evidence."""
    # This function is called while operator_policy imports primitive_evidence.
    # Hash sources directly so validation is bootstrap-safe and cannot observe a
    # partially initialized registry.  Including canonical keeps each record
    # independently bound even when a shared policy/signature table changes.
    policy_source = _source_hash(FE_ROOT / "cleaned_operators" / "operator_policy.py")
    signature_source = _source_hash(FE_ROOT / "backend" / "production_signature.py")
    semantic_sources = (
        FE_ROOT / "backend" / "numeric_semantics.py",
        FE_ROOT / "backend" / "cross_section_spec.py",
        FE_ROOT / "backend" / "production_signature.py",
    )
    semantic_contract = {
        str(path.relative_to(FE_ROOT)): _source_hash(path) for path in semantic_sources
    }
    variant_source = _source_hash(
        FE_ROOT / "tests" / "backend_parity" / "evidence_case_registry.py"
    )
    return {
        "operator_policy_hash": compute_payload_hash({"canonical": canonical, "source": policy_source}),
        "parameter_signature_hash": compute_payload_hash({"canonical": canonical, "source": signature_source}),
        "semantic_contract_hash": compute_payload_hash(semantic_contract),
        "planner_rewrite_hash": _source_hash(FE_ROOT / "planner" / "rewrite_fastpath.py"),
        "bridge_parameter_hash": _source_hash(FE_ROOT / "backend" / "cleaned_bridge.py"),
        "test_source_hash": _tree_hash(FE_ROOT / "tests" / "backend_parity", ("*.py",)),
        "golden_data_hash": compute_payload_hash({
            "operator_golden": _tree_hash(FE_ROOT / "tests" / "operator_golden", ("*.py", "*.csv", "*.json", "*.parquet")),
            "fixtures": _tree_hash(FE_ROOT / "tests" / "fixtures" / "golden", ("*.py", "*.csv", "*.json", "*.parquet")),
            "integration_fixture": _tree_hash(FE_ROOT / "tests" / "integration" / "fixtures" / "golden", ("*.csv", "*.json", "*.parquet")),
        }),
        "tolerance_policy_hash": compute_payload_hash({
            "numeric_semantics": _source_hash(FE_ROOT / "backend" / "numeric_semantics.py"),
            "parity_helpers": _source_hash(FE_ROOT / "tests" / "backend_parity" / "duckdb_parity_helpers.py"),
        }),
        "execution_variant_hash": compute_payload_hash({"canonical": canonical, "source": variant_source}),
    }


def parameter_domain_hash_for(canonical: str) -> str:
    """Recompute the exact signature-domain digest stored in evidence."""
    from backend.operator_evidence_schema import compute_implementation_hash
    from backend.production_signature import signature_for

    sig = signature_for(canonical)
    if sig is None:
        return ""
    return compute_implementation_hash(
        json.dumps(
            {
                "canonical": sig.canonical,
                "default_status": sig.default_status,
                "params": [(p.name, p.constraint, p.status) for p in sig.params],
            },
            sort_keys=True,
        )
    )


def implementation_hashes_for(canonical: str) -> dict[str, str]:
    """Hash the registered implementation sources for one canonical operator."""
    from cleaned_operators.registry import OperatorRegistry

    hashes: dict[str, str] = {}
    for backend, key in (("pandas_numpy", "pandas"), ("polars", "polars")):
        implementation = OperatorRegistry.get(canonical, backend)
        if implementation is None:
            continue
        source_file = getattr(implementation.__class__, "__module__", "")
        try:
            module = __import__(source_file, fromlist=["*"])
            path = Path(getattr(module, "__file__", ""))
        except (ImportError, TypeError):
            continue
        digest = _source_hash(path)
        if digest:
            hashes[f"implementation_hash_{key}"] = digest
    # SQL implementations are emitter fragments, not the registry's generic
    # SqlCapableOperator marker.  Binding every certified SQL canonical to the
    # emitter source is bootstrap-safe and invalidates evidence on any fragment
    # change even before SQL markers are registered.
    duckdb_emitter = _source_hash(FE_ROOT / "backend" / "sql_pushdown" / "emitter.py")
    if duckdb_emitter:
        hashes["implementation_hash_duckdb"] = duckdb_emitter
    return hashes


def implementation_sources_for(canonical: str) -> dict[str, str]:
    """Record reviewable relative source paths for bootstrap-safe validation."""
    from cleaned_operators.registry import OperatorRegistry

    sources: dict[str, str] = {}
    for backend, key in (("pandas_numpy", "pandas"), ("polars", "polars")):
        implementation = OperatorRegistry.get(canonical, backend)
        if implementation is None:
            continue
        try:
            module = __import__(implementation.__class__.__module__, fromlist=["*"])
            path = Path(getattr(module, "__file__", "")).resolve()
            sources[f"implementation_source_{key}"] = str(path.relative_to(FE_ROOT.resolve()))
        except (ImportError, TypeError, ValueError):
            continue
    sources["implementation_source_duckdb"] = "backend/sql_pushdown/emitter.py"
    return sources

def enrich_operator_metadata(
    *,
    certified: frozenset[str],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    from backend.operator_evidence_schema import compute_implementation_hash
    from backend.production_signature import signature_for
    from tests.backend_parity.evidence_case_registry import EXECUTION_VARIANTS, operator_evidence_meta

    emitter = emitter_hashes()
    polars_h = emitter.get("implementation_hash_polars_emitter", "")
    duck_h = emitter.get("implementation_hash_duckdb_emitter", "")
    base = operator_evidence_meta(certified=certified)
    merged: dict[str, dict[str, Any]] = {}
    if existing:
        for k, v in existing.items():
            if k in certified and isinstance(v, dict):
                merged[k] = dict(v)
    for name in sorted(certified):
        rec = dict(merged.get(name) or {})
        rec.setdefault("semantic_version", 2)
        rec.update(implementation_hashes_for(name))
        rec.update(implementation_sources_for(name))
        rec.update(semantic_hashes_for(name))
        if polars_h and "implementation_hash_polars" not in rec:
            rec["implementation_hash_polars_emitter"] = polars_h
        if duck_h and "implementation_hash_duckdb" not in rec:
            rec["implementation_hash_duckdb_emitter"] = duck_h
        sig = signature_for(name)
        if sig is not None:
            rec["parameter_domain_hash"] = parameter_domain_hash_for(name)
        variants = EXECUTION_VARIANTS.get(name)
        if variants:
            rec["execution_variants"] = variants
        merged[name] = rec
    return merged
