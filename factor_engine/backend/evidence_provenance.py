# -*- coding: utf-8
"""Evidence artifact 溯源：绑定 commit、依赖版本、case 与完整运行时实现。"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

FE_ROOT = Path(__file__).resolve().parents[1]
CASE_REGISTRY_JSON = FE_ROOT / "evidence" / "primitive_case_registry.json"
VERIFIED_JSON = FE_ROOT / "evidence" / "primitive_verified.json"


def current_commit_sha() -> str:
    override = os.environ.get("FACTOR_ENGINE_EVIDENCE_COMMIT_SHA", "").strip()
    if override:
        return override
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
    Reachability plus exact case-registry and implementation hashes gives
    stable, fail-closed provenance without that self-referential SHA cycle.
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
        "python": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
    }
    for mod, key in (
        ("polars", "polars"),
        ("duckdb", "duckdb"),
        ("pyarrow", "pyarrow"),
    ):
        try:
            module = __import__(mod)
            versions[key] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[key] = "missing"
    return versions


def compute_payload_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
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
    subset = {key: sorted(registry.get(key) or []) for key in keys}
    return compute_payload_hash(subset)


def compute_implementation_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _runtime_source_paths() -> list[Path]:
    """Return every source file that can change primitive runtime semantics.

    Earlier evidence only hashed the two emitters.  Final registry overrides,
    checkpoint implementations or routing guards could therefore change while
    the old artifact remained valid.  Hash the complete active operator tree
    and the backend/planner modules that select or compile those operators.
    """
    paths: set[Path] = set()
    cleaned = FE_ROOT / "cleaned_operators"
    if cleaned.is_dir():
        paths.update(cleaned.rglob("*.py"))
    sql_pushdown = FE_ROOT / "backend" / "sql_pushdown"
    if sql_pushdown.is_dir():
        paths.update(sql_pushdown.rglob("*.py"))
    for relative in (
        "backend/polars_expr_emitter.py",
        "backend/operator_capability.py",
        "backend/sql_tiers.py",
        "backend/runtime_hardening.py",
        "backend/primitive_evidence.py",
        "backend/production_signature.py",
        "backend/evidence_provenance.py",
        "stateful_runtime.py",
        "planner/composite_lowering.py",
        "planner/logical_plan.py",
        "runtime/planner_runtime.py",
    ):
        path = FE_ROOT / relative
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(FE_ROOT).as_posix())


@lru_cache(maxsize=1)
def _runtime_tree_hash() -> str:
    payload: dict[str, str] = {}
    for path in _runtime_source_paths():
        relative = path.relative_to(FE_ROOT).as_posix()
        payload[relative] = compute_implementation_hash(
            path.read_text(encoding="utf-8")
        )
    return compute_payload_hash(payload)


@lru_cache(maxsize=1)
def emitter_hashes() -> dict[str, str]:
    """Compatibility name returning all evidence-bound implementation hashes."""
    polars_path = FE_ROOT / "backend" / "polars_expr_emitter.py"
    duck_path = FE_ROOT / "backend" / "sql_pushdown" / "emitter.py"
    out: dict[str, str] = {
        "implementation_hash_operator_runtime": _runtime_tree_hash(),
    }
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
        raise FileNotFoundError(
            f"missing case registry: {CASE_REGISTRY_JSON}"
        )
    return json.loads(CASE_REGISTRY_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _certification_candidate() -> dict[str, Any] | None:
    """Load an explicit, short-lived certification candidate context.

    This context exists only to break the evidence-certification bootstrap
    cycle: the parity tests must route candidate operators before a fresh
    artifact can be written.  It is fail-closed, process-local and requires a
    nonce-matched owner-only temporary file with current registry/runtime
    hashes.  Normal production processes do not set these variables.
    """
    raw_path = os.environ.get("FACTOR_ENGINE_CERTIFICATION_CANDIDATE", "").strip()
    nonce = os.environ.get("FACTOR_ENGINE_CERTIFICATION_NONCE", "").strip()
    if not raw_path or not nonce:
        return None
    path = Path(raw_path)
    try:
        stat = path.stat()
        if not path.is_file():
            return None
        if hasattr(os, "getuid") and stat.st_uid != os.getuid():
            return None
        if stat.st_mode & 0o077:
            return None
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if wrapper.get("artifact_kind") != "certification_candidate":
        return None
    if not hmac.compare_digest(str(wrapper.get("nonce") or ""), nonce):
        return None
    try:
        if float(wrapper.get("expires_at", 0.0)) <= time.time():
            return None
    except (TypeError, ValueError):
        return None
    try:
        registry = load_case_registry()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if wrapper.get("case_registry_hash") != compute_case_registry_hash(registry):
        return None
    if dict(wrapper.get("implementation_hashes") or {}) != emitter_hashes():
        return None
    artifact = wrapper.get("artifact")
    return artifact if isinstance(artifact, dict) else None


def certification_candidate_active() -> bool:
    return _certification_candidate() is not None


def load_verified_artifact() -> dict[str, Any]:
    candidate = _certification_candidate()
    if candidate is not None:
        return candidate
    if not VERIFIED_JSON.is_file():
        raise FileNotFoundError(
            f"missing verified artifact: {VERIFIED_JSON}"
        )
    return json.loads(VERIFIED_JSON.read_text(encoding="utf-8"))


def evidence_artifact_valid(*, require_commit_match: bool = False) -> bool:
    """验证 evidence 是否绑定当前代码、case registry 与完整实现树。"""
    candidate = _certification_candidate()
    if candidate is not None:
        return not require_commit_match
    try:
        data = load_verified_artifact()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    provenance = data.get("provenance") or {}
    if provenance.get("artifact_kind") != "test_passed":
        return False
    if not provenance.get("passed_at") or not provenance.get("commit_sha"):
        return False
    if require_commit_match and not _commit_is_ancestor(
        str(provenance.get("commit_sha") or ""), current_commit_sha()
    ):
        return False
    try:
        registry = load_case_registry()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if provenance.get("case_registry_hash") != compute_case_registry_hash(
        registry
    ):
        return False
    if dict(provenance.get("emitter_hashes") or {}) != emitter_hashes():
        return False
    return True


def enrich_operator_metadata(
    *,
    certified: frozenset[str],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    from backend.operator_evidence_schema import compute_implementation_hash
    from backend.production_signature import signature_for
    from tests.backend_parity.evidence_case_registry import (
        EXECUTION_VARIANTS,
        operator_evidence_meta,
    )

    implementation = emitter_hashes()
    polars_hash = implementation.get(
        "implementation_hash_polars_emitter", ""
    )
    duckdb_hash = implementation.get(
        "implementation_hash_duckdb_emitter", ""
    )
    runtime_hash = implementation.get(
        "implementation_hash_operator_runtime", ""
    )
    base = operator_evidence_meta(certified=certified)
    merged: dict[str, dict[str, Any]] = {}
    if existing:
        for key, value in existing.items():
            if key in certified and isinstance(value, dict):
                merged[key] = dict(value)
    for name in sorted(certified):
        record = dict(merged.get(name) or base.get(name) or {})
        record.setdefault("semantic_version", 2)
        if polars_hash:
            record["implementation_hash_polars"] = polars_hash
        if duckdb_hash:
            record["implementation_hash_duckdb"] = duckdb_hash
        if runtime_hash:
            record["implementation_hash_operator_runtime"] = runtime_hash
        signature = signature_for(name)
        if signature is not None:
            record["parameter_domain_hash"] = compute_implementation_hash(
                json.dumps(
                    {
                        "canonical": signature.canonical,
                        "default_status": signature.default_status,
                        "params": [
                            (param.name, param.constraint, param.status)
                            for param in signature.params
                        ],
                    },
                    sort_keys=True,
                )
            )
        variants = EXECUTION_VARIANTS.get(name)
        if variants:
            record["execution_variants"] = variants
        merged[name] = record
    return merged
