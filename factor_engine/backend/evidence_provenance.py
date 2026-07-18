# -*- coding: utf-8
"""Evidence artifact 溯源：绑定 commit、依赖版本、case hash、implementation hash。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
        ("polars", "polars"),
        ("duckdb", "duckdb"),
        ("pyarrow", "pyarrow"),
    ):
        try:
            m = __import__(mod)
            versions[key] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            versions[key] = "missing"
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


def evidence_artifact_valid(*, require_commit_match: bool = False) -> bool:
    """验证 evidence 是否仍绑定当前代码、case registry 与 emitter 实现。"""
    try:
        data = load_verified_artifact()
    except FileNotFoundError:
        return False
    prov = data.get("provenance") or {}
    if prov.get("artifact_kind") != "test_passed":
        return False
    if not prov.get("passed_at") or not prov.get("commit_sha"):
        return False
    if require_commit_match and not _commit_is_ancestor(
        str(prov.get("commit_sha") or ""), current_commit_sha()
    ):
        return False
    try:
        registry = load_case_registry()
    except FileNotFoundError:
        return False
    if prov.get("case_registry_hash") != compute_case_registry_hash(registry):
        return False
    if dict(prov.get("emitter_hashes") or {}) != emitter_hashes():
        return False
    return True


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
        if polars_h:
            rec["implementation_hash_polars"] = polars_h
        if duck_h:
            rec["implementation_hash_duckdb"] = duck_h
        sig = signature_for(name)
        if sig is not None:
            rec["parameter_domain_hash"] = compute_implementation_hash(
                json.dumps(
                    {
                        "canonical": sig.canonical,
                        "default_status": sig.default_status,
                        "params": [(p.name, p.constraint, p.status) for p in sig.params],
                    },
                    sort_keys=True,
                )
            )
        variants = EXECUTION_VARIANTS.get(name)
        if variants:
            rec["execution_variants"] = variants
        merged[name] = rec
    return merged
