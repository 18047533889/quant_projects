# -*- coding: utf-8 -*-
"""Fail-closed production evidence for factor-shaped Pandas operators.

Two certification modes are supported:

``direct``
    The normal mode emitted by ``certify_factor_operator_evidence.py`` after a
    fresh runtime audit.  It binds the explicit operator set to the complete
    execution-semantic source hashes.

``inherited_runtime_audit``
    A narrowly scoped bridge for a previously audited commit.  It is accepted
    only when the certified commit is an ancestor of the checkout, every
    FactorEngine path changed afterwards is explicitly allow-listed, every
    allow-listed source blob has the recorded Git object id, and the current
    production target cardinalities still match the audited registry.  Any
    numerical kernel, operator-surface or policy change outside that list fails
    closed and requires a fresh direct certification.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FE_ROOT.parent
VERIFIED_PATH = FE_ROOT / "evidence" / "factor_operator_verified.json"


def _hashes() -> dict[str, str]:
    from backend.evidence_provenance import _tree_hash, compute_implementation_hash

    audit = FE_ROOT / "scripts" / "audit_all_factor_production.py"
    files = {
        "operator_capability_hash": FE_ROOT / "backend" / "operator_capability.py",
        "pandas_signature_hash": FE_ROOT / "backend" / "pandas_first_signature.py",
        "execution_context_hash": FE_ROOT / "backend" / "context.py",
        "production_policy_hash": FE_ROOT / "runtime" / "production_policy.py",
        "incremental_hash": FE_ROOT / "runtime" / "incremental.py",
        "warmup_hash": FE_ROOT / "runtime" / "warmup_service.py",
        "run_window_hash": FE_ROOT / "runtime" / "run_window.py",
        "source_window_contract_hash": FE_ROOT / "runtime" / "source_window_contract_v2.py",
        "time_window_hash": FE_ROOT / "storage" / "time_window.py",
    }
    hashes = {
        "cleaned_operator_python_tree_hash": _tree_hash(
            FE_ROOT / "cleaned_operators", patterns=("*.py",)
        ),
        "api_python_tree_hash": _tree_hash(FE_ROOT / "api", patterns=("*.py",)),
        "ir_python_tree_hash": _tree_hash(FE_ROOT / "ir", patterns=("*.py",)),
        "planner_python_tree_hash": _tree_hash(
            FE_ROOT / "planner", patterns=("*.py",)
        ),
        "logical_source_python_tree_hash": _tree_hash(
            FE_ROOT / "storage" / "sources", patterns=("*.py",)
        ),
        "audit_source_hash": compute_implementation_hash(
            audit.read_text(encoding="utf-8")
        ),
    }
    for name, path in files.items():
        hashes[name] = compute_implementation_hash(path.read_text(encoding="utf-8"))
    return hashes


def load_factor_operator_evidence() -> dict[str, Any]:
    if not VERIFIED_PATH.is_file():
        return {}
    try:
        payload = json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _production_sets() -> tuple[set[str], set[str]]:
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.production_hardening import factor_production_targets

    all_targets = set(factor_production_targets())
    return all_targets, all_targets.difference(DAILY_CANONICALS)


def _run_git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_blob_sha(relative_path: str) -> str:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return ""
    if not path.is_file():
        return ""
    value = _run_git(
        "hash-object",
        f"--path={relative_path}",
        str(path),
    )
    if value:
        return value
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _inherited_validation_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    certified = str(payload.get("certified_commit_sha") or "")
    if not certified:
        return ["certified_commit_sha missing"]

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", certified, "HEAD"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        errors.append(
            "certified commit is unavailable or is not an ancestor of the checkout"
        )
        return errors

    raw_changed = _run_git(
        "diff", "--name-only", f"{certified}..HEAD", "--", "factor_engine"
    )
    if raw_changed is None:
        return errors + ["cannot enumerate post-certification FactorEngine changes"]
    changed = {line.strip() for line in raw_changed.splitlines() if line.strip()}

    blob_shas = {
        str(path): str(value)
        for path, value in dict(payload.get("allowed_post_certification_blob_shas") or {}).items()
    }
    unhashed = {
        str(path)
        for path in (payload.get("allowed_unhashed_artifact_paths") or [])
    }
    allowed = set(blob_shas) | unhashed
    unexpected = changed.difference(allowed)
    if unexpected:
        errors.append(
            "uncertified FactorEngine changes after runtime audit: "
            + ", ".join(sorted(unexpected))
        )

    for relative, expected in sorted(blob_shas.items()):
        actual = _git_blob_sha(relative)
        if not actual or actual != expected:
            errors.append(
                f"post-certification blob mismatch for {relative}: "
                f"expected={expected!r} actual={actual!r}"
            )

    try:
        from backend.evidence_provenance import evidence_artifact_valid

        if not evidence_artifact_valid():
            errors.append("primitive evidence is invalid")
        all_targets, nonprimitive = _production_sets()
    except Exception as exc:
        return errors + [f"target resolution failed: {type(exc).__name__}: {exc}"]

    expected_all = int(payload.get("audited_factor_target_count") or -1)
    expected_nonprimitive = int(payload.get("audited_nonprimitive_target_count") or -1)
    if len(all_targets) != expected_all:
        errors.append(
            f"factor target count changed: expected={expected_all} actual={len(all_targets)}"
        )
    if len(nonprimitive) != expected_nonprimitive:
        errors.append(
            "non-primitive target count changed: "
            f"expected={expected_nonprimitive} actual={len(nonprimitive)}"
        )
    return errors


def _direct_validation_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if dict(payload.get("hashes") or {}) != _hashes():
        errors.append("execution-semantic source hashes are stale")
    try:
        _, expected = _production_sets()
    except Exception as exc:
        return errors + [f"target resolution failed: {type(exc).__name__}: {exc}"]
    observed = {str(value) for value in (payload.get("operators") or [])}
    if observed != expected:
        errors.append(
            f"operator set mismatch: missing={sorted(expected-observed)!r} "
            f"extra={sorted(observed-expected)!r}"
        )
    return errors


def validation_errors() -> list[str]:
    payload = load_factor_operator_evidence()
    if not payload:
        return ["missing factor_operator_verified.json"]
    errors: list[str] = []
    if payload.get("artifact_kind") != "test_passed":
        errors.append("artifact_kind != test_passed")
    if not payload.get("passed_at"):
        errors.append("passed_at missing")

    mode = str(payload.get("certification_mode") or "direct")
    if mode == "inherited_runtime_audit":
        errors.extend(_inherited_validation_errors(payload))
    elif mode == "direct":
        errors.extend(_direct_validation_errors(payload))
    else:
        errors.append(f"unknown certification_mode: {mode!r}")
    return errors


@lru_cache(maxsize=1)
def factor_operator_evidence_valid() -> bool:
    return not validation_errors()


@lru_cache(maxsize=256)
def pandas_reference_production_safe(canonical: str) -> bool:
    if not factor_operator_evidence_valid():
        return False
    payload = load_factor_operator_evidence()
    mode = str(payload.get("certification_mode") or "direct")
    if mode == "inherited_runtime_audit":
        try:
            _, targets = _production_sets()
        except Exception:
            return False
        return str(canonical) in targets
    return str(canonical) in {
        str(value) for value in (payload.get("operators") or [])
    }


def current_hashes() -> dict[str, str]:
    return _hashes()
