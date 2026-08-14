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

# review #253: the factor evidence artifact must bind the FULL execution TCB,
# not just operator sources.  Editing any component below invalidates previously
# certified factor evidence (fail-closed) until a fresh runtime audit re-binds.
# ``ExecutionTCB`` is the spec tuple; ``_EXECUTION_TCB_SOURCES`` resolves each
# component to its source file(s) / tree(s).
ExecutionTCB = (
    "operator_implementation",
    "logical_contract",
    "field_catalog",
    "ir_analyzer",
    "planner",
    "backend_bridge",
    "backend_router",
    "panel_conversion",
    "source_alignment",
    "calendar_session_semantics",
)

# Value is one of:
#   Path                                  -> single source file
#   (Path, (pattern, ...))                -> tracked tree hash under Path
#   (Path, Path, ...)                     -> multiple single source files
_EXECUTION_TCB_SOURCES: dict[str, object] = {
    "operator_implementation": (FE_ROOT / "cleaned_operators", ("*.py",)),
    "logical_contract": (
        FE_ROOT / "cleaned_operators" / "edge_requirements.py",
        FE_ROOT / "cleaned_operators" / "operator_policy.py",
    ),
    "field_catalog": (FE_ROOT / "fields", ("*.py",)),
    "ir_analyzer": (FE_ROOT / "ir", ("*.py",)),
    "planner": (FE_ROOT / "planner", ("*.py",)),
    "backend_bridge": FE_ROOT / "backend" / "cleaned_bridge.py",
    "backend_router": FE_ROOT / "backend" / "backend_router.py",
    "panel_conversion": FE_ROOT / "backend" / "panel_polars.py",
    "source_alignment": (FE_ROOT / "storage" / "sources", ("*.py",)),
    "calendar_session_semantics": (
        FE_ROOT / "runtime" / "session_calendar.py",
        FE_ROOT / "storage" / "trading_calendar.py",
        FE_ROOT / "market" / "session.py",
    ),
}


def execution_tcb_hash() -> str:
    """Composite hash over every ExecutionTCB component source.

    Uses the same Git-clean-tree digest machinery as the rest of the evidence
    system so an unstaged edit to the bridge / router / panel-conversion / any
    TCB source invalidates previously certified factor evidence.
    """
    from backend.evidence_provenance import (
        _source_hash,
        _tree_hash,
        compute_implementation_hash,
        compute_payload_hash,
    )

    parts: dict[str, str] = {}
    for name in ExecutionTCB:
        spec = _EXECUTION_TCB_SOURCES.get(name)
        if spec is None:
            parts[name] = ""
            continue
        if isinstance(spec, Path):
            parts[name] = _source_hash(spec)
        elif (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[1], tuple)
        ):
            root, patterns = spec
            parts[name] = _tree_hash(root, patterns)
        elif isinstance(spec, tuple):
            parts[name] = compute_payload_hash(
                {
                    str(p.relative_to(FE_ROOT)): _source_hash(p)
                    for p in spec
                    if isinstance(p, Path)
                }
            )
        else:
            parts[name] = ""
    return compute_payload_hash(parts)


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
        # P0-21: stateful runtime / contract and session / calendar logic are
        # execution-semantic inputs — a change must invalidate the artifact.
        "stateful_runtime_hash": FE_ROOT / "stateful_runtime.py",
        "stateful_contract_hash": FE_ROOT / "stateful_contract.py",
        "session_calendar_hash": FE_ROOT / "runtime" / "session_calendar.py",
        "trading_calendar_hash": FE_ROOT / "storage" / "trading_calendar.py",
        "market_session_hash": FE_ROOT / "market" / "session.py",
        "read_session_hash": FE_ROOT / "storage" / "read_session.py",
        "composite_lowering_hash": FE_ROOT / "planner" / "composite_lowering.py",
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
        # P0-21: composite lowerings and the session/calendar modules are hash-
        # covered explicitly so a change to any lowering / calendar file
        # invalidates the execution-semantic evidence.
        "composite_lowerings_python_tree_hash": _tree_hash(
            FE_ROOT / "planner" / "lowerings", patterns=("*.py",)
        ),
        "logical_source_python_tree_hash": _tree_hash(
            FE_ROOT / "storage" / "sources", patterns=("*.py",)
        ),
        "audit_source_hash": compute_implementation_hash(
            audit.read_text(encoding="utf-8")
        ),
        # review #253: full execution TCB composite — editing the backend bridge
        # (cleaned_bridge), backend router, panel conversion, field catalog,
        # IR/analyzer, planner, source alignment or calendar/session semantics
        # invalidates previously certified factor evidence.
        "execution_tcb_hash": execution_tcb_hash(),
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
    import cleaned_operators
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry

    if (
        OperatorRegistry.lifecycle() == "building"
        and not getattr(cleaned_operators, "_LOADED", False)
    ):
        cleaned_operators.load_all()
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
    # SHA-1 used to mimic git blob hash format, not for cryptographic security
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data, usedforsecurity=False).hexdigest()


def _git_available() -> bool:
    """R39 #75：当前部署是否还有 live .git（wheel/container 没有）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _inherited_validation_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    certified = str(payload.get("certified_commit_sha") or "")
    if not certified:
        return ["certified_commit_sha missing"]

    if not _git_available():
        # R39 #75：无 .git（wheel/container 部署）→ 读构建期 SCMManifest，取代
        # ``git merge-base`` / ``git diff``。manifest 缺失 → fail-closed（明确报错，
        # 绝不假装通过）。
        from evidence.scm_manifest import SCMManifest

        manifest = SCMManifest.load()
        if manifest is None:
            return errors + [
                "SCMManifest missing and .git unavailable: cannot verify inherited "
                "evidence in this deployment (run scripts/generate_scm_manifest.py "
                "at build time)"
            ]
        if manifest.build_commit_sha and manifest.build_commit_sha != certified:
            # 部署树来自与被审提交不同的提交 → 必须有构建期记录的祖先证明 + 变更清单。
            if not manifest.certified_is_ancestor:
                errors.append(
                    "certified commit is not an ancestor of the build commit "
                    "(per SCMManifest)"
                )
                return errors
            changed = set(manifest.changed_since_certified)
        else:
            # build commit == certified → 无 post-certification 变更。
            changed = set()
    else:
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
    # FE-P0-024: production hard verification must reject any unhashed artifacts
    if unhashed:
        errors.append(
            "production evidence cannot reference unhashed artifacts: "
            + ", ".join(sorted(unhashed))
        )
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

    # FE-P0-038: reject "unknown" or missing hashes in implementation/emitter/artifact
    for field in ("implementation_hash", "emitter_hash", "artifact_hash"):
        for key, value in dict(payload.get("allowed_post_certification_blob_shas") or {}).items():
            if not value or value == "unknown":
                errors.append(
                    f"production evidence cannot contain unknown/missing hash for {key}: {field}"
                )
    # Check emitter hashes explicitly
    emitter_hashes = dict(payload.get("emitter_hashes") or {})
    for emitter_name, emitter_hash in emitter_hashes.items():
        if not emitter_hash or emitter_hash == "unknown":
            errors.append(
                f"production evidence cannot contain unknown/missing emitter hash: {emitter_name}"
            )

    try:
        from backend.evidence_provenance import (
            evidence_artifact_valid,
            implementation_hashes_for,
        )

        if not evidence_artifact_valid():
            errors.append("primitive evidence is invalid")
        all_targets, nonprimitive = _production_sets()
    except Exception as exc:
        return errors + [f"target resolution failed: {type(exc).__name__}: {exc}"]

    # FE-P0-023: reject legacy count-only validation; require exact canonical set and hashes
    expected_canonicals = [
        str(c) for c in (payload.get("audited_factor_canonicals") or [])
    ]
    if not expected_canonicals:
        # Legacy payload without exact canonical list: reject in production
        errors.append(
            "production evidence requires exact canonical set; "
            "legacy count-only validation is not sufficient for hard verification"
        )
    else:
        if sorted(all_targets) != sorted(expected_canonicals):
            errors.append(
                "factor target set changed: "
                f"expected={sorted(expected_canonicals)!r} "
                f"actual={sorted(all_targets)!r}"
            )
    expected_hashes = dict(
        payload.get("audited_factor_implementation_hashes") or {}
    )
    if expected_hashes:
        actual_hashes = {
            canon: implementation_hashes_for(canon) for canon in sorted(all_targets)
        }
        if actual_hashes != expected_hashes:
            errors.append(
                "factor target implementation hashes changed: "
                f"expected={expected_hashes!r} actual={actual_hashes!r}"
            )
    else:
        # No implementation hashes: also reject in production
        errors.append(
            "production evidence requires per-canonical implementation hashes"
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
