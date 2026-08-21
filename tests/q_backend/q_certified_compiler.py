"""Shared fixture: build a production-certified q compiler for R21 tests.

A QCompiler with an unvalidated registry refuses production admission for
every operator ("validation_context: missing").  These helpers build a
registry with a current validation context and PASSing evidence artifacts so
the R21-Q-PHYSICAL-REGION-EXECUTOR tests can exercise the real production
compile path without a live q runtime.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.q_backend.q_compiler import QCompiler, get_q_compiler
from backend.q_backend.q_physical_implementation_registry import (
    QEvidenceArtifact,
    QEvidenceValidationContext,
    QPhysicalImplementation,
    QPhysicalImplementationRegistry,
    compute_q_implementation_hash,
)

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / "evidence" / "r2"


def _git_sha() -> str:
    import hashlib
    import os

    # Deterministic synthetic sha matching _GIT_SHA_RE (40 hex chars).
    return hashlib.sha1(os.getcwd().encode()).hexdigest()


def _stage_payload(stage: str, git_sha: str, canonical: str, lowering_id: str) -> dict:
    impl_hash = compute_q_implementation_hash(canonical, lowering_id, lowering_id, "default")
    base = {
        "evidence_type": f"q_{stage}_evidence/v1",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical": canonical,
        "git_sha": git_sha,
        "implementation_hash": impl_hash,
        "parameter_domain_id": "default",
        "q_version": "4.1",
        "pykx_version": "2.6.0",
        "status": "PASS",
    }
    if stage == "compile":
        base["result"] = {"compiled": True, "compiled_cases": 1, "failure_count": 0}
    elif stage == "runtime":
        base["result"] = {"executed": True, "executed_cases": 1, "failure_count": 0}
    elif stage == "parity":
        base["result"] = {
            "reference_backend": "pandas",
            "compared_cases": 1,
            "mismatch_count": 0,
            "max_abs_error": 0.0,
            "tolerance": 1e-9,
        }
    elif stage == "null_semantics":
        base["result"] = {
            "checked_cases": 1,
            "mismatch_count": 0,
            "checks": ["null-mask", "warmup", "all-null-window"],
        }
    return base


def production_certified_compiler() -> QCompiler:
    """Return a QCompiler whose registry is production-certified for every
    lowering in the operator map (compile/runtime/parity/null evidence PASS)."""
    base = QCompiler()
    git_sha = _git_sha()
    ctx = QEvidenceValidationContext(
        artifact_root=EVIDENCE_ROOT,
        q_version="4.1",
        pykx_version="2.6.0",
        current_git_sha_provider=lambda: git_sha,
    )
    registry = QPhysicalImplementationRegistry(
        lowerings={},
        declared_targets=QPhysicalImplementationRegistry._DECLARED_TARGETS,
        validation_context=ctx,
    )
    for canonical, lowering_id in base.executable_lowerings().items():
        artifacts = {}
        for stage in ("compile", "runtime", "parity", "null_semantics"):
            filename = f"_test_certified_{canonical}_{stage}.json"
            payload = _stage_payload(stage, git_sha, canonical, lowering_id)
            path = EVIDENCE_ROOT / filename
            path.write_text(json.dumps(payload))
            artifacts[f"{stage}_evidence"] = QEvidenceArtifact(
                path=filename, sha256=_sha256_bytes(path.read_bytes())
            )
        registry.register(
            QPhysicalImplementation(
                canonical=canonical,
                lowering_id=lowering_id,
                lowering_source=lowering_id,
                parameter_domain_id="default",
                git_sha=git_sha,
                generation_timestamp=datetime.now(timezone.utc).isoformat(),
                implementation_hash=compute_q_implementation_hash(
                    canonical, lowering_id, lowering_id, "default"
                ),
                q_version_range=">=4.0",
                pykx_version_range=">=2.0",
                **artifacts,
            )
        )
    return QCompiler(capability=__import__(
        "backend.q_backend.q_capability", fromlist=["QBackendCapability"]
    ).QBackendCapability(registry))


def _sha256_bytes(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="session")
def production_certified_q_compiler() -> QCompiler:
    return production_certified_compiler()
