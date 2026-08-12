# -*- coding: utf-8
"""R32-P0-042 — ExecutionIdentity: code/build facts separation from data facts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExecutionIdentity:
    """Execution identity: code/build/runtime facts (separate from data snapshot).

    R32-P0-042: build SHA belongs here, not in DataSnapshot. Changing code without
    changing data should change ExecutionIdentity but not SourceSnapshot.

    Components:
    - build_sha: git SHA of the build
    - package_version: semantic version
    - build_id: unique build identifier
    - build_time: when this version was built
    - runtime_mode: production/strict/research/interactive
    - semantic_execution_version: versioned semantic rules
    """

    build_sha: str | None = None
    package_version: str | None = None
    build_id: str | None = None
    build_time: datetime | None = None
    runtime_mode: str | None = None
    semantic_execution_version: str | None = None

    def digest(self) -> str:
        """Canonical digest of execution identity."""
        payload = {
            "build_sha": self.build_sha or "",
            "package_version": self.package_version or "",
            "build_id": self.build_id or "",
            "runtime_mode": self.runtime_mode or "",
            "semantic_execution_version": self.semantic_execution_version or "",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_sha": self.build_sha,
            "package_version": self.package_version,
            "build_id": self.build_id,
            "build_time": (
                self.build_time.isoformat()
                if self.build_time
                else None
            ),
            "runtime_mode": self.runtime_mode,
            "semantic_execution_version": self.semantic_execution_version,
            "digest": self.digest(),
        }


def get_current_execution_identity() -> ExecutionIdentity:
    """Get current execution identity from build metadata and runtime config."""
    try:
        from data_access._build_meta import (
            build_sha,
            build_id,
            build_time,
            package_version,
        )

        sha = build_sha()
        ver = package_version()
        bid = build_id()
        btime = build_time()
    except Exception:
        sha = None
        ver = None
        bid = None
        btime = None

    try:
        from data_access.read.query_budget import get_runtime_mode

        mode = get_runtime_mode()
    except Exception:
        mode = None

    # Semantic execution version tracks PIT/compiler/operator semantic changes
    # that invalidate cached results even with same data
    semantic_version = "v1"

    return ExecutionIdentity(
        build_sha=sha,
        package_version=ver,
        build_id=bid,
        build_time=btime,
        runtime_mode=mode,
        semantic_execution_version=semantic_version,
    )


@dataclass(frozen=True)
class ArtifactIdentity:
    """R32-P0-076: Combined identity for result artifacts.

    Artifact identity = data facts + code facts + security facts + plan facts.
    This is what determines cache/result invalidation.
    """

    source_snapshot_digest: str
    execution_digest: str
    security_digest: str | None = None
    plan_digest: str | None = None

    def digest(self) -> str:
        """Full artifact identity digest."""
        payload = {
            "source": self.source_snapshot_digest,
            "execution": self.execution_digest,
            "security": self.security_digest or "",
            "plan": self.plan_digest or "",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_snapshot_digest": self.source_snapshot_digest,
            "execution_digest": self.execution_digest,
            "security_digest": self.security_digest,
            "plan_digest": self.plan_digest,
            "artifact_digest": self.digest(),
        }
