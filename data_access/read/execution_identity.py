# -*- coding: utf-8
"""R32-P0-042 — ExecutionIdentity: code/build facts separation from data facts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from data_access.core.exceptions import ValidationError


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

    def validate(self) -> None:
        """Reject incomplete build facts before forming a cache identity."""
        from data_access.core.build_metadata import BuildInfo

        BuildInfo(
            version=self.package_version,
            build_sha=self.build_sha,
            build_id=self.build_id,
            build_time=self.build_time.isoformat() if self.build_time else None,
            dirty=False,
        ).validate_production_ready()
        if not self.runtime_mode:
            raise ValidationError("execution identity runtime_mode is unavailable")
        if not self.semantic_execution_version:
            raise ValidationError("execution identity semantic_execution_version is unavailable")

    @property
    def available(self) -> bool:
        try:
            self.validate()
        except ValidationError:
            return False
        return True

    def digest(self) -> str:
        """Canonical full-width digest of a complete execution identity."""
        self.validate()
        build_time = self.build_time
        if build_time is not None and build_time.tzinfo is not None:
            build_time = build_time.astimezone(timezone.utc)
        payload = {
            "build_sha": self.build_sha or "",
            "package_version": self.package_version or "",
            "build_id": self.build_id or "",
            "build_time": build_time.isoformat() if build_time else "",
            "runtime_mode": self.runtime_mode or "",
            "semantic_execution_version": self.semantic_execution_version or "",
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    """Get complete code/build facts from the frozen build authority."""
    from data_access.core.build_metadata import load_build_info

    info = load_build_info(production=True)
    try:
        btime = datetime.fromisoformat(info.build_time.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"invalid frozen build_time {info.build_time!r}") from exc

    from data_access.runtime.mode_identity import current_runtime_mode

    mode = current_runtime_mode().value

    # Semantic execution version tracks PIT/compiler/operator semantic changes
    # that invalidate cached results even with same data
    semantic_version = "v1"

    identity = ExecutionIdentity(
        build_sha=info.build_sha,
        package_version=info.version,
        build_id=info.build_id,
        build_time=btime,
        runtime_mode=mode,
        semantic_execution_version=semantic_version,
    )
    identity.validate()
    return identity


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
