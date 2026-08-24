"""R41 DataAccess R30 cross-layer contracts.

This additive module gives callers strict contracts without changing the legacy
read/store/runtime path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import (
    AccessDeniedError,
    DataAccessError,
    DeadlineExceeded,
    PITUnavailable,
    ResourceAdmissionError,
    SchemaContractError,
    SourceResolutionError,
    SourceSnapshotUnavailable,
)
from data_access.r30._shared import stable_digest_full
from data_access.r30.versioning import data_access_version_manifest


class R30ContractError(DataAccessError):
    """Base class for strict R30 cross-layer contract failures."""


class CapabilityHandshakeError(R30ContractError):
    """FE and DataAccess disagree on required capabilities or versions."""


class SnapshotCompatibilityError(R30ContractError):
    """Inputs do not belong to one compatible experiment snapshot."""


class SnapshotFidelityError(SourceSnapshotUnavailable, R30ContractError):
    """An authoritative read cannot prove snapshot fidelity."""


class EmptyResultError(R30ContractError):
    """A successful read produced no rows where rows are required."""


class BackendReadError(R30ContractError):
    """The selected backend failed after resolution and admission succeeded."""


TYPED_ERROR_TAXONOMY: Mapping[str, type[DataAccessError]] = {
    "resolution": SourceResolutionError,
    "snapshot": SourceSnapshotUnavailable,
    "permission": AccessDeniedError,
    "deadline": DeadlineExceeded,
    "pit": PITUnavailable,
    "schema": SchemaContractError,
    "dq": R30ContractError,
    "resource": ResourceAdmissionError,
    "backend": BackendReadError,
}


class SnapshotFidelity(str, Enum):
    PROVEN = "proven"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"


class ReadRepresentation(str, Enum):
    ARROW = "arrow"
    FACTOR_BLOCK = "factor_block"


@dataclass(frozen=True)
class SourceColumnFreshness:
    column: str
    dataset: str
    snapshot_id: str
    vintage: str | None = None
    maturity: str | None = None


@dataclass(frozen=True)
class ReadIdentity:
    snapshot_id: str
    build_sha: str
    contract_version: str
    api_version: str
    representation: ReadRepresentation = ReadRepresentation.ARROW
    column_freshness: tuple[SourceColumnFreshness, ...] = ()

    def cache_key(self, *request_parts: Any) -> str:
        """Return a query-cache identity bound to snapshot/build/contract/API."""
        return stable_digest_full(
            "r41-read-identity-v1",
            self.snapshot_id,
            self.build_sha,
            self.contract_version,
            self.api_version,
            self.representation.value,
            tuple(request_parts),
        )

    def source_identity(self) -> dict[str, Any]:
        """FE source identity payload; contract changes invalidate reuse."""
        return {
            "snapshot_id": self.snapshot_id,
            "build_sha": self.build_sha,
            "contract_version": self.contract_version,
            "api_version": self.api_version,
            "representation": self.representation.value,
        }


@dataclass(frozen=True)
class CapabilityManifest:
    package_version: str
    build_sha: str
    api_version: str
    contract_version: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    snapshot_contract_version: str = "experiment-snapshot-v1"

    @classmethod
    def current(cls, capabilities: Sequence[str] = ()) -> "CapabilityManifest":
        version = data_access_version_manifest()
        return cls(
            package_version=str(version.get("package_version") or "unknown"),
            build_sha=str(version.get("build_sha") or "unknown"),
            api_version=str(version.get("api") or "unknown"),
            contract_version=str(version.get("contract_schema") or "unknown"),
            capabilities=frozenset(str(item) for item in capabilities),
        )

    def handshake(
        self,
        *,
        required_capabilities: Sequence[str] = (),
        min_api_version: str | None = None,
        max_api_version_exclusive: str | None = None,
        snapshot_contract_version: str | None = None,
    ) -> None:
        missing = sorted(set(required_capabilities) - self.capabilities)
        failures: list[str] = []
        if missing:
            failures.append(f"missing capabilities: {missing}")
        if min_api_version is not None and self.api_version < min_api_version:
            failures.append(f"api {self.api_version} < {min_api_version}")
        if max_api_version_exclusive is not None and self.api_version >= max_api_version_exclusive:
            failures.append(f"api {self.api_version} >= {max_api_version_exclusive}")
        if (
            snapshot_contract_version is not None
            and self.snapshot_contract_version != snapshot_contract_version
        ):
            failures.append(
                "snapshot contract "
                f"{self.snapshot_contract_version} != {snapshot_contract_version}"
            )
        if failures:
            raise CapabilityHandshakeError("; ".join(failures))


PUBLIC_R30_MODULES = frozenset(
    {
        "adapter", "api_surface", "artifact_meta", "cache_hierarchy",
        "calendar_snapshot", "change_impact", "change_impact_types", "concepts", "contracts",
        "coverage_service", "data_change", "data_quality", "distributed",
        "execution_lease", "experiment_snapshot", "formats_contract", "join_cost",
        "lineage", "metrics", "mining_profile", "multi_asset", "partition_index",
        "policy", "query_trace", "resolution_lease", "session", "specs",
        "training", "training_read_plan", "universe_snapshot", "versioning",
    }
)
INTERNAL_R30_MODULES = frozenset({"_shared"})


def module_inventory(package_dir: str | Path | None = None) -> dict[str, tuple[str, ...]]:
    """Classify every physical R30 module and expose inventory drift."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).parent
    physical = {p.stem for p in root.glob("*.py") if p.stem != "__init__"}
    classified = PUBLIC_R30_MODULES | INTERNAL_R30_MODULES
    return {
        "physical": tuple(sorted(physical)),
        "public": tuple(sorted(PUBLIC_R30_MODULES)),
        "internal": tuple(sorted(INTERNAL_R30_MODULES)),
        "unclassified": tuple(sorted(physical - classified)),
        "missing": tuple(sorted(classified - physical)),
    }


def import_all_public_modules() -> tuple[str, ...]:
    """Import the complete public inventory; import failure is never suppressed."""
    imported = []
    for name in sorted(PUBLIC_R30_MODULES):
        import_module(f"data_access.r30.{name}")
        imported.append(name)
    return tuple(imported)


__all__ = [
    "BackendReadError", "CapabilityHandshakeError", "CapabilityManifest",
    "EmptyResultError", "INTERNAL_R30_MODULES", "PUBLIC_R30_MODULES",
    "R30ContractError", "ReadIdentity", "ReadRepresentation",
    "SnapshotCompatibilityError", "SnapshotFidelity", "SnapshotFidelityError",
    "SourceColumnFreshness", "TYPED_ERROR_TAXONOMY", "import_all_public_modules",
    "module_inventory",
]
