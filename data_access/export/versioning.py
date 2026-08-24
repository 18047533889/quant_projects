# -*- coding: utf-8 -*-
"""Schema versioning and migration utilities.

Supports contract schema evolution with backward-compatible migrations
and version registry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "SchemaVersion",
    "VersionRegistry",
    "migrate_contract",
    "get_current_version",
    "register_migration",
]


@dataclass(frozen=True)
class SchemaVersion:
    """Schema version metadata."""

    version: str
    contract_type: str
    schema_hash: str
    created_at: datetime
    description: str = ""
    breaking_changes: tuple[str, ...] = ()
    deprecated_fields: tuple[str, ...] = ()
    added_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.version or not self.contract_type:
            raise ValueError("version and contract_type are required")

    @property
    def is_breaking(self) -> bool:
        """Check if this version introduces breaking changes."""
        return bool(self.breaking_changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "contract_type": self.contract_type,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "breaking_changes": list(self.breaking_changes),
            "deprecated_fields": list(self.deprecated_fields),
            "added_fields": list(self.added_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaVersion:
        return cls(
            version=data["version"],
            contract_type=data["contract_type"],
            schema_hash=data["schema_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            breaking_changes=tuple(data.get("breaking_changes", [])),
            deprecated_fields=tuple(data.get("deprecated_fields", [])),
            added_fields=tuple(data.get("added_fields", [])),
        )


MigrationFunc = Callable[[dict[str, Any]], dict[str, Any]]


class VersionRegistry:
    """Registry for contract schema versions and migrations."""

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, SchemaVersion]] = {}
        self._migrations: dict[str, dict[tuple[str, str], MigrationFunc]] = {}
        self._current_versions: dict[str, str] = {}

    def register_version(self, version: SchemaVersion) -> None:
        """Register a schema version.

        Args:
            version: SchemaVersion to register
        """
        contract_type = version.contract_type

        if contract_type not in self._versions:
            self._versions[contract_type] = {}

        if version.version in self._versions[contract_type]:
            existing = self._versions[contract_type][version.version]
            if existing.schema_hash != version.schema_hash:
                raise ValueError(
                    f"Version {version.version} already registered for {contract_type} "
                    f"with different schema hash"
                )

        self._versions[contract_type][version.version] = version

    def register_migration(
        self,
        contract_type: str,
        from_version: str,
        to_version: str,
        migration_func: MigrationFunc,
    ) -> None:
        """Register a migration function between versions.

        Args:
            contract_type: Contract type name
            from_version: Source version
            to_version: Target version
            migration_func: Migration function (dict -> dict)
        """
        if contract_type not in self._migrations:
            self._migrations[contract_type] = {}

        key = (from_version, to_version)
        self._migrations[contract_type][key] = migration_func

    def set_current_version(self, contract_type: str, version: str) -> None:
        """Set the current version for a contract type.

        Args:
            contract_type: Contract type name
            version: Current version string
        """
        if contract_type not in self._versions:
            raise ValueError(f"No versions registered for {contract_type}")

        if version not in self._versions[contract_type]:
            raise ValueError(f"Version {version} not registered for {contract_type}")

        self._current_versions[contract_type] = version

    def get_current_version(self, contract_type: str) -> str | None:
        """Get current version for a contract type.

        Args:
            contract_type: Contract type name

        Returns:
            Current version string or None
        """
        return self._current_versions.get(contract_type)

    def get_version(self, contract_type: str, version: str) -> SchemaVersion | None:
        """Get schema version metadata.

        Args:
            contract_type: Contract type name
            version: Version string

        Returns:
            SchemaVersion or None
        """
        return self._versions.get(contract_type, {}).get(version)

    def list_versions(self, contract_type: str) -> list[SchemaVersion]:
        """List all versions for a contract type.

        Args:
            contract_type: Contract type name

        Returns:
            List of SchemaVersion objects (sorted by version)
        """
        versions = self._versions.get(contract_type, {}).values()
        return sorted(versions, key=lambda v: v.version)

    def migrate(
        self,
        contract_data: dict[str, Any],
        contract_type: str,
        from_version: str,
        to_version: str | None = None,
    ) -> dict[str, Any]:
        """Migrate contract data between versions.

        Args:
            contract_data: Contract data dictionary
            contract_type: Contract type name
            from_version: Source version
            to_version: Target version (current if None)

        Returns:
            Migrated contract data

        Raises:
            ValueError: If migration path not found
        """
        if to_version is None:
            to_version = self.get_current_version(contract_type)
            if to_version is None:
                raise ValueError(f"No current version set for {contract_type}")

        if from_version == to_version:
            return contract_data

        migration_path = self._find_migration_path(contract_type, from_version, to_version)

        if not migration_path:
            raise ValueError(
                f"No migration path from {from_version} to {to_version} "
                f"for {contract_type}"
            )

        result = contract_data
        for step_from, step_to in migration_path:
            migration_func = self._migrations[contract_type][(step_from, step_to)]
            result = migration_func(result)

        return result

    def _find_migration_path(
        self,
        contract_type: str,
        from_version: str,
        to_version: str,
    ) -> list[tuple[str, str]] | None:
        """Find migration path between versions using BFS.

        Args:
            contract_type: Contract type name
            from_version: Source version
            to_version: Target version

        Returns:
            List of (from, to) version pairs forming migration path, or None
        """
        if contract_type not in self._migrations:
            return None

        migrations = self._migrations[contract_type]
        queue = [(from_version, [])]
        visited = {from_version}

        while queue:
            current, path = queue.pop(0)

            if current == to_version:
                return path

            for (step_from, step_to) in migrations.keys():
                if step_from == current and step_to not in visited:
                    visited.add(step_to)
                    queue.append((step_to, path + [(step_from, step_to)]))

        return None

    def save(self, path: str | Path) -> None:
        """Save registry to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "versions": {
                contract_type: {
                    version: v.to_dict()
                    for version, v in versions.items()
                }
                for contract_type, versions in self._versions.items()
            },
            "current_versions": self._current_versions,
        }

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VersionRegistry:
        """Load registry from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded VersionRegistry
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        registry = cls()

        for contract_type, versions in data.get("versions", {}).items():
            for version_str, version_data in versions.items():
                version = SchemaVersion.from_dict(version_data)
                registry.register_version(version)

        for contract_type, version in data.get("current_versions", {}).items():
            registry.set_current_version(contract_type, version)

        return registry


_GLOBAL_REGISTRY = VersionRegistry()


def get_current_version(contract_type: str) -> str | None:
    """Get current version for a contract type from global registry.

    Args:
        contract_type: Contract type name

    Returns:
        Current version string or None
    """
    return _GLOBAL_REGISTRY.get_current_version(contract_type)


def register_migration(
    contract_type: str,
    from_version: str,
    to_version: str,
) -> Callable[[MigrationFunc], MigrationFunc]:
    """Decorator to register a migration function.

    Args:
        contract_type: Contract type name
        from_version: Source version
        to_version: Target version

    Returns:
        Decorator function

    Example:
        @register_migration("ModelOperatorSpec", "1.0", "2.0")
        def migrate_spec_v1_to_v2(data: dict) -> dict:
            data["new_field"] = "default"
            return data
    """
    def decorator(func: MigrationFunc) -> MigrationFunc:
        _GLOBAL_REGISTRY.register_migration(contract_type, from_version, to_version, func)
        return func

    return decorator


def migrate_contract(
    contract_data: dict[str, Any],
    contract_type: str,
    from_version: str,
    to_version: str | None = None,
) -> dict[str, Any]:
    """Migrate contract data between versions using global registry.

    Args:
        contract_data: Contract data dictionary
        contract_type: Contract type name
        from_version: Source version
        to_version: Target version (current if None)

    Returns:
        Migrated contract data
    """
    return _GLOBAL_REGISTRY.migrate(contract_data, contract_type, from_version, to_version)


def compute_schema_hash(schema_dict: dict[str, Any]) -> str:
    """Compute deterministic hash of schema structure.

    Args:
        schema_dict: Schema dictionary (field names and types)

    Returns:
        SHA256 hash string
    """
    canonical = json.dumps(schema_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Initialize default versions for core contracts
def _initialize_default_versions() -> None:
    """Initialize version registry with core contract versions."""
    core_contracts = [
        ("ModelOperatorSpec", "1.0.0", "Initial model operator specification"),
        ("RichModelTiming", "1.0.0", "Initial timing contract"),
        ("SampleAdequacyContract", "1.0.0", "Initial sample adequacy contract"),
        ("LabelContract", "1.0.0", "Initial label contract"),
        ("DecisionClock", "1.0.0", "Initial decision clock"),
        ("PredictionBatch", "1.0.0", "Initial prediction batch format"),
        ("ParameterSearchPolicy", "1.0.0", "Initial parameter search policy"),
    ]

    for contract_type, version, description in core_contracts:
        schema_version = SchemaVersion(
            version=version,
            contract_type=contract_type,
            schema_hash=compute_schema_hash({"version": version}),
            created_at=datetime.now(timezone.utc),
            description=description,
        )
        _GLOBAL_REGISTRY.register_version(schema_version)
        _GLOBAL_REGISTRY.set_current_version(contract_type, version)


_initialize_default_versions()
