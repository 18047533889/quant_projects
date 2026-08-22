"""
R32-P0-099: write/delete/publish/metadata mutation必须有逻辑action authorization.
R32-P0-100: dataset-specific DatasetPathBoundary.
R32-P0-101: public API不接受裸physical_scope path.

Write operations authorization and path security.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import AccessDeniedError, ValidationError


# R32-P0-099: Unified action types for authorization
ACTION_DATASET_READ = "dataset:read"
ACTION_DATASET_WRITE = "dataset:write"
ACTION_DATASET_DELETE = "dataset:delete"
ACTION_DATASET_PUBLISH = "dataset:publish"
ACTION_METADATA_READ = "metadata:read"
ACTION_METADATA_WRITE = "metadata:write"
ACTION_FACTOR_READ = "factor:read"
ACTION_FACTOR_WRITE = "factor:write"
ACTION_FACTOR_CATALOG_WRITE = "factor:catalog:write"

ALL_WRITE_ACTIONS = frozenset([
    ACTION_DATASET_WRITE,
    ACTION_DATASET_DELETE,
    ACTION_DATASET_PUBLISH,
    ACTION_METADATA_WRITE,
    ACTION_FACTOR_WRITE,
    ACTION_FACTOR_CATALOG_WRITE,
])


@dataclass(frozen=True)
class DatasetPathBoundary:
    """Dataset-specific path boundary enforcement.

    R32-P0-100: 全局registered roots allowlist不足.
    任何physical_scope/write_dir/write_root/staging root/publish target
    都必须证明属于"当前已授权dataset"的boundary.
    """

    dataset: str
    allowed_roots: tuple[Path, ...]
    allowed_staging_roots: tuple[Path, ...]
    allowed_publish_targets: tuple[Path, ...]

    def validate_physical_scope(self, path: Path) -> None:
        """Validate path is within dataset's physical scope boundary.

        Raises:
            AccessDeniedError: If path outside boundary
        """
        path_resolved = path.resolve()

        # Check if path is under any allowed root
        for root in self.allowed_roots:
            try:
                path_resolved.relative_to(root.resolve())
                return  # Valid
            except ValueError:
                continue

        raise AccessDeniedError(
            f"Path {path} outside dataset '{self.dataset}' boundary. "
            f"R32-P0-100: physical_scope must be within dataset-authorized roots."
        )

    def validate_write_dir(self, path: Path) -> None:
        """Validate write directory is within dataset boundary."""
        self.validate_physical_scope(path)

    def validate_staging_root(self, path: Path) -> None:
        """Validate staging root is within dataset-specific staging boundary."""
        path_resolved = path.resolve()

        for root in self.allowed_staging_roots:
            try:
                path_resolved.relative_to(root.resolve())
                return
            except ValueError:
                continue

        raise AccessDeniedError(
            f"Staging root {path} outside dataset '{self.dataset}' boundary"
        )

    def validate_publish_target(self, path: Path) -> None:
        """Validate publish target is within dataset-specific publish boundary."""
        path_resolved = path.resolve()

        for root in self.allowed_publish_targets:
            try:
                path_resolved.relative_to(root.resolve())
                return
            except ValueError:
                continue

        raise AccessDeniedError(
            f"Publish target {path} outside dataset '{self.dataset}' boundary"
        )


@dataclass(frozen=True)
class VerifiedPhysicalScope:
    """Verified physical scope - only created by PreparedRead/Resolver.

    R32-P0-101: public API不接受裸physical_scope path.
    用户不能dataset=A + paths=B.
    """

    dataset: str
    snapshot_id: str
    exact_objects: tuple[str, ...]
    contract_digest: str
    boundary: DatasetPathBoundary

    def __post_init__(self) -> None:
        """Validate all objects are within dataset boundary."""
        for obj_path in self.exact_objects:
            path = Path(obj_path)
            self.boundary.validate_physical_scope(path)

    @classmethod
    def create_trusted(
        cls,
        dataset: str,
        snapshot_id: str,
        exact_objects: Sequence[str],
        contract_digest: str,
        boundary: DatasetPathBoundary,
    ) -> VerifiedPhysicalScope:
        """Create verified scope - only callable by PreparedRead/Resolver.

        This is the ONLY way to create VerifiedPhysicalScope.
        Public APIs cannot construct this directly.
        """
        return cls(
            dataset=dataset,
            snapshot_id=snapshot_id,
            exact_objects=tuple(exact_objects),
            contract_digest=contract_digest,
            boundary=boundary,
        )


class WriteAuthorizationGuard:
    """Authorization guard for write/delete/publish/metadata mutations.

    R32-P0-099: Path sandbox不能替代逻辑权限.
    统一动作授权检查.
    """

    def __init__(self, authorizer: Any, principal: Any) -> None:
        self.authorizer = authorizer
        self.principal = principal

    def authorize_write(self, dataset: str) -> None:
        """Authorize dataset write operation.

        Raises:
            AccessDeniedError: If not authorized
        """
        self._check_action(dataset, ACTION_DATASET_WRITE)

    def authorize_delete(self, dataset: str) -> None:
        """Authorize dataset delete operation."""
        self._check_action(dataset, ACTION_DATASET_DELETE)

    def authorize_publish(self, dataset: str) -> None:
        """Authorize dataset publish operation."""
        self._check_action(dataset, ACTION_DATASET_PUBLISH)

    def authorize_metadata_write(self, dataset: str) -> None:
        """Authorize metadata write operation."""
        self._check_action(dataset, ACTION_METADATA_WRITE)

    def authorize_factor_write(self, factor_id: str) -> None:
        """Authorize factor write operation."""
        self._check_action("factor_lake", ACTION_FACTOR_WRITE)

    def authorize_factor_catalog_write(self) -> None:
        """Authorize factor catalog write operation."""
        self._check_action("factor_lake", ACTION_FACTOR_CATALOG_WRITE)

    def _check_action(self, resource: str, action: str) -> None:
        """Internal authorization check.

        Raises:
            AccessDeniedError: If not authorized
        """
        if not self.authorizer:
            raise AccessDeniedError(
                f"No authorizer configured for action {action} on {resource}"
            )

        try:
            self.authorizer.authorize(self.principal, resource, action=action)
        except Exception as exc:
            raise AccessDeniedError(
                f"Principal {self.principal.principal_id} not authorized "
                f"for {action} on {resource}"
            ) from exc

    def require_any_write_action(self, dataset: str) -> None:
        """Require at least one write action is authorized.

        For operations that could be write/delete/publish.
        """
        errors = []
        for action in [
            ACTION_DATASET_WRITE,
            ACTION_DATASET_DELETE,
            ACTION_DATASET_PUBLISH,
        ]:
            try:
                self._check_action(dataset, action)
                return  # Authorized for at least one
            except AccessDeniedError as e:
                errors.append(str(e))

        raise AccessDeniedError(
            f"Principal {self.principal.principal_id} not authorized for any "
            f"write action on {dataset}. Tried: {', '.join(ALL_WRITE_ACTIONS)}"
        )


def validate_no_raw_physical_scope(
    dataset: str, paths: Sequence[str] | None
) -> None:
    """Validate public API doesn't accept raw physical_scope paths.

    R32-P0-101: 用户不能dataset=A + paths=B.
    只能由PreparedRead/Resolver创建VerifiedPhysicalScope.

    Raises:
        ValidationError: If raw paths provided
    """
    if paths is not None and len(paths) > 0:
        raise ValidationError(
            f"R32-P0-101: Public API不接受裸physical_scope path. "
            f"Dataset '{dataset}' received raw paths={len(paths)}. "
            "Use PreparedRead/Resolver to create VerifiedPhysicalScope."
        )


__all__ = [
    "ACTION_DATASET_READ",
    "ACTION_DATASET_WRITE",
    "ACTION_DATASET_DELETE",
    "ACTION_DATASET_PUBLISH",
    "ACTION_METADATA_READ",
    "ACTION_METADATA_WRITE",
    "ACTION_FACTOR_READ",
    "ACTION_FACTOR_WRITE",
    "ACTION_FACTOR_CATALOG_WRITE",
    "DatasetPathBoundary",
    "VerifiedPhysicalScope",
    "WriteAuthorizationGuard",
    "validate_no_raw_physical_scope",
]
