"""
Registry package — append-only repository and metadata management.
"""

from factor_assets.registry.repository import (
    AssetRepository,
    DuplicateIdentityError,
    AssetNotFoundError,
    RepositoryStats,
    LifecycleRepository,
    CommittedTransition,
)
from factor_assets.registry.factory import create_repository
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository
from factor_assets.registry.lifecycle import (
    LifecycleOrchestrator,
    TransitionRequest,
    TransitionResult,
)
from factor_assets.registry.lineage import (
    LineageGraph,
    LineageEdge,
    CampaignMetadata,
)
from factor_assets.registry.snapshots import (
    SnapshotManager,
    SnapshotQuery,
    SnapshotResult,
)

__all__ = [
    "AssetRepository",
    "DuplicateIdentityError",
    "AssetNotFoundError",
    "RepositoryStats",
    "LifecycleRepository",
    "CommittedTransition",
    "create_repository",
    "SQLiteLifecycleRepository",
    "LifecycleOrchestrator",
    "TransitionRequest",
    "TransitionResult",
    "LineageGraph",
    "LineageEdge",
    "CampaignMetadata",
    "SnapshotManager",
    "SnapshotQuery",
    "SnapshotResult",
]
