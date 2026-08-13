"""
Registry package — append-only repository and metadata management.
"""

from factor_assets.registry.repository import (
    AssetRepository,
    DuplicateIdentityError,
    AssetNotFoundError,
    RepositoryStats,
)
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
