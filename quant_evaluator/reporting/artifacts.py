"""
Artifact management for quant_evaluator reporting.

Provides ArtifactStore for saving, loading, listing, and deleting ChartSpec objects
with content-based deduplication.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from quant_evaluator.reporting.chart_spec import ChartSpec


@dataclass
class ArtifactInfo:
    """Metadata for a stored artifact.

    Attributes:
        artifact_id: Unique identifier for the artifact
        content_hash: SHA-256 hash for deduplication
        title: Chart title
        chart_type: Type of chart
        created_at: Creation timestamp
    """
    artifact_id: str
    content_hash: str
    title: str
    chart_type: str
    created_at: str


class ArtifactStore:
    """Manages storage and retrieval of ChartSpec artifacts.

    Provides content-based deduplication using SHA-256 hashes.

    Attributes:
        base_dir: Base directory for artifact storage
    """

    def __init__(self, base_dir: str = "artifacts"):
        """Initialize ArtifactStore.

        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load artifact index from disk."""
        if self._index_path.exists():
            with open(self._index_path, 'r') as f:
                self._index: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self._index: Dict[str, Dict[str, Any]] = {}

    def _save_index(self) -> None:
        """Save artifact index to disk."""
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2, default=str)

    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get file path for an artifact.

        Args:
            artifact_id: Unique identifier

        Returns:
            Path to artifact file
        """
        return self.base_dir / f"{artifact_id}.json"

    def save(self, chart_spec: ChartSpec, artifact_id: Optional[str] = None) -> str:
        """Save a ChartSpec to disk.

        If an artifact with the same content_hash exists, returns the existing
        artifact_id instead of creating a duplicate.

        Args:
            chart_spec: ChartSpec to save
            artifact_id: Optional custom artifact ID. If not provided, uses content_hash.

        Returns:
            artifact_id of the saved or existing artifact
        """
        content_hash = chart_spec.content_hash

        # Check for existing artifact with same content hash
        for existing_id, metadata in self._index.items():
            if metadata.get("content_hash") == content_hash:
                return existing_id

        # Use provided ID or generate from hash
        if artifact_id is None:
            artifact_id = content_hash[:16]

        # Save chart spec to file
        artifact_path = self._get_artifact_path(artifact_id)
        with open(artifact_path, 'w') as f:
            f.write(chart_spec.to_json())

        # Update index
        import datetime
        self._index[artifact_id] = {
            "content_hash": content_hash,
            "title": chart_spec.title,
            "chart_type": chart_spec.chart_type,
            "created_at": datetime.datetime.now().isoformat(),
            "file_path": str(artifact_path)
        }
        self._save_index()

        return artifact_id

    def load(self, artifact_id: str) -> Optional[ChartSpec]:
        """Load a ChartSpec from disk.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            ChartSpec if found, None otherwise
        """
        if artifact_id not in self._index:
            return None

        artifact_path = self._get_artifact_path(artifact_id)
        if not artifact_path.exists():
            return None

        with open(artifact_path, 'r') as f:
            return ChartSpec.from_json(f.read())

    def list_artifacts(self) -> List[ArtifactInfo]:
        """List all stored artifacts.

        Returns:
            List of ArtifactInfo objects
        """
        artifacts = []
        for artifact_id, metadata in self._index.items():
            artifacts.append(ArtifactInfo(
                artifact_id=artifact_id,
                content_hash=metadata["content_hash"],
                title=metadata["title"],
                chart_type=metadata["chart_type"],
                created_at=metadata["created_at"]
            ))
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactInfo]:
        """Get metadata for a specific artifact.

        Args:
            artifact_id: Unique identifier

        Returns:
            ArtifactInfo if found, None otherwise
        """
        if artifact_id not in self._index:
            return None

        metadata = self._index[artifact_id]
        return ArtifactInfo(
            artifact_id=artifact_id,
            content_hash=metadata["content_hash"],
            title=metadata["title"],
            chart_type=metadata["chart_type"],
            created_at=metadata["created_at"]
        )

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from disk.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            True if deleted, False if not found
        """
        if artifact_id not in self._index:
            return False

        artifact_path = self._get_artifact_path(artifact_id)
        if artifact_path.exists():
            artifact_path.unlink()

        del self._index[artifact_id]
        self._save_index()
        return True

    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists.

        Args:
            artifact_id: Unique identifier

        Returns:
            True if artifact exists, False otherwise
        """
        return artifact_id in self._index

    def clear(self) -> int:
        """Delete all artifacts.

        Returns:
            Number of artifacts deleted
        """
        count = len(self._index)
        for artifact_id in list(self._index.keys()):
            artifact_path = self._get_artifact_path(artifact_id)
            if artifact_path.exists():
                artifact_path.unlink()

        self._index.clear()
        self._save_index()
        return count
