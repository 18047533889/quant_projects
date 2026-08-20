"""
Durable artifact management for quant_evaluator reporting.

Provides ChartArtifactStore for saving, loading, listing, and deleting ChartSpec
objects with content-based deduplication, atomic writes, file locking, fsync,
checksum verification, path traversal protection, and concurrent writer handling.
"""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from quant_evaluator.reporting.chart_spec import ChartSpec


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""


class ChecksumError(Exception):
    """Raised when stored checksum does not match re-computed value."""


class ConcurrentWriteError(Exception):
    """Raised when a concurrent write lock cannot be acquired."""


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """Immutable metadata for a stored artifact.

    Attributes:
        artifact_id: Unique identifier for the artifact
        content_hash: SHA-256 hash for deduplication
        title: Chart title
        chart_type: Type of chart
        created_at: Creation timestamp (ISO-8601)
        file_size: Size of the artifact file in bytes
        checksum: SHA-256 checksum of the stored file content
    """
    artifact_id: str
    content_hash: str
    title: str
    chart_type: str
    created_at: str
    file_size: int = 0
    checksum: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_file_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MiB chunks
            h.update(chunk)
    return h.hexdigest()


def _validate_path(base_dir: Path, target: Path) -> None:
    """Ensure *target* resolves inside *base_dir*.

    Raises PathTraversalError for any attempt to escape the base directory.
    """
    try:
        resolved = target.resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathTraversalError(f"Cannot resolve path {target}: {exc}") from exc
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise PathTraversalError(
            f"Path traversal detected: {target} resolves outside {base_dir}"
        )


def _atomic_write(path: Path, data: bytes, *, fsync: bool = True) -> str:
    """Write *data* to *path* atomically (write-to-temp then rename).

    If *fsync* is True the temp file and the parent directory are synced to
    durable storage before the rename, guaranteeing crash safety.

    Returns the SHA-256 hex digest of the written bytes.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".chart_")
    try:
        os.write(fd, data)
        if fsync:
            os.fsync(fd)
        os.close(fd)
        if fsync:
            # fsync the parent directory so the directory entry is durable
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        # Atomic rename (on POSIX this is atomic if src and dst are same fs)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        os.close(fd) if not os.get_inheritable(fd) else None  # type: ignore[arg-type]
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return hashlib.sha256(data).hexdigest()


@contextmanager
def _file_lock(path: Path, *, timeout: float = 30.0) -> Generator[None, None, None]:
    """Acquire an exclusive advisory file lock based on *path*.

    Creates a ``.lock`` sidecar file and holds an exclusive ``flock`` on it.
    Raises ConcurrentWriteError if the lock cannot be acquired within *timeout*.
    """
    lock_path = path.parent / f"{path.name}.lock"
    lock_path.touch(exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        import time
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ConcurrentWriteError(
                        f"Could not acquire lock on {lock_path} within {timeout}s"
                    )
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# ChartArtifactStore
# ---------------------------------------------------------------------------

class ChartArtifactStore:
    """Durable store for ChartSpec artifacts.

    Features:
    - Content-based deduplication via SHA-256 content hashes
    - Atomic writes (temp-file + fsync + rename)
    - Exclusive file locking for concurrent writer safety
    - Per-file checksum verification on load
    - Path traversal protection

    Attributes:
        base_dir: Base directory for artifact storage
    """

    def __init__(self, base_dir: str = "artifacts", *, fsync: bool = True) -> None:
        """Initialize ChartArtifactStore.

        Args:
            base_dir: Base directory for storing artifacts
            fsync: If True (default), fsync every write for crash safety.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._index_path = self.base_dir / "_index.json"
        self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load artifact index from disk."""
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                self._index: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save artifact index to disk atomically."""
        data = json.dumps(self._index, indent=2, default=str).encode("utf-8")
        _atomic_write(self._index_path, data, fsync=self._fsync)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get file path for an artifact, with traversal protection."""
        candidate = self.base_dir / f"{artifact_id}.json"
        _validate_path(self.base_dir, candidate)
        return candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, chart_spec: ChartSpec, artifact_id: Optional[str] = None) -> str:
        """Save a ChartSpec to disk.

        If an artifact with the same content_hash exists, returns the existing
        artifact_id instead of creating a duplicate.

        Args:
            chart_spec: ChartSpec to save
            artifact_id: Optional custom artifact ID. If not provided, uses
                         the first 16 characters of content_hash.

        Returns:
            artifact_id of the saved or existing artifact
        """
        content_hash = chart_spec.content_hash

        # Fast dedup check against index (read-locked)
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            for existing_id, metadata in self._index.items():
                if metadata.get("content_hash") == content_hash:
                    return existing_id

            # Use provided ID or generate from hash
            if artifact_id is None:
                artifact_id = content_hash[:16]

            artifact_path = self._get_artifact_path(artifact_id)
            with _file_lock(artifact_path, timeout=10.0):
                chart_json = chart_spec.to_json()
                data_bytes = chart_json.encode("utf-8")
                checksum = _atomic_write(artifact_path, data_bytes, fsync=self._fsync)

                now = datetime.now(timezone.utc).isoformat()
                self._index[artifact_id] = {
                    "content_hash": content_hash,
                    "title": chart_spec.title,
                    "chart_type": chart_spec.chart_type,
                    "created_at": now,
                    "file_path": str(artifact_path),
                    "file_size": len(data_bytes),
                    "checksum": checksum,
                }
                self._save_index()

        return artifact_id

    def load(self, artifact_id: str, *, verify_checksum: bool = True) -> Optional[ChartSpec]:
        """Load a ChartSpec from disk.

        Args:
            artifact_id: Unique identifier of the artifact
            verify_checksum: If True (default), verify stored checksum against file

        Returns:
            ChartSpec if found and valid, None otherwise

        Raises:
            ChecksumError: If checksum verification fails
        """
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            if artifact_id not in self._index:
                return None

        metadata = self._index[artifact_id]
        artifact_path = self._get_artifact_path(artifact_id)
        if not artifact_path.exists():
            return None

        if verify_checksum and "checksum" in metadata:
            actual = _compute_file_checksum(artifact_path)
            expected = metadata["checksum"]
            if actual != expected:
                raise ChecksumError(
                    f"Checksum mismatch for {artifact_id}: "
                    f"expected {expected}, got {actual}"
                )

        with open(artifact_path, "r") as f:
            return ChartSpec.from_json(f.read())

    def list_artifacts(self) -> List[ArtifactInfo]:
        """List all stored artifacts.

        Returns:
            List of ArtifactInfo objects
        """
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            artifacts = []
            for artifact_id, metadata in self._index.items():
                artifacts.append(
                    ArtifactInfo(
                        artifact_id=artifact_id,
                        content_hash=metadata["content_hash"],
                        title=metadata["title"],
                        chart_type=metadata["chart_type"],
                        created_at=metadata["created_at"],
                        file_size=metadata.get("file_size", 0),
                        checksum=metadata.get("checksum", ""),
                    )
                )
        return artifacts

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactInfo]:
        """Get metadata for a specific artifact.

        Args:
            artifact_id: Unique identifier

        Returns:
            ArtifactInfo if found, None otherwise
        """
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            if artifact_id not in self._index:
                return None
            metadata = self._index[artifact_id]

        return ArtifactInfo(
            artifact_id=artifact_id,
            content_hash=metadata["content_hash"],
            title=metadata["title"],
            chart_type=metadata["chart_type"],
            created_at=metadata["created_at"],
            file_size=metadata.get("file_size", 0),
            checksum=metadata.get("checksum", ""),
        )

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from disk.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            True if deleted, False if not found
        """
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            if artifact_id not in self._index:
                return False

            artifact_path = self._get_artifact_path(artifact_id)
            if artifact_path.exists():
                artifact_path.unlink()

            # Also remove lock file if present
            lock_path = artifact_path.parent / f"{artifact_path.name}.lock"
            if lock_path.exists():
                lock_path.unlink()

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
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            return artifact_id in self._index

    def clear(self) -> int:
        """Delete all artifacts.

        Returns:
            Number of artifacts deleted
        """
        with _file_lock(self._index_path, timeout=10.0):
            self._load_index()
            count = len(self._index)
            for artifact_id in list(self._index.keys()):
                artifact_path = self._get_artifact_path(artifact_id)
                if artifact_path.exists():
                    artifact_path.unlink()
                lock_path = artifact_path.parent / f"{artifact_path.name}.lock"
                if lock_path.exists():
                    lock_path.unlink()

            self._index.clear()
            self._save_index()
        return count


# ---------------------------------------------------------------------------
# Backward compat alias
# ---------------------------------------------------------------------------

# Keep the old name around so existing imports don't break.
ArtifactStore = ChartArtifactStore
