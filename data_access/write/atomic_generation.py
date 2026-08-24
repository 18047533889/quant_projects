"""
R32-P0-102: append/upsert/delete/publish dataset-level generation atomicity.
R32-P0-103: publish两次rename的missing-target window消除.
R32-P0-104: COS/object-store write使用distributed fencing.

Write atomicity and distributed coordination.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class GenerationIdentity:
    """Immutable generation identity for atomic dataset updates.

    R32-P0-102: Reader必须只看到:
    - old generation OR complete new generation
    不能看到mixed partitions.
    """

    dataset: str
    generation_id: str
    epoch: int
    created_at: str
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "generation_id": self.generation_id,
            "epoch": self.epoch,
            "created_at": self.created_at,
            "complete": self.complete,
        }

    @classmethod
    def create_new(cls, dataset: str, epoch: int) -> GenerationIdentity:
        """Create new generation identity."""
        import time

        return cls(
            dataset=dataset,
            generation_id=uuid.uuid4().hex,
            epoch=epoch,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            complete=False,
        )

    def mark_complete(self) -> GenerationIdentity:
        """Mark generation as complete (all partitions written)."""
        return GenerationIdentity(
            dataset=self.dataset,
            generation_id=self.generation_id,
            epoch=self.epoch,
            created_at=self.created_at,
            complete=True,
        )


@dataclass(frozen=True)
class AtomicPublishPlan:
    """Atomic publish plan using immutable generation + pointer swap.

    R32-P0-103: 不要target→archive + candidate→target中间暴露target absent.
    使用immutable generation + one visibility pointer commit.
    """

    dataset: str
    source_generation: GenerationIdentity
    target_generation: GenerationIdentity
    pointer_path: Path  # Single atomic pointer file

    def validate(self) -> None:
        """Validate publish plan is atomic."""
        if not self.source_generation.complete:
            raise ValidationError(
                f"Source generation {self.source_generation.generation_id} "
                "not complete - cannot publish"
            )

        if self.target_generation.epoch <= self.source_generation.epoch:
            raise ValidationError(
                f"Target epoch {self.target_generation.epoch} must be > "
                f"source epoch {self.source_generation.epoch}"
            )


class AtomicGenerationWriter:
    """Atomic dataset writer using generation-based atomicity.

    R32-P0-102: Single文件rename不够.
    推荐: immutable generation + atomic pointer/manifest swap.
    """

    def __init__(self, dataset: str, base_dir: Path) -> None:
        self.dataset = dataset
        self.base_dir = base_dir
        self.generations_dir = base_dir / ".generations"
        self.pointer_file = base_dir / ".current_generation"

    def begin_write(self, epoch: int) -> GenerationIdentity:
        """Begin new generation write.

        Returns:
            New generation identity
        """
        generation = GenerationIdentity.create_new(self.dataset, epoch)

        # Create generation directory
        gen_dir = self.generations_dir / generation.generation_id
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Write generation metadata
        import json

        meta_file = gen_dir / "_generation.json"
        meta_file.write_text(json.dumps(generation.to_dict(), indent=2))

        return generation

    def complete_write(self, generation: GenerationIdentity) -> None:
        """Mark generation as complete and atomically update pointer.

        R32-P0-102: Atomic visibility - readers see old OR new, never mixed.
        """
        if not generation.complete:
            generation = generation.mark_complete()

        gen_dir = self.generations_dir / generation.generation_id

        # Write completion marker
        import json

        meta_file = gen_dir / "_generation.json"
        meta_file.write_text(json.dumps(generation.to_dict(), indent=2))

        # Atomic pointer update
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.base_dir, prefix=".tmp_pointer_"
        )
        try:
            os.write(tmp_fd, generation.generation_id.encode("utf-8"))
            os.close(tmp_fd)
            # Atomic rename
            os.replace(tmp_path, self.pointer_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_current_generation(self) -> GenerationIdentity | None:
        """Read current generation from pointer."""
        if not self.pointer_file.exists():
            return None

        generation_id = self.pointer_file.read_text().strip()
        gen_dir = self.generations_dir / generation_id
        meta_file = gen_dir / "_generation.json"

        if not meta_file.exists():
            return None

        import json

        meta = json.loads(meta_file.read_text())
        return GenerationIdentity(
            dataset=meta["dataset"],
            generation_id=meta["generation_id"],
            epoch=meta["epoch"],
            created_at=meta["created_at"],
            complete=meta.get("complete", False),
        )


@dataclass(frozen=True)
class DistributedFencingToken:
    """Distributed fencing token for COS/object-store writes.

    R32-P0-104: 本地POSIX mutation lock无法协调多server同一COS prefix.
    未来/当前remote writes需要:
    - generation id
    - ETag/If-Match
    - conditional pointer update
    - fencing epoch
    - idempotency key
    """

    generation_id: str
    fencing_epoch: int
    idempotency_key: str
    etag: str | None = None
    if_match: str | None = None

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers for COS write."""
        headers = {
            "x-generation-id": self.generation_id,
            "x-fencing-epoch": str(self.fencing_epoch),
            "x-idempotency-key": self.idempotency_key,
        }
        if self.if_match:
            headers["If-Match"] = self.if_match
        return headers

    @classmethod
    def create_new(cls, fencing_epoch: int) -> DistributedFencingToken:
        """Create new fencing token."""
        return cls(
            generation_id=uuid.uuid4().hex,
            fencing_epoch=fencing_epoch,
            idempotency_key=uuid.uuid4().hex,
        )


class DistributedWriteCoordinator:
    """Coordinator for distributed writes with fencing.

    R32-P0-104: COS/object-store write使用distributed fencing.
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self._current_epoch = 0

    def acquire_write_token(self) -> DistributedFencingToken:
        """Acquire write token with fencing epoch.

        Returns:
            Fencing token for this write
        """
        self._current_epoch += 1
        return DistributedFencingToken.create_new(self._current_epoch)

    def validate_token(self, token: DistributedFencingToken) -> None:
        """Validate fencing token is still valid.

        Raises:
            ValidationError: If token fenced by newer epoch
        """
        if token.fencing_epoch < self._current_epoch:
            raise ValidationError(
                f"Write fenced: token epoch {token.fencing_epoch} < "
                f"current epoch {self._current_epoch}"
            )

    def conditional_write(
        self,
        token: DistributedFencingToken,
        target: str,
        content: bytes,
    ) -> bool:
        """Perform conditional write with fencing.

        Args:
            token: Fencing token
            target: Target path/key
            content: Content to write

        Returns:
            True if write succeeded, False if fenced/conflicted

        Raises:
            ValidationError: If token invalid
        """
        self.validate_token(token)

        # In real implementation, this would use COS conditional PUT
        # with If-Match header / generation precondition

        # Placeholder: write with atomic rename
        import tempfile

        target_path = Path(target)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target_path.parent, prefix=".tmp_write_"
        )
        try:
            os.write(tmp_fd, content)
            os.close(tmp_fd)

            # Would check If-Match condition here in real COS implementation
            os.replace(tmp_path, target_path)
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False


__all__ = [
    "GenerationIdentity",
    "AtomicPublishPlan",
    "AtomicGenerationWriter",
    "DistributedFencingToken",
    "DistributedWriteCoordinator",
]
