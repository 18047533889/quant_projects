"""Artifact materializer — download + verify + write to a local path (spec §22).

Implements the ``ArtifactMaterializer`` port declared in
:mod:`quant_platform.app.contracts.storage`. Resolves the artifact bytes via an
``ArtifactResolver``, verifies the content_hash (fail closed), and writes the
bytes to ``dest`` atomically (temp + ``os.replace``).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from quant_platform.app.contracts import ArtifactRef

__all__ = ["ArtifactMaterializerImpl"]


class ArtifactMaterializerImpl:
    """Materializes a published artifact to a local path (implements
    ``ArtifactMaterializer``)."""

    def __init__(self, resolver: Any) -> None:
        """``resolver`` must implement ``ArtifactResolver`` (open/verify)."""
        self.resolver = resolver

    def materialize(self, artifact: ArtifactRef, dest: str) -> str:
        """Download + verify + write ``artifact`` to ``dest``; return ``dest``."""
        data = self.resolver.open(artifact)  # verifies content_hash, fails closed
        dest_path = Path(dest).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.parent / f".{dest_path.name}.tmp.{uuid.uuid4().hex}"
        tmp.write_bytes(data)
        os.replace(str(tmp), str(dest_path))
        return str(dest_path)
