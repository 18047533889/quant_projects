"""Platform storage package — artifact semantics over the DataAccess ObjectStore.

The platform does NOT own a second ObjectStore authority. ``data_access`` owns the
mature ObjectStore; this package owns *artifact semantics* only:

- :mod:`quant_platform.app.adapters.data_access_storage` — the ONE place the
  platform talks to ``data_access`` (ArtifactStorageAdapter / publisher /
  resolver / local cache).
- :mod:`quant_platform.app.storage.access_guard` — artifact security
  classification + formula-endpoint authorization + audit (spec §23).
- :mod:`quant_platform.app.storage.materializer` — download + verify + write to a
  local path (spec §22).
- :mod:`quant_platform.app.storage.registry` — artifact registry keyed by
  ``content_hash`` (memory + sqlite).
- :mod:`quant_platform.app.storage.cache` — on-disk LRU
  ``LocalArtifactCache`` keyed by ``content_hash``.
- :mod:`quant_platform.app.storage.cos_adapter` — minimal COS
  ``ArtifactPublisher`` deferring object semantics to ``data_access``.
"""

from __future__ import annotations

from .access_guard import (
    FORMULA_MIN_CLASSIFICATION,
    FORMULA_PERMISSION,
    ClassificationAccessGate,
    FormulaAccessGuardImpl,
)
from .materializer import ArtifactMaterializerImpl
from .registry import ArtifactRegistry, UnknownArtifactTypeError

__all__ = [
    "ClassificationAccessGate",
    "FormulaAccessGuardImpl",
    "FORMULA_PERMISSION",
    "FORMULA_MIN_CLASSIFICATION",
    "ArtifactMaterializerImpl",
    "ArtifactRegistry",
    "UnknownArtifactTypeError",
]
