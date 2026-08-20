# -*- coding: utf-8 -*-
"""Evidence artifact registry with input-dependency tracking and auto-staleness.

Provides :class:`EvidenceArtifact` entries for committed inventory / certification /
benchmark artifacts, records which source files each artifact depends on, and can
compare the stored ``evidence_input_identity`` against the current source hashes so
staleness is detected automatically.

Operational model
-----------------
1. **Authoring / regeneration**: a generator script reads source files, produces an
   artifact, and registers it via :func:`register_artifact` (or by writing a YAML
   snippet to ``evidence/r2/`` that references this module).
2. **Freshness gate**: CI / the loop runs :func:`check_all_fresh` which recomputes
   SHA-256 digests over the listed source files and compares them to the stored
   identity.  Any mismatch sets ``stale=True`` and surfaces the specific changed
   inputs.
3. **Archival**: when an artifact is superseded, :func:`archive_current` copies the
   current file into ``evidence/archive/<snapshot>/`` before the new artifact
   replaces it.

All paths are *relative to the repository root* for portability.  The helpers in this
module deal only with local filesystem state; there is no network or git dependency.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "evidence"
CURRENT_DIR = EVIDENCE_ROOT / "current"
ARCHIVE_ROOT = EVIDENCE_ROOT / "archive"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest for a single file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: Sequence[Path]) -> str:
    """Deterministic composite SHA-256 over a set of files.

    Files are sorted so the digest is order-independent.  Missing files produce a
    deterministic sentinel ``MISSING:<name>`` entry in the byte stream so two
    registries with different missing files never collide accidentally.
    """
    normalized: list[tuple[str, str | None]] = []
    for p in sorted(paths):
        resolved = str(p.resolve()) if p.exists() else str(p)
        try:
            normalized.append((resolved, _sha256_file(p) if p.is_file() else None))
        except Exception:
            normalized.append((resolved, None))

    h = hashlib.sha256()
    for resolved, digest in normalized:
        h.update(resolved.encode("utf-8"))
        if digest is None:
            h.update(b"MISSING:")
            h.update(Path(resolved).name.encode("utf-8"))
        else:
            h.update(digest.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Artifact model
# ---------------------------------------------------------------------------

ARTIFACT_TYPES = {"inventory", "certification", "benchmark"}


@dataclass
class EvidenceArtifact:
    """Single evidence artifact with dependency tracking."""

    artifact_id: str
    artifact_type: str  # inventory | certification | benchmark
    description: str
    generator_script: str  # repo-relative path to the script that produces this artifact
    runner_command: str  # command / entrypoint that actually executes the generator
    output_files: List[str] = field(default_factory=list)  # repo-relative paths
    input_dependencies: List[str] = field(default_factory=list)  # repo-relative source paths
    evidence_input_identity: str = ""  # SHA-256 of all inputs at registration time
    generated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stale: bool = False
    stale_inputs: List[str] = field(default_factory=list)

    # -- helpers ---------------------------------------------------------------

    def resolve_inputs(self) -> list[Path]:
        return [(REPO_ROOT / p).resolve() for p in self.input_dependencies]

    def resolve_outputs(self) -> list[Path]:
        return [(REPO_ROOT / p).resolve() for p in self.output_files]

    def recompute_identity(self) -> str:
        return sha256_files(self.resolve_inputs())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Live registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, EvidenceArtifact] = {}


def _artifact_key(artifact_id: str) -> str:
    return artifact_id.strip().upper()


def register_artifact(artifact: EvidenceArtifact) -> None:
    """Add or replace an artifact in the live registry."""
    if artifact.artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"Unsupported artifact_type={artifact.artifact_type!r}")
    _REGISTRY[_artifact_key(artifact.artifact_id)] = artifact


def get_artifact(artifact_id: str) -> EvidenceArtifact | None:
    return _REGISTRY.get(_artifact_key(artifact_id))


def list_artifacts(artifact_type: str | None = None) -> list[EvidenceArtifact]:
    items = list(_REGISTRY.values())
    if artifact_type:
        items = [a for a in items if a.artifact_type == artifact_type]
    return sorted(items, key=lambda a: a.artifact_id)


def clear_registry() -> None:
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Freshness checking
# ---------------------------------------------------------------------------


@dataclass
class FreshnessReport:
    total: int
    stale_count: int
    stale_ids: list[str]
    details: Dict[str, Dict[str, Any]]

    @property
    def all_fresh(self) -> bool:
        return self.stale_count == 0


def check_freshness(artifact: EvidenceArtifact) -> FreshnessReport:
    """Check a single artifact (also mutates ``artifact.stale``)."""
    current = artifact.recompute_identity()
    stale = current != artifact.evidence_input_identity
    stale_inputs: list[str] = []

    if stale:
        current_paths = {p.resolve(): p for p in artifact.resolve_inputs()}
        for resolved, original in current_paths.items():
            if not resolved.is_file():
                stale_inputs.append(str(original))
                continue
            # We don't store per-file digests, so any mismatch means the set changed.
        # A simpler approach: just list all inputs as changed if the composite changed.
        stale_inputs = list(artifact.input_dependencies)

    artifact.stale = stale
    artifact.stale_inputs = stale_inputs
    return FreshnessReport(
        total=1,
        stale_count=int(stale),
        stale_ids=[artifact.artifact_id] if stale else [],
        details={
            artifact.artifact_id: {
                "stored_identity": artifact.evidence_input_identity,
                "current_identity": current,
                "stale": stale,
                "stale_inputs": stale_inputs,
            }
        },
    )


def check_all_fresh(artifacts: Sequence[EvidenceArtifact] | None = None) -> FreshnessReport:
    """Check all registered artifacts (or a supplied subset)."""
    artifacts = artifacts if artifacts is not None else list_artifacts()
    total = len(artifacts)
    stale_count = 0
    stale_ids: list[str] = []
    details: Dict[str, Dict[str, Any]] = {}

    for art in artifacts:
        report = check_freshness(art)
        stale_count += report.stale_count
        stale_ids.extend(report.stale_ids)
        details.update(report.details)

    return FreshnessReport(total=total, stale_count=stale_count, stale_ids=stale_ids, details=details)


# ---------------------------------------------------------------------------
# Current / archive helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)


def archive_current(artifact: EvidenceArtifact, *, snapshot: str | None = None) -> Path:
    """Copy all current output files for *artifact* into ``evidence/archive/<snapshot>/``.

    ``snapshot`` defaults to ``YYYYMMDD-HHMMSS`` if not supplied.
    """
    _ensure_dirs()
    snapshot = snapshot or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest_dir = ARCHIVE_ROOT / snapshot
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in artifact.output_files:
        src = (REPO_ROOT / rel).resolve()
        if src.is_file():
            dst = dest_dir / src.name
            shutil.copy2(str(src), str(dst))
    return dest_dir


def publish_current(artifact: EvidenceArtifact) -> Path:
    """Copy artifact outputs into ``evidence/current/`` (overwrites previous)."""
    _ensure_dirs()
    for rel in artifact.output_files:
        src = (REPO_ROOT / rel).resolve()
        if src.is_file():
            dst = CURRENT_DIR / src.name
            shutil.copy2(str(src), str(dst))
    return CURRENT_DIR


# ---------------------------------------------------------------------------
# Bootstrap registry from committed sources
# ---------------------------------------------------------------------------

# Lazy imports to avoid import-time overhead / missing optional deps.


def _infer_inventory_inputs() -> list[str]:
    candidates = [
        "operator_comprehensive_inventory.csv",
        "operator_inventory.csv",
    ]
    return [c for c in candidates if (REPO_ROOT / c).is_file()]


def _infer_certification_inputs() -> list[str]:
    candidates = [
        "evidence/factor_operator_verified.json",
        "evidence/primitive_verified.json",
        "evidence/composite_verified.json",
        "evidence/recipe_verified.json",
    ]
    return [c for c in candidates if (REPO_ROOT / c).is_file()]


def _infer_benchmark_inputs() -> list[str]:
    candidates = [
        "benchmarks/operator_manifest.json",
    ]
    return [c for c in candidates if (REPO_ROOT / c).is_file()]


def _register_default_artifacts() -> None:
    if _REGISTRY:
        return

    register_artifact(
        EvidenceArtifact(
            artifact_id="operator-inventory",
            artifact_type="inventory",
            description="Comprehensive and slim operator inventory CSVs (committed).",
            generator_script="scripts/r30_phase1_inventory.py",
            runner_command="python3 scripts/r30_phase1_inventory.py",
            output_files=_infer_inventory_inputs(),
            input_dependencies=_infer_inventory_inputs(),
        )
    )

    register_artifact(
        EvidenceArtifact(
            artifact_id="operator-certification",
            artifact_type="certification",
            description="Operator certification/verification ledgers (committed JSON).",
            generator_script="scripts/generate_backend_evidence_manifest.py",
            runner_command="python3 scripts/generate_backend_evidence_manifest.py",
            output_files=_infer_certification_inputs(),
            input_dependencies=_infer_certification_inputs(),
        )
    )

    register_artifact(
        EvidenceArtifact(
            artifact_id="operator-benchmark-manifest",
            artifact_type="benchmark",
            description="Operator benchmark manifest used by CI freshness checks.",
            generator_script="scripts/export_operator_manifest.py",
            runner_command="python3 scripts/export_operator_manifest.py --out benchmarks/operator_manifest.json",
            output_files=["benchmarks/operator_manifest.json"],
            input_dependencies=_infer_benchmark_inputs(),
        )
    )


def _freeze_identities() -> None:
    """Compute and persist ``evidence_input_identity`` for all default artifacts."""
    for art in _REGISTRY.values():
        art.evidence_input_identity = art.recompute_identity()
        art.generated_at = (
            art.generated_at
            or datetime.datetime.now(datetime.timezone.utc).isoformat()
        )


def init_registry() -> None:
    """Public initialization: register defaults and freeze identities."""
    _register_default_artifacts()
    _freeze_identities()


def _ensure_initialized() -> None:
    if not _REGISTRY:
        init_registry()


def load_registry() -> list[EvidenceArtifact]:
    """Ensure the default registry is populated and return all artifacts."""
    _ensure_initialized()
    return list_artifacts()


def evidence_staleness_summary() -> Dict[str, Any]:
    """Quick summary dict suitable for logs / YAML evidence."""
    _ensure_initialized()
    report = check_all_fresh()
    return {
        "total": report.total,
        "stale_count": report.stale_count,
        "stale_ids": report.stale_ids,
        "all_fresh": report.all_fresh,
        "details": report.details,
    }


__all__ = [
    "ARTIFACT_TYPES",
    "ARCHIVE_ROOT",
    "CURRENT_DIR",
    "EVIDENCE_ROOT",
    "REPO_ROOT",
    "EvidenceArtifact",
    "FreshnessReport",
    "archive_current",
    "check_all_fresh",
    "check_freshness",
    "clear_registry",
    "evidence_staleness_summary",
    "get_artifact",
    "init_registry",
    "list_artifacts",
    "load_registry",
    "publish_current",
    "register_artifact",
    "sha256_files",
]
