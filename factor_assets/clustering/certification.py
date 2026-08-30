"""Production-mode certified-graph enforcement for clustering (R55 P0-13).

The audit found that every clustering entry point in this package
(:class:`~factor_assets.clustering.families.LeidenClustering`,
:class:`~factor_assets.clustering.families.ConnectedComponents`,
:class:`~factor_assets.clustering.families.ModularityClustering`,
:class:`~factor_assets.clustering.families.HierarchicalClustering` and
:class:`~factor_assets.clustering.lineage.LineageDetector`) accepted a raw
:class:`~factor_assets.graph.sparse.SparseCorrelationGraph` with **no** binding
to any certification evidence.  A production run could therefore silently
cluster an uncertified, stale, or algorithm-mismatched graph.

This module closes that hole with ONE gate —
:func:`enforce_certified_graph` — which every clustering entry point must call
before touching a graph:

- ``RESEARCH`` mode keeps the relaxed path: clustering is allowed with or
  without certification, and uncertified algorithms are allowed.
- ``PRODUCTION`` mode requires ALL of:
    1. a certification is supplied at all (``None`` raises);
    2. the certification status is ``CERTIFIED`` (``PENDING`` / ``REVOKED``
       raise);
    3. the certification is not expired relative to ``now`` (``frozen_until``
       is a hard freeze window — clustering past it raises);
    4. the certification's recorded ``graph_content_hash`` matches the graph's
       *independently recomputed* content hash (a stale / mutated graph
       raises);
    5. the certification's recorded node / edge counts match the graph;
    6. the certification's recorded ``graph_completeness_class`` is a
       certified class (``CERTIFIED_ANN_REFINED`` / ``CERTIFIED_EXACT`` —
       DLIB-FA-054); and
    7. the requested algorithm is on the certification's
       ``allowed_algorithms`` list.

Every violation raises :class:`~factor_assets.errors.ProductionClusterViolation`
— there is no bypass path, and the gate is deliberately re-entrant (wired in
both the constructor and each clustering method of every class, so a caller
cannot escape it by holding a pre-constructed research object).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts.similarity import is_unknown_identity
from factor_assets.errors import ProductionClusterViolation

__all__ = [
    "ExecutionMode",
    "GraphCertificationStatus",
    "CERTIFIED_COMPLETENESS_CLASSES",
    "PRODUCTION_ALLOWED_ALGORITHMS",
    "CertifiedGraphArtifact",
    "compute_graph_content_hash",
    "enforce_certified_graph",
    "graph_summary",
]


class ExecutionMode(Enum):
    """Execution mode of a clustering run (R55 P0-13).

    ``PRODUCTION`` runs are gated on a certified graph;
    ``RESEARCH`` runs are not.
    """

    RESEARCH = "RESEARCH"
    PRODUCTION = "PRODUCTION"

    @classmethod
    def coerce(cls, value: "ExecutionMode | str") -> "ExecutionMode":
        """Accept an :class:`ExecutionMode`, its name, or its value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            token = value.strip().upper()
            for member in cls:
                if token in (member.name, member.value):
                    return member
        raise ValueError(
            f"execution mode must be an ExecutionMode or one of "
            f"{[m.value for m in cls]}, got {value!r}"
        )


class GraphCertificationStatus(Enum):
    """Lifecycle status of a graph certification.

    Only ``CERTIFIED`` may back a production clustering run.  ``PENDING``
    means the certification evidence has not completed review; ``REVOKED``
    means a previously certified graph was withdrawn.
    """

    CERTIFIED = "CERTIFIED"
    PENDING = "PENDING"
    REVOKED = "REVOKED"


#: Graph completeness classes that count as certified for production Leiden
#: (DLIB-FA-054).  Imported read-only from the cluster-governance contract.
CERTIFIED_COMPLETENESS_CLASSES: Tuple[str, ...] = (
    "CERTIFIED_ANN_REFINED",
    "CERTIFIED_EXACT",
)

#: The ONLY algorithm a certification may allow-list for a production run
#: (R55 P0-13).  ConnectedComponents / ModularityClustering /
#: HierarchicalClustering / LineageDetector are research-only paths, so even a
#: certification that names them can never authorise a production run with
#: them.
PRODUCTION_ALLOWED_ALGORITHMS: Tuple[str, ...] = ("leiden",)


def compute_graph_content_hash(
    *,
    node_universe: Sequence[str],
    edge_list: Sequence[Mapping[str, object]],
) -> str:
    """Content hash over a graph's node universe and edge list.

    Order-invariant over both the nodes and the edges (each edge is hashed
    with its canonical, sorted endpoint pair), so a caller that serialises the
    same graph in a different order still hashes identically.  This is the
    hash a certification records and the hash the gate *recomputes* from the
    live graph — so a mutated or stale graph can never pass.
    """
    digest = hashlib.sha256()
    for node in sorted(str(node) for node in node_universe):
        encoded = node.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    canonical_edges = sorted(
        tuple(sorted((str(e["factor_a"]), str(e["factor_b"]))))
        + (repr(float(e["correlation"])),)
        for e in edge_list
    )
    for edge in canonical_edges:
        encoded = "\x00".join(edge).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def graph_summary(graph: object) -> Tuple[Tuple[str, ...], Tuple[Mapping[str, object]]]:
    """Extract ``(node_universe, edge_records)`` from a SparseCorrelationGraph.

    Uses only the public graph surface (``nodes`` / ``to_edge_list``), so the
    gate never depends on private graph state.
    """
    nodes = tuple(sorted(str(node) for node in graph.nodes))  # type: ignore[attr-defined]
    edges: list[Mapping[str, object]] = []
    for edge in graph.to_edge_list():  # type: ignore[attr-defined]
        edges.append(
            {
                "factor_a": str(edge.factor_a),
                "factor_b": str(edge.factor_b),
                "correlation": float(edge.correlation),
            }
        )
    return nodes, tuple(edges)


@dataclass(frozen=True)
class CertifiedGraphArtifact:
    """Frozen, content-hashed certification evidence for a similarity graph.

    A production clustering run may only consume a graph bound to one of these.
    The artifact records which graph (``graph_content_hash`` + node/edge
    counts), how complete it is (``graph_completeness_class``), which
    algorithms are allowed on it, who certified it and when, and until when it
    stays frozen (``frozen_until``).  ``content_hash`` is derived over every
    semantic field, so a caller cannot self-report an arbitrary hash.
    """

    certification_id: str
    graph_content_hash: str
    node_count: int
    edge_count: int
    graph_completeness_class: str
    allowed_algorithms: Tuple[str, ...] = ("leiden",)
    status: GraphCertificationStatus = GraphCertificationStatus.CERTIFIED
    certified_at: str = ""
    frozen_until: Optional[str] = None
    certified_by: str = ""
    evidence_refs: Tuple[str, ...] = ()
    notes: Mapping[str, object] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.certification_id:
            raise ValueError("certification_id is required")
        if not self.graph_content_hash:
            raise ValueError("graph_content_hash is required")
        # P0-12 (R55 audit): an UNKNOWN placeholder certification / content
        # hash is not an identity — two certifications of *different* graphs
        # would both read UNKNOWN and production clustering would accept
        # whichever one arrived first.  Fail closed.
        for _field_name in ("certification_id", "graph_content_hash"):
            _value = getattr(self, _field_name)
            if isinstance(_value, str) and is_unknown_identity(_value):
                raise ValueError(
                    f"{_field_name} is UNKNOWN ({_value!r}); a graph "
                    "certification may not carry an unknown identity — "
                    "certify the concrete graph first (fail closed)."
                )
        if not isinstance(self.node_count, int) or isinstance(self.node_count, bool):
            raise TypeError("node_count must be an integer")
        if self.node_count < 0:
            raise ValueError("node_count must be >= 0")
        if not isinstance(self.edge_count, int) or isinstance(self.edge_count, bool):
            raise TypeError("edge_count must be an integer")
        if self.edge_count < 0:
            raise ValueError("edge_count must be >= 0")
        if not self.graph_completeness_class:
            raise ValueError("graph_completeness_class is required")
        if not isinstance(self.status, GraphCertificationStatus):
            raise TypeError(
                "status must be a GraphCertificationStatus, got "
                f"{type(self.status).__name__}"
            )
        if not isinstance(self.certified_at, str):
            raise TypeError("certified_at must be a string")
        if self.frozen_until is not None and not isinstance(self.frozen_until, str):
            raise TypeError("frozen_until must be a string or None")
        object.__setattr__(
            self, "allowed_algorithms", tuple(str(a) for a in self.allowed_algorithms)
        )
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if not isinstance(self.notes, Mapping):
            raise TypeError("notes must be a mapping")
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))
        computed = canonical_digest(
            self.certification_id,
            self.graph_content_hash,
            self.node_count,
            self.edge_count,
            self.graph_completeness_class,
            self.allowed_algorithms,
            self.status.value,
            self.certified_at,
            self.frozen_until,
            self.certified_by,
            self.evidence_refs,
            self.notes,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed graph-certification "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )

    @property
    def is_certified(self) -> bool:
        """Whether the certification is in the ``CERTIFIED`` state."""
        return self.status is GraphCertificationStatus.CERTIFIED

    @property
    def has_certified_completeness(self) -> bool:
        """Whether the certified completeness class is a production class."""
        return self.graph_completeness_class in CERTIFIED_COMPLETENESS_CLASSES

    def to_dict(self) -> dict:
        return {
            "certification_id": self.certification_id,
            "graph_content_hash": self.graph_content_hash,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "graph_completeness_class": self.graph_completeness_class,
            "allowed_algorithms": list(self.allowed_algorithms),
            "status": self.status.value,
            "certified_at": self.certified_at,
            "frozen_until": self.frozen_until,
            "certified_by": self.certified_by,
            "evidence_refs": list(self.evidence_refs),
            "notes": dict(self.notes),
            "content_hash": self.content_hash,
        }


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (``Z`` suffix tolerated). ``None`` passes."""
    if value is None:
        return None
    token = value.strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def enforce_certified_graph(
    mode: ExecutionMode | str,
    graph: object,
    certification: Optional[CertifiedGraphArtifact],
    *,
    algorithm: str,
    now: Optional[object] = None,
) -> CertifiedGraphArtifact:
    """Single production gate for clustering (R55 P0-13).

    In ``RESEARCH`` mode the gate is a no-op (the relaxed path is retained) and
    returns the certification as-is (possibly ``None``).

    In ``PRODUCTION`` mode the gate returns the certification only when every
    rule holds; otherwise it raises
    :class:`~factor_assets.errors.ProductionClusterViolation`.  The
    certification's recorded ``graph_content_hash`` is compared against a
    **freshly recomputed** hash of the graph actually handed to the clustering
    entry point, so a stale or mutated graph can never ride in under a valid
    certification.

    :param mode: ``RESEARCH`` or ``PRODUCTION`` (name or value string accepted).
    :param graph: the :class:`~factor_assets.graph.sparse.SparseCorrelationGraph`
        about to be clustered.
    :param certification: the certification evidence attached to that graph, or
        ``None`` when the graph is uncertified.
    :param algorithm: the clustering algorithm about to run (``"leiden"`` ...).
    :param now: optional ``datetime`` (naive treated as UTC) or ISO-8601 string
        used as the freshness clock; defaults to the current UTC time.
    :returns: the certification that authorised the run.
    :raises ProductionClusterViolation: in production mode on any rule failure.
    """
    resolved_mode = ExecutionMode.coerce(mode)
    if resolved_mode is ExecutionMode.RESEARCH:
        # Relaxed research path: no gate, certification passed through.
        return certification  # type: ignore[return-value]

    violation_prefix = (
        f"PRODUCTION clustering with algorithm {algorithm!r} refused: "
    )

    if certification is None:
        raise ProductionClusterViolation(
            violation_prefix
            + "no graph certification was supplied.  Production runs require a "
            "frozen, content-hash-verified CertifiedGraphArtifact."
        )
    if not isinstance(certification, CertifiedGraphArtifact):
        raise ProductionClusterViolation(
            violation_prefix
            + f"certification must be a CertifiedGraphArtifact, got "
            f"{type(certification).__name__}."
        )

    # Rule: certification status.
    if not certification.is_certified:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph certification {certification.certification_id!r} has status "
            f"{certification.status.value!r}; only 'CERTIFIED' may back a "
            "production clustering run."
        )

    # Rule: certified completeness class (DLIB-FA-054).
    if not certification.has_certified_completeness:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph certification {certification.certification_id!r} records "
            f"graph_completeness_class "
            f"{certification.graph_completeness_class!r}; production requires "
            f"one of {list(CERTIFIED_COMPLETENESS_CLASSES)}."
        )

    # Rule: algorithm allow-list.  A certification may never authorise a
    # non-production algorithm (only sparse-graph Leiden is production
    # capable), regardless of what its allow-list names.
    if algorithm not in PRODUCTION_ALLOWED_ALGORITHMS:
        raise ProductionClusterViolation(
            violation_prefix
            + f"algorithm {algorithm!r} is not production-capable; production "
            f"clustering may only use {list(PRODUCTION_ALLOWED_ALGORITHMS)}."
        )
    if algorithm not in certification.allowed_algorithms:
        raise ProductionClusterViolation(
            violation_prefix
            + f"algorithm {algorithm!r} is not in the certification's "
            f"allowed_algorithms {list(certification.allowed_algorithms)}."
        )

    # Rule: freshness / freeze window.
    clock = _resolve_now(now)
    frozen_until = _parse_timestamp(certification.frozen_until)
    if frozen_until is not None and clock > frozen_until:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph certification {certification.certification_id!r} is stale: "
            f"its freeze window ended at {certification.frozen_until!r} "
            f"(now {clock.isoformat()}); re-certify the graph before a "
            "production run."
        )

    # Rule: content-hash binding against the graph ACTUALLY supplied.
    node_universe, edge_records = graph_summary(graph)
    actual_hash = compute_graph_content_hash(
        node_universe=node_universe, edge_list=edge_records
    )
    if actual_hash != certification.graph_content_hash:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph content hash mismatch: certification "
            f"{certification.certification_id!r} certifies graph "
            f"{certification.graph_content_hash[:16]!r} but the supplied graph "
            f"hashes to {actual_hash[:16]!r}.  A stale or mutated graph must "
            "never be clustered in production — re-certify it."
        )

    # Rule: structural counts.
    if len(node_universe) != certification.node_count:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph node count mismatch: certification "
            f"{certification.certification_id!r} certifies "
            f"{certification.node_count} nodes but the supplied graph has "
            f"{len(node_universe)}."
        )
    if len(edge_records) != certification.edge_count:
        raise ProductionClusterViolation(
            violation_prefix
            + f"graph edge count mismatch: certification "
            f"{certification.certification_id!r} certifies "
            f"{certification.edge_count} edges but the supplied graph has "
            f"{len(edge_records)}."
        )

    return certification


def _resolve_now(now: Optional[object]) -> datetime:
    """Resolve the gate clock: datetime, ISO-8601 string, or current UTC."""
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, str):
        parsed = _parse_timestamp(now)
        assert parsed is not None  # _parse_timestamp raises on garbage
        return parsed
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now
    raise TypeError("now must be a datetime, an ISO-8601 string, or None")