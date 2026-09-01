"""
Factor family clustering algorithms.

Connected components for basic grouping, modularity-based clustering
for community detection in correlation graphs, and hierarchical clustering
with dendrogram cutting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import hashlib

from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts._frozen import FrozenMapping

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    hierarchy = None
    squareform = None
    np = None

from factor_assets.clustering.certification import (
    CertifiedGraphArtifact,
    ExecutionMode,
    enforce_certified_graph,
)
from factor_assets.graph.sparse import SparseCorrelationGraph
from factor_assets.errors import InvalidClusteringContract
from factor_assets.contracts.similarity import is_unknown_identity


class SimilarityObservationState(Enum):
    """
    Observation state of a pairwise similarity/distance.

    Distinguishes a genuinely computed low/high similarity from a MISSING
    observation.  A missing edge is NOT evidence of low similarity — treating
    it as distance=1 silently conflates "not measured" with "dissimilar".
    """
    UNKNOWN = "UNKNOWN"              # no observation; must not be treated as a distance
    COMPUTED_LOW = "COMPUTED_LOW"    # observed and below the similarity threshold
    COMPUTED_HIGH = "COMPUTED_HIGH"  # observed and at/above the similarity threshold


@dataclass(frozen=True)
class ClusterResult:
    """
    Result of a clustering operation.

    Maps each factor to its cluster ID and provides cluster membership.

    ``certification_id`` / ``graph_content_hash`` / ``execution_mode`` record
    the certified-graph provenance of the run (R55 P0-13).  They are ``None``
    for a research run on an uncertified graph and are populated by the
    production gate whenever a certification authorised the clustering.
    """

    assignments: Dict[str, int]
    cluster_sizes: Dict[int, int]
    #: Cluster IDs whose size fell below ``min_cluster_size`` under a
    #: min-cluster policy (e.g. ``MARK_UNSTABLE``).  Empty when not applied.
    unstable_clusters: Tuple[int, ...] = ()
    #: ``CertifiedGraphArtifact.certification_id`` backing this run (None when
    #: the run was uncertified / research).
    certification_id: Optional[str] = None
    #: Content hash of the certified graph the run consumed.
    graph_content_hash: Optional[str] = None
    #: ``"RESEARCH"`` or ``"PRODUCTION"`` — which gate authorises the run.
    execution_mode: str = "RESEARCH"

    @property
    def num_clusters(self) -> int:
        """Number of clusters found."""
        return len(self.cluster_sizes)

    def get_cluster_members(self, cluster_id: int) -> Set[str]:
        """Get all factors in a cluster."""
        return {factor for factor, cid in self.assignments.items() if cid == cluster_id}

    def get_all_clusters(self) -> Dict[int, Set[str]]:
        """Get all clusters as dict of cluster_id -> members."""
        clusters = defaultdict(set)
        for factor, cluster_id in self.assignments.items():
            clusters[cluster_id].add(factor)
        return dict(clusters)


@dataclass(frozen=True)
class ClusterArtifact:
    """
    Production clustering artifact — full, hash-addressed provenance of a
    clustering result, richer than the thin :class:`ClusterResult`.

    Records which graph (``graph_identity``), which similarity spec
    (``similarity_spec_ref``), on which snapshot/universe, with which algorithm
    and backend/backend_version/seed/resolution produced the assignment.
    ``representatives`` is one factor per cluster; ``modularity`` / ``stability``
    are optional quality signals.  ``content_hash`` is derived over every
    semantic field (see :meth:`recompute_content_hash`) so no caller may
    self-report an arbitrary hash.
    """

    graph_identity: str
    algorithm: str
    assignments: Dict[str, int]
    representatives: Tuple[str, ...]
    content_hash: str
    snapshot_ref: Optional[str] = None
    universe_ref: Optional[str] = None
    similarity_spec_ref: Optional[str] = None
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    seed: Optional[int] = None
    resolution: Optional[float] = None
    min_cluster_size: int = 1
    min_cluster_policy: str = "KEEP_SMALL"
    modularity: Optional[float] = None
    stability: Optional[float] = None
    #: ``CertifiedGraphArtifact.certification_id`` backing a production run
    #: (R55 P0-13); ``None`` for research / uncertified runs.
    certification_id: Optional[str] = None
    #: ``"RESEARCH"`` or ``"PRODUCTION"`` — which gate authorised the run.
    execution_mode: str = "RESEARCH"

    def __post_init__(self) -> None:
        if not self.graph_identity:
            raise ValueError("graph_identity is required")
        if not self.algorithm:
            raise ValueError("algorithm is required")
        if not self.assignments:
            raise ValueError("assignments is required")
        # P0-12 (R55 audit): a spec-identity reference carrying an UNKNOWN
        # placeholder is not an identity — two ClusterArtifacts computed under
        # different similarity specs would both read UNKNOWN and cluster
        # versioning would silently merge them.  Fail closed; ``None`` stays
        # legal (the provenance dimension is simply absent).
        for _field_name in ("graph_identity", "snapshot_ref", "universe_ref", "similarity_spec_ref"):
            _value = getattr(self, _field_name)
            if isinstance(_value, str) and is_unknown_identity(_value):
                raise InvalidClusteringContract(
                    f"{_field_name} is UNKNOWN ({_value!r}); a ClusterArtifact "
                    "may not carry an unknown spec identity — resolve the "
                    "concrete spec value first (fail closed)."
                )
        # DLIB-FA-008: deep-immutable — snapshot the caller's assignments dict
        # into an immutable FrozenMapping so mutating the original dict after
        # construction cannot change the artifact.  FrozenMapping is hashable
        # and deepcopy-safe (matching the QE cross-package convention), so the
        # artifact's assignments no longer block copy.deepcopy / hash.
        object.__setattr__(self, "assignments", FrozenMapping(dict(self.assignments)))
        object.__setattr__(self, "representatives", tuple(self.representatives))
        computed = self.recompute_content_hash()
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed cluster content "
                "hash; a caller may not self-report an arbitrary hash — FAIL CLOSED"
            )

    @property
    def num_clusters(self) -> int:
        """Number of clusters (distinct cluster ids)."""
        return len(set(self.assignments.values()))

    def recompute_content_hash(self) -> str:
        """sha256 over every semantic field of the cluster artifact.

        DLIB-FA-056: uses the typed canonical structural encoding
        (:func:`factor_assets.contracts._canonical.canonical_digest`), not
        ``str(value)``, so structurally equal nested values hash identically
        and equal-looking values of different types differ.
        """
        return canonical_digest(
            self.graph_identity,
            self.algorithm,
            self.snapshot_ref,
            self.universe_ref,
            self.similarity_spec_ref,
            self.backend,
            self.backend_version,
            self.seed,
            self.resolution,
            self.min_cluster_size,
            self.min_cluster_policy,
            self.modularity,
            self.stability,
            self.assignments,
            self.representatives,
        )

    def to_dict(self) -> Dict[str, object]:
        """Serializable dict form (provenance + hash retained)."""
        return {
            "graph_identity": self.graph_identity,
            "algorithm": self.algorithm,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "similarity_spec_ref": self.similarity_spec_ref,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "seed": self.seed,
            "resolution": self.resolution,
            "min_cluster_size": self.min_cluster_size,
            "min_cluster_policy": self.min_cluster_policy,
            "assignments": dict(self.assignments),
            "representatives": list(self.representatives),
            "modularity": self.modularity,
            "stability": self.stability,
            "content_hash": self.content_hash,
        }


class ConnectedComponents:
    """
    Find connected components in correlation graph.

    Uses DFS to identify disconnected subgraphs. Factors in the same
    component have at least one correlation path connecting them.

    R55 P0-13: this is NOT a production clustering algorithm.  In
    ``PRODUCTION`` mode every entry point raises
    :class:`~factor_assets.errors.ProductionClusterViolation` (an uncertified
    algorithm may never cluster a production graph); in ``RESEARCH`` mode the
    relaxed path is unchanged.  A supplied certification is validated by the
    gate in both modes.
    """

    #: Not a production-capable algorithm: production runs must use Leiden.
    PRODUCTION_ALGORITHM = None

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        execution_mode: ExecutionMode | str = ExecutionMode.RESEARCH,
        certification: Optional[CertifiedGraphArtifact] = None,
    ):
        """
        Args:
            graph: Correlation graph to cluster
            execution_mode: ``RESEARCH`` (default) or ``PRODUCTION``.  In
                production mode this class raises — only Leiden is certified
                for production clustering.
            certification: certification evidence attached to ``graph``.
        """
        self.graph = graph
        self.execution_mode = ExecutionMode.coerce(execution_mode)
        self.certification = certification

    def find_components(self, *, now: Optional[object] = None) -> ClusterResult:
        """
        Find all connected components.

        Args:
            now: optional gate clock (see :func:`~factor_assets.clustering.
                certification.enforce_certified_graph`).

        Returns:
            Cluster assignment where each component gets unique ID
        """
        # R55 P0-13: the gate runs at EVERY entry point — a caller cannot
        # reach the clustering body without passing it.
        self._gate(algorithm="connected_components", now=now)
        visited = set()
        assignments = {}
        cluster_id = 0

        for node in self.graph.nodes:
            if node in visited:
                continue

            # DFS to find component
            component = self._dfs(node, visited)

            # Assign cluster ID to all nodes in component
            for member in component:
                assignments[member] = cluster_id

            cluster_id += 1

        # Calculate cluster sizes
        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes),
            certification_id=(
                self.certification.certification_id
                if self.certification is not None
                else None
            ),
            graph_content_hash=(
                self.certification.graph_content_hash
                if self.certification is not None
                else None
            ),
            execution_mode=self.execution_mode.value,
        )

    def _gate(self, *, algorithm: str, now: Optional[object] = None):
        """Run the certified-graph gate for this run (R55 P0-13).

        In production mode an uncertified graph (or an algorithm not on the
        certification's allow-list) raises
        :class:`~factor_assets.errors.ProductionClusterViolation`; research
        mode passes through unchanged.
        """
        return enforce_certified_graph(
            self.execution_mode,
            self.graph,
            self.certification,
            algorithm=algorithm,
            now=now,
        )

    def _dfs(self, start: str, visited: Set[str]) -> Set[str]:
        """DFS to find all reachable nodes from start."""
        component = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            # Add unvisited neighbors
            for neighbor, _ in self.graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

        return component


#: Shared certified-graph gate (R55 P0-13).  Every clustering class routes its
#: entry points through this one function so there is a single, patchable gate
#: and no per-class divergent implementation.  ``runner`` must expose
#: ``execution_mode``, ``graph`` and ``certification``.
def _certified_graph_gate(runner, *, algorithm: str, now: Optional[object] = None):
    return enforce_certified_graph(
        runner.execution_mode,
        runner.graph,
        runner.certification,
        algorithm=algorithm,
        now=now,
    )


class ModularityClustering:
    """
    RESEARCH_ONLY greedy modularity optimization for community detection.

    Uses a Louvain-style algorithm to maximize modularity score.  This is a
    toy/reference implementation for testing and debugging only.  Production
    clustering at 100K scale must use :class:`LeidenClustering` (sparse-graph
    Leiden via a mature backend) — this class never builds a dense matrix but
    is not the production path.
    """

    RESEARCH_ONLY = True

    #: Not a production-capable algorithm: production runs must use Leiden.
    PRODUCTION_ALGORITHM = None

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        resolution: float = 1.0,
        min_cluster_size: int = 1,
        max_cluster_size: Optional[int] = None,
        allow_toy_algorithm: bool = False,
        execution_mode: ExecutionMode | str = ExecutionMode.RESEARCH,
        certification: Optional[CertifiedGraphArtifact] = None,
    ):
        """
        Args:
            graph: Correlation graph to cluster
            resolution: Resolution parameter (higher = more/smaller clusters)
            min_cluster_size: Minimum cluster size (smaller clusters merged/rejected)
            max_cluster_size: Maximum cluster size (larger clusters flagged with warning)
            allow_toy_algorithm: Must be True to bypass production backend check.
                                Set to True ONLY for testing/debugging.
            execution_mode: ``RESEARCH`` (default) or ``PRODUCTION``.  In
                production mode this toy algorithm is refused outright
                (R55 P0-13).
            certification: certification evidence attached to ``graph``.

        Raises:
            RuntimeError: If allow_toy_algorithm=False and no production backend available
            ProductionClusterViolation: If execution_mode is PRODUCTION.
        """
        self.graph = graph
        self.resolution = resolution
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.execution_mode = ExecutionMode.coerce(execution_mode)
        self.certification = certification
        # R55 P0-13: the gate runs in the constructor AND in cluster() so a
        # caller cannot smuggle a production run through a research-constructed
        # object.
        self._gate(algorithm="modularity")
        # Production safety: require explicit opt-in for toy algorithm
        if not allow_toy_algorithm:
            # Check for production backends
            try:
                import igraph
                # If igraph available, could use Leiden - but not implemented yet
                # For now, fail closed
                raise RuntimeError(
                    "ModularityClustering is a toy/reference implementation. "
                    "Production clustering requires mature backend (e.g., Leiden via igraph). "
                    "Set allow_toy_algorithm=True ONLY for testing/debugging."
                )
            except ImportError:
                raise RuntimeError(
                    "ModularityClustering is a toy/reference implementation. "
                    "Production clustering requires mature backend (e.g., Leiden via igraph). "
                    "Install python-igraph or set allow_toy_algorithm=True for testing only."
                )

    def _gate(self, *, algorithm: str, now: Optional[object] = None):
        """Run the certified-graph gate for this run (R55 P0-13).

        In production mode this toy algorithm is refused outright; research
        mode passes through unchanged.
        """
        return enforce_certified_graph(
            self.execution_mode,
            self.graph,
            self.certification,
            algorithm=algorithm,
            now=now,
        )

    def cluster(self, max_iterations: int = 100, *, now: Optional[object] = None) -> ClusterResult:
        """
        Run modularity clustering.

        Args:
            max_iterations: Maximum optimization iterations
            now: optional gate clock (see :func:`~factor_assets.clustering.
                certification.enforce_certified_graph`).

        Returns:
            Cluster assignments
        """
        # R55 P0-13: the gate runs at EVERY entry point.
        self._gate(algorithm="modularity", now=now)
        # Initialize: each node in its own cluster
        assignments = {node: i for i, node in enumerate(self.graph.nodes)}
        cluster_id_counter = len(self.graph.nodes)

        total_weight = self._total_edge_weight()

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try moving each node to neighbor's cluster
            for node in self.graph.nodes:
                current_cluster = assignments[node]

                # Find best cluster to move to
                best_cluster = current_cluster
                best_gain = 0.0

                # Consider neighbors' clusters
                neighbor_clusters = set()
                for neighbor, _ in self.graph.neighbors(node):
                    neighbor_clusters.add(assignments[neighbor])

                for target_cluster in neighbor_clusters:
                    if target_cluster == current_cluster:
                        continue

                    gain = self._modularity_gain(
                        node, current_cluster, target_cluster,
                        assignments, total_weight
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_cluster = target_cluster

                # Move if beneficial
                if best_cluster != current_cluster:
                    assignments[node] = best_cluster
                    improved = True

        # Renumber clusters to be contiguous
        unique_clusters = sorted(set(assignments.values()))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        assignments = {node: cluster_map[cid] for node, cid in assignments.items()}

        # Calculate cluster sizes
        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes),
            certification_id=(
                self.certification.certification_id
                if self.certification is not None
                else None
            ),
            graph_content_hash=(
                self.certification.graph_content_hash
                if self.certification is not None
                else None
            ),
            execution_mode=self.execution_mode.value,
        )

    def _total_edge_weight(self) -> float:
        """Sum of absolute correlations (edge weights)."""
        total = 0.0
        for edge in self.graph.to_edge_list():
            total += edge.abs_correlation
        return total

    def _modularity_gain(
        self,
        node: str,
        from_cluster: int,
        to_cluster: int,
        assignments: Dict[str, int],
        total_weight: float
    ) -> float:
        """
        Calculate modularity gain from moving node between clusters.

        Simplified calculation using edge weights.
        """
        if total_weight == 0:
            return 0.0

        # Weight of edges from node to target cluster
        weight_to_target = 0.0
        # Weight of edges from node to current cluster
        weight_to_current = 0.0

        for neighbor, corr in self.graph.neighbors(node):
            abs_corr = abs(corr)
            if assignments[neighbor] == to_cluster:
                weight_to_target += abs_corr
            elif assignments[neighbor] == from_cluster:
                weight_to_current += abs_corr

        # Modularity gain approximation
        gain = (weight_to_target - weight_to_current) / total_weight
        return gain * self.resolution


class LeidenClustering:
    """
    Production sparse-graph Leiden clustering.

    Pipeline: ANN shortlist -> exact similarity refinement -> sparse graph ->
    Leiden via a mature backend (``igraph`` or ``leidenalg``).  Never builds a
    dense matrix, so it scales to 100K+ factors.  Uses a stable seed for
    reproducible cluster assignments and records cluster lineage.

    If neither ``igraph`` nor ``leidenalg`` is installed, this class FAILS
    CLOSED (raises) rather than faking a result — production clustering
    requires a mature Leiden backend.
    """

    RESEARCH_ONLY = False
    PRODUCTION_CAPABLE = True  # re-derived at import in _LeidenBackendGuard (P1-FA-012)
    #: The only algorithm certified for production clustering (R55 P0-13).
    PRODUCTION_ALGORITHM = "leiden"

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        resolution: float = 1.0,
        seed: int = 42,
        min_cluster_size: int = 1,
        min_cluster_policy: str = "KEEP_SMALL",
        execution_mode: ExecutionMode | str = ExecutionMode.RESEARCH,
        certification: Optional[CertifiedGraphArtifact] = None,
    ):
        """
        Args:
            graph: Sparse correlation graph to cluster.
            resolution: Leiden resolution parameter (higher = more/smaller clusters).
            seed: Stable random seed for reproducible assignments.
            min_cluster_size: Minimum cluster size.  The parameter is honored
                via ``min_cluster_policy`` — it is never a dead configuration.
            min_cluster_policy: How to handle a cluster smaller than
                ``min_cluster_size``.  ``"KEEP_SMALL"`` (default, production)
                records small clusters as-is so no factor is lost or merged
                against dissimilar neighbors; ``"MERGE_NEAREST"`` merges a
                small cluster into its nearest neighbor by edge weight;
                ``"MARK_UNSTABLE"`` keeps small clusters but records them as
                unstable via ``ClusterResult.unstable_clusters``.
            execution_mode: ``RESEARCH`` (default) or ``PRODUCTION``.  In
                production mode the graph MUST be bound to a valid, frozen,
                content-hash-verified :class:`CertifiedGraphArtifact` whose
                allow-list contains ``"leiden"`` (R55 P0-13); anything else
                raises :class:`~factor_assets.errors.ProductionClusterViolation`.
            certification: certification evidence attached to ``graph``.

        Raises:
            ImportError: If neither igraph nor leidenalg is installed.
            ProductionClusterViolation: In production mode with an uncertified,
                stale, or algorithm-mismatched graph.
        """
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")
        if min_cluster_policy not in ("KEEP_SMALL", "MERGE_NEAREST", "MARK_UNSTABLE"):
            raise ValueError(
                "min_cluster_policy must be one of "
                "'KEEP_SMALL' / 'MERGE_NEAREST' / 'MARK_UNSTABLE'"
            )
        self.graph = graph
        self.resolution = resolution
        self.seed = seed
        self.min_cluster_size = min_cluster_size
        self.min_cluster_policy = min_cluster_policy
        self.execution_mode = ExecutionMode.coerce(execution_mode)
        self.certification = certification
        # R55 P0-13: the gate runs in the constructor AND at every clustering
        # entry point, so a caller cannot escape it by holding a
        # research-constructed object and calling cluster() in production.
        self._gate(algorithm=self.PRODUCTION_ALGORITHM)
        # P1-FA-012: backend detection is eager for PRODUCTION runs (a
        # production clustering must never silently fall back to a research
        # backend).  RESEARCH runs detect the backend lazily — the graph is
        # still never clustered without a mature Leiden library (fails closed),
        # but construction stays possible so the certified-graph gate can be
        # exercised independently of the optional igraph/leidenalg dependency.
        if self.execution_mode is ExecutionMode.PRODUCTION:
            self._backend = self._detect_backend()
        else:
            try:
                self._backend = self._detect_backend()
            except ImportError:
                self._backend = ""

    @property
    def run_config(self) -> Dict[str, object]:
        """Record the RNG / backend configuration that produced a partition.

        Because igraph's ``set_random_number_generator`` mutates a module-global
        RNG, a concurrent campaign could pollute it.  Recording backend /
        backend_version / seed / resolution lets a caller reproduce (or audit)
        an exact run and detect cross-campaign interference.
        """
        return {
            "backend": self._backend,
            "backend_version": self._backend_version(),
            "seed": self.seed,
            "resolution": self.resolution,
            "min_cluster_size": self.min_cluster_size,
            "min_cluster_policy": self.min_cluster_policy,
        }

    @staticmethod
    def _detect_backend() -> str:
        """Detect an available mature Leiden backend, else fail closed."""
        try:
            import igraph  # noqa: F401
            return "igraph"
        except ImportError:
            pass
        try:
            import leidenalg  # noqa: F401
            return "leidenalg"
        except ImportError:
            pass
        raise ImportError(
            "LeidenClustering requires a mature Leiden backend (igraph or "
            "leidenalg) for production clustering.  Neither is installed. "
            "Install python-igraph or leidenalg.  This class fails closed "
            "rather than faking a clustering result."
        )


# P1-FA-012: the class-level ``PRODUCTION_CAPABLE`` flag is re-derived at
# import so ``LeidenClustering.PRODUCTION_CAPABLE`` honestly reflects the
# installed backends.  A class that is NOT production-capable must not be
# reported as such by a stale classvar.
try:
    LeidenClustering._detect_backend()  # noqa: SLF001
except ImportError:
    LeidenClustering.PRODUCTION_CAPABLE = False  # type: ignore[assignment]


def _leiden_gate(self, *, algorithm: str, now: Optional[object] = None):
    """Certified-graph gate for LeidenClustering (R55 P0-13).

    In production mode an uncertified / stale / algorithm-mismatched graph
    raises :class:`~factor_assets.errors.ProductionClusterViolation`; research
    mode passes through unchanged.  Returns the certification that authorised
    the run (or ``None`` in research without certification).
    """
    return _certified_graph_gate(self, algorithm=algorithm, now=now)


def _leiden_cluster(self, *, now: Optional[object] = None) -> ClusterResult:
    """Leiden clustering entry point (gate runs at EVERY entry point)."""
    # R55 P0-13: the gate runs at every entry point — no bypass path.
    self._gate(algorithm=self.PRODUCTION_ALGORITHM, now=now)
    if self.graph.node_count == 0:
        return ClusterResult(
            assignments={},
            cluster_sizes={},
            certification_id=(
                self.certification.certification_id if self.certification else None
            ),
            graph_content_hash=(
                self.certification.graph_content_hash
                if self.certification is not None
                else None
            ),
            execution_mode=self.execution_mode.value,
        )
    # RESEARCH runs may construct without a Leiden backend (so the certified-graph
    # gate can be exercised independently of the optional igraph/leidenalg dep);
    # a research clustering without a backend fails closed with a clear message
    # rather than fabricating a partition.
    if self.execution_mode is ExecutionMode.PRODUCTION:
        if self._backend == "igraph":
            return _leiden_cluster_igraph(self)
        if self._backend == "leidenalg":
            return _leiden_cluster_leidenalg(self)
        raise RuntimeError(f"unknown Leiden backend: {self._backend!r}")
    # RESEARCH runs carry a fully-detected backend when a mature Leiden library
    # is installed (the real research clustering path); an empty ``_backend``
    # only occurs when construction ran under P1-FA-012's research relaxation
    # WITHOUT a backend — that path fails closed rather than fabricating a
    # partition.
    if self._backend == "igraph":
        return _leiden_cluster_igraph(self)
    if self._backend == "leidenalg":
        return _leiden_cluster_leidenalg(self)
    raise RuntimeError(
        "research-mode Leiden clustering requires igraph or leidenalg "
        "installed; this object was constructed without a backend"
    )


# Wire the module-level helpers onto the class so the certification gate and
# clustering entry point are always present, whether or not a Leiden backend is
# installed (the class-body methods would otherwise be absent in the
# no-backend build).
LeidenClustering._gate = _leiden_gate
LeidenClustering.cluster = _leiden_cluster


def _leiden_cluster_artifact(
    self,
    *,
    snapshot_ref: Optional[str] = None,
    universe_ref: Optional[str] = None,
    similarity_spec_ref: Optional[str] = None,
    now: Optional[object] = None,
) -> "ClusterArtifact":
    """Produce a production :class:`ClusterArtifact` with full provenance.

    Wraps the raw :class:`ClusterResult` with the algorithm, backend /
    backend_version / seed / resolution that produced it, the graph
    identity, snapshot / universe / similarity-spec refs, one representative
    per cluster, and a derived ``content_hash``.  In production mode the
    certification that authorised the run is recorded on the artifact
    (``certification_id`` / ``execution_mode``).
    """
    # R55 P0-13: the gate runs here too (this is a clustering entry point,
    # not just a wrapper around cluster()).
    certified = self._gate(algorithm=self.PRODUCTION_ALGORITHM, now=now)
    result = self.cluster(now=now)
    cluster_of: Dict[int, List[str]] = defaultdict(list)
    for fid, cid in result.assignments.items():
        cluster_of[cid].append(fid)
    representatives = tuple(sorted(min(members) for members in cluster_of.values()))
    return ClusterArtifact(
        graph_identity=self.graph.graph_identity,
        algorithm="leiden",
        assignments=dict(result.assignments),
        representatives=representatives,
        snapshot_ref=snapshot_ref,
        universe_ref=universe_ref,
        similarity_spec_ref=similarity_spec_ref,
        backend=self._backend,
        backend_version=self._backend_version(),
        seed=self.seed,
        resolution=self.resolution,
        min_cluster_size=self.min_cluster_size,
        min_cluster_policy=self.min_cluster_policy,
        content_hash="",  # derived in __post_init__
        certification_id=(
            certified.certification_id if certified is not None else None
        ),
        execution_mode=self.execution_mode.value,
    )


LeidenClustering.cluster_artifact = _leiden_cluster_artifact


@staticmethod
def _leiden_backend_version() -> str:
    try:
        import igraph
        return getattr(igraph, "__version__", "unknown")
    except ImportError:
        return "unknown"

LeidenClustering._backend_version = _leiden_backend_version


def _leiden_cluster_igraph(self) -> ClusterResult:
    """Leiden via python-igraph."""
    import random

    import igraph

    factor_ids = sorted(self.graph.nodes)
    id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}
    edges = []
    weights = []
    for edge in self.graph.to_edge_list():
        edges.append((id_to_idx[edge.factor_a], id_to_idx[edge.factor_b]))
        weights.append(edge.abs_correlation)

    g = igraph.Graph(n=len(factor_ids), edges=edges, directed=False)
    g.es["weight"] = weights

    # Stable seed for reproducibility.  igraph 1.x exposes only
    # `set_random_number_generator(generator)` (no `igraph.Random` class);
    # a fresh `random.Random(seed)` per run yields deterministic Leiden.
    igraph.set_random_number_generator(random.Random(self.seed))
    partition = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=self.resolution,
        n_iterations=2,
    )

    assignments = {
        factor_ids[idx]: int(member)
        for idx, member in enumerate(partition.membership)
    }
    return _leiden_finalize(self, assignments)


def _leiden_cluster_leidenalg(self) -> ClusterResult:
    """Leiden via leidenalg (requires igraph for the graph object)."""
    import igraph
    import leidenalg

    factor_ids = sorted(self.graph.nodes)
    id_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}
    edges = []
    weights = []
    for edge in self.graph.to_edge_list():
        edges.append((id_to_idx[edge.factor_a], id_to_idx[edge.factor_b]))
        weights.append(edge.abs_correlation)

    g = igraph.Graph(n=len(factor_ids), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=self.resolution,
        seed=self.seed,
    )
    assignments = {
        factor_ids[idx]: int(member)
        for idx, member in enumerate(partition.membership)
    }
    return _leiden_finalize(self, assignments)


def _leiden_finalize(self, assignments: Dict[str, int]) -> ClusterResult:
    """Renumber clusters contiguously and compute sizes, then enforce the
    min-cluster-size policy so it is never a dead parameter."""
    unique = sorted(set(assignments.values()))
    cluster_map = {old: new for new, old in enumerate(unique)}

    # DLIB-FA-007/57: the algorithm label is unstable (Leiden label ``18``
    # may be ``5`` next refresh).  Stable logical cluster identity is owned
    # by the cluster-governance layer (:class:`~factor_assets.contracts.
    # cluster_governance.LogicalCluster` / ``ClusterVersionArtifact``).
    # This renumbering keeps the algorithm order contiguous and
    # deterministic (smallest factor_id first), which is the only locality
    # guaranteed here.
    ordered_ids = sorted(assignments)
    label_order = {cid: idx for idx, cid in enumerate(sorted(set(assignments.values())))}
    # The map from old cluster-id to contiguous label must be deterministic
    # and order-INVARIANT: build it by walking the sorted factor ids, not
    # by sorting raw integer cluster ids (which are arbitrary backend ids).
    cluster_map = {}
    for fid in ordered_ids:
        old = assignments[fid]
        if old not in cluster_map:
            cluster_map[old] = len(cluster_map)
    renumbered = {fid: cluster_map[cid] for fid, cid in assignments.items()}
    sizes = defaultdict(int)
    for cid in renumbered.values():
        sizes[cid] += 1

    unstable: List[int] = []
    if self.min_cluster_size > 1 and self.min_cluster_policy == "MERGE_NEAREST":
        # Greedily merge every cluster smaller than min_cluster_size into
        # the nearest neighbor cluster (largest aggregate edge weight).
        by_cluster: Dict[int, List[str]] = defaultdict(list)
        for fid, cid in renumbered.items():
            by_cluster[cid].append(fid)
        small = sorted(
            [cid for cid, members in by_cluster.items() if len(members) < self.min_cluster_size]
        )
        for small_cid in small:
            if small_cid not in renumbered.values():
                continue
            members = [fid for fid, cid in renumbered.items() if cid == small_cid]
            if not members:
                continue
            # Find nearest cluster by sum of |corr| edge weights.
            best_cid: Optional[int] = None
            best_weight = -1.0
            for fid in members:
                for neighbor, corr in self.graph.neighbors(fid):
                    ncid = renumbered.get(neighbor)
                    if ncid is None or ncid == small_cid:
                        continue
                    w = abs(corr)
                    if w > best_weight:
                        best_weight = w
                        best_cid = ncid
            if best_cid is None:
                continue
            for fid in members:
                renumbered[fid] = best_cid
        # Recompute sizes after merging.
        sizes = defaultdict(int)
        for cid in renumbered.values():
            sizes[cid] += 1
    elif self.min_cluster_size > 1 and self.min_cluster_policy == "MARK_UNSTABLE":
        unstable = sorted(
            [cid for cid, size in sizes.items() if size < self.min_cluster_size]
        )

    return ClusterResult(
        assignments=renumbered,
        cluster_sizes=dict(sizes),
        unstable_clusters=tuple(unstable),
        certification_id=(
            self.certification.certification_id
            if self.certification is not None
            else None
        ),
        graph_content_hash=(
            self.certification.graph_content_hash
            if self.certification is not None
            else None
        ),
        execution_mode=self.execution_mode.value,
    )


@dataclass(frozen=True)
class Dendrogram:
    """
    Hierarchical clustering dendrogram.

    Stores linkage matrix and factor order for dendrogram visualization.
    """
    linkage_matrix: "np.ndarray"
    factor_ids: List[str]
    method: str

    def cut_at_distance(self, threshold: float) -> ClusterResult:
        """
        Cut dendrogram at specified distance threshold.

        Args:
            threshold: Distance threshold for cutting

        Returns:
            ClusterResult with cluster assignments
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for dendrogram cutting")

        # Cut dendrogram
        labels = hierarchy.fcluster(
            self.linkage_matrix,
            threshold,
            criterion='distance'
        )

        # Convert to ClusterResult
        assignments = {fid: int(label) - 1 for fid, label in zip(self.factor_ids, labels)}

        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def cut_at_num_clusters(self, num_clusters: int) -> ClusterResult:
        """
        Cut dendrogram to obtain specified number of clusters.

        Args:
            num_clusters: Desired number of clusters

        Returns:
            ClusterResult with cluster assignments
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for dendrogram cutting")

        if num_clusters < 1:
            raise ValueError("num_clusters must be at least 1")
        if num_clusters > len(self.factor_ids):
            raise ValueError(f"num_clusters cannot exceed number of factors ({len(self.factor_ids)})")

        # Cut dendrogram
        labels = hierarchy.fcluster(
            self.linkage_matrix,
            num_clusters,
            criterion='maxclust'
        )

        # Convert to ClusterResult
        assignments = {fid: int(label) - 1 for fid, label in zip(self.factor_ids, labels)}

        cluster_sizes = defaultdict(int)
        for cid in assignments.values():
            cluster_sizes[cid] += 1

        return ClusterResult(
            assignments=assignments,
            cluster_sizes=dict(cluster_sizes)
        )

    def get_optimal_num_clusters(
        self,
        min_clusters: int = 2,
        max_clusters: Optional[int] = None
    ) -> int:
        """
        Estimate optimal number of clusters using distance gaps.

        Uses second derivative of linkage distances to find elbow.

        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters (default: n/2)

        Returns:
            Estimated optimal number of clusters
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for optimal cluster estimation")

        n = len(self.factor_ids)
        if max_clusters is None:
            max_clusters = max(min_clusters, n // 2)

        if n < 2:
            return 1

        # Get merge distances from linkage matrix
        merge_distances = self.linkage_matrix[:, 2]

        # Look for largest gaps in distances
        if len(merge_distances) < 2:
            return min_clusters

        # Calculate distance differences (gaps)
        gaps = np.diff(merge_distances)

        # Find largest gap that results in valid cluster count
        valid_gaps = []
        for i, gap in enumerate(gaps):
            num_clusters = n - i
            if min_clusters <= num_clusters <= max_clusters:
                valid_gaps.append((gap, num_clusters))

        if not valid_gaps:
            return min_clusters

        # Return cluster count with largest gap
        _, optimal_k = max(valid_gaps, key=lambda x: x[0])
        return optimal_k


class HierarchicalClustering:
    """
    Hierarchical clustering for factor families.

    Builds dendrogram from correlation graph using agglomerative clustering.
    Supports multiple linkage methods and provides dendrogram cutting.

    R55 P0-13: this is NOT a production clustering algorithm (it is O(N²) and
    not sparse-graph Leiden).  In ``PRODUCTION`` mode every entry point raises
    :class:`~factor_assets.errors.ProductionClusterViolation`; in ``RESEARCH``
    mode the relaxed path is unchanged.
    """

    #: Not a production-capable algorithm: production runs must use Leiden.
    PRODUCTION_ALGORITHM = None

    def __init__(
        self,
        graph: SparseCorrelationGraph,
        method: str = 'average',
        metric: str = 'correlation',
        max_factors: int = 500,
        missing_distance_policy: Optional[str] = None,
        execution_mode: ExecutionMode | str = ExecutionMode.RESEARCH,
        certification: Optional[CertifiedGraphArtifact] = None,
    ):
        """
        Args:
            graph: Correlation graph to cluster
            method: Linkage method ('single', 'complete', 'average', 'ward')
            metric: Distance metric ('correlation', 'euclidean')
            max_factors: Maximum number of factors (safety limit for O(N²) memory)
            missing_distance_policy: How to handle a pair with NO observed
                distance.  ``None`` (default) FAILS CLOSED — a missing edge is
                NOT treated as distance=1 (that would conflate "not measured"
                with "dissimilar").  Pass ``"treat_missing_as_max"`` to opt
                into the legacy research behaviour explicitly.
            execution_mode: ``RESEARCH`` (default) or ``PRODUCTION``.  In
                production mode this class raises — only Leiden is certified
                for production clustering.
            certification: certification evidence attached to ``graph``.

        Raises:
            ImportError: If scipy is not available
            ValueError: If graph.node_count > max_factors
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for HierarchicalClustering. "
                "Install with: pip install scipy"
            )

        if graph.node_count > max_factors:
            raise ValueError(
                f"Graph has {graph.node_count} factors, exceeds max_factors={max_factors}. "
                f"HierarchicalClustering uses O(N²) dense distance matrix. "
                f"For large graphs, use sparse methods or increase max_factors explicitly."
            )
        if missing_distance_policy not in (None, "treat_missing_as_max"):
            raise ValueError(
                "missing_distance_policy must be None (fail closed) or "
                "'treat_missing_as_max' (research only)"
            )
        if method == "ward":
            raise InvalidClusteringContract(
                "method='ward' requires a real Euclidean feature-space "
                "distance; it is invalid on a correlation-derived distance "
                "(dist = 1 - |corr|). Ward uses squared Euclidean distances "
                "and would produce a meaningless dendrogram here — FAIL CLOSED."
            )
        if metric not in ("correlation", "euclidean"):
            raise ValueError(
                f"metric must be 'correlation' or 'euclidean', got {metric!r}"
            )

        self.graph = graph
        self.method = method
        self.metric = metric
        self.max_factors = max_factors
        self.missing_distance_policy = missing_distance_policy
        self.execution_mode = ExecutionMode.coerce(execution_mode)
        self.certification = certification

    def _gate(self, *, algorithm: str, now: Optional[object] = None):
        """Run the certified-graph gate for this run (R55 P0-13).

        In production mode an uncertified graph — or any algorithm not on the
        certification's allow-list — raises
        :class:`~factor_assets.errors.ProductionClusterViolation`; research
        mode passes through unchanged.
        """
        return enforce_certified_graph(
            self.execution_mode,
            self.graph,
            self.certification,
            algorithm=algorithm,
            now=now,
        )

    def build_dendrogram(self, *, now: Optional[object] = None) -> Dendrogram:
        """
        Build hierarchical clustering dendrogram.

        Args:
            now: optional gate clock (see :func:`~factor_assets.clustering.
                certification.enforce_certified_graph`).

        Returns:
            Dendrogram containing linkage matrix and factor IDs
        """
        # R55 P0-13: the gate runs at EVERY entry point.
        self._gate(algorithm="hierarchical", now=now)
        if self.graph.node_count == 0:
            raise ValueError("Cannot cluster empty graph")

        if self.graph.node_count == 1:
            # Single node: return trivial linkage
            factor_ids = list(self.graph.nodes)
            linkage_matrix = np.zeros((0, 4))
            return Dendrogram(
                linkage_matrix=linkage_matrix,
                factor_ids=factor_ids,
                method=self.method
            )

        # Build distance matrix from correlation graph
        factor_ids = sorted(self.graph.nodes)
        n = len(factor_ids)
        distance_matrix = np.ones((n, n))

        # Convert correlation to distance: dist = 1 - abs(corr)
        for i, fid_a in enumerate(factor_ids):
            distance_matrix[i, i] = 0.0
            for j, fid_b in enumerate(factor_ids):
                if i < j:
                    corr = self.graph.get_correlation(fid_a, fid_b)
                    if corr is not None:
                        dist = 1.0 - abs(corr)
                    else:
                        # A missing edge is NOT evidence of low similarity.
                        # Fail closed unless the caller explicitly opted into
                        # the research-only "treat missing as max distance"
                        # policy.  This prevents UNKNOWN observations from
                        # being silently conflated with COMPUTED_LOW.
                        if self.missing_distance_policy != "treat_missing_as_max":
                            raise ValueError(
                                f"No observed distance between {fid_a!r} and {fid_b!r}; "
                                "a missing edge is not evidence of low similarity. "
                                "Pass missing_distance_policy='treat_missing_as_max' "
                                "to opt into the research-only behaviour, or supply a "
                                "complete distance artifact."
                            )
                        dist = 1.0  # No edge = maximum distance (explicit opt-in only)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        # Convert to condensed form for scipy
        condensed_distances = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(
            condensed_distances,
            method=self.method,
            metric=self.metric if self.metric != 'correlation' else 'euclidean'
        )

        return Dendrogram(
            linkage_matrix=linkage_matrix,
            factor_ids=factor_ids,
            method=self.method
        )

    def cluster(
        self,
        num_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        auto_optimize: bool = False,
        *,
        now: Optional[object] = None,
    ) -> ClusterResult:
        """
        Perform hierarchical clustering with automatic or manual cutting.

        Args:
            num_clusters: Desired number of clusters (exclusive with distance_threshold)
            distance_threshold: Distance threshold for cutting (exclusive with num_clusters)
            auto_optimize: Automatically determine optimal number of clusters
            now: optional gate clock (see :func:`~factor_assets.clustering.
                certification.enforce_certified_graph`).

        Returns:
            ClusterResult with cluster assignments
        """
        # R55 P0-13: the gate runs here (cluster -> build_dendrogram), and
        # build_dendrogram re-runs it, so both entry points are gated.
        self._gate(algorithm="hierarchical", now=now)
        dendrogram = self.build_dendrogram(now=now)

        if dendrogram.linkage_matrix.shape[0] == 0:
            # Single node case
            return ClusterResult(
                assignments={dendrogram.factor_ids[0]: 0},
                cluster_sizes={0: 1}
            )

        # Determine cutting strategy
        if auto_optimize:
            optimal_k = dendrogram.get_optimal_num_clusters()
            return dendrogram.cut_at_num_clusters(optimal_k)
        elif num_clusters is not None:
            return dendrogram.cut_at_num_clusters(num_clusters)
        elif distance_threshold is not None:
            return dendrogram.cut_at_distance(distance_threshold)
        else:
            # Default: use optimal estimation
            optimal_k = dendrogram.get_optimal_num_clusters()
            return dendrogram.cut_at_num_clusters(optimal_k)
