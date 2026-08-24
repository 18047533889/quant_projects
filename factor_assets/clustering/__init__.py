"""
Clustering module for factor family detection.

Connected components and modularity-based community detection
for identifying factor families and subgroups. Hierarchical clustering
with dendrogram cutting for hierarchical family structures.
"""

from factor_assets.clustering.families import (
   
        ConnectedComponents,
    ModularityClustering,
    ClusterResult,
    SimilarityObservationState,
)

try:
    from factor_assets.clustering.families import (
       
        HierarchicalClustering,
        Dendrogram,
        LeidenClustering,
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    HierarchicalClustering = None
    Dendrogram = None
    LeidenClustering = None

from factor_assets.clustering.lineage import (
   
        LineageDetector,
    ParentChildRelation,
    FamilyLineage,
)

__all__ = [
    "ConnectedComponents",
    "ModularityClustering",
    "ClusterResult",
    "SimilarityObservationState",
    "HierarchicalClustering",
    "Dendrogram",
    "LeidenClustering",
    "LineageDetector",
    "ParentChildRelation",
    "FamilyLineage",
    "HIERARCHICAL_AVAILABLE",
]
