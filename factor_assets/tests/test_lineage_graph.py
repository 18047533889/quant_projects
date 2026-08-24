"""
Test lineage graph and parent-child tracking.
"""

import pytest

from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.registry.lineage import (
    LineageGraph,
    LineageEdge,
    CampaignMetadata,
)


def test_lineage_graph_register_factor_no_parents():
    """Test registering a root factor with no parents."""
    graph = LineageGraph()

    lineage = LineageRef(
        factor_id="F001",
        parents=(),
    )

    graph.register_factor(lineage)

    assert graph.get_parents("F001") == ()
    assert graph.get_children("F001") == ()
    assert graph.is_root("F001")
    assert graph.is_leaf("F001")


def test_lineage_graph_register_factor_with_parent():
    """Test registering a factor with a parent."""
    graph = LineageGraph()

    # Register parent first
    parent_lineage = LineageRef(factor_id="F001", parents=())
    graph.register_factor(parent_lineage)

    # Register child
    child_lineage = LineageRef(
        factor_id="F002",
        parents=(
            ParentRef(
                factor_id="F001",
                relationship="mutation",
                operation="add_lag",
            ),
        ),
    )
    graph.register_factor(child_lineage)

    assert graph.get_parents("F002") == ("F001",)
    assert graph.get_children("F001") == ("F002",)
    assert graph.is_root("F001")
    assert not graph.is_root("F002")
    assert graph.is_leaf("F002")
    assert not graph.is_leaf("F001")


def test_lineage_graph_multiple_parents():
    """Test factor with multiple parents (combination)."""
    graph = LineageGraph()

    # Register parents
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(factor_id="F002", parents=()))

    # Register child with two parents
    child_lineage = LineageRef(
        factor_id="F003",
        parents=(
            ParentRef(factor_id="F001", relationship="combination"),
            ParentRef(factor_id="F002", relationship="combination"),
        ),
    )
    graph.register_factor(child_lineage)

    assert set(graph.get_parents("F003")) == {"F001", "F002"}
    assert graph.get_children("F001") == ("F003",)
    assert graph.get_children("F002") == ("F003",)


def test_lineage_graph_get_edge():
    """Test retrieving edge metadata."""
    graph = LineageGraph()

    graph.register_factor(LineageRef(factor_id="F001", parents=()))

    child_lineage = LineageRef(
        factor_id="F002",
        parents=(
            ParentRef(
                factor_id="F001",
                relationship="mutation",
                mutation_id="mut_123",
                operation="smooth",
            ),
        ),
    )
    graph.register_factor(child_lineage)

    edge = graph.get_edge("F001", "F002")

    assert edge is not None
    assert edge.parent_id == "F001"
    assert edge.child_id == "F002"
    assert edge.relationship == "mutation"
    assert edge.mutation_id == "mut_123"
    assert edge.operation == "smooth"


def test_lineage_graph_get_ancestors():
    """Test getting all ancestors via BFS."""
    graph = LineageGraph()

    # Create chain: F001 -> F002 -> F003 -> F004
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F002", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(ParentRef(factor_id="F003", relationship="derived"),)
    ))

    ancestors = graph.get_ancestors("F004")

    assert set(ancestors) == {"F001", "F002", "F003"}


def test_lineage_graph_get_ancestors_max_depth():
    """Test limiting ancestor depth."""
    graph = LineageGraph()

    # Create chain: F001 -> F002 -> F003 -> F004
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F002", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(ParentRef(factor_id="F003", relationship="derived"),)
    ))

    # Get ancestors with max_depth=2
    ancestors = graph.get_ancestors("F004", max_depth=2)

    assert set(ancestors) == {"F002", "F003"}


def test_lineage_graph_get_descendants():
    """Test getting all descendants via BFS."""
    graph = LineageGraph()

    # Create tree: F001 -> F002, F003
    #              F002 -> F004
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(ParentRef(factor_id="F002", relationship="derived"),)
    ))

    descendants = graph.get_descendants("F001")

    assert set(descendants) == {"F002", "F003", "F004"}


def test_lineage_graph_get_lineage_depth():
    """Test computing lineage depth from root."""
    graph = LineageGraph()

    # Create chain: F001 -> F002 -> F003
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F002", relationship="derived"),)
    ))

    assert graph.get_lineage_depth("F001") == 0
    assert graph.get_lineage_depth("F002") == 1
    assert graph.get_lineage_depth("F003") == 2


def test_lineage_graph_get_lineage_depth_multiple_paths():
    """Test lineage depth with multiple paths (DAG)."""
    graph = LineageGraph()

    # Create diamond: F001 -> F002, F003 -> F004
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(
            ParentRef(factor_id="F002", relationship="combination"),
            ParentRef(factor_id="F003", relationship="combination"),
        )
    ))

    # F004 is at depth 2 (max of paths through F002 and F003)
    assert graph.get_lineage_depth("F004") == 2


def test_lineage_graph_campaign_registration():
    """Test registering and querying campaign metadata."""
    graph = LineageGraph()

    campaign = CampaignMetadata(
        campaign_id="camp_001",
        name="Alpha Search Q1",
        start_timestamp="2026-01-01T00:00:00Z",
        objective="Find momentum factors",
        total_trials=100,
        successful_trials=12,
    )

    graph.register_campaign(campaign)

    retrieved = graph.get_campaign("camp_001")

    assert retrieved is not None
    assert retrieved.campaign_id == "camp_001"
    assert retrieved.name == "Alpha Search Q1"
    assert retrieved.total_trials == 100


def test_lineage_graph_campaign_factor_association():
    """Test associating factors with campaigns."""
    graph = LineageGraph()

    campaign = CampaignMetadata(
        campaign_id="camp_001",
        name="Test Campaign",
        start_timestamp="2026-01-01T00:00:00Z",
    )
    graph.register_campaign(campaign)

    # Register factors from campaign
    lineage1 = LineageRef(
        factor_id="F001",
        parents=(),
        campaign_id="camp_001",
    )
    lineage2 = LineageRef(
        factor_id="F002",
        parents=(),
        campaign_id="camp_001",
    )
    lineage3 = LineageRef(
        factor_id="F003",
        parents=(),
    )

    graph.register_factor(lineage1)
    graph.register_factor(lineage2)
    graph.register_factor(lineage3)

    campaign_factors = graph.get_campaign_factors("camp_001")

    assert set(campaign_factors) == {"F001", "F002"}
    assert graph.get_factor_campaign("F001") == "camp_001"
    assert graph.get_factor_campaign("F002") == "camp_001"
    assert graph.get_factor_campaign("F003") is None


def test_lineage_graph_has_cycle_detection():
    """Test cycle detection in lineage graph."""
    graph = LineageGraph()

    # Register chain: F001 -> F002 -> F003
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))

    # Check that F002 is not its own ancestor (no cycle)
    assert not graph.has_cycle("F002")


def test_lineage_graph_get_roots_and_leaves():
    """Test getting all root and leaf factors."""
    graph = LineageGraph()

    # Create tree: F001, F002 (roots) -> F003 -> F004, F005 (leaves)
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(factor_id="F002", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(
            ParentRef(factor_id="F001", relationship="combination"),
            ParentRef(factor_id="F002", relationship="combination"),
        )
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(ParentRef(factor_id="F003", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F005",
        parents=(ParentRef(factor_id="F003", relationship="derived"),)
    ))

    roots = graph.get_roots()
    leaves = graph.get_leaves()

    assert set(roots) == {"F001", "F002"}
    assert set(leaves) == {"F004", "F005"}


def test_lineage_graph_stats():
    """Test lineage graph statistics."""
    graph = LineageGraph()

    # Empty graph
    stats = graph.stats()
    assert stats["total_factors"] == 0
    assert stats["total_edges"] == 0

    # Add factors
    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F002",
        parents=(ParentRef(factor_id="F001", relationship="derived"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F002", relationship="derived"),)
    ))

    stats = graph.stats()

    assert stats["total_factors"] == 3
    assert stats["total_edges"] == 2
    assert stats["root_factors"] == 1
    assert stats["leaf_factors"] == 1
    assert stats["factors_with_parents"] == 2
    assert stats["factors_with_children"] == 2


def test_lineage_graph_complex_dag():
    """Test complex DAG with multiple paths and combinations."""
    graph = LineageGraph()

    # Build complex DAG
    # Roots: F001, F002
    # F003 = mutation(F001)
    # F004 = mutation(F002)
    # F005 = combination(F001, F002)
    # F006 = combination(F003, F004, F005)

    graph.register_factor(LineageRef(factor_id="F001", parents=()))
    graph.register_factor(LineageRef(factor_id="F002", parents=()))
    graph.register_factor(LineageRef(
        factor_id="F003",
        parents=(ParentRef(factor_id="F001", relationship="mutation"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F004",
        parents=(ParentRef(factor_id="F002", relationship="mutation"),)
    ))
    graph.register_factor(LineageRef(
        factor_id="F005",
        parents=(
            ParentRef(factor_id="F001", relationship="combination"),
            ParentRef(factor_id="F002", relationship="combination"),
        )
    ))
    graph.register_factor(LineageRef(
        factor_id="F006",
        parents=(
            ParentRef(factor_id="F003", relationship="combination"),
            ParentRef(factor_id="F004", relationship="combination"),
            ParentRef(factor_id="F005", relationship="combination"),
        )
    ))

    # F006 should have all factors as ancestors
    ancestors = graph.get_ancestors("F006")
    assert set(ancestors) == {"F001", "F002", "F003", "F004", "F005"}

    # F006 depth should be 2 (longest path)
    assert graph.get_lineage_depth("F006") == 2

    # Roots and leaves
    assert set(graph.get_roots()) == {"F001", "F002"}
    assert set(graph.get_leaves()) == {"F006"}
