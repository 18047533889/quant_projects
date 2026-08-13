"""Tests for mutation lineage tracking."""

import pytest
from datetime import datetime

from factor_optimizer.search.lineage import (
    LineageNode,
    LineageTree,
    LineageAnalyzer,
)


def test_lineage_node_creation():
    node = LineageNode(
        trial_id="t1",
        parent_ids=["t0"],
        mutation_type="parameter_tune",
        generation=1,
        score=0.8,
    )

    assert node.trial_id == "t1"
    assert node.parent_ids == ["t0"]
    assert node.mutation_type == "parameter_tune"
    assert node.generation == 1
    assert node.score == 0.8
    assert not node.is_seed()


def test_lineage_node_seed():
    node = LineageNode(trial_id="seed1", generation=0)

    assert node.is_seed()
    assert len(node.parent_ids) == 0


def test_lineage_node_serialization():
    node = LineageNode(
        trial_id="t1",
        parent_ids=["t0"],
        mutation_type="operator_swap",
        generation=2,
        score=0.75,
        metadata={"cost": 10.0},
    )

    data = node.to_dict()

    assert data["trial_id"] == "t1"
    assert data["parent_ids"] == ["t0"]
    assert data["mutation_type"] == "operator_swap"
    assert data["generation"] == 2
    assert data["score"] == 0.75

    # Deserialize
    restored = LineageNode.from_dict(data)
    assert restored.trial_id == "t1"
    assert restored.score == 0.75


def test_lineage_tree_initialization():
    tree = LineageTree()

    assert len(tree.nodes) == 0
    assert len(tree.children) == 0


def test_lineage_tree_add_seed():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    assert len(tree.nodes) == 1
    assert tree.get_node("seed1") is not None


def test_lineage_tree_add_child():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    child = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(child)

    assert len(tree.nodes) == 2
    assert tree.get_node("t1") is not None

    children = tree.get_children("seed1")
    assert len(children) == 1
    assert children[0].trial_id == "t1"


def test_lineage_tree_add_duplicate():
    tree = LineageTree()

    node = LineageNode(trial_id="t1", generation=0)
    tree.add_node(node)

    with pytest.raises(ValueError, match="already exists"):
        tree.add_node(node)


def test_lineage_tree_add_missing_parent():
    tree = LineageTree()

    child = LineageNode(trial_id="t1", parent_ids=["missing"], generation=1)

    with pytest.raises(ValueError, match="not in tree"):
        tree.add_node(child)


def test_lineage_tree_get_parents():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    child = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(child)

    parents = tree.get_parents("t1")
    assert len(parents) == 1
    assert parents[0].trial_id == "seed1"


def test_lineage_tree_get_ancestors():
    tree = LineageTree()

    # Build a chain: seed1 -> t1 -> t2
    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2)
    tree.add_node(t2)

    ancestors = tree.get_ancestors("t2")
    assert ancestors == {"seed1", "t1"}


def test_lineage_tree_get_descendants():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["seed1"], generation=1)
    tree.add_node(t2)

    descendants = tree.get_descendants("seed1")
    assert descendants == {"t1", "t2"}


def test_lineage_tree_get_seeds():
    tree = LineageTree()

    seed1 = LineageNode(trial_id="seed1", generation=0)
    seed2 = LineageNode(trial_id="seed2", generation=0)
    tree.add_node(seed1)
    tree.add_node(seed2)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    seeds = tree.get_seeds()
    assert len(seeds) == 2
    assert {s.trial_id for s in seeds} == {"seed1", "seed2"}


def test_lineage_tree_get_leaves():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2)
    tree.add_node(t2)

    leaves = tree.get_leaves()
    assert len(leaves) == 1
    assert leaves[0].trial_id == "t2"


def test_lineage_tree_depth():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2)
    tree.add_node(t2)

    assert tree.depth("seed1") == 0
    assert tree.depth("t1") == 1
    assert tree.depth("t2") == 2


def test_lineage_tree_path_to_seed():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2)
    tree.add_node(t2)

    path = tree.path_to_seed("t2")
    assert path == ["seed1", "t1", "t2"]


def test_lineage_tree_best_in_lineage():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1, score=0.7)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2, score=0.6)
    tree.add_node(t2)

    best = tree.best_in_lineage("t2")
    assert best is not None
    assert best.trial_id == "t1"
    assert best.score == 0.7


def test_lineage_tree_subtree_stats():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1, score=0.6)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["seed1"], generation=1, score=0.7)
    tree.add_node(t2)

    stats = tree.subtree_stats("seed1")

    assert stats["num_descendants"] == 2
    assert stats["num_scored"] == 2
    assert stats["best_score"] == 0.7
    assert abs(stats["mean_score"] - 0.65) < 0.001


def test_lineage_tree_prune_lineage_keep_ancestors():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["t1"], generation=2)
    tree.add_node(t2)

    removed = tree.prune_lineage("t1", keep_ancestors=True)

    assert removed == 2  # t1 and t2
    assert tree.get_node("seed1") is not None
    assert tree.get_node("t1") is None
    assert tree.get_node("t2") is None


def test_lineage_tree_prune_lineage_remove_all():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    removed = tree.prune_lineage("t1", keep_ancestors=False)

    assert removed == 2  # seed1 and t1
    assert tree.get_node("seed1") is None
    assert tree.get_node("t1") is None


def test_lineage_tree_serialization():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1, score=0.6)
    tree.add_node(t1)

    data = tree.to_dict()

    assert "nodes" in data
    assert len(data["nodes"]) == 2

    # Deserialize
    restored = LineageTree.from_dict(data)
    assert len(restored.nodes) == 2
    assert restored.get_node("seed1") is not None
    assert restored.get_node("t1") is not None


def test_lineage_analyzer_initialization():
    tree = LineageTree()
    analyzer = LineageAnalyzer(tree)

    assert analyzer.tree is tree


def test_lineage_analyzer_success_rate_by_mutation_type():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    # Successful mutation
    t1 = LineageNode(
        trial_id="t1",
        parent_ids=["seed1"],
        mutation_type="parameter_tune",
        generation=1,
        score=0.7,
    )
    tree.add_node(t1)

    # Failed mutation
    t2 = LineageNode(
        trial_id="t2",
        parent_ids=["seed1"],
        mutation_type="parameter_tune",
        generation=1,
        score=0.4,
    )
    tree.add_node(t2)

    analyzer = LineageAnalyzer(tree)
    rates = analyzer.success_rate_by_mutation_type()

    assert "parameter_tune" in rates
    assert rates["parameter_tune"] == 0.5  # 1 success out of 2


def test_lineage_analyzer_generation_statistics():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1, score=0.6)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["seed1"], generation=1, score=0.8)
    tree.add_node(t2)

    analyzer = LineageAnalyzer(tree)
    stats = analyzer.generation_statistics()

    assert 0 in stats
    assert 1 in stats
    assert stats[1]["count"] == 2
    assert stats[1]["best_score"] == 0.8
    assert stats[1]["mean_score"] == 0.7


def test_lineage_analyzer_most_productive_lineage():
    tree = LineageTree()

    seed1 = LineageNode(trial_id="seed1", generation=0, score=0.5)
    tree.add_node(seed1)

    seed2 = LineageNode(trial_id="seed2", generation=0, score=0.4)
    tree.add_node(seed2)

    # seed1 has more descendants
    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1, score=0.6)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["seed1"], generation=1, score=0.7)
    tree.add_node(t2)

    t3 = LineageNode(trial_id="t3", parent_ids=["seed2"], generation=1, score=0.5)
    tree.add_node(t3)

    analyzer = LineageAnalyzer(tree)
    most_productive = analyzer.most_productive_lineage()

    assert most_productive == "seed1"


def test_lineage_analyzer_diversity_score():
    tree = LineageTree()

    seed1 = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed1)

    seed2 = LineageNode(trial_id="seed2", generation=0)
    tree.add_node(seed2)

    # Balanced descendants
    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    t2 = LineageNode(trial_id="t2", parent_ids=["seed2"], generation=1)
    tree.add_node(t2)

    analyzer = LineageAnalyzer(tree)
    diversity = analyzer.diversity_score()

    assert 0 <= diversity <= 1


def test_lineage_analyzer_diversity_single_seed():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    analyzer = LineageAnalyzer(tree)
    diversity = analyzer.diversity_score()

    assert diversity == 0.0


def test_lineage_tree_multiple_parents():
    tree = LineageTree()

    seed1 = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed1)

    seed2 = LineageNode(trial_id="seed2", generation=0)
    tree.add_node(seed2)

    # Composition with two parents
    composition = LineageNode(
        trial_id="comp1",
        parent_ids=["seed1", "seed2"],
        mutation_type="composition",
        generation=1,
    )
    tree.add_node(composition)

    parents = tree.get_parents("comp1")
    assert len(parents) == 2
    assert {p.trial_id for p in parents} == {"seed1", "seed2"}


def test_lineage_tree_get_node_missing():
    tree = LineageTree()

    node = tree.get_node("missing")
    assert node is None


def test_lineage_tree_depth_missing():
    tree = LineageTree()

    depth = tree.depth("missing")
    assert depth == -1


def test_lineage_tree_path_to_seed_missing():
    tree = LineageTree()

    path = tree.path_to_seed("missing")
    assert path == []


def test_lineage_analyzer_most_productive_no_seeds():
    tree = LineageTree()
    analyzer = LineageAnalyzer(tree)

    most_productive = analyzer.most_productive_lineage()
    assert most_productive is None


def test_lineage_tree_best_in_lineage_no_scores():
    tree = LineageTree()

    seed = LineageNode(trial_id="seed1", generation=0)
    tree.add_node(seed)

    t1 = LineageNode(trial_id="t1", parent_ids=["seed1"], generation=1)
    tree.add_node(t1)

    best = tree.best_in_lineage("t1")
    assert best is None
