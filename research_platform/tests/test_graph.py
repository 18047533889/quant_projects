from research_platform.graph import ArtifactGraph, DataInvalidationGraph


def test_lineage_ancestors():
    g = ArtifactGraph()
    g.add("factor", "snap")
    g.add("eval", "factor")
    g.add("model", "eval")
    assert g.lineage("model") == ["eval", "factor", "snap"]
    assert g.lineage("snap") == []


def test_dependents_downstream():
    g = ArtifactGraph()
    g.add("factor", "snap")
    g.add("eval", "factor")
    g.add("model", "eval")
    assert g.dependents("snap") == ["snap", "factor", "eval", "model"]
    assert g.dependents("eval") == ["eval", "model"]


def test_multi_source():
    g = ArtifactGraph()
    g.add("combo", "a", "b")
    assert set(g.lineage("combo")) == {"a", "b"}


def test_invalidation_via_snapshot_reverse_lookup():
    g = DataInvalidationGraph()
    g.add("factor", snapshot_ref="S1")
    g.add("eval", "S1", "factor")
    g.add("model", "S1", "eval")
    g.add("other", snapshot_ref="S2")  # unrelated
    affected = g.mark_invalid("S1")
    assert set(affected) == {"factor", "eval", "model"}
    assert "other" not in affected
    # ancestors-first order
    assert affected.index("factor") < affected.index("eval")
    assert affected.index("eval") < affected.index("model")


def test_cycle_is_handled():
    g = ArtifactGraph()
    g.add("a", "b")
    g.add("b", "a")
    # no infinite recursion; lineage is finite
    assert set(g.lineage("a")) == {"b"}
