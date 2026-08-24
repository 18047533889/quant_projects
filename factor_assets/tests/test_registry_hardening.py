"""
Pinning tests for FA registry hardening:

1. Lifecycle dead-end: APPROVED/DEPRECATED was illegal — deprecation must be
   reachable from every live state.
2. Timestamp mixing: "+00:00" vs "Z" suffixes compare wrongly as raw strings.
3. Lineage cycle: register_factor must refuse edges that close a cycle
   (and roll back partially-registered edges), not silently corrupt the DAG.
"""
import pytest

from factor_assets.contracts.lifecycle import (
    LifecycleState,
    is_legal_transition,
    get_required_evidence,
    StateEvent,
)
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.registry.lineage import LineageGraph
from factor_assets.registry.snapshots import SnapshotManager, SnapshotQuery


class TestDeprecationFromAnyLiveState:
    @pytest.mark.parametrize("state", [
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        LifecycleState.PRODUCTION_READY,
    ])
    def test_deprecation_is_reachable(self, state):
        assert is_legal_transition(state, LifecycleState.DEPRECATED)

    def test_deprecated_to_retired(self):
        assert is_legal_transition(LifecycleState.DEPRECATED, LifecycleState.RETIRED)

    def test_no_forward_transition_after_retired(self):
        assert not is_legal_transition(LifecycleState.RETIRED, LifecycleState.REGISTERED)
        assert not is_legal_transition(LifecycleState.RETIRED, LifecycleState.EVALUATED)

    def test_deprecation_requires_no_evidence(self):
        assert get_required_evidence(
            LifecycleState.APPROVED, LifecycleState.DEPRECATED
        ) == ()


class TestTimestampNormalization:
    def _event(self, factor_id, ts, to_state=LifecycleState.EVALUATED):
        return StateEvent(
            factor_id=factor_id,
            from_state=LifecycleState.REGISTERED,
            to_state=to_state,
            timestamp=ts,
            evidence_refs=("evaluation_bundle_ref",),
        )

    def test_z_suffix_event_seen_by_plus_offset_query(self):
        """Event at ...:00Z; query as_of ...:00+00:00 (same instant).
        Raw-string comparison would treat +00:00 < Z and drop the event."""
        mgr = SnapshotManager()
        events = [self._event("F1", "2026-01-01T00:00:00Z")]
        state = mgr.get_state_at_time("F1", events, "2026-01-01T00:00:00+00:00")
        assert state == LifecycleState.EVALUATED

    def test_mixed_suffix_events_sort_by_instant(self):
        mgr = SnapshotManager()
        events = [
            self._event("F1", "2026-01-02T00:00:00+00:00", to_state=LifecycleState.APPROVED),
            self._event("F1", "2026-01-01T00:00:00Z"),
        ]
        state = mgr.get_state_at_time("F1", events, "2026-01-01T12:00:00Z")
        assert state == LifecycleState.EVALUATED  # earlier event only

    def test_unparseable_timestamp_falls_back_to_raw_comparison(self):
        # Unparseable strings fall back to raw comparison; the function must
        # not raise, and ordering follows lexicographic raw comparison.
        mgr = SnapshotManager()
        events = [self._event("F1", "not-a-timestamp")]
        state = mgr.get_state_at_time("F1", events, "z" * 20)
        assert state == LifecycleState.EVALUATED
        state = mgr.get_state_at_time("F1", events, "0000")
        assert state == LifecycleState.REGISTERED


class TestLineageCycleGuard:
    def test_self_parent_rejected(self):
        graph = LineageGraph()
        lineage = LineageRef(
            factor_id="F1",
            parents=(ParentRef(factor_id="F1", relationship="mutation"),),
        )
        with pytest.raises(ValueError, match="cannot list itself"):
            graph.register_factor(lineage)

    def test_cycle_closing_edge_rejected_and_rolled_back(self):
        # C registered with parent B; then B with parent C closes B->C->B.
        graph = LineageGraph()
        graph.register_factor(
            LineageRef(factor_id="C", parents=(ParentRef(factor_id="B", relationship="mutation"),))
        )
        with pytest.raises(ValueError, match="cycle"):
            graph.register_factor(
                LineageRef(factor_id="B", parents=(ParentRef(factor_id="C", relationship="mutation"),))
            )
        # Rollback: B's parent set must be empty; C must not list B as a child
        assert graph.get_parents("B") == ()
        assert "B" not in graph.get_children("C")

    def test_valid_dag_still_registers(self):
        graph = LineageGraph()
        graph.register_factor(LineageRef(factor_id="A", parents=()))
        graph.register_factor(
            LineageRef(factor_id="B", parents=(ParentRef(factor_id="A", relationship="mutation"),))
        )
        graph.register_factor(
            LineageRef(factor_id="C", parents=(ParentRef(factor_id="B", relationship="mutation"),))
        )
        assert graph.get_ancestors("C") == ("B", "A")
        assert not graph.has_cycle("C")
