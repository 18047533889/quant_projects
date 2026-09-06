from __future__ import annotations

import pytest

from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalRegionPlan,
    Representation,
    StateContract,
    TransferEdge,
    TransferTransform,
)


def _region(region_id: str, backend: PhysicalBackend, representation: Representation) -> BackendRegion:
    return BackendRegion(
        region_id=region_id,
        backend=backend,
        representation=representation,
        node_ids=(region_id + "_node",),
        execution_axis=ExecutionAxis.RELATIONAL,
        estimated_rows=1,
        estimated_compute_ms=1.0,
        estimated_memory_bytes=8,
        state_contract=StateContract(),
    )


def _edge(edge_id: str, producer: BackendRegion, consumer: BackendRegion) -> TransferEdge:
    if producer.backend == consumer.backend and producer.representation == consumer.representation:
        transform = TransferTransform.SAME_BACKEND_NATIVE
    elif (
        producer.representation is Representation.PANDAS_LONG
        and consumer.representation is Representation.POLARS_LAZY_LONG
    ):
        transform = TransferTransform.PANDAS_TO_POLARS
    elif (
        producer.representation is Representation.POLARS_LAZY_LONG
        and consumer.representation is Representation.NUMPY_PANEL
    ):
        transform = TransferTransform.POLARS_TO_NUMPY
    else:
        raise AssertionError("fixture requires an explicit supported transfer transform")
    return TransferEdge(
        edge_id=edge_id,
        producer_region=producer.region_id,
        consumer_region=consumer.region_id,
        source_backend=producer.backend,
        target_backend=consumer.backend,
        source_representation=producer.representation,
        target_representation=consumer.representation,
        transform=transform,
        estimated_rows=1,
        estimated_bytes=8,
        estimated_transfer_ms=1.0,
    )


def _plan(regions: tuple[BackendRegion, ...], edges: tuple[TransferEdge, ...], switches: int) -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="test_plan",
        regions=regions,
        edges=edges,
        topological_order=tuple(region.region_id for region in regions),
        root_region_ids=(regions[-1].region_id,) if regions else (),
        total_compute_ms=1.0,
        total_transfer_ms=1.0 if edges else 0.0,
        backend_switch_count=switches,
    )


class TestUniqueRegionIDs:
    """Validate that region IDs must be unique within a plan."""

    def test_rejects_duplicate_region_ids(self) -> None:
        r1 = _region("duplicate", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        r2 = _region("duplicate", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)

        with pytest.raises(ValueError, match="duplicate region IDs"):
            _plan((r1, r2), (), 0)

    def test_accepts_unique_region_ids(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        r2 = _region("r2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)

        plan = _plan((r1, r2), (), 0)
        assert len(plan.regions) == 2


class TestUniqueEdgeIDs:
    """Validate that edge IDs must be unique within a plan."""

    def test_rejects_duplicate_edge_ids(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge1 = _edge("dup_edge", producer, consumer)
        edge2 = _edge("dup_edge", producer, consumer)

        with pytest.raises(ValueError, match="duplicate edge ID"):
            _plan((producer, consumer), (edge1, edge2), 1)

    def test_accepts_unique_edge_ids(self) -> None:
        """Multiple edges with unique IDs are allowed."""
        p1 = _region("p1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        p2 = _region("p2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        p3 = _region("p3", PhysicalBackend.PANDAS_NUMPY, Representation.NUMPY_PANEL)
        edge1 = _edge("edge1", p1, p2)
        edge2 = _edge("edge2", p2, p3)

        plan = _plan((p1, p2, p3), (edge1, edge2), 2)
        assert len(plan.edges) == 2


class TestTopologicalOrder:
    """Validate topological_order contains every region exactly once."""

    def test_rejects_missing_region_in_topo_order(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        r2 = _region("r2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)

        with pytest.raises(ValueError, match="topological_order must contain every region"):
            PhysicalRegionPlan(
                plan_id="missing_topo",
                regions=(r1, r2),
                edges=(),
                topological_order=("r1",),  # Missing r2
                root_region_ids=("r2",),
                total_compute_ms=1.0,
                total_transfer_ms=0.0,
                backend_switch_count=0,
            )

    def test_rejects_extra_region_in_topo_order(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)

        with pytest.raises(ValueError, match="topological_order must contain every region"):
            PhysicalRegionPlan(
                plan_id="extra_topo",
                regions=(r1,),
                edges=(),
                topological_order=("r1", "r2"),  # r2 doesn't exist
                root_region_ids=("r1",),
                total_compute_ms=1.0,
                total_transfer_ms=0.0,
                backend_switch_count=0,
            )

    def test_rejects_duplicate_in_topo_order(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)

        with pytest.raises(ValueError, match="topological_order contains duplicate"):
            PhysicalRegionPlan(
                plan_id="dup_topo",
                regions=(r1,),
                edges=(),
                topological_order=("r1", "r1"),
                root_region_ids=("r1",),
                total_compute_ms=1.0,
                total_transfer_ms=0.0,
                backend_switch_count=0,
            )

    def test_accepts_valid_topo_order(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        r2 = _region("r2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)

        plan = _plan((r1, r2), (), 0)
        assert plan.topological_order == ("r1", "r2")


class TestRootRegionIDs:
    """Validate root_region_ids reference valid regions."""

    def test_rejects_unknown_root_region(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)

        with pytest.raises(ValueError, match="root_region_ids contains an unknown region"):
            PhysicalRegionPlan(
                plan_id="unknown_root",
                regions=(r1,),
                edges=(),
                topological_order=("r1",),
                root_region_ids=("unknown",),
                total_compute_ms=1.0,
                total_transfer_ms=0.0,
                backend_switch_count=0,
            )

    def test_accepts_valid_root_region(self) -> None:
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)

        plan = PhysicalRegionPlan(
            plan_id="valid_root",
            regions=(r1,),
            edges=(),
            topological_order=("r1",),
            root_region_ids=("r1",),
            total_compute_ms=1.0,
            total_transfer_ms=0.0,
            backend_switch_count=0,
        )
        assert plan.root_region_ids == ("r1",)


class TestEdgeRegionReferences:
    """Validate edges reference valid producer and consumer regions."""

    def test_rejects_edge_with_unknown_producer(self) -> None:
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region="unknown_producer",
            consumer_region=consumer.region_id,
            source_backend=PhysicalBackend.PANDAS_NUMPY,
            target_backend=consumer.backend,
            source_representation=Representation.PANDAS_LONG,
            target_representation=consumer.representation,
            transform=TransferTransform.PANDAS_TO_POLARS,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="references an unknown region"):
            _plan((consumer,), (edge,), 1)

    def test_rejects_edge_with_unknown_consumer(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region=producer.region_id,
            consumer_region="unknown_consumer",
            source_backend=producer.backend,
            target_backend=PhysicalBackend.POLARS_LONG,
            source_representation=producer.representation,
            target_representation=Representation.POLARS_LAZY_LONG,
            transform=TransferTransform.PANDAS_TO_POLARS,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="references an unknown region"):
            _plan((producer,), (edge,), 1)

    def test_rejects_self_referential_edge(self) -> None:
        region = _region("r", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        edge = TransferEdge(
            edge_id="self_edge",
            producer_region=region.region_id,
            consumer_region=region.region_id,
            source_backend=region.backend,
            target_backend=region.backend,
            source_representation=region.representation,
            target_representation=region.representation,
            transform=TransferTransform.SAME_BACKEND_NATIVE,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="cannot be self-referential"):
            _plan((region,), (edge,), 0)

    def test_accepts_valid_edge_references(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = _edge("edge", producer, consumer)

        plan = _plan((producer, consumer), (edge,), 1)
        assert len(plan.edges) == 1


class TestEdgeResidencyMatch:
    """Validate edge residency matches producer and consumer regions."""

    def test_rejects_wrong_source_backend(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region=producer.region_id,
            consumer_region=consumer.region_id,
            source_backend=PhysicalBackend.DUCKDB_SQL,  # Wrong: should be PANDAS_NUMPY
            target_backend=consumer.backend,
            source_representation=producer.representation,
            target_representation=consumer.representation,
            transform=TransferTransform.PANDAS_TO_POLARS,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="source residency"):
            _plan((producer, consumer), (edge,), 1)

    def test_rejects_wrong_target_backend(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region=producer.region_id,
            consumer_region=consumer.region_id,
            source_backend=producer.backend,
            target_backend=PhysicalBackend.DUCKDB_SQL,  # Wrong: should be POLARS_LONG
            source_representation=producer.representation,
            target_representation=consumer.representation,
            transform=TransferTransform.PANDAS_TO_POLARS,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="target residency does not match consumer"):
            _plan((producer, consumer), (edge,), 1)

    def test_rejects_wrong_target_representation(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region=producer.region_id,
            consumer_region=consumer.region_id,
            source_backend=producer.backend,
            target_backend=consumer.backend,
            source_representation=producer.representation,
            target_representation=Representation.POLARS_WIDE,  # Wrong
            transform=TransferTransform.PANDAS_TO_POLARS,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        with pytest.raises(ValueError, match="target residency does not match consumer"):
            _plan((producer, consumer), (edge,), 1)

    def test_accepts_duckdb_arrow_boundary(self) -> None:
        """MB-P1-018: DuckDB → Arrow boundary is allowed."""
        producer = _region("p", PhysicalBackend.DUCKDB_SQL, Representation.DUCKDB_RELATION)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = TransferEdge(
            edge_id="edge",
            producer_region=producer.region_id,
            consumer_region=consumer.region_id,
            source_backend=PhysicalBackend.DUCKDB_SQL,
            target_backend=consumer.backend,
            source_representation=Representation.ARROW_TABLE,  # Boundary form
            target_representation=consumer.representation,
            transform=TransferTransform.DUCKDB_TO_ARROW,
            estimated_rows=1,
            estimated_bytes=8,
            estimated_transfer_ms=1.0,
        )

        plan = _plan((producer, consumer), (edge,), 1)
        assert plan.edges[0].source_representation == Representation.ARROW_TABLE


class TestTopologicalEdgeDirection:
    """Validate edges follow topological order (producer before consumer)."""

    def test_rejects_backward_edge(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        # Edge direction is correct, but topological order is wrong
        edge = _edge("edge", producer, consumer)

        with pytest.raises(ValueError, match="topological_order"):
            PhysicalRegionPlan(
                plan_id="backward",
                regions=(producer, consumer),
                edges=(edge,),
                topological_order=("c", "p"),  # Wrong order
                root_region_ids=("p",),
                total_compute_ms=1.0,
                total_transfer_ms=1.0,
                backend_switch_count=1,
            )

    def test_accepts_forward_edge(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = _edge("edge", producer, consumer)

        plan = _plan((producer, consumer), (edge,), 1)
        assert plan.topological_order == ("p", "c")


class TestBackendSwitchCount:
    """Validate backend_switch_count matches actual backend-changing edges."""

    def test_rejects_incorrect_switch_count_same_backend(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        edge = _edge("edge", producer, consumer)

        with pytest.raises(ValueError, match="backend_switch_count"):
            _plan((producer, consumer), (edge,), 1)  # Claims 1 switch but none happened

    def test_rejects_incorrect_switch_count_different_backend(self) -> None:
        producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        edge = _edge("edge", producer, consumer)

        with pytest.raises(ValueError, match="backend_switch_count"):
            _plan((producer, consumer), (edge,), 0)  # Claims 0 switches but one happened

    def test_counts_multiple_backend_switches(self) -> None:
        p1 = _region("p1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        p2 = _region("p2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
        p3 = _region("p3", PhysicalBackend.PANDAS_NUMPY, Representation.NUMPY_PANEL)

        e1 = _edge("e1", p1, p2)
        e2 = _edge("e2", p2, p3)

        plan = _plan((p1, p2, p3), (e1, e2), 2)
        assert plan.backend_switch_count == 2

    def test_accepts_zero_switches_same_backend(self) -> None:
        p1 = _region("p1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        p2 = _region("p2", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        edge = _edge("edge", p1, p2)

        plan = _plan((p1, p2), (edge,), 0)
        assert plan.backend_switch_count == 0


class TestEmptyAndSingleRegionPlans:
    """Validate edge cases: empty plans and single-region plans."""

    def test_accepts_empty_plan(self) -> None:
        plan = PhysicalRegionPlan(
            plan_id="empty",
            regions=(),
            edges=(),
            topological_order=(),
            root_region_ids=(),
            total_compute_ms=0.0,
            total_transfer_ms=0.0,
            backend_switch_count=0,
        )
        assert len(plan.regions) == 0
        assert len(plan.edges) == 0

    def test_accepts_single_region_no_edges(self) -> None:
        region = _region("single", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        plan = _plan((region,), (), 0)
        assert len(plan.regions) == 1
        assert len(plan.edges) == 0
        assert plan.backend_switch_count == 0

    def test_accepts_multiple_regions_no_edges(self) -> None:
        """Multiple disconnected regions (parallel execution)."""
        r1 = _region("r1", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
        r2 = _region("r2", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)

        plan = PhysicalRegionPlan(
            plan_id="parallel",
            regions=(r1, r2),
            edges=(),
            topological_order=("r1", "r2"),
            root_region_ids=("r1", "r2"),
            total_compute_ms=2.0,
            total_transfer_ms=0.0,
            backend_switch_count=0,
        )
        assert len(plan.regions) == 2
        assert len(plan.edges) == 0
