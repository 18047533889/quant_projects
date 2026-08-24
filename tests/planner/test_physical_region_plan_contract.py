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


def _edge(producer: BackendRegion, consumer: BackendRegion) -> TransferEdge:
    return TransferEdge(
        edge_id="edge",
        producer_region=producer.region_id,
        consumer_region=consumer.region_id,
        source_backend=producer.backend,
        target_backend=consumer.backend,
        source_representation=producer.representation,
        target_representation=consumer.representation,
        estimated_rows=1,
        estimated_bytes=8,
        estimated_transfer_ms=1.0,
    )


def _plan(regions: tuple[BackendRegion, ...], edges: tuple[TransferEdge, ...], switches: int) -> PhysicalRegionPlan:
    return PhysicalRegionPlan(
        plan_id="contract",
        regions=regions,
        edges=edges,
        topological_order=tuple(region.region_id for region in regions),
        root_region_ids=(regions[-1].region_id,),
        total_compute_ms=1.0,
        total_transfer_ms=1.0 if edges else 0.0,
        backend_switch_count=switches,
    )


def test_plan_rejects_edge_with_wrong_source_residency() -> None:
    producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
    consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
    edge = _edge(producer, consumer)
    invalid = TransferEdge(
        **{**edge.__dict__, "source_backend": PhysicalBackend.DUCKDB_SQL}
    )

    with pytest.raises(ValueError, match="source residency"):
        _plan((producer, consumer), (invalid,), 1)


def test_plan_rejects_edge_that_runs_backwards() -> None:
    producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
    consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
    edge = TransferEdge(
        **{
            **_edge(producer, consumer).__dict__,
            "producer_region": "c",
            "consumer_region": "p",
            "source_backend": consumer.backend,
            "source_representation": consumer.representation,
            "target_backend": producer.backend,
            "target_representation": producer.representation,
        }
    )

    with pytest.raises(ValueError, match="topological_order"):
        _plan((producer, consumer), (edge,), 1)


def test_plan_counts_only_backend_changing_edges() -> None:
    producer = _region("p", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)
    consumer = _region("c", PhysicalBackend.PANDAS_NUMPY, Representation.PANDAS_LONG)

    with pytest.raises(ValueError, match="backend_switch_count"):
        _plan((producer, consumer), (_edge(producer, consumer),), 1)


def test_plan_accepts_explicit_duckdb_arrow_boundary() -> None:
    producer = _region("p", PhysicalBackend.DUCKDB_SQL, Representation.DUCKDB_RELATION)
    consumer = _region("c", PhysicalBackend.POLARS_LONG, Representation.POLARS_LAZY_LONG)
    edge = TransferEdge(
        **{
            **_edge(producer, consumer).__dict__,
            "source_representation": Representation.ARROW_TABLE,
        }
    )

    plan = _plan((producer, consumer), (edge,), 1)
    assert plan.edges[0].source_representation is Representation.ARROW_TABLE
