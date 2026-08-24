from __future__ import annotations

"""R21-PLAN-HASH-STRENGTHEN regression tests.

PhysicalRegionPlan.compute_plan_hash must bind ALL physical plan semantics:
two plans that would execute differently must produce different hashes, and
the same plan must always produce the same hash.  The pre-strengthen hash
only covered region_id/backend/node_count and edge producer/consumer/
source_repr/target_repr, so e.g. two regions with identical node counts but
different node IDs collided.
"""
import copy

from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalProperties,
    PhysicalRegionPlan,
    Representation,
    StateContract,
    TransferEdge,
)


def _region(**overrides: object) -> BackendRegion:
    base: dict[str, object] = {
        "region_id": "r1",
        "backend": PhysicalBackend.POLARS_PANEL,
        "representation": Representation.POLARS_LONG,
        "node_ids": ("n1", "n2"),
        "execution_axis": ExecutionAxis.TIME_PER_INSTRUMENT,
        "estimated_rows": 1000,
        "estimated_compute_ms": 10.0,
        "estimated_memory_bytes": 8000,
    }
    base.update(overrides)
    return BackendRegion(**base)  # type: ignore[arg-type]


def _edge(**overrides: object) -> TransferEdge:
    base: dict[str, object] = {
        "edge_id": "e1",
        "producer_region": "r1",
        "consumer_region": "r2",
        "source_backend": PhysicalBackend.POLARS_PANEL,
        "target_backend": PhysicalBackend.DUCKDB_SQL,
        "source_representation": Representation.POLARS_LONG,
        "target_representation": Representation.DUCKDB_RELATION,
        "estimated_rows": 1000,
        "estimated_bytes": 8000,
        "estimated_transfer_ms": 5.0,
    }
    base.update(overrides)
    return TransferEdge(**base)  # type: ignore[arg-type]


class TestSamePlanSameHash:
    """Regression 1: identical physical plans hash identically."""

    def test_same_plan_produces_same_hash(self) -> None:
        region = _region(
            implementation_id="pi:v1:abc",
            liveness="lazy",
            parameter_domain_identity="pd:ts_mean:window:int:min=2",
            source_snapshot="snap_001",
            universe_contract="CSI300",
            grain="1d",
            available_at="eod",
            pit_safe=True,
            required_properties=PhysicalProperties(sorted_by=("datetime",)),
            output_properties=PhysicalProperties(
                sorted_by=("datetime",), partitioned_by=("instrument",)
            ),
            state_contract=StateContract(
                requires_checkpoint=False,
                checkpoint_seed=None,
                stateful_operators=("ema",),
                sequential_only=True,
            ),
        )
        edge = _edge(
            requires_sort=True,
            requires_repartition=False,
            requires_reshape=False,
            requires_dtype_cast=True,
            producer_sorted_by=("datetime",),
            producer_guarantees_order=True,
            preserves_pit=True,
            preserves_universe=True,
            preserves_grain=True,
            source_snapshot_id="snap_001",
        )

        hash1 = PhysicalRegionPlan.compute_plan_hash(
            (region,), (edge,), "logical-dag-hash"
        )
        hash2 = PhysicalRegionPlan.compute_plan_hash(
            (region,), (edge,), "logical-dag-hash"
        )
        # Also via independently-constructed (structurally equal) objects.
        region2 = _region(
            implementation_id="pi:v1:abc",
            liveness="lazy",
            parameter_domain_identity="pd:ts_mean:window:int:min=2",
            source_snapshot="snap_001",
            universe_contract="CSI300",
            grain="1d",
            available_at="eod",
            pit_safe=True,
            required_properties=PhysicalProperties(sorted_by=("datetime",)),
            output_properties=PhysicalProperties(
                sorted_by=("datetime",), partitioned_by=("instrument",)
            ),
            state_contract=StateContract(
                requires_checkpoint=False,
                checkpoint_seed=None,
                stateful_operators=("ema",),
                sequential_only=True,
            ),
        )
        edge2 = _edge(
            requires_sort=True,
            requires_repartition=False,
            requires_reshape=False,
            requires_dtype_cast=True,
            producer_sorted_by=("datetime",),
            producer_guarantees_order=True,
            preserves_pit=True,
            preserves_universe=True,
            preserves_grain=True,
            source_snapshot_id="snap_001",
        )
        hash3 = PhysicalRegionPlan.compute_plan_hash(
            (region2,), (edge2,), "logical-dag-hash"
        )

        assert hash1 == hash2
        assert hash1 == hash3
        assert len(hash1) == 64  # full sha256 hex digest


class TestDifferentPlansDifferentHash:
    """Regression 2: plans differing in any physical semantics must not collide."""

    def _base(self) -> BackendRegion:
        return _region(
            implementation_id="pi:v1:base",
            liveness="eager",
            parameter_domain_identity="pd:v1",
            source_snapshot="snap_1",
            universe_contract="CSI300",
            grain="1d",
            available_at="eod",
            required_properties=PhysicalProperties(sorted_by=("datetime",)),
            state_contract=StateContract(),
        )

    def test_different_logical_dag_hash(self) -> None:
        region = self._base()
        h1 = PhysicalRegionPlan.compute_plan_hash((region,), (), "logical-a")
        h2 = PhysicalRegionPlan.compute_plan_hash((region,), (), "logical-b")
        assert h1 != h2

    def test_different_node_ids_same_node_count(self) -> None:
        # The old hash only used len(node_ids) — this was a silent collision.
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(node_ids=("n1", "n2")),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(node_ids=("n1", "n3")),), (), "logical"
        )
        assert h1 != h2

    def test_different_representation(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(representation=Representation.POLARS_LONG),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(representation=Representation.POLARS_WIDE),), (), "logical"
        )
        assert h1 != h2

    def test_different_execution_axis(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT),
            ),
            (),
            "logical",
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(execution_axis=ExecutionAxis.CROSS_SECTION_PER_DATE),
            ),
            (),
            "logical",
        )
        assert h1 != h2

    def test_different_implementation_id(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(implementation_id="pi:v1:aaa"),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(implementation_id="pi:v1:bbb"),), (), "logical"
        )
        assert h1 != h2

    def test_different_liveness(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(liveness="eager"),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(liveness="lazy"),), (), "logical"
        )
        assert h1 != h2

    def test_different_parameter_domain_identity(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(parameter_domain_identity="pd:ts_mean:window=5"),),
            (),
            "logical",
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(parameter_domain_identity="pd:ts_mean:window=20"),),
            (),
            "logical",
        )
        assert h1 != h2

    def test_different_source_identity(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(source_snapshot="snap_1"),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(source_snapshot="snap_2"),), (), "logical"
        )
        assert h1 != h2

    def test_different_semantic_contract(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (_region(grain="1d", available_at="eod"),), (), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (_region(grain="1m", available_at="eod"),), (), "logical"
        )
        h3 = PhysicalRegionPlan.compute_plan_hash(
            (_region(grain="1d", available_at="next_open"),), (), "logical"
        )
        h4 = PhysicalRegionPlan.compute_plan_hash(
            (_region(grain="1d", available_at="eod", pit_safe=False),),
            (),
            "logical",
        )
        assert len({h1, h2, h3, h4}) == 4

    def test_different_physical_properties(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(
                    required_properties=PhysicalProperties(
                        sorted_by=("datetime",)
                    )
                ),
            ),
            (),
            "logical",
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(
                    required_properties=PhysicalProperties(
                        sorted_by=("instrument",)
                    )
                ),
            ),
            (),
            "logical",
        )
        h3 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(
                    output_properties=PhysicalProperties(
                        partitioned_by=("instrument",)
                    )
                ),
            ),
            (),
            "logical",
        )
        assert len({h1, h2, h3}) == 3

    def test_different_state_contract(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(
                    state_contract=StateContract(stateful_operators=("ema",))
                ),
            ),
            (),
            "logical",
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (
                _region(
                    state_contract=StateContract(
                        stateful_operators=("ema",),
                        requires_checkpoint=True,
                        checkpoint_seed="seed-1",
                    )
                ),
            ),
            (),
            "logical",
        )
        assert h1 != h2


class TestDifferentEdgesDifferentHash:
    """Regression 3: edges differing in transforms/contracts must not collide."""

    def test_different_edge_transforms(self) -> None:
        def _h(**overrides: object) -> str:
            return PhysicalRegionPlan.compute_plan_hash(
                (), (_edge(**overrides),), "logical"
            )

        hashes = {
            _h(),
            _h(requires_sort=True),
            _h(requires_repartition=True),
            _h(requires_reshape=True),
            _h(requires_dtype_cast=True),
            _h(producer_sorted_by=("datetime",)),
            _h(producer_guarantees_order=True),
        }
        assert len(hashes) == 7

    def test_different_edge_semantic_contract(self) -> None:
        def _h(**overrides: object) -> str:
            return PhysicalRegionPlan.compute_plan_hash(
                (), (_edge(**overrides),), "logical"
            )

        hashes = {
            _h(),
            _h(preserves_pit=False),
            _h(preserves_universe=False),
            _h(preserves_grain=False),
            _h(source_snapshot_id="snap_9"),
        }
        assert len(hashes) == 5

    def test_different_edge_backends_and_representations(self) -> None:
        h1 = PhysicalRegionPlan.compute_plan_hash(
            (), (_edge(),), "logical"
        )
        h2 = PhysicalRegionPlan.compute_plan_hash(
            (),
            (
                _edge(
                    source_backend=PhysicalBackend.PANDAS_NUMPY,
                    source_representation=Representation.PANDAS_LONG,
                ),
            ),
            "logical",
        )
        assert h1 != h2


class TestHashExcludesEstimates:
    """Cost/size estimates are predictions, not plan identity (§68)."""

    def test_estimates_do_not_change_hash(self) -> None:
        r1 = _region(
            estimated_rows=1000,
            estimated_compute_ms=10.0,
            estimated_memory_bytes=8000,
            input_bytes=100,
            output_bytes=200,
        )
        r2 = _region(
            estimated_rows=9_999_999,
            estimated_compute_ms=999.0,
            estimated_memory_bytes=8_000_000,
            input_bytes=9_000_000,
            output_bytes=9_000_000,
        )
        e1 = _edge(estimated_rows=1000, estimated_transfer_ms=5.0)
        e2 = _edge(estimated_rows=123_456, estimated_transfer_ms=77.0)

        h1 = PhysicalRegionPlan.compute_plan_hash((r1,), (e1,), "logical")
        h2 = PhysicalRegionPlan.compute_plan_hash((r2,), (e2,), "logical")
        assert h1 == h2
