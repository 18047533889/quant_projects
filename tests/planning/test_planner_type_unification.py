# -*- coding: utf-8 -*-
"""Regression tests for R21-PLANNER-TYPE-UNIFICATION.

Verifies:
1. planner.backend_region and planning.backend_region both import cleanly.
2. No dual physical ABI — both modules expose the SAME enum classes (identity).
3. Numba is an Accelerator, NOT a PhysicalBackend (NUMBA_CPU must not exist).
4. Arrow is an interchange Representation, NOT a PhysicalBackend (ARROW_COMPUTE
   must not exist).
5. The single authority PhysicalBackend enum has exactly the six canonical
   backends: PANDAS_NUMPY / POLARS_PANEL / POLARS_LONG / DUCKDB_SQL /
   CLICKHOUSE_SQL / Q_KDB.
"""
from __future__ import annotations

import pytest

from planner.backend_region import (
    BackendRegion as PlannerBackendRegion,
    PhysicalBackend as PlannerPhysicalBackend,
    PhysicalProperties as PlannerPhysicalProperties,
    Representation as PlannerRepresentation,
    StateContract as PlannerStateContract,
)
from planning.backend_region import (
    BackendRegion as PlanningBackendRegion,
    PhysicalBackend as PlanningPhysicalBackend,
    PhysicalProperties as PlanningPhysicalProperties,
    Representation as PlanningRepresentation,
    StateContract as PlanningStateContract,
)

# Canonical six-backend physical ABI (single authority).
_CANONICAL_BACKEND_MEMBERS = {
    "PANDAS_NUMPY": "pandas_numpy",
    "POLARS_PANEL": "polars_panel",
    "POLARS_LONG": "polars_long",
    "DUCKDB_SQL": "duckdb_sql",
    "CLICKHOUSE_SQL": "clickhouse_sql",
    "Q_KDB": "q_kdb",
}


class TestImportsResolve:
    """Test 1: both planner.backend_region and planning.backend_region import."""

    def test_planner_backend_region_imports(self):
        assert PlannerPhysicalBackend is not None
        assert PlannerRepresentation is not None
        assert PlannerBackendRegion is not None
        assert PlannerStateContract is not None

    def test_planning_backend_region_imports(self):
        assert PlanningPhysicalBackend is not None
        assert PlanningRepresentation is not None
        assert PlanningBackendRegion is not None
        assert PlanningStateContract is not None

    def test_planning_reexports_execution_axis(self):
        from planning.backend_region import ExecutionAxis as PlanningAxis
        from planner.backend_region import ExecutionAxis as PlannerAxis

        assert PlanningAxis is PlannerAxis


class TestNoDualAbi:
    """Test 2: no dual physical type system — planning re-exports planner's classes."""

    def test_physical_backend_single_authority(self):
        assert PlanningPhysicalBackend is PlannerPhysicalBackend, (
            "planning.backend_region must re-export planner.backend_region.PhysicalBackend"
        )

    def test_representation_single_authority(self):
        assert PlanningRepresentation is PlannerRepresentation, (
            "planning.backend_region must re-export planner.backend_region.Representation"
        )

    def test_state_contract_single_authority(self):
        assert PlanningStateContract is PlannerStateContract, (
            "planning.backend_region must re-export planner.backend_region.StateContract"
        )

    def test_backend_region_single_authority(self):
        assert PlanningBackendRegion is PlannerBackendRegion, (
            "planning.backend_region must re-export planner.backend_region.BackendRegion"
        )

    def test_physical_properties_single_authority(self):
        # planning consumers use the alias PhysicalProperty (MB-P2-002 name).
        from planning.backend_region import PhysicalProperty as PlanningProperty

        assert PlanningProperty is PlannerPhysicalProperties

    def test_physical_backend_members_exact(self):
        actual = {m.name for m in PlannerPhysicalBackend}
        assert actual == set(_CANONICAL_BACKEND_MEMBERS), (
            f"PhysicalBackend must contain exactly the six canonical backends, got {actual}"
        )


class TestNumbaAndArrowNotBackends:
    """Test 3: Numba = Accelerator, Arrow = interchange Representation only."""

    def test_numba_not_a_backend(self):
        assert not hasattr(PlannerPhysicalBackend, "NUMBA_CPU"), (
            "NUMBA_CPU must NOT be a PhysicalBackend (Numba is an Accelerator)"
        )

    def test_arrow_compute_not_a_backend(self):
        assert not hasattr(PlannerPhysicalBackend, "ARROW_COMPUTE"), (
            "ARROW_COMPUTE must NOT be a PhysicalBackend (Arrow is an interchange Representation)"
        )

    def test_numba_is_accelerator(self):
        from backend.contracts import Accelerator, ExecutionKind

        assert hasattr(Accelerator, "NUMBA_CPU")
        assert Accelerator.NUMBA_CPU.value == "numba_cpu"
        assert hasattr(ExecutionKind, "NUMBA_CPU_KERNEL")
        assert ExecutionKind.NUMBA_CPU_KERNEL.value == "numba_cpu_kernel"

    def test_arrow_is_representation(self):
        assert hasattr(PlannerRepresentation, "ARROW_TABLE")
        assert PlannerRepresentation.ARROW_TABLE.value == "arrow_table"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
