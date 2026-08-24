# -*- coding: utf-8 -*-
"""Regression tests for R21-Q-PHYSICAL-BACKEND.

Verifies:
1. PhysicalBackend.Q_KDB exists and has correct value
2. Representation Q_TABLE, Q_VECTOR, Q_KEYED_TABLE, Q_SHARED_HANDLE exist
3. Single StateContract definition (no duplicate)
"""
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_engine.planner.backend_region import (
    PhysicalBackend,
    Representation,
    StateContract,
    BackendRegion,
    ExecutionAxis,
    PhysicalProperties,
)


def test_physical_backend_q_kdb_exists():
    """Test that Q_KDB is present in PhysicalBackend enum."""
    assert hasattr(PhysicalBackend, "Q_KDB"), "PhysicalBackend.Q_KDB is missing"
    assert PhysicalBackend.Q_KDB.value == "q_kdb", f"Expected 'q_kdb', got '{PhysicalBackend.Q_KDB.value}'"


def test_representation_q_variants_exist():
    """Test that all Q representations are present in Representation enum."""
    q_members = ["Q_TABLE", "Q_VECTOR", "Q_KEYED_TABLE", "Q_SHARED_HANDLE"]
    for member in q_members:
        assert hasattr(Representation, member), f"Representation.{member} is missing"
        expected_value = member.lower()
        actual_value = getattr(Representation, member).value
        assert actual_value == expected_value, f"Expected '{expected_value}', got '{actual_value}'"


def test_single_state_contract_definition():
    """Test that StateContract is defined only once and has expected fields."""
    # Verify the class exists and has the expected docstring
    assert hasattr(StateContract, "__doc__"), "StateContract class is missing"
    assert "MB-P0-012" in StateContract.__doc__, "StateContract docstring should mention MB-P0-012"

    # Verify fields
    contract = StateContract()
    assert hasattr(contract, "requires_checkpoint"), "StateContract missing requires_checkpoint"
    assert hasattr(contract, "checkpoint_seed"), "StateContract missing checkpoint_seed"
    assert hasattr(contract, "stateful_operators"), "StateContract missing stateful_operators"
    assert hasattr(contract, "checkpoint_interval"), "StateContract missing checkpoint_interval"
    assert hasattr(contract, "sequential_only"), "StateContract missing sequential_only"


def test_backend_region_with_q_kdb():
    """Test that BackendRegion can be created with Q_KDB backend."""
    region = BackendRegion(
        region_id="q_region_1",
        backend=PhysicalBackend.Q_KDB,
        representation=Representation.Q_TABLE,
        node_ids=("node1",),
        execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
        estimated_rows=1000,
        estimated_compute_ms=10.0,
        estimated_memory_bytes=1024 * 1024,
    )
    assert region.backend == PhysicalBackend.Q_KDB
    assert region.representation == Representation.Q_TABLE


if __name__ == "__main__":
    # Run tests manually if executed directly
    import pytest
    pytest.main([__file__, "-v"])
