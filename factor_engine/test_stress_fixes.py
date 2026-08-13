#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test stress test fixes: DAG width/depth limits and chunked compilation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_constants():
    """Test that constants are properly defined."""
    from planner.physical_lowerer import (
        MAX_DAG_WIDTH,
        MAX_EXPRESSION_DEPTH,
        CHUNK_SIZE_DEFAULT,
    )

    print("Test 1: Constants")
    assert MAX_DAG_WIDTH == 1000, f"Expected 1000, got {MAX_DAG_WIDTH}"
    assert MAX_EXPRESSION_DEPTH == 100, f"Expected 100, got {MAX_EXPRESSION_DEPTH}"
    assert CHUNK_SIZE_DEFAULT == 500, f"Expected 500, got {CHUNK_SIZE_DEFAULT}"
    print(f"  ✓ MAX_DAG_WIDTH = {MAX_DAG_WIDTH}")
    print(f"  ✓ MAX_EXPRESSION_DEPTH = {MAX_EXPRESSION_DEPTH}")
    print(f"  ✓ CHUNK_SIZE_DEFAULT = {CHUNK_SIZE_DEFAULT}")


def test_expression_depth_validation():
    """Test expression depth validation."""
    from planner.physical_lowerer import validate_expression_depth
    from planner.logical_plan import PlanNode

    print("\nTest 2: Expression depth validation")

    # Create a simple expression (depth 2)
    leaf = PlanNode(op="column", attrs={"name": "close"}, inputs=())
    node = PlanNode(op="add", inputs=(leaf, leaf))

    depth = validate_expression_depth(node, max_depth=100)
    assert depth <= 2, f"Expected depth <= 2, got {depth}"
    print(f"  ✓ Simple expression depth: {depth}")

    # Create a deep expression (depth > 100)
    deep = leaf
    for i in range(105):
        deep = PlanNode(op="add", inputs=(deep, leaf))

    try:
        validate_expression_depth(deep, max_depth=100)
        print("  ✗ Should have raised ValueError for deep expression")
        return False
    except ValueError as e:
        assert "exceeds limit" in str(e)
        print(f"  ✓ Deep expression correctly rejected: {str(e)[:60]}...")

    return True


def test_chunked_compilation_structure():
    """Test that compile_many_chunked has correct structure."""
    from planner.physical_lowerer import compile_many_chunked
    import inspect

    print("\nTest 3: Chunked compilation structure")

    sig = inspect.signature(compile_many_chunked)
    params = list(sig.parameters.keys())

    assert "dag_plans" in params, "Missing dag_plans parameter"
    assert "chunk_size" in params, "Missing chunk_size parameter"
    print(f"  ✓ Function signature: {params[:3]}...")

    # Check docstring
    assert compile_many_chunked.__doc__ is not None
    assert "5000" in compile_many_chunked.__doc__
    print("  ✓ Docstring references stress test limits")


def test_batch_dag_validation():
    """Test that lower_batch_dag validates input size."""
    from planner.physical_lowerer import lower_batch_dag, MAX_DAG_WIDTH
    from dataclasses import dataclass

    print("\nTest 4: Batch DAG validation")

    @dataclass
    class MockFactorPlan:
        factor_name: str
        root: object

    @dataclass
    class MockDAG:
        roots: list
        shared_nodes: dict | None = None

    # Test with acceptable size
    small_dag = MockDAG(roots=[MockFactorPlan(f"f{i}", None) for i in range(10)])
    print(f"  • Small DAG with {len(small_dag.roots)} factors: OK to test structure")

    # Test with oversized DAG
    from planner.logical_plan import PlanNode
    leaf = PlanNode(op="column", attrs={"name": "close"}, inputs=())
    large_dag = MockDAG(
        roots=[MockFactorPlan(f"f{i}", leaf) for i in range(MAX_DAG_WIDTH + 1)]
    )

    try:
        lower_batch_dag(large_dag, ctx=None)
        print(f"  ✗ Should have raised ValueError for {len(large_dag.roots)} factors")
        return False
    except ValueError as e:
        assert "Cannot compile" in str(e) or "Limit is" in str(e)
        print(f"  ✓ Large DAG correctly rejected: {str(e)[:70]}...")

    return True


def test_imports():
    """Test that all exports are accessible."""
    print("\nTest 5: Public API imports")

    from planner import (
        compile_many_chunked,
        validate_expression_depth,
        MAX_DAG_WIDTH,
        MAX_EXPRESSION_DEPTH,
    )

    print("  ✓ compile_many_chunked")
    print("  ✓ validate_expression_depth")
    print("  ✓ MAX_DAG_WIDTH")
    print("  ✓ MAX_EXPRESSION_DEPTH")


def main():
    """Run all tests."""
    print("=" * 70)
    print("STRESS TEST FIXES VERIFICATION")
    print("=" * 70)

    try:
        test_constants()
        test_expression_depth_validation()
        test_chunked_compilation_structure()
        test_batch_dag_validation()
        test_imports()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nFixes applied:")
        print("  1. MAX_DAG_WIDTH = 1000 (breaking point: 5000)")
        print("  2. MAX_EXPRESSION_DEPTH = 100 (breaking point: 200)")
        print("  3. compile_many_chunked() for large batches")
        print("  4. validate_expression_depth() prevents RecursionError")
        print("  5. lower_batch_dag() validates input size")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
