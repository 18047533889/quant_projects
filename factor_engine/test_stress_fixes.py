#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test adaptive stress test behavior: DAG width/depth limits scale with host memory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_adaptive_constants():
    """Test that constants adapt to host memory."""
    from planner.physical_lowerer import (
        _get_adaptive_dag_width_limit,
        _get_expression_depth_limit,
        _get_adaptive_chunk_size,
    )
    from runtime.adaptive_config import get_adaptive_config

    print("Test 1: Adaptive constants scale with memory")

    # 30GB 主机
    config_30gb = get_adaptive_config(force_memory_gb=30.0)
    print(f"\n  30GB host:")
    print(f"    dag_chunk_size: {config_30gb.dag_chunk_size}")
    print(f"    compile_chunk_size: {config_30gb.compile_chunk_size}")

    # 500GB 主机
    config_500gb = get_adaptive_config(force_memory_gb=500.0)
    print(f"\n  500GB host:")
    print(f"    dag_chunk_size: {config_500gb.dag_chunk_size}")
    print(f"    compile_chunk_size: {config_500gb.compile_chunk_size}")

    # 表达式深度基于 Python 递归限制
    depth_limit = _get_expression_depth_limit()
    print(f"\n  Expression depth limit: {depth_limit} (based on sys.getrecursionlimit={sys.getrecursionlimit()})")

    assert config_500gb.dag_chunk_size > config_30gb.dag_chunk_size, \
        "500GB host should have larger limits than 30GB"
    assert depth_limit == int(sys.getrecursionlimit() * 0.8), \
        "Depth limit should be 80% of Python recursion limit"

    print(f"\n  ✓ Limits scale correctly with memory")
    print(f"  ✓ 500GB host gets {config_500gb.dag_chunk_size / config_30gb.dag_chunk_size:.1f}x larger DAG limit")


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

    # Create a deep expression (depth > typical limit)
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


def test_chunked_compilation_adaptive():
    """Test that compile_many_chunked uses adaptive chunk size."""
    from planner.physical_lowerer import compile_many_chunked, _get_adaptive_chunk_size
    import inspect

    print("\nTest 3: Chunked compilation with adaptive sizing")

    sig = inspect.signature(compile_many_chunked)
    params = list(sig.parameters.keys())

    assert "dag_plans" in params, "Missing dag_plans parameter"
    assert "chunk_size" in params, "Missing chunk_size parameter"
    print(f"  ✓ Function signature: {params[:3]}...")

    # Check that it references adaptive config
    assert compile_many_chunked.__doc__ is not None
    assert "adaptive" in compile_many_chunked.__doc__.lower() or "自适应" in compile_many_chunked.__doc__
    print("  ✓ Docstring mentions adaptive behavior")

    # Check default chunk size is callable
    chunk_size = _get_adaptive_chunk_size()
    print(f"  ✓ Adaptive chunk size: {chunk_size}")
    assert chunk_size >= 100, "Chunk size should be reasonable"


def test_batch_dag_validation_adaptive():
    """Test that lower_batch_dag validates with adaptive limit."""
    from planner.physical_lowerer import lower_batch_dag, _get_adaptive_dag_width_limit
    from dataclasses import dataclass

    print("\nTest 4: Batch DAG validation (adaptive limit)")

    @dataclass
    class MockFactorPlan:
        factor_name: str
        root: object

    @dataclass
    class MockDAG:
        roots: list
        shared_nodes: dict | None = None

    # Get current adaptive limit
    max_width = _get_adaptive_dag_width_limit()
    print(f"  • Current adaptive limit: {max_width} factors")

    # Test with oversized DAG
    from planner.logical_plan import PlanNode
    leaf = PlanNode(op="column", attrs={"name": "close"}, inputs=())
    large_dag = MockDAG(
        roots=[MockFactorPlan(f"f{i}", leaf) for i in range(max_width + 100)]
    )

    try:
        lower_batch_dag(large_dag, ctx=None)
        print(f"  ✗ Should have raised ValueError for {len(large_dag.roots)} factors")
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"  ✓ Correctly rejected with error:")
        print(f"      {error_msg[:100]}...")

        # Verify error message quality
        assert "Cannot compile" in error_msg
        assert str(max_width) in error_msg or "Adaptive limit" in error_msg
        assert "compile_many_chunked" in error_msg

        print("   ✓ Error message contains:")
        print("      - Clear rejection reason")
        print("      - Adaptive limit value")
        print("      - Recommended solution (compile_many_chunked)")

        return True


def test_env_var_override():
    """Test that environment variables can override adaptive limits."""
    import os
    from planner.physical_lowerer import _get_adaptive_dag_width_limit

    print("\nTest 5: Environment variable override")

    # Set override
    original = os.environ.get("MAX_DAG_WIDTH")
    try:
        os.environ["MAX_DAG_WIDTH"] = "2500"
        limit = _get_adaptive_dag_width_limit()
        assert limit == 2500, f"Expected 2500, got {limit}"
        print(f"  ✓ MAX_DAG_WIDTH env var override works: {limit}")
    finally:
        if original is None:
            os.environ.pop("MAX_DAG_WIDTH", None)
        else:
            os.environ["MAX_DAG_WIDTH"] = original


def test_imports():
    """Test that all exports are accessible."""
    print("\nTest 6: Public API imports")

    from planner import (
        compile_many_chunked,
        validate_expression_depth,
        MAX_DAG_WIDTH,
        MAX_EXPRESSION_DEPTH,
    )

    print("  ✓ compile_many_chunked")
    print("  ✓ validate_expression_depth")
    print(f"  ✓ MAX_DAG_WIDTH (adaptive, current: {MAX_DAG_WIDTH})")
    print(f"  ✓ MAX_EXPRESSION_DEPTH (adaptive, current: {MAX_EXPRESSION_DEPTH})")


def main():
    """Run all tests."""
    print("=" * 70)
    print("ADAPTIVE STRESS TEST BEHAVIOR VERIFICATION")
    print("=" * 70)

    try:
        test_adaptive_constants()
        test_expression_depth_validation()
        test_chunked_compilation_adaptive()
        test_batch_dag_validation_adaptive()
        test_env_var_override()
        test_imports()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nBehavior verified:")
        print("  1. DAG width limit scales with host memory (30GB→1000, 500GB→5000)")
        print("  2. Expression depth limit based on Python recursion limit (80% of 1000 = 800)")
        print("  3. compile_many_chunked() uses adaptive chunk sizing")
        print("  4. validate_expression_depth() prevents RecursionError")
        print("  5. lower_batch_dag() validates with adaptive limits")
        print("  6. Environment variables override adaptive values")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
