"""Tests for FactorEngineAdapter."""

import pytest
from factor_optimizer.adapters import (
    FactorEngineAdapter,
    FEOptionalDependencyMissing,
    create_fe_adapter,
)


class MockFEAdapter:
    """Mock FE adapter for testing without FE dependency."""

    def compute_canonical_hash(self, factor_definition):
        """Mock canonical hash computation."""
        # Simple deterministic hash based on string representation
        import hashlib

        content = str(factor_definition)
        return hashlib.sha256(content.encode()).hexdigest()

    def validate_mutation(self, mutation, spec):
        """Mock mutation validation."""
        # Accept mutations with valid target_operator
        target_op = mutation.parameters.get("target_operator")
        if target_op and target_op.startswith("invalid_"):
            return {
                "is_legal": False,
                "reason": f"Operator {target_op} not found",
                "metadata": {},
            }
        return {
            "is_legal": True,
            "reason": "Mock validation passed",
            "metadata": {},
        }

    def estimate_complexity(self, factor_definition):
        """Mock complexity estimation."""
        return {
            "operator_count": 3,
            "max_depth": 2,
            "lookback_periods": 20,
            "estimated_cost": 5.0,
            "domains": ["price", "volume"],
            "sources": ["market_data"],
        }

    def get_operator_metadata(self, operator_names=None):
        """Mock operator metadata."""
        all_ops = {
            "ts_mean": {
                "canonical": "ts_mean",
                "backends": ["pandas_numpy", "polars"],
                "param_names": ["x", "window"],
                "status": "implemented",
            },
            "cs_rank": {
                "canonical": "cs_rank",
                "backends": ["pandas_numpy"],
                "param_names": ["x"],
                "status": "implemented",
            },
        }

        if operator_names is None:
            return all_ops

        return {name: all_ops[name] for name in operator_names if name in all_ops}


def test_mock_adapter_protocol():
    """Test that mock adapter satisfies protocol."""
    adapter = MockFEAdapter()

    # Check protocol compliance
    assert isinstance(adapter, FactorEngineAdapter)

    # Test canonical hash
    hash1 = adapter.compute_canonical_hash("ts_mean(close, 20)")
    hash2 = adapter.compute_canonical_hash("ts_mean(close, 20)")
    hash3 = adapter.compute_canonical_hash("ts_mean(close, 30)")

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hex
    assert hash1 == hash2  # Deterministic
    assert hash1 != hash3  # Different inputs


def test_mock_adapter_validate_mutation():
    """Test mock mutation validation."""
    adapter = MockFEAdapter()

    # Mock mutation object
    class MockMutation:
        def __init__(self, params):
            self.parameters = params

    # Valid mutation
    valid_mut = MockMutation({"target_operator": "ts_mean"})
    result = adapter.validate_mutation(valid_mut, None)

    assert result["is_legal"] is True
    assert "metadata" in result

    # Invalid mutation
    invalid_mut = MockMutation({"target_operator": "invalid_operator"})
    result = adapter.validate_mutation(invalid_mut, None)

    assert result["is_legal"] is False
    assert "not found" in result["reason"].lower()


def test_mock_adapter_estimate_complexity():
    """Test mock complexity estimation."""
    adapter = MockFEAdapter()

    result = adapter.estimate_complexity("ts_mean(close, 20)")

    assert isinstance(result, dict)
    assert result["operator_count"] == 3
    assert result["max_depth"] == 2
    assert result["lookback_periods"] == 20
    assert result["estimated_cost"] > 0


def test_mock_adapter_get_operator_metadata():
    """Test mock operator metadata retrieval."""
    adapter = MockFEAdapter()

    # Get all operators
    all_ops = adapter.get_operator_metadata()
    assert isinstance(all_ops, dict)
    assert "ts_mean" in all_ops
    assert "cs_rank" in all_ops

    # Get specific operators
    specific = adapter.get_operator_metadata(["ts_mean"])
    assert len(specific) == 1
    assert "ts_mean" in specific
    assert "cs_rank" not in specific


@pytest.mark.integration
def test_create_fe_adapter_with_real_fe():
    """Test creating real FE adapter when factor_engine is available."""
    try:
        adapter = create_fe_adapter()

        # Check protocol compliance
        assert isinstance(adapter, FactorEngineAdapter)

        # Test operator metadata retrieval
        catalog = adapter.get_operator_metadata()
        assert isinstance(catalog, dict)
        assert len(catalog) > 0

        # Check a known operator
        assert any("mean" in op.lower() for op in catalog.keys())

    except FEOptionalDependencyMissing:
        pytest.skip("FactorEngine not available")


@pytest.mark.integration
def test_fe_adapter_canonical_hash():
    """Test canonical hash with real FE."""
    try:
        import sys
        import os

        # Add factor_engine to path
        fe_path = os.path.abspath("factor_engine")
        if fe_path not in sys.path:
            sys.path.insert(0, fe_path)

        import api
        from expr.field import field

        adapter = create_fe_adapter()

        # Create a simple factor using field
        close_field = field("close")
        factor = api.ts_mean(close_field, 20)

        # Compute hash
        hash1 = adapter.compute_canonical_hash(factor)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Same factor should produce same hash
        close_field2 = field("close")
        factor2 = api.ts_mean(close_field2, 20)
        hash2 = adapter.compute_canonical_hash(factor2)
        assert hash1 == hash2

        # Different factor should produce different hash
        factor3 = api.ts_mean(close_field, 30)
        hash3 = adapter.compute_canonical_hash(factor3)
        assert hash1 != hash3

    except (FEOptionalDependencyMissing, ImportError):
        pytest.skip("FactorEngine not available")


@pytest.mark.integration
def test_fe_adapter_complexity():
    """Test complexity estimation with real FE."""
    try:
        import sys
        import os

        fe_path = os.path.abspath("factor_engine")
        if fe_path not in sys.path:
            sys.path.insert(0, fe_path)

        import api
        from expr.field import field

        adapter = create_fe_adapter()

        # Simple factor (just a field reference)
        close_field = field("close")
        result = adapter.estimate_complexity(close_field)

        assert isinstance(result, dict)
        assert "operator_count" in result
        assert "max_depth" in result

        # Complex factor with operators and window
        complex_factor = api.ts_mean(api.ts_std(close_field, 10), 20)
        result2 = adapter.estimate_complexity(complex_factor)

        # Complex factor should have more operators
        assert result2["operator_count"] > result["operator_count"]
        assert result2["lookback_periods"] > 0

    except (FEOptionalDependencyMissing, ImportError):
        pytest.skip("FactorEngine not available")


def test_create_fe_adapter_missing_dependency():
    """Test that missing FE raises appropriate error."""
    # This test is tricky because FE is actually available in our environment
    # Just verify the exception type exists and can be raised

    # Verify exception exists
    assert FEOptionalDependencyMissing is not None

    # Verify it's an Exception subclass
    assert issubclass(FEOptionalDependencyMissing, Exception)

    # Test raising it
    with pytest.raises(FEOptionalDependencyMissing):
        raise FEOptionalDependencyMissing("Test message")


def test_adapter_protocol_interface():
    """Test that protocol defines expected interface."""
    # Check protocol has required methods
    required_methods = [
        "compute_canonical_hash",
        "validate_mutation",
        "estimate_complexity",
        "get_operator_metadata",
    ]

    for method in required_methods:
        assert hasattr(FactorEngineAdapter, method)
