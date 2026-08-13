#!/usr/bin/env python3
"""Verification script for adapters layer implementation."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../factor_engine"))

def test_fe_adapter():
    """Test FE adapter."""
    print("Testing FE Adapter...")

    from factor_optimizer.adapters import create_fe_adapter, FEOptionalDependencyMissing

    try:
        # Try to add FE to path if not already there
        fe_path = os.path.abspath("../../factor_engine")
        if os.path.exists(fe_path) and fe_path not in sys.path:
            sys.path.insert(0, fe_path)

        import api
        from expr.field import FieldRef

        adapter = create_fe_adapter()

        # Test canonical hash
        close_field = FieldRef('close')
        factor = api.ts_mean(close_field, 20)
        hash_val = adapter.compute_canonical_hash(factor)
        assert len(hash_val) == 64, "Hash should be 64 chars"

        # Test complexity
        complexity = adapter.estimate_complexity(factor)
        assert "operator_count" in complexity
        assert complexity["operator_count"] >= 0

        # Test operator metadata
        catalog = adapter.get_operator_metadata(["ts_mean"])
        assert "ts_mean" in catalog

        print("  ✓ FE adapter working with real FactorEngine")
        return True

    except (FEOptionalDependencyMissing, ImportError) as e:
        print(f"  ⚠ FE not available (this is OK): {e}")
        return True  # This is acceptable - FE is optional


def test_qe_adapter():
    """Test QE adapter."""
    print("Testing QE Adapter...")

    from factor_optimizer.adapters import create_mock_qe_adapter

    adapter = create_mock_qe_adapter()

    # Test evaluation
    result = adapter.evaluate(
        factor_batch="test_factor",
        labels="returns",
        metrics=["rank_ic", "turnover"]
    )

    assert "evaluation_id" in result
    assert "metrics" in result
    assert "rank_ic" in result["metrics"]
    assert "turnover" in result["metrics"]

    # Test evidence retrieval
    evidence = adapter.get_evidence(result["evaluation_id"])
    assert evidence["evaluation_id"] == result["evaluation_id"]

    # Test metrics catalog
    metrics = adapter.list_metrics(tier="core")
    assert len(metrics) > 0
    assert all("metric_id" in m for m in metrics)

    print("  ✓ QE mock adapter working")
    return True


def test_protocol_compliance():
    """Test protocol compliance."""
    print("Testing Protocol Compliance...")

    from factor_optimizer.adapters import FactorEngineAdapter, QuantEvaluatorAdapter

    # Simple mock that implements all FE methods
    class SimpleFEMock:
        def compute_canonical_hash(self, factor_definition):
            return "hash"

        def validate_mutation(self, mutation, spec):
            return {"is_legal": True, "reason": "", "metadata": {}}

        def estimate_complexity(self, factor_definition):
            return {
                "operator_count": 1,
                "max_depth": 1,
                "lookback_periods": 0,
                "estimated_cost": 1.0,
                "domains": [],
                "sources": [],
            }

        def get_operator_metadata(self, operator_names=None):
            return {}

    fe_mock = SimpleFEMock()
    assert isinstance(fe_mock, FactorEngineAdapter), "Mock should satisfy FE protocol"

    # Simple mock that implements all QE methods
    class SimpleQEMock:
        def evaluate(self, factor_batch, labels, metrics=None, context=None):
            return {"evaluation_id": "test", "metrics": {}, "diagnostics": {}, "evidence_ref": ""}

        def get_evidence(self, evaluation_id):
            return {"evaluation_id": evaluation_id, "metrics": {}, "diagnostics": {}, "timeseries": {}}

        def list_metrics(self, tier=None):
            return []

    qe_mock = SimpleQEMock()
    assert isinstance(qe_mock, QuantEvaluatorAdapter), "Mock should satisfy QE protocol"

    print("  ✓ Protocol compliance verified")
    return True


def test_integration():
    """Test integration with FO core."""
    print("Testing Integration with FO Core...")

    from factor_optimizer.adapters import create_mock_qe_adapter

    # Mock adapter for testing
    class MockFEAdapter:
        def compute_canonical_hash(self, factor_definition):
            import hashlib
            return hashlib.sha256(str(factor_definition).encode()).hexdigest()

        def validate_mutation(self, mutation, spec):
            return {"is_legal": True, "reason": "", "metadata": {}}

        def estimate_complexity(self, factor_definition):
            return {
                "operator_count": 1,
                "max_depth": 1,
                "lookback_periods": 0,
                "estimated_cost": 1.0,
                "domains": [],
                "sources": [],
            }

        def get_operator_metadata(self, operator_names=None):
            return {}

    fe_adapter = MockFEAdapter()

    # Test with SeenCache
    from factor_optimizer.seen import SeenCache
    cache = SeenCache(fe_adapter=fe_adapter)

    was_seen, record = cache.check_and_mark("factor1", "trial_001")
    assert not was_seen

    was_seen, record = cache.check_and_mark("factor1", "trial_002")
    assert was_seen

    # Test with ComplexityEstimator
    from factor_optimizer.complexity import ComplexityEstimator
    estimator = ComplexityEstimator(fe_adapter=fe_adapter)

    profile = estimator.estimate("factor1")
    assert profile.operator_count > 0

    print("  ✓ Integration with FO core working")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Adapters Layer Verification")
    print("=" * 60)
    print()

    results = []

    results.append(("Protocol Compliance", test_protocol_compliance()))
    results.append(("FE Adapter", test_fe_adapter()))
    results.append(("QE Adapter", test_qe_adapter()))
    results.append(("FO Core Integration", test_integration()))

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    print()

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("✓ All verification tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
