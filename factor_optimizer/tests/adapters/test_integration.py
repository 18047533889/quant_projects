"""Integration tests for adapters working with FO core."""

import pytest
from factor_optimizer.adapters import (
    create_mock_qe_adapter,
    FEOptionalDependencyMissing,
    create_fe_adapter,
)
from factor_optimizer.seen import SeenCache
from factor_optimizer.complexity import ComplexityEstimator


class TestFEAdapterIntegration:
    """Test FE adapter integration with FO core components."""

    def test_seen_cache_with_mock_adapter(self):
        """Test SeenCache with mock FE adapter."""
        # Import at runtime to avoid issues
        import sys
        import os

        # Ensure factor_optimizer is in path
        fo_path = os.path.abspath(".")
        if fo_path not in sys.path:
            sys.path.insert(0, fo_path)

        from factor_optimizer.adapters.factor_engine import FactorEngineAdapter

        # Create mock adapter with all required methods
        class SimpleMockAdapter:
            def compute_canonical_hash(self, factor_definition):
                import hashlib

                return hashlib.sha256(str(factor_definition).encode()).hexdigest()

            def validate_mutation(self, mutation, spec):
                return {"is_legal": True, "reason": "mock", "metadata": {}}

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

        adapter = SimpleMockAdapter()
        assert isinstance(adapter, FactorEngineAdapter)

        # Create cache with adapter
        cache = SeenCache(fe_adapter=adapter)

        # Mark some factors as seen
        factor1 = "ts_mean(close, 20)"
        factor2 = "ts_mean(close, 30)"

        # Check and mark
        was_seen, record = cache.check_and_mark(factor1, "trial_001")
        assert not was_seen
        assert record.trial_id == "trial_001"

        # Same factor should be seen
        was_seen, record = cache.check_and_mark(factor1, "trial_002")
        assert was_seen
        assert record.trial_id == "trial_001"  # Original trial

        # Different factor should not be seen
        was_seen, record = cache.check_and_mark(factor2, "trial_003")
        assert not was_seen

        # Cache should have 2 factors
        assert cache.size() == 2

    def test_complexity_estimator_with_mock_adapter(self):
        """Test ComplexityEstimator with mock FE adapter."""

        class SimpleMockAdapter:
            def estimate_complexity(self, factor_definition):
                # Count words as proxy for complexity
                content = str(factor_definition)
                word_count = len(content.split())

                return {
                    "operator_count": word_count,
                    "max_depth": min(word_count, 5),
                    "lookback_periods": 20 if "20" in content else 0,
                    "estimated_cost": float(word_count * 1.5),
                    "domains": [],
                    "sources": [],
                }

        adapter = SimpleMockAdapter()
        estimator = ComplexityEstimator(fe_adapter=adapter)

        # Estimate simple factor
        simple = "close"
        profile = estimator.estimate(simple)

        assert profile.operator_count > 0
        assert profile.estimated_cost > 0

        # Estimate complex factor
        complex_factor = "ts_mean(ts_std(close, 10), 20)"
        complex_profile = estimator.estimate(complex_factor)

        # Complex should have higher cost
        assert complex_profile.operator_count > profile.operator_count
        assert complex_profile.estimated_cost > profile.estimated_cost

    @pytest.mark.integration
    def test_seen_cache_with_real_fe(self):
        """Test SeenCache with real FE adapter."""
        try:
            import sys
            import os

            fe_path = os.path.abspath("factor_engine")
            if fe_path not in sys.path:
                sys.path.insert(0, fe_path)

            import api
            from expr.field import field

            adapter = create_fe_adapter()
            cache = SeenCache(fe_adapter=adapter)

            # Create factors
            close_field = field("close")
            factor1 = api.ts_mean(close_field, 20)
            factor2 = api.ts_mean(field("close"), 20)  # Same
            factor3 = api.ts_mean(close_field, 30)  # Different

            # Check and mark
            was_seen, record1 = cache.check_and_mark(factor1, "trial_001")
            assert not was_seen

            # Same semantic factor should be detected
            was_seen, record2 = cache.check_and_mark(factor2, "trial_002")
            assert was_seen
            assert record2.canonical_hash == record1.canonical_hash

            # Different factor
            was_seen, record3 = cache.check_and_mark(factor3, "trial_003")
            assert not was_seen
            assert record3.canonical_hash != record1.canonical_hash

        except (FEOptionalDependencyMissing, ImportError):
            pytest.skip("FactorEngine not available")

    @pytest.mark.integration
    def test_complexity_estimator_with_real_fe(self):
        """Test ComplexityEstimator with real FE adapter."""
        try:
            import sys
            import os

            fe_path = os.path.abspath("factor_engine")
            if fe_path not in sys.path:
                sys.path.insert(0, fe_path)

            import api
            from expr.field import field

            adapter = create_fe_adapter()
            estimator = ComplexityEstimator(fe_adapter=adapter)

            # Simple factor
            close_field = field("close")
            profile1 = estimator.estimate(close_field)

            assert profile1.operator_count >= 0
            assert profile1.max_depth >= 0

            # Complex factor with operators
            complex_factor = api.ts_mean(api.ts_std(close_field, 10), 20)
            profile2 = estimator.estimate(complex_factor)

            # Complex should have more operators
            assert profile2.operator_count > profile1.operator_count
            assert profile2.lookback_periods > 0

        except (FEOptionalDependencyMissing, ImportError):
            pytest.skip("FactorEngine not available")


class TestQEAdapterIntegration:
    """Test QE adapter integration with FO core components."""

    def test_mock_qe_adapter_in_search_loop(self):
        """Test mock QE adapter in a simulated search loop."""
        adapter = create_mock_qe_adapter()

        # Simulate search loop
        results = []
        for i in range(10):
            result = adapter.evaluate(
                factor_batch=f"candidate_{i}",
                labels="returns",
                metrics=["rank_ic", "turnover"],
            )
            results.append(result)

        # All evaluations should complete
        assert len(results) == 10

        # All should have metrics
        for result in results:
            assert "rank_ic" in result["metrics"]
            assert "turnover" in result["metrics"]

        # Can retrieve all evidence
        for result in results:
            evidence = adapter.get_evidence(result["evaluation_id"])
            assert evidence["evaluation_id"] == result["evaluation_id"]

    def test_mock_qe_adapter_metric_catalog(self):
        """Test using QE metric catalog for optimization."""
        adapter = create_mock_qe_adapter()

        # Get metrics to optimize
        core_metrics = adapter.list_metrics(tier="core")

        # Should have multiple core metrics
        assert len(core_metrics) >= 3

        # Extract metric IDs
        metric_ids = [m["metric_id"] for m in core_metrics]

        # Evaluate with specific metrics
        result = adapter.evaluate(
            factor_batch="candidate",
            labels="returns",
            metrics=metric_ids,
        )

        # All requested metrics should be present
        for metric_id in metric_ids:
            assert metric_id in result["metrics"]

    def test_qe_adapter_pareto_optimization(self):
        """Test QE adapter for multi-objective optimization."""
        adapter = create_mock_qe_adapter()

        # Evaluate multiple candidates
        candidates = []
        for i in range(20):
            result = adapter.evaluate(
                factor_batch=f"candidate_{i}",
                labels="returns",
                metrics=["rank_ic", "turnover"],
            )
            candidates.append(result)

        # Find Pareto frontier: maximize IC, minimize turnover
        pareto_frontier = []
        for candidate in candidates:
            ic = candidate["metrics"]["rank_ic"]
            turnover = candidate["metrics"]["turnover"]

            # Check if dominated
            is_dominated = False
            for other in candidates:
                other_ic = other["metrics"]["rank_ic"]
                other_turnover = other["metrics"]["turnover"]

                # Other dominates if better IC and lower turnover
                if other_ic > ic and other_turnover < turnover:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(candidate)

        # Should have some Pareto optimal solutions
        assert len(pareto_frontier) > 0
        assert len(pareto_frontier) <= len(candidates)


class TestAdapterErrorHandling:
    """Test adapter error handling and missing dependencies."""

    def test_seen_cache_without_adapter(self):
        """Test that SeenCache fails gracefully without adapter."""
        cache = SeenCache(fe_adapter=None)

        # Should raise when trying to compute hash
        with pytest.raises(RuntimeError, match="FE adapter required"):
            cache.compute_canonical_hash("ts_mean(close, 20)")

    def test_complexity_estimator_without_adapter(self):
        """Test that ComplexityEstimator fails gracefully without adapter."""
        estimator = ComplexityEstimator(fe_adapter=None)

        # Should raise when trying to estimate
        with pytest.raises(RuntimeError, match="FE adapter required"):
            estimator.estimate("ts_mean(close, 20)")

    def test_adapter_with_incompatible_object(self):
        """Test adapter behavior with incompatible objects."""

        class IncompatibleAdapter:
            """Adapter missing required methods."""

            def compute_canonical_hash(self, factor_definition):
                return "hash"

            # Missing other methods

        adapter = IncompatibleAdapter()
        cache = SeenCache(fe_adapter=adapter)

        # Should work for methods that exist
        hash_val = cache.compute_canonical_hash("factor")
        assert hash_val == "hash"

    def test_mock_adapter_reusability(self):
        """Test that mock adapters can be reused across components."""
        # Import at test time to get fresh protocol
        import sys
        import os

        fo_path = os.path.abspath(".")
        if fo_path not in sys.path:
            sys.path.insert(0, fo_path)

        from factor_optimizer.adapters.factor_engine import FactorEngineAdapter

        class ReusableMockAdapter:
            def compute_canonical_hash(self, factor_definition):
                import hashlib

                return hashlib.sha256(str(factor_definition).encode()).hexdigest()

            def estimate_complexity(self, factor_definition):
                return {
                    "operator_count": 1,
                    "max_depth": 1,
                    "lookback_periods": 0,
                    "estimated_cost": 1.0,
                    "domains": [],
                    "sources": [],
                }

            def validate_mutation(self, mutation, spec):
                return {"is_legal": True, "reason": "mock", "metadata": {}}

            def get_operator_metadata(self, operator_names=None):
                return {}

        adapter = ReusableMockAdapter()
        # Check it satisfies protocol
        assert isinstance(adapter, FactorEngineAdapter)

        # Use in seen cache
        cache = SeenCache(fe_adapter=adapter)
        hash1 = cache.compute_canonical_hash("factor1")

        # Use in complexity estimator
        estimator = ComplexityEstimator(fe_adapter=adapter)
        profile = estimator.estimate("factor1")

        # Both should work
        assert isinstance(hash1, str)
        assert profile.operator_count > 0
