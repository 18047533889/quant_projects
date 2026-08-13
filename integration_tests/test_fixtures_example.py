"""
Example integration test using the new fixtures.

Demonstrates end-to-end workflow with synthetic data, mock adapters, and assertions.
"""

import pytest
import numpy as np

from integration_tests.fixtures import (
    # Data generation
    generate_factor_panel,
    generate_label_bundle,
    generate_exposure_matrix,
    PanelConfig,
    DataCharacteristics,
    # Mock adapters
    create_stable_adapters,
    MockEvaluationBackend,
    AdapterStatus,
    # Test helpers
    assert_panel_shape,
    assert_factor_properties,
    assert_timing_consistency,
    assert_no_lookahead,
    assert_numeric_close,
    compare_evaluation_results,
    summarize_panel,
)


def test_end_to_end_factor_evaluation_workflow():
    """
    Complete end-to-end test: generate factors -> store in DA -> evaluate -> validate.
    """
    # Step 1: Configure and generate synthetic data
    config = PanelConfig(
        num_times=100,
        num_assets=50,
        num_factors=3,
        start_date="2024-01-01",
        seed=42,
    )

    characteristics = DataCharacteristics(
        factor_mean=0.0,
        factor_std=1.0,
        signal_strength=0.05,
        missing_rate=0.05,
    )

    factor_values, time_index, asset_ids = generate_factor_panel(config, characteristics)
    label_bundle = generate_label_bundle(factor_values, time_index, characteristics)

    # Step 2: Validate data quality
    assert_panel_shape(factor_values, (100, 50, 3), "factors")
    assert_factor_properties(
        factor_values,
        check_finite=True,
        check_range=(-5.0, 5.0),
        max_missing_rate=0.1,
    )
    assert_timing_consistency(
        label_bundle["decision_time"],
        label_bundle["label_start_time"],
        label_bundle["label_end_time"],
    )
    assert_no_lookahead(
        factor_values,
        label_bundle["values"],
        label_bundle["decision_time"],
        label_bundle["label_start_time"],
    )

    # Step 3: Setup mock adapters
    da_adapter, fe_adapter = create_stable_adapters(seed=42)
    eval_backend = MockEvaluationBackend(seed=42)

    # Step 4: Write factors to DataAccess
    for f_idx in range(config.num_factors):
        response = da_adapter.write_factor(
            factor_id=f"test_factor_{f_idx}",
            data=factor_values[:, :, f_idx],
            metadata={
                "timing": "daily",
                "frequency": "D",
                "pit_safe": True,
            },
        )
        assert response.status == AdapterStatus.SUCCESS

    # Step 5: Read factors back
    retrieved_factors = []
    for f_idx in range(config.num_factors):
        response = da_adapter.read_factor(
            factor_id=f"test_factor_{f_idx}",
            start_date="2024-01-01",
            end_date="2024-04-10",
        )
        assert response.status == AdapterStatus.SUCCESS
        retrieved_factors.append(response.data)

    # Step 6: Compute derived factor via FactorEngine
    def compute_combination(inputs, params):
        weights = params["weights"]
        result = np.zeros_like(inputs["factor_0"])
        for i, w in enumerate(weights):
            result += w * inputs[f"factor_{i}"]
        return result

    fe_adapter.register_operator("weighted_combination", compute_combination)

    inputs = {f"factor_{i}": factor_values[:, :, i] for i in range(3)}
    params = {"weights": [0.5, 0.3, 0.2]}

    response = fe_adapter.execute_operator(
        "weighted_combination",
        inputs=inputs,
        params=params,
    )
    assert response.status == AdapterStatus.SUCCESS
    combined_factor = response.data

    # Step 7: Evaluate factors
    def simple_ic(factor_vals, label_vals):
        # Flatten and compute correlation
        f_flat = factor_vals.flatten()
        l_flat = label_vals.flatten()
        valid_mask = ~(np.isnan(f_flat) | np.isnan(l_flat))
        if np.sum(valid_mask) < 10:
            return {"value": np.nan, "n_obs": 0}
        corr = np.corrcoef(f_flat[valid_mask], l_flat[valid_mask])[0, 1]
        return {"value": corr, "n_obs": np.sum(valid_mask)}

    eval_backend.register_metric("pearson_ic", simple_ic)

    # Evaluate each factor
    results = {}
    for f_idx in range(config.num_factors):
        response = eval_backend.evaluate(
            factor_values=factor_values[:, :, f_idx:f_idx+1],
            label_values=label_bundle["values"],
            metric_ids=("pearson_ic",),
        )
        assert response.status == AdapterStatus.SUCCESS
        results[f"factor_{f_idx}"] = response.data

    # Step 8: Validate evaluation results
    for f_idx in range(config.num_factors):
        ic_value = results[f"factor_{f_idx}"]["pearson_ic"]["value"]
        # Should have some signal (not exactly zero, but may be small)
        assert not np.isnan(ic_value)
        assert -1.0 <= ic_value <= 1.0

    # Step 9: Verify adapter tracking
    assert da_adapter.request_count > 0
    assert fe_adapter.execution_count > 0
    assert eval_backend.evaluation_count > 0

    print(f"✓ DA requests: {da_adapter.request_count}")
    print(f"✓ FE executions: {fe_adapter.execution_count}")
    print(f"✓ Evaluations: {eval_backend.evaluation_count}")


def test_preprocessing_and_neutralization_workflow():
    """
    Test preprocessing workflow with exposure neutralization.
    """
    # Generate data
    config = PanelConfig(num_times=100, num_assets=60, num_factors=2, seed=42)
    characteristics = DataCharacteristics(exposure_rank=3)

    factor_values, time_index, asset_ids = generate_factor_panel(config)
    exposures, exposure_names = generate_exposure_matrix(
        config.num_times,
        config.num_assets,
        characteristics,
        seed=42,
    )

    # Validate shapes
    assert_panel_shape(factor_values, (100, 60, 2))
    assert_panel_shape(exposures, (100, 60, 3))

    # Setup adapters
    _, fe_adapter = create_stable_adapters(seed=42)

    # Register neutralization operator
    def neutralize_factor(inputs, params):
        factor = inputs["factor"]
        exposures = inputs["exposures"]
        T, N = factor.shape

        neutralized = np.zeros_like(factor)
        for t in range(T):
            f_t = factor[t]
            e_t = exposures[t]

            valid_mask = ~np.isnan(f_t)
            if np.sum(valid_mask) < exposures.shape[2] + 10:
                neutralized[t] = f_t
                continue

            # Orthogonalize factor to exposures
            f_valid = f_t[valid_mask]
            e_valid = e_t[valid_mask]

            # Simple projection removal
            for k in range(e_valid.shape[1]):
                proj = np.dot(f_valid, e_valid[:, k]) / (np.dot(e_valid[:, k], e_valid[:, k]) + 1e-10)
                f_valid -= proj * e_valid[:, k]

            neutralized[t, valid_mask] = f_valid

        return neutralized

    fe_adapter.register_operator("neutralize", neutralize_factor)

    # Apply neutralization
    response = fe_adapter.execute_operator(
        "neutralize",
        inputs={
            "factor": factor_values[:, :, 0],
            "exposures": exposures,
        },
        params={},
    )

    assert response.status == AdapterStatus.SUCCESS
    neutralized_factor = response.data

    # Validate result
    assert_panel_shape(neutralized_factor, (100, 60))
    assert_factor_properties(neutralized_factor, check_finite=True)

    print("✓ Neutralization workflow completed")


def test_robustness_with_flaky_adapters():
    """
    Test system robustness with occasional adapter failures.
    """
    from integration_tests.fixtures import create_flaky_adapters

    # Generate data
    config = PanelConfig(num_times=50, num_assets=30, num_factors=2, seed=42)
    factor_values, _, _ = generate_factor_panel(config)

    # Setup flaky adapters (10-15% failure rate)
    da_adapter, fe_adapter = create_flaky_adapters(seed=42)

    # Attempt to write factors with retry logic
    max_retries = 5
    for f_idx in range(2):
        success = False
        for attempt in range(max_retries):
            response = da_adapter.write_factor(
                factor_id=f"factor_{f_idx}",
                data=factor_values[:, :, f_idx],
                metadata={"timing": "daily"},
            )
            if response.status == AdapterStatus.SUCCESS:
                success = True
                break

        # With 5 retries and 10% failure rate, should succeed
        assert success, f"Failed to write factor_{f_idx} after {max_retries} retries"

    print(f"✓ Handled flaky adapters with {da_adapter.request_count} total requests")


def test_data_summary_and_diagnostics():
    """
    Test data quality diagnostics using summary helpers.
    """
    # Generate data with various characteristics
    config = PanelConfig(num_times=200, num_assets=100, num_factors=5, seed=42)
    characteristics = DataCharacteristics(
        factor_mean=0.0,
        factor_std=1.0,
        missing_rate=0.08,
        factor_autocorr=0.2,
    )

    factor_values, time_index, asset_ids = generate_factor_panel(config, characteristics)

    # Generate summaries for each factor
    summaries = []
    for f_idx in range(config.num_factors):
        summary = summarize_panel(
            factor_values[:, :, f_idx],
            name=f"factor_{f_idx}",
        )
        summaries.append(summary)

    # Validate summaries
    for summary in summaries:
        assert not summary["all_missing"]
        assert 0.0 <= summary["missing_rate"] <= 0.15
        assert -0.5 < summary["mean"] < 0.5
        assert 0.5 < summary["std"] < 1.5
        assert summary["n_inf"] == 0

    print("✓ Data quality diagnostics:")
    for summary in summaries:
        print(f"  {summary['name']}: mean={summary['mean']:.3f}, "
              f"std={summary['std']:.3f}, missing={summary['missing_rate']:.1%}")


def test_evaluation_result_comparison():
    """
    Test comparing evaluation results from different runs.
    """
    # Run 1: Baseline evaluation
    config = PanelConfig(num_times=100, num_assets=50, num_factors=2, seed=42)
    factor_values_1, time_index, _ = generate_factor_panel(config)
    label_bundle_1 = generate_label_bundle(factor_values_1, time_index)

    eval_backend_1 = MockEvaluationBackend(seed=100)

    response_1 = eval_backend_1.evaluate(
        factor_values_1[:, :, 0:1],
        label_bundle_1["values"],
        metric_ids=("metric_a", "metric_b"),
    )

    # Run 2: Identical setup (should get same results)
    factor_values_2, _, _ = generate_factor_panel(config)  # Same seed
    label_bundle_2 = generate_label_bundle(factor_values_2, time_index)

    eval_backend_2 = MockEvaluationBackend(seed=100)  # Same seed

    response_2 = eval_backend_2.evaluate(
        factor_values_2[:, :, 0:1],
        label_bundle_2["values"],
        metric_ids=("metric_a", "metric_b"),
    )

    # Compare results
    comparison = compare_evaluation_results(
        response_1.data,
        response_2.data,
        rtol=1e-10,
    )

    assert comparison["metric_a"] is True
    assert comparison["metric_b"] is True

    print("✓ Evaluation results are reproducible")


if __name__ == "__main__":
    # Run tests
    test_end_to_end_factor_evaluation_workflow()
    test_preprocessing_and_neutralization_workflow()
    test_robustness_with_flaky_adapters()
    test_data_summary_and_diagnostics()
    test_evaluation_result_comparison()
    print("\n✓ All fixture example tests passed!")
