"""Example integration of validation module with factor computation.

Demonstrates:
1. Schema validation for factor batches
2. Quality checks on computed factors
3. Input sanitization pipeline
4. Integration with strict mode for production
"""
import numpy as np
from datetime import datetime

# Example: Post-computation validation pipeline


def compute_factor_with_validation():
    """Compute a factor with full validation pipeline."""
    from validation import (
        check_data_quality,
        sanitize_factor_inputs,
        validate_factor_batch,
        ValidationConfig,
    )

    # Simulate factor computation
    timestamps = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    instruments = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]

    # Generate sample data with some issues
    np.random.seed(42)
    values = np.random.randn(5, 4)
    values[0, 0] = np.inf  # inject inf
    values[1, 1] = np.nan  # inject nan
    values[2, 2] = 100.0   # outlier

    print("=" * 80)
    print("FACTOR VALIDATION PIPELINE EXAMPLE")
    print("=" * 80)

    # Step 1: Quality check
    print("\n[1] Quality Check")
    print("-" * 80)
    report = check_data_quality(
        values,
        max_nan_fraction=0.5,
        outlier_method="iqr",
        max_outlier_fraction=0.1,
    )
    print(report.summary())

    # Step 2: Sanitization if needed
    if not report.passed or report.nan_fraction > 0 or report.inf_fraction > 0:
        print("\n[2] Sanitization Required")
        print("-" * 80)
        result = sanitize_factor_inputs(
            values,
            replace_inf="clip",
            max_abs_value=1e10,
            winsorize=(0.05, 0.95),
        )
        print(result.summary())
        values = result.sanitized
    else:
        print("\n[2] No sanitization needed")

    # Step 3: Schema validation (strict mode)
    print("\n[3] Schema Validation")
    print("-" * 80)
    config = ValidationConfig(
        strict=True,
        allow_nan=True,  # allow after sanitization
        allow_inf=False,
        max_nan_fraction=0.2,
    )

    try:
        batch = validate_factor_batch(
            factor_name="momentum_20d",
            timestamps=timestamps,
            instruments=instruments,
            values=values,
            metadata={
                "source": "example",
                "version": "1.0",
                "computation_time": datetime.now().isoformat(),
            },
            config=config,
        )
        print(f"✓ Factor batch validated: {batch.factor_name}")
        print(f"  Shape: {batch.values.shape}")
        print(f"  Timestamps: {len(batch.timestamps)}")
        print(f"  Instruments: {len(batch.instruments)}")
        print(f"  Metadata: {batch.metadata}")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")

    print("\n" + "=" * 80)


def example_feature_bundle_validation():
    """Example of feature bundle validation."""
    from validation import validate_feature_bundle

    print("\n" + "=" * 80)
    print("FEATURE BUNDLE VALIDATION EXAMPLE")
    print("=" * 80)

    values = np.random.randn(100)
    row_ids = [f"row_{i:03d}" for i in range(100)]

    bundle = validate_feature_bundle(
        canonical="momentum",
        operator_semantic_version="1.0.0",
        params={"window": 20, "decay": 0.95},
        normalized_ast_hash="abc123def456789",
        source_snapshot="snapshot_2024_01_01",
        values=values,
        row_ids=row_ids,
    )

    print(f"✓ Feature bundle validated: {bundle.canonical}")
    print(f"  Operator version: {bundle.operator_semantic_version}")
    print(f"  Parameters: {bundle.params}")
    print(f"  Values shape: {bundle.values.shape if bundle.values is not None else 'None'}")
    print(f"  Row count: {len(bundle.row_ids)}")


def example_prediction_validation():
    """Example of prediction batch validation."""
    from validation import validate_prediction_batch

    print("\n" + "=" * 80)
    print("PREDICTION BATCH VALIDATION EXAMPLE")
    print("=" * 80)

    predictions = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
    row_ids = ["stock_a", "stock_b", "stock_c", "stock_d", "stock_e"]
    timestamps = ["2024-01-01"] * 5

    batch = validate_prediction_batch(
        values=predictions,
        row_ids=row_ids,
        timestamps=timestamps,
        status="ok",
        model_version="v1.0.0",
        artifact_id="artifact_12345",
        allow_nan=False,
    )

    print(f"✓ Prediction batch validated")
    print(f"  Status: {batch.status}")
    print(f"  Model version: {batch.model_version}")
    print(f"  Predictions: {len(batch.values)}")
    print(f"  Range: [{batch.values.min():.4f}, {batch.values.max():.4f}]")


if __name__ == "__main__":
    # Run examples
    compute_factor_with_validation()
    example_feature_bundle_validation()
    example_prediction_validation()

    print("\n" + "=" * 80)
    print("All validation examples completed successfully!")
    print("=" * 80 + "\n")
