"""
Example usage of the diagnosis module.
"""

import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.diagnosis import (
    diagnose_factor,
    diagnose_all_factors,
    WarningSystem,
    WarningSeverity,
)


def example_basic_diagnosis():
    """Basic factor diagnosis example."""
    # Create a test factor batch
    T, N = 100, 50
    values = np.random.randn(T, N, 1)

    # Add some issues for demonstration
    values[0:20, :, 0] = np.nan  # Missing data

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("momentum_5d",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # Diagnose the factor
    diagnosis = diagnose_factor(batch, factor_idx=0)

    print(f"Factor: {diagnosis.factor_id}")
    print(f"Coverage: {diagnosis.coverage:.2%}")
    print(f"Valid observations: {diagnosis.num_valid_observations}")
    print(f"Missing: {diagnosis.num_missing}")
    print(f"Is constant: {diagnosis.is_constant}")
    print(f"Has NaNs: {diagnosis.has_nans}")
    print(f"Has Infs: {diagnosis.has_infs}")
    print(f"Range: [{diagnosis.min_value:.3f}, {diagnosis.max_value:.3f}]")
    print(f"Mean: {diagnosis.mean_value:.3f}")

    if diagnosis.warnings:
        print("\nWarnings:")
        for warning in diagnosis.warnings:
            print(f"  - {warning}")


def example_warning_system():
    """Warning system with structured diagnostics."""
    # Create test batch with multiple issues
    T, N = 100, 50
    values = np.random.randn(T, N, 1)

    # Add critical coverage issue
    values[0:60, :, 0] = np.nan

    # Add extreme outlier
    values[70, 0, 0] = 100.0

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("test_factor",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # Create diagnosis
    diagnosis = diagnose_factor(batch, factor_idx=0)

    # Create warning system
    ws = WarningSystem(
        missing_rate_high=0.5,
        outlier_z_threshold=5.0,
    )

    # Diagnose with structured warnings
    warnings = ws.diagnose_factor(diagnosis, factor_batch=batch, factor_idx=0)

    print(f"\nFound {len(warnings)} issues:\n")
    print(ws.format_warnings(warnings, include_details=True))

    # Filter by severity
    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
    if critical:
        print(f"\n{len(critical)} critical issues require immediate attention:")
        for w in critical:
            print(f"  [{w.category}] {w.message}")


def example_batch_diagnosis():
    """Diagnose multiple factors in a batch."""
    T, N, F = 100, 50, 3
    values = np.random.randn(T, N, F)

    # Create different issues for each factor
    values[0:60, :, 0] = np.nan  # Factor 0: low coverage
    values[:, :, 1] = 5.0  # Factor 1: constant
    values[0, 0, 2] = 1000.0  # Factor 2: extreme outlier

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("factor_a", "factor_b", "factor_c"),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # Diagnose all factors
    diagnostics = diagnose_all_factors(batch)

    # Create warning system
    ws = WarningSystem()

    # Get warnings for all factors
    warnings_by_factor = ws.diagnose_all_factors(diagnostics, factor_batch=batch)

    print("\nBatch Diagnosis Summary:")
    print("=" * 60)

    for factor_id, warnings in warnings_by_factor.items():
        diag = diagnostics[factor_id]
        print(f"\n{factor_id}:")
        print(f"  Coverage: {diag.coverage:.2%}")
        print(f"  Valid: {diag.num_valid_observations}, Missing: {diag.num_missing}")

        if warnings:
            print(f"  Issues ({len(warnings)}):")
            for w in warnings:
                print(f"    [{w.severity.value.upper()}] {w.message}")
        else:
            print("  Status: OK")


if __name__ == "__main__":
    print("Example 1: Basic Diagnosis")
    print("=" * 60)
    example_basic_diagnosis()

    print("\n\n")
    print("Example 2: Structured Warning System")
    print("=" * 60)
    example_warning_system()

    print("\n\n")
    print("Example 3: Batch Diagnosis")
    example_batch_diagnosis()
