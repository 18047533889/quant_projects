"""
Integration test: QE + FP pipeline.

Test the flow from raw factors through preprocessing to evaluation.
"""

import pytest
import numpy as np
from datetime import datetime


def test_qe_import_isolation():
    """Test that QE can be imported independently without FP dependencies."""
    import quant_evaluator
    from quant_evaluator import (
        FactorBatch,
        AxisRef,
        LabelBundle,
        EvaluationRequest,
    )

    assert quant_evaluator.__version__ is not None
    assert FactorBatch is not None
    assert AxisRef is not None


def test_fp_import_isolation():
    """Test that FP can be imported independently without QE dependencies."""
    import factor_preprocess
    from factor_preprocess import (
        PreprocessingPolicy,
        TransformSpec,
        FeatureBundle,
    )

    assert factor_preprocess.__version__ is not None
    assert PreprocessingPolicy is not None


def test_qe_basic_batch_creation():
    """Test creating a FactorBatch for QE evaluation."""
    from quant_evaluator import FactorBatch, AxisRef

    # Create synthetic factor data
    T, N, F = 10, 5, 2
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    values = np.random.randn(T, N, F)

    # Use helper to create factor IDs
    factor_ids = (pytest.make_factor_id("factor_a"), pytest.make_factor_id("factor_b"))

    batch = FactorBatch(
        factor_ids=factor_ids,
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
        layout="wide",
        dtype="float64",
    )

    assert batch.num_factors == 2
    assert batch.num_times == T
    assert batch.num_assets == N
    assert batch.values.shape == (T, N, F)


def test_qe_label_bundle_creation():
    """Test creating a LabelBundle with explicit timing."""
    from quant_evaluator import LabelBundle

    T, N = 10, 5
    values = np.random.randn(T, N)

    decision_times = tuple(f"2024-01-{i:02d}" for i in range(1, T + 1))
    label_start = tuple(f"2024-01-{i:02d}" for i in range(2, T + 2))
    label_end = tuple(f"2024-01-{i:02d}" for i in range(3, T + 3))

    label_bundle = LabelBundle(
        target_id="fwd_ret_1d",
        values=values,
        horizon=1,
        execution_delay=0,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    assert label_bundle.target_id == "fwd_ret_1d"
    assert label_bundle.horizon == 1
    assert label_bundle.num_observations() == T


def test_qe_evaluation_request_creation():
    """Test creating an EvaluationRequest."""
    from quant_evaluator import (
        EvaluationRequest,
        FactorBatch,
        AxisRef,
        LabelBundle,
    )

    T, N, F = 10, 5, 1

    # Create batch
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, F)

    batch = FactorBatch(
        factor_ids=("test_factor",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # Create labels
    label_values = np.random.randn(T, N)
    decision_times = tuple(f"2024-01-{i:02d}" for i in range(1, T + 1))
    label_start = tuple(f"2024-01-{i:02d}" for i in range(2, T + 2))
    label_end = tuple(f"2024-01-{i:02d}" for i in range(3, T + 3))

    labels = LabelBundle(
        target_id="fwd_ret_1d",
        values=label_values,
        horizon=1,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Create request
    request = EvaluationRequest(
        batch_or_factor_ids=batch,
        label_bundle=labels,
        metric_ids=("pearson_ic", "rank_ic", "coverage"),
        tier="core",
    )

    assert request.batch_or_factor_ids == batch
    assert request.label_bundle == labels
    assert "pearson_ic" in request.metric_ids


def test_fp_basic_policy_creation():
    """Test creating a preprocessing policy."""
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    policy = PreprocessingPolicy(
        policy_id="test_policy_001",
        transforms=[
            TransformSpec(
                name="winsorize",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0",
                parameters={"lower": 0.01, "upper": 0.99}
            ),
            TransformSpec(
                name="standardize",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0",
                parameters={"method": "robust"}
            ),
        ],
    )

    assert len(policy.transforms) == 2
    assert policy.transforms[0].name == "winsorize"


def test_qe_fp_contract_compatibility():
    """Test that QE FactorBatch shape is compatible with FP expectations."""
    from quant_evaluator import FactorBatch, AxisRef

    T, N, F = 20, 10, 3
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, F)

    batch = FactorBatch(
        factor_ids=("f1", "f2", "f3"),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # FP would expect (T, N, F) shaped data
    assert batch.values.shape == (T, N, F)
    assert batch.values.ndim == 3


def test_qe_error_taxonomy():
    """Test QE error classes are correctly imported."""
    from quant_evaluator import (
        QuantEvaluatorError,
        ContractError,
        DataError,
        CapabilityError,
        InsufficientObservations,
    )

    assert issubclass(ContractError, QuantEvaluatorError)
    assert issubclass(DataError, QuantEvaluatorError)
    assert issubclass(InsufficientObservations, DataError)


def test_fp_package_info():
    """Test FP package metadata."""
    from factor_preprocess import package_info

    info = package_info()

    assert info["name"] == "factor_preprocess"
    assert info["version"] is not None
    assert "capabilities" in info
    assert info["capabilities"]["stateless_transforms"]
    assert info["capabilities"]["fitted_transforms"]


def test_qe_batch_validation():
    """Test FactorBatch validates inputs correctly."""
    from quant_evaluator import FactorBatch, AxisRef

    T, N, F = 5, 3, 1
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    # Valid batch
    values = np.random.randn(T, N, F)
    batch = FactorBatch(
        factor_ids=("test_factor",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )
    assert batch.num_factors == 1

    # Test validation: empty factor_ids
    with pytest.raises(ValueError, match="factor_ids cannot be empty"):
        FactorBatch(
            factor_ids=(),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )


def test_qe_label_bundle_validation():
    """Test LabelBundle validates timing requirements."""
    from quant_evaluator import LabelBundle

    T, N = 5, 3
    values = np.random.randn(T, N)
    decision_times = tuple(f"2024-01-{i:02d}" for i in range(1, T + 1))
    label_start = tuple(f"2024-01-{i:02d}" for i in range(2, T + 2))
    label_end = tuple(f"2024-01-{i:02d}" for i in range(3, T + 3))

    # Valid bundle
    bundle = LabelBundle(
        target_id="fwd_ret",
        values=values,
        horizon=1,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )
    assert bundle.horizon == 1

    # Test validation: missing decision_time
    with pytest.raises(ValueError, match="decision_time .* is required"):
        LabelBundle(
            target_id="fwd_ret",
            values=values,
            horizon=1,
            decision_time=(),
            label_start_time=label_start,
            label_end_time=label_end,
        )
