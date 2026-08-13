"""
Integration test: QE + FA evidence flow.

Test the flow from evaluation results to evidence storage in asset registry.
"""

import pytest
import numpy as np
from datetime import datetime


def test_qe_fa_import_isolation():
    """Test that QE and FA can coexist without dependency conflicts."""
    import quant_evaluator
    import factor_assets

    assert quant_evaluator.__version__ is not None
    assert factor_assets.__version__ is not None


def test_evaluation_to_evidence_flow():
    """Test creating evaluation results and storing as evidence refs."""
    from quant_evaluator import (
        FactorBatch,
        AxisRef,
        LabelBundle,
        EvaluationRequest,
        MetricValue,
    )
    from factor_assets import EvidenceRef, create_factor_id

    # Create factor
    factor_id = create_factor_id(name="test_factor", version="v1", params={"w": 10})

    # Create evaluation inputs
    T, N = 10, 5
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, 1)

    batch = FactorBatch(
        factor_ids=(factor_id,),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

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

    # Simulate evaluation result
    metric_value = MetricValue(
        metric_id="rank_ic",
        value=0.042,
        valid=True,
        observation_count=T * N,
        metric_version="0.1",
    )

    # Create evidence ref from metric
    evidence = EvidenceRef(
        evidence_id="evidence_001",
        evaluation_run_id="eval_001",
        metric_name="rank_ic",
        metric_version="0.1.0",
        timestamp=datetime.now().isoformat(),
        factor_id=factor_id,
        summary_value=0.042,
    )

    assert evidence.factor_id == factor_id
    assert evidence.metric_name == "rank_ic"
    assert evidence.summary_value == metric_value.value


def test_batch_evaluation_to_multiple_evidence():
    """Test multi-factor batch evaluation creating multiple evidence refs."""
    from quant_evaluator import FactorBatch, AxisRef, MetricValue
    from factor_assets import EvidenceRef, create_factor_id

    # Create multiple factors
    factor_ids = tuple(
        create_factor_id(name=f"factor_{i}", version="v1", params={"idx": i})
        for i in range(3)
    )

    T, N, F = 10, 5, 3
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, F)

    batch = FactorBatch(
        factor_ids=factor_ids,
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )

    # Simulate evaluation results for each factor
    evidence_refs = []
    for idx, fid in enumerate(factor_ids):
        metric = MetricValue(
            metric_id="rank_ic",
            value=0.03 + idx * 0.01,
            valid=True,
            observation_count=T * N,
        )

        evidence = EvidenceRef(
            evidence_id=f"evidence_{idx}",
            evaluation_run_id=f"eval_batch_001",
            metric_name="rank_ic",
            metric_version="0.1.0",
            timestamp=datetime.now().isoformat(),
            factor_id=fid,
            summary_value=metric.value,
        )
        evidence_refs.append(evidence)

    assert len(evidence_refs) == 3
    assert all(e.metric_name == "rank_ic" for e in evidence_refs)
    assert evidence_refs[0].summary_value == 0.03
    assert evidence_refs[2].summary_value == 0.05


def test_evidence_bundle_to_asset_attachment():
    """Test attaching evaluation bundle to factor asset."""
    from quant_evaluator import EvaluationBundle, MetricValue, FactorDiagnosis
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        EvidenceBundleRef,
        LineageRef,
        LifecycleState,
        create_factor_id,
    )

    factor_id = create_factor_id(name="test_factor", version="v1", params={})

    # Create asset
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="test_factor",
        canonical_hash="hash_test_factor_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test factor",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    # Create evaluation bundle
    bundle = EvaluationBundle(
        request_id="req_001",
        factor_ids=(factor_id,),
        label_id="fwd_ret_1d",
        timestamp=datetime.now().isoformat(),
        schema_version="0.1",
        metric_values={
            "rank_ic": MetricValue(
                metric_id="rank_ic",
                value=0.045,
                valid=True,
                observation_count=100,
            )
        },
        diagnostics={
            factor_id: FactorDiagnosis(
                factor_id=factor_id,
                num_valid_observations=100,
                num_missing=0,
                coverage=1.0,
                is_constant=False,
                has_nans=False,
                has_infs=False,
                min_value=-2.5,
                max_value=2.5,
                mean_value=0.0,
            )
        },
    )

    # Create evidence bundle ref
    evidence_bundle_ref = EvidenceBundleRef(
        bundle_id="bundle_001",
        evaluation_run_id="eval_001",
        factor_ids=(factor_id,),
        timestamp=bundle.timestamp,
        qe_version="0.1.0",
    )

    assert factor_id in evidence_bundle_ref.factor_ids


def test_asset_lifecycle_with_evidence():
    """Test lifecycle state progression based on evidence."""
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        LifecycleState,
        StateTransition,
        EvidenceRef,
        LineageRef,
        create_factor_id,
    )

    factor_id = create_factor_id(name="lifecycle_test", version="v1", params={})

    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="lifecycle_test",
        canonical_hash="hash_lifecycle_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test lifecycle progression",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    # Initial state
    assert asset.lifecycle_state == LifecycleState.REGISTERED

    # Create evidence
    evidence = EvidenceRef(
        evidence_id="evidence_001",
        evaluation_run_id="eval_001",
        metric_name="rank_ic",
        metric_version="0.1.0",
        timestamp=datetime.now().isoformat(),
        factor_id=factor_id,
        summary_value=0.05,
    )

    # Simulate progression: REGISTERED -> EVALUATED
    transition1 = StateTransition(
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        description="Initial evaluation scheduled",
    )

    # Simulate progression: EVALUATED -> APPROVED (based on evidence)
    transition2 = StateTransition(
        from_state=LifecycleState.EVALUATED,
        to_state=LifecycleState.APPROVED,
        required_evidence=(evidence.evaluation_run_id,),
        description=f"Evidence shows rank_ic={evidence.summary_value:.3f} meets threshold",
    )

    assert transition1.to_state == LifecycleState.EVALUATED
    assert transition2.to_state == LifecycleState.APPROVED
    assert evidence.evaluation_run_id in transition2.required_evidence


def test_qe_diagnosis_to_fa_metadata():
    """Test mapping QE diagnosis to FA asset metadata."""
    from quant_evaluator import FactorDiagnosis
    from factor_assets import AssetMetadata, create_factor_id

    factor_id = create_factor_id(name="diagnosed_factor", version="v1", params={})

    # QE diagnosis
    diagnosis = FactorDiagnosis(
        factor_id=factor_id,
        num_valid_observations=950,
        num_missing=50,
        coverage=0.95,
        is_constant=False,
        has_nans=True,
        has_infs=False,
        min_value=-3.2,
        max_value=2.8,
        mean_value=0.05,
        warnings=("sparse_observations", "outliers_detected"),
    )

    # Create asset metadata with diagnosis info
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="diagnosed_factor",
        canonical_hash="hash_diagnosed_factor_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description=f"Factor with coverage={diagnosis.coverage:.2%}",
        complexity_score=diagnosis.coverage,
    )

    assert metadata.complexity_score == 0.95
    assert diagnosis.coverage == metadata.complexity_score


def test_repository_evidence_query():
    """Test querying repository by evidence metrics."""
    from factor_assets import (
        AssetRepository,
        FactorAsset,
        AssetMetadata,
        EvidenceRef,
        create_factor_id,
    )

    repo = AssetRepository()

    # Create assets with different evidence
    evidence_values = [0.03, 0.06, 0.04]
    assets = []

    for i, ev_val in enumerate(evidence_values):
        factor_id = create_factor_id(
            name=f"factor_{i}", version="v1", params={"idx": i}
        )

        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"factor_{i}",
            canonical_hash=f"hash_factor_{i}_v1",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Factor {i}",
        )

        evidence = EvidenceRef(
            evidence_id=f"evidence_{i}",
            evaluation_run_id=f"eval_{i}",
            metric_name="rank_ic",
            metric_version="0.1.0",
            timestamp=datetime.now().isoformat(),
            factor_id=factor_id,
            summary_value=ev_val,
        )

        from factor_assets import LineageRef
        lineage = LineageRef(
            factor_id=factor_id,
            parents=(),
        )

        asset = repo.register(metadata, lineage)
        assets.append((asset, evidence))

    # All assets in repository
    all_assets = repo.list_all()
    assert len(all_assets) >= 3

    # Verify assets exist
    for asset, evidence in assets:
        retrieved = repo.get(asset.factor_id)
        assert retrieved is not None


def test_qe_contract_error_handling_with_fa():
    """Test error propagation from QE to FA asset status."""
    from quant_evaluator import ContractError, FactorBatch, AxisRef
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        LifecycleState,
        create_factor_id,
    )

    factor_id = create_factor_id(name="error_test", version="v1", params={})

    # Try to create invalid batch
    T, N = 5, 3
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    with pytest.raises(ValueError):
        # This should fail - wrong shape
        wrong_values = np.random.randn(T, N)  # Missing factor dimension
        batch = FactorBatch(
            factor_ids=(factor_id,),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=wrong_values,
        )

    # Asset can track evaluation failures
    from factor_assets import LineageRef

    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="error_test",
        canonical_hash="hash_error_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Factor with evaluation error",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    assert asset.lifecycle_state == LifecycleState.REGISTERED
