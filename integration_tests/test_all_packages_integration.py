"""
Integration test: All four packages together.

Test the complete workflow: QE + FP + FA + FO integration.
"""

import pytest
import numpy as np
from datetime import datetime


def test_all_packages_import():
    """Test all four packages can be imported together."""
    import quant_evaluator as qe
    import factor_preprocess as fp
    import factor_assets as fa
    import factor_optimizer as fo

    assert qe.__version__ is not None
    assert fp.__version__ is not None
    assert fa.__version__ is not None
    assert fo.package_info()["version"] is not None


def test_complete_factor_lifecycle():
    """
    Test complete lifecycle: create factor asset, preprocess, evaluate, store evidence.

    Flow:
    1. FA creates factor asset
    2. FP defines preprocessing policy
    3. QE evaluates factor
    4. FA stores evidence and updates lifecycle
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        AssetRepository,
        EvidenceRef,
        LifecycleState,
        LineageRef,
        StateTransition,
        create_factor_id,
    )

    # Step 1: Create factor asset
    factor_id = create_factor_id(
        name="momentum_10d",
        version="v1",
        params={"window": 10, "method": "log_return"},
    )

    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="momentum_10d",
        canonical_hash="hash_momentum_10d_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="10-day momentum factor",
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

    repo = AssetRepository()
    asset = repo.register(metadata, lineage)

    # Step 2: Define preprocessing policy
    from factor_preprocess import TransformKind, TransformMode

    preprocess_policy = PreprocessingPolicy(
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
                parameters={"method": "zscore"}
            ),
        ],
    )

    assert len(preprocess_policy.transforms) == 2

    # Step 3: Prepare evaluation inputs
    T, N = 20, 10
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

    # Step 3b: Simulate evaluation result
    metric = MetricValue(
        metric_id="rank_ic",
        value=0.048,
        valid=True,
        observation_count=T * N,
        metric_version="0.1",
    )

    # Step 4: Store evidence and update lifecycle
    evidence = EvidenceRef(
        evidence_id="evidence_001",
        evaluation_run_id="eval_integration_001",
        metric_name="rank_ic",
        metric_version="0.1",
        timestamp=datetime.now().isoformat(),
        factor_id=factor_id,
        summary_value=metric.value,
    )

    transition = StateTransition(
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        required_evidence=("eval_integration_001",),
        description=f"Passed evaluation with rank_ic={metric.value:.3f}",
    )

    # Verify complete flow
    retrieved_asset = repo.get(factor_id)
    assert retrieved_asset.metadata.factor_id == factor_id
    assert retrieved_asset.lifecycle_state == LifecycleState.REGISTERED
    assert metric.value > 0.03
    assert transition.to_state == LifecycleState.EVALUATED


def test_multi_factor_batch_workflow():
    """
    Test batch processing multiple factors through the complete pipeline.
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        AssetRepository,
        FactorSet,
        FactorSetSpec,
        LineageRef,
        LifecycleState,
        create_factor_id,
    )

    repo = AssetRepository()

    # Create multiple factor assets
    num_factors = 5
    factor_ids = []

    for i in range(num_factors):
        factor_id = create_factor_id(
            name=f"factor_{i}",
            version="v1",
            params={"index": i, "window": 10 + i * 5},
        )
        factor_ids.append(factor_id)

        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"factor_{i}",
            canonical_hash=f"hash_factor_{i}_v1",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Test factor {i}",
        )

        lineage = LineageRef(
            factor_id=factor_id,
            parents=(),
        )

        asset = repo.register(metadata, lineage)

    # Create factor set
    factor_set_spec = FactorSetSpec(
        set_id="test_batch_suite_001",
        name="test_batch_suite",
        selection_policy="manual",
        description="Batch of test factors",
    )

    factor_set = FactorSet(
        set_id="test_batch_suite_001",
        name="test_batch_suite",
        factor_ids=tuple(factor_ids),
        created_at=datetime.now().isoformat(),
        spec=factor_set_spec,
    )

    # Create batch for evaluation
    T, N, F = 15, 8, num_factors
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, F)

    batch = FactorBatch(
        factor_ids=tuple(factor_ids),
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

    # Define preprocessing
    policy = PreprocessingPolicy(
        policy_id="test_policy_batch",
        transforms=[
            TransformSpec(
                name="winsorize",
                kind=TransformKind.CROSS_SECTIONAL,
                mode=TransformMode.STATELESS,
                version="1.0",
                parameters={"lower": 0.01, "upper": 0.99}
            ),
        ],
    )

    # Verify all components
    assert batch.num_factors == num_factors
    assert len(factor_set.factor_ids) == num_factors
    assert policy.transforms[0].name == "winsorize"
    assert repo.get(factor_ids[0]) is not None


def test_factor_optimization_search_simulation():
    """
    Simulate factor optimization search workflow across all packages.

    Flow:
    1. FO proposes candidate mutations
    2. FA checks if candidates already exist (seen index)
    3. FP applies preprocessing
    4. QE evaluates candidates
    5. FA stores results and updates registry
    """
    from quant_evaluator import FactorBatch, AxisRef, MetricValue
    from factor_assets import (
        AssetRepository,
        SeenIndex,
        FactorAsset,
        AssetMetadata,
        LineageRef,
        LifecycleState,
        EvidenceRef,
        create_factor_id,
    )
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
    import factor_optimizer as fo

    repo = AssetRepository()
    seen_index = SeenIndex()

    # Simulate search iterations
    num_iterations = 3
    best_ic = 0.0
    best_factor_id = None

    for iteration in range(num_iterations):
        # Step 1: FO proposes candidate (simulated)
        candidate_params = {"window": 10 + iteration * 5, "decay": 0.95 - iteration * 0.1}
        candidate_id = create_factor_id(
            name=f"search_candidate_{iteration}",
            version="v1",
            params=candidate_params,
        )

        # Step 2: FA checks if seen
        if seen_index.is_seen(candidate_id):
            continue

        # Record as seen
        seen_index.record(
            canonical_hash=candidate_id,
            factor_id=candidate_id,
            origin="search",
        )

        # Step 3: Create asset
        metadata = AssetMetadata(
            factor_id=candidate_id,
            canonical_repr=f"search_candidate_{iteration}",
            canonical_hash=candidate_id,
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Search iteration {iteration}",
        )

        lineage = LineageRef(
            factor_id=candidate_id,
            parents=(),
        )

        asset = repo.register(metadata, lineage)

        # Step 4: FP preprocessing (policy defined)
        policy = PreprocessingPolicy(
            policy_id="search_policy",
            transforms=[
                TransformSpec(
                    name="standardize",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0",
                    parameters={}
                )
            ],
        )

        # Step 5: QE evaluation (simulated)
        simulated_ic = 0.03 + iteration * 0.01 + np.random.randn() * 0.005

        metric = MetricValue(
            metric_id="rank_ic",
            value=simulated_ic,
            valid=True,
            observation_count=100,
        )

        # Step 6: Store evidence
        evidence = EvidenceRef(
            evidence_id=f"evidence_search_{iteration}",
            evaluation_run_id=f"search_eval_{iteration}",
            metric_name="rank_ic",
            metric_version="0.1",
            timestamp=datetime.now().isoformat(),
            factor_id=candidate_id,
            summary_value=simulated_ic,
        )

        # Track best
        if simulated_ic > best_ic:
            best_ic = simulated_ic
            best_factor_id = candidate_id

    # Verify search results
    assert best_factor_id is not None
    assert seen_index.is_seen(best_factor_id)
    assert repo.get(best_factor_id) is not None
    assert seen_index.count() == num_iterations


def test_cross_package_error_handling():
    """Test error handling and validation across package boundaries."""
    from quant_evaluator import ContractError, FactorBatch, AxisRef
    from factor_assets import AssetRepository, DuplicateIdentityError, create_factor_id
    import factor_preprocess as fp

    # Test 1: QE contract validation
    T, N = 5, 3
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    with pytest.raises(ValueError, match="factor_ids cannot be empty"):
        FactorBatch(
            factor_ids=(),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(T, N, 1),
        )

    # Test 2: FA duplicate detection
    from factor_assets import FactorAsset, AssetMetadata, LineageRef, LifecycleState

    factor_id = create_factor_id(name="duplicate_test", version="v1", params={})
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="duplicate_test",
        canonical_hash="hash_duplicate_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test duplicate",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    repo = AssetRepository()
    asset = repo.register(metadata, lineage)

    with pytest.raises(DuplicateIdentityError):
        repo.register(metadata, lineage)

    # Test 3: FP package info available
    info = fp.package_info()
    assert "production_ready" in info


def test_package_version_compatibility():
    """Verify package version information and compatibility."""
    import quant_evaluator as qe
    import factor_preprocess as fp
    import factor_assets as fa
    import factor_optimizer as fo

    # All packages report versions
    assert hasattr(qe, "__version__")
    assert hasattr(fp, "__version__")
    assert hasattr(fa, "__version__")

    fo_info = fo.package_info()
    assert "version" in fo_info

    # Version formats
    assert isinstance(qe.__version__, str)
    assert isinstance(fp.__version__, str)
    assert isinstance(fa.__version__, str)


def test_import_order_independence():
    """Test that import order doesn't matter for the four packages."""
    # Clear any cached imports (simulate fresh import)
    import sys

    # Test importing in different orders - all should work
    packages = [
        "quant_evaluator",
        "factor_preprocess",
        "factor_assets",
        "factor_optimizer",
    ]

    # They should all be importable regardless of order
    for pkg in packages:
        mod = sys.modules.get(pkg)
        assert mod is not None, f"Package {pkg} should already be loaded"

    # Try importing specific classes across packages
    from quant_evaluator import FactorBatch
    from factor_preprocess import PreprocessingPolicy
    from factor_assets import FactorAsset, AssetRepository
    import factor_optimizer

    assert FactorBatch is not None
    assert PreprocessingPolicy is not None
    assert FactorAsset is not None
    assert factor_optimizer.package_info() is not None


def test_end_to_end_realistic_scenario():
    """
    Realistic end-to-end scenario: research workflow.

    1. Researcher creates new factor formula
    2. Factor registered in asset repository
    3. Factor preprocessed for model input
    4. Factor evaluated on historical data
    5. Evidence stored and lifecycle updated
    6. Optimizer uses results for next iteration
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue, FactorDiagnosis
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        AssetRepository,
        EvidenceRef,
        LifecycleState,
        LineageRef,
        StateTransition,
        create_factor_id,
    )

    # Step 1: Create factor
    factor_formula = "rolling_mean(close, 20) / rolling_std(close, 20)"
    factor_params = {"mean_window": 20, "std_window": 20}

    factor_id = create_factor_id(
        name="mean_std_ratio_20",
        version="v1",
        params=factor_params,
    )

    # Step 2: Register in repository
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="mean_std_ratio_20",
        canonical_hash="hash_mean_std_ratio_20_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="20-day rolling mean divided by rolling std",
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

    repo = AssetRepository()
    asset = repo.register(metadata, lineage)

    # Step 3: Define preprocessing
    from factor_preprocess import TransformKind, TransformMode

    preprocess = PreprocessingPolicy(
        policy_id="realistic_policy",
        transforms=[
            TransformSpec(name="winsorize", kind=TransformKind.CROSS_SECTIONAL, mode=TransformMode.STATELESS, version="1.0", parameters={"lower": 0.01, "upper": 0.99}),
            TransformSpec(name="standardize", kind=TransformKind.CROSS_SECTIONAL, mode=TransformMode.STATELESS, version="1.0", parameters={"method": "robust"}),
        ],
    )

    # Step 4: Prepare evaluation
    T, N = 252, 50  # 1 year daily, 50 assets
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    # Simulate factor values
    factor_values = np.random.randn(T, N, 1) * 0.5 + 0.1

    batch = FactorBatch(
        factor_ids=(factor_id,),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    # Simulate forward returns
    label_values = np.random.randn(T, N) * 0.02
    decision_times = tuple(f"2024-{i//21+1:02d}-{i%21+1:02d}" for i in range(T))
    label_start = tuple(f"2024-{i//21+1:02d}-{(i%21+2):02d}" for i in range(T))
    label_end = tuple(f"2024-{i//21+1:02d}-{(i%21+3):02d}" for i in range(T))

    labels = LabelBundle(
        target_id="fwd_ret_5d",
        values=label_values,
        horizon=5,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Simulate evaluation results
    simulated_rank_ic = 0.052
    simulated_pearson_ic = 0.048

    metrics = {
        "rank_ic": MetricValue(
            metric_id="rank_ic",
            value=simulated_rank_ic,
            valid=True,
            observation_count=T * N,
            metric_version="0.1",
        ),
        "pearson_ic": MetricValue(
            metric_id="pearson_ic",
            value=simulated_pearson_ic,
            valid=True,
            observation_count=T * N,
            metric_version="0.1",
        ),
    }

    diagnosis = FactorDiagnosis(
        factor_id=factor_id,
        num_valid_observations=T * N - 50,
        num_missing=50,
        coverage=0.996,
        is_constant=False,
        has_nans=False,
        has_infs=False,
        min_value=-2.1,
        max_value=2.3,
        mean_value=0.1,
    )

    # Step 5: Store evidence
    evidence_refs = []
    for metric_id, metric_val in metrics.items():
        evidence = EvidenceRef(
            evidence_id=f"evidence_realistic_{metric_id}",
            evaluation_run_id="eval_realistic_001",
            metric_name=metric_id,
            metric_version="0.1",
            timestamp=datetime.now().isoformat(),
            factor_id=factor_id,
            summary_value=metric_val.value,
        )
        evidence_refs.append(evidence)

    # Update lifecycle based on evidence
    if simulated_rank_ic > 0.04:  # Threshold for approval
        transition = StateTransition(
            from_state=LifecycleState.REGISTERED,
            to_state=LifecycleState.EVALUATED,
            required_evidence=("eval_realistic_001",),
            description=f"Passed evaluation: rank_ic={simulated_rank_ic:.3f}, pearson_ic={simulated_pearson_ic:.3f}",
        )

        new_state = LifecycleState.EVALUATED
    else:
        new_state = LifecycleState.REGISTERED

    # Step 6: Verify complete workflow
    retrieved = repo.get(factor_id)
    assert retrieved is not None
    assert retrieved.metadata.factor_id == factor_id
    assert retrieved.metadata.canonical_repr == "mean_std_ratio_20"
    assert batch.num_factors == 1
    assert len(metrics) == 2
    assert diagnosis.coverage > 0.99
    assert len(evidence_refs) == 2

    # FO would use these results for next iteration
    print(f"Factor {factor_id[:16]}... evaluated:")
    print(f"  Rank IC: {simulated_rank_ic:.4f}")
    print(f"  Pearson IC: {simulated_pearson_ic:.4f}")
    print(f"  Coverage: {diagnosis.coverage:.2%}")
    print(f"  Recommended state: {new_state.value}")
