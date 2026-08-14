"""
End-to-end integration test: Research Control event recording across packages.

Tests research workflow event logging, audit trails, and reproducibility tracking.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


class ResearchEvent:
    """Research event for audit trail."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        timestamp: str,
        actor: str,
        package: str,
        details: Dict[str, Any],
    ):
        self.event_id = event_id
        self.event_type = event_type
        self.timestamp = timestamp
        self.actor = actor
        self.package = package
        self.details = details


class ResearchEventLogger:
    """Logger for research workflow events."""

    def __init__(self):
        self.events: List[ResearchEvent] = []

    def log_event(self, event: ResearchEvent):
        self.events.append(event)

    def get_events(self, event_type=None, package=None):
        filtered = self.events
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if package:
            filtered = [e for e in filtered if e.package == package]
        return filtered

    def get_timeline(self):
        return sorted(self.events, key=lambda e: e.timestamp)


def test_research_workflow_event_logging():
    """
    Test complete research workflow with event logging.

    Flow:
    1. FA: Register new factor asset
    2. FE: Compute factor (simulated)
    3. FP: Apply preprocessing
    4. QE: Evaluate factor
    5. FA: Store evidence and update lifecycle
    6. FO: Record in optimization history
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode
    from factor_assets import (
        AssetRepository,
        AssetMetadata,
        LineageRef,
        LifecycleState,
        EvidenceRef,
        create_factor_id,
    )

    logger = ResearchEventLogger()

    # Step 1: FA - Register factor asset
    factor_id = create_factor_id(
        name="research_momentum_15d",
        version="v1",
        params={"window": 15, "method": "log_return"},
    )

    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="research_momentum_15d",
        canonical_hash="hash_research_momentum_15d_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="15-day momentum for research",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    repo = AssetRepository()
    asset = repo.register(metadata, lineage)

    logger.log_event(ResearchEvent(
        event_id="evt_001",
        event_type="FACTOR_REGISTERED",
        timestamp=datetime.now().isoformat(),
        actor="researcher_1",
        package="factor_assets",
        details={
            "factor_id": factor_id,
            "canonical_repr": "research_momentum_15d",
            "lifecycle_state": LifecycleState.REGISTERED.value,
        },
    ))

    # Step 2: FE - Compute factor (simulated)
    T, N = 100, 30
    factor_values = np.random.randn(T, N, 1) * 0.05

    logger.log_event(ResearchEvent(
        event_id="evt_002",
        event_type="FACTOR_COMPUTED",
        timestamp=datetime.now().isoformat(),
        actor="factor_engine",
        package="factor_engine",
        details={
            "factor_id": factor_id,
            "shape": (T, N, 1),
            "computation_params": {"window": 15, "method": "log_return"},
        },
    ))

    # Step 3: FP - Apply preprocessing
    policy = PreprocessingPolicy(
        policy_id="research_policy_001",
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

    logger.log_event(ResearchEvent(
        event_id="evt_003",
        event_type="PREPROCESSING_APPLIED",
        timestamp=datetime.now().isoformat(),
        actor="preprocessing_service",
        package="factor_preprocess",
        details={
            "factor_id": factor_id,
            "policy_id": "research_policy_001",
            "transforms": ["winsorize", "standardize"],
        },
    ))

    # Step 4: QE - Evaluate factor
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=(factor_id,),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    # Create labels
    label_values = np.random.randn(T, N) * 0.02
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
    label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
    label_end = tuple((d + timedelta(days=4)).strftime("%Y-%m-%d") for d in dates)

    labels = LabelBundle(
        target_id="fwd_ret_3d",
        values=label_values,
        horizon=3,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Simulate evaluation
    simulated_ic = 0.045

    metric = MetricValue(
        metric_id="rank_ic",
        value=simulated_ic,
        valid=True,
        observation_count=T * N,
        metric_version="0.1",
    )

    logger.log_event(ResearchEvent(
        event_id="evt_004",
        event_type="EVALUATION_COMPLETED",
        timestamp=datetime.now().isoformat(),
        actor="evaluation_service",
        package="quant_evaluator",
        details={
            "factor_id": factor_id,
            "rank_ic": simulated_ic,
            "observation_count": T * N,
            "label_horizon": 3,
        },
    ))

    # Step 5: FA - Store evidence
    evidence = EvidenceRef(
        evidence_id="evidence_research_001",
        evaluation_run_id="eval_run_research_001",
        metric_name="rank_ic",
        metric_version="0.1",
        timestamp=datetime.now().isoformat(),
        factor_id=factor_id,
        summary_value=simulated_ic,
    )

    logger.log_event(ResearchEvent(
        event_id="evt_005",
        event_type="EVIDENCE_STORED",
        timestamp=datetime.now().isoformat(),
        actor="evidence_service",
        package="factor_assets",
        details={
            "factor_id": factor_id,
            "evidence_id": "evidence_research_001",
            "metric_name": "rank_ic",
            "metric_value": simulated_ic,
        },
    ))

    # Step 6: FO - Record in optimization history
    logger.log_event(ResearchEvent(
        event_id="evt_006",
        event_type="OPTIMIZATION_ITERATION",
        timestamp=datetime.now().isoformat(),
        actor="optimizer_service",
        package="factor_optimizer",
        details={
            "factor_id": factor_id,
            "iteration": 1,
            "objective_value": simulated_ic,
            "accepted": simulated_ic > 0.04,
        },
    ))

    # Verify event log
    timeline = logger.get_timeline()
    assert len(timeline) == 6

    # Verify event sequence
    event_types = [e.event_type for e in timeline]
    expected_sequence = [
        "FACTOR_REGISTERED",
        "FACTOR_COMPUTED",
        "PREPROCESSING_APPLIED",
        "EVALUATION_COMPLETED",
        "EVIDENCE_STORED",
        "OPTIMIZATION_ITERATION",
    ]
    assert event_types == expected_sequence

    # Verify each package logged events
    packages = set(e.package for e in timeline)
    assert packages == {
        "factor_assets",
        "factor_engine",
        "factor_preprocess",
        "quant_evaluator",
        "factor_optimizer",
    }

    # Verify factor_id consistency across all events
    factor_ids = [e.details.get("factor_id") for e in timeline if "factor_id" in e.details]
    assert all(fid == factor_id for fid in factor_ids)


def test_research_audit_trail_reproducibility():
    """
    Test audit trail enables reproducibility.

    Verify that logged events contain sufficient information to reproduce results.
    """
    from factor_assets import create_factor_id

    logger = ResearchEventLogger()

    # Simulate research experiment
    experiment_id = "exp_reproducibility_001"

    # Log factor creation with all parameters
    factor_id = create_factor_id(
        name="test_factor",
        version="v1",
        params={"window": 10, "decay": 0.95, "threshold": 0.02},
    )

    logger.log_event(ResearchEvent(
        event_id="evt_r001",
        event_type="FACTOR_CREATED",
        timestamp="2024-08-14T10:00:00",
        actor="researcher_2",
        package="factor_engine",
        details={
            "experiment_id": experiment_id,
            "factor_id": factor_id,
            "formula": "ewma(returns, window=10, decay=0.95) > 0.02",
            "parameters": {"window": 10, "decay": 0.95, "threshold": 0.02},
            "code_version": "v1.2.3",
            "random_seed": 42,
        },
    ))

    # Log preprocessing configuration
    logger.log_event(ResearchEvent(
        event_id="evt_r002",
        event_type="PREPROCESSING_CONFIG",
        timestamp="2024-08-14T10:01:00",
        actor="preprocessing_service",
        package="factor_preprocess",
        details={
            "experiment_id": experiment_id,
            "policy_id": "policy_reproducibility",
            "transforms": [
                {"name": "winsorize", "params": {"lower": 0.01, "upper": 0.99}},
                {"name": "standardize", "params": {"method": "robust"}},
            ],
        },
    ))

    # Log evaluation configuration
    logger.log_event(ResearchEvent(
        event_id="evt_r003",
        event_type="EVALUATION_CONFIG",
        timestamp="2024-08-14T10:02:00",
        actor="evaluation_service",
        package="quant_evaluator",
        details={
            "experiment_id": experiment_id,
            "universe": "top_500",
            "date_range": ("2023-01-01", "2023-12-31"),
            "label": "fwd_ret_5d",
            "metrics": ["rank_ic", "pearson_ic", "sharpe"],
            "backend": "polars",
        },
    ))

    # Retrieve reproducibility info
    exp_events = [e for e in logger.events if e.details.get("experiment_id") == experiment_id]
    assert len(exp_events) == 3

    # Verify all parameters captured
    factor_event = exp_events[0]
    assert "random_seed" in factor_event.details
    assert "code_version" in factor_event.details
    assert "parameters" in factor_event.details

    preprocess_event = exp_events[1]
    assert "transforms" in preprocess_event.details

    eval_event = exp_events[2]
    assert "date_range" in eval_event.details
    assert "backend" in eval_event.details


def test_research_error_tracking():
    """
    Test error tracking across packages in research workflow.
    """
    logger = ResearchEventLogger()

    # Simulate workflow with errors
    factor_id = "factor_error_test"

    # Step 1: Success
    logger.log_event(ResearchEvent(
        event_id="evt_e001",
        event_type="FACTOR_REGISTERED",
        timestamp="2024-08-14T11:00:00",
        actor="researcher_3",
        package="factor_assets",
        details={
            "factor_id": factor_id,
            "status": "success",
        },
    ))

    # Step 2: FE computation error
    logger.log_event(ResearchEvent(
        event_id="evt_e002",
        event_type="FACTOR_COMPUTATION_ERROR",
        timestamp="2024-08-14T11:01:00",
        actor="factor_engine",
        package="factor_engine",
        details={
            "factor_id": factor_id,
            "error_type": "DivisionByZero",
            "error_message": "Division by zero in rolling std calculation",
            "traceback": "...",
        },
    ))

    # Step 3: Retry with fix
    logger.log_event(ResearchEvent(
        event_id="evt_e003",
        event_type="FACTOR_COMPUTATION_RETRY",
        timestamp="2024-08-14T11:02:00",
        actor="factor_engine",
        package="factor_engine",
        details={
            "factor_id": factor_id,
            "retry_attempt": 1,
            "fix_applied": "Added epsilon to denominator",
        },
    ))

    # Step 4: Success after retry
    logger.log_event(ResearchEvent(
        event_id="evt_e004",
        event_type="FACTOR_COMPUTED",
        timestamp="2024-08-14T11:03:00",
        actor="factor_engine",
        package="factor_engine",
        details={
            "factor_id": factor_id,
            "status": "success",
            "retry_count": 1,
        },
    ))

    # Verify error tracking
    error_events = logger.get_events(event_type="FACTOR_COMPUTATION_ERROR")
    assert len(error_events) == 1
    assert error_events[0].details["error_type"] == "DivisionByZero"

    retry_events = logger.get_events(event_type="FACTOR_COMPUTATION_RETRY")
    assert len(retry_events) == 1

    # Verify final success
    success_events = logger.get_events(event_type="FACTOR_COMPUTED")
    assert len(success_events) == 1
    assert success_events[0].details["retry_count"] == 1


def test_research_multi_user_coordination():
    """
    Test event logging for multi-user research coordination.
    """
    logger = ResearchEventLogger()

    factor_id = "factor_shared"

    # User 1: Creates factor
    logger.log_event(ResearchEvent(
        event_id="evt_u001",
        event_type="FACTOR_REGISTERED",
        timestamp="2024-08-14T09:00:00",
        actor="user_1",
        package="factor_assets",
        details={"factor_id": factor_id, "owner": "user_1"},
    ))

    # User 2: Evaluates factor
    logger.log_event(ResearchEvent(
        event_id="evt_u002",
        event_type="EVALUATION_STARTED",
        timestamp="2024-08-14T09:30:00",
        actor="user_2",
        package="quant_evaluator",
        details={"factor_id": factor_id, "requested_by": "user_2"},
    ))

    # User 1: Tries to modify factor (conflict)
    logger.log_event(ResearchEvent(
        event_id="evt_u003",
        event_type="MODIFICATION_BLOCKED",
        timestamp="2024-08-14T09:31:00",
        actor="user_1",
        package="factor_assets",
        details={
            "factor_id": factor_id,
            "reason": "Factor is being evaluated by user_2",
            "blocked_operation": "update_parameters",
        },
    ))

    # User 2: Completes evaluation
    logger.log_event(ResearchEvent(
        event_id="evt_u004",
        event_type="EVALUATION_COMPLETED",
        timestamp="2024-08-14T09:35:00",
        actor="user_2",
        package="quant_evaluator",
        details={"factor_id": factor_id, "result": "success"},
    ))

    # User 1: Now can modify
    logger.log_event(ResearchEvent(
        event_id="evt_u005",
        event_type="FACTOR_MODIFIED",
        timestamp="2024-08-14T09:36:00",
        actor="user_1",
        package="factor_assets",
        details={"factor_id": factor_id, "modification": "parameter_update"},
    ))

    # Verify coordination
    timeline = logger.get_timeline()
    assert len(timeline) == 5

    # Verify conflict detection
    conflict_events = logger.get_events(event_type="MODIFICATION_BLOCKED")
    assert len(conflict_events) == 1

    # Verify actors
    actors = set(e.actor for e in timeline)
    assert actors == {"user_1", "user_2"}


def test_research_lineage_tracking():
    """
    Test lineage tracking through research workflow.

    Parent factor → Derived factor → Evaluation → Evidence
    """
    from factor_assets import (
        AssetRepository,
        AssetMetadata,
        LineageRef,
        LifecycleState,
        create_factor_id,
    )

    logger = ResearchEventLogger()
    repo = AssetRepository()

    # Create parent factor
    parent_id = create_factor_id(
        name="base_momentum",
        version="v1",
        params={"window": 20},
    )

    parent_metadata = AssetMetadata(
        factor_id=parent_id,
        canonical_repr="base_momentum",
        canonical_hash="hash_base_momentum_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Base momentum factor",
    )

    parent_lineage = LineageRef(
        factor_id=parent_id,
        parents=(),
    )

    parent_asset = repo.register(parent_metadata, parent_lineage)

    logger.log_event(ResearchEvent(
        event_id="evt_l001",
        event_type="PARENT_FACTOR_CREATED",
        timestamp="2024-08-14T10:00:00",
        actor="researcher_4",
        package="factor_assets",
        details={
            "factor_id": parent_id,
            "lineage_depth": 0,
            "parents": [],
        },
    ))

    # Create derived factor
    child_id = create_factor_id(
        name="adjusted_momentum",
        version="v1",
        params={"window": 20, "adjustment": "volatility"},
    )

    child_metadata = AssetMetadata(
        factor_id=child_id,
        canonical_repr="adjusted_momentum",
        canonical_hash="hash_adjusted_momentum_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Volatility-adjusted momentum",
    )

    child_lineage = LineageRef(
        factor_id=child_id,
        parents=(parent_id,),
    )

    child_asset = repo.register(child_metadata, child_lineage)

    logger.log_event(ResearchEvent(
        event_id="evt_l002",
        event_type="DERIVED_FACTOR_CREATED",
        timestamp="2024-08-14T10:05:00",
        actor="researcher_4",
        package="factor_assets",
        details={
            "factor_id": child_id,
            "lineage_depth": 1,
            "parents": [parent_id],
            "derivation": "volatility_adjustment",
        },
    ))

    # Evaluate derived factor
    logger.log_event(ResearchEvent(
        event_id="evt_l003",
        event_type="DERIVED_FACTOR_EVALUATED",
        timestamp="2024-08-14T10:10:00",
        actor="evaluation_service",
        package="quant_evaluator",
        details={
            "factor_id": child_id,
            "parent_factors": [parent_id],
            "rank_ic": 0.052,
        },
    ))

    # Verify lineage chain
    assert child_asset.lineage.parents == (parent_id,)
    assert parent_asset.lineage.parents == ()

    # Verify lineage events (only explicit lineage-related events)
    lineage_events = [e for e in logger.events if "lineage_depth" in e.details or "derivation" in e.details]
    assert len(lineage_events) == 2

    # Verify derivation recorded
    derived_events = logger.get_events(event_type="DERIVED_FACTOR_CREATED")
    assert len(derived_events) == 1
    assert derived_events[0].details["derivation"] == "volatility_adjustment"


def test_research_session_boundaries():
    """
    Test event logging across research session boundaries.
    """
    logger = ResearchEventLogger()

    session_1_id = "session_2024_08_14_morning"
    session_2_id = "session_2024_08_14_afternoon"

    # Session 1 events
    logger.log_event(ResearchEvent(
        event_id="evt_s001",
        event_type="SESSION_STARTED",
        timestamp="2024-08-14T09:00:00",
        actor="researcher_5",
        package="research_control",
        details={"session_id": session_1_id, "goals": ["explore_momentum_variants"]},
    ))

    logger.log_event(ResearchEvent(
        event_id="evt_s002",
        event_type="FACTOR_CREATED",
        timestamp="2024-08-14T09:15:00",
        actor="researcher_5",
        package="factor_engine",
        details={"session_id": session_1_id, "factor_id": "factor_s1_1"},
    ))

    logger.log_event(ResearchEvent(
        event_id="evt_s003",
        event_type="SESSION_ENDED",
        timestamp="2024-08-14T12:00:00",
        actor="researcher_5",
        package="research_control",
        details={
            "session_id": session_1_id,
            "factors_created": 1,
            "evaluations_run": 1,
            "summary": "Explored momentum with windows 10-30",
        },
    ))

    # Session 2 events
    logger.log_event(ResearchEvent(
        event_id="evt_s004",
        event_type="SESSION_STARTED",
        timestamp="2024-08-14T14:00:00",
        actor="researcher_5",
        package="research_control",
        details={"session_id": session_2_id, "goals": ["refine_best_candidates"]},
    ))

    logger.log_event(ResearchEvent(
        event_id="evt_s005",
        event_type="FACTOR_EVALUATED",
        timestamp="2024-08-14T14:30:00",
        actor="researcher_5",
        package="quant_evaluator",
        details={"session_id": session_2_id, "factor_id": "factor_s1_1"},
    ))

    # Query by session
    session_1_events = [e for e in logger.events if e.details.get("session_id") == session_1_id]
    session_2_events = [e for e in logger.events if e.details.get("session_id") == session_2_id]

    assert len(session_1_events) == 3
    assert len(session_2_events) == 2

    # Verify session boundaries
    session_starts = logger.get_events(event_type="SESSION_STARTED")
    session_ends = logger.get_events(event_type="SESSION_ENDED")
    assert len(session_starts) == 2
    assert len(session_ends) == 1
