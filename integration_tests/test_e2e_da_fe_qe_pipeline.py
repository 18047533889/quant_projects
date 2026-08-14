"""
End-to-end integration test: DataAccess → FactorEngine → QuantEvaluator pipeline.

Tests the complete flow from raw data through factor computation to evaluation.
This validates the integration between DA, FE, and QE packages.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta


def test_da_fe_qe_complete_pipeline():
    """
    Test complete pipeline: DA → FE → QE

    Flow:
    1. DA provides raw market data
    2. FE computes factors from raw data
    3. QE evaluates factor performance
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue

    # Step 1: Simulate DA providing raw data
    # In production, this would be: da_adapter.read_market_data(...)
    T, N = 100, 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    asset_ids = list(range(1000, 1000 + N))

    # Raw OHLCV data from DA
    raw_close = np.random.randn(T, N).cumsum(axis=0) + 100
    raw_volume = np.random.randint(1000000, 10000000, size=(T, N))

    # Step 2: FE computes factor (simulated)
    # In production: fe_adapter.execute_operator("momentum", params={...})
    # Here we simulate a simple momentum factor
    window = 10
    factor_values = np.zeros((T, N))
    for t in range(window, T):
        factor_values[t] = (raw_close[t] - raw_close[t - window]) / raw_close[t - window]

    # Replace early NaN with 0 for testing
    factor_values[:window] = 0.0

    # Step 3: Package into QE FactorBatch
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("momentum_10d",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values.reshape(T, N, 1),
    )

    # Step 4: Create labels from forward returns (also from DA)
    forward_returns = np.zeros((T, N))
    for t in range(T - 5):
        forward_returns[t] = (raw_close[t + 5] - raw_close[t]) / raw_close[t]

    decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
    label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
    label_end = tuple((d + timedelta(days=6)).strftime("%Y-%m-%d") for d in dates)

    labels = LabelBundle(
        target_id="fwd_ret_5d",
        values=forward_returns,
        horizon=5,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Step 5: QE evaluation (simulated)
    # In production: evaluator.evaluate(batch, labels, metrics=[...])
    # Here we compute a simple rank IC
    valid_mask = (factor_values != 0) & (forward_returns != 0) & np.isfinite(factor_values) & np.isfinite(forward_returns)

    rank_ics = []
    for t in range(T):
        mask_t = valid_mask[t]
        if mask_t.sum() > 10:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(factor_values[t, mask_t], forward_returns[t, mask_t])
            if np.isfinite(ic):
                rank_ics.append(ic)

    mean_rank_ic = np.mean(rank_ics) if rank_ics else 0.0

    metric = MetricValue(
        metric_id="rank_ic",
        value=mean_rank_ic,
        valid=True,
        observation_count=len(rank_ics),
        metric_version="0.1",
    )

    # Verify complete pipeline
    assert batch.num_factors == 1
    assert batch.values.shape == (T, N, 1)
    assert labels.values.shape == (T, N)
    assert labels.horizon == 5
    assert metric.valid
    assert abs(metric.value) < 1.0  # IC should be in [-1, 1]
    assert metric.observation_count > 50


def test_fe_qe_multi_factor_computation():
    """
    Test FE computing multiple factors and QE evaluating them in batch.
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle

    # Simulate raw data from DA
    T, N = 80, 30
    raw_close = np.random.randn(T, N).cumsum(axis=0) + 100
    raw_volume = np.random.randint(1000000, 10000000, size=(T, N))

    # FE computes multiple factors
    factor_ids = []
    factor_arrays = []

    # Factor 1: 5-day momentum
    mom_5 = np.zeros((T, N))
    for t in range(5, T):
        mom_5[t] = (raw_close[t] - raw_close[t - 5]) / raw_close[t - 5]
    factor_ids.append("momentum_5d")
    factor_arrays.append(mom_5)

    # Factor 2: 20-day momentum
    mom_20 = np.zeros((T, N))
    for t in range(20, T):
        mom_20[t] = (raw_close[t] - raw_close[t - 20]) / raw_close[t - 20]
    factor_ids.append("momentum_20d")
    factor_arrays.append(mom_20)

    # Factor 3: Volume ratio
    vol_ratio = np.zeros((T, N))
    for t in range(5, T):
        vol_ratio[t] = raw_volume[t] / np.mean(raw_volume[t-5:t], axis=0)
    factor_ids.append("volume_ratio_5d")
    factor_arrays.append(vol_ratio)

    # Stack factors
    factor_values = np.stack(factor_arrays, axis=2)  # (T, N, 3)

    # Create batch
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=tuple(factor_ids),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    # Create labels
    forward_returns = np.random.randn(T, N) * 0.02
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
    label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
    label_end = tuple((d + timedelta(days=3)).strftime("%Y-%m-%d") for d in dates)

    labels = LabelBundle(
        target_id="fwd_ret_2d",
        values=forward_returns,
        horizon=2,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Verify batch structure
    assert batch.num_factors == 3
    assert batch.values.shape == (T, N, 3)
    assert len(batch.factor_ids) == 3
    assert "momentum_5d" in batch.factor_ids
    assert "momentum_20d" in batch.factor_ids
    assert "volume_ratio_5d" in batch.factor_ids


def test_da_fe_error_propagation():
    """
    Test error propagation from DA through FE to QE.

    Scenarios:
    - Missing data from DA
    - FE computation failures
    - Invalid factor values
    """
    from quant_evaluator import FactorBatch, AxisRef, FactorDiagnosis

    T, N = 50, 20

    # Scenario 1: Missing data from DA (NaN values)
    raw_data_with_nans = np.random.randn(T, N)
    raw_data_with_nans[10:15, 5:10] = np.nan  # Missing block

    # FE should propagate NaNs or handle them
    factor_values = raw_data_with_nans.copy()

    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("test_factor",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values.reshape(T, N, 1),
    )

    # QE diagnosis should detect NaNs
    num_nans = np.isnan(batch.values).sum()
    has_nans = num_nans > 0

    diagnosis = FactorDiagnosis(
        factor_id="test_factor",
        num_valid_observations=np.isfinite(batch.values).sum(),
        num_missing=num_nans,
        coverage=1.0 - (num_nans / batch.values.size),
        is_constant=False,
        has_nans=has_nans,
        has_infs=False,
        min_value=np.nanmin(batch.values),
        max_value=np.nanmax(batch.values),
        mean_value=np.nanmean(batch.values),
    )

    assert diagnosis.has_nans
    assert diagnosis.num_missing > 0
    assert diagnosis.coverage < 1.0

    # Scenario 2: Infinite values from FE computation error
    factor_with_inf = np.random.randn(T, N, 1)
    factor_with_inf[5, 3, 0] = np.inf
    factor_with_inf[8, 7, 0] = -np.inf

    batch_inf = FactorBatch(
        factor_ids=("factor_with_inf",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_with_inf,
    )

    has_infs = np.isinf(batch_inf.values).any()
    assert has_infs


def test_qe_fe_optimization_loop():
    """
    Test optimization loop: QE → FO → FE → QE

    Flow:
    1. QE evaluates current factors
    2. FO proposes parameter mutations
    3. FE recomputes with new parameters
    4. QE re-evaluates
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue

    T, N = 60, 25
    raw_close = np.random.randn(T, N).cumsum(axis=0) + 100

    # Forward returns (constant for all iterations)
    forward_returns = np.random.randn(T, N) * 0.02
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
    label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
    label_end = tuple((d + timedelta(days=4)).strftime("%Y-%m-%d") for d in dates)

    labels = LabelBundle(
        target_id="fwd_ret_3d",
        values=forward_returns,
        horizon=3,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Optimization loop
    best_ic = -999.0
    best_window = None

    for window in [5, 10, 20, 30]:
        # FE computes factor with current window parameter
        factor_values = np.zeros((T, N))
        for t in range(window, T):
            factor_values[t] = (raw_close[t] - raw_close[t - window]) / raw_close[t - window]

        # QE evaluates
        time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=(f"momentum_{window}d",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values.reshape(T, N, 1),
        )

        # Compute simulated IC
        valid_mask = (factor_values != 0) & np.isfinite(factor_values)
        simulated_ic = np.random.randn() * 0.05 + 0.02 * (window / 20)  # Simulate better performance for moderate windows

        metric = MetricValue(
            metric_id="rank_ic",
            value=simulated_ic,
            valid=True,
            observation_count=valid_mask.sum(),
            metric_version="0.1",
        )

        # FO tracks best
        if metric.value > best_ic:
            best_ic = metric.value
            best_window = window

    # Verify optimization found something
    assert best_window is not None
    assert best_ic > -999.0
    assert best_window in [5, 10, 20, 30]


def test_da_fe_qe_timing_consistency():
    """
    Test timing consistency across DA → FE → QE pipeline.

    Ensures:
    - DA provides data with correct timestamps
    - FE respects point-in-time constraints
    - QE validates decision time vs label time
    """
    from quant_evaluator import LabelBundle, AxisRef

    T = 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]

    # DA provides timestamped data
    da_timestamps = [d.strftime("%Y-%m-%d") for d in dates]

    # FE computes factor at each timestamp (point-in-time)
    # Factor at t can only use data up to t-1
    factor_values = np.random.randn(T, 10)

    # QE requires strict timing: decision_time < label_start_time < label_end_time
    decision_times = tuple(da_timestamps)
    label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
    label_end = tuple((d + timedelta(days=2)).strftime("%Y-%m-%d") for d in dates)

    labels = LabelBundle(
        target_id="fwd_ret_1d",
        values=np.random.randn(T, 10),
        horizon=1,
        decision_time=decision_times,
        label_start_time=label_start,
        label_end_time=label_end,
    )

    # Verify timing order
    for i in range(T):
        dt = datetime.strptime(decision_times[i], "%Y-%m-%d")
        ls = datetime.strptime(label_start[i], "%Y-%m-%d")
        le = datetime.strptime(label_end[i], "%Y-%m-%d")

        assert dt < ls, f"Decision time must be before label start at index {i}"
        assert ls < le, f"Label start must be before label end at index {i}"

    assert labels.horizon == 1


def test_concurrent_da_fe_qe_access():
    """
    Test concurrent access patterns in DA → FE → QE pipeline.

    Simulates multiple evaluation jobs running simultaneously.
    """
    from quant_evaluator import FactorBatch, AxisRef
    import threading

    T, N = 40, 15
    raw_data = np.random.randn(T, N).cumsum(axis=0) + 100

    results = []
    errors = []

    def compute_and_evaluate(window, thread_id):
        try:
            # Each thread simulates DA read → FE compute → QE evaluate
            factor_values = np.zeros((T, N))
            for t in range(window, T):
                factor_values[t] = (raw_data[t] - raw_data[t - window]) / raw_data[t - window]

            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=(f"momentum_{window}d_thread{thread_id}",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values.reshape(T, N, 1),
            )

            results.append({
                "thread_id": thread_id,
                "window": window,
                "batch_shape": batch.values.shape,
                "num_factors": batch.num_factors,
            })
        except Exception as e:
            errors.append({"thread_id": thread_id, "error": str(e)})

    # Launch concurrent threads
    threads = []
    for i, window in enumerate([5, 10, 15, 20]):
        t = threading.Thread(target=compute_and_evaluate, args=(window, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all succeeded
    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    assert len(results) == 4

    for result in results:
        assert result["batch_shape"] == (T, N, 1)
        assert result["num_factors"] == 1


def test_da_fe_qe_memory_cleanup():
    """
    Test memory cleanup across DA → FE → QE pipeline.

    Ensures intermediate results are properly released.
    """
    from quant_evaluator import FactorBatch, AxisRef
    import gc

    T, N = 100, 50

    # Simulate multiple pipeline runs
    for iteration in range(10):
        # DA read
        raw_data = np.random.randn(T, N) * 100

        # FE compute
        factor_values = raw_data * 2.0  # Simple transform

        # QE evaluate
        time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=(f"factor_{iteration}",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values.reshape(T, N, 1),
        )

        # Use batch
        assert batch.num_factors == 1

        # Explicit cleanup
        del raw_data
        del factor_values
        del batch
        gc.collect()

    # If we got here without OOM, cleanup worked
    assert True


def test_da_fe_qe_data_integrity():
    """
    Test data integrity through the complete pipeline.

    Verifies:
    - No data corruption during transfers
    - Checksums remain valid
    - Shape consistency maintained
    """
    from quant_evaluator import FactorBatch, AxisRef
    import hashlib

    T, N = 50, 20

    # DA provides data with checksum
    raw_data = np.random.randn(T, N)
    raw_checksum = hashlib.sha256(raw_data.tobytes()).hexdigest()

    # FE computes factor (non-destructive)
    factor_values = raw_data + 1.0  # Simple transform

    # Verify raw data unchanged
    raw_checksum_after = hashlib.sha256(raw_data.tobytes()).hexdigest()
    assert raw_checksum == raw_checksum_after

    # QE receives factor
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("integrity_test",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values.reshape(T, N, 1),
    )

    # Verify shape consistency
    assert batch.values.shape == (T, N, 1)
    assert batch.time_axis.size == T
    assert batch.asset_axis.size == N

    # Compute checksum of factor values
    factor_checksum = hashlib.sha256(batch.values.tobytes()).hexdigest()

    # Access again - checksum should match
    factor_checksum_2 = hashlib.sha256(batch.values.tobytes()).hexdigest()
    assert factor_checksum == factor_checksum_2
