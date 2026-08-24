from __future__ import annotations

from data_access.read.scan_cost import (
    ScanCacheState,
    ScanCost,
    calibration_quantiles,
    load_calibration,
    record_scan_actual,
    reset_calibration,
    save_calibration,
    scan_shape_key,
    suggest_read_strategy,
)


def _cost(**updates):
    values = dict(
        dataset="prices", file_count=4, total_bytes=64 << 20,
        estimated_rows=100_000, projected_columns=2, total_columns=20,
        remote=False, selected_files=4, selected_bytes=64 << 20,
        projection_bytes=32 << 20, file_format="parquet", score=1_000_000,
    )
    values.update(updates)
    return ScanCost(**values)


def test_shape_calibration_isolated_by_cache_and_projection():
    reset_calibration()
    cold = scan_shape_key(_cost(), cache_state=ScanCacheState.COLD)
    warm = scan_shape_key(_cost(), cache_state=ScanCacheState.WARM)
    wide = scan_shape_key(_cost(projected_columns=15), cache_state=ScanCacheState.COLD)
    for elapsed in (1, 2, 3, 4):
        record_scan_actual("prices", estimated_score=1_000_000, actual_elapsed_ms=elapsed, shape_key=cold)
    record_scan_actual("prices", estimated_score=1_000_000, actual_elapsed_ms=0.1, shape_key=warm)
    stats = calibration_quantiles(cold)
    assert stats is not None and stats.sample_count == 4
    assert stats.p50 < stats.p95 and stats.mad > 0
    assert calibration_quantiles(warm).p50 == 0.1
    assert calibration_quantiles(wide) is None


def test_calibration_persistence_generation_binding(tmp_path):
    reset_calibration()
    key = scan_shape_key(_cost(), cache_state=ScanCacheState.STEADY_STATE)
    record_scan_actual("prices", estimated_score=1_000_000, actual_elapsed_ms=2, shape_key=key)
    path = tmp_path / "calibration.json"
    save_calibration(path, host_class="h1", storage_class="nvme", build_id="b1")
    reset_calibration()
    assert not load_calibration(path, host_class="h2", storage_class="nvme", build_id="b1")
    assert load_calibration(path, host_class="h1", storage_class="nvme", build_id="b1")
    assert calibration_quantiles(key).p50 == 2.0


def test_routing_accounts_for_downstream_backend_and_projection_bytes():
    engine, result = suggest_read_strategy(
        _cost(projection_bytes=2 << 30), downstream_backend="polars_native"
    )
    assert engine == "polars"
    assert result == "lazy"
