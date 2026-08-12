from __future__ import annotations

from planner.read_wave_planner import ConservativeScanClass, ReadWavePlanner


def test_typed_fallback_replaces_global_500k_rows():
    planner = ReadWavePlanner(conservative_class=ConservativeScanClass.REFERENCE_TABLE)
    planner.register_scan_task(
        "s", dataset="security_master", source_scope="reference", snapshot_id="v1",
        time_range=None, columns=["id", "name"],
    )
    wave = planner.plan().waves[0]
    assert wave.estimated_memory_bytes == 2 * 25_000 * 8
    assert wave.estimated_memory_bytes != 2 * 500_000 * 8


def test_session_ordinal_drives_range_density():
    ordinals = {"2024-01-05": 10, "2024-01-08": 11}
    planner = ReadWavePlanner(
        conservative_class=ConservativeScanClass.DAILY_PANEL,
        session_ordinal=ordinals.__getitem__,
    )
    planner.register_scan_task(
        "s", dataset="daily_prices", source_scope="ashare", snapshot_id="v1",
        time_range=("2024-01-05", "2024-01-08"), columns=["close"],
    )
    assert planner.plan().waves[0].estimated_memory_bytes == 2 * 8_000 * 8


def test_historical_shape_estimate_precedes_conservative_class():
    planner = ReadWavePlanner(
        conservative_class=ConservativeScanClass.INTRADAY_PANEL,
        historical_rows_estimator=lambda dataset, scope, time_range: 1234,
    )
    planner.register_scan_task(
        "s", dataset="minute", source_scope="ashare", snapshot_id="v1",
        time_range=None, columns=["close"],
    )
    assert planner.plan().waves[0].estimated_memory_bytes == 1234 * 8
