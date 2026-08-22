from __future__ import annotations

import threading
import types

from runtime.buffer_ref import SourceWaveExecutor


def test_source_wave_executor_serializes_concurrent_duplicate_reads():
    calls = []
    entered = threading.Event()
    release = threading.Event()

    class Source:
        snapshot_token = "s1"

        def prefetch_columns(self, columns):
            calls.append(list(columns))
            entered.set()
            assert release.wait(2)

    wave = types.SimpleNamespace(
        wave_id=7,
        columns=frozenset({"close"}),
        estimated_scan_bytes=8,
        estimated_memory_bytes=8,
    )
    executor = SourceWaveExecutor(Source())
    results = []

    def run():
        results.append(executor.execute_wave(wave, consumer_ids=("t1",)))

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert entered.wait(2)
    second.start()
    release.set()
    first.join(2)
    second.join(2)

    assert len(calls) == 1
    assert len(results) == 2
    assert results[0] is results[1]
    assert executor.summary()["waves_executed"] == 1
