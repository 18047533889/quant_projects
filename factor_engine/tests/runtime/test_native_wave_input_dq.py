import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.runtime.input_dq import (
    InputDQError, InputDQThresholds, assert_input_dq, assert_native_input_dq,
)


@pytest.mark.parametrize("values", [[1., 2., 3.], [None, np.nan, 1.], [np.inf, -np.inf, 0.]])
@pytest.mark.parametrize("duplicate", [False, True])
def test_native_dq_matches_reference_without_source_read(values, duplicate):
    frame = pl.DataFrame({"ts": [1, 1 if duplicate else 2, 3],
                          "inst": ["a", "a", "b"], "x": values})
    series = frame.to_pandas().set_index(["ts", "inst"])["x"]
    thresholds = InputDQThresholds(min_non_null_ratio=.8, min_instruments=2)
    expected = assert_input_dq(None, ["x"], prefetched={"x": series},
                               thresholds=thresholds, raise_on_fail=False)
    actual = assert_native_input_dq(frame, ["x"], thresholds=thresholds,
                                    raise_on_fail=False)
    assert actual.to_dict() == expected.to_dict()
    if not expected.passed:
        with pytest.raises(InputDQError):
            assert_native_input_dq(frame, ["x"], thresholds=thresholds)


def test_native_dq_missing_and_empty_fail_closed():
    frame = pl.DataFrame(schema={"ts": pl.Int64, "inst": pl.String, "x": pl.Float64})
    with pytest.raises(InputDQError):
        assert_native_input_dq(frame, ["x"])
    with pytest.raises(InputDQError):
        assert_native_input_dq(frame, ["missing"])
    with pytest.raises(TypeError):
        assert_native_input_dq(frame.lazy(), ["x"])


def test_scheduler_native_wave_dq_does_not_reopen_source():
    from types import SimpleNamespace
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.planner.read_wave_planner import ReadWave, ReadWavePlan
    from factor_engine.planner.physical_factor_dag import TASK_SOURCE_SCAN

    class Source:
        dataset = ""
        reads = 0
        def scan_polars_long(self, columns):
            self.reads += 1
            return pl.DataFrame({"ts":[1,2], "inst":["a","b"], "x":[1.,2.]}).lazy()
        def load_columns(self, names):
            raise AssertionError("DQ must reuse the certified native buffer")
        load_column = load_columns

    source = Source()
    wave = ReadWave(wave_id=0, source_scope="d", dataset="d", snapshot_id="s",
                    columns=frozenset({"x"}), time_range=None, source_tasks=("scan",),
                    estimated_memory_bytes=80, preferred_representation="polars_lazy")
    scheduler = AdaptiveBatchScheduler.__new__(AdaptiveBatchScheduler)
    scheduler.job_lease = None
    scheduler._lease_scope = "test"
    scheduler._wave_executor = None
    for key in ("_wave_refs", "_wave_pending_consumers", "_wave_source_tasks",
                "_buffer_results", "_task_started_at", "_task_timing"):
        setattr(scheduler, key, {})
    scheduler._wave_summary = {}
    scheduler._input_dq_reports = []
    scheduler._done = 0
    scheduler._wave_covered_tasks = set()
    scheduler.broker = SimpleNamespace(acquire_memory=lambda *a,**kw:SimpleNamespace(release=lambda:None))
    task = SimpleNamespace(task_id="scan", task_type=TASK_SOURCE_SCAN, inputs=(),
                           op="source_scan", factor_name="", preferred_backend="polars")
    scheduler._execute_read_waves(
        SimpleNamespace(read_waves=ReadWavePlan([wave])),
        SimpleNamespace(tasks={"scan":task}), set(), {"scan"},
        SimpleNamespace(data_source=source, runtime_stats={}),
        input_dq_check=True, input_dq_strict=True,
    )
    assert source.reads == 1
    assert len(scheduler._input_dq_reports) == 1
    assert scheduler._input_dq_reports[0].passed
