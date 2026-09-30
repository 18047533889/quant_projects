"""Real COS panels use temporal axes, not only integer test coordinates."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime import backend_calibration as c
from quant_evaluator.tests.test_backend_calibration import _inputs


@pytest.mark.parametrize("array", [
    np.arange(12).astype("datetime64[D]"),
    np.arange(12).astype("datetime64[ns]"),
    np.array("NaT", dtype="datetime64[ns]"),
    np.arange(12).astype("timedelta64[h]"),
    np.arange(24).astype("datetime64[ns]").reshape(4, 6)[:, ::2],
    np.empty((0, 4), dtype="datetime64[ns]"),
    np.arange(12).astype(">i8"),
])
def test_temporal_and_endian_hash_is_exact_logical_c_bytes(array):
    actual = hashlib.sha256()
    c._hash_array(actual, array, chunk_bytes=16)
    oracle = hashlib.sha256()
    oracle.update(str(array.dtype).encode("ascii"))
    oracle.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    oracle.update(array.tobytes(order="C"))
    assert actual.digest() == oracle.digest()


def test_datetime_axis_calibration_produces_a_fresh_cache_hit(monkeypatch):
    from quant_evaluator import AutoCalibrationOptions, evaluate
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.scripts.benchmark_backend_tournament import panel

    reason, _ = c._device_admission(GPUExecutionPolicy())
    if reason:
        pytest.skip("device admission: " + str(reason))
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    batch, original_label = panel(64, 512, 8, 20261001)
    times = (np.datetime64("2020-01-01", "ns")
             + np.arange(64).astype("timedelta64[D]"))
    axis = AxisRef("time", "datetime64[ns]", 64, times)
    assets = AxisRef("asset", "str", 512, np.asarray([f"a{i}" for i in range(512)]))
    batch = replace(batch, time_axis=axis, asset_axis=assets)
    label = LabelBundle(
        target_id="temporal", values=original_label.values, horizon=1,
        decision_time=tuple(times), label_start_time=tuple(times + np.timedelta64(1, "D")),
        label_end_time=tuple(times + np.timedelta64(2, "D")), asset_axis=batch.asset_axis,
    )
    options = AutoCalibrationOptions(policy=c.CalibrationPolicy(repetitions=1, warmups=0))
    kwargs = dict(metrics=("pearson_ic", "pearson_ic_series"), auto_calibration=options)
    first = evaluate(batch, label, **kwargs)
    second = evaluate(batch, label, **kwargs)
    assert first.metadata["auto_calibration"]["calibration_record"]["parity"] == "pass"
    assert second.metadata["auto_calibration"]["status"] == "cache_hit"
    assert second is not first
