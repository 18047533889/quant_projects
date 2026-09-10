"""H11 bounded, finite and atomic calibration regression tests."""
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from data_access.read import scan_cost as sc

BINDING = dict(host_class="host", storage_class="disk", build_id="build")

@pytest.fixture(autouse=True)
def reset():
    sc.reset_calibration()
    yield
    sc.reset_calibration()

def record(value, key=("data", "cold", "proj=2")):
    sc.record_scan_actual("data", estimated_score=1e6, actual_elapsed_ms=value, shape_key=key)

def test_numpy_real_telemetry_and_large_integer_rejection():
    import numpy as np
    for value in (np.int64(1), np.float32(2), np.float64(3)):
        record(value)
    q = sc.calibration_quantiles(("data", "cold", "proj=2"))
    assert q.sample_count == 3 and q.p50 == 2
    for value in (np.bool_(True), 10 ** 1000, np.float32("inf")):
        assert not sc._positive_finite(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_measurements_ignored(value):
    record(value)
    assert sc.calibration_quantiles(("data", "cold", "proj=2")) is None
    assert sc._calibrated_factor("data") == 1

def test_true_median_mad():
    for v in [1, 2, 3, 100]:
        record(v)
    q = sc.calibration_quantiles(("data", "cold", "proj=2"))
    assert q.p50 == 2.5
    assert q.mad == 1
    assert q.mean == 26.5
    assert q.p95 == 100

def test_recent_sample_and_key_bounds():
    for i in range(sc._MAX_CALIBRATION_SAMPLES * 3):
        record(i + 1)
    assert sc.calibration_quantiles(("data", "cold", "proj=2")).sample_count == sc._MAX_CALIBRATION_SAMPLES
    for i in range(sc._MAX_CALIBRATION_KEYS * 3):
        record(i + 1, (str(i), "cold"))
    assert len(sc._shape_calibration_samples) == sc._MAX_CALIBRATION_KEYS
    assert sc.calibration_quantiles(("data", "cold", "proj=2")) is None

def test_atomic_concurrent_save_and_roundtrip(tmp_path):
    target = tmp_path / "calibration.json"
    record(3, ("a::b", "c"))
    record(7, ("a", "b::c"))
    def work(i):
        record(i + 1)
        sc.save_calibration(target, **BINDING)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(20)))
    saved = sc._calibrated_factor("data")
    sc.reset_calibration()
    assert sc.load_calibration(target, **BINDING)
    assert sc._calibrated_factor("data") == saved
    assert sc.calibration_quantiles(("a::b", "c")).p50 == 3
    assert sc.calibration_quantiles(("a", "b::c")).p50 == 7
    assert not list(tmp_path.glob(".scan-calibration-*"))

def test_invalid_load_does_not_modify_live_state(tmp_path):
    record(2)
    path = tmp_path / "bad.json"
    sc.save_calibration(path, **BINDING)
    payload = json.loads(path.read_text())
    payload["samples"][0]["values"] = [float("nan")]
    path.write_text(json.dumps(payload))
    assert not sc.load_calibration(path, **BINDING)
    assert sc.calibration_quantiles(("data", "cold", "proj=2")).p50 == 2
    path.write_text("[]")
    assert not sc.load_calibration(path, **BINDING)
