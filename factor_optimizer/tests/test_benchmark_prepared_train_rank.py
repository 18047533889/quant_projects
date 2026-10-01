import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_prepared_train_rank.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_prepared_train_rank", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_dimensions_are_positive_and_capped():
    assert bench._validate_dimensions(300, 5_000) == 1_500_000
    assert bench._validate_dimensions(1_700, 5_000) == 8_500_000
    with pytest.raises(ValueError):
        bench._validate_dimensions(0, 10)
    with pytest.raises(ValueError):
        bench._validate_dimensions(3_000, 3_001)
    with pytest.raises(ValueError):
        bench._validate_dimensions(True, 2)


def test_fixture_uses_exact_integer_cross_section_ids_and_ties_nonfinite_values():
    frame = bench._build_frame(4, 16, 17)
    assert frame["date"].dtype == np.dtype("int32")
    assert frame["asset_id"].dtype == np.dtype("int32")
    assert frame["date"].tolist() == [0] * 16 + [1] * 16 + [2] * 16 + [3] * 16
    assert frame.groupby("date", sort=False).size().tolist() == [16] * 4
    assert frame["value"].duplicated().any()
    assert np.isnan(frame.loc[0, "value"])
    assert np.isposinf(frame.loc[1, "value"])
    assert np.isneginf(frame.loc[2, "value"])
    assert frame.loc[3, "value"] == frame.loc[4, "value"] == 0.0


def test_output_preflight_refuses_existing_files_and_missing_parent(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        bench._preflight_output(output)
    with pytest.raises(ValueError, match="parent"):
        bench._preflight_output(tmp_path / "missing" / "new.json")


def test_output_preflight_requires_bounded_free_space(tmp_path, monkeypatch):
    class Stats:
        f_flag = 0
        f_bavail = bench.MAX_JSON_BYTES
        f_frsize = 1

    monkeypatch.setattr(bench.os, "statvfs", lambda _path: Stats())
    with pytest.raises(OSError, match="less than twice"):
        bench._preflight_output(tmp_path / "new.json")


def test_two_modes_use_same_workload_and_record_real_call_counts(monkeypatch):
    from factor_optimizer.adapters import repair_execution

    monkeypatch.setattr(bench, "_available_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(bench, "_head_sha", lambda: "test-head")
    monkeypatch.setattr(bench, "_source_evidence", lambda: (
        {"test": "test-source"}, {"implementation_digest": "test-loaded"}))

    def fake_fe_rank(frame):
        rank = frame.groupby("date", sort=False)["value"].rank(
            method="average", pct=True, na_option="keep")
        return rank.astype(np.float64).rename("value")

    monkeypatch.setattr(repair_execution, "_execute_fe_cs_rank", fake_fe_rank)
    public = bench.run_benchmark(time_points=3, assets=20, mode="public", seed=5)
    prepared = bench.run_benchmark(time_points=3, assets=20, mode="prepared", seed=5)

    assert public["plan_count"] == prepared["plan_count"] == 14
    assert public["distinct_plan_count"] == prepared["distinct_plan_count"] == 14
    assert public["output_sha256"] == prepared["output_sha256"]
    assert public["input_frame_sha256"] == prepared["input_frame_sha256"]
    assert public["fingerprint_calls"] == 14
    assert prepared["fingerprint_calls"] == 1
    assert public["fe_rank_calls"] == prepared["fe_rank_calls"] == 1
    assert public["fingerprint_frame_object_count"] == prepared["fingerprint_frame_object_count"] == 1
    assert public["input_frame_preserved"] is prepared["input_frame_preserved"] is True
