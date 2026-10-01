from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_native_ewma.py"
SPEC = importlib.util.spec_from_file_location("benchmark_native_ewma", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


@pytest.mark.parametrize(
    ("maximum", "small", "large"),
    [
        (float("nan"), 8.0, 16.0),
        (float("inf"), 8.0, 16.0),
        (513.0, 8.0, 16.0),
        (-1.0, 8.0, 16.0),
        (512.0, 0.0, 16.0),
        (512.0, 8.0, float("nan")),
    ],
)
def test_limit_validation_rejects_nonfinite_nonpositive_and_over_cap(
    maximum, small, large
):
    with pytest.raises(ValueError):
        bench.validate_limits(maximum, small, large)


def test_input_frame_layout_missingness_and_parity():
    asset_major = bench._make_frame(16, 3, "asset-major", "long_run_20pct", 7)
    time_major = bench._make_frame(16, 3, "time-major", "all_nan", 7)
    assert len(asset_major) == 48
    assert asset_major["asset_id"].dtype == np.int32
    assert asset_major["date"].dtype == np.int32
    assert asset_major["value"].isna().sum() > len(asset_major) * 0.01
    assert time_major["value"].isna().all()

    output = pd.Series([np.nan, 1.0, 2.0], index=[4, 5, 6])
    result = bench.check_parity(output, output.copy())
    assert result["allclose"]
    assert result["bit_exact"]
    with pytest.raises(AssertionError, match="parity failed"):
        bench.check_parity(output, output + 1.0)


def _pandas_route(frame, halflife, min_periods=1):
    result = np.full(len(frame), np.nan, dtype=np.float64)
    for _, group in frame.groupby("asset_id", sort=False):
        positions = group.index.to_numpy(dtype=np.int64, copy=False)
        result[positions] = pd.Series(
            group["value"].to_numpy(copy=False)
        ).shift(1).ewm(
            halflife=halflife, min_periods=min_periods, adjust=False
        ).mean().to_numpy()
    return pd.Series(result, index=frame.index)


def _args(receipt: Path):
    return SimpleNamespace(
        receipt=receipt,
        include_large=False,
        only_small=True,
        max_input_mib=512.0,
        min_available_gib_small=8.0,
        min_available_gib_large=16.0,
        seed=20261001,
    )


def _mock_runtime(monkeypatch, *, available=32 * 1024**3, heads=None):
    monkeypatch.setattr(bench, "SIZES", {"small": (16, 3), "large": (25, 5)})
    monkeypatch.setattr(bench, "_available_ram_bytes", lambda: available)
    monkeypatch.setattr(bench, "_source_fingerprint", lambda root: {"source": "hash"})
    if heads is None:
        heads = iter(("test-head", "test-head"))
    monkeypatch.setattr(bench, "_head", lambda: next(heads))
    monkeypatch.setattr(bench.importlib.metadata, "version", lambda name: "test-version")


def test_benchmark_status_complete_and_source_change(monkeypatch, tmp_path):
    _mock_runtime(monkeypatch, heads=iter(("head-a", "head-a")))
    receipt_path = tmp_path / "complete.json"
    result = bench.run_benchmark(
        _args(receipt_path), _pandas_route, _pandas_route, Path("/unused")
    )
    assert result["status"] == "complete"
    assert len(result["cases"]) == 16
    assert json.loads(receipt_path.read_text())["status"] == "complete"
    assert all(case["rows"] == 48 for case in result["cases"])

    _mock_runtime(monkeypatch, heads=iter(("head-a", "head-b")))
    changed_path = tmp_path / "changed.json"
    changed = bench.run_benchmark(
        _args(changed_path), _pandas_route, _pandas_route, Path("/unused")
    )
    assert changed["status"] == "source_changed"
    assert changed["status_before_source_check"] == "complete"


def test_benchmark_status_skipped_before_allocating(monkeypatch, tmp_path):
    _mock_runtime(monkeypatch, available=1 * 1024**3)
    path = tmp_path / "skipped.json"
    result = bench.run_benchmark(
        _args(path), _pandas_route, _pandas_route, Path("/unused")
    )
    assert result["status"] == "skipped"
    assert not result["cases"]
    assert json.loads(path.read_text())["status"] == "skipped"


def test_benchmark_status_failed_after_parity_exception(monkeypatch, tmp_path):
    _mock_runtime(monkeypatch)

    def wrong_route(frame, halflife, min_periods=1):
        return _pandas_route(frame, halflife, min_periods) + 1.0

    path = tmp_path / "failed.json"
    with pytest.raises(AssertionError, match="parity failed"):
        bench.run_benchmark(
            _args(path), _pandas_route, wrong_route, Path("/unused")
        )
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure"]["type"] == "AssertionError"


def test_receipt_refuses_existing_file(monkeypatch, tmp_path):
    _mock_runtime(monkeypatch)
    path = tmp_path / "existing.json"
    path.write_text("keep")
    with pytest.raises(FileExistsError):
        bench.run_benchmark(
            _args(path), _pandas_route, _pandas_route, Path("/unused")
        )
    assert path.read_text() == "keep"
