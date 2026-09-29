import json
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


def _fixtures():
    times = pd.date_range("2024-01-01", periods=3, freq="B")
    time_axis = AxisRef("time", "datetime64[ns]", 3,
                        times.to_numpy(dtype="datetime64[ns]"))
    asset_axis = AxisRef("asset", "str", 4,
                         np.asarray(["A", "B", "C", "D"]))
    values = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
    batch = FactorBatch(("f0", "f1"), time_axis, asset_axis, values)
    label = LabelBundle(
        "ret", np.ones((3, 4), dtype=np.float64), 1,
        decision_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + pd.Timedelta(days=1)),
        asset_axis=asset_axis,
    )
    records = (
        ("f0", "cos://test/f0", "0" * 64, 1),
        ("f1", "cos://test/f1", "1" * 64, 1),
    )
    source_rows = tuple((*row, "etag-" + row[0], "2" * 64) for row in records)
    return times, asset_axis, batch, label, records, source_rows


def test_real_cos_source_repeats_share_snapshot_and_preserve_tile_axes(monkeypatch):
    times, asset_axis, batch, label, records, source_rows = _fixtures()
    reader_calls = []

    def fake_reader(name):
        def read(rows, *args, **kwargs):
            reader_calls.append((name, tuple(row[0] for row in rows)))
            return iter(())
        return read

    monkeypatch.setattr(harness.tiles, "iter_frames", fake_reader("serial"))
    monkeypatch.setattr(harness.tiles, "iter_frames_prefetched", fake_reader("prefetch"))

    def make_tile(stream, expected, dates, assets, labels, *, load_phases=None):
        ids = tuple(row[0] for row in expected)
        indices = [batch.factor_ids.index(fid) for fid in ids]
        tile_batch = FactorBatch(
            ids, batch.time_axis, batch.asset_axis,
            batch.values[:, :, indices],
        )
        return tile_batch

    monkeypatch.setattr(harness.tiles, "make_tile", make_tile)
    sources = [
        harness.RealCosSource(
            records, source_rows, pd.DatetimeIndex(times), asset_axis.values,
            label, "a" * 64, 1, 16, prefetch_objects=bool(index),
        )
        for index in range(2)
    ]
    assert sources[0] is not sources[1]
    assert sources[0].snapshot_id == sources[1].snapshot_id
    for source in sources:
        for start in range(2):
            tile = source.read_tile(start, start + 1)
            assert tile.snapshot_id == source.snapshot_id
            assert tile.batch.factor_ids == (f"f{start}",)
            assert np.array_equal(tile.batch.time_axis.values, source.time_axis.values)
            assert np.array_equal(tile.batch.asset_axis.values, source.asset_axis.values)
        assert source.reads == [(0, 1), (1, 2)]
        assert [item["tile_range"] for item in source.tile_read_timings] == [[0, 1], [1, 2]]
    assert reader_calls == [
        ("serial", ("f0",)), ("serial", ("f1",)),
        ("prefetch", ("f0",)), ("prefetch", ("f1",)),
    ]


def test_run_backend_forwards_object_cap_and_validates_receipt(monkeypatch):
    _, asset_axis, _, label, records, source_rows = _fixtures()
    sources = []

    def fake_evaluate(source, labels, *, metrics, backend, max_tile_size, gpu_policy):
        sources.append(source)
        for start in range(0, len(source.factor_ids), max_tile_size):
            source.reads.append(
                (start, min(start + max_tile_size, len(source.factor_ids))))
        return SimpleNamespace(metadata={
            "factor_tiles_processed": len(source.reads),
            "backend_used": "cuda" if backend == "cuda_strict" else "cpu",
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", fake_evaluate)
    results = []
    for backend in ("cpu", "cuda_strict"):
        result, timing = harness.run_backend(
            backend, records, source_rows, pd.date_range("2024-01-01", periods=3, freq="B"),
            asset_axis.values, label, "a" * 64, 1, 16, GPUExecutionPolicy(),
            harness.DEFAULT_METRICS, prefetch_objects=True,
        )
        results.append((result, timing))

    assert sources[0] is not sources[1]
    assert all(source.max_object_mib == 16 for source in sources)
    assert all(source.reads == [(0, 1), (1, 2)] for source in sources)
    assert [timing["backend_used"] for _, timing in results] == ["cpu", "cuda"]
    assert [timing["factor_tiles_processed"] for _, timing in results] == [2, 2]
    assert [timing["prefetch_objects"] for _, timing in results] == [True, True]
    assert [timing["total_wall_seconds"] for _, timing in results] == [
        timing["seconds"] for _, timing in results]


def test_run_backend_auto_validates_effective_not_requested_tile_width(monkeypatch):
    _, asset_axis, _, label, _, _ = _fixtures()
    records = tuple((f"f{i}", f"cos://test/f{i}", "0" * 64, 1) for i in range(5))
    source_rows = tuple((*row, "etag", "2" * 64) for row in records)

    def fake_evaluate(source, labels, *, metrics, backend, max_tile_size, gpu_policy):
        source.reads.extend([(0, 2), (2, 4), (4, 5)])
        return SimpleNamespace(metadata={
            "factor_tiles_processed": 3, "backend_used": "cuda",
            "effective_max_tile_size": 2,
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", fake_evaluate)
    _, receipt = harness.run_backend(
        "auto", records, source_rows,
        pd.date_range("2024-01-01", periods=3, freq="B"),
        asset_axis.values, label, "a" * 64, 4, 16, GPUExecutionPolicy(),
        harness.DEFAULT_METRICS, expected_auto_cuda=True,
    )
    assert receipt["tile_ranges"] == [(0, 2), (2, 4), (4, 5)]


def test_auto_reference_requires_opposite_order_and_matching_hashes(tmp_path):
    metric = "pearson_ic"
    digest = "a" * 64

    def report(order, shape=(2586, 5461, 61)):
        return {
            "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
            "manifest_sha256": "b" * 64, "metric_ids": [metric],
            "factor_dtype": "float64", "tile_size": 16,
            "shape": list(shape), "run_order": list(order),
            "runs": [{"backend_used": "cpu"}, {"backend_used": "cuda"}],
            "comparison": {"pass": True, "compared_metric_count": 61,
                           "metrics": {metric: {
                               "pass": True, "artifact_kind": "scalar",
                               "cuda_values_sha256": digest,
                               "cpu_values_sha256": "c" * 64,
                               "cuda_shape": [61], "finite_value_count": 61,
                           }}},
        }

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(report(("cpu", "cuda_strict"))), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"))), encoding="utf-8")
    assert harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)["shape"] == [2586, 5461, 61]
    second_shape = (2400, 5000, 61)
    left.write_text(json.dumps(report(("cpu", "cuda_strict"), second_shape)), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"), second_shape)), encoding="utf-8")
    assert harness.certified_cuda_hashes(
        (left, right), (metric,), "b" * 64)["shape"] == list(second_shape)
    unsupported_shape = (2400, 5001, 61)
    left.write_text(json.dumps(report(("cpu", "cuda_strict"), unsupported_shape)), encoding="utf-8")
    right.write_text(json.dumps(report(("cuda_strict", "cpu"), unsupported_shape)), encoding="utf-8")
    with pytest.raises(ValueError, match="opposite-order"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)
    bad = report(("cpu", "cuda_strict"))
    left.write_text(json.dumps(report(("cpu", "cuda_strict"))), encoding="utf-8")
    right.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="opposite-order"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)
    bad["run_order"] = ["cuda_strict", "cpu"]
    bad["comparison"]["metrics"][metric]["cuda_values_sha256"] = "d" * 64
    right.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        harness.certified_cuda_hashes((left, right), (metric,), "b" * 64)


def test_preflight_enforces_ram_and_bounded_disk_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(harness.tiles, "cos_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        harness.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=8 * 1024**3),
    )
    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 40 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is True

    monkeypatch.setattr(harness, "available_ram_bytes",
                        lambda: 31 * 1024**3)
    assert harness.preflight(128, 4096)["pass"] is False


def test_compare_reads_fingerprint_from_public_bundle_metadata():
    def bundle(fingerprint="a" * 64):
        return SimpleNamespace(
            factor_ids=("f0", "f1"),
            metadata={"source_request_fingerprint": fingerprint},
            scalar_metrics={name: np.array([0.1, 0.2])
                            for name in ("pearson_ic",)},
            observation_counts={name: np.array([3, 3])
                                for name in ("pearson_ic",)},
        )

    result = harness.compare(bundle(), bundle(), harness.PEARSON_SINGLE)
    assert result["pass"] is True
    assert result["compared_factor_count"] == 2
    assert result["compared_metric_count"] == 2
    with pytest.raises(ValueError, match="fingerprints differ"):
        harness.compare(bundle(), bundle("b" * 64), harness.PEARSON_SINGLE)


def test_auto_direct_comparison_rejects_same_shape_count_mismatch():
    def bundle(counts):
        return SimpleNamespace(
            factor_ids=("f0", "f1"),
            metadata={"source_request_fingerprint": "a" * 64},
            scalar_metrics={"pearson_ic": np.asarray([0.1, 0.2])},
            observation_counts={"pearson_ic": np.asarray(counts, dtype=np.int64)},
        )

    result = harness.compare(
        bundle([3, 3]), bundle([3, 2]), harness.PEARSON_SINGLE)
    metric = result["metrics"]["pearson_ic"]
    assert metric["observation_counts_shape_valid"] is True
    assert metric["observation_counts_equal"] is False
    assert result["pass"] is False


def test_compare_pearson_chain_checks_full_series_counts_masks_and_fingerprint():
    factors = ("f0", "f1")
    scalars = {
        metric: np.asarray([0.1, 0.2])
        for metric in ("pearson_ic", "pearson_ic_std", "pearson_ic_ir")
    }
    series = np.asarray([[0.1, np.nan], [0.2, 0.3], [np.nan, 0.4]])
    def bundle(*, values=series, counts=None, fingerprint="a" * 64):
        return SimpleNamespace(
            factor_ids=factors,
            metadata={"source_request_fingerprint": fingerprint},
            scalar_metrics=scalars,
            series_metrics={"pearson_ic_series": values},
            observation_counts={
                metric: np.asarray([2, 2]) for metric in harness.PEARSON_CHAIN
            } if counts is None else counts,
        )

    result = harness.compare(
        bundle(), bundle(), harness.PEARSON_CHAIN, expected_days=3)
    assert result["pass"]
    assert result["compared_metric_count"] == 12
    assert result["metrics"]["pearson_ic_series"]["artifact_kind"] == "series"
    assert result["metrics"]["pearson_ic_series"]["cpu_shape"] == [3, 2]
    assert result["metrics"]["pearson_ic_series"]["finite_value_count"] == 4
    assert len(result["metrics"]["pearson_ic_series"]["cpu_values_sha256"]) == 64

    wrong_mask = series.copy()
    wrong_mask[0, 1] = 0.0
    assert not harness.compare(
        bundle(), bundle(values=wrong_mask), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    assert not harness.compare(
        bundle(), bundle(values=series[:2]), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    wrong_counts = {metric: np.asarray([2, 2]) for metric in harness.PEARSON_CHAIN}
    wrong_counts["pearson_ic_series"] = np.asarray([2, 1])
    assert not harness.compare(
        bundle(), bundle(counts=wrong_counts), harness.PEARSON_CHAIN,
        expected_days=3)["pass"]
    with pytest.raises(ValueError, match="fingerprints differ"):
        harness.compare(
            bundle(fingerprint="z" * 64), bundle(fingerprint="z" * 64),
            harness.PEARSON_CHAIN, expected_days=3)


def test_pearson_chain_rejects_gpu_worker_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "benchmark", "--factors", "61", "--metrics",
        ",".join(harness.PEARSON_CHAIN), "--gpu-worker", "--output", "/tmp/unused.json"])
    with pytest.raises(SystemExit):
        harness.main()


def test_pearson_chain_benchmark_accepts_only_complete_permutations():
    assert harness.is_pearson_chain(harness.PEARSON_CHAIN)
    assert harness.is_pearson_chain(tuple(reversed(harness.PEARSON_CHAIN)))
    assert not harness.is_pearson_chain(harness.PEARSON_CHAIN[:-1])
    assert not harness.is_pearson_chain(("pearson_ic",) * 4)


def test_all_source_benchmark_profile_covers_public_source_metric_set():
    from quant_evaluator.api.factor_source import _SOURCE_METRICS

    assert len(harness.ALL_SOURCE_METRICS) == len(_SOURCE_METRICS) == 15
    assert frozenset(harness.ALL_SOURCE_METRICS) == _SOURCE_METRICS


def test_gpu_tile_width_ab_interleaves_and_checks_every_run(monkeypatch, tmp_path):
    reports, widths = [], []
    assert harness.DEFAULT_METRICS == ("rank_ic", "quantile_spread", "factor_turnover_rate")
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "61",
                                  "--gpu-tile-widths", "8", "16",
                                  "--metrics", "pearson_ic", "--output", str(tmp_path / "summary.json")])

    def fake_run(command, **kwargs):
        assert command[command.index("--metrics") + 1] == "pearson_ic"
        width = int(command[command.index("--tile-size") + 1])
        widths.append(width)
        output = harness.Path(command[command.index("--output") + 1])
        receipt = {
            "kind": "real_cos_source_gpu_worker.v1",
            "tile_size": width, "metric_ids": ["pearson_ic"], "manifest_sha256": "a" * 64,
            "shape": [3, 4, 2], "factor_ids": ["f0", "f1"],
            "source_request_fingerprint": "b" * 64,
            "run": {"seconds": width / 10, "backend_used": "cuda"},
            "scalar_metrics": {name: [0.1, 0.2] for name in ("pearson_ic",)},
            "observation_counts": {name: [3, 3] for name in ("pearson_ic",)},
        }
        output.write_text(json.dumps(receipt), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "emit_report",
                        lambda report, output: reports.append(json.loads(json.dumps(report))))
    harness.main()
    assert widths == [8, 16, 16, 8]
    assert [item["status"] for item in reports] == ["partial"] * 4 + ["complete"]
    assert len(reports[-1]["comparisons_to_first_run"]) == 3
    assert all(item["pass"] for item in reports[-1]["comparisons_to_first_run"])
    assert reports[-1]["median_seconds_by_width"] == {"8": 0.8, "16": 1.6}
    assert reports[-1]["metric_ids"] == ["pearson_ic"]
    assert all("factor_ids" not in item and "scalar_metrics" not in item
               and "observation_counts" not in item for item in reports[-1]["runs"])


def test_gpu_tile_width_ab_failure_keeps_partial_receipt(monkeypatch, tmp_path):
    args = SimpleNamespace(
        gpu_tile_widths=(8, 16), factors=61, days=0, assets=5500,
        max_object_mib=128, max_total_mib=4096, axis_index=None,
        output=tmp_path / "summary.json",
    )
    monkeypatch.setattr(
        harness.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=9),
    )
    with pytest.raises(SystemExit) as error:
        harness.run_gpu_tile_width_ab(args, harness.DEFAULT_METRICS)
    assert error.value.code == 1
    receipt = json.loads(args.output.read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["runs"] == []
    assert "median_seconds_by_width" not in receipt
