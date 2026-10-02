"""Offline contract checks for bounded research factor tile evaluation."""
import io
import numpy as np
import pandas as pd
import pytest
import sys
import json
import copy
import threading
import time
import weakref
from dataclasses import replace
from types import SimpleNamespace

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles


def _mapping(count=34):
    return {f"f{i:02d}": {"verified": True,
            "status": "materialized_not_evaluated", "bytes": i + 1,
            "sha256": f"{i+1:064x}",
            "uri": f"{tiles.POOL}/{i+1:064x}/f{i:02d}.parquet"}
            for i in range(count)}


def test_selection_bound_order_and_manifest_identity():
    selected = tiles.select_records(_mapping(), 33, 128, 1)
    assert [r[0] for r in selected] == [f"f{i:02d}" for i in range(33)]
    with pytest.raises(ValueError, match="count"):
        tiles.select_records(_mapping(), 65, 128, 1)
    changed = _mapping()
    changed["f00"]["uri"] = "cos://wrong/object.parquet"
    with pytest.raises(ValueError, match="manifest bound"):
        tiles.select_records(changed, 33, 128, 1)
    changed = _mapping()
    changed["f01"]["sha256"] = changed["f00"]["sha256"]
    changed["f01"]["uri"] = changed["f00"]["uri"].replace("f00", "f01")
    with pytest.raises(ValueError, match="duplicated"):
        tiles.select_records(changed, 33, 128, 1)


@pytest.mark.parametrize("initial,available_gib,disk_free_mib,expected_pass,required_gib", [
    (True, 58, 2048, True, 58),
    (True, 57, 2048, False, 58),  # Initial check does not subtract existing RSS.
    (False, 48, 2048, True, 48),  # 50 GiB peak - 10 GiB RSS + 8 GiB guard.
    (False, 47, 2048, False, 48),
    (False, 60, 1024, False, 48),  # Memory is sufficient, cache disk is not.
])
def test_memory_preflight_initial_incremental_and_disk(
        monkeypatch, tmp_path, initial, available_gib, disk_free_mib,
        expected_pass, required_gib):
    monkeypatch.setattr(tiles, "open", lambda *args, **kwargs: io.StringIO(
        f"MemAvailable: {available_gib * 1024**2} kB\n"), raising=False)
    monkeypatch.setattr(tiles, "_current_process_rss_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(tiles, "cos_cache_root", lambda: tmp_path)
    monkeypatch.setattr(tiles.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=disk_free_mib * 1024**2))
    result = tiles.memory_preflight(8, 64, 50, initial)
    assert result["pass"] is expected_pass
    assert result["observed_process_rss_bytes"] == 10 * 1024**3
    assert result["remaining_incremental_peak_bytes"] == 40 * 1024**3
    assert result["required_incremental_headroom_bytes"] == 48 * 1024**3
    assert result["required_mem_available_bytes"] == required_gib * 1024**3
    assert result["required_disk_bytes"] == 1152 * 1024**2


def test_memory_preflight_rss_read_failure_is_closed(monkeypatch, tmp_path):
    def read(path, **kwargs):
        if path == "/proc/self/statm":
            raise OSError("RSS unavailable")
        return io.StringIO(f"MemAvailable: {80 * 1024**2} kB\n")
    monkeypatch.setattr(tiles, "open", read, raising=False)
    monkeypatch.setattr(tiles, "cos_cache_root", lambda: tmp_path)
    monkeypatch.setattr(tiles.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=2048 * 1024**2))
    with pytest.raises(OSError, match="RSS unavailable"):
        tiles.memory_preflight(8, 64, 50, False)


def test_current_process_rss_reads_resident_pages(monkeypatch):
    monkeypatch.setattr(tiles, "open", lambda *args, **kwargs: io.StringIO(
        "100 25 0 0 0 0 0\n"), raising=False)
    assert tiles._current_process_rss_bytes() == 25 * tiles.os.sysconf("SC_PAGE_SIZE")


def test_unavailable_manifest_never_starts_factor_reads(monkeypatch, capsys):
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(tiles, "iter_frames", lambda *args: pytest.fail("factor read started"))
    monkeypatch.setattr(sys, "argv", ["tiles", "--factors", "33"])
    tiles.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unavailable"


def test_collection_write_is_atomic_on_failure(monkeypatch, tmp_path):
    target = tmp_path / "report.json"
    target.write_text("old")
    def fail_replace(*args):
        raise OSError("disk error")
    monkeypatch.setattr(tiles.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk error"):
        tiles.write_collection_atomic(target, {"status": "complete"})
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("requested,actual,explicit", [
    ("cpu", "cpu", True), ("cuda_strict", "cuda", True),
    ("auto", "cpu", True), ("auto", "cuda", False),
])
def test_report_contains_phase_and_tile_timings(monkeypatch, tmp_path, capsys,
                                                requested, actual, explicit):
    output = tmp_path / "collection.json"
    dates = pd.date_range("2024-01-01", periods=2)
    sources = tuple((f"f{i:02d}", "uri", "sha", 1, "etag", "axis")
                    for i in range(33))
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: _mapping(33))
    preflights = []
    def fake_memory_preflight(*args):
        snapshot = {"pass": True, "observed_process_rss_bytes": 100 + len(preflights),
                    "required_incremental_headroom_bytes": 200 + len(preflights)}
        preflights.append(snapshot)
        return snapshot
    monkeypatch.setattr(tiles, "memory_preflight", fake_memory_preflight)
    monkeypatch.setattr(tiles, "iter_frames", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(tiles, "intersect_axes", lambda stream, count:
                        (dates, {"A.SZ"}, sources))
    monkeypatch.setattr(tiles, "load_labels", lambda *args:
                        (dates, ["A.SZ"], SimpleNamespace(content_hash="label-hash")))
    monkeypatch.setattr(tiles, "make_tile", lambda *args, **kwargs: object())
    calls = []
    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs["backend"])
        return object()
    monkeypatch.setattr(tiles, "evaluate", fake_evaluate)
    monkeypatch.setattr(tiles, "summarize_tile", lambda bundle, ids:
                        {"factor_ids": list(ids), "config_hash": "tile-hash",
                         "backend_requested": requested, "backend_used": actual,
                         "execution_receipt": {"receipt_hash": "tile-receipt",
                                               "config_hash": "tile-hash",
                                               "backend_requested": requested,
                                               "backend_used": actual}})
    argv = ["tiles", "--factors", "33", "--tile-size", "8"]
    if explicit:
        argv += ["--backend", requested]
    argv += ["--run", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    tiles.main()
    report = json.loads(output.read_text())
    perf = report["performance"]
    assert len(report["tiles"]) == len(perf["tiles"]) == 5
    assert len(preflights) == 6  # Initial gate plus one snapshot for each tile.
    assert [tile["memory_preflight"] for tile in perf["tiles"]] == preflights[1:]
    assert preflights[0] not in [tile["memory_preflight"] for tile in perf["tiles"]]
    assert calls == [requested] * 5
    assert report["backend_requested"] == requested
    assert {tile["backend_used"] for tile in report["tiles"]} == {actual}
    assert report["tiles"][0]["execution_receipt"]["receipt_hash"] == "tile-receipt"
    assert all(perf[key] >= 0 for key in
               ("manifest_preflight_s", "first_pass_s", "labels_s", "total_s"))
    assert perf["peak_process_rss_kib"] > 0
    assert set(perf["load_phases_s"]) == set(tiles.LOAD_PHASES)
    assert all(perf["load_phases_s"][phase] == sum(
        tile["load_phases_s"][phase] for tile in perf["tiles"])
        for phase in tiles.LOAD_PHASES)
    assert all(set(tile["load_phases_s"]) == set(tiles.LOAD_PHASES)
               for tile in perf["tiles"])
    assert all(tile["load_s"] >= 0 and tile["evaluate_s"] >= 0 and
               tile["total_s"] >= tile["load_s"] + tile["evaluate_s"] and
               tile["peak_process_rss_kib"] > 0 for tile in perf["tiles"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    tile_lines = [line for line in lines if "tile_complete" in line]
    assert [line["tile_complete"] for line in tile_lines] == [1, 2, 3, 4, 5]
    assert [line["memory_preflight"] for line in tile_lines] == preflights[1:]


def _collection_for_comparison():
    ids = [f"f{i:02d}" for i in range(33)]
    sources = [[fid, "uri", f"sha-{fid}", 1, "etag", "axis"] for fid in ids]
    report = {"status": "complete", "kind": "research_factor_tile_collection.v1",
              "metric_ids": list(tiles.METRICS), "manifest_sha256": "manifest",
              "label_content_hash": "label", "decision_time_first": "first",
              "decision_time_last": "last", "asset_axis_sha256": "asset",
              "selected_source_sha256": "source-set", "shape": [24, 40, 33],
              "sources": sources, "backend": "cpu", "tiles": []}
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        metrics = {metric: {fid: {"value": float(start+i)/10,
                                   "valid": True, "observation_count": 20,
                                   "metric_version": "1", "sample_unit": "daily_ic"}
                            for i, fid in enumerate(selected)}
                   for metric in tiles.METRICS}
        report["tiles"].append({"factor_ids": selected, "config_hash": f"config-{start}",
                                "execution_receipt": {"backend_requested": "cpu",
                                                      "backend_used": "cpu",
                                                      "config_hash": f"config-{start}"},
                                "metrics": metrics})
    return report


def test_compare_collections_accepts_legacy_receipt_only_route_and_tolerance():
    reference = _collection_for_comparison()  # saved F48 baseline schema: route in receipt
    candidate = copy.deepcopy(reference)
    candidate.pop("backend")
    candidate["backend_requested"] = "cuda_strict"
    for tile in candidate["tiles"]:
        tile["backend_requested"] = "cuda_strict"
        tile["backend_used"] = "cuda"
        tile["execution_receipt"].update(backend_requested="cuda_strict", backend_used="cuda")
    candidate["tiles"][0]["metrics"]["rank_ic"]["f00"]["value"] += 1e-11
    comparison = tiles.compare_collections(reference, candidate)
    assert comparison["pass"] is True
    assert comparison["compared_factor_count"] == 33
    assert comparison["compared_metric_count"] == 33 * 3


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(label_content_hash="different"),
    lambda r: r["sources"][0].__setitem__(2, "different"),
    lambda r: r.update(asset_axis_sha256="different"),
    lambda r: r["tiles"][0]["metrics"]["rank_ic"]["f00"].update(value=1.0),
    lambda r: r["tiles"][0]["metrics"]["rank_ic"]["f00"].update(valid=False),
    lambda r: r["tiles"][0]["metrics"]["rank_ic"]["f00"].update(observation_count=19),
    lambda r: r["tiles"][0]["metrics"]["rank_ic"]["f00"].update(metric_version="2"),
    lambda r: r["tiles"][0]["metrics"]["rank_ic"]["f00"].update(sample_unit="wrong"),
    lambda r: r["tiles"].pop(),
    lambda r: r["tiles"][0]["execution_receipt"].update(backend_used="cuda"),
])
def test_compare_collections_fails_closed_on_mismatch(mutation):
    reference = _collection_for_comparison()
    candidate = copy.deepcopy(reference)
    mutation(candidate)
    assert tiles.compare_collections(reference, candidate)["pass"] is False


def test_two_pass_axes_and_source_drift():
    dates = pd.date_range("2024-01-01", periods=4)
    a = pd.DataFrame({"B.SZ": [1., 2., 3., 4.], "A.SZ": [5., 6., 7., 8.]}, index=dates)
    b = pd.DataFrame({"A.SZ": [10., 11., 12.]}, index=dates[1:])
    source_a = ("a", "uri-a", "sha-a", 1, "etag-a", tiles.axis_hash(a))
    source_b = ("b", "uri-b", "sha-b", 1, "etag-b", tiles.axis_hash(b))
    common_dates, common_assets, sources = tiles.intersect_axes(
        iter([(a, source_a), (b, source_b)]), 2)
    assert common_dates.equals(dates[1:])
    assert common_assets == {"A.SZ"}
    axis = AxisRef("asset", "str", 1, np.array(["A.SZ"]))
    class Labels:
        asset_axis = axis
    batch = tiles.make_tile(iter([(a, source_a), (b, source_b)]), sources,
                            common_dates, ["A.SZ"], Labels())
    np.testing.assert_array_equal(batch.values[:, 0, 0], [6., 7., 8.])
    np.testing.assert_array_equal(batch.values[:, 0, 1], [10., 11., 12.])
    with pytest.raises(ValueError, match="changed"):
        tiles.make_tile(iter([(a, source_a), (b, source_b[:-1] + ("new",))]),
                        sources, common_dates, ["A.SZ"], Labels())


def test_make_tile_reuses_axis_indexers_for_shuffled_subset_and_nan():
    source_dates = pd.DatetimeIndex([
        "2024-01-03", "2024-01-01", "2024-01-02",
    ])
    frame = pd.DataFrame({
        "B.SZ": np.asarray([30.0, 10.0, np.nan], dtype=np.float64),
        "EXTRA.SH": np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
        "A.SZ": np.asarray([300.0, 100.0, 200.0], dtype=np.float64),
    }, index=source_dates)
    target_dates = source_dates[[1, 2]]
    target_assets = ["A.SZ", "B.SZ"]
    source = ("f0", "uri", "sha", 1, "etag", tiles.axis_hash(frame))
    labels = type("Labels", (), {
        "asset_axis": AxisRef("asset", "str", 2,
                              np.asarray(target_assets, dtype=str)),
    })()

    batch = tiles.make_tile(iter([(frame, source)]), (source,), target_dates,
                            target_assets, labels)
    expected = frame.reindex(index=target_dates,
                             columns=target_assets).to_numpy(dtype=np.float64)

    assert batch.values.dtype == np.float64
    np.testing.assert_array_equal(batch.values[:, :, 0], expected)
    assert np.isnan(batch.values[1, 1, 0])


def test_axis_materializer_memory_bounded_mode_uses_multiple_row_chunks():
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=3)
    assets = ["A.SZ", "B.SH"]
    frame = pd.DataFrame({
        "B.SH": [2.0, np.nan, 6.0],
        "EXTRA.SZ": [9.0, 8.0, 7.0],
        "A.SZ": [1.0, 3.0, 5.0],
    }, index=dates[::-1])
    target_dates = dates[:2]
    output = np.empty((len(target_dates), len(assets)), dtype=np.float64)

    module.write_axis_aligned_float64(
        frame, target_dates, assets, output,
        memory_bounded=True, chunk_bytes=40)
    expected = frame.reindex(index=target_dates, columns=assets).to_numpy(
        dtype=np.float64, copy=False)

    assert output.dtype == np.float64
    np.testing.assert_array_equal(output, expected)
    assert np.isnan(output[1, 1])


@pytest.mark.parametrize("memory_bounded", [1, None, "yes"])
def test_axis_materializer_rejects_non_boolean_memory_mode(memory_bounded):
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": [1.0, 2.0]}, index=dates)
    output = np.empty((len(dates), 1), dtype=np.float64)

    with pytest.raises(ValueError, match="memory_bounded must be bool"):
        module.write_axis_aligned_float64(
            frame, dates, ["A.SZ"], output, memory_bounded=memory_bounded)


@pytest.mark.parametrize("output_kind", ["shape", "dtype", "readonly"])
def test_axis_materializer_rejects_invalid_output(output_kind):
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": [1.0, 2.0]}, index=dates)
    if output_kind == "shape":
        output = np.empty((1, 1), dtype=np.float64)
    elif output_kind == "dtype":
        output = np.empty((len(dates), 1), dtype=np.float32)
    else:
        output = np.empty((len(dates), 1), dtype=np.float64)
        output.flags.writeable = False

    with pytest.raises(ValueError, match="output must be a writable float64 array"):
        module.write_axis_aligned_float64(frame, dates, ["A.SZ"], output)


@pytest.mark.parametrize("missing_axis", ["dates", "assets"])
def test_axis_materializer_requires_shared_axes_as_a_pair(missing_axis):
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": [1.0, 2.0]}, index=dates)
    output = np.empty((len(dates), 1), dtype=np.float64)
    kwargs = ({"required_dates": dates} if missing_axis == "assets" else
              {"required_assets": {"A.SZ"}})

    with pytest.raises(ValueError, match="must be provided together"):
        module.write_axis_aligned_float64(
            frame, dates, ["A.SZ"], output, **kwargs)


def test_axis_materializer_rejects_chunk_budget_below_one_logical_row():
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=3)
    assets = ["A.SZ", "B.SH"]
    frame = pd.DataFrame({
        "B.SH": [2.0, np.nan, 6.0],
        "EXTRA.SZ": [9.0, 8.0, 7.0],
        "A.SZ": [1.0, 3.0, 5.0],
    }, index=dates[::-1])
    output = np.empty((2, len(assets)), dtype=np.float64)

    with pytest.raises(ValueError, match="at least one logical row"):
        module.write_axis_aligned_float64(
            frame, dates[:2], assets, output,
            memory_bounded=True, chunk_bytes=39)


def test_axis_materializer_empty_date_axis_needs_no_chunk_budget():
    module = tiles.source_axis_materializer
    dates = pd.date_range("2024-01-01", periods=2)
    assets = ["A.SZ", "B.SH"]
    frame = pd.DataFrame({"A.SZ": [1.0, 2.0], "B.SH": [3.0, 4.0]},
                         index=dates)
    output = np.empty((0, len(assets)), dtype=np.float64)

    module.write_axis_aligned_float64(
        frame, dates[:0], assets, output,
        memory_bounded=True, chunk_bytes=1)

    assert output.shape == (0, len(assets))


@pytest.mark.parametrize("missing_axis", ["date", "asset"])
def test_make_tile_fails_closed_when_target_axis_is_missing(missing_axis):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": [1.0, np.nan], "B.SH": [2.0, 3.0]},
                         index=dates)
    target_dates = dates if missing_axis == "asset" else dates.append(
        pd.DatetimeIndex(["2024-01-03"]))
    target_assets = ["A.SZ", "B.SH"] if missing_axis == "date" else [
        "A.SZ", "MISSING.SZ"]
    source = ("f0", "uri", "sha", 1, "etag", tiles.axis_hash(frame))
    labels = type("Labels", (), {
        "asset_axis": AxisRef("asset", "str", len(target_assets),
                              np.asarray(target_assets, dtype=str)),
    })()

    with pytest.raises(ValueError, match="factor final axes changed"):
        tiles.make_tile(iter([(frame, source)]), (source,), target_dates,
                        target_assets, labels)


def test_make_tile_checks_required_shared_axes_before_final_axes():
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": [1.0, 2.0]}, index=dates)
    source = ("f0", "uri", "sha", 1, "etag", tiles.axis_hash(frame))
    labels = type("Labels", (), {
        "asset_axis": AxisRef("asset", "str", 1, np.asarray(["A.SZ"])),
    })()

    with pytest.raises(ValueError, match="indexed shared axes changed"):
        tiles.make_tile(
            iter([(frame, source)]), (source,),
            dates.append(pd.DatetimeIndex(["2024-01-03"])), ["MISSING.SZ"],
            labels, required_dates=dates.append(
                pd.DatetimeIndex(["2024-01-04"])), required_assets={"A.SZ"})


def test_make_tile_rejects_non_numeric_object_values():
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A.SZ": np.asarray([1.0, "bad"], dtype=object)},
                         index=dates)
    source = ("f0", "uri", "sha", 1, "etag", tiles.axis_hash(frame))
    labels = type("Labels", (), {
        "asset_axis": AxisRef("asset", "str", 1, np.asarray(["A.SZ"])),
    })()

    with pytest.raises((TypeError, ValueError)):
        tiles.make_tile(iter([(frame, source)]), (source,), dates,
                        ["A.SZ"], labels)


@pytest.mark.parametrize("duplicate_axis", ["date", "asset"])
def test_make_tile_rejects_duplicate_source_axes(duplicate_axis):
    dates = pd.date_range("2024-01-01", periods=2)
    if duplicate_axis == "date":
        frame = pd.DataFrame({"A.SZ": [1.0, 2.0]},
                             index=pd.DatetimeIndex([dates[0], dates[0]]))
    else:
        frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                             index=dates, columns=["A.SZ", "A.SZ"])
    source = ("f0", "uri", "sha", 1, "etag", tiles.axis_hash(frame))
    labels = type("Labels", (), {
        "asset_axis": AxisRef("asset", "str", 1, np.asarray(["A.SZ"])),
    })()

    with pytest.raises((ValueError, pd.errors.InvalidIndexError)):
        tiles.make_tile(iter([(frame, source)]), (source,), dates[:1],
                        ["A.SZ"], labels)


def test_load_phase_timing_preserves_factor_values_and_axis_checks(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2)
    table = SimpleNamespace(to_pandas=lambda: pd.DataFrame(
        {"timestamp": dates, "A.SZ": [1.0, np.nan]}))
    record = ("factor", "cos://declared/factor.parquet", "a" * 64, 123)
    factor = SimpleNamespace(table=table, source_uri=record[1],
                             content_sha256=record[2], downloaded_bytes=record[3],
                             source_etag="etag")
    monkeypatch.setattr(tiles, "_ds", lambda name, *args: SimpleNamespace(name=name))
    monkeypatch.setattr(tiles, "DuckDBEngine", lambda **kwargs:
                        SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(tiles, "DatasetRegistry", lambda mapping: mapping)
    monkeypatch.setattr(tiles, "DataAccessStore", lambda *args: object())
    monkeypatch.setattr(tiles, "read_bound_factor", lambda *args, **kwargs:
                        SimpleNamespace(factor=factor, manifest_sha256="m" * 64))
    phases = {phase: 0.0 for phase in tiles.LOAD_PHASES}
    expected_frame = table.to_pandas().set_index("timestamp")
    expected_source = (*record, "etag", tiles.axis_hash(expected_frame))
    labels = SimpleNamespace(asset_axis=AxisRef(
        "asset", "str", 1, np.array(["A.SZ"])))
    batch = tiles.make_tile(tiles.iter_frames([record], "m" * 64, 128,
                                             load_phases=phases),
                            (expected_source,), dates, ["A.SZ"], labels,
                            load_phases=phases)
    np.testing.assert_array_equal(batch.values[:, 0, 0], [1.0, np.nan])
    assert all(phases[phase] > 0 for phase in tiles.LOAD_PHASES)
    assert set(phases) == set(tiles.LOAD_PHASES)


def test_default_metrics_tile_values_match_full_batch_cpu():
    rng = np.random.default_rng(7)
    times = pd.date_range("2024-01-01", periods=24).to_numpy()
    assets = np.array([f"S{i}.SZ" for i in range(40)])
    values = rng.normal(size=(24, 40, 33))
    values[2, 3, 32] = np.nan
    returns = rng.normal(scale=0.01, size=(24, 40))
    t_axis = AxisRef("time", "datetime64[ns]", len(times), times)
    a_axis = AxisRef("asset", "str", len(assets), assets)
    ids = tuple(f"f{i}" for i in range(33))
    labels = LabelBundle("test-return", returns, 1, decision_time=tuple(times),
        label_start_time=tuple(times + np.timedelta64(1, "D")),
        label_end_time=tuple(times + np.timedelta64(2, "D")), asset_axis=a_axis)
    full = FactorBatch(ids, t_axis, a_axis, values, validity=np.isfinite(values))
    reference = evaluate(full, labels, metrics=tiles.METRICS, backend="cpu")
    observed = {}
    for start in (0, 8, 16, 24, 32):
        stop = min(start + 8, 33)
        tile = FactorBatch(ids[start:stop], t_axis, a_axis, values[:, :, start:stop],
                           validity=np.isfinite(values[:, :, start:stop]))
        bundle = evaluate(tile, labels, metrics=tiles.METRICS, backend="cpu")
        summary = tiles.summarize_tile(bundle, ids[start:stop])
        assert summary["backend_requested"] == summary["backend_used"] == "cpu"
        assert summary["execution_receipt"]["backend_used"] == "cpu"
        if start == 0:
            original = bundle.grouped_metrics[ids[0]]["rank_ic"]
            bundle.grouped_metrics[ids[0]]["rank_ic"] = replace(
                original, observation_count=original.observation_count + 1)
            with pytest.raises(ValueError, match="differs from artifact"):
                tiles.summarize_tile(bundle, ids[start:stop])
        for metric in tiles.METRICS:
            observed.setdefault(metric, {}).update(summary["metrics"][metric])
    for metric in tiles.METRICS:
        for fid in ids:
            expected = reference.grouped_metrics[fid][metric]
            actual = observed[metric][fid]
            assert actual["value"] == pytest.approx(expected.value, nan_ok=True)
            assert actual["valid"] == expected.valid
            assert actual["observation_count"] == expected.observation_count


def test_max_total_mib_can_explicitly_cover_f61():
    mapping = _mapping(61)
    for row in mapping.values():
        row["bytes"] = 80 * 1024**2
    with pytest.raises(ValueError, match="sample unavailable"):
        tiles.select_records(mapping, 61, 128, 4096)
    assert len(tiles.select_records(mapping, 61, 128, 6144)) == 61
    with pytest.raises(ValueError, match="6144"):
        tiles.select_records(mapping, 61, 128, 6145)


def _small_axis_index():
    records = tiles.select_records(_mapping(33), 33, 128, 1)
    dates = pd.date_range("2024-01-01", periods=3)
    assets = {"A.SZ", "B.SH"}
    sources = tuple((*record, "etag", "a" * 64) for record in records)
    return records, dates, assets, sources


def test_axis_index_round_trip_identity_and_tamper(tmp_path):
    records, dates, assets, sources = _small_axis_index()
    index = tiles.build_axis_index("m" * 64, records, dates, assets, sources)
    path = tmp_path / "axes.json"
    tiles.write_axis_index_atomic(path, index)
    loaded_dates, loaded_assets, loaded_sources = tiles.read_axis_index(
        path, "m" * 64, records)
    assert loaded_dates.equals(dates)
    assert loaded_assets == assets
    assert loaded_sources == sources
    with pytest.raises(ValueError, match="selection or manifest"):
        tiles.read_axis_index(path, "n" * 64, records)
    with pytest.raises(ValueError, match="selection or manifest"):
        tiles.read_axis_index(path, "m" * 64, records[::-1])
    changed = json.loads(path.read_text())
    changed["dates_ns"].pop()
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum"):
        tiles.read_axis_index(path, "m" * 64, records)


def test_axis_index_bounded_and_atomic(monkeypatch, tmp_path):
    records, dates, assets, sources = _small_axis_index()
    with pytest.raises(ValueError, match="bound"):
        tiles.build_axis_index("m" * 64, records,
                               pd.date_range("2024-01-01", periods=10001), assets, sources)
    index = tiles.build_axis_index("m" * 64, records, dates, assets, sources)
    path = tmp_path / "axes.json"
    tiles.write_axis_index_atomic(path, index)
    old = path.read_bytes()
    monkeypatch.setattr(tiles.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        tiles.write_axis_index_atomic(path, index)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]
    path.write_bytes(b"x" * (tiles.AXIS_INDEX_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="bound"):
        tiles.read_axis_index(path, "m" * 64, records)


def test_axis_index_reuse_skips_first_pass_and_preserves_report(monkeypatch, tmp_path):
    records, dates, assets, _ = _small_axis_index()
    columns = sorted(assets)
    frames = [pd.DataFrame({name: [1., 2., 3.] for name in columns}, index=dates)
              for _ in records]
    source_rows = tuple((*record, "etag", tiles.axis_hash(frame))
                        for record, frame in zip(records, frames))
    reads = []
    def frames_for(selected, *args, **kwargs):
        reads.extend(row[0] for row in selected)
        for row in selected:
            idx = records.index(row)
            yield frames[idx], source_rows[idx]
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: _mapping(33))
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": True})
    monkeypatch.setattr(tiles, "iter_frames", frames_for)
    label = SimpleNamespace(content_hash="label",
                            asset_axis=AxisRef("asset", "str", 2, np.array(columns)))
    label_reads = []
    def fresh_labels(*args):
        label_reads.append(True)
        return dates, columns, label
    monkeypatch.setattr(tiles, "load_labels", fresh_labels)
    monkeypatch.setattr(tiles, "evaluate", lambda *args, **kwargs: object())
    monkeypatch.setattr(tiles, "summarize_tile", lambda bundle, ids:
                        {"factor_ids": list(ids), "backend_requested": "cpu",
                         "backend_used": "cpu", "config_hash": "same",
                         "execution_receipt": {"backend_requested": "cpu",
                                               "backend_used": "cpu",
                                               "config_hash": "same"}})
    index_path = tmp_path / "axes.json"
    def run(output):
        monkeypatch.setattr(sys, "argv", ["tiles", "--factors", "33", "--run",
                                        "--backend", "cpu", "--axis-index", str(index_path),
                                        "--output", str(output)])
        tiles.main()
        return json.loads(output.read_text())
    first = run(tmp_path / "first.json")
    assert len(reads) == 66
    assert index_path.exists()
    reads.clear()
    monkeypatch.setattr(tiles, "intersect_axes", lambda *args:
                        pytest.fail("reused index started first pass"))
    second = run(tmp_path / "second.json")
    assert len(reads) == 33
    assert len(label_reads) == 2
    assert first["performance"]["axis_index_reused"] is False
    assert second["performance"]["axis_index_reused"] is True
    first.pop("performance")
    second.pop("performance")
    assert first == second
    frames[0] = frames[0].drop(columns="B.SH")
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path / "drift.json")
    frames[0] = pd.DataFrame({name: [1., 2., 3.] for name in columns}, index=dates)
    source_rows = ((source_rows[0][0], source_rows[0][1], source_rows[0][2],
                    source_rows[0][3], "different-etag", source_rows[0][5]), *source_rows[1:])
    with pytest.raises(ValueError, match="source or axes changed"):
        run(tmp_path / "source-drift.json")


def test_prefetch_is_bounded_ordered_and_owns_engines(monkeypatch):
    records = tuple((f"f{i}", f"{tiles.POOL}/{i+1:064x}/f{i}.parquet",
                     f"{i+1:064x}", i + 1) for i in range(4))
    dates = pd.date_range("2024-01-01", periods=2)
    main_thread = threading.get_ident()
    engines = []
    calls = []
    first_frame_ref = [None]
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    class Engine:
        def __init__(self, **kwargs):
            self.owner = threading.get_ident()
            self.closed = False
            engines.append(self)
        def close(self):
            self.closed = True

    monkeypatch.setattr(tiles, "DuckDBEngine", Engine)
    monkeypatch.setattr(tiles, "_ds", lambda name, *args: SimpleNamespace(name=name))
    monkeypatch.setattr(tiles, "DatasetRegistry", lambda mapping: mapping)
    monkeypatch.setattr(tiles, "DataAccessStore",
                        lambda registry, engine: SimpleNamespace(engine=engine))
    snapshot = SimpleNamespace(manifest_sha256="m" * 64)
    def read_manifest(store, name, *, allow_research):
        assert store.engine.owner == main_thread
        assert allow_research is True
        return snapshot
    verify_calls = []
    def verify_manifest(store, name, actual):
        verify_calls.append((store.engine.owner, actual))
    monkeypatch.setattr(tiles, "read_bound_manifest", read_manifest)
    monkeypatch.setattr(tiles, "verify_bound_manifest_unchanged", verify_manifest)
    original_axis_hash = tiles.axis_hash
    def capture_axis_hash(frame):
        if frame.iloc[0, 0] == 1.0:
            first_frame_ref[0] = weakref.ref(frame)
        return original_axis_hash(frame)
    monkeypatch.setattr(tiles, "axis_hash", capture_axis_hash)

    def read_factor(store, manifest_name, factor_name, fid, **kwargs):
        if fid == "f2":
            assert first_frame_ref[0] is not None and first_frame_ref[0]() is None
        assert store.engine.owner == threading.get_ident()
        assert store.engine.owner != main_thread
        assert kwargs["allow_research"] is True
        assert kwargs["manifest_snapshot"] is snapshot
        assert kwargs["max_object_mib"] == 8
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            calls.append((fid, store.engine))
        time.sleep(0.02)
        record = next(row for row in records if row[0] == fid)
        value = float(int(fid[1:]) + 1)
        frame = pd.DataFrame({"timestamp": dates, "A.SZ": [value, value + 1.0]})
        factor = SimpleNamespace(
            table=SimpleNamespace(to_pandas=lambda: frame.copy()),
            source_uri=record[1], content_sha256=record[2],
            downloaded_bytes=record[3], source_etag=f"etag-{fid}")
        with lock:
            state["active"] -= 1
        return SimpleNamespace(factor=factor, manifest_sha256="m" * 64)
    monkeypatch.setattr(tiles, "read_bound_factor", read_factor)

    common_dates, common_assets, sources = tiles.intersect_axes(
        tiles.iter_frames_prefetched(records, "m" * 64, 8), len(records))
    assert common_dates.equals(dates)
    assert common_assets == {"A.SZ"}
    assert [source[0] for source in sources] == [row[0] for row in records]
    assert state["max_active"] == 2
    assert len({id(engine) for _, engine in calls}) == len(records)
    assert all(engine.owner != main_thread and engine.closed for _, engine in calls)
    assert verify_calls == [(main_thread, snapshot)]
    controller_engines = [engine for engine in engines if engine.owner == main_thread]
    assert len(controller_engines) == 1 and controller_engines[0].closed


def test_prefetch_failure_closes_engines_and_does_not_verify_partial_shard(monkeypatch):
    records = tuple((f"f{i}", f"{tiles.POOL}/{i+1:064x}/f{i}.parquet",
                     f"{i+1:064x}", i + 1) for i in range(3))
    engines = []
    class Engine:
        def __init__(self, **kwargs):
            self.closed = False
            engines.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr(tiles, "DuckDBEngine", Engine)
    monkeypatch.setattr(tiles, "_ds", lambda name, *args: SimpleNamespace(name=name))
    monkeypatch.setattr(tiles, "DatasetRegistry", lambda mapping: mapping)
    monkeypatch.setattr(tiles, "DataAccessStore",
                        lambda registry, engine: SimpleNamespace(engine=engine))
    snapshot = SimpleNamespace(manifest_sha256="m" * 64)
    monkeypatch.setattr(tiles, "read_bound_manifest", lambda *args, **kwargs: snapshot)
    verified = []
    monkeypatch.setattr(tiles, "verify_bound_manifest_unchanged",
                        lambda *args: verified.append(True))
    def fail_factor(store, manifest_name, factor_name, fid, **kwargs):
        raise RuntimeError("offline read failure")
    monkeypatch.setattr(tiles, "read_bound_factor", fail_factor)
    stream = tiles.iter_frames_prefetched(records, "m" * 64, 8)
    with pytest.raises(RuntimeError, match="offline read failure"):
        next(stream)
    assert verified == []
    assert engines and all(engine.closed for engine in engines)


def test_axis_and_tile_consumers_release_frames_before_next_request():
    dates = pd.date_range("2024-01-01", periods=2)
    records = tuple((f"f{i}", f"uri-{i}", f"sha-{i}", 1, f"etag-{i}", f"axis-{i}")
                    for i in range(2))

    class WeakFrameStream:
        def __init__(self):
            self.index = 0
            self.previous = None
            self.closed = False
        def __iter__(self):
            return self
        def __next__(self):
            if self.index >= len(records):
                raise StopIteration
            if self.previous is not None:
                assert self.previous() is None
            frame = pd.DataFrame({"A.SZ": [float(self.index + 1), 2.0]}, index=dates)
            self.previous = weakref.ref(frame)
            result = (frame, records[self.index])
            self.index += 1
            return result
        def close(self):
            self.closed = True

    axis_stream = WeakFrameStream()
    common_dates, common_assets, sources = tiles.intersect_axes(axis_stream, len(records))
    assert common_dates.equals(dates) and common_assets == {"A.SZ"}
    assert sources == records and axis_stream.closed

    asset_axis = AxisRef("asset", "str", 1, np.array(["A.SZ"]))
    labels = SimpleNamespace(asset_axis=asset_axis)
    tile_stream = WeakFrameStream()
    batch = tiles.make_tile(tile_stream, records, dates, ["A.SZ"], labels)
    np.testing.assert_array_equal(batch.values[:, 0, :], [[1.0, 2.0], [2.0, 2.0]])
    assert tile_stream.closed


def test_intersect_axes_closes_stream_after_iteration_failure():
    class FailingStream:
        closed = False
        def __iter__(self):
            return self
        def __next__(self):
            raise RuntimeError("offline iteration failure")
        def close(self):
            self.closed = True
    stream = FailingStream()
    with pytest.raises(RuntimeError, match="offline iteration failure"):
        tiles.intersect_axes(stream, 1)
    assert stream.closed


def test_prefetch_cli_reports_manifest_snapshot_reuse(monkeypatch, tmp_path, capsys):
    output = tmp_path / "prefetch.json"
    dates = pd.date_range("2024-01-01", periods=2)
    records = tuple((f"f{i:02d}", "uri", f"sha-{i}", 1, "etag", "axis")
                    for i in range(33))
    calls = []
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: _mapping(33))
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": True})
    def fake_prefetch(selected, manifest_sha, object_mib, **kwargs):
        calls.append(tuple(selected))
        return iter(())
    monkeypatch.setattr(tiles, "iter_frames_prefetched", fake_prefetch)
    monkeypatch.setattr(tiles, "intersect_axes", lambda stream, count:
                        (dates, {"A.SZ"}, records))
    labels = SimpleNamespace(content_hash="label-hash", asset_axis=AxisRef(
        "asset", "str", 1, np.array(["A.SZ"])))
    monkeypatch.setattr(tiles, "load_labels", lambda *args: (dates, ["A.SZ"], labels))
    monkeypatch.setattr(tiles, "make_tile", lambda *args, **kwargs: object())
    monkeypatch.setattr(tiles, "evaluate", lambda *args, **kwargs: object())
    def summarize(bundle, ids):
        return {"factor_ids": list(ids), "backend_requested": "auto",
                "backend_used": "cpu", "execution_receipt": {
                    "backend_requested": "auto", "backend_used": "cpu"}}
    monkeypatch.setattr(tiles, "summarize_tile", summarize)
    monkeypatch.setattr(sys, "argv", ["tiles", "--factors", "33", "--tile-size", "32",
                                      "--run", "--prefetch", "--output", str(output)])
    tiles.main()
    report = json.loads(output.read_text())
    assert len(calls) == 3  # first pass and two tile shards
    assert report["performance"]["manifest_reused"] is True
    assert report["performance"]["object_prefetch"] == {
        "enabled": True, "max_unconsumed_objects": 2}
    capsys.readouterr()
