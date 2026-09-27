"""Offline contract checks for bounded research factor tile evaluation."""
import numpy as np
import pandas as pd
import pytest
import sys
import json
import copy
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
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": True})
    monkeypatch.setattr(tiles, "iter_frames", lambda *args: iter(()))
    monkeypatch.setattr(tiles, "intersect_axes", lambda stream, count:
                        (dates, {"A.SZ"}, sources))
    monkeypatch.setattr(tiles, "load_labels", lambda *args:
                        (dates, ["A.SZ"], SimpleNamespace(content_hash="label-hash")))
    monkeypatch.setattr(tiles, "make_tile", lambda *args: object())
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
    assert calls == [requested] * 5
    assert report["backend_requested"] == requested
    assert {tile["backend_used"] for tile in report["tiles"]} == {actual}
    assert report["tiles"][0]["execution_receipt"]["receipt_hash"] == "tile-receipt"
    assert all(perf[key] >= 0 for key in
               ("manifest_preflight_s", "first_pass_s", "labels_s", "total_s"))
    assert perf["peak_process_rss_kib"] > 0
    assert all(tile["load_s"] >= 0 and tile["evaluate_s"] >= 0 and
               tile["total_s"] >= tile["load_s"] + tile["evaluate_s"] and
               tile["peak_process_rss_kib"] > 0 for tile in perf["tiles"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["tile_complete"] for line in lines if "tile_complete" in line] == [1, 2, 3, 4, 5]


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
