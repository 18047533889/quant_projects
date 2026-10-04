"""F61 orchestration tests replace external data and timed runs, not report checks."""
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from quant_evaluator.scripts import benchmark_real_cos_f61_profile_abba as cli
from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report
from test_f61_source_profile_report_reader import _fixture, _payload, METRICS_24


def test_full61_disk_budget_not_swapped_with_source_budget(monkeypatch):
    monkeypatch.setattr(cli, "preflight_research_cos_cli", lambda: {"configured_cli_resolvable": True})
    calls = []
    monkeypatch.setattr(cli.source_batch, "preflight", lambda *a: calls.append(a) or {"pass": True})
    assert cli.preflight()["pass"] is True
    assert calls == [(128, 6144, 4096)]


@pytest.fixture(params=("15", "24"))
def live(monkeypatch, tmp_path, request):
    payload = _payload(request.param)
    metric_ids = tuple(payload["metric_ids"])
    records = _fixture(metric_ids)
    calls = {"closed": 0, "auto": [], "warm": [], "oracle": []}
    dates, assets, rows = tuple(range(2586)), tuple(range(5461)), tuple(range(61))
    labels, oracle_bundle = object(), object()
    monkeypatch.setattr(cli, "preflight", lambda: dict(payload["preflight"]))
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda _: {})
    def select(mapping, count, object_mib, total_mib):
        assert (count, object_mib, total_mib) == (61, 128, 6144)
        return rows
    monkeypatch.setattr(cli.tiles, "select_source_records", select)
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (dates, assets, rows))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda d, a, *_: (d, a, labels))
    class Source:
        def close(self):
            calls["closed"] += 1
    source = Source()
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: source)
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: None, raising=False)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True, raising=False)
    monkeypatch.setattr(cli, "warm_source_profile", lambda p, b, m: calls["warm"].append((b, m)), raising=False)
    def independent(s, lb, **kw):
        assert s is source and lb is labels and kw["metrics"] == metric_ids
        assert kw["max_tile_size"] == 1
        calls["oracle"].append(True)
        return oracle_bundle
    monkeypatch.setattr(cli, "reference_source_all24", independent, raising=False)
    def produce(**kw):
        assert kw["run_kwargs"]["selected_metrics"] == metric_ids
        assert kw["run_kwargs"]["source_auto_policy"] == "qualified_only"
        for i in range(4):
            kw["progress_observer"]({"run_index": i})
        return SimpleNamespace(records=records, run_order=tuple(payload["run_order"]),
                               oracle_reports=tuple(payload["oracle_reports"]))
    monkeypatch.setattr(cli, "produce_source_route_profile_abba", produce, raising=False)
    def auto(backend, **kw):
        assert backend == "auto"
        calls["auto"].append(kw)
        return object(), {}
    monkeypatch.setattr(cli, "_checked_run_backend", auto, raising=False)
    def verify(bundle, receipt, qualification, context, oracle, **kw):
        assert oracle is oracle_bundle and context == records[0].context
        assert kw["selected_profile"] == records[0].cpu
        assert kw["run_index"] in (4, 5)
        assert kw["require_cache_hit"] is (kw["run_index"] == 5)
        return deepcopy(payload["default_auto_verification" if kw["run_index"] == 5 else "auto_verification"])
    monkeypatch.setattr(cli, "verify_source_profile_auto", verify, raising=False)
    axis, output = tmp_path / "axis.json", tmp_path / "report.json"
    axis.write_text("{}")
    args = ["--run", "--axis-index", str(axis), "--output", str(output)]
    if request.param == "15":
        args += ["--metric-set", "all15"]
    return SimpleNamespace(calls=calls, records=records, output=output, args=args, kind=payload["kind"])


def test_live_pipeline_verifies_explicit_then_ordinary_auto_and_writes_typed_report(live):
    # Supplying qualification on both auto calls would not test default adoption.
    assert cli.main(live.args) == 0
    assert load_source_profile_report(live.output).kind == live.kind
    assert live.calls["closed"] == 1 and live.calls["oracle"] == [True]
    assert [b for b, _ in live.calls["warm"]] == ["cpu", "cuda_strict"]
    explicit, default = live.calls["auto"]
    assert explicit["source_qualification"] == live.records
    assert "source_qualification" not in default
    progress = json.loads(live.output.with_suffix(".json.progress.json").read_text())
    assert progress["status"] == "complete" and progress["qualification_available"] is False
    assert len(progress["validated_runs"]) == 4


def test_cuda_unavailable_stops_before_cos(live, monkeypatch):
    monkeypatch.setattr(cli, "_runtime_ready", lambda: False)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("COS without CUDA"))
    with pytest.raises(SystemExit, match="CuPy unavailable"):
        cli.main(live.args)
    assert not live.output.exists()


def test_noncanonical_axis_stops_before_oracle(live, monkeypatch):
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (tuple(range(2400)), tuple(range(5000)), ()))
    monkeypatch.setattr(cli, "reference_source_all24", lambda *a, **k: pytest.fail("oracle on wrong axes"))
    loads = []
    original = cli.tiles.load_labels
    def load(*args):
        loads.append(True)
        return original(*args)
    monkeypatch.setattr(cli.tiles, "load_labels", load)
    with pytest.raises(SystemExit, match="F61"):
        cli.main(live.args)
    assert not live.output.exists()
    assert not loads, "noncanonical axes triggered a real label read"


def test_oracle_failure_closes_source_and_cannot_authorize_routing(live, monkeypatch):
    def fail(*a, **k):
        raise ValueError("independent oracle failed")
    monkeypatch.setattr(cli, "reference_source_all24", fail)
    with pytest.raises(ValueError, match="oracle failed"):
        cli.main(live.args)
    assert live.calls["closed"] == 1 and not live.output.exists()
    progress = json.loads(live.output.with_suffix(".json.progress.json").read_text())
    assert progress["status"] == "failed" and progress["qualification_available"] is False
