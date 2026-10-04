"""CLI safety contract for fresh-provider verification of F61 reports."""
from __future__ import annotations

import json

import pytest

from quant_evaluator.scripts import verify_real_cos_f61_provider as cli
from quant_evaluator.runtime import source_qualification_provider as provider_api
from test_f61_source_profile_report_reader import _payload


def _inputs(tmp_path, version="24"):
    axis, profile, output = (tmp_path / "axis.json", tmp_path / "profile.json",
                             tmp_path / "result.json")
    axis.write_text("{}", encoding="utf-8")
    payload = _payload(version)
    profile.write_text(json.dumps(payload), encoding="utf-8")
    return axis, profile, output, payload


def _prepare_run(tmp_path, monkeypatch, payload, *, oracle_error=False):
    axis, profile, output, _ = _inputs(tmp_path)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "has_validated_records", lambda: False)
    monkeypatch.setattr(cli, "warm_source_profile", lambda *a: None)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: object())
    records = tuple(f"factor-{i}" for i in range(61))
    dates, assets, rows = tuple(range(2586)), tuple(range(5461)), tuple(range(61))
    labels = object()
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (dates, assets, rows))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda d, a, *args: (d, a, labels))

    class Source:
        closed = False

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: source)
    if oracle_error:
        def fail_oracle(*a, **k):
            raise RuntimeError("oracle failed")
        monkeypatch.setattr(cli, "reference_source_all24", fail_oracle)
    else:
        monkeypatch.setattr(cli, "reference_source_all24", lambda *a, **k: object())
    args = ["--run", "--axis-index", str(axis), "--profile-report", str(profile),
            "--output", str(output)]
    return args, output, source


def test_default_cli_is_preflight_only_and_reports_both_profile_kinds(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True, "ram": "ok"})
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("dry-run read COS"))
    monkeypatch.setattr(cli.source_batch, "_make_cos_source",
                        lambda *a, **k: pytest.fail("dry-run made COS source"))
    assert cli.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "real_cos_f61_fresh_provider.v1"
    assert result["status"] == "preflight_only" and result["run_started"] is False
    assert result["shape"] == [2586, 5461, 61]
    assert result["allowed_profile_kinds"] == [
        "real_cos_profile_abba_f61_all15.v1", "real_cos_profile_abba_f61_all24.v1"]
    assert "metric_ids" not in result


def test_run_requires_all_paths_before_preflight(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("invalid CLI reached preflight"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--run"])
    assert error.value.code == 2


def test_existing_output_is_preserved_before_preflight(tmp_path, monkeypatch):
    axis, profile, output, _ = _inputs(tmp_path)
    output.write_text("owned", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("collision reached preflight"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                  "--output", str(output)])
    assert error.value.code == 2 and output.read_text(encoding="utf-8") == "owned"


@pytest.mark.parametrize("version", ["15", "24"])
def test_strict_reader_accepts_only_the_two_f61_domains_before_cos(
        tmp_path, monkeypatch, capsys, version):
    axis, profile, output, payload = _inputs(tmp_path, version)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    reads = []
    monkeypatch.setattr(cli.tiles, "read_manifest",
                        lambda *a: (reads.append(a), (_ for _ in ()).throw(RuntimeError("stop")))[1])
    result = cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                       "--output", str(output)])
    assert result == 1 and reads
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["kind", "manifest"])
def test_wrong_report_kind_or_manifest_stops_before_cos(tmp_path, monkeypatch, mutation):
    axis, profile, output, payload = _inputs(tmp_path)
    if mutation == "kind":
        payload["kind"] = "real_cos_profile_abba_f61_all15.v1"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256",
                        "f" * 64 if mutation == "manifest" else _payload()["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("invalid report read COS"))
    assert cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                     "--output", str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize("guard", ["provider", "cache"])
def test_existing_provider_or_live_validated_cache_blocks_cos(tmp_path, monkeypatch, capsys, guard):
    axis, profile, output, payload = _inputs(tmp_path)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    existing = provider_api.FileSourceQualificationProvider({"b" * 64: profile})
    if guard == "provider":
        provider_api.configure_source_qualification_provider(existing)
    monkeypatch.setattr(cli, "has_validated_records", lambda: guard == "cache")
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("guard ran after COS"))
    try:
        result = cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                           "--output", str(output)])
        assert result == 1 and not output.exists()
        assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
        assert provider_api.get_source_qualification_provider() is (
            existing if guard == "provider" else None)
    finally:
        if provider_api.get_source_qualification_provider() is existing:
            provider_api.clear_source_qualification_provider()


def test_bad_axis_shape_is_rejected_before_loading_labels(tmp_path, monkeypatch, capsys):
    axis, profile, output, payload = _inputs(tmp_path)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "get_source_qualification_provider", lambda: None)
    monkeypatch.setattr(cli, "has_validated_records", lambda: False)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: object())
    records = tuple(f"factor-{i}" for i in range(61))
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(cli.tiles, "read_axis_index",
                        lambda *a: (tuple(range(2585)), tuple(range(5461)), tuple(range(61))))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda *a: pytest.fail("labels loaded before shape check"))
    assert cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                     "--output", str(output)]) == 1
    assert capsys.readouterr().err.strip() == "error_code=ValueError"


def test_success_uses_two_ordinary_auto_runs_and_cleans_only_owned_provider(
        tmp_path, monkeypatch):
    axis, profile, output, payload = _inputs(tmp_path)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "has_validated_records", lambda: False)
    warm = []
    monkeypatch.setattr(cli, "warm_source_profile",
                        lambda policy, backend, metrics: warm.append((backend, tuple(metrics))))
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: object())
    records = tuple(f"factor-{i}" for i in range(61))
    dates, assets, source_rows = tuple(range(2586)), tuple(range(5461)), tuple(range(61))
    labels = object()
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (dates, assets, source_rows))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda d, a, *args: (d, a, labels))
    class OracleSource:
        def close(self):
            pass
    oracle_source = OracleSource()
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: oracle_source)
    monkeypatch.setattr(cli, "reference_source_all24", lambda *a, **k: object())
    runs = []
    def run_backend(backend, **kwargs):
        assert backend == "auto" and "source_qualification" not in kwargs
        assert kwargs["source_auto_policy"] == "qualified_only"
        assert callable(kwargs["context_observer"])
        assert isinstance(provider_api.get_source_qualification_provider(),
                          provider_api.FileSourceQualificationProvider)
        lookup = provider_api.lookup_source_qualification_candidate(
            payload["profile_records"][0]["context"]["request_content_sha256"])
        assert lookup.reason_code == "candidate_found"
        runs.append(kwargs)
        return object(), {"backend_used": "cpu"}
    monkeypatch.setattr(cli, "_checked_run_backend", run_backend)
    verified = []
    monkeypatch.setattr(cli, "verify_provider_run",
                        lambda *a, **kw: (verified.append(kw) or {"verified": True}))
    assert cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                     "--output", str(output)]) == 0
    assert len(runs) == 2 and all("source_qualification" not in run for run in runs)
    assert [call["run_index"] for call in verified] == [0, 1]
    assert warm == [("cpu", tuple(payload["metric_ids"]))]
    assert provider_api.get_source_qualification_provider() is None
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["kind"] == "real_cos_f61_fresh_provider.v1"
    assert result["status"] == "complete" and result["shape"] == [2586, 5461, 61]


def test_runtime_exception_emits_only_exception_type(tmp_path, monkeypatch, capsys):
    axis, profile, output, _ = _inputs(tmp_path)
    monkeypatch.setattr(cli, "preflight",
                        lambda: (_ for _ in ()).throw(RuntimeError("secret path and token")))
    assert cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
                     "--output", str(output)]) == 1
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"


def test_oracle_exception_closes_source_without_creating_output(tmp_path, monkeypatch, capsys):
    _, _, _, payload = _inputs(tmp_path)
    args, output, source = _prepare_run(tmp_path, monkeypatch, payload, oracle_error=True)
    assert cli.main(args) == 1
    assert source.closed
    assert not output.exists()
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    assert provider_api.get_source_qualification_provider() is None


@pytest.mark.parametrize("failure_point", ["backend", "verifier", "foreign_backend", "foreign_verifier"])
def test_run_failure_cleans_only_owned_provider(tmp_path, monkeypatch, capsys, failure_point):
    _, _, _, payload = _inputs(tmp_path)
    args, output, _ = _prepare_run(tmp_path, monkeypatch, payload)
    foreign = provider_api.FileSourceQualificationProvider({"c" * 64: output})

    def fail_backend(*a, **k):
        assert isinstance(provider_api.get_source_qualification_provider(),
                          provider_api.FileSourceQualificationProvider)
        if failure_point == "foreign_backend":
            provider_api.configure_source_qualification_provider(foreign)
        raise RuntimeError("backend failed")

    def fail_verifier(*a, **k):
        assert isinstance(provider_api.get_source_qualification_provider(),
                          provider_api.FileSourceQualificationProvider)
        if failure_point == "foreign_verifier":
            provider_api.configure_source_qualification_provider(foreign)
        raise RuntimeError("verifier failed")

    monkeypatch.setattr(cli, "_checked_run_backend", fail_backend)
    if failure_point in ("verifier", "foreign_verifier"):
        monkeypatch.setattr(cli, "_checked_run_backend", lambda *a, **k: (object(), object()))
        monkeypatch.setattr(cli, "verify_provider_run", fail_verifier)
    assert cli.main(args) == 1
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    assert not output.exists()
    assert provider_api.get_source_qualification_provider() is (
        foreign if failure_point.startswith("foreign_") else None)
    if provider_api.get_source_qualification_provider() is foreign:
        provider_api.clear_source_qualification_provider()


def test_runtime_unavailable_rejects_cpu_winner_before_cos(tmp_path, monkeypatch, capsys):
    _, _, _, payload = _inputs(tmp_path)
    args, output, _ = _prepare_run(tmp_path, monkeypatch, payload)
    monkeypatch.setattr(cli, "_runtime_ready", lambda: False)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("runtime gate read COS"))
    assert cli.main(args) == 1
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    assert not output.exists()


def test_cuda_resource_gate_rejects_before_cos(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    _, _, _, payload = _inputs(tmp_path)
    args, output, _ = _prepare_run(tmp_path, monkeypatch, payload)
    schema = cli.F61_ALL24_SCHEMA
    monkeypatch.setattr(cli, "_report_contract", lambda report: (
        schema, object(), SimpleNamespace(winning_backend="cuda")))
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: True)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("CUDA gate read COS"))
    assert cli.main(args) == 1
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    assert not output.exists()

def test_raw_source_dates_allow_label_horizon_trim_without_changing_domain(
        tmp_path, monkeypatch):
    # Real F61 source index has 2588 days; t+1/t+2 labels have 2586.
    # Rejecting the raw index against the final domain prevents every real run.
    args, output, _ = _prepare_run(tmp_path, monkeypatch, _payload())
    raw_dates = tuple(range(2588))
    assets, rows = tuple(range(5461)), tuple(range(61))
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (raw_dates, assets, rows))
    monkeypatch.setattr(cli.tiles, "load_labels",
                        lambda d, a, *args: (d[:-2], a, object()))
    runs = []
    def backend(*args, **kwargs):
        assert kwargs["dates"] == raw_dates[:-2]
        assert kwargs["source_rows"] is rows
        runs.append(kwargs)
        return object(), {}
    monkeypatch.setattr(cli, "_checked_run_backend", backend)
    monkeypatch.setattr(cli, "verify_provider_run", lambda *a, **kw: {"verified": True})
    assert cli.main(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["shape"] == [2586, 5461, 61]
    assert len(runs) == 2
    assert provider_api.get_source_qualification_provider() is None


@pytest.mark.parametrize("aligned_days", [2585, 2587])
def test_raw_source_superset_does_not_relax_final_label_domain(
        tmp_path, monkeypatch, capsys, aligned_days):
    args, output, _ = _prepare_run(tmp_path, monkeypatch, _payload())
    raw_dates = tuple(range(2588))
    assets, rows = tuple(range(5461)), tuple(range(61))
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (raw_dates, assets, rows))
    loaded = []
    def labels(d, a, *args):
        loaded.append(True)
        return d[:aligned_days], a, object()
    monkeypatch.setattr(cli.tiles, "load_labels", labels)
    monkeypatch.setattr(cli, "reference_source_all24",
                        lambda *a, **kw: pytest.fail("oracle on wrong label domain"))
    monkeypatch.setattr(cli, "_checked_run_backend",
                        lambda *a, **kw: pytest.fail("backend on wrong label domain"))
    assert cli.main(args) == 1
    assert loaded == [True]
    assert capsys.readouterr().err.strip() == "error_code=ValueError"
    assert not output.exists()
    assert provider_api.get_source_qualification_provider() is None


def test_raw_axis_beyond_registered_label_tail_stops_before_label_io(
        tmp_path, monkeypatch, capsys):
    args, output, _ = _prepare_run(tmp_path, monkeypatch, _payload())
    monkeypatch.setattr(cli.tiles, "read_axis_index",
                        lambda *a: (tuple(range(2589)), tuple(range(5461)), tuple(range(61))))
    monkeypatch.setattr(cli.tiles, "load_labels",
                        lambda *a: pytest.fail("labels read outside bounded raw domain"))
    assert cli.main(args) == 1
    assert capsys.readouterr().err.strip() == "error_code=ValueError"
    assert not output.exists()
