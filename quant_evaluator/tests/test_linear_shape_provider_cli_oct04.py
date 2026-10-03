"""Public CLI contract for fresh-provider verification of fixed F48 reports."""
from __future__ import annotations

import json
import sys
from types import ModuleType
from quant_evaluator.runtime.source_profile_report_schema import LINEAR_SHAPE_METRICS, LINEAR_SHAPE_REPORT_KIND

import pytest


def test_validated_cache_presence_query_observes_empty_live_and_expired_states(monkeypatch):
    """The preflight predicate must not prune an entry merely because it expired."""
    import time
    from types import SimpleNamespace
    from test_source_linear_shape_report_reader_oct04 import _payload
    from quant_evaluator.runtime import source_qualification_cache as cache_api
    from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report

    assert cache_api.has_validated_records() is False
    report = parse_source_profile_report(json.dumps(_payload()))
    cache_key = "test-validated-cache-presence-query"
    cache_api.discard_validated_records(cache_key)
    cache_api.remember_validated_records(cache_key, report.records)
    assert cache_api.has_validated_records() is True
    original_time = cache_api.time
    future = time.monotonic() + cache_api._TTL_SECONDS + 1
    monkeypatch.setattr(cache_api, "time", SimpleNamespace(monotonic=lambda: future))
    try:
        assert cache_api.has_validated_records() is False
    finally:
        monkeypatch.setattr(cache_api, "time", original_time)
    assert cache_api.get_validated_records(cache_key) is report.records
    cache_api.discard_validated_records(cache_key)


def test_run_rejects_live_validated_cache_before_cos_or_runtime(tmp_path, monkeypatch, capsys):
    """A fresh-provider claim cannot be masked by cached qualification records."""
    import json as json_module
    from test_source_linear_shape_report_reader_oct04 import _payload
    from quant_evaluator.runtime import source_qualification_cache as cache_api
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli

    axis, profile, output = (tmp_path / "axis.json", tmp_path / "profile.json", tmp_path / "result.json")
    axis.write_text("{}", encoding="utf-8")
    profile_payload = _payload()
    profile.write_text(json_module.dumps(profile_payload), encoding="utf-8")
    report = cli.load_source_profile_report(profile)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", report.manifest_sha256)
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})

    cache_key = "test-fresh-linear-shape-provider-preflight"
    cache_api.discard_validated_records(cache_key)
    cache_api.remember_validated_records(cache_key, report.records)
    assert cache_api.get_validated_records(cache_key) is report.records
    manifest_reads = []
    monkeypatch.setattr(cli.tiles, "read_manifest",
        lambda *args: (manifest_reads.append(args), object())[1])
    monkeypatch.setattr(cli.source_batch, "_make_cos_source",
        lambda *args, **kwargs: pytest.fail("cache rejection occurred after COS construction"))
    monkeypatch.setattr(cli, "_runtime_ready",
        lambda: pytest.fail("cache rejection occurred after runtime initialization"))

    try:
        result = cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
            "--output", str(output)])
        assert result == 1
        assert manifest_reads == []
        assert cache_api.get_validated_records(cache_key) is report.records
        assert not output.exists()
        assert capsys.readouterr().err.strip() == "error_code=RuntimeError"
    finally:
        cache_api.discard_validated_records(cache_key)


def test_default_cli_is_preflight_only(monkeypatch, capsys):
    """A CLI that performs runtime/COS work without --run violates the dry-run boundary."""
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli

    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True, "ram": "ok"})
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("dry-run read manifest"))
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: pytest.fail("dry-run made COS source"))
    monkeypatch.setattr(cli, "_runtime_ready", lambda: pytest.fail("dry-run initialized runtime"))
    assert cli.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "real_cos_f48_linear_shape_fresh_provider.v1"
    assert report["status"] == "preflight_only" and report["run_started"] is False
    assert report["shape"] == [2586, 5461, 48]


def test_run_requires_existing_profile_and_axis_and_new_output(tmp_path, monkeypatch):
    """Missing provenance inputs must be rejected before preflight or external reads."""
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli

    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("invalid CLI reached preflight"))
    with pytest.raises(SystemExit) as missing:
        cli.main(["--run", "--axis-index", str(tmp_path / "axis"), "--output", str(tmp_path / "result")])
    assert missing.value.code == 2

def test_preexisting_provider_is_preserved_and_fails_closed(tmp_path, monkeypatch, capsys):
    """The CLI must not override a process provider installed by its caller."""
    from quant_evaluator.runtime import source_qualification_provider as provider_api
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli
    axis, profile, output = (tmp_path / "axis", tmp_path / "profile", tmp_path / "output")
    axis.write_text("{}", encoding="utf-8")
    profile.write_text("opaque", encoding="utf-8")
    existing = provider_api.FileSourceQualificationProvider({"b" * 64: profile})
    monkeypatch.setattr(provider_api, "_configured_provider", existing)
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: pytest.fail("COS before refusal"))
    args = ["--run", "--axis-index", str(axis), "--profile-report", str(profile), "--output", str(output)]
    assert cli.main(args) == 1
    assert provider_api.get_source_qualification_provider() is existing
    assert not output.exists()
    assert capsys.readouterr().err.strip() == "error_code=RuntimeError"

def test_module_entrypoint_exposes_public_flags():
    """The documented python -m invocation must execute the actual CLI parser."""
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-m",
        "quant_evaluator.scripts.verify_real_cos_linear_shape_provider", "--help"],
        cwd=repo, text=True, capture_output=True, check=False)
    assert proc.returncode == 0
    assert "--profile-report" in proc.stdout and "--axis-index" in proc.stdout

def test_successful_run_uses_real_profile_reader_and_provider_lifecycle(tmp_path, monkeypatch):
    """A live CLI run must parse the report, install its real file provider, and write its receipt."""
    import json as json_module
    from test_source_linear_shape_report_reader_oct04 import _payload
    from quant_evaluator.runtime import source_qualification_provider as provider_api
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli

    axis, profile, output = (tmp_path / "axis.json", tmp_path / "profile.json", tmp_path / "result.json")
    axis.write_text("{}", encoding="utf-8")
    profile_payload = _payload()
    profile.write_text(json_module.dumps(profile_payload), encoding="utf-8")
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", profile_payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True, "test_gate": True})
    monkeypatch.setattr(cli, "_auto_batch_cuda_rejection", lambda *a: pytest.fail("CPU winner should not request VRAM gate"))
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    warm = []
    monkeypatch.setattr(cli, "_warm", lambda policy, backend: warm.append(backend))
    factor_records = tuple(f"factor-{i}" for i in range(48))
    dates, assets, source_rows = tuple(range(cli.SHAPE[0])), tuple(range(cli.SHAPE[1])), tuple(range(48))
    labels = object()
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda _: object())
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *a: factor_records)
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (dates, assets, source_rows))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda d, a, *args: (d, a, labels))

    closed = []
    class OracleSource:
        def close(self):
            closed.append(True)
    oracle_source, oracle_bundle = OracleSource(), object()
    def make_source(*args, **kwargs):
        assert args[0:3] == (factor_records, source_rows, dates)
        assert kwargs == {"prefetch": "auto", "max_source_memory_mib": 4096,
            "prefetch_workers": 2, "max_prefetch_memory_mib": 512}
        return oracle_source
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", make_source)
    oracle_module = ModuleType("quant_evaluator.scripts.source_linear_shape_oracle")
    oracle_module.reference_source_linear_shape = lambda source, got_labels, **kwargs: (
        oracle_bundle if source is oracle_source and got_labels is labels
        and kwargs == {"max_tile_size": 1, "max_result_bytes": 64 * 1024**2}
        else pytest.fail("unexpected oracle inputs"))
    monkeypatch.setitem(sys.modules, "quant_evaluator.scripts.source_linear_shape_oracle", oracle_module)

    run_calls = []
    def run_backend(backend, **kwargs):
        assert backend == "auto" and kwargs["expected_auto_cuda"] is False
        assert "source_qualification" not in kwargs
        assert kwargs["context_observer"] is cli.live_source_profile_context_observer
        lookup = provider_api.lookup_source_qualification_candidate(
            _payload()["profile_records"][0]["context"]["request_content_sha256"])
        assert lookup.reason_code == "candidate_found"
        assert kwargs["source_adapter"] == "cos" and kwargs["cos_prefetch"] == "auto"
        assert kwargs["tile_size"] == 16 and kwargs["max_object_mib"] == 128
        assert kwargs["max_source_memory_mib"] == 4096
        assert kwargs["cos_prefetch_workers"] == 2 and kwargs["max_prefetch_memory_mib"] == 512
        run_calls.append(lookup.candidate.candidate_id)
        return object(), {"context_before": "fixture", "context_after": "fixture"}
    monkeypatch.setattr(cli, "_checked_run_backend", run_backend)
    verification_calls = []
    verification_module = ModuleType("quant_evaluator.scripts.linear_shape_provider_verification")
    def verify(bundle, receipt, report, oracle, *, run_index, require_cache_hit):
        assert report.kind == LINEAR_SHAPE_REPORT_KIND and oracle is oracle_bundle
        assert require_cache_hit is (run_index == 1)
        verification_calls.append(run_index)
        return {"verified": True, "cache_hit_required": require_cache_hit}
    verification_module.verify_provider_run = verify
    monkeypatch.setitem(sys.modules,
        "quant_evaluator.scripts.linear_shape_provider_verification", verification_module)

    assert cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
        "--output", str(output)]) == 0
    assert len(run_calls) == 2 and run_calls[0] == run_calls[1]
    assert verification_calls == [0, 1] and warm == ["cpu"] and closed == [True]
    assert provider_api.get_source_qualification_provider() is None
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["kind"] == "real_cos_f48_linear_shape_fresh_provider.v1"
    assert receipt["status"] == "complete" and receipt["run_started"] is True
    assert receipt["shape"] == [2586, 5461, 48] and receipt["metric_ids"] == list(LINEAR_SHAPE_METRICS)
    assert [run["run_index"] for run in receipt["runs"]] == [0, 1]
    assert str(profile) not in output.read_text(encoding="utf-8")
    assert receipt["manifest_sha256"] == profile_payload["manifest_sha256"]

def test_run_stage_system_exit_is_sanitized_without_leaking_external_text(tmp_path, monkeypatch, capsys):
    """A lower runner's SystemExit must not expose paths or credential-like diagnostics."""
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli
    axis, profile, output = (tmp_path / "axis", tmp_path / "profile", tmp_path / "output")
    axis.write_text("{}", encoding="utf-8")
    profile.write_text("opaque", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight",
        lambda: (_ for _ in ()).throw(SystemExit("secret-token /private/report.json")))
    result = cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
        "--output", str(output)])
    assert result == 1
    diagnostic = capsys.readouterr().err
    assert diagnostic.strip() == "error_code=SystemExit"
    assert "secret-token" not in diagnostic and "/private/report.json" not in diagnostic
    assert not output.exists()

def test_profile_manifest_mismatch_is_rejected_before_manifest_or_cos_reads(tmp_path, monkeypatch, capsys):
    """A profile from another source manifest must be rejected before source selection."""
    from types import SimpleNamespace
    from quant_evaluator.scripts import verify_real_cos_linear_shape_provider as cli
    axis, profile, output = (tmp_path / "axis", tmp_path / "profile", tmp_path / "output")
    axis.write_text("{}", encoding="utf-8")
    profile.write_text("caller profile", encoding="utf-8")
    context = SimpleNamespace(request_content_sha256="a" * 64,
        request_shape=cli.SHAPE, metric_ids=cli.METRICS)
    records = (SimpleNamespace(context=context), SimpleNamespace(context=context))
    report = SimpleNamespace(kind=LINEAR_SHAPE_REPORT_KIND, status="complete",
        manifest_sha256="b" * 64, records=records,
        qualification=SimpleNamespace(winning_backend="cpu"))
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "load_source_profile_report", lambda _: report)
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", "c" * 64)
    manifest_reads = []
    monkeypatch.setattr(cli.tiles, "read_manifest",
        lambda sha: (manifest_reads.append(sha), object())[1])
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *args: tuple(range(48)))
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *args: ((0,), (0,), tuple(range(48))))
    result = cli.main(["--run", "--axis-index", str(axis), "--profile-report", str(profile),
        "--output", str(output)])
    assert result == 1 and manifest_reads == []
    assert capsys.readouterr().err.strip() == "error_code=ValueError"
    assert not output.exists()
