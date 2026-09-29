"""Small non-COS tests for the real-COS factor-count preflight CLI."""
import json
import sys
from types import SimpleNamespace

import pytest

from quant_evaluator.scripts import benchmark_real_cos_factor_batch as benchmark


def test_preflight_only_exits_before_factor_load_or_backend_run(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("preflight-only must not load factor objects or run an evaluator")

    monkeypatch.setattr(benchmark, "preflight_factor_count_profile", lambda args: {
        "status": "ready", "pass": True, "verified_factor_count": 61,
        "selected_object_bytes": 1_946_203_073, "mem_available_bytes": 30_000_000_000,
        "gpu_memory": {"status": "ok", "devices": [{"total_mib": 80_000, "free_mib": 70_000}]},
        "estimated_peak_bytes": 50_688_000_000,
    })
    monkeypatch.setattr(benchmark, "load_real_batch", fail)
    monkeypatch.setattr(benchmark.routes, "run_one", fail)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_factor_batch.py", "--factor-count-profile", "32",
        "--preflight-only",
    ])

    benchmark.main()

    lines = capsys.readouterr().out.splitlines()
    result = json.loads(lines[0])
    assert result["preflight"]["verified_factor_count"] == 61
    assert result["preflight"]["selected_object_bytes"] == 1_946_203_073
    assert result["preflight"]["gpu_memory"]["status"] == "ok"


def test_f64_count_profile_reports_unavailable_without_loading(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("F64 unsupported path must not load or run")

    monkeypatch.setattr(benchmark, "load_real_batch", fail)
    monkeypatch.setattr(benchmark.routes, "run_one", fail)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_factor_batch.py", "--factor-count-profile", "64",
    ])

    benchmark.main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unavailable"
    assert result["factor_count"] == 64
    assert result["loader_max_factor_count"] == 32


def test_preflight_requires_14_gib_effective_vram_at_conservative_40_percent():
    assert benchmark._gpu_memory_pass({
        "status": "ok", "devices": [{"free_mib": 35_840}],
    })
    assert not benchmark._gpu_memory_pass({
        "status": "ok", "devices": [{"free_mib": 35_839}],
    })
    assert not benchmark._gpu_memory_pass({
        "status": "unavailable", "devices": [{"free_mib": 80_000}],
    })


def test_f32_working_set_estimate_is_manifest_bound():
    from quant_evaluator.scripts import benchmark_real_cos_metric_batch as full

    assert benchmark._estimate_f32_peak_bytes(full.MANIFEST_SHA256) == 50 * 1024**3
    array_bound = 3000 * 5500 * 32 * 8
    scaled_f24 = (46 * 1024**3 * 32 + 23) // 24
    conservative = max(array_bound * 16, scaled_f24)
    assert benchmark._estimate_f32_peak_bytes() == conservative
    assert benchmark._estimate_f32_peak_bytes("0" * 64) == conservative


def _stub_preflight_data_access(monkeypatch, read_declared):
    import data_access.core.engine as engine_module
    import data_access.cos.research as research_module
    import data_access.registry.loader as registry_module
    import data_access.store as store_module
    import quant_evaluator.scripts.load_real_cos_factor_batch as loader_module

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class FakeStore:
        def __init__(self, registry, engine):
            self.authorized = []

        def authorize_dataset(self, dataset):
            self.authorized.append(dataset)

        def _authorize_factor_params(self, dataset, params):
            assert params is None

    monkeypatch.setattr(engine_module, "DuckDBEngine", FakeEngine)
    monkeypatch.setattr(registry_module, "DatasetRegistry", lambda datasets: datasets)
    monkeypatch.setattr(store_module, "DataAccessStore", FakeStore)
    monkeypatch.setattr(research_module, "read_declared_cos_object", read_declared)
    monkeypatch.setattr(loader_module, "_ds", lambda name, *args: SimpleNamespace(name=name))
    monkeypatch.setattr(benchmark, "_estimate_f32_peak_bytes", lambda _sha: 1024**3)
    monkeypatch.setattr(benchmark, "_mem_available_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(benchmark, "_gpu_memory_snapshot", lambda: {
        "status": "ok", "devices": [{"free_mib": 35_840}],
    })
    monkeypatch.setattr(benchmark, "_gpu_memory_pass", lambda _snapshot: True)


def test_f32_preflight_uses_authorized_data_access_read_without_s3_creds(monkeypatch):
    from data_access.cos import remote

    calls = []

    def read_manifest(store, dataset, **kwargs):
        assert store.authorized == [dataset]
        calls.append((dataset, kwargs))
        factors = {}
        for i in range(1, 33):
            factor_id = f"factor_{i}"
            digest = f"{i:064x}"
            factors[factor_id] = {
                "verified": True,
                "status": "evaluated_optimization_pending",
                "bytes": 1024,
                "sha256": digest,
                "uri": (
                    "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/"
                    f"factor_values/{digest}/{factor_id}.parquet"
                ),
            }
        return SimpleNamespace(
            content_sha256=benchmark.full.MANIFEST_SHA256,
            table=SimpleNamespace(to_pylist=lambda: [{"factors": factors}]),
        )

    _stub_preflight_data_access(monkeypatch, read_manifest)
    monkeypatch.setattr(remote, "resolve_s3_credentials", lambda: pytest.fail(
        "the DataAccess read path must not require S3 credentials"))
    args = SimpleNamespace(
        manifest_sha256=benchmark.full.MANIFEST_SHA256,
        profile_max_object_mib=128, profile_max_total_mib=2048,
        max_working_gib=50,
    )

    result = benchmark.preflight_factor_count_profile(args)

    assert result["status"] == "ready"
    assert result["verified_factor_count"] == 32
    assert result["gpu_memory_pass"] is True
    assert calls == [("source_manifest", {"allow_research": True})]


def test_f32_preflight_fails_closed_when_data_access_read_is_unavailable(monkeypatch):
    from data_access.core.exceptions import ValidationError

    def reject_read(*args, **kwargs):
        raise ValidationError("COS metadata unavailable")

    _stub_preflight_data_access(monkeypatch, reject_read)
    args = SimpleNamespace(
        manifest_sha256=benchmark.full.MANIFEST_SHA256,
        profile_max_object_mib=128, profile_max_total_mib=2048,
        max_working_gib=50,
    )

    result = benchmark.preflight_factor_count_profile(args)

    assert result["status"] == "unavailable"
    assert result["pass"] is False
    assert "COS metadata unavailable" in result["reason"]


def test_f32_preflight_does_not_reclassify_data_access_authorization_denial(monkeypatch):
    from data_access.core.exceptions import AccessDeniedError

    def reject_read(*args, **kwargs):
        raise AccessDeniedError("dataset read denied")

    _stub_preflight_data_access(monkeypatch, reject_read)
    args = SimpleNamespace(
        manifest_sha256=benchmark.full.MANIFEST_SHA256,
        profile_max_object_mib=128, profile_max_total_mib=2048,
        max_working_gib=50,
    )

    with pytest.raises(AccessDeniedError):
        benchmark.preflight_factor_count_profile(args)


def test_f32_whole_batch_uses_complete_metric_batch_worker(monkeypatch, capsys):
    from types import SimpleNamespace

    import numpy as np

    calls = []
    batch = SimpleNamespace(values=np.zeros((2, 3, 32), dtype=np.float64))
    labels = object()
    provenance = {"sources": [{"manifest_sha256": "f" * 64}]}
    args = SimpleNamespace(
        repeats=1, timeout_s=5, factor_profile_mode="whole_batch",
        manifest_sha256="f" * 64,
        profile_max_object_mib=128, profile_max_total_mib=2048,
        max_working_gib=80, output=None,
    )

    def fake_run(ctx, backend, repeats, timeout_s):
        calls.append((backend, repeats, timeout_s))
        return {
            "backend_requested": backend,
            "backend_used": "cpu" if backend == "cpu" else "cuda",
            "peak_vram": 1 if backend != "cpu" else None,
            "cold_s": 0.1, "warm_median_s": 0.1, "peak_rss_kib": 100,
            "config_hash": "same", "artifacts": {},
        }

    monkeypatch.setattr(benchmark.full, "_run_one", fake_run)
    monkeypatch.setattr(benchmark.full, "_compare", lambda reference, run: {"pass": True})
    monkeypatch.setattr(benchmark.full, "_BATCH", None)
    monkeypatch.setattr(benchmark.full, "_LABELS", None)
    monkeypatch.setattr(benchmark.full, "METRICS", ())

    benchmark.run_factor_count_profile(None, batch, labels, provenance, args, 0.0)

    assert benchmark.full._BATCH is batch
    assert benchmark.full._LABELS is labels
    assert benchmark.full.METRICS == benchmark.DEFAULT_METRICS
    assert [call[0] for call in calls] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu",
    ]
    assert all(call[1] == 2 for call in calls)
    assert '"profile_mode": "whole_batch"' in capsys.readouterr().out


def test_f8_extended_trial_honors_configured_timeout(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np

    calls = []
    batch = SimpleNamespace(values=np.zeros((2, 3, 8), dtype=np.float64))
    output = tmp_path / "extended.json"
    args = SimpleNamespace(
        factors=8, days=0, assets=5500,
        manifest_sha256=benchmark.full.MANIFEST_SHA256,
        max_object_mib=64, max_total_mib=256,
        repeats=1, timeout_s=321, output=output,
    )

    def fake_run(ctx, backend, repeats, timeout_s):
        calls.append((backend, repeats, timeout_s))
        return {
            "backend_requested": backend,
            "backend_used": "cpu" if backend == "cpu" else "cuda",
            "peak_vram": None, "cold_s": 0.1, "warm_median_s": 0.1,
            "peak_rss_kib": 100, "config_hash": "same", "artifacts": {},
        }

    monkeypatch.setattr(benchmark.full, "_run_one", fake_run)
    monkeypatch.setattr(benchmark.full, "_compare", lambda *_: {"pass": True})
    benchmark.run_extended(
        None, ("rank_ic_series",), batch, object(), {}, args, 0.0)

    assert [backend for backend, _, _ in calls] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu",
    ]
    assert all(repeats == 2 and timeout_s == 321
               for _, repeats, timeout_s in calls)
    report = json.loads(output.read_text())
    assert report["request"]["timeout_s"] == 321
    assert report["pass"] is True


def test_f32_preflight_classifies_metadata_failure_without_s3_credential_probe(monkeypatch):
    from data_access.core.exceptions import ValidationError
    from data_access.store import DataAccessStore
    from data_access.cos import remote, research

    monkeypatch.setattr(DataAccessStore, "authorize_dataset", lambda *_: None)
    monkeypatch.setattr(DataAccessStore, "_authorize_factor_params", lambda *_: None)
    calls = []

    def metadata_probe(*_args, **_kwargs):
        calls.append("metadata")
        raise ValidationError("exact object metadata is missing or mismatched")

    monkeypatch.setattr(remote, "resolve_s3_credentials", lambda: pytest.fail(
        "metadata failure classification must not probe S3 credentials"))
    monkeypatch.setattr(research, "read_declared_cos_object", metadata_probe)
    result = benchmark.preflight_factor_count_profile(SimpleNamespace(manifest_sha256=None))

    assert result["status"] == "unavailable"
    assert result["category"] == "object_metadata_unavailable"
    assert calls == ["metadata"]
