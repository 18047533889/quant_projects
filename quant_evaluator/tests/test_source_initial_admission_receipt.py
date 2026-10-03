"""Rejected source requests must not be recorded as performance results."""
import json
import sys
import pytest
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


@pytest.mark.parametrize("output_enabled", [True, False])
def test_initial_admission_rejection_never_reads_or_evaluates(monkeypatch, tmp_path, output_enabled):
    output = tmp_path / "admission.json"
    argv = ["benchmark", "--factors", "48", "--source-adapter", "cos", "--tile-size", "2"]
    if output_enabled:
        argv.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    gate = {"pass": False, "available_ram_bytes": 1,
            "minimum_available_ram_bytes": 32 * 1024**3,
            "cos_cache_disk_free_bytes": 0, "required_disk_bytes": 1024}
    monkeypatch.setattr(harness, "preflight", lambda *args: gate)
    monkeypatch.setattr(harness.tiles, "read_manifest",
                        lambda *_args: pytest.fail("must not read manifest"))
    monkeypatch.setattr(harness, "run_backend",
                        lambda *_args, **_kwargs: pytest.fail("must not evaluate"))
    with pytest.raises(SystemExit, match="insufficient RAM or COS cache disk headroom"):
        harness.main()
    assert output.exists() is output_enabled
    if output_enabled:
        report = json.loads(output.read_text())
        assert report["status"] == "preflight_rejected"
        assert report["pass"] is False and report["evaluation_started"] is False
        assert report["factor_objects_read"] == 0
        assert report["requested_factor_count"] == 48
        assert report["preflight"] == gate
        assert "seconds" not in report and "runs" not in report
