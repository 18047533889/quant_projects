"""Public F61 profiling safety contract; no external data in these tests."""
import json

import pytest

from quant_evaluator.scripts import benchmark_real_cos_f61_profile_abba as cli


@pytest.mark.parametrize("domain,kind,count", [
    ("all15", "real_cos_profile_abba_f61_all15.v1", 15),
    ("all24", "real_cos_profile_abba_f61_all24.v1", 24),
])
def test_dry_run_reports_exact_domain_without_external_reads(monkeypatch, capsys, domain, kind, count):
    # A default run must never begin COS reads or grant performance qualification.
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("dry run read COS"))
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **k: pytest.fail("dry run built source"))
    assert cli.main(["--metric-set", domain]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["kind"], report["status"], report["run_started"]) == (kind, "preflight_only", False)
    assert report["shape"] == [2586, 5461, 61]
    assert len(report["metric_ids"]) == count
    assert report["requested_tile_cap"] == 16
    assert "profile_records" not in report


def test_default_domain_is_all24(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    assert cli.main([]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "real_cos_profile_abba_f61_all24.v1"


def test_run_requires_explicit_paths_before_preflight(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("readiness before permission"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--run"])
    assert error.value.code == 2


@pytest.mark.parametrize("which", ["output", "progress"])
def test_existing_owned_output_is_not_overwritten(tmp_path, monkeypatch, which):
    axis, output = tmp_path / "axis.json", tmp_path / "report.json"
    axis.write_text("{}")
    target = output if which == "output" else tmp_path / "report.json.progress.json"
    target.write_text("owned-by-another-worker")
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("collision must precede reads"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--run", "--axis-index", str(axis), "--output", str(output)])
    assert error.value.code == 2
    assert target.read_text() == "owned-by-another-worker"


def test_rejected_resources_do_not_start_real_reads(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": False})
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: pytest.fail("insufficient memory read COS"))
    with pytest.raises(SystemExit, match="preflight"):
        cli.main([])


def test_unknown_metric_domain_is_not_silently_changed(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: pytest.fail("invalid request began preflight"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--metric-set", "arbitrary"])
    assert error.value.code == 2
