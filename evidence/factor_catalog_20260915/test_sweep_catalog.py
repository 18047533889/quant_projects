import argparse
import gzip
import json
from pathlib import Path
import sys
import pytest
import sweep_catalog

class FakeRunner:
    """Test cursor/ledger mechanics only; never supplies computed factor values."""
    def __init__(self, fail_call=None):
        self.calls = 0
        self.fail_call = fail_call
        self.last_argv = None
    def current_rss_mib(self):
        return 1
    def main(self):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("simulated process-local batch failure")
        argv = sys.argv
        self.last_argv = list(argv)
        get = lambda name: argv[argv.index(name) + 1]
        output = Path(get("--output"))
        with gzip.open(output, "wt") as stream:
            stream.write("{}\n")
        take, offset = int(get("--limit")), int(get("--offset"))
        summary = {"processed": take, "next_offset": offset + take + 2,
                   "counts": {"TEST_ROUTING_ONLY": take}, "seconds": 0}
        output.with_suffix("").with_suffix(".summary.json").write_text(json.dumps(summary))

def args():
    return argparse.Namespace(limit=4, chunk=2, offset=0, max_rss_mib=100,
                              rotate_rss_mib=90, timeout_seconds=1,
                              deadline_mode="external-watchdog", backend="pandas",
                              daily_only=True, output_prefix="test-sweep")

def test_warm_batches_follow_source_cursor_and_do_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_catalog, "__file__", str(tmp_path / "sweep_catalog.py"))
    old = sys.argv
    runner = FakeRunner()
    result = sweep_catalog.sweep(args(), runner=runner)
    assert runner.calls == 2
    assert result["next_offset"] == 8
    assert result["requested_count_completed"]
    assert sys.argv is old
    assert "--deadline-mode" in runner.last_argv
    assert runner.last_argv[runner.last_argv.index("--backend") + 1] == "pandas"
    assert (tmp_path / "test-sweep-000004.jsonl.gz").exists()
    with pytest.raises(FileExistsError):
        sweep_catalog.sweep(args(), runner=runner)

def test_failed_later_batch_preserves_closed_prior_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_catalog, "__file__", str(tmp_path / "sweep_catalog.py"))
    old = sys.argv
    with pytest.raises(RuntimeError, match="simulated"):
        sweep_catalog.sweep(args(), runner=FakeRunner(fail_call=2))
    assert sys.argv is old
    with gzip.open(tmp_path / "test-sweep-000000.jsonl.gz", "rt") as stream:
        assert stream.read() == "{}\n"
    lines = (tmp_path / "test-sweep.ledger.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["next_offset"] == 4
    assert not (tmp_path / "test-sweep-000004.jsonl.gz").exists()


def test_reviewed_input_path_is_forwarded_to_every_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_catalog, "__file__", str(tmp_path / "sweep_catalog.py"))
    config = args()
    config.input = str(tmp_path / "reviewed.input.jsonl.gz")

    class InputRunner(FakeRunner):
        def main(self):
            assert "--input" in sys.argv
            assert sys.argv[sys.argv.index("--input") + 1] == config.input
            super().main()

    result = sweep_catalog.sweep(config, runner=InputRunner())
    assert result["processed"] == 4
    start = json.loads((tmp_path / "test-sweep.ledger.jsonl").read_text().splitlines()[0])
    assert start["config"]["input"] == config.input
