import json
import os
from pathlib import Path
import subprocess
import sys


def test_ledger_distinguishes_reference_rejection_skip_and_xfail(tmp_path):
    fixture = tmp_path / "test_outcomes.py"
    fixture.write_text('''import pytest
@pytest.mark.v8_evidence("math_reference", reference_id="independent-fixture")
def test_math(): assert 2 + 2 == 4
@pytest.mark.v8_evidence("rejection")
def test_rejection():
    with pytest.raises(ValueError): raise ValueError("unsupported")
@pytest.mark.skip(reason="optional fixture")
def test_optional(): pass
@pytest.mark.xfail(strict=True, reason="unimplemented fixture")
def test_pending(): assert False
''')
    report = tmp_path / "ledger.jsonl"
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), PYTEST_ADDOPTS="")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p",
         "factor_engine.scripts.v8_pytest_evidence", f"--v8-report={report}", str(fixture)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert sum(row["reference_assertion_passed"] for row in records) == 1
    assert any(row["category"] == "rejection" and row["outcome"] == "passed"
               and row["phase"] == "call" for row in records)
    assert any(row["wasxfail"] for row in records)
    assert any(row["outcome"] == "skipped" and not row["wasxfail"] for row in records)
    assert records[-1]["record_type"] == "session_finish"
    assert records[-1]["successful"]
    assert records[-1]["completed_nodes"] == 4
    assert any(row.get("source_files") for row in records)


def test_failed_teardown_cannot_certify_reference(tmp_path):
    fixture = tmp_path / "test_teardown.py"
    fixture.write_text('''import pytest
@pytest.fixture
def broken():
    yield
    raise RuntimeError("teardown failed")
@pytest.mark.v8_evidence("math_reference", reference_id="teardown-oracle")
def test_math(broken): assert 2 + 2 == 4
''')
    report = tmp_path / "ledger.jsonl"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p",
         "factor_engine.scripts.v8_pytest_evidence", f"--v8-report={report}", str(fixture)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(root), PYTEST_ADDOPTS=""),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert not any(row["reference_assertion_passed"] for row in records)
    assert records[-1]["exitstatus"] == 1
    assert not records[-1]["successful"]


def test_abrupt_exit_has_no_completion_footer(tmp_path):
    fixture = tmp_path / "test_interrupted.py"
    fixture.write_text('''import os
def test_first(): assert True
def test_exit(): os._exit(9)
''')
    report = tmp_path / "ledger.jsonl"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p",
         "factor_engine.scripts.v8_pytest_evidence", f"--v8-report={report}", str(fixture)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(root), PYTEST_ADDOPTS=""),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 9
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert any(row["record_type"] == "node_final" for row in records)
    assert not any(row["record_type"] == "session_finish" for row in records)
