from pathlib import Path
from unittest.mock import patch
import importlib.util

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_missing_profile_resource_fails_explicitly(tmp_path):
    from factor_engine.runtime.quality import dq_profiles
    dq_profiles._load_profiles_payload.cache_clear()
    try:
        with patch.object(dq_profiles, "files", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="Required packaged DQ"):
                dq_profiles.list_dq_profiles()
    finally:
        dq_profiles._load_profiles_payload.cache_clear()


def test_packaged_production_profile_is_resolved():
    from factor_engine.runtime.quality.dq_profiles import resolve_input_dq_thresholds
    assert resolve_input_dq_thresholds("us_equity_daily_prod").min_rows > 0


def test_cli_passes_canonical_installed_module_to_server():
    from factor_engine.service.app import main
    with patch("uvicorn.run") as run:
        assert main(["--host", "127.0.0.1", "--port", "8088"]) == 0
    assert run.call_args.args == ("factor_engine.service.app:create_app",)


@pytest.mark.parametrize("body,accepted", [
    ("", False), ('<testcase name="skipped"><skipped/></testcase>', False),
    ('<testcase name="fail"><failure/></testcase>', False),
    ('<testcase name="ok"/>', True),
])
def test_ci_cannot_certify_zero_tests_or_all_skipped(tmp_path, body, accepted):
    spec = importlib.util.spec_from_file_location("ci_execution_check", ROOT / "scripts/verify_junit_execution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    if accepted:
        assert module.verify(str(path))["passed"] == 1
    else:
        with pytest.raises(ValueError):
            module.verify(str(path))


def test_workflow_transfers_wheels_and_keeps_manual_gate_explicit():
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert any(s.get("uses") == "actions/upload-artifact@v4" for s in jobs["PACKAGE_BUILD"]["steps"])
    assert any(s.get("uses") == "actions/download-artifact@v4" for s in jobs["IMPORT"]["steps"])
    assert "inputs.run_100k" in jobs["manual_100k"]["if"]
    assert "FE_EXECUTE_PERSIST_SYNTHETIC" in jobs
    assert 'ZipFile("$whl")' not in str(workflow)
    assert 'python3 -m build --outdir' in str(jobs["PACKAGE_BUILD"])
    assert 'pip wheel --no-deps --no-build-isolation' not in str(jobs["PACKAGE_BUILD"])
