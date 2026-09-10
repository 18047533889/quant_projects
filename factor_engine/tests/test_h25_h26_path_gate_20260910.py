from __future__ import annotations

import shutil
import subprocess
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "gate-root"
    (root / "scripts").mkdir(parents=True)
    for name in (
        "factor_engine", "data_access", "factor_preprocess", "factor_optimizer",
        "quant_evaluator", "factor_assets", ".github",
    ):
        (root / name).mkdir()
    shutil.copy2(REPO / "scripts" / "forbidden_path_reference_gate.sh", root / "scripts")
    (root / ".pre-commit-config.yaml").write_text("repos: []\n")
    (root / "scripts" / "audit_legacy_branch_content.py").write_text(
        'LEGACY_READ_ONLY_PATH_PREFIXES = {"dataaccess/": "data_access/"}  # PATH_GATE_HISTORICAL_MAPPING\n'
    )
    return root


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "forbidden_path_reference_gate.sh")],
        cwd=root, text=True, capture_output=True, check=False,
    )


def test_h26_exact_historical_mapping_is_the_only_positive_exception(tmp_path: Path) -> None:
    result = _run(_sandbox(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    spec = importlib.util.spec_from_file_location(
        "h26_history_audit", REPO / "scripts" / "audit_legacy_branch_content.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.current_path_candidates("dataaccess/read/scan_handle.py")[-1] == (
        "data_access/read/scan_handle.py"
    )


def test_h26_runtime_import_dynamic_import_and_config_paths_are_rejected(tmp_path: Path) -> None:
    probes = {
        "runtime.py": "import dataaccess\n",
        "dynamic.py": 'import importlib\nimportlib.import_module("dataaccess")\n',
        "runtime.json": '{"package": "dataaccess/read"}\n',
    }
    for filename, content in probes.items():
        root = _sandbox(tmp_path / filename.replace(".", "_"))
        (root / "factor_engine" / filename).write_text(content)
        result = _run(root)
        assert result.returncode == 1, (filename, result.stdout, result.stderr)


def test_h25_merge_residue_is_rejected_in_runtime_and_evidence(tmp_path: Path) -> None:
    for relative in (Path("factor_engine/kernel.py.orig"), Path("factor_engine/tests/evidence/sample.py.rej")):
        root = _sandbox(tmp_path / relative.name)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("residue\n")
        result = _run(root)
        assert result.returncode == 1, (relative, result.stdout, result.stderr)
        assert "MERGE RESIDUAL" in result.stdout
