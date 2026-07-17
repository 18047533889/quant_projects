from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from platform_bootstrap import PROJECT_ROOT, activate_platform, platform_diagnostics


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = []
    for asset, values in {
        "A": [1.0, -2.0, 0.0, 4.0, -5.0, 6.0],
        "B": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }.items():
        for date, close in zip(dates, values):
            rows.append(
                {
                    "datetime": date,
                    "asset": asset,
                    "close": close,
                    "open": close,
                    "high": close,
                    "low": close,
                    "volume": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_embedded_platform_is_selected_by_default():
    layout = activate_platform(force=True)
    assert layout.mode == "bundled"
    assert layout.root == PROJECT_ROOT
    assert (PROJECT_ROOT / "factor_engine" / "runtime" / "engine.py").is_file()
    assert (PROJECT_ROOT / "data_access" / "store.py").is_file()


def test_runtime_modules_are_loaded_from_distribution():
    activate_platform(force=True)
    import data_access
    import runtime.engine

    assert Path(data_access.__file__).resolve().is_relative_to(PROJECT_ROOT)
    assert Path(runtime.engine.__file__).resolve().is_relative_to(PROJECT_ROOT)


def test_embedded_manifest_has_platform_provenance():
    manifest = json.loads(
        (PROJECT_ROOT / "embedded_platform_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_repository"] == "18047533889/quant_projects"
    assert manifest["source_commit"]
    assert manifest["modules"]["factor_engine"]["files"] > 100
    assert manifest["modules"]["data_access"]["files"] > 20
    assert len(manifest["modules"]["factor_engine"]["sha256"]) == 64
    assert "workspace_data" in manifest["exclusions"]["data_access"]
    assert not (PROJECT_ROOT / "data_access" / "workspace_data").exists()
    assert (
        PROJECT_ROOT
        / "factor_engine"
        / "cleaned_operators"
        / "common"
        / "daily_panel.py"
    ).is_file()


def test_standalone_diagnostics_pass():
    report = platform_diagnostics(import_runtime=True)
    assert report["status"] == "PASS", report


def test_manifest_remains_valid_after_runtime_imports_create_caches():
    activate_platform(force=True)
    import data_access  # noqa: F401
    import runtime.engine  # noqa: F401

    proc = subprocess.run(
        [sys.executable, "scripts/sync_embedded_platform.py", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_embedded_semantic_hardening_executes_signed_rolling_product():
    activate_platform(force=True)
    from integrations.quant_platform import execute_factor_on_frame

    execution = execute_factor_on_frame(
        "ts_product(close, 2)",
        _frame(),
        factor_name="signed_product",
    )
    a = execution.result.xs("A", level=1).to_numpy(dtype=float)
    assert np.isnan(a[0])
    np.testing.assert_allclose(a[1:], [-2.0, -0.0, 0.0, -20.0, -30.0])


def test_nested_lookback_uses_dependency_path():
    activate_platform(force=True)
    from integrations.quant_platform import validate_factor_formula

    _, _, analysis = validate_factor_formula("ts_mean(ts_delay(close, 5), 20)")
    assert analysis.lookback == 24


def test_latest_daily_panel_operator_executes_from_embedded_engine():
    activate_platform(force=True)
    from integrations.quant_platform import execute_factor_on_frame

    execution = execute_factor_on_frame(
        "ts_true_streak(close > 0)",
        _frame(),
        factor_name="positive_streak",
    )
    a = execution.result.xs("A", level=1).to_numpy(dtype=float)
    b = execution.result.xs("B", level=1).to_numpy(dtype=float)
    np.testing.assert_allclose(a, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(b, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
