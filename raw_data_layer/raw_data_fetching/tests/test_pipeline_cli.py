from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def test_pipeline_cli_validate_parquet(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00"]),
            "price": [10.0, 10.5],
            "size": [100.0, 200.0],
        }
    )
    target = tmp_path / "2024-01-02.parquet"
    frame.to_parquet(target, index=False)

    script_path = Path(__file__).resolve().parents[1] / "run_pipeline.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), "validate-parquet", str(tmp_path), "--workers", "1"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )

    payload = json.loads(completed.stdout)
    assert payload["command"] == "validate-parquet"
    assert payload["directory"] == str(tmp_path.resolve())
    assert payload["total_files"] == 1
    assert payload["healthy_files"] == 1