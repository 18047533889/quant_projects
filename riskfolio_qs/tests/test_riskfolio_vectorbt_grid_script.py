from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_generic_optimizer_parallel_resume_preserves_order_and_ranking(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "riskfolio_vectorbt_grid.py"
    data_root = tmp_path / "data"
    barra_root = tmp_path / "barra"
    data_root.mkdir()
    barra_root.mkdir()

    base_config = tmp_path / "base.yaml"
    _write_yaml(
        base_config,
        {
            "config_version": 1,
            "adapter": {
                "type": "auto",
            },
            "inputs": {
                "alpha": {
                    "path": str(tmp_path / "unused-alpha.parquet"),
                    "layout": "wide",
                    "date_column": "date",
                }
            },
            "run": {
                "optimizer_name": "meanvar_absolute_return",
                "runtime_overrides": {},
            },
            "output": {"directory": "unused", "overwrite": False},
        },
    )

    output_root = tmp_path / "output"
    stats_root = output_root / "trial_stats"
    stats_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trial_id": "trial_0001",
                "risk_weight": 3.0,
                "pipeline_status": "success",
                "Information Ratio": 0.5,
            }
        ]
    ).to_csv(stats_root / "trial_0001.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial_id": "trial_0002",
                "risk_weight": 10.0,
                "pipeline_status": "success",
                "Information Ratio": 1.5,
            }
        ]
    ).to_csv(stats_root / "trial_0002.csv", index=False)

    pipeline_config = tmp_path / "pipeline.yaml"
    _write_yaml(
        pipeline_config,
        {
            "config_version": 1,
            "riskfolio": {
                "base_config": str(base_config),
                "fixed_overrides": {
                    "alpha_weight": 1.0,
                    "single_name_max": 0.05,
                },
                "grid": {"risk_weight": [3.0, 10.0]},
            },
            "backtest": {
                "profile": "standard_accurate_benchmark_v1",
                "data_root": str(data_root),
            },
            "selection": {
                "metric": "Information Ratio",
                "ascending": False,
            },
            "execution": {
                "output_directory": str(output_root),
                "max_trials": 2,
                "workers": 2,
                "resume": True,
                "overwrite": False,
                "on_trial_error": "raise",
            },
            "reports": {
                "enabled": False,
                "top_n": 0,
                "on_error": "raise",
            },
        },
    )

    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, str(script), "-c", str(pipeline_config)],
        cwd=project_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    search = pd.read_csv(output_root / "search_results.csv")
    ranked = pd.read_csv(output_root / "ranked_results.csv")
    assert search["trial_id"].tolist() == ["trial_0001", "trial_0002"]
    assert ranked["trial_id"].tolist() == ["trial_0002", "trial_0001"]
