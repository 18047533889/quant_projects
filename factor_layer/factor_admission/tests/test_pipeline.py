from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_layer.factor_admission.pipeline import run_config_directory, run_from_config
from factor_layer.factor_evaluation.config_runner import run_from_config as run_evaluation_from_config


def _write_factor(lake_root: Path, factor_id: str, rows: list[tuple[str, str, float]]) -> None:
    frame = pd.DataFrame(rows, columns=["datetime", "asset", "value"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    year = int(frame["datetime"].dt.year.iloc[0])
    target = lake_root / "factors" / factor_id / f"year={year}"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "data.parquet", index=False)


def _write_market(path: Path, rows: list[tuple[str, str, float]]) -> None:
    frame = pd.DataFrame(rows, columns=["timestamp", "symbol", "open"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _register_factor(lake_root: Path, factor_id: str) -> None:
    db_path = lake_root / "_catalog.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS factor_registry (
                factor_id   TEXT PRIMARY KEY,
                author      TEXT NOT NULL,
                frequency   TEXT NOT NULL,
                description TEXT,
                ast_hash    TEXT NOT NULL,
                expression  TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS factor_watermark (
                factor_id    TEXT PRIMARY KEY,
                start_date   TEXT NOT NULL,
                end_date     TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                row_count    INTEGER,
                FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id)
            );
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO factor_registry (factor_id, author, frequency, description, ast_hash, expression, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                factor_id,
                "tester",
                "1d",
                None,
                "hash_v1",
                None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def _bootstrap_evaluation_run(tmp_path: Path, factor_id: str) -> tuple[Path, str, Path]:
    lake_root = tmp_path / "factor_lake"
    evaluation_root = tmp_path / "evaluations"
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    factor_rows: list[tuple[str, str, float]] = []
    market_rows: list[tuple[str, str, float]] = []
    daily_gross = {"AAA": 1.01, "BBB": 1.02, "CCC": 1.03, "DDD": 1.04}
    for date in dates:
        for rank, symbol in enumerate(symbols, start=1):
            factor_rows.append((str(date.date()), symbol, float(rank)))
    for symbol in symbols:
        price = 100.0
        for date in dates:
            market_rows.append((str(date.date()), symbol, price))
            price *= daily_gross[symbol]

    _write_factor(lake_root, factor_id, factor_rows)
    market_path = tmp_path / "market" / "daily_market.parquet"
    _write_market(market_path, market_rows)
    _register_factor(lake_root, factor_id)

    config_path = tmp_path / "factor_evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "factor_id": factor_id,
                    "run_name": "for_admission",
                    "primary_horizon": 1,
                },
                "source": {
                    "factor_lake_root": str(lake_root),
                    "market_data_path": str(market_path),
                    "market_price_col": "open",
                },
                "run": {
                    "horizons": [1, 2],
                    "n_quantiles": 4,
                    "min_assets_per_date": 4,
                },
                "output": {
                    "root": str(evaluation_root),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = run_evaluation_from_config(config_path)
    return lake_root, str(result["meta"]["run_id"]), evaluation_root


def _write_admission_config(
    path: Path,
    *,
    factor_id: str,
    run_id: str,
    lake_root: Path,
    evaluation_root: Path,
    approve: bool | None = None,
) -> None:
    decision_payload: dict[str, object]
    if approve is None:
        decision_payload = {
            "mode": "rule_based",
            "decided_by": "system",
            "policy_name": "smoke_policy",
            "primary_horizon": 1,
            "thresholds": {
                "min_rank_ic_mean": 0.9,
                "min_long_short_sharpe": 0.5,
            },
        }
    else:
        decision_payload = {
            "mode": "manual",
            "approve": approve,
            "reason": "manual override" if approve is False else None,
        }

    path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "factor_id": factor_id,
                    "run_id": run_id,
                },
                "source": {
                    "factor_lake_root": str(lake_root),
                    "evaluation_root": str(evaluation_root),
                },
                "decision": decision_payload,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_pipeline_run_from_config_writes_summary_and_result_json(tmp_path: Path):
    factor_id = "daily_quality_v1"
    lake_root, run_id, evaluation_root = _bootstrap_evaluation_run(tmp_path, factor_id)

    config_path = tmp_path / "factor_admission.yaml"
    _write_admission_config(
        config_path,
        factor_id=factor_id,
        run_id=run_id,
        lake_root=lake_root,
        evaluation_root=evaluation_root,
    )

    output_root = tmp_path / "pipeline_output"
    out = run_from_config(config_path, output_root=output_root)

    assert out["summary"]["configs_total"] == 1
    assert out["summary"]["configs_success"] == 1
    assert out["summary"]["decisions_approved"] == 1
    result = out["results"][0]
    assert result["status"] == "success"
    assert result["decision"] == "approved"
    assert result["approved"] is True
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshot.yaml").exists()
    assert (output_root / "results" / "factor_admission.json").exists()

    payload = json.loads((output_root / "results" / "factor_admission.json").read_text(encoding="utf-8"))
    assert payload["factor_id"] == factor_id
    assert payload["run_id"] == run_id
    assert payload["decision_file"] is not None


def test_pipeline_run_config_directory_writes_batch_summary(tmp_path: Path):
    factor_id = "daily_quality_v1"
    lake_root, run_id, evaluation_root = _bootstrap_evaluation_run(tmp_path, factor_id)

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_admission_config(
        config_dir / "approve.yaml",
        factor_id=factor_id,
        run_id=run_id,
        lake_root=lake_root,
        evaluation_root=evaluation_root,
    )
    _write_admission_config(
        config_dir / "reject.yaml",
        factor_id=factor_id,
        run_id=run_id,
        lake_root=lake_root,
        evaluation_root=evaluation_root,
        approve=False,
    )

    output_root = tmp_path / "batch_output"
    out = run_config_directory(config_dir, output_root=output_root)

    assert out["summary"]["configs_total"] == 2
    assert out["summary"]["configs_success"] == 2
    assert out["summary"]["decisions_rejected"] == 1
    assert len(out["results"]) == 2
    assert (output_root / "run_summary.json").exists()
    assert (output_root / "config_snapshots" / "approve.yaml").exists()
    assert (output_root / "config_snapshots" / "reject.yaml").exists()
    assert (output_root / "results" / "approve.json").exists()
    assert (output_root / "results" / "reject.json").exists()