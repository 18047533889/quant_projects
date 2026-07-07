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

from factor_layer.factor_admission.catalog import AdmissionCatalog
from factor_layer.factor_admission.config_runner import run_from_config
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


def _bootstrap_evaluation_run(tmp_path: Path, factor_id: str) -> tuple[Path, str]:
    lake_root = tmp_path / "factor_lake"
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
                    "root": str(tmp_path / "evaluations"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = run_evaluation_from_config(config_path)
    return lake_root, str(result["meta"]["run_id"])


def test_run_from_config_indexes_and_approves_evaluation(tmp_path: Path):
    factor_id = "daily_quality_v1"
    lake_root, run_id = _bootstrap_evaluation_run(tmp_path, factor_id)

    config_path = tmp_path / "factor_admission.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "factor_id": factor_id,
                    "run_id": run_id,
                },
                "source": {
                    "factor_lake_root": str(lake_root),
                    "evaluation_root": str(tmp_path / "evaluations"),
                },
                "decision": {
                    "mode": "rule_based",
                    "decided_by": "system",
                    "policy_name": "smoke_policy",
                    "primary_horizon": 1,
                    "thresholds": {
                        "min_rank_ic_mean": 0.9,
                        "min_long_short_sharpe": 0.5,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    decision = run_from_config(config_path)

    assert decision["approved"] is True
    assert decision["decision"] == "approved"
    run_dir = tmp_path / "evaluations" / factor_id / run_id
    assert (run_dir / "admission_decision.json").exists()
    payload = json.loads((run_dir / "admission_decision.json").read_text(encoding="utf-8"))
    assert payload["approved"] is True

    with AdmissionCatalog(lake_root / "_catalog.sqlite") as catalog:
        status = catalog.get_factor_status(factor_id)
        assert status is not None
        assert status["status"] == "approved"
        assert status["latest_run_id"] == run_id
        assert status["latest_approved_run_id"] == run_id
        library_rows = catalog.list_factor_library()
        assert any(row["factor_id"] == factor_id and row["status"] == "approved" for row in library_rows)


def test_manual_rejection_preserves_latest_approved_run(tmp_path: Path):
    factor_id = "daily_quality_v1"
    lake_root, run_id = _bootstrap_evaluation_run(tmp_path, factor_id)

    approved_config = tmp_path / "approve.yaml"
    approved_config.write_text(
        yaml.safe_dump(
            {
                "meta": {"factor_id": factor_id, "run_id": run_id},
                "source": {
                    "factor_lake_root": str(lake_root),
                    "evaluation_root": str(tmp_path / "evaluations"),
                },
                "decision": {
                    "mode": "manual",
                    "approve": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_from_config(approved_config)

    rejected_config = tmp_path / "reject.yaml"
    rejected_config.write_text(
        yaml.safe_dump(
            {
                "meta": {"factor_id": factor_id, "run_id": run_id},
                "source": {
                    "factor_lake_root": str(lake_root),
                    "evaluation_root": str(tmp_path / "evaluations"),
                },
                "decision": {
                    "mode": "manual",
                    "approve": False,
                    "reason": "manual review failed",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    decision = run_from_config(rejected_config)

    assert decision["approved"] is False
    with AdmissionCatalog(lake_root / "_catalog.sqlite") as catalog:
        status = catalog.get_factor_status(factor_id)
        assert status is not None
        assert status["status"] == "evaluated"
        assert status["latest_run_id"] == run_id
        assert status["latest_approved_run_id"] == run_id