from __future__ import annotations

from pathlib import Path
import sys

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_layer.factor_evaluation.config_runner import run_from_config


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


def test_run_from_config_produces_evaluation_artifacts(tmp_path: Path):
    lake_root = tmp_path / "factor_lake"
    factor_id = "daily_quality_v1"
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

    config_path = tmp_path / "factor_evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "factor_id": factor_id,
                    "run_name": "demo_eval",
                    "primary_horizon": 1,
                },
                "source": {
                    "factor_lake_root": str(lake_root),
                    "market_data_path": str(market_path),
                    "market_timestamp_col": "timestamp",
                    "market_symbol_col": "symbol",
                    "market_price_col": "open",
                },
                "run": {
                    "start": "2024-01-02",
                    "end": "2024-01-11",
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

    result = run_from_config(config_path)

    output_dir = Path(result["output_dir"])
    summary = result["summary_df"].set_index("horizon")
    assert output_dir.exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "daily_ic.parquet").exists()
    assert (output_dir / "quantile_backtest.parquet").exists()
    assert (output_dir / "long_short_returns.parquet").exists()
    assert summary.loc[1, "ic_mean"] > 0.99
    assert summary.loc[1, "rank_ic_mean"] > 0.99
    assert summary.loc[1, "top_minus_bottom_mean"] > 0.0
    assert summary.loc[1, "long_short_total_return"] > 0.0