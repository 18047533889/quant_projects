"""本地 US StockDailyBar 因子 smoke（需本地 parquet）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from runtime.engine import FactorEngine
from workspace_paths import quant_projects_root, resolve_path


def _stock_daily_bar_root() -> Path:
    return resolve_path("data/us_stock/massive_data/StockDailyBar")


@pytest.mark.integration
def test_factor_engine_run_from_config_on_local_stock_daily_bar(tmp_path: Path):
    if os.environ.get("RUN_LOCAL_MASSIVE_SMOKE") != "1":
        pytest.skip("set RUN_LOCAL_MASSIVE_SMOKE=1 to enable")

    root = _stock_daily_bar_root()
    if not root.is_dir() or not any(root.glob("*.parquet")):
        pytest.skip(f"no parquet under {root}")

    config_path = tmp_path / "local_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "factor": {
                    "name": "local_close_rank",
                    "expr": 'rank(col("close"))',
                },
                "data_source": {
                    "type": "parquet",
                    "root": str(root),
                    "timestamp_col": "TradeDate",
                    "instrument_col": "Ticker",
                    "fields": {"close": "Close"},
                    "max_files": 3,
                },
                "backend": {"type": "pandas"},
                "engine": {"enable_cache": True},
            }
        ),
        encoding="utf-8",
    )

    out = FactorEngine.run_from_config(config_path)
    result = out["result"]
    assert len(result) > 0
    assert int(result.notna().sum()) > 0
    assert quant_projects_root().is_dir()
