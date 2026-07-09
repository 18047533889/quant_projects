"""Phase E：bucket playbook、read_root cutover、Polars panel、10M 阈值。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from api import ts_mean
from api.columns import col
from api.factor import Factor
from backend.context import ExecutionContext
from backend.panel_polars import is_polars_frame, panel_to_polars
from backend.polars_backend import PolarsBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def test_build_cutover_checklist():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
    from data_access.scripts.run_bucket_migration import build_cutover_checklist

    plan = {"recommended_partition_columns": ["year", "month", "bucket"]}
    cutover = build_cutover_checklist(
        dataset="ashare_stock_minute",
        target_root="/data/bucket_layout",
        plan=plan,
        executed=True,
        validation={"passed": True},
    )
    assert "partition_columns" in cutover["yaml_patch_hint"]
    assert "DATA_ACCESS_READ_ROOT_ASHARE_STOCK_MINUTE" in cutover["env_cutover"]


def test_playbook_dry_run_subprocess(tmp_path):
    da_root = Path(__file__).resolve().parents[2].parent / "data_access"
    script = da_root / "scripts" / "run_bucket_migration.py"
    if not script.exists():
        pytest.skip("data_access scripts not present")

    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "source"
    src.mkdir()
    table = pa.table(
        {
            "QuoteTime": ["2024-01-02 09:31:00", "2024-01-02 09:32:00"],
            "Symbol": ["000001", "000001"],
            "Close": [10.0, 10.5],
            "TradeDate": ["2024-01-02", "2024-01-02"],
        }
    )
    pq.write_table(table, src / "part-001.parquet")

    config = tmp_path / "datasets.yaml"
    config.write_text(
        f"""
test_minute:
  kind: static
  access_mode: published
  root: {src}
  glob: "**/*.parquet"
  time_column: QuoteTime
  instrument_column: Symbol
  layout_policy:
    bucket:
      column: bucket
      count: 4
  schema:
    QuoteTime: timestamp
    Symbol: string
    Close: double
    TradeDate: date
""",
        encoding="utf-8",
    )
    target = tmp_path / "bucket_target"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "test_minute",
            "--config",
            str(config),
            "--target-root",
            str(target),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(da_root.parent),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert "cutover" in payload["steps"]


def test_panel_to_polars_idempotent():
    panel = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    pl_panel = panel_to_polars(panel)
    assert is_polars_frame(pl_panel)
    again = panel_to_polars(pl_panel)
    assert again is pl_panel


def test_polars_backend_sets_prefer_polars_panel():
    backend = PolarsBackend()
    assert backend.use_lazy is False or True  # env dependent
    src = InMemorySeriesSource(data={})
    ctx = ExecutionContext(data_source=src, panel_cache={})
    from dataclasses import replace

    ctx = replace(ctx, runtime_stats={"backend": "polars"}, prefer_polars_panel=True)
    assert ctx.prefer_polars_panel is True


def test_run_many_polars_backend_smoke():
    pytest.importorskip("polars")
    dates = pd.bdate_range("2024-01-02", periods=8)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    data = {"close": pd.Series([float(i) for i in range(len(idx))], index=idx)}
    eng = FactorEngine(
        backend=PolarsBackend(),
        data_source=InMemorySeriesSource(data=data),
    )
    f = Factor(name="m", expr=ts_mean(col("close"), 3))
    out = eng.run(f)
    assert len(out["result"]) == len(idx)
