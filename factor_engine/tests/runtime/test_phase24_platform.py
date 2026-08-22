"""Phase 24：事件 CLI + read_auto + Numba rolling。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from pipeline_event import run_data_event
from runtime.engine import FactorEngine
from storage.data_access_source import DataAccessSource
from storage.materializer import ParquetMaterializer
from tests.helpers import InMemorySeriesSource


def _close_panel(n_days: int = 5) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["timestamp", "instrument"])
    return pd.Series([float(i + 1) for i in range(len(idx))], index=idx)


def test_run_data_event_dry_run(tmp_path):
    close = _close_panel(5)
    factor = Factor(name="mom", expr=rank(ts_mean(col("close"), 2)))
    fid = "mom_evt_cli"
    lake = tmp_path / "lake"
    eng = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    eng.materialize(
        factor,
        factor_id=fid,
        lake_root=lake,
        expression='rank(ts_mean(col("close"), 2))',
    )
    ParquetMaterializer(lake_root=lake).catalog.record_factor_dependency(
        fid,
        referenced_columns=["close"],
        lookback=2,
        source_dataset="mock_ds",
        frequency="1d",
    )

    out = run_data_event(
        dataset="mock_ds",
        column="close",
        updated_date="2024-01-08",
        lake_root=lake,
        dry_run=True,
    )
    assert out["summary"]["mode"] == "data_event"
    assert out["summary"]["dry_run"] is True
    assert out["summary"]["factor_count"] == 1
    assert Path(out["summary_path"]).is_file()


def test_data_access_source_read_auto_flag():
    from workspace_paths import quant_projects_root

    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    src = DataAccessSource(dataset="test_ds", read_auto=True)
    assert src.read_auto is True

    table = pa.table(
        {
            "trade_date": pa.array(["2024-01-01", "2024-01-02"]),
            "instrument": pa.array(["A", "A"]),
            "close": pa.array([1.0, 2.0]),
        }
    )
    mock_store = MagicMock()
    mock_ds = MagicMock()
    mock_ds.time_column = "trade_date"
    mock_ds.instrument_column = "instrument"
    mock_store.get_dataset.return_value = mock_ds
    mock_store.describe_dataset.return_value = MagicMock(snapshot_id="snap_test")

    mock_result = MagicMock()
    mock_result.table = table
    mock_result.snapshot = MagicMock(snapshot_id="snap_test")
    mock_store.read_result.return_value = mock_result

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    expected = pd.Series([1.0, 2.0], index=idx, name="close")

    with patch("storage.data_access_source._get_store", return_value=mock_store):
        with patch(
            "data_access.store.adapter_options_for_dataset",
            return_value={"normalize_timestamp": False},
        ):
            with patch(
                "data_access.read.adapters.arrow_table_to_multiindex_columns",
                return_value={"close": expected},
            ):
                got = src.load_columns(["close"])

    assert "close" in got
    mock_store.read_result.assert_called_once()
    mock_store.read_auto.assert_not_called()


def test_polars_backend_enables_read_auto_on_data_access_source():
    from backend.polars_backend import PolarsBackend
    from backend.context import ExecutionContext
    from backend.pandas_backend import PandasBackend as PB
    from planner.logical_plan import PlanNode

    src = DataAccessSource(dataset="ds", read_auto=False)
    backend = PolarsBackend(use_lazy=True)
    ctx = ExecutionContext(data_source=src)
    plan = PlanNode(op="col", attrs={"name": "close"}, inputs=[])

    with patch.object(PB, "_eval", return_value=pd.DataFrame()):
        backend.execute(plan, ctx)
    # R13 P1-66: a lazy run must NOT leave a persistent mutation on the source —
    # the pre-run eager read semantics are restored afterwards.
    assert src.read_auto is False


@pytest.mark.parametrize("use_numba", [False, True])
def test_ts_rank_numba_optional(use_numba, monkeypatch):
    from cleaned_operators.common.time_series import TSRank

    if use_numba:
        monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    else:
        monkeypatch.delenv("FACTOR_ENGINE_USE_NUMBA", raising=False)

    dates = pd.bdate_range("2024-01-02", periods=10)
    panel = pd.DataFrame(
        np.arange(10, dtype=float).reshape(-1, 1),
        index=dates,
        columns=["A"],
    )
    op = TSRank()
    fast = op._calculate_series(panel, window=5)
    ref = panel.rolling(window=5, min_periods=1).rank(pct=True)
    pd.testing.assert_frame_equal(fast, ref)
