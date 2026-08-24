# -*- coding: utf-8 -*-
"""ClickHouseMaterializer 单元测试。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from workspace_paths import quant_projects_root


@pytest.fixture(autouse=True)
def _ensure_data_access_on_path():
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
        yield
        sys.path.remove(root)
    else:
        yield


def test_materializer_delegates_to_insert():
    from factor_engine.storage.clickhouse_materializer import ClickHouseMaterializer

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    mock_insert = MagicMock(return_value=2)
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", mock_insert):
            mat = ClickHouseMaterializer(table="fv")
            summary = mat.materialize("fid", series, ensure_table=False)

    assert summary.rows_written == 2
    assert summary.factor_id == "fid"
    assert summary.table == "fv"
    mock_insert.assert_called_once()


def test_materializer_preserve_invalid_rows_propagates_reason():
    from factor_engine.storage.clickhouse_materializer import ClickHouseMaterializer

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, float("inf"), float("nan")], index=idx)
    mock_insert = MagicMock(return_value=3)
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", mock_insert):
            mat = ClickHouseMaterializer(table="fv")
            summary = mat.materialize(
                "fid_invalid",
                series,
                ensure_table=False,
                preserve_invalid_rows=True,
            )

    assert summary.rows_written == 3
    frame = mock_insert.call_args.kwargs["frame"]
    assert len(frame) == 3
    assert int(frame["is_valid"].sum()) == 1
    invalid = frame.loc[frame["is_valid"] == 0]
    assert len(invalid) == 2
    assert (invalid["invalid_reason"] == "inf_or_nan").all()
