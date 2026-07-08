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
    from storage.clickhouse_materializer import ClickHouseMaterializer

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    mock_insert = MagicMock(return_value=2)
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse_panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse_write.insert_factor_series", mock_insert):
            mat = ClickHouseMaterializer(table="fv")
            summary = mat.materialize("fid", series, ensure_table=False)

    assert summary.rows_written == 2
    assert summary.factor_id == "fid"
    assert summary.table == "fv"
    mock_insert.assert_called_once()
