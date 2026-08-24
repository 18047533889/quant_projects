# -*- coding: utf-8
"""store.sql_result 合并 snapshot 测试。"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest


@pytest.fixture
def sql_store(tmp_path, monkeypatch):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    pd.DataFrame({"x": [1, 2], "y": [10.0, 20.0]}).to_parquet(root_a / "data.parquet")
    pd.DataFrame({"x": [1, 2], "z": [100.0, 200.0]}).to_parquet(root_b / "data.parquet")

    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        dedent(
            f"""
        ds_a:
          kind: static
          access_mode: published
          root: {root_a}
          glob: "**/*.parquet"
          time_column: x
          instrument_column: y
          schema:
            x: int
            y: double
        ds_b:
          kind: static
          access_mode: published
          root: {root_b}
          glob: "**/*.parquet"
          time_column: x
          instrument_column: z
          schema:
            x: int
            z: double
    """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(cfg))
    from data_access import reset_store

    reset_store()
    from data_access import get_store

    return get_store()


def test_sql_result_returns_merged_snapshot(sql_store):
    result = sql_store.sql_result(
        "SELECT a.x, a.y, b.z FROM {{ds_a}} a JOIN {{ds_b}} b ON a.x = b.x",
        read_datasets=["ds_a", "ds_b"],
        view_columns={"ds_a": ["x", "y"], "ds_b": ["x", "z"]},
    )
    assert result.table.num_rows == 2
    assert result.snapshot.snapshot_id
    assert "ds_a" in result.snapshot.dataset
    assert "ds_b" in result.snapshot.dataset
