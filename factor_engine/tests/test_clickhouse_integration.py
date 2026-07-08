# -*- coding: utf-8
"""ClickHouse 集成测试（需本地 docker compose 或远程 CH 可达）。"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

from workspace_paths import load_env_file, quant_projects_root


def _ch_available() -> bool:
    load_env_file()
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from data_access.clickhouse_panel import ClickHouseConfig, execute_query

        cfg = ClickHouseConfig.from_env()
        table = execute_query(config=cfg, sql="SELECT 1 AS value")
        return table.num_rows == 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ch_available(),
    reason="ClickHouse 不可达（见 examples/infra/clickhouse-compose.yaml）",
)


@pytest.fixture(autouse=True)
def _path():
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    yield


@pytest.mark.integration
def test_clickhouse_config_from_env():
    from data_access.clickhouse_panel import ClickHouseConfig

    cfg = ClickHouseConfig.from_env()
    assert cfg.host
    assert cfg.port > 0
    assert cfg.database


@pytest.mark.integration
def test_clickhouse_materialize_engine_path(tmp_path):
    """端到端：run + materialize(write_target=clickhouse) 写入真实 CH。"""
    from api.columns import col
    from api.factor import Factor
    from backend.pandas_backend import PandasBackend
    from runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=4), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([float(i) for i in range(len(idx))], index=idx)
    factor = Factor(name="ch_it", expr=col("close"))
    fid = f"ch_it_{os.getpid()}"

    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    out = engine.materialize(
        factor,
        factor_id=fid,
        lake_root=tmp_path,
        write_target="clickhouse",
        clickhouse_table="factor_values_it",
        ch_ensure_table=True,
        expression='col("close")',
    )
    mat = out["materialization"]
    assert mat.get("dual_write_committed") is True or mat.get("clickhouse")
    ch = mat.get("clickhouse") or {}
    assert ch.get("rows_written", mat.get("rows_written", 0)) >= 4

    from data_access.clickhouse_panel import ClickHouseConfig, execute_query

    cfg = ClickHouseConfig.from_env()
    result = execute_query(
        config=cfg,
        sql=(
            "SELECT count() AS c FROM factor_values_it "
            f"WHERE factor_id = '{fid}'"
        ),
    )
    assert int(result.column("c")[0].as_py()) >= 4
