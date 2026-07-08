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
    reason="ClickHouse 不可达（见 examples/infrastructure/docker-compose.clickhouse.yml）",
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


@pytest.mark.integration
def test_clickhouse_panel_read_sql_pushdown():
    """ClickHouse 长表读 + clickhouse_sql 因子下推（真实 CH）。"""
    from api import rank, ts_mean
    from api.columns import col
    from api.factor import Factor
    from backend.factory import build_backend
    from data_access.clickhouse_panel import ClickHouseConfig, execute_query
    from data_access.clickhouse_write import ensure_panel_table, insert_dataframe
    from runtime.engine import FactorEngine
    from storage.factory import build_data_source

    table = f"panel_sql_it_{os.getpid()}"
    cfg = ClickHouseConfig.from_env()
    execute_query(config=cfg, sql=f"DROP TABLE IF EXISTS {table}")
    ensure_panel_table(
        config=cfg,
        table=table,
        value_columns=["close"],
        timestamp_column="trade_date",
        instrument_column="instrument",
    )
    frame = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-01-02").date(), "instrument": "AAPL", "close": 100.0},
            {"trade_date": pd.Timestamp("2024-01-03").date(), "instrument": "AAPL", "close": 102.0},
            {"trade_date": pd.Timestamp("2024-01-04").date(), "instrument": "AAPL", "close": 104.0},
            {"trade_date": pd.Timestamp("2024-01-02").date(), "instrument": "MSFT", "close": 200.0},
            {"trade_date": pd.Timestamp("2024-01-03").date(), "instrument": "MSFT", "close": 204.0},
            {"trade_date": pd.Timestamp("2024-01-04").date(), "instrument": "MSFT", "close": 208.0},
        ]
    )
    insert_dataframe(
        config=cfg,
        table=table,
        frame=frame,
        column_order=["trade_date", "instrument", "close"],
    )

    source = build_data_source(
        {
            "type": "clickhouse",
            "table": table,
            "timestamp_column": "trade_date",
            "instrument_column": "instrument",
            "fields": {"close": "close"},
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
        }
    )
    factor = Factor(name="ch_sql_push", expr=rank(ts_mean(col("close"), 2)))
    engine = FactorEngine(backend=build_backend("clickhouse_sql"), data_source=source)
    result = engine.run(factor)["result"]
    assert len(result) >= 4
    assert result.notna().any()

    from api import coalesce, ffill

    frame.iloc[1, frame.columns.get_loc("close")] = float("nan")
    execute_query(config=cfg, sql=f"TRUNCATE TABLE {table}")
    insert_dataframe(
        config=cfg,
        table=table,
        frame=frame,
        column_order=["trade_date", "instrument", "close"],
    )
    factor2 = Factor(
        name="ch_coalesce",
        expr=coalesce(ffill(col("close")), col("close")),
    )
    result2 = engine.run(factor2)["result"]
    assert len(result2) >= 4
    assert result2.notna().any()

    from api import clip, protected_log, nan_to_num

    factor3 = Factor(name="ch_clip", expr=clip(col("close"), 50, 250))
    result3 = engine.run(factor3)["result"]
    assert len(result3) >= 4
    assert result3.notna().all()

    factor4 = Factor(name="ch_plog", expr=protected_log(col("close")))
    result4 = engine.run(factor4)["result"]
    assert len(result4) >= 4
    assert result4.notna().all()

    factor5 = Factor(name="ch_nan", expr=nan_to_num(col("close"), 0))
    result5 = engine.run(factor5)["result"]
    assert len(result5) >= 4
    assert result5.notna().all()

    from api import where, is_finite, normalize

    zero = col("close") * 0
    factor6 = Factor(
        name="ch_where_finite",
        expr=where(is_finite(col("close")), col("close"), zero),
    )
    result6 = engine.run(factor6)["result"]
    assert len(result6) >= 4

    factor7 = Factor(name="ch_normalize", expr=normalize(col("close")))
    result7 = engine.run(factor7)["result"]
    assert len(result7) >= 4
    assert result7.notna().any()

    from api import group_normalize

    execute_query(config=cfg, sql=f"DROP TABLE IF EXISTS {table}")
    ensure_panel_table(
        config=cfg,
        table=table,
        value_columns=["close", "grp"],
        timestamp_column="trade_date",
        instrument_column="instrument",
    )
    frame_grp = frame.copy()
    frame_grp["grp"] = [1, 1, 2, 1, 1, 2]
    insert_dataframe(
        config=cfg,
        table=table,
        frame=frame_grp,
        column_order=["trade_date", "instrument", "close", "grp"],
    )
    source_grp = build_data_source(
        {
            "type": "clickhouse",
            "table": table,
            "timestamp_column": "trade_date",
            "instrument_column": "instrument",
            "fields": {"close": "close", "grp": "grp"},
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
        }
    )
    result8 = FactorEngine(
        backend=build_backend("clickhouse_sql"),
        data_source=source_grp,
    ).run(Factor(name="ch_gnorm", expr=group_normalize(col("close"), col("grp"))))["result"]
    assert len(result8) >= 4
    assert result8.notna().any()

    execute_query(config=cfg, sql=f"DROP TABLE IF EXISTS {table}")
