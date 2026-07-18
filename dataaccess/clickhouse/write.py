# -*- coding: utf-8 -*-
"""
ClickHouse 写入：因子长表 / panel 长表落盘。

与 ``ParquetMaterializer`` 长表格式对齐：``(timestamp, instrument, value)`` +
可选元数据列。读路径见 ``data_access.clickhouse.panel``。
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .panel import ClickHouseConfig, _quote_ident
from data_access.core.exceptions import ValidationError

_FORBIDDEN_DML = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|ATTACH|DETACH|RENAME|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

DEFAULT_FACTOR_TABLE = "factor_values"
DEFAULT_PANEL_TABLE = "panel_daily"


def _require_clickhouse_connect():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise ImportError(
            "ClickHouse 写入需要 clickhouse-connect。pip install clickhouse-connect 后重试。"
        ) from exc
    return clickhouse_connect


def get_client(config: ClickHouseConfig):
    clickhouse_connect = _require_clickhouse_connect()
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )


def _assert_select_only(sql: str) -> None:
    if not sql or not sql.strip():
        raise ValidationError("SQL 不能为空")
    if _FORBIDDEN_DML.search(sql):
        raise ValidationError("仅允许 SELECT 查询")


def ensure_factor_table(
    *,
    config: ClickHouseConfig,
    table: str = DEFAULT_FACTOR_TABLE,
    timestamp_column: str = "trade_date",
    instrument_column: str = "instrument",
    factor_id_column: str = "factor_id",
) -> None:
    """创建因子长表（ReplacingMergeTree，按 factor_id + 日期 + 标的 upsert）。"""
    t = _quote_ident(table)
    ts = _quote_ident(timestamp_column)
    inst = _quote_ident(instrument_column)
    fid = _quote_ident(factor_id_column)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {t} (
        {ts} Date,
        {inst} String,
        {fid} String,
        value Float32,
        calc_time DateTime DEFAULT now(),
        factor_version String DEFAULT '',
        data_snapshot_id String DEFAULT '',
        is_valid UInt8 DEFAULT 1,
        invalid_reason String DEFAULT ''
    ) ENGINE = ReplacingMergeTree(calc_time)
    PARTITION BY toYYYYMM({ts})
    ORDER BY ({fid}, {ts}, {inst})
    """
    client = get_client(config)
    client.command(ddl)


def ensure_panel_table(
    *,
    config: ClickHouseConfig,
    table: str = DEFAULT_PANEL_TABLE,
    timestamp_column: str = "trade_date",
    instrument_column: str = "instrument",
    value_columns: Sequence[str],
    column_types: Mapping[str, str] | None = None,
) -> None:
    """创建 panel 长表（MergeTree，宽列存于行）。"""
    if not value_columns:
        raise ValidationError("value_columns 不能为空")
    t = _quote_ident(table)
    ts = _quote_ident(timestamp_column)
    inst = _quote_ident(instrument_column)
    types = dict(column_types or {})
    col_defs = []
    for col in value_columns:
        ch_type = types.get(col, "Float64")
        col_defs.append(f"{_quote_ident(col)} {ch_type}")
    cols_sql = ",\n        ".join(col_defs)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {t} (
        {ts} Date,
        {inst} String,
        {cols_sql}
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM({ts})
    ORDER BY ({ts}, {inst})
    """
    client = get_client(config)
    client.command(ddl)


def insert_dataframe(
    *,
    config: ClickHouseConfig,
    table: str,
    frame,
    column_order: Sequence[str] | None = None,
) -> int:
    """批量 INSERT（``clickhouse-connect`` Arrow/pandas 路径）。"""
    if frame is None or len(frame) == 0:
        return 0
    client = get_client(config)
    cols = list(column_order) if column_order else list(frame.columns)
    client.insert_df(table, frame[cols])
    return len(frame)


def _series_to_long_table(
    series,
    *,
    timestamp_col: str,
    asset_col: str,
    value_col: str = "value",
):
    import pandas as pd

    if not isinstance(series, pd.Series):
        raise ValueError(f"期望 pd.Series，实际 {type(series).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise ValueError("期望 MultiIndex(timestamp, instrument) Series")
    frame = series.reset_index()
    cols = list(frame.columns)
    frame = frame.rename(
        columns={cols[0]: timestamp_col, cols[1]: asset_col, cols[-1]: value_col}
    )
    out = frame[[timestamp_col, asset_col, value_col]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out[asset_col] = out[asset_col].astype("string")
    out[value_col] = out[value_col].astype("float32")
    return out


def insert_factor_dataframe(
    *,
    config: ClickHouseConfig,
    table: str = DEFAULT_FACTOR_TABLE,
    frame,
    timestamp_column: str = "trade_date",
    instrument_column: str = "instrument",
    factor_id_column: str = "factor_id",
    ensure_table: bool = True,
) -> int:
    """长表 DataFrame 写入 ClickHouse（含 is_valid / invalid_reason 列）。"""
    import pandas as pd

    if frame is None or len(frame) == 0:
        return 0
    if ensure_table:
        ensure_factor_table(
            config=config,
            table=table,
            timestamp_column=timestamp_column,
            instrument_column=instrument_column,
            factor_id_column=factor_id_column,
        )
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column]).dt.date
    work[instrument_column] = work[instrument_column].astype(str)
    if "is_valid" not in work.columns:
        work["is_valid"] = 1
    if "invalid_reason" not in work.columns:
        work["invalid_reason"] = ""
    if "calc_time" not in work.columns:
        work["calc_time"] = pd.Timestamp.utcnow().tz_localize(None)
    cols = [
        timestamp_column,
        instrument_column,
        factor_id_column,
        "value",
        "calc_time",
        "factor_version",
        "data_snapshot_id",
        "is_valid",
        "invalid_reason",
    ]
    cols = [c for c in cols if c in work.columns]
    return insert_dataframe(config=config, table=table, frame=work, column_order=cols)


def insert_factor_series(
    *,
    config: ClickHouseConfig,
    table: str = DEFAULT_FACTOR_TABLE,
    factor_id: str,
    series,
    timestamp_column: str = "trade_date",
    instrument_column: str = "instrument",
    factor_id_column: str = "factor_id",
    factor_version: str = "",
    data_snapshot_id: str | None = None,
    is_valid: int = 1,
    ensure_table: bool = True,
) -> int:
    """MultiIndex Series → 因子长表写入 ClickHouse。"""
    import pandas as pd

    if ensure_table:
        ensure_factor_table(
            config=config,
            table=table,
            timestamp_column=timestamp_column,
            instrument_column=instrument_column,
            factor_id_column=factor_id_column,
        )

    long_df = _series_to_long_table(
        series,
        timestamp_col=timestamp_column,
        asset_col=instrument_column,
    )
    long_df[timestamp_column] = pd.to_datetime(long_df[timestamp_column]).dt.date
    long_df[factor_id_column] = str(factor_id)
    long_df["factor_version"] = str(factor_version)
    long_df["data_snapshot_id"] = data_snapshot_id or ""
    long_df["is_valid"] = int(is_valid)
    long_df["invalid_reason"] = ""
    long_df["calc_time"] = pd.Timestamp.utcnow().tz_localize(None)

    cols = [
        timestamp_column,
        instrument_column,
        factor_id_column,
        "value",
        "calc_time",
        "factor_version",
        "data_snapshot_id",
        "is_valid",
        "invalid_reason",
    ]
    return insert_dataframe(config=config, table=table, frame=long_df, column_order=cols)


def load_parquet_to_panel_table(
    *,
    config: ClickHouseConfig,
    parquet_path: str,
    table: str = DEFAULT_PANEL_TABLE,
    timestamp_column: str = "trade_date",
    instrument_column: str = "instrument",
    value_columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    batch_size: int = 100_000,
    ensure_table: bool = True,
) -> int:
    """Parquet 长表/宽表 → ClickHouse panel 表（本地 ETL）。"""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if value_columns is None:
        value_columns = [
            c
            for c in df.columns
            if c not in {timestamp_column, instrument_column}
        ]
    if ensure_table:
        ensure_panel_table(
            config=config,
            table=table,
            timestamp_column=timestamp_column,
            instrument_column=instrument_column,
            value_columns=value_columns,
        )

    if time_range is not None:
        start, end = time_range
        ts = pd.to_datetime(df[timestamp_column])
        if start is not None:
            df = df.loc[ts >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[ts <= pd.Timestamp(end)]

    if len(df) == 0:
        return 0

    work = df.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column]).dt.date
    work[instrument_column] = work[instrument_column].astype(str)

    total = 0
    cols = [timestamp_column, instrument_column, *value_columns]
    for start in range(0, len(work), batch_size):
        chunk = work.iloc[start : start + batch_size]
        total += insert_dataframe(config=config, table=table, frame=chunk, column_order=cols)
    return total


def execute_select(*, config: ClickHouseConfig, sql: str):
    """只读 SELECT → Arrow Table（因子 SQL 下推）。"""
    _assert_select_only(sql)
    client = get_client(config)
    return client.query(sql).arrow()


def verify_factor_write(
    *,
    config: ClickHouseConfig,
    table: str,
    factor_id: str,
    expected_rows: int,
    factor_id_column: str = "factor_id",
) -> dict[str, Any]:
    """写后校验：FINAL 计数应不少于本次写入行数（ReplacingMergeTree 去重语义）。"""
    if expected_rows <= 0:
        return {"verified_rows": 0, "expected_rows": 0, "ok": True}

    t = _quote_ident(table)
    fid_col = _quote_ident(factor_id_column)
    escaped_id = str(factor_id).replace("'", "''")
    sql = f"SELECT count() AS cnt FROM {t} FINAL WHERE {fid_col} = '{escaped_id}'"
    client = get_client(config)
    result = client.query(sql)
    verified = int(result.result_rows[0][0]) if result.result_rows else 0
    if verified < expected_rows:
        raise ValidationError(
            f"ClickHouse 写后校验失败：factor_id={factor_id!r} FINAL 计数 {verified} "
            f"< 期望 {expected_rows}（table={table}）。"
        )
    return {
        "verified_rows": verified,
        "expected_rows": expected_rows,
        "ok": True,
    }
