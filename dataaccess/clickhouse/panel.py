# -*- coding: utf-8 -*-
"""
ClickHouse 长表读取（只读）—— 供 factor_engine 与 data_access 消费 panel 数据。

配置来源（优先级：构造参数 > 环境变量）：
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DATABASE, CLICKHOUSE_SECURE (0/1)

依赖 ``clickhouse-connect``（可选）；未安装时在首次连接时给出明确提示。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import ValidationError

_ENV_BOOTSTRAPPED = False


def _bootstrap_env() -> None:
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return
    if os.environ.get("DATA_ACCESS_CLICKHOUSE_LOAD_DOTENV", "").lower() not in {"1", "true", "yes"}:
        _ENV_BOOTSTRAPPED = True
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = val.strip().strip('"').strip("'")
    _ENV_BOOTSTRAPPED = True


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "default"
    secure: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "ClickHouseConfig":
        _bootstrap_env()

        def _field(name: str, env_key: str, default: str) -> str:
            if name in overrides and overrides[name] is not None:
                return str(overrides[name])
            return os.environ.get(env_key, default)

        secure_raw = _field("secure", "CLICKHOUSE_SECURE", "0").lower()
        return cls(
            host=_field("host", "CLICKHOUSE_HOST", "localhost"),
            port=int(_field("port", "CLICKHOUSE_PORT", "8123")),
            username=_field("username", "CLICKHOUSE_USER", "default"),
            password=_field("password", "CLICKHOUSE_PASSWORD", ""),
            database=_field("database", "CLICKHOUSE_DATABASE", "default"),
            secure=secure_raw in {"1", "true", "yes"},
        )


_CLICKHOUSE_RESERVED = {"select", "from", "where", "order", "group", "limit", "date", "table", "value"}


def _quote_ident(name: str) -> str:
    if not name or not name.replace("_", "").isalnum():
        raise ValidationError(f"非法 ClickHouse 标识符: {name!r}")
    if name.lower() in _CLICKHOUSE_RESERVED:
        return f"`{name.replace('`', '``')}`"
    return name


def _build_select_sql(
    *,
    table: str,
    columns: Sequence[str],
    timestamp_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None,
    instrument_filter: Sequence[str] | None,
) -> tuple[str, dict[str, Any]]:
    """组装参数化 SELECT（ClickHouse ``%(name)s`` 风格）。"""
    table_q = _quote_ident(table)
    ts_q = _quote_ident(timestamp_column)
    inst_q = _quote_ident(instrument_column)
    value_cols = [_quote_ident(c) for c in columns]
    select_cols = [ts_q, inst_q, *value_cols]
    sql = f"SELECT {', '.join(select_cols)} FROM {table_q} WHERE 1=1"
    params: dict[str, Any] = {}

    if time_range is not None:
        start, end = time_range
        if start is not None:
            sql += f" AND {ts_q} >= %(start)s"
            params["start"] = start
        if end is not None:
            sql += f" AND {ts_q} <= %(end)s"
            params["end"] = end

    if instrument_filter:
        sql += f" AND {inst_q} IN %(instruments)s"
        params["instruments"] = tuple(instrument_filter)

    sql += f" ORDER BY {ts_q}, {inst_q}"
    return sql, params


def read_columns(
    *,
    config: ClickHouseConfig,
    table: str,
    columns: Sequence[str],
    timestamp_column: str,
    instrument_column: str,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    output_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """从 ClickHouse 读多列，返回 ``{name: MultiIndex Series}``。"""
    if not columns:
        return {}

    try:
        import clickhouse_connect
    except ImportError as exc:
        raise ImportError(
            "ClickHouse 读取需要 clickhouse-connect。pip install clickhouse-connect 后重试。"
        ) from exc

    from data_access.read.adapters import arrow_table_to_multiindex_columns

    sql, params = _build_select_sql(
        table=table,
        columns=columns,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        time_range=time_range,
        instrument_filter=instrument_filter,
    )

    client = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )
    result = client.query(sql, parameters=params)
    table_arrow = result.arrow()

    output_names = output_names or {}
    physical = list(columns)
    return arrow_table_to_multiindex_columns(
        table_arrow,
        timestamp_column=timestamp_column,
        instrument_column=instrument_column,
        value_columns=physical,
        output_names=output_names,
    )


def execute_query(*, config: ClickHouseConfig, sql: str):
    """执行只读 SELECT，返回 Arrow Table（因子 SQL 下推用）。"""
    from .write import execute_select

    return execute_select(config=config, sql=sql)
