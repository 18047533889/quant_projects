# -*- coding: utf-8 -*-
"""因子结果写入 ClickHouse（ReplacingMergeTree 长表）。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from logging_utils import get_logger
from workspace_paths import quant_projects_root

logger = get_logger("storage.clickhouse_materializer")


def _ensure_data_access() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class ClickHouseMaterializeSummary:
    factor_id: str
    table: str
    rows_written: int
    database: str


class ClickHouseMaterializer:
    """将 ``pd.Series`` 因子结果写入 ClickHouse ``factor_values`` 表。"""

    def __init__(
        self,
        *,
        table: str = "factor_values",
        timestamp_column: str = "trade_date",
        instrument_column: str = "instrument",
        factor_id_column: str = "factor_id",
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
    ) -> None:
        self.table = table
        self.timestamp_column = timestamp_column
        self.instrument_column = instrument_column
        self.factor_id_column = factor_id_column
        self._ch_overrides = {
            k: v
            for k, v in {
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "secure": secure,
            }.items()
            if v is not None
        }

    def materialize(
        self,
        factor_id: str,
        result,
        *,
        factor_version: str = "",
        data_snapshot_id: str | None = None,
        is_valid: int = 1,
        ensure_table: bool = True,
    ) -> ClickHouseMaterializeSummary:
        _ensure_data_access()
        from data_access.clickhouse_panel import ClickHouseConfig
        from data_access.clickhouse_write import insert_factor_series

        config = ClickHouseConfig.from_env(**self._ch_overrides)
        logger.info(
            "ClickHouse 落盘 factor_id=%s table=%s database=%s",
            factor_id,
            self.table,
            config.database,
        )
        rows = insert_factor_series(
            config=config,
            table=self.table,
            factor_id=factor_id,
            series=result,
            timestamp_column=self.timestamp_column,
            instrument_column=self.instrument_column,
            factor_id_column=self.factor_id_column,
            factor_version=factor_version,
            data_snapshot_id=data_snapshot_id,
            is_valid=is_valid,
            ensure_table=ensure_table,
        )
        return ClickHouseMaterializeSummary(
            factor_id=factor_id,
            table=self.table,
            rows_written=rows,
            database=config.database,
        )
