# -*- coding: utf-8
"""因子结果写入 ClickHouse（ReplacingMergeTree 长表，与 Parquet 路径同构清洗/DQ）。"""
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
    dq_report: dict[str, Any] | None = None


class ClickHouseMaterializer:
    """将 ``pd.Series`` 因子结果写入 ClickHouse，复用 Parquet 规范化与清洗逻辑。"""

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
        ensure_table: bool = True,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_metadata: bool = True,
    ) -> ClickHouseMaterializeSummary:
        _ensure_data_access()
        from data_access.clickhouse_panel import ClickHouseConfig
        from data_access.clickhouse_write import insert_factor_dataframe
        from storage.factor_frame import prepare_factor_dataframe

        version_key = str(factor_version) if factor_version else None
        df, dq_dict, _ = prepare_factor_dataframe(
            result,
            ast_hash=version_key,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            write_metadata=write_metadata,
        )
        if df.empty:
            logger.warning("因子 '%s' ClickHouse 清洗后无数据，跳过写入", factor_id)
            config = ClickHouseConfig.from_env(**self._ch_overrides)
            return ClickHouseMaterializeSummary(
                factor_id=factor_id,
                table=self.table,
                rows_written=0,
                database=config.database,
                dq_report=dq_dict,
            )

        out = df.rename(
            columns={
                "datetime": self.timestamp_column,
                "asset": self.instrument_column,
            }
        )
        out[self.factor_id_column] = str(factor_id)
        if "factor_version" not in out.columns:
            out["factor_version"] = str(factor_version)
        if "data_snapshot_id" not in out.columns:
            out["data_snapshot_id"] = data_snapshot_id or ""

        config = ClickHouseConfig.from_env(**self._ch_overrides)
        logger.info(
            "ClickHouse 落盘 factor_id=%s table=%s database=%s rows=%d",
            factor_id,
            self.table,
            config.database,
            len(out),
        )
        rows = insert_factor_dataframe(
            config=config,
            table=self.table,
            frame=out,
            timestamp_column=self.timestamp_column,
            instrument_column=self.instrument_column,
            factor_id_column=self.factor_id_column,
            ensure_table=ensure_table,
        )
        return ClickHouseMaterializeSummary(
            factor_id=factor_id,
            table=self.table,
            rows_written=rows,
            database=config.database,
            dq_report=dq_dict,
        )
