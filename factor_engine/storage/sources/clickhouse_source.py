# -*- coding: utf-8 -*-
"""ClickHouse 数据源：长表只读，返回与引擎一致的 MultiIndex Series。"""
from __future__ import annotations

from typing import Any

from factor_engine.util.logging_utils import get_logger

from .datasource import DataSource
from .field_plan import NormalizedFieldPlan, plan_from_field_spec

logger = get_logger("factor_engine.storage.clickhouse_source")


def _ensure_data_access_importable() -> None:
    """R21-130..133: import the installed ``data_access`` (no sys.path injection)."""
    from factor_engine.storage.data_access_loader import ensure_data_access_importable

    ensure_data_access_importable()


class ClickHouseSource(DataSource):
    """从 ClickHouse 表读取 MultiIndex Series。
    
    参数:
        table: ClickHouse 表名（可选）
        timestamp_column: 见函数签名（可选）
        instrument_column: 见函数签名（可选）
        fields: 逻辑列到物理列映射（可选）
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        instrument_filter: 见函数签名（可选）
        host: 见函数签名（可选）
        port: 见函数签名（可选）
        database: 见函数签名（可选）
        username: 见函数签名（可选）
        password: 见函数签名（可选）
        secure: 见函数签名（可选）
    """

    def __init__(
        self,
        *,
        table: str,
        timestamp_column: str = "trade_date",
        instrument_column: str = "instrument",
        fields: dict[str, str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        instrument_filter: list[str] | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
        frequency: str | None = None,
        session_column: str | None = None,
    ) -> None:
        """初始化实例。
        
        参数:
            table: ClickHouse 表名（可选）
            timestamp_column: 见函数签名（可选）
            instrument_column: 见函数签名（可选）
            fields: 逻辑列到物理列映射（可选）
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）
            instrument_filter: 见函数签名（可选）
            host: 见函数签名（可选）
            port: 见函数签名（可选）
            database: 见函数签名（可选）
            username: 见函数签名（可选）
            password: 见函数签名（可选）
            secure: 见函数签名（可选）
        
        返回:
            无
        """
        self.table = table
        self.timestamp_column = timestamp_column
        self.instrument_column = instrument_column
        self.fields = dict(fields or {})
        self.start_date = start_date
        self.end_date = end_date
        self.instrument_filter = list(instrument_filter) if instrument_filter else None
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
        self.frequency = frequency
        self.session_column = session_column

    def execution_spec(self) -> dict[str, Any]:
        """返回可重建（``storage.factory.build_data_source``）的 canonical 配置。

        凭据字段（host/port/database/username/password/secure）进入 lineage 身份
        前会被 ``runtime.lineage._canonicalize_config_value`` redact，不会把密钥
        hash 进 factor_version / data_snapshot_id。
        """
        from .datasource import clean_execution_spec

        return clean_execution_spec(
            {
                "type": "clickhouse",
                "table": self.table,
                "timestamp_col": self.timestamp_column,
                "instrument_col": self.instrument_column,
                "fields": dict(self.fields or {}),
                "start_date": self.start_date,
                "end_date": self.end_date,
                "instrument_filter": (
                    list(self.instrument_filter)
                    if self.instrument_filter is not None
                    else None
                ),
                "host": self._ch_overrides.get("host"),
                "port": self._ch_overrides.get("port"),
                "database": self._ch_overrides.get("database"),
                "username": self._ch_overrides.get("username"),
                "password": self._ch_overrides.get("password"),
                "secure": self._ch_overrides.get("secure"),
            }
        )
        #: Unified field-resolution plans keyed by logical name (P0-11).
        self._field_plans: dict[str, NormalizedFieldPlan] = {}
        self._column_cache: dict[str, Any] = {}
        self._panel_cache: dict[str, Any] = {}

    def _time_range(self) -> tuple[Any, Any] | None:
        """_time_range。
        
        参数:
            无
        
        返回:
            tuple[Any, Any] | None
        """
        if self.start_date is None and self.end_date is None:
            return None
        return (self.start_date, self.end_date)

    def _resolve_columns(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        """_resolve_columns：逻辑名 → 物理列，并产出统一 NormalizedFieldPlan（P0-11）。

        scale 归一化读取 FE FIELD_REGISTRY 的 ``scale_to_canonical``，使 Return(bp)
        等字段在 ClickHouse 直查路径与 DataAccess 一致归一化（此前完全没有
        unit/scale 感知）。

        参数:
            names: 逻辑列名列表

        返回:
            tuple[list[str], dict[str, str]]
        """
        # R17-001: resolve within the table's market registry when the table name
        # declares one (us_* / ashare_*); otherwise keep the legacy A-share
        # registry for bare logical-table names.
        table_lower = str(self.table or "").strip().lower()
        clickhouse_market = (
            "us" if table_lower.startswith("us_")
            else "ashare" if table_lower.startswith("ashare_") or table_lower.startswith("cn_")
            else None
        )

        physical: list[str] = []
        output_names: dict[str, str] = {}
        plans: dict[str, NormalizedFieldPlan] = {}
        for name in names:
            src = self.fields.get(name, name)
            physical.append(src)
            if src != name:
                output_names[src] = name
            spec = None
            try:
                if clickhouse_market:
                    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                    spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(
                        clickhouse_market, name, table=self.table, strict=False
                    )
                else:
                    from factor_engine.fields import FIELD_REGISTRY

                    spec = FIELD_REGISTRY.get(name, table=self.table)
            except Exception:
                spec = None
            if spec is not None:
                plans[name] = plan_from_field_spec(name, spec)
            else:
                plans[name] = NormalizedFieldPlan(
                    logical_concept=name, physical_fields=(src,)
                )
        self._field_plans = plans
        return physical, output_names

    def _normalize_from_plans(self, fetched: dict[str, Any], names: list[str]) -> None:
        """按统一 field plan 的 scale 就地归一化已读取的列（P0-11）。"""
        for name in names:
            plan = self._field_plans.get(name)
            if plan is None or not plan.is_scale_applicable:
                continue
            scale = float(plan.scale)
            if scale != 1.0:
                fetched[name] = fetched[name] * scale

    def load_column(self, name: str):
        """load_column。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._column_cache:
            return self._column_cache[name]
        return self.load_columns([name])[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        """load_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            dict[str, Any]
        """
        needed = [n for n in names if n not in self._column_cache]
        if not needed:
            return {n: self._column_cache[n] for n in names}

        _ensure_data_access_importable()
        from data_access.clickhouse.panel import ClickHouseConfig, read_columns

        physical, output_names = self._resolve_columns(needed)
        config = ClickHouseConfig.from_env(**self._ch_overrides)
        logger.info(
            "ClickHouse 读表 '%s' 列 %s (time_range=%s)",
            self.table,
            needed,
            self._time_range(),
        )
        fetched = read_columns(
            config=config,
            table=self.table,
            columns=physical,
            timestamp_column=self.timestamp_column,
            instrument_column=self.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            output_names=output_names or None,
        )
        # P0-11: ClickHouse 直查无 DataAccess 输出层 unit 归一化，按统一 plan 补 scale。
        self._normalize_from_plans(fetched, needed)
        for n in needed:
            self._column_cache[n] = fetched[n]
        return {n: self._column_cache[n] for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        """prefetch_columns。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        if names:
            self.load_columns(names)

    def prefetch_panels(self, names: list[str]) -> None:
        """批量预加载宽表 panel（一次 SQL 读 + 本地 unstack）。
        
        参数:
            names: 逻辑列名列表
        
        返回:
            无
        """
        needed = [n for n in names if n not in self._panel_cache]
        if not needed:
            return
        batch = self.load_columns(needed)
        level = self.instrument_column
        for name, series in batch.items():
            if name not in self._panel_cache:
                inst_level = (
                    series.index.names[-1]
                    if series.index.names[-1] is not None
                    else level
                )
                self._panel_cache[name] = series.unstack(level=inst_level)

    def load_column_panel(self, name: str):
        """load_column_panel。
        
        参数:
            name: 逻辑列名
        
        返回:
            无
        """
        if name in self._panel_cache:
            return self._panel_cache[name]
        series = self.load_column(name)
        inst_level = (
            series.index.names[-1]
            if series.index.names[-1] is not None
            else self.instrument_column
        )
        panel = series.unstack(level=inst_level)
        self._panel_cache[name] = panel
        return panel

    def scan_polars_long(self, columns: list[str]):
        """ClickHouse 直查 long LazyFrame（无 unstack / 无宽表 panel）。

        Ordering guarantee (P0-10): SQL 携带 ``ORDER BY <instrument>, <time>``，
        post-query 再经 ``scan_clickhouse_long`` 强制排序（分钟数据追加 session）。

        参数:
            columns: 列名列表

        返回:
            无
        """
        physical, output_names = self._resolve_columns(columns)
        _ensure_data_access_importable()
        from data_access.clickhouse.panel import ClickHouseConfig
        from factor_engine.backend.polars_lazy import scan_clickhouse_long

        config = ClickHouseConfig.from_env(**self._ch_overrides)
        lf = scan_clickhouse_long(
            config,
            table=self.table,
            timestamp_column=self.timestamp_column,
            instrument_column=self.instrument_column,
            physical_columns=physical,
            output_names=output_names,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            frequency=self.frequency,
            session_column=self.session_column,
        )
        # P0-11: ClickHouse 直查路径按统一 field plan 补 scale 归一化（此前无 unit/scale 感知）。
        try:
            import polars as pl

            plans = self._field_plans
            expressions = []
            for name in columns:
                plan = plans.get(name)
                if plan is None or not plan.is_scale_applicable:
                    continue
                scale = float(plan.scale)
                if scale != 1.0:
                    expressions.append((pl.col(name).cast(pl.Float64) * scale).alias(name))
            if expressions:
                lf = lf.with_columns(expressions)
        except ImportError:
            pass
        return lf

    def dataset_axis_columns(self) -> tuple[str, str]:
        """dataset_axis_columns。
        
        参数:
            无
        
        返回:
            tuple[str, str]
        """
        return self.timestamp_column, self.instrument_column
