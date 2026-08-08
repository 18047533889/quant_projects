# -*- coding: utf-8 -*-
"""Runtime resolver for opaque LQTP logical DataTable references.

Phase 5 P1-8：本模块是 **legacy/base**——生产 canonical resolver 是
``lqtp_logical_source_v2.LQTPLogicalDataSource``（``ExecutionContext`` 安装时优先
v2，覆盖 financial/minute/batch）。本类保留 alignment / anchor / 单字段 PIT 等
共享基实现，供 v2 继承；不再新增业务路径。
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .datasource import DataSource
from .data_access_source import DataAccessSource, MissingDataDependencyError
from .logical_tables import TABLE_DATASETS as _TABLE_DATASETS


class LQTPLogicalDataSource(DataSource):
    """Decorate an existing daily source with LQTP DataTable resolution."""
    def __init__(self, inner: DataSource, *, factor_freq: str = "1d",
                 factor_lake_root: str | None = None) -> None:
        self.inner = inner
        self.factor_freq = str(factor_freq or "1d")
        self.factor_lake_root = factor_lake_root
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _anchor_index(self) -> pd.MultiIndex:
        cached = self._cache.get("__anchor_index__")
        if cached is not None:
            return cached
        for candidate in ("close", "Close", "volume", "Volume", "open", "Open"):
            try:
                series = self.inner.load_column(candidate)
                if isinstance(series.index, pd.MultiIndex):
                    self._cache["__anchor_index__"] = series.index
                    return series.index
            except MissingDataDependencyError:
                # 该列缺失：尝试下一个候选（capability 类，允许继续）
                continue
            except (KeyError, ValueError, FileNotFoundError):
                # 列不存在 / 数据不可用：继续尝试下一个候选
                continue
        if hasattr(self.inner, "scan_index_long"):
            frame = self.inner.scan_index_long()
            if hasattr(frame, "collect"):
                frame = frame.collect()
            if hasattr(frame, "to_pandas"):
                frame = frame.to_pandas()
            cols = list(frame.columns)
            if len(cols) >= 2:
                idx = pd.MultiIndex.from_frame(frame[cols[:2]], names=["timestamp", "instrument"])
                self._cache["__anchor_index__"] = idx
                return idx
        raise MissingDataDependencyError("cannot resolve LQTP logical source without an anchor panel index")

    def _child(self, dataset: str, *, instrument_filter: list[str] | None = None) -> DataAccessSource:
        start = getattr(self.inner, "start_date", None)
        end = getattr(self.inner, "end_date", None)
        return DataAccessSource(dataset=dataset, start_date=start, end_date=end,
                                instrument_filter=instrument_filter)

    @staticmethod
    def _align_by_instrument(anchor: pd.MultiIndex, series: pd.Series) -> pd.Series:
        if series.index.equals(anchor):
            return series
        left = anchor.to_frame(index=False); left.columns = ["timestamp", "instrument"]
        left["_row"] = np.arange(len(left))
        right = series.rename("value").reset_index(); right.columns = ["timestamp", "instrument", "value"]
        left["timestamp"] = pd.to_datetime(left["timestamp"])
        right["timestamp"] = pd.to_datetime(right["timestamp"])
        merged = pd.merge_asof(left.sort_values(["timestamp","instrument"]),
                               right.sort_values(["timestamp","instrument"]),
                               on="timestamp", by="instrument", direction="backward",
                               allow_exact_matches=True).sort_values("_row")
        return pd.Series(merged["value"].to_numpy(), index=anchor, name=series.name)

    @staticmethod
    def _broadcast_by_date(anchor: pd.MultiIndex, series: pd.Series) -> pd.Series:
        left = anchor.to_frame(index=False); left.columns = ["timestamp", "instrument"]
        left["_row"] = np.arange(len(left)); left["timestamp"] = pd.to_datetime(left["timestamp"])
        right = series.rename("value").reset_index()
        right.columns = ["timestamp", "_source_instrument", "value"]
        right["timestamp"] = pd.to_datetime(right["timestamp"])
        right = right.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        merged = pd.merge_asof(left.sort_values("timestamp"), right[["timestamp","value"]],
                               on="timestamp", direction="backward", allow_exact_matches=True).sort_values("_row")
        return pd.Series(merged["value"].to_numpy(), index=anchor, name=series.name)

    def _load_turnover_base(self, field: str) -> pd.Series:
        # #11 走 DataAccess 登记数据集（ashare_turnover_base_daily），不再直接
        # ParquetSource 裸读 LQTP 镜像。
        from .data_access_source import _get_store

        store = _get_store()
        ds = store.get_dataset("ashare_turnover_base_daily")
        physical = "EffectiveFloatShares" if field == "TurnoverBase" else field
        handle = store.read(
            "ashare_turnover_base_daily",
            columns=[ds.instrument_column, physical],
            time_range=(getattr(self.inner, "start_date", None),
                        getattr(self.inner, "end_date", None)),
            instrument_filter=getattr(self.inner, "instrument_filter", None),
        )
        frame = handle.to_arrow().to_pandas()
        instrument = next(
            (c for c in ("Symbol", "ticker", "Ticker") if c in frame.columns),
            ds.instrument_column,
        )
        ts = pd.to_datetime(frame[ds.time_column])
        idx = pd.MultiIndex.from_arrays([ts, frame[instrument]], names=["timestamp", "instrument"])
        return pd.Series(frame[physical].to_numpy(), index=idx, name=field)

    def _load_intermediate(self, spec) -> pd.Series:
        # #11 走 DataAccess 因子湖（factor_lake 参数化数据集），不再 pd.read_parquet。
        from .data_access_source import _get_store

        store = _get_store()
        factor_id = str(spec.params_dict().get("name", ""))
        version = int(spec.params_dict().get("version", 0))
        if not factor_id or version <= 0:
            raise MissingDataDependencyError("invalid intermediate(name, version) reference")
        ds = store.get_dataset("factor_lake")
        handle = store.read(
            "factor_lake",
            columns=[ds.time_column, ds.instrument_column, "value"],
            time_range=(getattr(self.inner, "start_date", None),
                        getattr(self.inner, "end_date", None)),
            instrument_filter=getattr(self.inner, "instrument_filter", None),
            factor_id=factor_id,
        )
        frame = handle.to_arrow().to_pandas()
        if frame.empty:
            raise MissingDataDependencyError(
                f"intermediate {factor_id!r} version={version} is not materialized; "
                "backfill it before downstream execution"
            )
        idx = pd.MultiIndex.from_frame(
            frame[[ds.time_column, ds.instrument_column]],
            names=["timestamp", "instrument"],
        )
        return pd.Series(frame["value"].to_numpy(), index=idx, name=factor_id)

    def _financial_raw(self, dataset: str, field: str) -> pd.DataFrame:
        from .data_access_source import _get_store
        store = _get_store(); ds = store.get_dataset(dataset)
        required = [ds.instrument_column, "ReportPeriodEndDate", "PubDate", field]
        result = store.read_result(dataset, columns=required,
                                   time_range=(getattr(self.inner,"start_date",None), getattr(self.inner,"end_date",None)))
        return result.table.to_pandas()

    def _financial(self, dataset: str, field: str, transform: str | None, params: dict[str, Any]) -> pd.Series:
        raw = self._financial_raw(dataset, field)
        instrument = next(c for c in ("Symbol","ticker","Ticker") if c in raw.columns)
        events = raw.rename(columns={instrument:"instrument", "ReportPeriodEndDate":"period_end",
                                    "PubDate":"available_at", field:"value"})
        events = events[["instrument","period_end","available_at","value"]].dropna(subset=["available_at","period_end"])
        if transform == "financial_lag":
            quarters = int(params.get("quarters", 1))
            if quarters <= 0:
                raise ValueError("financial_lag quarters must be positive")
            emitted: list[dict[str, Any]] = []
            for inst, grp in events.sort_values("available_at").groupby("instrument", sort=False):
                visible: dict[pd.Timestamp, Any] = {}
                for row in grp.itertuples(index=False):
                    period = pd.Timestamp(row.period_end)
                    visible[period] = row.value
                    periods = sorted(visible)
                    pos = periods.index(period) - quarters
                    value = visible[periods[pos]] if pos >= 0 else np.nan
                    emitted.append({"instrument":inst,"period_end":period,
                                    "available_at":row.available_at,"value":value})
            events = pd.DataFrame(emitted)
        from pit_contract import PITColumns, pit_asof_join
        anchor = self._anchor_index()
        decisions = anchor.to_frame(index=False); decisions.columns = ["decision_timestamp","instrument"]
        # Conservative next_trading_day visibility (audit §2.9): a same-day
        # announcement must not inform a bar dated PubDate.
        joined = pit_asof_join(
            decisions,
            events,
            columns=PITColumns(),
            max_age_days=None,
            available_policy="next_trading_day",
        )
        return pd.Series(joined["value"].to_numpy(), index=anchor, name=field)

    @staticmethod
    def _minute_semantic_aggregate(frame: pd.DataFrame, field: str) -> float:
        if frame.empty: return np.nan
        key = field.lower()
        vals = pd.to_numeric(frame["value"], errors="coerce")
        if key.endswith("open"): return float(vals.iloc[0])
        if key.endswith("high"): return float(vals.max())
        if key.endswith("low"): return float(vals.min())
        if key.endswith("volume") or key.endswith("amount"): return float(vals.sum())
        return float(vals.iloc[-1])

    def _minute_daily(self, field: str, transform: str, params: dict[str, Any]) -> pd.Series:
        src = self._child("ashare_stock_minute")
        series = src.load_column(field)
        frame = series.rename("value").reset_index(); frame.columns = ["timestamp","instrument","value"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["date"] = frame["timestamp"].dt.normalize()
        frame["hhmm"] = frame["timestamp"].dt.strftime("%H:%M")
        if transform == "minute_at":
            frame = frame[frame["hhmm"] == str(params["hhmm"])]
            agg = frame.sort_values("timestamp").groupby(["date","instrument"], sort=False).tail(1)
        elif transform == "minute_range":
            start, end = str(params["start"]), str(params["end"])
            if start >= end: raise ValueError("minute_range start must be earlier than end")
            sub = frame[(frame["hhmm"] >= start) & (frame["hhmm"] <= end)]
            rows=[]
            for (date,inst), grp in sub.groupby(["date","instrument"], sort=False):
                rows.append((date,inst,self._minute_semantic_aggregate(grp, field)))
            agg = pd.DataFrame(rows, columns=["date","instrument","value"])
        elif transform in {"minute_bar","minute_resample"}:
            period = int(params.get("period", 1)); offset = int(params.get("index", 0))
            if period <= 0 or offset < 0: raise ValueError("minute period must be positive and index non-negative")
            minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
            morning = (minute >= 570) & (minute <= 690); afternoon = (minute >= 780) & (minute <= 900)
            elapsed = pd.Series(np.nan, index=frame.index)
            elapsed.loc[morning] = minute.loc[morning] - 570
            elapsed.loc[afternoon] = 120 + minute.loc[afternoon] - 780
            frame = frame[elapsed.notna()].copy(); frame["bar"] = (elapsed[elapsed.notna()] // period).astype(int)
            bars=[]
            for (date,inst,bar), grp in frame.groupby(["date","instrument","bar"], sort=False):
                bars.append((date,inst,bar,self._minute_semantic_aggregate(grp, field)))
            barf = pd.DataFrame(bars, columns=["date","instrument","bar","value"])
            if transform == "minute_resample" and self.factor_freq != "1d":
                idx = pd.MultiIndex.from_arrays([barf["date"] + pd.to_timedelta(barf["bar"]*period, unit="m"), barf["instrument"]], names=["timestamp","instrument"])
                return pd.Series(barf["value"].to_numpy(), index=idx, name=field).sort_index()
            chosen = barf.sort_values("bar").groupby(["date","instrument"], sort=False).nth(-(offset+1)).reset_index()
            agg = chosen.rename(columns={"date":"date"})
        else:
            raise MissingDataDependencyError(f"unsupported minute transform {transform!r}")
        idx = pd.MultiIndex.from_arrays([pd.to_datetime(agg["date"]), agg["instrument"]], names=["timestamp","instrument"])
        daily = pd.Series(agg["value"].to_numpy(), index=idx, name=field)
        return self._align_by_instrument(self._anchor_index(), daily)

    def _load_source_ref(self, spec) -> pd.Series:
        table, field = spec.table, spec.field
        transform, tparams = spec.transform, spec.transform_params_dict()
        if table in {"DailyBar","StockDailyBar"}:
            return self.inner.load_column(field)
        if table == "Intermediate":
            return self._align_by_instrument(self._anchor_index(), self._load_intermediate(spec))
        if table == "TurnoverBaseDaily":
            return self._align_by_instrument(self._anchor_index(), self._load_turnover_base(field))
        if table in {"StockIncome","StockCashFlow","StockBalance"}:
            dataset = _TABLE_DATASETS[table]
            return self._financial(dataset, field, transform, tparams)
        if table in {"StockMinuteBar","MinuteBar"}:
            if transform is None:
                raise MissingDataDependencyError("minute fields require minute_at/range/resample/bar transform in a daily factor")
            return self._minute_daily(field, transform, tparams)
        dataset = _TABLE_DATASETS.get(table)
        if dataset is None:
            raise MissingDataDependencyError(f"no FactorEngine dataset mapping for LQTP DataTable {table!r}")
        params = spec.params_dict()
        filt = [str(params["index"])] if table == "BenchmarkIndexDailyBar" and "index" in params else None
        child = self._child(dataset, instrument_filter=filt)
        series = child.load_column(field)
        if table == "BenchmarkIndexDailyBar":
            return self._broadcast_by_date(self._anchor_index(), series)
        return self._align_by_instrument(self._anchor_index(), series)

    def load_column(self, name: str):
        from api.source_ref import decode_source_ref
        spec = decode_source_ref(name)
        if spec is None:
            return self.inner.load_column(name)
        if name not in self._cache:
            self._cache[name] = self._load_source_ref(spec)
        return self._cache[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        return {name:self.load_column(name) for name in names}

    def prefetch_columns(self, names: list[str]) -> None:
        for name in names: self.load_column(name)

    def prefetch_panels(self, names: list[str]) -> None:
        for name in names: self.load_column_panel(name)

    def load_column_panel(self, name: str):
        series = self.load_column(name)
        return series.unstack(level=series.index.names[-1] or "instrument")

    def scan_polars_long(self, columns: list[str]):
        from api.source_ref import decode_source_ref
        if any(decode_source_ref(name) is not None for name in columns):
            raise NotImplementedError("SourceRef columns currently execute through the certified Pandas/Arrow path")
        return self.inner.scan_polars_long(columns)
