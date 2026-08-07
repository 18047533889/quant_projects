# -*- coding: utf-8 -*-
"""PIT/sequence-safe LQTP logical-source resolver with dependency lineage."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_access_source import MissingDataDependencyError
from .lqtp_logical_source import LQTPLogicalDataSource as _Base
from .lqtp_logical_source import _TABLE_DATASETS
from .logical_tables import logical_table_contract


class LQTPLogicalDataSource(_Base):
    prefer_series_panel_loading = True

    def _dependency_store(self) -> dict[str, dict[str, Any]]:
        return self._cache.setdefault("__source_dependencies__", {})

    def _record_dependency(
        self,
        key: str,
        *,
        kind: str,
        snapshot_id: str | None = None,
        **metadata: Any,
    ) -> None:
        row: dict[str, Any] = {"key": str(key), "kind": str(kind), **metadata}
        if snapshot_id:
            row["snapshot_id"] = str(snapshot_id)
        self._dependency_store()[str(key)] = row

    def collect_source_dependencies(self) -> list[dict[str, Any]]:
        """Return deterministic secondary-source dependencies used this execution."""
        return [dict(self._dependency_store()[k]) for k in sorted(self._dependency_store())]

    def source_dependency_hash(self) -> str:
        payload = json.dumps(
            self.collect_source_dependencies(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_source_ref_name(name: str) -> bool:
        from api.source_ref import decode_source_ref
        return decode_source_ref(str(name)) is not None

    def load_column(self, name: str):
        if not self._is_source_ref_name(name):
            return self.inner.load_column(name)
        return super().load_column(name)

    def load_columns(self, names: Iterable[str]):
        names = list(names)
        ordinary = [n for n in names if not self._is_source_ref_name(n)]
        refs = [n for n in names if self._is_source_ref_name(n)]
        out: dict[str, Any] = {}
        if ordinary:
            fn = getattr(self.inner, "load_columns", None)
            if callable(fn):
                out.update(fn(ordinary))
            else:
                out.update({n: self.inner.load_column(n) for n in ordinary})
        if refs:
            out.update(self.load_source_refs_batch(refs))
        return out

    def load_source_refs_batch(self, names: list[str]) -> dict[str, Any]:
        """#10 SourceRef 批量 coalesce：按 (dataset, params, transform) 分组，
        同一物理表的所有字段合并成一次 DataRequest/读取，替代逐 SourceRef 单独
        load_column。返回 {ref_name: MultiIndex Series}。

        分组规则：
            - 财务报表（StockBalance/StockIncome/...）按表合并字段，一次
              ``store.read`` 拿全部列，再逐字段做 PIT join
            - 其余 dataset-mapped 表（IndexConstituent/StockIndustry/...）按
              (dataset, params) 合并，一次 ``child.load_columns`` 拿全部字段
            - DailyBar/Intermediate/TurnoverBaseDaily/MinuteBar/DerivedField
              等特殊路径仍走单字段 ``_load_source_ref``
        """
        from api.source_ref import decode_source_ref

        decoded = [(str(n), decode_source_ref(str(n))) for n in names]
        decoded = [(n, s) for n, s in decoded if s is not None]
        out: dict[str, Any] = {}

        financial_by_table: dict[str, list[tuple[str, Any]]] = {}
        dataset_groups: dict[tuple[str, str, str], list[tuple[str, Any]]] = {}
        singles: list[tuple[str, Any]] = []
        for name, spec in decoded:
            table = spec.table
            if table in {"StockIncome", "StockCashFlow", "StockBalance", "StockIndicator"}:
                financial_by_table.setdefault(table, []).append((name, spec))
            elif table in {
                "DailyBar", "StockDailyBar", "Intermediate",
                "TurnoverBaseDaily", "StockMinuteBar", "MinuteBar", "DerivedField",
            }:
                singles.append((name, spec))
            else:
                dataset = _TABLE_DATASETS.get(table)
                params = spec.params_dict()
                key = (table, str(dataset), repr(sorted(params.items())))
                dataset_groups.setdefault(key, []).append((name, spec))

        # 财务：一表一次 store read（多字段一次 scan）
        for table, group in financial_by_table.items():
            dataset = _TABLE_DATASETS.get(table)
            if dataset is None:
                for name, spec in group:
                    out[name] = self._load_source_ref(spec)
                continue
            fields = list(dict.fromkeys(spec.field for _, spec in group))
            raw = self._financial_raw_multi(dataset, fields)
            for name, spec in group:
                out[name] = self._financial_from_raw(
                    dataset, spec.field, raw,
                    transform=spec.transform,
                    params=spec.transform_params_dict(),
                )

        # dataset-mapped：同 (dataset, params) 一次 child.load_columns
        for (table, dataset, _), group in dataset_groups.items():
            contract = logical_table_contract(table)
            params = dict(group[0][1].params_dict())
            required_param = contract.required_parameter
            if required_param:
                legacy = {"IndexSymbol": "index", "IndustrySource": "industry_source"}.get(required_param)
                legacy_value = params.get(legacy) if legacy else None
                if params.get(required_param) is None and legacy_value is not None:
                    params[required_param] = legacy_value
            filt = None
            if contract.required_parameter == "IndexSymbol" and table != "IndexConstituent":
                filt = [str(params["IndexSymbol"])]
            child = self._child(dataset, instrument_filter=filt)
            fields = list(dict.fromkeys(spec.field for _, spec in group))
            if table == "IndexConstituent":
                fields.append("IndexSymbol")
            elif contract.required_parameter == "IndustrySource":
                fields.append("IndustrySource")
            loaded = child.load_columns(fields)
            snapshot = getattr(child, "data_snapshot_id", None)
            for name, spec in group:
                field = spec.field
                if table == "IndexConstituent":
                    series = loaded[field].where(
                        loaded["IndexSymbol"].astype(str) == str(params["IndexSymbol"])
                    ).dropna()
                elif contract.required_parameter == "IndustrySource":
                    series = loaded[field].where(
                        loaded["IndustrySource"].astype(str) == str(params["IndustrySource"])
                    ).dropna()
                else:
                    series = loaded[field]
                policy = contract.join_policy
                self._record_dependency(
                    dataset,
                    kind=(
                        "benchmark_daily" if policy == "exact_date"
                        else "daily_exact" if policy == "exact"
                        else "dividend_effective_only" if policy == "effective_only"
                        else "shareholder_relation_pit" if policy == "relation_pit"
                        else "relation_asof"
                    ),
                    snapshot_id=snapshot,
                    field=field,
                    join_policy=policy,
                    **({"IndexSymbol": str(params["IndexSymbol"])} if "IndexSymbol" in params else {}),
                    **({"IndustrySource": str(params["IndustrySource"])} if "IndustrySource" in params else {}),
                )
                if policy == "exact_date":
                    out[name] = self._broadcast_exact_by_date(self._anchor_index(), series)
                elif policy in {"exact", "effective_only"}:
                    out[name] = self._align_exact_by_instrument(self._anchor_index(), series)
                else:
                    out[name] = self._align_by_instrument(self._anchor_index(), series)

        # 特殊路径逐个处理
        for name, spec in singles:
            out[name] = self._load_source_ref(spec)
        return out

    def prefetch_columns(self, names: Iterable[str]) -> None:
        names = list(names)
        ordinary = [n for n in names if not self._is_source_ref_name(n)]
        refs = [n for n in names if self._is_source_ref_name(n)]
        if ordinary:
            fn = getattr(self.inner, "prefetch_columns", None)
            if callable(fn):
                fn(ordinary)
            else:
                load_many = getattr(self.inner, "load_columns", None)
                if callable(load_many):
                    load_many(ordinary)
                else:
                    for n in ordinary:
                        self.inner.load_column(n)
        for n in refs:
            super(LQTPLogicalDataSource, self).load_column(n)

    def load_column_panel(self, name: str):
        if not self._is_source_ref_name(name):
            fn = getattr(self.inner, "load_column_panel", None)
            if callable(fn):
                return fn(name)
            return self.inner.load_column(name).unstack(level=-1)
        return super().load_column(name).unstack(level=-1)

    def prefetch_panels(self, names: Iterable[str]) -> None:
        for name in names:
            self.load_column_panel(str(name))

    def scan_polars_long(self, columns: list[str]):
        if any(self._is_source_ref_name(name) for name in columns):
            raise NotImplementedError(
                "SourceRef columns require the certified logical/Pandas-Arrow boundary"
            )
        return self.inner.scan_polars_long(columns)

    def scan_index_long(self):
        fn = getattr(self.inner, "scan_index_long", None)
        if not callable(fn):
            raise NotImplementedError
        return fn()

    @staticmethod
    def _align_exact_by_instrument(anchor: pd.MultiIndex, series: pd.Series) -> pd.Series:
        """Exact date×instrument join; never carry a stale daily observation."""
        s = series.copy()
        if not isinstance(s.index, pd.MultiIndex):
            raise MissingDataDependencyError("exact logical daily source requires MultiIndex")
        s.index = s.index.set_names(["timestamp", "instrument"])
        idx = anchor.set_names(["timestamp", "instrument"])
        out = s.reindex(idx)
        out.index = anchor
        return out

    @staticmethod
    def _broadcast_exact_by_date(anchor: pd.MultiIndex, series: pd.Series) -> pd.Series:
        """Exact benchmark-date broadcast; missing benchmark dates remain missing."""
        right = series.rename("value").reset_index()
        right.columns = ["timestamp", "_source_instrument", "value"]
        right["timestamp"] = pd.to_datetime(right["timestamp"]).dt.normalize()
        values = right.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")["value"]
        ts = pd.DatetimeIndex(anchor.get_level_values(0)).normalize()
        return pd.Series(values.reindex(ts).to_numpy(), index=anchor, name=series.name)

    def _financial_raw(self, dataset: str, field: str) -> pd.DataFrame:
        """Read all pre-end PIT history; start-date truncation is forbidden."""
        return self._financial_raw_multi(dataset, [field])

    def _financial_raw_multi(self, dataset: str, fields: list[str]) -> pd.DataFrame:
        """#10 批量财务读：同一报表表的多个字段一次 ``store.read``（一次 scan）。"""
        from .data_access_source import _get_store

        store = _get_store()
        ds = store.get_dataset(dataset)
        required = [ds.instrument_column, "ReportPeriodEndDate", "PubDate", *fields]
        end = getattr(self.inner, "end_date", None)
        kwargs: dict[str, Any] = {"columns": list(dict.fromkeys(required))}
        if end is not None:
            kwargs["time_range"] = (None, end)
        result = store.read_result(dataset, **kwargs)
        snapshot = getattr(getattr(result, "snapshot", None), "snapshot_id", None)
        for field in fields:
            self._record_dependency(
                dataset,
                kind="financial",
                snapshot_id=snapshot,
                field=field,
                availability_column="PubDate",
                join_policy="asof_backward",
            )
        return result.table.to_pandas()

    def _financial_from_raw(
        self,
        dataset: str,
        field: str,
        raw: pd.DataFrame,
        *,
        transform: str | None,
        params: dict[str, Any],
    ) -> pd.Series:
        """从共享 raw frame 构建单个字段的 PIT 事件 + 对齐到锚点。"""
        instrument = next(c for c in ("Symbol", "ticker", "Ticker") if c in raw.columns)
        events = raw.rename(
            columns={
                instrument: "instrument",
                "ReportPeriodEndDate": "period_end",
                "PubDate": "available_at",
                field: "value",
            }
        )[["instrument", "period_end", "available_at", "value"]]
        events = events.dropna(subset=["available_at", "period_end"]).copy()
        events["period_end"] = pd.to_datetime(events["period_end"]).dt.normalize()
        events["available_at"] = pd.to_datetime(events["available_at"])

        if transform == "financial_lag":
            quarters = int(params.get("quarters", 1))
            if quarters <= 0:
                raise ValueError("financial_lag quarters must be positive")
            emitted: list[dict[str, Any]] = []
            for inst, grp in events.sort_values(
                ["available_at", "period_end"], kind="stable"
            ).groupby("instrument", sort=False):
                visible: dict[pd.Period, tuple[pd.Timestamp, Any]] = {}
                for row in grp.itertuples(index=False):
                    q = pd.Timestamp(row.period_end).to_period("Q")
                    visible[q] = (pd.Timestamp(row.period_end), row.value)
                    current_q = max(visible)
                    target = visible.get(current_q - quarters)
                    emitted.append(
                        {
                            "instrument": inst,
                            "period_end": visible[current_q][0],
                            "available_at": pd.Timestamp(row.available_at),
                            "value": target[1] if target is not None else np.nan,
                        }
                    )
            events = (
                pd.DataFrame(emitted)
                .sort_values(["instrument", "available_at", "period_end"], kind="stable")
                .drop_duplicates(["instrument", "available_at"], keep="last")
            )
        elif transform not in {None, "financial_asof"}:
            raise MissingDataDependencyError(f"unsupported financial transform {transform!r}")

        from pit_contract import PITColumns, pit_asof_join

        anchor = self._anchor_index()
        decisions = anchor.to_frame(index=False)
        decisions.columns = ["decision_timestamp", "instrument"]
        # A-share earnings land after close; a bar dated PubDate must not act on
        # the same-day announcement.  Conservative next_trading_day visibility
        # (audit §2.9): the value is usable from the first decision after PubDate.
        joined = pit_asof_join(
            decisions,
            events,
            columns=PITColumns(),
            max_age_days=None,
            available_policy="next_trading_day",
        )
        return pd.Series(joined["value"].to_numpy(), index=anchor, name=field)

    def _financial(
        self,
        dataset: str,
        field: str,
        transform: str | None,
        params: dict[str, Any],
    ) -> pd.Series:
        """单字段财务 PIT（批量路径的便捷包装）。"""
        raw = self._financial_raw(dataset, field)
        return self._financial_from_raw(
            dataset, field, raw,
            transform=transform,
            params=params,
        )

    def _load_intermediate(self, spec) -> pd.Series:
        from runtime.intermediate_registry import (
            ensure_intermediate_materialized,
            intermediate_dependency_lineage,
        )

        params = spec.params_dict()
        name, version = str(params.get("name", "")), int(params.get("version", 0))
        ensure_intermediate_materialized(
            name,
            version,
            lake_root=self.factor_lake_root,
            require_version_pin=True,
        )
        lineage = intermediate_dependency_lineage(name, version)
        self._record_dependency(
            f"intermediate:{name}@{version}",
            kind="intermediate",
            snapshot_id=lineage["expected_factor_version"],
            **lineage,
        )
        return super()._load_intermediate(spec)

    def _load_source_ref(self, spec) -> pd.Series:
        table, field = spec.table, spec.field
        transform, tparams = spec.transform, spec.transform_params_dict()
        anchor = self._anchor_index()

        if table == "DerivedField":
            from runtime.derived_field_registry import evaluate_derived_field, derived_field_lineage

            lineage = derived_field_lineage(field)
            self._record_dependency(
                f"derived:{field}",
                kind="derived_field",
                snapshot_id=lineage["definition_hash"],
                **lineage,
            )
            series = evaluate_derived_field(field, self.inner)
            if isinstance(series, pd.DataFrame):
                if series.shape[1] != 1:
                    raise MissingDataDependencyError(
                        f"derived field {field!r} returned multiple columns"
                    )
                series = series.iloc[:, 0]
            if not isinstance(series, pd.Series):
                raise MissingDataDependencyError(
                    f"derived field {field!r} did not return a Series"
                )
            return self._align_by_instrument(anchor, series.rename(field))

        if table == "Intermediate":
            return self._align_by_instrument(anchor, self._load_intermediate(spec))

        if table in {"DailyBar", "StockDailyBar"}:
            return self.inner.load_column(field)

        if table == "TurnoverBaseDaily":
            series = self._load_turnover_base(field)
            self._record_dependency(
                "TurnoverBaseDaily",
                kind="daily_exact",
                field=field,
                join_policy="exact",
            )
            return self._align_exact_by_instrument(anchor, series)

        if table in {"StockIncome", "StockCashFlow", "StockBalance", "StockIndicator"}:
            return self._financial(_TABLE_DATASETS[table], field, transform, tparams)

        if table in {"StockMinuteBar", "MinuteBar"}:
            if transform is None:
                raise MissingDataDependencyError(
                    "minute fields require minute_at/range/resample/bar transform"
                )
            return self._minute_daily(field, transform, tparams)

        dataset = _TABLE_DATASETS.get(table)
        if dataset is None:
            raise MissingDataDependencyError(
                f"no FactorEngine dataset mapping for LQTP DataTable {table!r}"
            )
        params = spec.params_dict()
        contract = logical_table_contract(table)
        required_param = contract.required_parameter
        if required_param:
            legacy = {"IndexSymbol": "index", "IndustrySource": "industry_source"}.get(required_param)
            canonical_value = params.get(required_param)
            legacy_value = params.get(legacy) if legacy else None
            if canonical_value is not None and legacy_value is not None and str(canonical_value) != str(legacy_value):
                raise MissingDataDependencyError(
                    f"LQTP DataTable {table!r} has conflicting {required_param} and {legacy}"
                )
            if canonical_value is None:
                if legacy_value is None:
                    raise MissingDataDependencyError(
                        f"LQTP DataTable {table!r} requires exact {required_param} parameter"
                    )
                params[required_param] = legacy_value
        filt = None
        if contract.required_parameter == "IndexSymbol" and table != "IndexConstituent":
            filt = [str(params["IndexSymbol"])]
        child = self._child(dataset, instrument_filter=filt)
        if table == "IndexConstituent":
            loaded = child.load_columns([field, "IndexSymbol"])
            series = loaded[field].where(
                loaded["IndexSymbol"].astype(str) == str(params["IndexSymbol"])
            ).dropna()
        elif contract.required_parameter == "IndustrySource":
            loaded = child.load_columns([field, "IndustrySource"])
            series = loaded[field].where(
                loaded["IndustrySource"].astype(str) == str(params["IndustrySource"])
            ).dropna()
        else:
            series = child.load_column(field)
        snapshot = getattr(child, "data_snapshot_id", None)

        if contract.join_policy == "exact_date":
            self._record_dependency(
                f"{dataset}:{params.get('IndexSymbol','')}",
                kind="benchmark_daily",
                snapshot_id=snapshot,
                field=field,
                join_policy="exact_date",
            )
            return self._broadcast_exact_by_date(anchor, series)
        if contract.join_policy == "exact":
            self._record_dependency(
                dataset,
                kind="daily_exact",
                snapshot_id=snapshot,
                field=field,
                join_policy="exact",
                **({"IndexSymbol": str(params["IndexSymbol"])} if "IndexSymbol" in params else {}),
            )
            return self._align_exact_by_instrument(anchor, series)
        if contract.join_policy == "effective_only":
            self._record_dependency(
                dataset,
                kind="dividend_effective_only",
                snapshot_id=snapshot,
                field=field,
                join_policy="exact",
                effective_only=True,
            )
            return self._align_exact_by_instrument(anchor, series)
        if contract.join_policy == "relation_pit":
            self._record_dependency(
                dataset,
                kind="shareholder_relation_pit",
                snapshot_id=snapshot,
                field=field,
                join_policy="asof_backward",
            )
            return self._align_by_instrument(anchor, series)
        if contract.join_policy == "asof_backward":
            self._record_dependency(
                dataset,
                kind="classification_asof" if contract.required_parameter == "IndustrySource" else "relation_asof",
                snapshot_id=snapshot,
                field=field,
                join_policy="asof_backward",
                **({"IndustrySource": str(params["IndustrySource"])} if "IndustrySource" in params else {}),
            )
            return self._align_by_instrument(anchor, series)

        raise MissingDataDependencyError(
            f"logical table {table!r} has unhandled join policy {contract.join_policy!r}"
        )

    def _anchor_is_intraday(self) -> bool:
        anchor = self._anchor_index()
        if len(anchor) == 0:
            return False
        ts = pd.DatetimeIndex(anchor.get_level_values(0))
        return bool((ts != ts.normalize()).any())

    @staticmethod
    def _minute_semantic_aggregate_frame(grp: pd.DataFrame, field: str) -> float:
        if grp.empty:
            return np.nan
        vals = pd.to_numeric(grp["value"], errors="coerce")
        key = field.lower()
        if key.endswith("open"):
            return float(vals.iloc[0])
        if key.endswith("high"):
            return float(vals.max())
        if key.endswith("low"):
            return float(vals.min())
        if key.endswith("volume") or key.endswith("amount"):
            return float(vals.sum())
        return float(vals.iloc[-1])

    @staticmethod
    def _session_slots(frame: pd.DataFrame) -> pd.DataFrame:
        from runtime.session_calendar import SessionCalendar

        out = frame.copy()
        timestamps = pd.to_datetime(out["timestamp"])
        minute = timestamps.dt.hour * 60 + timestamps.dt.minute
        has_start_labels = bool((minute == 570).any() or (minute == 780).any())
        convention = "bar_start" if has_start_labels else "bar_end"
        calendar = SessionCalendar.ashare(timestamp_convention=convention)
        segment = calendar.segment_id(timestamps)
        slot = calendar.segment_ordinal(timestamps)
        valid = segment >= 0
        out = out.loc[valid].copy()
        out["session"] = np.where(segment[valid] == 0, "am", "pm")
        out["session_slot"] = slot[valid].astype(int)
        return out

    def _minute_weighted_vwap(
        self,
        transform: str,
        params: dict[str, Any],
    ) -> pd.Series:
        """Aggregate multi-minute VWAP from additive Amount and Volume legs."""
        amount = self._minute_daily("Amount", transform, params)
        volume = self._minute_daily("Volume", transform, params)
        output = amount / volume.replace(0, np.nan)
        output.name = "Vwap"
        return output

    def _minute_dispatch(
        self,
        field: str,
        transform: str,
        params: dict[str, Any],
    ) -> pd.Series | None:
        """Explicit extension dispatcher; ``None`` delegates to legacy transforms."""
        if transform == "intraday_feature":
            from .intraday_feature_runtime_v2 import load_intraday_feature

            return load_intraday_feature(self, dict(params))
        return None

    def _minute_daily(self, field: str, transform: str, params: dict[str, Any]) -> pd.Series:
        dispatched = self._minute_dispatch(field, transform, params)
        if dispatched is not None:
            return dispatched

        if transform == "minute_resample" and self.factor_freq == "1d" and not self._anchor_is_intraday():
            raise MissingDataDependencyError(
                "minute_resample returns an intraday bar sequence; use minute_bar(..., index) "
                "for a daily factor or configure an intraday anchor"
            )

        if field.lower().endswith("vwap") and transform in {
            "minute_range", "minute_bar", "minute_resample"
        }:
            return self._minute_weighted_vwap(transform, params)

        # #14 minute_at / minute_range 下沉 DuckDB（不把整份分钟数据拉 Pandas）；
        # 任何失败（数据集未注册等）回退原 pandas 路径。
        if transform in {"minute_at", "minute_range"}:
            try:
                from data_access.read.aggregation import (
                    AggregationSpec,
                    aggregate_minute_to_daily,
                )

                from .data_access_source import _get_store

                spec = AggregationSpec(
                    aggregation=transform,
                    start=str(params["start"]) if "start" in params else None,
                    end=str(params["end"]) if "end" in params else None,
                    hhmm=str(params["hhmm"]) if "hhmm" in params else None,
                )
                table = aggregate_minute_to_daily(
                    _get_store(),
                    "ashare_stock_minute",
                    field,
                    spec,
                    time_range=(getattr(self.inner, "start_date", None),
                                getattr(self.inner, "end_date", None)),
                    instrument_filter=getattr(self.inner, "instrument_filter", None),
                )
                df = table.to_pandas()
                idx = pd.MultiIndex.from_arrays(
                    [pd.to_datetime(df["ts"]), df["inst"]],
                    names=["timestamp", "instrument"],
                )
                daily = pd.Series(df["value"].to_numpy(), index=idx, name=field)
                self._record_dependency(
                    "ashare_stock_minute",
                    kind="minute_session",
                    field=field,
                    transform=transform,
                    join_policy="exact_session",
                    pushdown="duckdb",
                )
                return self._align_by_instrument(self._anchor_index(), daily)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "minute %s pushdown 失败回退 pandas 路径: %s", transform, exc
                )

        src = self._child("ashare_stock_minute")
        series = src.load_column(field)
        self._record_dependency(
            "ashare_stock_minute",
            kind="minute_session",
            snapshot_id=getattr(src, "data_snapshot_id", None),
            field=field,
            transform=transform,
            join_policy="exact_session",
        )
        frame = series.rename("value").reset_index()
        frame.columns = ["timestamp", "instrument", "value"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["date"] = frame["timestamp"].dt.normalize()
        frame["hhmm"] = frame["timestamp"].dt.strftime("%H:%M")

        if transform == "minute_at":
            sub = frame[frame["hhmm"] == str(params["hhmm"])]
            agg = sub.sort_values("timestamp").groupby(["date", "instrument"], sort=False).tail(1)
        elif transform == "minute_range":
            start, end = str(params["start"]), str(params["end"])
            if start >= end:
                raise ValueError("minute_range start must be earlier than end")
            sub = frame[(frame["hhmm"] >= start) & (frame["hhmm"] <= end)]
            rows: list[tuple[Any, ...]] = []
            for (date, inst), grp in sub.groupby(["date", "instrument"], sort=False):
                rows.append((date, inst, self._minute_semantic_aggregate_frame(grp.sort_values("timestamp"), field)))
            agg = pd.DataFrame(rows, columns=["date", "instrument", "value"])
        elif transform in {"minute_bar", "minute_resample"}:
            period = int(params.get("period", 1))
            offset = int(params.get("index", 0))
            if period <= 0 or offset < 0:
                raise ValueError("minute period must be positive and index non-negative")
            slotted = self._session_slots(frame)
            slotted["bar"] = (slotted["session_slot"] // period).astype(int)
            rows: list[tuple[Any, ...]] = []
            for (date, inst, session, bar), grp in slotted.groupby(
                ["date", "instrument", "session", "bar"], sort=False
            ):
                ordered = grp.sort_values("timestamp")
                rows.append((
                    date, inst, session, int(bar), ordered["timestamp"].max(),
                    self._minute_semantic_aggregate_frame(ordered, field),
                ))
            barf = pd.DataFrame(
                rows,
                columns=["date", "instrument", "session", "bar", "bar_timestamp", "value"],
            ).sort_values(["date", "instrument", "bar_timestamp"], kind="stable")

            if transform == "minute_resample":
                if self.factor_freq == "1d" and not self._anchor_is_intraday():
                    raise MissingDataDependencyError(
                        "minute_resample returns an intraday sequence; use minute_bar(..., index) "
                        "for a daily factor or configure an intraday anchor"
                    )
                idx = pd.MultiIndex.from_arrays(
                    [barf["bar_timestamp"], barf["instrument"]],
                    names=["timestamp", "instrument"],
                )
                return pd.Series(barf["value"].to_numpy(), index=idx, name=field).sort_index()

            pieces: list[pd.DataFrame] = []
            for _, grp in barf.groupby(["date", "instrument"], sort=False):
                ordered = grp.sort_values("bar_timestamp")
                pos = len(ordered) - 1 - offset
                if pos >= 0:
                    pieces.append(ordered.iloc[[pos]])
            agg = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
                columns=["date", "instrument", "value"]
            )
        else:
            raise MissingDataDependencyError(f"unsupported minute transform {transform!r}")

        idx = pd.MultiIndex.from_arrays(
            [pd.to_datetime(agg["date"]), agg["instrument"]],
            names=["timestamp", "instrument"],
        )
        daily = pd.Series(agg["value"].to_numpy(), index=idx, name=field)
        return self._align_by_instrument(self._anchor_index(), daily)
