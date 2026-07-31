# -*- coding: utf-8 -*-
"""PIT/sequence-safe LQTP logical-source resolver."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_access_source import MissingDataDependencyError
from .lqtp_logical_source import LQTPLogicalDataSource as _Base


class LQTPLogicalDataSource(_Base):
    prefer_series_panel_loading = True

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
        out.update({n: super(LQTPLogicalDataSource, self).load_column(n) for n in refs})
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

    def _financial_raw(self, dataset: str, field: str) -> pd.DataFrame:
        """Read all pre-end PIT history, never truncate at factor start_date.

        An as-of value on the first requested trading day normally comes from a
        report published before that start date.  A lagged financial transform
        needs even more pre-start quarters. Reading from the beginning of the
        registered financial dataset is the fail-safe production behavior.
        """
        from .data_access_source import _get_store

        store = _get_store()
        ds = store.get_dataset(dataset)
        required = [ds.instrument_column, "ReportPeriodEndDate", "PubDate", field]
        end = getattr(self.inner, "end_date", None)
        kwargs: dict[str, Any] = {"columns": required}
        if end is not None:
            kwargs["time_range"] = (None, end)
        result = store.read_result(dataset, **kwargs)
        return result.table.to_pandas()

    def _financial(
        self,
        dataset: str,
        field: str,
        transform: str | None,
        params: dict[str, Any],
    ) -> pd.Series:
        raw = self._financial_raw(dataset, field)
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
                # visible is updated only when a report has actually become
                # available, so later revisions never leak backward.
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
        joined = pit_asof_join(decisions, events, columns=PITColumns(), max_age_days=None)
        return pd.Series(joined["value"].to_numpy(), index=anchor, name=field)

    def _load_intermediate(self, spec) -> pd.Series:
        """Validate the requested external version before reading any parquet."""
        from runtime.intermediate_registry import ensure_intermediate_materialized

        params = spec.params_dict()
        name = str(params.get("name", ""))
        version = int(params.get("version", 0))
        ensure_intermediate_materialized(
            name,
            version,
            lake_root=self.factor_lake_root,
            require_version_pin=True,
        )
        return super()._load_intermediate(spec)

    def _load_source_ref(self, spec) -> pd.Series:
        if spec.table == "DerivedField":
            from runtime.derived_field_registry import evaluate_derived_field

            series = evaluate_derived_field(spec.field, self.inner)
            if isinstance(series, pd.DataFrame):
                if series.shape[1] != 1:
                    raise MissingDataDependencyError(
                        f"derived field {spec.field!r} returned multiple columns"
                    )
                series = series.iloc[:, 0]
            if not isinstance(series, pd.Series):
                raise MissingDataDependencyError(
                    f"derived field {spec.field!r} did not return a Series"
                )
            return self._align_by_instrument(self._anchor_index(), series.rename(spec.field))
        return super()._load_source_ref(spec)

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
        """Assign collision-free CN regular-session minute slots.

        Both open-labelled (09:30, 13:00) and close-labelled (09:31, 13:01)
        minute datasets are supported. Morning and afternoon are separate
        sessions, so the lunch boundary can never map 11:30 and 13:00 into the
        same bucket.
        """
        out = frame.copy()
        ts = pd.to_datetime(out["timestamp"])
        minute = ts.dt.hour * 60 + ts.dt.minute
        has_open_labels = bool((minute == 570).any() or (minute == 780).any())
        label_offset = 0 if has_open_labels else 1

        am = (minute >= 570 + label_offset) & (minute <= 690)
        pm = (minute >= 780 + label_offset) & (minute <= 900)
        valid = am | pm
        out = out.loc[valid].copy()
        minute = minute.loc[valid]
        out["session"] = np.where(am.loc[valid], "am", "pm")
        out["session_slot"] = np.where(
            am.loc[valid],
            minute - (570 + label_offset),
            minute - (780 + label_offset),
        ).astype(int)
        return out

    def _minute_daily(self, field: str, transform: str, params: dict[str, Any]) -> pd.Series:
        if field.lower().endswith("vwap") and transform in {
            "minute_range", "minute_bar", "minute_resample"
        }:
            raise MissingDataDependencyError(
                "multi-minute VWAP requires Amount/Volume weighted aggregation"
            )

        src = self._child("ashare_stock_minute")
        series = src.load_column(field)
        frame = series.rename("value").reset_index()
        frame.columns = ["timestamp", "instrument", "value"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["date"] = frame["timestamp"].dt.normalize()
        frame["hhmm"] = frame["timestamp"].dt.strftime("%H:%M")

        if transform == "minute_at":
            sub = frame[frame["hhmm"] == str(params["hhmm"])]
            agg = (
                sub.sort_values("timestamp")
                .groupby(["date", "instrument"], sort=False)
                .tail(1)
                .rename(columns={"timestamp": "bar_timestamp"})
            )
        elif transform == "minute_range":
            start, end = str(params["start"]), str(params["end"])
            if start >= end:
                raise ValueError("minute_range start must be earlier than end")
            sub = frame[(frame["hhmm"] >= start) & (frame["hhmm"] <= end)]
            rows: list[tuple[Any, ...]] = []
            for (date, inst), grp in sub.groupby(["date", "instrument"], sort=False):
                rows.append(
                    (
                        date,
                        inst,
                        grp["timestamp"].max(),
                        self._minute_semantic_aggregate_frame(grp.sort_values("timestamp"), field),
                    )
                )
            agg = pd.DataFrame(
                rows,
                columns=["date", "instrument", "bar_timestamp", "value"],
            )
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
                rows.append(
                    (
                        date,
                        inst,
                        session,
                        int(bar),
                        ordered["timestamp"].max(),
                        self._minute_semantic_aggregate_frame(ordered, field),
                    )
                )
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

            # minute_bar(period, index): nth completed bar from the end of each
            # trading day, ordered by the real exchange timestamp across lunch.
            pieces: list[pd.DataFrame] = []
            for _, grp in barf.groupby(["date", "instrument"], sort=False):
                ordered = grp.sort_values("bar_timestamp")
                pos = len(ordered) - 1 - offset
                if pos >= 0:
                    pieces.append(ordered.iloc[[pos]])
            agg = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
                columns=["date", "instrument", "bar_timestamp", "value"]
            )
        else:
            raise MissingDataDependencyError(f"unsupported minute transform {transform!r}")

        idx = pd.MultiIndex.from_arrays(
            [pd.to_datetime(agg["date"]), agg["instrument"]],
            names=["timestamp", "instrument"],
        )
        daily = pd.Series(agg["value"].to_numpy(), index=idx, name=field)
        return self._align_by_instrument(self._anchor_index(), daily)
