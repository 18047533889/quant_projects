# -*- coding: utf-8 -*-
"""PIT/sequence-safe overrides for the LQTP logical-source resolver."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_access_source import MissingDataDependencyError
from .lqtp_logical_source import LQTPLogicalDataSource as _Base


class LQTPLogicalDataSource(_Base):
    """Production-safe v2 logical source resolver."""

    def _financial(self, dataset: str, field: str, transform: str | None,
                   params: dict[str, Any]) -> pd.Series:
        raw = self._financial_raw(dataset, field)
        instrument = next(c for c in ("Symbol", "ticker", "Ticker") if c in raw.columns)
        events = raw.rename(columns={
            instrument: "instrument",
            "ReportPeriodEndDate": "period_end",
            "PubDate": "available_at",
            field: "value",
        })[["instrument", "period_end", "available_at", "value"]]
        events = events.dropna(subset=["available_at", "period_end"]).copy()
        events["period_end"] = pd.to_datetime(events["period_end"]).dt.normalize()
        events["available_at"] = pd.to_datetime(events["available_at"])

        if transform == "financial_lag":
            quarters = int(params.get("quarters", 1))
            if quarters <= 0:
                raise ValueError("financial_lag quarters must be positive")
            emitted: list[dict[str, Any]] = []
            for inst, grp in events.sort_values(["available_at", "period_end"], kind="stable").groupby("instrument", sort=False):
                visible: dict[pd.Period, tuple[pd.Timestamp, Any]] = {}
                for row in grp.itertuples(index=False):
                    q = pd.Timestamp(row.period_end).to_period("Q")
                    visible[q] = (pd.Timestamp(row.period_end), row.value)
                    current_q = max(visible)
                    target_q = current_q - quarters
                    target = visible.get(target_q)
                    emitted.append({
                        "instrument": inst,
                        "period_end": visible[current_q][0],
                        "available_at": pd.Timestamp(row.available_at),
                        "value": target[1] if target is not None else np.nan,
                    })
            events = pd.DataFrame(emitted)
            # Multiple filings can arrive at the same timestamp. The final
            # state after processing all same-timestamp revisions is the only
            # state visible immediately after that timestamp.
            events = events.sort_values(["instrument", "available_at", "period_end"], kind="stable")
            events = events.drop_duplicates(["instrument", "available_at"], keep="last")
        elif transform not in {None, "financial_asof"}:
            raise MissingDataDependencyError(f"unsupported financial transform {transform!r}")

        from pit_contract import PITColumns, pit_asof_join
        anchor = self._anchor_index()
        decisions = anchor.to_frame(index=False)
        decisions.columns = ["decision_timestamp", "instrument"]
        joined = pit_asof_join(decisions, events, columns=PITColumns(), max_age_days=None)
        return pd.Series(joined["value"].to_numpy(), index=anchor, name=field)

    def _load_intermediate(self, spec) -> pd.Series:
        try:
            return super()._load_intermediate(spec)
        except MissingDataDependencyError:
            from runtime.intermediate_registry import ensure_intermediate_materialized
            params = spec.params_dict()
            ensure_intermediate_materialized(
                str(params.get("name", "")),
                int(params.get("version", 0)),
                lake_root=self.factor_lake_root,
            )
            return super()._load_intermediate(spec)

    def _minute_daily(self, field: str, transform: str,
                      params: dict[str, Any]) -> pd.Series:
        if transform == "minute_resample" and self.factor_freq == "1d":
            raise MissingDataDependencyError(
                "minute_resample returns an intraday bar sequence and cannot be silently collapsed "
                "inside a daily factor; use minute_bar(..., index) for a daily scalar or configure "
                "an intraday FactorEngine run"
            )
        if field.lower().endswith("vwap") and transform in {"minute_range", "minute_bar", "minute_resample"}:
            raise MissingDataDependencyError(
                "multi-minute VWAP requires Amount/Volume weighted aggregation; the current logical "
                "source refuses to average/last-fill minute VWAP values"
            )
        return super()._minute_daily(field, transform, params)
