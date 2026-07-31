# -*- coding: utf-8 -*-
"""PIT/sequence-safe LQTP logical-source resolver with zero native-path drift."""
from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import pandas as pd
from .data_access_source import MissingDataDependencyError
from .lqtp_logical_source import LQTPLogicalDataSource as _Base

class LQTPLogicalDataSource(_Base):
    @staticmethod
    def _is_source_ref_name(name: str) -> bool:
        from api.source_ref import decode_source_ref
        return decode_source_ref(str(name)) is not None

    def load_column(self, name: str):
        if not self._is_source_ref_name(name):
            return self.inner.load_column(name)
        return super().load_column(name)

    def load_columns(self, names: Iterable[str]):
        names=list(names)
        ordinary=[n for n in names if not self._is_source_ref_name(n)]
        refs=[n for n in names if self._is_source_ref_name(n)]
        out={}
        if ordinary:
            fn=getattr(self.inner,"load_columns",None)
            if callable(fn):out.update(fn(ordinary))
            else:out.update({n:self.inner.load_column(n) for n in ordinary})
        out.update({n:super(LQTPLogicalDataSource,self).load_column(n) for n in refs})
        return out

    def prefetch_columns(self, names: Iterable[str]) -> None:
        names=list(names)
        ordinary=[n for n in names if not self._is_source_ref_name(n)]
        refs=[n for n in names if self._is_source_ref_name(n)]
        if ordinary:
            fn=getattr(self.inner,"prefetch_columns",None)
            if callable(fn):fn(ordinary)
            else:
                load_many=getattr(self.inner,"load_columns",None)
                if callable(load_many):load_many(ordinary)
                else:
                    for n in ordinary:self.inner.load_column(n)
        for n in refs:super(LQTPLogicalDataSource,self).load_column(n)

    def load_column_panel(self, name: str):
        if not self._is_source_ref_name(name):
            fn=getattr(self.inner,"load_column_panel",None)
            if callable(fn):return fn(name)
            return self.inner.load_column(name).unstack(level=-1)
        series=super().load_column(name)
        return series.unstack(level=-1)

    def prefetch_panels(self, names: Iterable[str]) -> None:
        names=list(names)
        ordinary=[n for n in names if not self._is_source_ref_name(n)]
        refs=[n for n in names if self._is_source_ref_name(n)]
        if ordinary:
            fn=getattr(self.inner,"prefetch_panels",None)
            if callable(fn):fn(ordinary)
            else:
                for n in ordinary:self.load_column_panel(n)
        for n in refs:self.load_column_panel(n)

    def scan_polars_long(self, columns: list[str]):
        if any(self._is_source_ref_name(name) for name in columns):
            raise NotImplementedError("SourceRef columns require the logical source/Pandas-Arrow boundary")
        return self.inner.scan_polars_long(columns)

    def scan_index_long(self):
        fn=getattr(self.inner,"scan_index_long",None)
        if not callable(fn):raise NotImplementedError
        return fn()

    def _financial(self,dataset:str,field:str,transform:str|None,params:dict[str,Any])->pd.Series:
        raw=self._financial_raw(dataset,field)
        instrument=next(c for c in ("Symbol","ticker","Ticker") if c in raw.columns)
        events=raw.rename(columns={instrument:"instrument","ReportPeriodEndDate":"period_end","PubDate":"available_at",field:"value"})[["instrument","period_end","available_at","value"]]
        events=events.dropna(subset=["available_at","period_end"]).copy();events["period_end"]=pd.to_datetime(events["period_end"]).dt.normalize();events["available_at"]=pd.to_datetime(events["available_at"])
        if transform=="financial_lag":
            quarters=int(params.get("quarters",1))
            if quarters<=0:raise ValueError("financial_lag quarters must be positive")
            emitted=[]
            for inst,grp in events.sort_values(["available_at","period_end"],kind="stable").groupby("instrument",sort=False):
                visible:dict[pd.Period,tuple[pd.Timestamp,Any]]={}
                for row in grp.itertuples(index=False):
                    q=pd.Timestamp(row.period_end).to_period("Q");visible[q]=(pd.Timestamp(row.period_end),row.value);current_q=max(visible);target=visible.get(current_q-quarters)
                    emitted.append({"instrument":inst,"period_end":visible[current_q][0],"available_at":pd.Timestamp(row.available_at),"value":target[1] if target is not None else np.nan})
            events=pd.DataFrame(emitted).sort_values(["instrument","available_at","period_end"],kind="stable").drop_duplicates(["instrument","available_at"],keep="last")
        elif transform not in {None,"financial_asof"}:raise MissingDataDependencyError(f"unsupported financial transform {transform!r}")
        from pit_contract import PITColumns,pit_asof_join
        anchor=self._anchor_index();decisions=anchor.to_frame(index=False);decisions.columns=["decision_timestamp","instrument"]
        joined=pit_asof_join(decisions,events,columns=PITColumns(),max_age_days=None)
        return pd.Series(joined["value"].to_numpy(),index=anchor,name=field)

    def _load_intermediate(self,spec)->pd.Series:
        try:return super()._load_intermediate(spec)
        except MissingDataDependencyError:
            from runtime.intermediate_registry import ensure_intermediate_materialized
            params=spec.params_dict();ensure_intermediate_materialized(str(params.get("name","")),int(params.get("version",0)),lake_root=self.factor_lake_root)
            return super()._load_intermediate(spec)

    def _load_source_ref(self,spec)->pd.Series:
        if spec.table=="DerivedField":
            from runtime.derived_field_registry import evaluate_derived_field
            series=evaluate_derived_field(spec.field,self.inner)
            if isinstance(series,pd.DataFrame):
                if series.shape[1]!=1:raise MissingDataDependencyError(f"derived field {spec.field!r} returned multiple columns")
                series=series.iloc[:,0]
            if not isinstance(series,pd.Series):raise MissingDataDependencyError(f"derived field {spec.field!r} did not return a Series")
            return self._align_by_instrument(self._anchor_index(),series.rename(spec.field))
        return super()._load_source_ref(spec)

    def _minute_daily(self,field:str,transform:str,params:dict[str,Any])->pd.Series:
        if transform=="minute_resample" and self.factor_freq=="1d":
            raise MissingDataDependencyError("minute_resample returns an intraday bar sequence and cannot be silently collapsed inside a daily factor; use minute_bar(..., index) or configure an intraday run")
        if field.lower().endswith("vwap") and transform in {"minute_range","minute_bar","minute_resample"}:
            raise MissingDataDependencyError("multi-minute VWAP requires Amount/Volume weighted aggregation; refusing an incorrect average/last-fill")
        return super()._minute_daily(field,transform,params)
