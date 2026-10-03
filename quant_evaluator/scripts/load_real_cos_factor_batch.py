#!/usr/bin/env python3
"""Read two verified COS factor values and registered AdjVwap labels for QE research."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_access import get_store
from data_access.core.engine import DuckDBEngine
from data_access.core.storage import StorageSpec
from data_access.cos.research import read_declared_cos_object
from data_access.read.formats import FormatSpec
from data_access.read.query_budget import QueryBudget
from data_access.registry.loader import DatasetRegistry, StaticDataset
from data_access.store import DataAccessStore
from factor_optimizer.research_manifest import read_bound_factor
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

ROOT = Path("/home/sunhaiwei/quant_projects")
MIRROR = Path("/home/sunhaiwei/cos_data/StockDailyBarAdj")
SHA = "00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66"
BASE = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show"
POOL = f"{BASE}/factor_values"

def _ds(name, uri, filename, fmt):
    return StaticDataset(name=name, access_mode="published", layout="plain",
        time_column=None, instrument_column=None, hive_partitioning=False,
        union_by_name=True, root=ROOT/"workspace_data/research_factor_panel"/name,
        glob=filename, format_spec=FormatSpec.from_yaml(fmt),
        storage=StorageSpec(type="cos", uri=uri, layout="plain"))

def _iter_factors(count, max_mib, manifest_sha256, max_total_mib):
    if (type(count) is not int or type(max_mib) is not int
            or not 1 <= count <= 48 or not 1 <= max_mib <= 128):
        raise ValueError("factor count 1..48, object limit 1..128 MiB")
    if type(max_total_mib) is not int or not 1 <= max_total_mib <= 2048:
        raise ValueError("total verified object limit must be 1..2048 MiB")
    if len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA256 digest")
    engine = DuckDBEngine(threads=2)
    manifest_uri = f"{BASE}/metadata/{manifest_sha256}/landing_manifest.json"
    md = _ds("source_manifest", manifest_uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    try:
        store = DataAccessStore(DatasetRegistry({md.name: md}), engine)
        manifest = read_declared_cos_object(store, md.name, allow_research=True)
        if manifest.content_sha256 != manifest_sha256:
            raise ValueError("landing manifest identity mismatch")
        rows = manifest.table.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
            raise ValueError("invalid landing manifest")
        selected = []
        for name, record in sorted(rows[0]["factors"].items()):
            if not isinstance(record, dict) or record.get("verified") is not True:
                continue
            if record.get("status") not in {"materialized_not_evaluated", "evaluated_optimization_pending"}:
                continue
            size, digest = record.get("bytes"), record.get("sha256")
            if type(size) is not int or not 0 < size <= max_mib*1024**2:
                continue
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid factor digest")
            if record.get("uri") != f"{POOL}/{digest}/{name}.parquet":
                raise ValueError("factor URI is not manifest bound")
            selected.append((name, record))
        # Prefer the smallest independently verified objects so larger
        # research tiles fit the explicit bounded transfer budget.
        selected = sorted(selected, key=lambda item: (item[1]["bytes"], item[0]))[:count]
        if len(selected) != count or sum(r["bytes"] for _, r in selected) > max_total_mib*1024**2:
            raise ValueError("bounded verified sample unavailable")
        if len({record["sha256"] for _, record in selected}) != count:
            raise ValueError("selected factors do not have unique content hashes")
        # Stream verified panels so their data does not overlap with np.stack.
        for name, record in selected:
            fd = _ds("factor_panel", record["uri"].rsplit("/", 1)[0], name+".parquet", "parquet")
            fs = DataAccessStore(DatasetRegistry({md.name: md, fd.name: fd}), engine)
            bound = read_bound_factor(fs, md.name, fd.name, name,
                                      max_object_mib=max_mib, allow_research=True)
            obj = bound.factor
            if obj.content_sha256 != record["sha256"] or obj.downloaded_bytes != record["bytes"]:
                raise ValueError("factor object identity mismatch")
            frame = obj.table.to_pandas()
            if "timestamp" not in frame.columns:
                raise ValueError("factor timestamp missing")
            frame = frame.set_index("timestamp")
            frame.index = pd.to_datetime(frame.index).normalize()
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError("duplicate factor axis")
            frame = frame.loc[:, [c for c in frame if c.endswith((".SZ", ".SH"))]].sort_index()
            source = {"factor_id": name, "uri": obj.source_uri, "sha256": obj.content_sha256,
                      "bytes": obj.downloaded_bytes, "etag": obj.source_etag,
                      "manifest_sha256": bound.manifest_sha256,
                      "source_status": bound.source_status}
            del bound, obj, fs, fd
            yield frame, source
            del frame, source
    finally:
        engine.close()


def _factors(count, max_mib, manifest_sha256, max_total_mib):
    """Compatibility helper for callers that explicitly need all panels."""
    loaded = list(_iter_factors(count, max_mib, manifest_sha256, max_total_mib))
    return [frame for frame, _ in loaded], [source for _, source in loaded]


def _assemble_factor_values(panel_stream, factor_count):
    """Align panels into one array while retaining at most one pandas panel."""
    iterator = iter(panel_stream)
    try:
        first, first_source = next(iterator)
    except StopIteration as exc:
        raise ValueError("no verified factor panels") from exc
    base_dates, base_assets = first.index, first.columns
    values = np.empty((len(base_dates), len(base_assets), factor_count), dtype=np.float64)
    common_dates = base_dates
    common_assets = set(base_assets)
    sources = []
    frame = first
    del first
    for position in range(factor_count):
        if position:
            try:
                frame, source = next(iterator)
            except StopIteration as exc:
                raise ValueError("verified factor panel count changed") from exc
            common_dates = common_dates.intersection(frame.index)
            common_assets.intersection_update(frame.columns)
        aligned = frame.reindex(index=base_dates, columns=base_assets).to_numpy(dtype=np.float64, copy=False)
        values[:, :, position] = aligned
        sources.append(first_source if position == 0 else source)
        del aligned, frame
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("verified factor panel count changed")
    return values, common_dates, common_assets, sources, base_dates, base_assets


def _trim_factor_values_in_place(values, base_dates, base_assets, dates, assets):
    """Compact a factor cube onto selected axes without allocating another cube.

    Time selections must preserve source order. Asset selections may reorder
    freely; one selected time row is staged at a time while the cube is
    compacted into its own leading bytes. The returned view is contiguous and
    aliases the writable input buffer. FactorBatch takes its normal immutable
    ownership copy when constructed.
    """
    if (not isinstance(values, np.ndarray) or values.ndim != 3
            or values.dtype != np.float64 or not values.flags.c_contiguous
            or not values.flags.writeable):
        raise ValueError("factor values must be a writable C-contiguous float64 cube")
    if values.shape[:2] != (len(base_dates), len(base_assets)):
        raise ValueError("factor values do not match source axes")
    time_positions = base_dates.get_indexer(dates)
    asset_positions = base_assets.get_indexer(assets)
    if (np.any(time_positions < 0) or np.any(asset_positions < 0)
            or len(np.unique(time_positions)) != len(time_positions)
            or len(np.unique(asset_positions)) != len(asset_positions)
            or (len(time_positions) > 1 and np.any(np.diff(time_positions) <= 0))):
        raise ValueError("selected factor axes are missing, duplicated, or out of source order")

    selected_times, selected_assets, factor_count = (
        len(time_positions), len(asset_positions), values.shape[2]
    )
    flat = values.reshape(-1)
    row = np.empty((selected_assets, factor_count), dtype=np.float64)
    row_size = selected_assets * factor_count
    for output_time, source_time in enumerate(time_positions):
        np.take(values[source_time], asset_positions, axis=0, out=row)
        start = output_time * row_size
        flat[start:start + row_size] = row.reshape(-1)
    return flat[:selected_times * row_size].reshape(
        selected_times, selected_assets, factor_count
    )


def load_real_batch(*, factors=2, days=500, assets=5500, max_object_mib=8, max_total_mib=128,
                    manifest_sha256=SHA):
    """Decision t, execution t+1, label AdjVwap(t+2)/AdjVwap(t+1)-1."""
    if not 0 <= days <= 3000 or not 1 <= assets <= 5500:
        raise ValueError("days 0 (all) or 1..3000, assets 1..5500")
    values, common_dates, common_assets, sources, base_dates, base_assets = _assemble_factor_values(
        _iter_factors(factors, max_object_mib, manifest_sha256, max_total_mib), factors
    )
    if common_dates.empty:
        raise ValueError("no shared factor dates")
    store = get_store()
    calendar = store.read("ashare_calendar", columns=["TradeDate", "IsTradeDay"],
        time_range=(common_dates.min(), common_dates.max()), result="arrow",
        query_budget=QueryBudget(max_scan_files=1, max_rows=6000,
                                 max_result_bytes=2*1024**2)).to_pandas()
    trading = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[calendar["IsTradeDay"].eq(True), "TradeDate"])).normalize().sort_values()
    if trading.empty or trading.has_duplicates:
        raise ValueError("registered trading calendar invalid")
    local_days = {pd.Timestamp(p.stem) for p in MIRROR.glob("*.parquet") if len(p.stem) == 10}
    missing = [str(d.date()) for d in trading if d not in local_days]
    shared = set(common_dates)
    positions = [i for i in range(len(trading)-2)
                 if trading[i] in shared and all(trading[i+j] in local_days for j in (0,1,2))]
    positions = positions[-days:] if days else positions
    if not positions or (days and len(positions) != days):
        raise ValueError(f"only {len(positions)} complete calendar/factor/price triplets")
    decision = trading[positions]
    names = sorted(common_assets)[:assets]
    if len(names) < 30:
        raise ValueError("fewer than 30 common A-share assets")
    start, end = trading[min(positions)], trading[max(positions)+2]
    # read_root is DataAccess's registered-dataset relocation. It prevents
    # mirror sync of known-absent objects while retaining the dataset contract.
    rows = store.read("ashare_stock_daily_adj",
        columns=["TradeDate","Symbol","AdjVwap"], time_range=(start,end),
        instrument_filter=names, read_root=str(MIRROR), result="arrow",
        query_budget=QueryBudget(max_scan_files=min(4000,(end-start).days+2),
            max_rows=20_000_000,max_result_bytes=1024**3)).to_pandas()
    if rows.duplicated(["TradeDate","Symbol"]).any():
        raise ValueError("duplicate AdjVwap observation")
    prices = rows.pivot(index="TradeDate",columns="Symbol",values="AdjVwap")
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.reindex(index=trading,columns=names)
    p1 = prices.iloc[[i+1 for i in positions]].to_numpy(dtype=np.float64)
    p2 = prices.iloc[[i+2 for i in positions]].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        labels_array = p2/p1-1
    labels_array[~np.isfinite(labels_array)] = np.nan
    values = _trim_factor_values_in_place(values, base_dates, base_assets, decision, names)
    times = decision.to_numpy(dtype="datetime64[ns]")
    time_axis = AxisRef("time","datetime64[ns]",len(times),times)
    asset_axis = AxisRef("asset","str",len(names),np.asarray(names,dtype=str))
    batch = FactorBatch(tuple(s["factor_id"] for s in sources),time_axis,asset_axis,
                        np.ascontiguousarray(values),validity=np.isfinite(values))
    t1 = tuple(trading[[i+1 for i in positions]].to_numpy())
    t2 = tuple(trading[[i+2 for i in positions]].to_numpy())
    labels = LabelBundle("adj_vwap_tplus1_to_tplus2_return",
        np.ascontiguousarray(labels_array),1,execution_delay=1,
        decision_time=tuple(times),observation_time=tuple(times),
        signal_available_time=tuple(times),execution_time=t1,
        label_start_time=t1,label_end_time=t2,validity=np.isfinite(labels_array),
        asset_axis=asset_axis,source_ref="data_access:ashare_stock_daily_adj:AdjVwap",
        calendar_ref="data_access:ashare_calendar")
    provenance = {"sources":sources,"days":len(times),"assets":len(names),
        "date_span":[str(decision.min().date()),str(decision.max().date())],
        "calendar_sessions":len(trading),"missing_adj_vwap_partitions":missing,
        "label":"decision t; execution t+1; AdjVwap(t+2)/AdjVwap(t+1)-1",
        "factor_finite_ratio":float(np.isfinite(values).mean()),
        "label_finite_ratio":float(np.isfinite(labels_array).mean()),
        "limitations":"research lineage; no upstream PIT/investability certification"}
    return batch,labels,provenance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors",type=int,default=2)
    parser.add_argument("--days",type=int,default=500)
    parser.add_argument("--assets",type=int,default=5500)
    parser.add_argument("--manifest-sha256",default=SHA)
    parser.add_argument("--max-total-mib",type=int,default=128)
    args = parser.parse_args()
    _,_,info = load_real_batch(factors=args.factors,days=args.days,assets=args.assets,
                               manifest_sha256=args.manifest_sha256,max_total_mib=args.max_total_mib)
    print(json.dumps(info,ensure_ascii=False,indent=2))
