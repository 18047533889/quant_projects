"""Bounded research COS factor tiles; emits a collection of genuine QE receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data_access import get_store
from data_access.core.engine import DuckDBEngine
from data_access.cos.research import read_declared_cos_object
from data_access.cos.remote import cos_cache_root
from data_access.read.query_budget import QueryBudget
from data_access.registry.loader import DatasetRegistry
from data_access.store import DataAccessStore
from factor_optimizer.research_manifest import read_bound_factor
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.scripts.benchmark_real_cos_metric_batch import MANIFEST_SHA256
from quant_evaluator.scripts.load_real_cos_factor_batch import BASE, MIRROR, POOL, _ds

METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def select_records(mapping, count, object_mib, total_mib):
    if not (type(count) is int and 33 <= count <= 64 and
            type(object_mib) is int and 1 <= object_mib <= 128 and
            type(total_mib) is int and 1 <= total_mib <= 4096):
        raise ValueError("count 33..64, object 1..128 MiB, total 1..4096 MiB required")
    if not isinstance(mapping, dict):
        raise ValueError("manifest factors mapping required")
    eligible = []
    for name, row in mapping.items():
        if not isinstance(name, str) or not isinstance(row, dict):
            continue
        if row.get("verified") is not True or row.get("status") not in {
            "materialized_not_evaluated", "evaluated_optimization_pending"}:
            continue
        size, digest = row.get("bytes"), row.get("sha256")
        if type(size) is not int or not 0 < size <= object_mib * 1024**2:
            continue
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid factor digest")
        uri = f"{POOL}/{digest}/{name}.parquet"
        if row.get("uri") != uri:
            raise ValueError("factor URI is not manifest bound")
        eligible.append((name, uri, digest, size))
    selected = sorted(eligible, key=lambda item: (item[3], item[0]))[:count]
    if len(selected) != count or sum(r[3] for r in selected) > total_mib * 1024**2:
        raise ValueError("bounded verified factor sample unavailable")
    if len({r[2] for r in selected}) != count:
        raise ValueError("selected factor content hashes are duplicated")
    return tuple(selected)


def read_manifest(sha):
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("manifest_sha256 must be lowercase SHA256")
    uri = f"{BASE}/metadata/{sha}/landing_manifest.json"
    md = _ds("source_manifest", uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(DatasetRegistry({md.name: md}), engine)
        obj = read_declared_cos_object(store, md.name, allow_research=True)
        if obj.content_sha256 != sha:
            raise ValueError("manifest identity mismatch")
        rows = obj.table.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
            raise ValueError("invalid landing manifest")
        return rows[0]["factors"]
    finally:
        engine.close()


def axis_hash(frame):
    h = hashlib.sha256()
    h.update(np.asarray(frame.index, dtype="datetime64[ns]").view("int64").tobytes())
    for name in frame.columns:
        h.update(name.encode())
        h.update(b"\0")
    return h.hexdigest()


def iter_frames(records, manifest_sha, object_mib):
    uri = f"{BASE}/metadata/{manifest_sha}/landing_manifest.json"
    md = _ds("source_manifest", uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=2)
    try:
        for name, uri, digest, size in records:
            fd = _ds("factor_panel", uri.rsplit("/", 1)[0], name + ".parquet", "parquet")
            store = DataAccessStore(DatasetRegistry({md.name: md, fd.name: fd}), engine)
            bound = read_bound_factor(store, md.name, fd.name, name,
                                      max_object_mib=object_mib, allow_research=True)
            obj = bound.factor
            if (bound.manifest_sha256 != manifest_sha or obj.source_uri != uri or
                    obj.content_sha256 != digest or obj.downloaded_bytes != size):
                raise ValueError("factor source changed or disagrees with selected manifest")
            frame = obj.table.to_pandas()
            if "timestamp" not in frame:
                raise ValueError("factor timestamp missing")
            frame = frame.set_index("timestamp")
            frame.index = pd.to_datetime(frame.index).normalize()
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError("duplicate factor axis")
            frame = frame.loc[:, [c for c in frame if c.endswith((".SZ", ".SH"))]].sort_index()
            source = (name, uri, digest, size, obj.source_etag, axis_hash(frame))
            yield frame, source
            del frame, bound, obj, store
    finally:
        engine.close()


def intersect_axes(stream, count):
    dates = assets = None
    sources = []
    for frame, source in stream:
        dates = frame.index if dates is None else dates.intersection(frame.index)
        columns = set(frame.columns)
        assets = columns if assets is None else assets.intersection(columns)
        sources.append(source)
    if len(sources) != count or dates is None or dates.empty or not assets:
        raise ValueError("incomplete first pass or no common factor axes")
    return dates, assets, tuple(sources)


def load_labels(common_dates, common_assets, days, assets):
    """Use the same registered t/t+1/t+2 AdjVwap rule as load_real_batch."""
    if not (type(days) is int and 0 <= days <= 3000 and type(assets) is int and 30 <= assets <= 5500):
        raise ValueError("days 0..3000 and assets 30..5500 required")
    store = get_store()
    cal = store.read("ashare_calendar", columns=["TradeDate", "IsTradeDay"],
        time_range=(common_dates.min(), common_dates.max()), result="arrow",
        query_budget=QueryBudget(max_scan_files=1, max_rows=6000,
                                 max_result_bytes=2*1024**2)).to_pandas()
    trading = pd.DatetimeIndex(pd.to_datetime(
        cal.loc[cal["IsTradeDay"].eq(True), "TradeDate"])).normalize().sort_values()
    if trading.empty or trading.has_duplicates:
        raise ValueError("registered trading calendar invalid")
    local_days = {pd.Timestamp(p.stem) for p in MIRROR.glob("*.parquet") if len(p.stem) == 10}
    shared = set(common_dates)
    positions = [i for i in range(len(trading)-2) if trading[i] in shared and
                 all(trading[i+j] in local_days for j in (0, 1, 2))]
    positions = positions[-days:] if days else positions
    if not positions or (days and len(positions) != days):
        raise ValueError(f"only {len(positions)} complete calendar/factor/price triplets")
    dates = trading[positions]
    names = sorted(common_assets)[:assets]
    if len(names) < 30:
        raise ValueError("fewer than 30 common A-share assets")
    start, end = trading[min(positions)], trading[max(positions)+2]
    rows = store.read("ashare_stock_daily_adj", columns=["TradeDate", "Symbol", "AdjVwap"],
        time_range=(start, end), instrument_filter=names, read_root=str(MIRROR),
        result="arrow", query_budget=QueryBudget(
            max_scan_files=min(4000, (end-start).days+2), max_rows=20_000_000,
            max_result_bytes=1024**3)).to_pandas()
    if rows.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError("duplicate AdjVwap observation")
    prices = rows.pivot(index="TradeDate", columns="Symbol", values="AdjVwap")
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.reindex(index=trading, columns=names)
    p1 = prices.iloc[[i+1 for i in positions]].to_numpy(dtype=np.float64)
    p2 = prices.iloc[[i+2 for i in positions]].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = p2/p1-1
    values[~np.isfinite(values)] = np.nan
    times = dates.to_numpy(dtype="datetime64[ns]")
    asset_axis = AxisRef("asset", "str", len(names), np.asarray(names, dtype=str))
    t1 = tuple(trading[[i+1 for i in positions]].to_numpy())
    t2 = tuple(trading[[i+2 for i in positions]].to_numpy())
    labels = LabelBundle("adj_vwap_tplus1_to_tplus2_return", np.ascontiguousarray(values), 1,
        execution_delay=1, decision_time=tuple(times), observation_time=tuple(times),
        signal_available_time=tuple(times), execution_time=t1,
        label_start_time=t1, label_end_time=t2, validity=np.isfinite(values),
        asset_axis=asset_axis, source_ref="data_access:ashare_stock_daily_adj:AdjVwap",
        calendar_ref="data_access:ashare_calendar")
    return dates, names, labels


def make_tile(stream, expected, dates, assets, labels):
    if not 1 <= len(expected) <= 32:
        raise ValueError("tile must contain 1..32 factors")
    values = np.empty((len(dates), len(assets), len(expected)), dtype=np.float64)
    seen = 0
    for frame, source in stream:
        if seen >= len(expected) or source != expected[seen]:
            raise ValueError("factor source or axes changed between passes")
        if not dates.isin(frame.index).all() or not set(assets).issubset(frame.columns):
            raise ValueError("factor final axes changed between passes")
        values[:, :, seen] = frame.reindex(index=dates, columns=assets).to_numpy(dtype=np.float64, copy=False)
        seen += 1
    if seen != len(expected):
        raise ValueError("incomplete factor tile")
    times = dates.to_numpy(dtype="datetime64[ns]")
    return FactorBatch(tuple(s[0] for s in expected),
        AxisRef("time", "datetime64[ns]", len(dates), times), labels.asset_axis,
        values, validity=np.isfinite(values))


def summarize_tile(bundle, expected_ids):
    if tuple(bundle.factor_ids) != tuple(expected_ids):
        raise ValueError("evaluator factor coordinates changed")
    metrics = {}
    for metric in METRICS:
        artifact = bundle.artifacts[metric]
        if (artifact.artifact_kind != "scalar" or
                tuple(artifact.factor_axis.factor_ids) != tuple(expected_ids) or
                np.shape(artifact.values) != (len(expected_ids),)):
            raise ValueError("unsupported tile artifact coordinates")
        counts = artifact.provenance.get("observation_counts")
        if counts is None or len(counts) != len(expected_ids):
            raise ValueError("tile artifact observation counts are missing")
        for i, fid in enumerate(expected_ids):
            item = bundle.grouped_metrics[fid][metric]
            numeric = float(artifact.values[i])
            if (item.observation_count != counts[i] or
                    item.valid != bool(np.isfinite(numeric)) or
                    item.metric_version != artifact.producer_version or
                    (item.valid and item.value != numeric) or
                    (not item.valid and item.value is not None)):
                raise ValueError("tile metric value or evidence differs from artifact")
        metrics[metric] = {fid: {
            "value": bundle.grouped_metrics[fid][metric].value,
            "valid": bundle.grouped_metrics[fid][metric].valid,
            "observation_count": bundle.grouped_metrics[fid][metric].observation_count,
            "sample_unit": bundle.grouped_metrics[fid][metric].sample_unit,
            "metric_version": bundle.grouped_metrics[fid][metric].metric_version,
            "warnings": list(bundle.grouped_metrics[fid][metric].warnings),
        } for fid in expected_ids}
    receipt = dict(bundle.metadata["execution_receipt"])
    if (receipt.get("config_hash") != bundle.config_hash or
            receipt.get("backend_requested") not in {"cpu", "cuda_strict", "auto"} or
            receipt.get("backend_used") not in {"cpu", "cuda"}):
        raise ValueError("tile execution receipt is incomplete or inconsistent")
    return {"factor_ids": list(expected_ids), "config_hash": bundle.config_hash,
            "execution_receipt": receipt,
            "backend_requested": receipt["backend_requested"],
            "backend_used": receipt["backend_used"],
            "metric_backends": receipt.get("metric_backends"),
            "metrics": metrics}


def compare_collections(reference, candidate, *, rtol=1e-8, atol=1e-10):
    """Compare complete research collections; backend and timing may differ.

    Structural or evidence mismatch fails closed. No metric is averaged across
    factors or tiles. Differences are capped at 50 to keep the result bounded.
    """
    if (type(rtol) not in {int, float} or type(atol) not in {int, float} or
            not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0):
        raise ValueError("finite nonnegative numeric tolerances required")

    def failure(reason):
        return {"pass": False, "reason": reason, "differences": []}

    for label, report in (("reference", reference), ("candidate", candidate)):
        if (not isinstance(report, Mapping) or report.get("status") != "complete" or
                report.get("kind") != "research_factor_tile_collection.v1" or
                tuple(report.get("metric_ids", ())) != METRICS):
            return failure(f"{label} is not a complete default-metric collection")
    identity_fields = ("manifest_sha256", "label_content_hash", "decision_time_first",
                       "decision_time_last", "asset_axis_sha256", "selected_source_sha256",
                       "shape", "sources")
    for field in identity_fields:
        if json.dumps(reference.get(field), sort_keys=True) != json.dumps(
                candidate.get(field), sort_keys=True):
            return failure(f"collection identity differs: {field}")

    def index(report):
        sources = report["sources"]
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("sources are missing")
        ids = [row[0] for row in sources]
        if len(ids) != len(set(ids)) or len(ids) != report["shape"][2]:
            raise ValueError("source factor coordinates are invalid")
        found = {}
        for tile in report.get("tiles", ()):
            tile_ids = tile["factor_ids"]
            receipt = tile.get("execution_receipt", {})
            requested = tile.get("backend_requested", receipt.get("backend_requested"))
            actual = tile.get("backend_used", receipt.get("backend_used"))
            if (not tile_ids or len(tile_ids) > 32 or
                    requested not in {"cpu", "cuda_strict", "auto"} or
                    actual not in {"cpu", "cuda"} or
                    receipt.get("backend_used") != actual or
                    receipt.get("backend_requested") != requested or
                    receipt.get("config_hash") != tile.get("config_hash") or
                    (requested == "cpu" and actual != "cpu") or
                    (requested == "cuda_strict" and actual != "cuda") or
                    report.get("backend_requested", report.get("backend")) != requested):
                raise ValueError("tile receipt or size is invalid")
            if set(tile["metrics"]) != set(METRICS):
                raise ValueError("tile metric set is incomplete")
            for fid in tile_ids:
                if fid in found:
                    raise ValueError("duplicate factor coordinate")
                found[fid] = {metric: tile["metrics"][metric][fid] for metric in METRICS}
        if list(found) != ids:
            raise ValueError("tile factor order or coverage differs from sources")
        return found

    try:
        left, right = index(reference), index(candidate)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return failure(f"invalid collection coordinates or receipt: {exc}")
    differences = []
    checked = 0
    for fid in left:
        for metric in METRICS:
            a, b = left[fid][metric], right[fid][metric]
            checked += 1
            fields = ("valid", "observation_count", "metric_version", "sample_unit")
            if any(a.get(field) != b.get(field) for field in fields):
                if len(differences) < 50:
                    differences.append({"factor_id": fid, "metric_id": metric,
                                        "reason": "metric evidence differs"})
                continue
            if (type(a.get("valid")) is not bool or
                    type(a.get("observation_count")) is not int or
                    not isinstance(a.get("metric_version"), str) or
                    not isinstance(a.get("sample_unit"), str)):
                if len(differences) < 50:
                    differences.append({"factor_id": fid, "metric_id": metric,
                                        "reason": "metric evidence is malformed"})
                continue
            av, bv = a.get("value"), b.get("value")
            if not a["valid"]:
                same = av is None and bv is None
            else:
                same = (type(av) in {int, float} and type(bv) in {int, float} and
                        math.isfinite(av) and math.isfinite(bv) and
                        math.isclose(av, bv, rel_tol=rtol, abs_tol=atol))
            if not same and len(differences) < 50:
                differences.append({"factor_id": fid, "metric_id": metric,
                                    "reason": "metric value differs"})
    return {"pass": not differences, "compared_factor_count": len(left),
            "compared_metric_count": checked, "rtol": rtol, "atol": atol,
            "differences": differences}


def memory_preflight(tile_size, object_mib, max_working_gib):
    with open("/proc/meminfo", encoding="ascii") as stream:
        available = next(int(line.split()[1])*1024 for line in stream
                         if line.startswith("MemAvailable:"))
    peak = 50 * 1024**3  # conservative observed F32 parent plus worker envelope
    cache_path = cos_cache_root()
    while not cache_path.exists():
        cache_path = cache_path.parent
    free = shutil.disk_usage(cache_path).free
    disk_need = (2*object_mib + 1024) * 1024**2
    return {"pass": 1 <= tile_size <= 32 and peak <= max_working_gib*1024**3 and
                    available >= peak + 8*1024**3 and free >= disk_need,
            "estimated_tile_peak_bytes": peak, "mem_available_bytes": available,
            "cos_cache_disk_free_bytes": free, "required_disk_bytes": disk_need}


def write_collection_atomic(path, collection):
    """Publish complete, bounded JSON only after every tile succeeds."""
    payload = json.dumps(collection, indent=2, ensure_ascii=False) + "\n"
    if len(payload.encode("utf-8")) > 4 * 1024**2:
        raise ValueError("collection report exceeds 4 MiB bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".qe-factor-tiles-", suffix=".json", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--max-working-gib", type=int, default=50)
    parser.add_argument("--backend", choices=("cpu", "cuda_strict", "auto"), default="auto")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        records = select_records(read_manifest(args.manifest_sha256), args.factors,
                                 args.max_object_mib, args.max_total_mib)
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "manifest_sha256": args.manifest_sha256,
                          "reason": f"{type(exc).__name__}: {exc}",
                          "manifest_preflight_s": time.perf_counter() - started}), flush=True)
        if args.run:
            raise SystemExit(2) from exc
        return
    preflight = memory_preflight(args.tile_size, args.max_object_mib, args.max_working_gib)
    preflight_s = time.perf_counter() - started
    print(json.dumps({"preflight": preflight, "selected_factor_ids": [r[0] for r in records],
                      "selected_object_bytes": sum(r[3] for r in records),
                      "manifest_preflight_s": preflight_s,
                      "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),
          flush=True)
    if not args.run:
        return
    if not preflight["pass"]:
        raise SystemExit("insufficient tile memory or temporary disk headroom")
    first_pass_started = time.perf_counter()
    common_dates, common_assets, sources = intersect_axes(
        iter_frames(records, args.manifest_sha256, args.max_object_mib), len(records))
    first_pass_s = time.perf_counter() - first_pass_started
    labels_started = time.perf_counter()
    dates, assets, labels = load_labels(common_dates, common_assets, args.days, args.assets)
    labels_s = time.perf_counter() - labels_started
    tiles = []
    tile_timings = []
    for start in range(0, len(records), args.tile_size):
        tile_started = time.perf_counter()
        if not memory_preflight(args.tile_size, args.max_object_mib,
                                args.max_working_gib)["pass"]:
            raise RuntimeError("tile resources changed below the required headroom")
        selected = records[start:start+args.tile_size]
        expected = sources[start:start+args.tile_size]
        load_started = time.perf_counter()
        batch = make_tile(iter_frames(selected, args.manifest_sha256,
                                     args.max_object_mib), expected, dates, assets, labels)
        load_s = time.perf_counter() - load_started
        evaluate_started = time.perf_counter()
        bundle = evaluate(batch, labels, metrics=METRICS, backend=args.backend)
        evaluate_s = time.perf_counter() - evaluate_started
        tile_result = summarize_tile(bundle, [r[0] for r in selected])
        if (tile_result["backend_requested"] != args.backend or
                (args.backend == "cuda_strict" and tile_result["backend_used"] != "cuda") or
                (args.backend == "cpu" and tile_result["backend_used"] != "cpu")):
            raise ValueError("tile execution route differs from requested backend")
        tiles.append(tile_result)
        del batch, bundle
        timing = {"factor_start": start, "factor_count": len(selected),
                  "load_s": load_s, "evaluate_s": evaluate_s,
                  "total_s": time.perf_counter() - tile_started,
                  "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        tile_timings.append(timing)
        print(json.dumps({"tile_complete": len(tile_timings), "tiles_total":
                          (len(records) + args.tile_size - 1) // args.tile_size,
                          "backend_requested": args.backend,
                          "backend_used": tile_result["backend_used"], **timing}), flush=True)
    performance = {"manifest_preflight_s": preflight_s, "first_pass_s": first_pass_s,
                   "labels_s": labels_s, "tiles": tile_timings,
                   "total_s": time.perf_counter() - started,
                   "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "total_scope": "CLI start through final tile; report serialization and write excluded",
                   "rss_note": "Linux process high-water mark; tile values are cumulative, not per-tile deltas"}
    result = {"status": "complete", "kind": "research_factor_tile_collection.v1",
        "manifest_sha256": args.manifest_sha256, "label_content_hash": labels.content_hash,
        "decision_time_first": str(dates[0]), "decision_time_last": str(dates[-1]),
        "asset_axis_sha256": hashlib.sha256("\0".join(assets).encode()).hexdigest(),
        "selected_source_sha256": hashlib.sha256(json.dumps(
            [list(row) for row in sources], separators=(",", ":")).encode()).hexdigest(),
        "shape": [len(dates), len(assets), len(records)], "backend_requested": args.backend,
        "metric_ids": list(METRICS), "sources": sources, "tiles": tiles,
        "performance": performance,
        "limitations": "Tile receipts only; no whole-batch EvaluationBundle or production/PIT certification"}
    if args.output:
        write_collection_atomic(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "complete", "tiles": len(tiles), "factors": len(records),
                      "total_wall_s": time.perf_counter() - started,
                      "peak_process_rss_kib": performance["peak_process_rss_kib"],
                      "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
