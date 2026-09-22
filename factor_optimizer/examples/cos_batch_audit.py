#!/usr/bin/env python3
"""Read a small, deterministic COS factor sample via DataAccess and diagnose TRAIN."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_access.core.engine import DuckDBEngine
from data_access.core.storage import StorageSpec
from data_access.registry.loader import DatasetRegistry, StaticDataset
from data_access.read.formats import FormatSpec
from data_access.store import DataAccessStore
from data_access.cos.research import read_declared_cos_object
from factor_optimizer.research_diagnostics import diagnose_training_batch
from factor_optimizer.research_manifest import read_bound_factor
from factor_optimizer.research_batch import automatic_time_split
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

POOL = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values"
MANIFEST = ("cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/"
            "00e545d254ca742305a37405a69ffe55e9a04448d0f176282c4f07997f1feb66")


def _validate_sample_limits(n_factors, max_factor_bytes):
    if type(n_factors) is not int or not 1 <= n_factors <= 16:
        raise ValueError("n_factors must be a positive integer <= 16")
    if type(max_factor_bytes) is not int or not 0 < max_factor_bytes <= 64*1024**2:
        raise ValueError("max_factor_bytes must be an integer in 1..64 MiB")


def select_manifest_records(rows, n_factors, *, max_factor_bytes=8*1024**2):
    """Deterministic sample; quality and 128 MiB batch cap precede downloads."""
    _validate_sample_limits(n_factors, max_factor_bytes)
    if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
        raise ValueError("one manifest with a factors mapping is required")
    eligible = []
    for name, record in sorted(rows[0]["factors"].items()):
        if not isinstance(record, dict) or record.get("verified") is not True:
            continue
        if record.get("status") not in {"materialized_not_evaluated", "evaluated_optimization_pending"}:
            continue
        size = record.get("bytes")
        if type(size) is not int or not 0 < size <= max_factor_bytes:
            continue
        sha = record.get("sha256")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or record.get("uri") != f"{POOL}/{sha}/{name}.parquet"):
            raise ValueError("manifest factor URI must match the authorized hash directory")
        eligible.append((name, record))
    if len(eligible) < n_factors:
        raise ValueError(f"requested {n_factors} factors but only {len(eligible)} eligible bounded records")
    selected = eligible[:n_factors]
    if sum(record["bytes"] for _, record in selected) > 128*1024**2:
        raise ValueError("selected factor batch exceeds 128 MiB; request fewer factors")
    return selected


def choose_assets(panels, train_dates, n_assets=512):
    if type(n_assets) is not int or n_assets < 1:
        raise ValueError("n_assets must be a positive integer")
    common = sorted(set.intersection(*(set(p.columns) for p in panels)))
    coverage = np.vstack([
        np.isfinite(p.reindex(train_dates)[common].to_numpy()).mean(axis=0)
        for p in panels])
    eligible = [i for i in range(len(common)) if coverage[:, i].min() >= .90]
    ranked = sorted(eligible, key=lambda i: (-coverage[:, i].min(), -coverage[:, i].mean(), common[i]))
    if len(ranked) < n_assets:
        raise ValueError("insufficient TRAIN-covered assets for a real 20-bin audit")
    return sorted(common[i] for i in ranked[:n_assets])


def load_cos_sample(n_factors=2, n_assets=256, *, include_lineages=False,
                    manifest_uri=None, max_factor_bytes=8*1024**2,
                    coverage_policy="isolate"):
    _validate_sample_limits(n_factors, max_factor_bytes)
    if coverage_policy not in {"isolate", "strict"}:
        raise ValueError("coverage_policy must be isolate or strict")
    if type(n_assets) is not int or n_assets < 1:
        raise ValueError("n_assets must be a positive integer")
    manifest_uri = MANIFEST + "/landing_manifest.json" if manifest_uri is None else manifest_uri
    prefix = POOL.rsplit("/", 1)[0] + "/metadata/"
    if (not isinstance(manifest_uri, str) or not re.fullmatch(
            re.escape(prefix) + r"[0-9a-f]{64}/landing_manifest\.json", manifest_uri)):
        raise ValueError("manifest must be an exact content-addressed landing manifest in the authorized pool")
    from real_batch_audit import _load_vwap, DAILY_ADJ
    root = Path("/home/sunhaiwei/quant_projects/workspace_data/research_factor_panel")
    def dataset(name, uri, filename, fmt):
        return StaticDataset(name=name, access_mode="published", layout="plain",
            time_column=None, instrument_column=None, hive_partitioning=False, union_by_name=True,
            root=root/name, glob=filename, format_spec=FormatSpec.from_yaml(fmt),
            storage=StorageSpec(type="cos", uri=uri, layout="plain"))
    manifest_ds = dataset("source_manifest", manifest_uri.rsplit("/", 1)[0], "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=2)
    panels, sources, lineages = [], [], {}
    try:
        store = DataAccessStore(DatasetRegistry({manifest_ds.name: manifest_ds}), engine)
        manifest = read_declared_cos_object(store, manifest_ds.name, allow_research=True)
        selected = select_manifest_records(manifest.table.to_pylist(), n_factors,
                                           max_factor_bytes=max_factor_bytes)
        factor_ids = tuple(name for name, _ in selected)
        for factor_id, record in selected:
            ds = dataset("factor_panel", record["uri"].rsplit("/", 1)[0],
                         factor_id+".parquet", "parquet")
            store = DataAccessStore(DatasetRegistry({manifest_ds.name: manifest_ds, ds.name: ds}), engine)
            bound = read_bound_factor(store, manifest_ds.name, ds.name, factor_id, allow_research=True)
            result = bound.factor
            lineages[factor_id] = bound.treatment_signature
            table = result.table.to_pandas()
            if "timestamp" in table.columns:
                table = table.set_index("timestamp")
            if table.index.name != "timestamp":
                raise ValueError("factor has no explicit timestamp identity")
            table.index = pd.to_datetime(table.index).normalize()
            if table.index.has_duplicates:
                raise ValueError("duplicate factor dates")
            table = table.loc[:, [c for c in table if c.endswith((".SZ", ".SH"))]].sort_index()
            panels.append(table)
            sources.append({"factor": factor_id, "uri": result.source_uri,
                            "etag": result.source_etag, "sha256": result.content_sha256,
                            "downloaded_bytes": result.downloaded_bytes,
                            "manifest_sha256": bound.manifest_sha256,
                            "source_status": bound.source_status, "expression": bound.expression})
    finally:
        engine.close()
    dates = panels[0].index
    for p in panels[1:]:
        dates = dates.intersection(p.index)
    candidate_dates = dates[-510:]
    calendar = pd.DatetimeIndex([pd.Timestamp(p.stem) for p in sorted(DAILY_ADJ.glob("*.parquet"))
                                if candidate_dates.min() <= pd.Timestamp(p.stem) <= dates.max()])
    dates = dates[dates.isin(calendar)]
    positions = calendar.get_indexer(dates)
    dates = dates[positions + 2 < len(calendar)][-500:]
    if len(dates) != 500:
        raise ValueError("need 500 aligned trading dates with full label endpoints")
    pos = calendar.get_indexer(dates)
    # Split metadata only: no return values or TEST scores are read for selection.
    selection_split = automatic_time_split(SimpleNamespace(
        decision_time=tuple(dates.to_numpy(dtype="datetime64[ns]")),
        label_end_time=tuple(calendar[pos+2].to_numpy(dtype="datetime64[ns]"))))
    train_dates = dates[list(selection_split.train_indices)]
    quarantined, retained, factor_coverage = [], [], {}
    for factor_id, panel in zip(factor_ids, panels):
        coverage = np.isfinite(panel.reindex(train_dates).to_numpy()).mean(axis=0)
        eligible_count = int((coverage >= .90).sum())
        factor_coverage[factor_id] = eligible_count
        if coverage_policy == "isolate" and eligible_count < n_assets:
            quarantined.append({"factor": factor_id, "eligible_assets": eligible_count,
                                "required_assets": n_assets,
                                "reason": "insufficient_training_coverage"})
        else:
            retained.append((factor_id, panel))
    if not retained:
        raise ValueError("all requested factors lack sufficient TRAIN-covered assets: "
                         + json.dumps(quarantined, sort_keys=True))
    factor_ids = tuple(name for name, _ in retained)
    panels = [panel for _, panel in retained]
    lineages = {name: lineages[name] for name in factor_ids}
    assets = choose_assets(panels, train_dates, n_assets)
    prices = _load_vwap(calendar.min(), calendar.max(), assets).reindex(calendar)
    pos = calendar.get_indexer(dates)
    vwap = prices.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        y = vwap[pos+2] / vwap[pos+1] - 1
    values = np.stack([p.reindex(dates)[assets].to_numpy(dtype=float) for p in panels], axis=-1)
    times = dates.to_numpy(dtype="datetime64[ns]")
    ta = AxisRef("time", "datetime64[ns]", len(times), times)
    aa = AxisRef("asset", "str", len(assets), np.asarray(assets, dtype=str))
    batch = FactorBatch(factor_ids, ta, aa, np.ascontiguousarray(values), validity=np.isfinite(values))
    labels = LabelBundle("adj_vwap_tplus1_to_tplus2_return", np.ascontiguousarray(y), 1,
        execution_delay=1, decision_time=tuple(times), observation_time=tuple(times),
        signal_available_time=tuple(times), execution_time=tuple(calendar[pos+1].to_numpy()),
        label_start_time=tuple(calendar[pos+1].to_numpy()), label_end_time=tuple(calendar[pos+2].to_numpy()),
        validity=np.isfinite(y), asset_axis=aa, source_ref="data_access:ashare_stock_daily_adj:AdjVwap",
        calendar_ref=str(DAILY_ADJ))
    provenance = {"sources": sources, "days": len(times), "assets": len(assets),
        "coverage_policy": coverage_policy,
        "retained_factor_ids": list(factor_ids),
        "quarantined_factors": quarantined,
        "train_eligible_assets": factor_coverage,
        "manifest_uri": manifest_uri,
        "max_factor_bytes": max_factor_bytes,
        "max_batch_factor_bytes": 128*1024**2,
        "asset_selection_split": selection_split.identity,
        "asset_selection_train_days": len(selection_split.train_indices),
        "date_span": [str(dates.min().date()), str(dates.max().date())],
        "selection": "first eligible bound manifest factor IDs; assets selected on TRAIN coverage only",
        "limitations": "source-declared lineage; no upstream PIT or investability certification"}
    return (batch, labels, provenance, lineages) if include_lineages else (batch, labels, provenance)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="also run frozen TRAIN/VALIDATION selection")
    parser.add_argument("--manifest", default=None, help="exact in-pool landing_manifest.json COS URI")
    parser.add_argument("--factors", type=int, default=2, help="deterministic factor sample, 1..16")
    parser.add_argument("--assets", type=int, default=256, help="TRAIN-covered asset count")
    parser.add_argument("--coverage-policy", choices=("isolate", "strict"), default="isolate",
                        help="isolate individually low-coverage factors or reject the whole sample")
    parser.add_argument("--max-factor-mib", type=int, choices=range(1, 65), default=8,
                        help="per-factor admission cap; total factor objects capped at 128 MiB")
    args = parser.parse_args()
    batch, labels, provenance, lineages = load_cos_sample(
        n_factors=args.factors, n_assets=args.assets, include_lineages=True,
        manifest_uri=args.manifest, max_factor_bytes=args.max_factor_mib*1024**2,
        coverage_policy=args.coverage_policy)
    report = {"inputs": provenance, "test_evaluated": False,
              "diagnostics": diagnose_training_batch(batch, labels)}
    if args.optimize:
        from factor_optimizer.research_batch import optimize_factor_batch
        result = optimize_factor_batch(batch, labels, allow_research=True, lineages=lineages)
        report["automatic"] = {name: {
            "status": r.status, "selected_family": r.selected_family,
            "train_gain": r.train_gain, "validation_lower_bound": r.validation_lower_bound,
            "reason": r.reason, "plan_identity": r.plan_identity,
            "baseline_diagnostics": dict(r.baseline_diagnostics),
            "joint_diagnostics": dict(r.joint_diagnostics),
            "candidates": [dict(c) for c in r.candidates],
        } for name, r in result.factors.items()}
        report["selection_objective"] = "joint.v1: Sharpe, RankICIR, RankIC, drawdown, worst-block Sharpe, turnover"
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
