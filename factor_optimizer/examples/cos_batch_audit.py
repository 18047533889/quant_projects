#!/usr/bin/env python3
"""Read a small, deterministic COS factor sample via DataAccess and diagnose TRAIN."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_access.core.engine import DuckDBEngine
from data_access.core.storage import StorageSpec
from data_access.registry.loader import DatasetRegistry, ParametricDataset
from data_access.store import DataAccessStore
from data_access.cos.remote import cos_cli_ls
from data_access.cos.research import read_declared_cos_object
from factor_optimizer.research_diagnostics import diagnose_training_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

POOL = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values"


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


def load_cos_sample(n_factors=3, n_assets=512):
    from real_batch_audit import _load_vwap, DAILY_ADJ
    cli = os.environ.get("DATA_ACCESS_COS_CLI", "clean-cos-ro")
    objects = cos_cli_ls(POOL + "/", cli=cli, timeout_s=20)
    # Prespecified metadata-only sampling: never select on future performance.
    selected = sorted((r for r in objects if r["key"].endswith(".parquet")
                       and 0 < r["size"] <= 8*1024**2), key=lambda r: r["key"])[:n_factors]
    if len(selected) != n_factors:
        raise ValueError("insufficient bounded COS sample objects")
    factor_ids = tuple(Path(r["key"]).stem for r in selected)
    root = Path("/home/sunhaiwei/quant_projects/workspace_data/research_factor_panel")
    ds = ParametricDataset(
        name="lqtp_research_panel", access_mode="published", layout="plain",
        time_column=None, instrument_column=None, hive_partitioning=False, union_by_name=True,
        root_template=str(root), glob_template="{factor_id}.parquet",
        params_schema={"factor_id": "str"}, static_root=root,
        storage=StorageSpec(type="cos", uri=POOL, layout="plain"))
    engine = DuckDBEngine(threads=2)
    panels, sources = [], []
    try:
        store = DataAccessStore(DatasetRegistry({ds.name: ds}), engine)
        for factor_id in factor_ids:
            result = read_declared_cos_object(store, ds.name,
                params={"factor_id": factor_id}, allow_research=True)
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
                            "downloaded_bytes": result.downloaded_bytes})
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
    assets = choose_assets(panels, dates[:300], n_assets)
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
    return batch, labels, {"sources": sources, "days": len(times), "assets": len(assets),
        "date_span": [str(dates.min().date()), str(dates.max().date())],
        "selection": "first bounded object keys; assets selected on TRAIN coverage only",
        "limitations": "research replay; no upstream PIT or investability certification; recipe lineage not loaded"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="also run frozen TRAIN/VALIDATION selection")
    args = parser.parse_args()
    batch, labels, provenance = load_cos_sample()
    report = {"inputs": provenance, "test_evaluated": False,
              "diagnostics": diagnose_training_batch(batch, labels)}
    if args.optimize:
        from factor_optimizer.research_batch import optimize_factor_batch
        result = optimize_factor_batch(batch, labels, allow_research=True)
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
