#!/usr/bin/env python3
"""Bounded read-only real sample for the research batch optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from data_access import get_store
from data_access.read.query_budget import QueryBudget

from factor_optimizer.research_batch import optimize_factor_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

ROOT = Path("/home/sunhaiwei/quant_projects")
FACTOR_ROOT = ROOT / "weekly_backtest_output/factor_matrices_all"
DAILY_ADJ = Path("/home/sunhaiwei/cos_data/StockDailyBarAdj")
FACTORS = ("price_smoothness", "volume_ratio_persistence",
           "ema_trend_simple", "cyclical_trend_confidence")
N_DAYS, N_ASSETS = 500, 64


def _index_name(path: Path) -> str:
    names = _column_names(path)
    for name in ("timestamp", "date"):
        if name in names:
            return name
    raise ValueError(f"{path}: no date index column")


def _column_names(path: Path) -> list[str]:
    # Metadata-only projection, still subject to DataAccess authorization.
    return get_store().read_uri(str(path), limit=0, result="arrow",
        query_budget=QueryBudget(max_scan_files=1, max_result_bytes=1024*1024)).to_arrow().column_names


def _choose_assets(paths: list[Path], train_dates: pd.DatetimeIndex) -> list[str]:
    common = None
    for path in paths:
        names = {x for x in _column_names(path)
                 if x.endswith((".SZ", ".SH"))}
        common = names if common is None else common & names
    candidates = sorted(common or ())[:600]
    scores = {asset: [] for asset in candidates}
    for path in paths:
        # Align exact TRAIN dates, not row positions in differently dated files.
        table = _load_factor(path, candidates).reindex(train_dates)
        for asset in candidates:
            scores[asset].append(float(np.isfinite(
                table[asset].to_numpy()).mean()))
    ranked = sorted(candidates,
                    key=lambda a: (min(scores[a]), sum(scores[a]), a),
                    reverse=True)
    selected = sorted(ranked[:N_ASSETS])
    if len(selected) != N_ASSETS or min(min(scores[a]) for a in selected) < .90:
        raise RuntimeError("insufficient jointly covered assets")
    return selected


def _load_factor(path: Path, assets: list[str]) -> pd.DataFrame:
    index = _index_name(path)
    frame = get_store().read_uri(str(path), columns=[*assets, index],
        result="arrow", query_budget=QueryBudget(
            max_scan_files=1, max_rows=20000, max_result_bytes=128*1024*1024)).to_pandas()
    if index in frame.columns:
        frame = frame.set_index(index)
    elif frame.index.name != index:
        raise ValueError(f"{path}: projected index {index!r} was not restored")
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame.sort_index()


def _load_vwap(start: pd.Timestamp, end: pd.Timestamp,
               assets: list[str]) -> pd.DataFrame:
    files = [p for p in sorted(DAILY_ADJ.glob("*.parquet"))
             if start <= pd.Timestamp(p.stem) <= end]
    if not files:
        raise FileNotFoundError("no StockDailyBarAdj files in bounded interval")
    # The registered dataset owns partition pruning, filtering and governance.
    # ASHARE_PARQUET_ROOT must point to the same configured mirror as DAILY_ADJ.
    rows = get_store().read("ashare_stock_daily_adj",
        columns=["TradeDate", "Symbol", "AdjVwap"], time_range=(start, end),
        instrument_filter=assets, result="arrow",
        query_budget=QueryBudget(max_scan_files=600, max_rows=600*len(assets),
                                 max_result_bytes=32*1024*1024)).to_pandas()
    rows = rows.rename(columns={"TradeDate": "date", "Symbol": "asset", "AdjVwap": "vwap"})
    out = rows.pivot(index="date", columns="asset", values="vwap")
    out.index = pd.to_datetime(out.index).normalize()
    return out.reindex(columns=assets).sort_index()


def load_sample():
    """Read a bounded sample; select assets using the final aligned TRAIN dates."""
    paths = [FACTOR_ROOT / f"{name}.parquet" for name in FACTORS]
    # Calendar alignment requires only index columns, not factor/label values.
    panels = [_load_factor(path, []) for path in paths]
    dates = panels[0].index
    for panel in panels[1:]:
        dates = dates.intersection(panel.index)
    candidate_dates = dates[-(N_DAYS + 10):]
    calendar = pd.DatetimeIndex([pd.Timestamp(p.stem) for p in sorted(DAILY_ADJ.glob("*.parquet"))
                                if candidate_dates.min() <= pd.Timestamp(p.stem) <= dates.max()])
    eligible = dates[dates.isin(calendar)]
    positions = np.asarray([calendar.get_loc(d) for d in eligible])
    eligible = eligible[positions + 2 < len(calendar)][-N_DAYS:]
    if len(eligible) != N_DAYS:
        raise RuntimeError(f"expected {N_DAYS} eligible dates, got {len(eligible)}")
    assets = _choose_assets(paths, eligible[:int(N_DAYS * .6)])
    panels = [_load_factor(path, assets) for path in paths]
    vwap = _load_vwap(candidate_dates.min(), dates.max(), assets).reindex(calendar)
    positions = np.asarray([vwap.index.get_loc(d) for d in eligible])
    entry = tuple(vwap.index[positions + 1].to_numpy(dtype="datetime64[ns]"))
    end = tuple(vwap.index[positions + 2].to_numpy(dtype="datetime64[ns]"))
    labels = vwap.to_numpy()[positions + 2] / vwap.to_numpy()[positions + 1] - 1.
    values = np.stack([p.reindex(eligible)[assets].to_numpy(dtype=np.float64)
                       for p in panels], axis=-1)
    times = eligible.to_numpy(dtype="datetime64[ns]")
    asset_values = np.asarray(assets, dtype=str)
    time_axis = AxisRef("time", "datetime64[ns]", len(times), times)
    asset_axis = AxisRef("asset", "str", len(assets), asset_values)
    batch = FactorBatch(FACTORS, time_axis, asset_axis,
                        np.ascontiguousarray(values), validity=np.isfinite(values))
    decisions = tuple(times)
    target = LabelBundle(
        "adj_vwap_tplus1_to_tplus2_return",
        np.ascontiguousarray(labels), 1, execution_delay=1,
        decision_time=decisions, execution_time=entry,
        signal_available_time=decisions, label_start_time=entry,
        label_end_time=end, observation_time=decisions,
        validity=np.isfinite(labels), asset_axis=asset_axis,
        source_ref=str(DAILY_ADJ / "YYYY-MM-DD.parquet") + ":AdjVwap",
        calendar_ref="StockDailyBarAdj trading dates",
        metadata={"timing": "close(t); entry AdjVwap(t+1); exit AdjVwap(t+2)"},
    )
    inputs = {"factor_paths": [str(p) for p in paths],
                   "label_source": target.source_ref,
                   "date_span": [str(eligible[0].date()), str(eligible[-1].date())],
                   "days": len(eligible), "assets": len(assets)}
    return batch, target, inputs


def main() -> int:
    batch, target, inputs = load_sample()
    result = optimize_factor_batch(batch, target, allow_research=True)
    summary = {
        "inputs": inputs,
        "asset_selection": "joint coverage on the exact first 300 aligned TRAIN dates",
        "provenance_limit": "research replay of existing matrices; upstream factor-build PIT is not re-certified",
        "split_sizes": {"train": len(result.split.train_indices),
                        "validation": len(result.split.validation_indices),
                        "test_reserved": len(result.split.test_indices)},
        "test_evaluated": result.test_evaluated,
        "factors": {key: {"status": item.status,
                           "selected_family": item.selected_family,
                           "train_gain": item.train_gain,
                           "validation_lower_bound": item.validation_lower_bound,
                           "validation_candidate_identity": item.validation_candidate_identity,
                           "validation_coverage": item.validation_coverage,
                           "reason": item.reason,
                           "candidate_count": len(item.candidates),
                           "families": {
                               family: {
                                   "train_evaluated": sum(
                                       r.get("family") == family and r.get("status") == "train_evaluated"
                                       for r in item.candidates),
                                   "ineligible": sum(
                                       r.get("family") == family and r.get("status") == "ineligible"
                                       for r in item.candidates),
                                   "ineligible_reasons": sorted({
                                       str(r.get("reason")) for r in item.candidates
                                       if r.get("family") == family and r.get("status") == "ineligible"
                                   }),
                               }
                               for family in sorted({str(r.get("family")) for r in item.candidates})
                           }}
                    for key, item in result.factors.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
