from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"
EVALUATOR = PROJECT / "evaluation" / "gtja185_batch.py"
TESTS = PROJECT / "tests" / "test_gtja185_evaluation_contracts.py"
DOCS = PROJECT / "docs" / "gtja185_batch_evaluation.md"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_evaluator() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")

    text = replace_exact(
        text,
        '''    annualization: int = 252
    forward_price_field: str = "vwap"
    materialize_staging: bool = False
''',
        '''    annualization: int = 252
    forward_price_field: str = "vwap"
    entry_lag: int = 1
    cost_bps: float = 10.0
    universe_dataset: str | None = None
    universe_membership_field: str = "is_member"
    tradability_field: str | None = "is_tradable"
    require_point_in_time_universe: bool = False
    materialize_staging: bool = False
''',
        "evaluation config fields",
    )
    text = replace_exact(
        text,
        '''        if self.publish and not self.materialize_staging:
            raise ValueError("publish=True requires materialize_staging=True")
''',
        '''        if self.entry_lag < 1:
            raise ValueError("entry_lag must be >= 1 to prevent same-bar execution bias")
        if not math.isfinite(float(self.cost_bps)) or self.cost_bps < 0:
            raise ValueError("cost_bps must be a finite non-negative number")
        if self.publish and not self.materialize_staging:
            raise ValueError("publish=True requires materialize_staging=True")
        if str(self.run_mode).lower() == "production" and not self.require_point_in_time_universe:
            raise ValueError(
                "production GTJA185 evaluation requires require_point_in_time_universe=True"
            )
        if self.require_point_in_time_universe and not self.universe_dataset:
            raise ValueError(
                "require_point_in_time_universe=True requires a registered universe_dataset"
            )
''',
        "evaluation config validation",
    )

    text = replace_exact(
        text,
        '''def _price_forward_returns(frame: pd.DataFrame, horizons: Sequence[int], price_field: str) -> dict[int, pd.Series]:
    if price_field not in frame.columns:
        raise ValueError(f"forward price field not found: {price_field}")
    panel = frame.set_index(["datetime", "asset"])[price_field].astype(float).sort_index()
    outputs: dict[int, pd.Series] = {}
    grouped = panel.groupby(level="asset", group_keys=False)
    for horizon in sorted(set(int(h) for h in horizons)):
        future = grouped.shift(-horizon)
        value = future / panel - 1.0
        outputs[horizon] = value.replace([np.inf, -np.inf], np.nan).rename(f"forward_return_{horizon}")
    return outputs
''',
        '''def _price_forward_returns(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    price_field: str,
    *,
    entry_lag: int,
) -> dict[int, pd.Series]:
    """Return next-entry holding-period returns.

    A signal observed on date ``t`` enters at ``t + entry_lag`` and exits after
    ``horizon`` additional trading observations.  This prevents same-bar execution
    bias and makes the label convention explicit.
    """
    if price_field not in frame.columns:
        raise ValueError(f"forward price field not found: {price_field}")
    if entry_lag < 1:
        raise ValueError("entry_lag must be >= 1")
    panel = frame.set_index(["datetime", "asset"])[price_field].astype(float).sort_index()
    outputs: dict[int, pd.Series] = {}
    grouped = panel.groupby(level="asset", group_keys=False)
    entry = grouped.shift(-entry_lag)
    for horizon in sorted(set(int(h) for h in horizons)):
        exit_price = grouped.shift(-(entry_lag + horizon))
        value = exit_price / entry - 1.0
        outputs[horizon] = value.replace([np.inf, -np.inf], np.nan).rename(
            f"forward_return_{horizon}"
        )
    return outputs


def _purged_split_mask(
    index: pd.MultiIndex,
    split: str,
    bounds: SplitBoundaries,
    trading_dates: pd.DatetimeIndex,
    *,
    entry_lag: int,
    horizon: int,
) -> np.ndarray:
    """Chronological split mask with label-end purging/embargo.

    A row belongs to a split only when both its signal date and label exit date
    are inside that same split.  Training labels therefore never consume
    validation observations, and validation labels never consume test data.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(index.get_level_values("datetime"))).normalize()
    calendar = pd.DatetimeIndex(trading_dates).normalize().unique().sort_values()
    positions = calendar.get_indexer(dates)
    exit_positions = positions + int(entry_lag) + int(horizon)
    valid_exit = (positions >= 0) & (exit_positions >= 0) & (exit_positions < len(calendar))
    exit_dates = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_dates[valid_exit] = calendar.to_numpy(dtype="datetime64[ns]")[exit_positions[valid_exit]]
    exit_index = pd.DatetimeIndex(exit_dates)
    train_end = pd.Timestamp(bounds.train_end)
    valid_end = pd.Timestamp(bounds.valid_end)
    last_date = pd.Timestamp(bounds.last_date)
    if split == "train":
        return np.asarray(valid_exit & (dates <= train_end) & (exit_index <= train_end))
    if split == "valid":
        return np.asarray(
            valid_exit
            & (dates > train_end)
            & (dates <= valid_end)
            & (exit_index <= valid_end)
        )
    if split == "test":
        return np.asarray(valid_exit & (dates > valid_end) & (exit_index <= last_date))
    if split == "all":
        return np.asarray(valid_exit)
    raise ValueError(f"unknown split: {split}")
''',
        "next-entry forward returns and purged splits",
    )

    text = replace_exact(
        text,
        '''def _summary_stats(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": None, "std": None, "ir": None, "t_stat": None, "positive_ratio": None}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    t_stat = mean / (std / math.sqrt(len(clean))) if math.isfinite(std) and std > 0 else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "ir": ir if math.isfinite(ir) else None,
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "positive_ratio": float((clean > 0).mean()),
    }
''',
        '''def _summary_stats(
    values: pd.Series,
    *,
    hac_lags: int = 0,
) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "ir": None,
            "t_stat": None,
            "hac_t_stat": None,
            "positive_ratio": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    t_stat = mean / (std / math.sqrt(len(clean))) if math.isfinite(std) and std > 0 else float("nan")

    values_np = clean.to_numpy(dtype=float)
    demeaned = values_np - mean
    n = len(values_np)
    max_lag = min(max(int(hac_lags), 0), max(n - 1, 0))
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    hac_se = math.sqrt(long_run_variance / n) if n else float("nan")
    hac_t_stat = mean / hac_se if math.isfinite(hac_se) and hac_se > 0 else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "ir": ir if math.isfinite(ir) else None,
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "hac_t_stat": hac_t_stat if math.isfinite(hac_t_stat) else None,
        "positive_ratio": float((clean > 0).mean()),
    }
''',
        "HAC summary statistics",
    )

    text = replace_exact(
        text,
        '''def _long_short_series(
    signal: pd.Series,
    returns: pd.Series,
    *,
    min_assets: int,
    n_quantiles: int,
) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1)
    spreads: dict[pd.Timestamp, float] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    previous_top: set[str] | None = None
    previous_bottom: set[str] | None = None
    for date, group in joined.groupby(level="datetime", sort=True):
        valid = group.dropna()
        if len(valid) < max(min_assets, n_quantiles * 2):
            continue
        ranked = valid["signal"].rank(method="average", pct=True)
        low_cut = 1.0 / n_quantiles
        high_cut = 1.0 - low_cut
        bottom_mask = ranked <= low_cut
        top_mask = ranked > high_cut
        if not bottom_mask.any() or not top_mask.any():
            continue
        spreads[pd.Timestamp(date)] = float(valid.loc[top_mask, "return"].mean() - valid.loc[bottom_mask, "return"].mean())
        top = set(valid.index.get_level_values("asset")[np.asarray(top_mask, dtype=bool)])
        bottom = set(valid.index.get_level_values("asset")[np.asarray(bottom_mask, dtype=bool)])
        if previous_top is not None and previous_bottom is not None:
            top_turn = 1.0 - len(top & previous_top) / max(len(top | previous_top), 1)
            bottom_turn = 1.0 - len(bottom & previous_bottom) / max(len(bottom | previous_bottom), 1)
            turnovers[pd.Timestamp(date)] = float((top_turn + bottom_turn) / 2)
        previous_top, previous_bottom = top, bottom
    return pd.Series(spreads, name="long_short").sort_index(), pd.Series(turnovers, name="turnover").sort_index()
''',
        '''def _long_short_series(
    signal: pd.Series,
    returns: pd.Series,
    *,
    min_assets: int,
    n_quantiles: int,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build equal-weight gross/net long-short returns and weight turnover."""
    joined = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1)
    gross_returns: dict[pd.Timestamp, float] = {}
    net_returns: dict[pd.Timestamp, float] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    previous_weights: dict[str, float] = {}
    for date, group in joined.groupby(level="datetime", sort=True):
        valid = group.dropna()
        if len(valid) < max(min_assets, n_quantiles * 2):
            continue
        ranked = valid["signal"].rank(method="average", pct=True)
        low_cut = 1.0 / n_quantiles
        high_cut = 1.0 - low_cut
        bottom_mask = np.asarray(ranked <= low_cut, dtype=bool)
        top_mask = np.asarray(ranked > high_cut, dtype=bool)
        if not bottom_mask.any() or not top_mask.any():
            continue
        assets = valid.index.get_level_values("asset").astype(str)
        top_assets = list(assets[top_mask])
        bottom_assets = list(assets[bottom_mask])
        weights = {
            **{asset: 1.0 / len(top_assets) for asset in top_assets},
            **{asset: -1.0 / len(bottom_assets) for asset in bottom_assets},
        }
        realized = valid["return"].to_dict()
        gross = float(sum(weights[str(asset)] * float(value) for asset, value in realized.items() if str(asset) in weights))
        all_assets = set(weights) | set(previous_weights)
        turnover = 0.5 * sum(
            abs(weights.get(asset, 0.0) - previous_weights.get(asset, 0.0))
            for asset in all_assets
        )
        transaction_cost = turnover * float(cost_bps) / 10_000.0
        timestamp = pd.Timestamp(date)
        gross_returns[timestamp] = gross
        net_returns[timestamp] = gross - transaction_cost
        turnovers[timestamp] = float(turnover)
        previous_weights = weights
    return (
        pd.Series(gross_returns, name="long_short_gross").sort_index(),
        pd.Series(net_returns, name="long_short_net").sort_index(),
        pd.Series(turnovers, name="turnover").sort_index(),
    )
''',
        "weight turnover and transaction costs",
    )

    text = replace_exact(
        text,
        '''def _route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    test = primary.get("test", {})
    rank_ic = ((test.get("rank_ic") or {}).get("mean"))
    icir = ((test.get("rank_ic") or {}).get("ir"))
    sharpe = ((test.get("long_short") or {}).get("sharpe"))
''',
        '''def _route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    test = primary.get("test", {})
    rank_ic = ((test.get("rank_ic") or {}).get("mean"))
    icir = ((test.get("rank_ic") or {}).get("ir"))
    sharpe = ((test.get("long_short_net") or {}).get("sharpe"))
''',
        "route on net performance",
    )

    text = replace_exact(
        text,
        '''def _evaluate_one(
    factor: FactorDefinition,
    raw: pd.Series,
    forward_returns: Mapping[int, pd.Series],
    bounds: SplitBoundaries,
    config: BatchEvaluationConfig,
''',
        '''def _evaluate_one(
    factor: FactorDefinition,
    raw: pd.Series,
    forward_returns: Mapping[int, pd.Series],
    bounds: SplitBoundaries,
    trading_dates: pd.DatetimeIndex,
    config: BatchEvaluationConfig,
''',
        "evaluate one signature",
    )
    text = replace_exact(
        text,
        '''    primary_forward = forward_returns[min(config.horizons)].reindex(purified.index)
    train_positions = np.flatnonzero(_split_mask(purified.index, "train", bounds))
''',
        '''    primary_horizon = min(config.horizons)
    primary_forward = forward_returns[primary_horizon].reindex(purified.index)
    train_positions = np.flatnonzero(
        _purged_split_mask(
            purified.index,
            "train",
            bounds,
            trading_dates,
            entry_lag=config.entry_lag,
            horizon=primary_horizon,
        )
    )
''',
        "training direction purged split",
    )
    text = replace_exact(
        text,
        '''        for split in ("train", "valid", "test", "all"):
            mask = _split_mask(signal.index, split, bounds)
            split_signal = signal.iloc[np.flatnonzero(mask)]
            split_returns = future.reindex(signal.index).iloc[np.flatnonzero(mask)]
            daily = _daily_ic(split_signal, split_returns, min_assets=config.min_assets)
            long_short, turnover = _long_short_series(
                split_signal,
                split_returns,
                min_assets=config.min_assets,
                n_quantiles=config.n_quantiles,
            )
            horizon_metrics[split] = {
                "ic": _summary_stats(daily.get("ic", pd.Series(dtype=float))),
                "rank_ic": _summary_stats(daily.get("rank_ic", pd.Series(dtype=float))),
                "long_short": _performance_stats(long_short, annualization=annualization),
                "turnover_mean": float(turnover.mean()) if not turnover.empty else None,
                "coverage": float(pd.concat([split_signal, split_returns], axis=1).dropna().shape[0] / max(len(split_signal), 1)),
            }
''',
        '''        for split in ("train", "valid", "test", "all"):
            mask = _purged_split_mask(
                signal.index,
                split,
                bounds,
                trading_dates,
                entry_lag=config.entry_lag,
                horizon=int(horizon),
            )
            positions = np.flatnonzero(mask)
            split_signal = signal.iloc[positions]
            split_returns = future.reindex(signal.index).iloc[positions]
            daily = _daily_ic(split_signal, split_returns, min_assets=config.min_assets)
            long_short_gross, long_short_net, turnover = _long_short_series(
                split_signal,
                split_returns,
                min_assets=config.min_assets,
                n_quantiles=config.n_quantiles,
                cost_bps=config.cost_bps,
            )
            horizon_metrics[split] = {
                "ic": _summary_stats(
                    daily.get("ic", pd.Series(dtype=float)),
                    hac_lags=max(int(horizon) - 1, 0),
                ),
                "rank_ic": _summary_stats(
                    daily.get("rank_ic", pd.Series(dtype=float)),
                    hac_lags=max(int(horizon) - 1, 0),
                ),
                "long_short_gross": _performance_stats(
                    long_short_gross,
                    annualization=annualization,
                ),
                "long_short_net": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                ),
                "long_short": _performance_stats(
                    long_short_net,
                    annualization=annualization,
                ),
                "cost_bps": float(config.cost_bps),
                "turnover_mean": float(turnover.mean()) if not turnover.empty else None,
                "coverage": float(
                    pd.concat([split_signal, split_returns], axis=1).dropna().shape[0]
                    / max(len(split_signal), 1)
                ),
            }
''',
        "purged horizon metrics and net returns",
    )
    text = replace_exact(
        text,
        '''                "test_long_short_sharpe": ((test.get("long_short") or {}).get("sharpe")),
                "test_long_short_annual_return": ((test.get("long_short") or {}).get("annual_return")),
                "test_long_short_max_drawdown": ((test.get("long_short") or {}).get("max_drawdown")),
''',
        '''                "test_long_short_gross_sharpe": ((test.get("long_short_gross") or {}).get("sharpe")),
                "test_long_short_net_sharpe": ((test.get("long_short_net") or {}).get("sharpe")),
                "test_long_short_net_annual_return": ((test.get("long_short_net") or {}).get("annual_return")),
                "test_long_short_net_max_drawdown": ((test.get("long_short_net") or {}).get("max_drawdown")),
''',
        "ranking net columns",
    )
    text = replace_exact(
        text,
        '''            row.get("test_long_short_sharpe") if row.get("test_long_short_sharpe") is not None else -999,
''',
        '''            row.get("test_long_short_net_sharpe") if row.get("test_long_short_net_sharpe") is not None else -999,
''',
        "ranking net sort",
    )

    insert_anchor = '''def run_gtja185_evaluation(
    config: BatchEvaluationConfig,
    *,
    output_dir: str | Path,
    market_frame: pd.DataFrame | None = None,
    snapshot_id: str | None = None,
) -> BatchRunSummary:
'''
    replacement = '''def _load_point_in_time_universe(
    config: BatchEvaluationConfig,
) -> tuple[pd.DataFrame, str]:
    if not config.universe_dataset:
        raise ValueError("universe_dataset is required")
    from data_access import get_store

    store = get_store()
    dataset = store.get_dataset(config.universe_dataset)
    columns = [
        dataset.time_column,
        dataset.instrument_column,
        config.universe_membership_field,
    ]
    if config.tradability_field:
        columns.append(config.tradability_field)
    result = store.read_result(
        config.universe_dataset,
        columns=list(dict.fromkeys(columns)),
        time_range=(config.start_date, config.end_date)
        if config.start_date or config.end_date
        else None,
        instrument_filter=list(config.instrument_filter)
        if config.instrument_filter
        else None,
    )
    frame = result.table.to_pandas().rename(
        columns={
            dataset.time_column: "datetime",
            dataset.instrument_column: "asset",
            config.universe_membership_field: "is_member",
            **(
                {config.tradability_field: "is_tradable"}
                if config.tradability_field
                else {}
            ),
        }
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["asset"] = frame["asset"].astype(str)
    if frame.duplicated(["datetime", "asset"]).any():
        raise ValueError(
            f"universe dataset {config.universe_dataset} contains duplicate keys"
        )
    frame["is_member"] = frame["is_member"].fillna(False).astype(bool)
    if "is_tradable" in frame.columns:
        frame["is_tradable"] = frame["is_tradable"].fillna(False).astype(bool)
    return frame, str(result.snapshot.snapshot_id)


def _apply_point_in_time_universe(
    market_frame: pd.DataFrame,
    universe_frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {"datetime", "asset", "is_member"}
    missing = sorted(required - set(universe_frame.columns))
    if missing:
        raise ValueError(f"universe frame missing required columns: {missing}")
    universe = universe_frame.copy()
    universe["datetime"] = pd.to_datetime(universe["datetime"], errors="raise")
    universe["asset"] = universe["asset"].astype(str)
    if universe.duplicated(["datetime", "asset"]).any():
        raise ValueError("universe frame contains duplicate (datetime, asset) keys")
    columns = ["datetime", "asset", "is_member"]
    if "is_tradable" in universe.columns:
        columns.append("is_tradable")
    merged = market_frame.merge(
        universe[columns],
        on=["datetime", "asset"],
        how="left",
        validate="one_to_one",
    )
    eligible = merged["is_member"].fillna(False).astype(bool)
    if "is_tradable" in merged.columns:
        eligible &= merged["is_tradable"].fillna(False).astype(bool)
    filtered = merged.loc[eligible, market_frame.columns].copy()
    if filtered.empty:
        raise ValueError("point-in-time universe filter removed all market rows")
    return filtered.sort_values(["asset", "datetime"]).reset_index(drop=True)


def run_gtja185_evaluation(
    config: BatchEvaluationConfig,
    *,
    output_dir: str | Path,
    market_frame: pd.DataFrame | None = None,
    snapshot_id: str | None = None,
    universe_frame: pd.DataFrame | None = None,
    universe_snapshot_id: str | None = None,
) -> BatchRunSummary:
'''
    text = replace_exact(text, insert_anchor, replacement, "PIT universe helpers and run signature")

    text = replace_exact(
        text,
        '''    market_frame = _normalize_market_frame(market_frame)
    bounds = _split_boundaries(market_frame, config)
''',
        '''    market_frame = _normalize_market_frame(market_frame)
    if config.require_point_in_time_universe:
        if universe_frame is None:
            universe_frame, universe_snapshot_id = _load_point_in_time_universe(config)
        market_frame = _apply_point_in_time_universe(market_frame, universe_frame)
        snapshot_id = (
            f"market={snapshot_id}|universe={universe_snapshot_id or 'provided'}"
        )
    trading_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(market_frame["datetime"]).dt.normalize().unique())
    )
    bounds = _split_boundaries(market_frame, config)
''',
        "apply PIT universe and trading calendar",
    )
    text = replace_exact(
        text,
        '''    forward_returns = _price_forward_returns(
        market_frame,
        config.horizons,
        config.forward_price_field,
    )
''',
        '''    forward_returns = _price_forward_returns(
        market_frame,
        config.horizons,
        config.forward_price_field,
        entry_lag=config.entry_lag,
    )
''',
        "forward return call",
    )
    text = replace_exact(
        text,
        '''                forward_returns,
                bounds,
                config,
''',
        '''                forward_returns,
                bounds,
                trading_dates,
                config,
''',
        "evaluate one call",
    )

    text = replace_exact(
        text,
        '''    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
''',
        '''    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--entry-lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--universe-dataset", default=None)
    parser.add_argument("--universe-membership-field", default="is_member")
    parser.add_argument("--tradability-field", default="is_tradable")
    parser.add_argument("--require-point-in-time-universe", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
''',
        "evaluation CLI arguments",
    )
    text = replace_exact(
        text,
        '''        n_quantiles=args.n_quantiles,
        materialize_staging=args.materialize_staging,
''',
        '''        n_quantiles=args.n_quantiles,
        entry_lag=args.entry_lag,
        cost_bps=args.cost_bps,
        universe_dataset=args.universe_dataset,
        universe_membership_field=args.universe_membership_field,
        tradability_field=args.tradability_field or None,
        require_point_in_time_universe=args.require_point_in_time_universe,
        materialize_staging=args.materialize_staging,
''',
        "evaluation CLI config",
    )

    EVALUATOR.write_text(text, encoding="utf-8")


def add_tests() -> None:
    TESTS.write_text(
        '''from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.gtja185_batch import (
    BatchEvaluationConfig,
    SplitBoundaries,
    _apply_point_in_time_universe,
    _long_short_series,
    _price_forward_returns,
    _purged_split_mask,
)


def _price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=8)
    return pd.DataFrame(
        {
            "datetime": dates,
            "asset": "A",
            "vwap": np.arange(100.0, 108.0),
        }
    )


def test_forward_return_enters_on_next_bar():
    frame = _price_frame()
    result = _price_forward_returns(frame, (2,), "vwap", entry_lag=1)[2]
    # signal at t0 enters at t1=101 and exits at t3=103
    assert result.iloc[0] == pytest.approx(103.0 / 101.0 - 1.0)
    assert result.iloc[-3:].isna().all()


def test_purged_split_never_crosses_boundary():
    dates = pd.bdate_range("2024-01-02", periods=10)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "asset"])
    bounds = SplitBoundaries(
        first_date=str(dates[0].date()),
        train_end=str(dates[4].date()),
        valid_end=str(dates[7].date()),
        last_date=str(dates[-1].date()),
    )
    train = _purged_split_mask(index, "train", bounds, dates, entry_lag=1, horizon=2)
    valid = _purged_split_mask(index, "valid", bounds, dates, entry_lag=1, horizon=2)
    test = _purged_split_mask(index, "test", bounds, dates, entry_lag=1, horizon=2)
    assert list(np.flatnonzero(train)) == [0, 1]
    assert list(np.flatnonzero(valid)) == []
    assert list(np.flatnonzero(test)) == []


def test_turnover_uses_weights_and_costs_net_returns():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C", "D"]],
        names=["datetime", "asset"],
    )
    signal = pd.Series([-2, -1, 1, 2] * 2, index=index, dtype=float)
    returns = pd.Series([0.0, 0.0, 0.01, 0.01] * 2, index=index, dtype=float)
    gross, net, turnover = _long_short_series(
        signal,
        returns,
        min_assets=4,
        n_quantiles=2,
        cost_bps=10.0,
    )
    assert turnover.iloc[0] == pytest.approx(1.0)
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.001)
    assert net.iloc[1] == pytest.approx(gross.iloc[1])


def test_point_in_time_universe_filters_by_date_and_tradability():
    market = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
            "asset": ["A", "B", "A", "B"],
            "close": [1.0, 2.0, 1.1, 2.1],
        }
    )
    universe = pd.DataFrame(
        {
            "datetime": market["datetime"],
            "asset": market["asset"],
            "is_member": [True, False, True, True],
            "is_tradable": [True, True, False, True],
        }
    )
    filtered = _apply_point_in_time_universe(market, universe)
    assert filtered[["datetime", "asset"]].to_records(index=False).tolist() == [
        (np.datetime64("2024-01-02T00:00:00.000000000"), "A"),
        (np.datetime64("2024-01-03T00:00:00.000000000"), "B"),
    ]


def test_production_requires_registered_point_in_time_universe():
    with pytest.raises(ValueError, match="require_point_in_time_universe"):
        BatchEvaluationConfig(run_mode="production").validate()
    with pytest.raises(ValueError, match="universe_dataset"):
        BatchEvaluationConfig(
            run_mode="production",
            require_point_in_time_universe=True,
        ).validate()
''',
        encoding="utf-8",
    )


def patch_docs() -> None:
    text = DOCS.read_text(encoding="utf-8")
    old = '''5. Build forward returns from the configured price field, defaulting to real `vwap`.
6. Split dates chronologically into train, validation and test samples.
7. Choose signal direction from **training RankIC only**. Validation and test observations
   never participate in sign selection or parameter fitting.
8. Report Pearson IC, RankIC, ICIR, t-statistic, positive ratio, coverage, quantile
   long-short return, turnover, annualized return/volatility, Sharpe, max drawdown, hit
   rate and yearly stability for every configured horizon.
9. Produce deterministic routing recommendations (`tier3a_core`, `tier3b_satellite`,
'''
    new = '''5. Build forward returns from the configured price field, defaulting to real `vwap`.
   A signal observed at `t` enters no earlier than `t+1`; same-bar execution is forbidden.
6. Split dates chronologically into train, validation and test samples and purge any row
   whose label exit date crosses the next split boundary.
7. Choose signal direction from **training RankIC only**. Validation and test observations
   never participate in sign selection or parameter fitting.
8. Report Pearson IC, RankIC, ICIR, ordinary and Newey-West/HAC t-statistics, positive
   ratio, coverage, equal-weight quantile long-short gross and transaction-cost-adjusted
   net returns, weight turnover, annualized return/volatility, Sharpe, max drawdown, hit
   rate and yearly stability for every configured horizon. Routing uses net performance.
9. Production mode requires a registered point-in-time universe dataset with explicit
   membership and optional tradability flags; missing membership fails closed.
10. Produce deterministic routing recommendations (`tier3a_core`, `tier3b_satellite`,
'''
    if old not in text:
        raise RuntimeError("documentation evaluation path anchor not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        '''10. Optionally upsert factor values to `factor_lake_staging`; publication remains a separate,
''',
        '''11. Optionally upsert factor values to `factor_lake_staging`; publication remains a separate,
''',
        1,
    )
    DOCS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_evaluator()
    add_tests()
    patch_docs()
    print("GTJA185 evaluation economics and PIT universe contracts hardened")


if __name__ == "__main__":
    main()
