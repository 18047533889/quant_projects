"""Production runner for any versioned external factor pack."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from integrations.quant_platform import (
    bootstrap_quant_platform,
    load_market_frame,
    materialize_factor_to_staging,
)
from .batch_engine import execute_pack, prepare_pack
from .batch_metrics import (
    apply_point_in_time_universe,
    apply_validation_fdr,
    daily_ic,
    long_short_series,
    normalize_market_frame,
    performance_stats,
    price_forward_returns,
    purged_split_mask,
    purify_series,
    ranking_rows,
    route_factor,
    split_boundaries,
    summary_stats,
)
from .batch_models import BatchEvaluationConfig, BatchRunSummary, FactorRunRecord, SplitBoundaries
from .factor_pack import FactorDefinition, FactorPack

bootstrap_quant_platform()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_synthetic_market_frame(
    *,
    periods: int = 320,
    symbols: int = 8,
    seed: int = 191,
) -> pd.DataFrame:
    if periods < 250:
        raise ValueError("synthetic panel requires at least 250 periods")
    if symbols < 6:
        raise ValueError("synthetic panel requires at least 6 symbols")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=periods)
    common = rng.normal(0.0003, 0.008, size=periods)
    frames: list[pd.DataFrame] = []
    for index in range(symbols):
        idiosyncratic = rng.normal(0.0, 0.012 + index * 0.0003, size=periods)
        returns = common * (0.7 + index / max(symbols, 1) * 0.3) + idiosyncratic
        close = (30.0 + index * 3.0) * np.exp(np.cumsum(returns))
        open_price = close * np.exp(rng.normal(0.0, 0.003, size=periods))
        spread = np.abs(rng.normal(0.008, 0.003, size=periods))
        high = np.maximum(open_price, close) * (1.0 + spread)
        low = np.minimum(open_price, close) * (1.0 - spread)
        volume = rng.lognormal(13.5 + index * 0.02, 0.35, size=periods)
        vwap = (open_price + high + low + close) / 4.0
        frames.append(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "asset": f"S{index:04d}",
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "preclose": np.r_[np.nan, close[:-1]],
                    "volume": volume,
                    "amount": volume * vwap,
                    "ret": np.r_[np.nan, close[1:] / close[:-1] - 1.0],
                    "vwap": vwap,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["asset", "datetime"]).reset_index(drop=True)


def _load_point_in_time_universe(config: BatchEvaluationConfig) -> tuple[pd.DataFrame, str]:
    dataset_name = config.universe_dataset
    if not dataset_name and config.market == "ashare":
        dataset_name = "ashare_universe_daily"
    if not dataset_name:
        raise ValueError("production evaluation requires a registered universe_dataset")
    from data_access import get_store

    store = get_store()
    dataset = store.get_dataset(dataset_name)
    columns = [
        dataset.time_column,
        dataset.instrument_column,
        config.universe_membership_field,
    ]
    if config.tradability_field:
        columns.append(config.tradability_field)
    parameters: dict[str, Any] = {}
    parameter_names = set(getattr(dataset, "params_schema", {}) or {})
    if "universe_id" in parameter_names:
        parameters["universe_id"] = str(config.universe_id)
    unsupported = parameter_names - {"universe_id"}
    if unsupported:
        raise ValueError(
            f"universe dataset {dataset_name} has unsupported parameters: {sorted(unsupported)}"
        )
    read = store.read_result(
        dataset_name,
        columns=list(dict.fromkeys(columns)),
        time_range=(config.start_date, config.end_date)
        if config.start_date or config.end_date
        else None,
        instrument_filter=list(config.instrument_filter)
        if config.instrument_filter
        else None,
        **parameters,
    )
    rename = {
        dataset.time_column: "datetime",
        dataset.instrument_column: "asset",
        config.universe_membership_field: "is_member",
    }
    if config.tradability_field:
        rename[config.tradability_field] = "is_tradable"
    frame = read.table.to_pandas().rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["asset"] = frame["asset"].astype(str)
    if frame.duplicated(["datetime", "asset"]).any():
        raise ValueError(f"universe dataset {dataset_name} contains duplicate keys")
    frame["is_member"] = frame["is_member"].fillna(False).astype(bool)
    if "is_tradable" in frame.columns:
        frame["is_tradable"] = frame["is_tradable"].fillna(False).astype(bool)
    return frame, str(read.snapshot.snapshot_id)


def _resume_record(
    path: Path,
    *,
    formula_hash: str,
    snapshot_id: str,
    config_hash: str,
) -> FactorRunRecord | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("formula_hash") != formula_hash:
        return None
    if payload.get("snapshot_id") != snapshot_id:
        return None
    if payload.get("config_hash") != config_hash:
        return None
    record = payload.get("record")
    if not isinstance(record, dict) or record.get("status") != "success":
        return None
    return FactorRunRecord(**record)


def _evaluate_one(
    definition: FactorDefinition,
    raw: pd.Series,
    forward_returns: Mapping[int, pd.Series],
    bounds: SplitBoundaries,
    trading_dates: pd.DatetimeIndex,
    config: BatchEvaluationConfig,
    *,
    compile_seconds: float,
    execute_seconds: float,
) -> FactorRunRecord:
    purified = purify_series(
        raw,
        mad_multiplier=config.winsor_mad,
        min_assets=config.min_assets,
    )
    finite_values = int(
        np.isfinite(pd.to_numeric(purified, errors="coerce").to_numpy(dtype=float)).sum()
    )
    coverage = finite_values / max(len(purified), 1)
    primary_horizon = min(config.horizons)
    primary_forward = forward_returns[primary_horizon].reindex(purified.index)
    train_positions = np.flatnonzero(
        purged_split_mask(
            purified.index,
            "train",
            bounds,
            trading_dates,
            entry_lag=config.entry_lag,
            horizon=primary_horizon,
        )
    )
    training_ic = daily_ic(
        purified.iloc[train_positions],
        primary_forward.iloc[train_positions],
        min_assets=config.min_assets,
    )
    direction_mean = pd.to_numeric(
        training_ic.get("rank_ic", pd.Series(dtype=float)),
        errors="coerce",
    ).mean()
    direction = -1 if pd.notna(direction_mean) and float(direction_mean) < 0 else 1
    signal = purified * direction
    metrics: dict[str, Any] = {}
    for horizon, future in forward_returns.items():
        future = future.reindex(signal.index)
        horizon_metrics: dict[str, Any] = {}
        annualization = config.annualization / max(int(horizon), 1)
        for split in ("train", "valid", "test", "all"):
            positions = np.flatnonzero(
                purged_split_mask(
                    signal.index,
                    split,
                    bounds,
                    trading_dates,
                    entry_lag=config.entry_lag,
                    horizon=int(horizon),
                )
            )
            split_signal = signal.iloc[positions]
            split_returns = future.iloc[positions]
            daily = daily_ic(split_signal, split_returns, min_assets=config.min_assets)
            gross, net, turnover = long_short_series(
                split_signal,
                split_returns,
                min_assets=config.min_assets,
                n_quantiles=config.n_quantiles,
                cost_bps=config.cost_bps,
            )
            paired_count = pd.concat(
                [split_signal.rename("signal"), split_returns.rename("return")],
                axis=1,
            ).dropna().shape[0]
            horizon_metrics[split] = {
                "ic": summary_stats(
                    daily.get("ic", pd.Series(dtype=float)),
                    hac_lags=max(int(horizon) - 1, 0),
                ),
                "rank_ic": summary_stats(
                    daily.get("rank_ic", pd.Series(dtype=float)),
                    hac_lags=max(int(horizon) - 1, 0),
                ),
                "long_short_gross": performance_stats(
                    gross,
                    annualization=annualization,
                    hac_lags=max(int(horizon) - 1, 0),
                    holding_period=int(horizon),
                ),
                "long_short_net": performance_stats(
                    net,
                    annualization=annualization,
                    hac_lags=max(int(horizon) - 1, 0),
                    holding_period=int(horizon),
                ),
                "cost_bps": float(config.cost_bps),
                "turnover_mean": float(turnover.mean()) if not turnover.empty else None,
                "coverage": paired_count / max(len(split_signal), 1),
            }
        metrics[str(int(horizon))] = horizon_metrics
    record = FactorRunRecord(
        factor_name=definition.name,
        formula_hash=definition.formula_hash,
        status="success",
        compile_seconds=round(float(compile_seconds), 6),
        execute_seconds=round(float(execute_seconds), 6),
        finite_values=finite_values,
        coverage=coverage,
        direction=direction,
        metrics=metrics,
    )
    record.route = route_factor(metrics, coverage=coverage)
    return record


def run_factor_pack_evaluation(
    pack: FactorPack,
    config: BatchEvaluationConfig,
    *,
    output_dir: str | Path,
    market_frame: pd.DataFrame | None = None,
    snapshot_id: str | None = None,
    universe_frame: pd.DataFrame | None = None,
    universe_snapshot_id: str | None = None,
) -> BatchRunSummary:
    pack.validate()
    config.validate()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    selected = pack.select(config.selected_factors or None, limit=config.limit)
    output = Path(output_dir)
    report_dir = output / "factor_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if market_frame is None:
        market_frame, snapshot = load_market_frame(
            market=config.market,
            dataset=config.dataset,
            fields=["open", "high", "low", "close", "preclose", "volume", "amount", "ret", "vwap"],
            start_date=config.start_date,
            end_date=config.end_date,
            instrument_filter=config.instrument_filter or None,
        )
        snapshot_id = str(snapshot.snapshot_id)
    else:
        snapshot_id = str(
            snapshot_id
            or f"synthetic:{_canonical_hash({'rows': len(market_frame), 'columns': list(market_frame.columns)})[:16]}"
        )
    market_frame = normalize_market_frame(market_frame)
    if config.require_point_in_time_universe:
        if universe_frame is None:
            universe_frame, universe_snapshot_id = _load_point_in_time_universe(config)
        market_frame = apply_point_in_time_universe(
            market_frame,
            universe_frame,
            universe_id=config.universe_id,
        )
        snapshot_id = (
            f"market={snapshot_id}|universe={universe_snapshot_id or 'provided'}"
            f"|universe_id={config.universe_id}"
        )
    trading_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(market_frame["datetime"]).dt.normalize().unique())
    )
    bounds = split_boundaries(market_frame, config)
    config_payload = asdict(config)
    config_hash = _canonical_hash(config_payload)
    run_id = (
        f"{pack.name}_{started_at.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{_canonical_hash(snapshot_id)[:8]}_{config_hash[:8]}"
    )

    resumed: dict[str, FactorRunRecord] = {}
    pending: list[FactorDefinition] = []
    for definition in selected:
        cached = None
        if config.resume:
            cached = _resume_record(
                report_dir / f"{definition.name}.json",
                formula_hash=definition.formula_hash,
                snapshot_id=snapshot_id,
                config_hash=config_hash,
            )
        if cached is None:
            pending.append(definition)
        else:
            resumed[definition.name] = cached

    raw_results: dict[str, pd.Series] = {}
    compile_timings: dict[str, float] = {}
    execute_timings: dict[str, float] = {}
    execution_errors: dict[str, str] = {}
    forward: dict[int, pd.Series] = {}
    if pending:
        engine, parsed, compile_timings = prepare_pack(
            pending,
            market_frame,
            backend=config.backend,
            run_mode=config.run_mode,
            universe=pack.name,
        )
        raw_results, execute_timings, execution_errors = execute_pack(
            engine,
            parsed,
            batch_size=config.batch_size,
            market=config.market,
        )
        forward = price_forward_returns(
            market_frame,
            config.horizons,
            config.forward_price_field,
            entry_lag=config.entry_lag,
        )

    records: list[FactorRunRecord] = []
    for definition in selected:
        if definition.name in resumed:
            records.append(resumed[definition.name])
            continue
        if definition.name in execution_errors or definition.name not in raw_results:
            records.append(
                FactorRunRecord(
                    factor_name=definition.name,
                    formula_hash=definition.formula_hash,
                    status="failed",
                    compile_seconds=compile_timings.get(definition.name, 0.0),
                    execute_seconds=execute_timings.get(definition.name, 0.0),
                    error=execution_errors.get(definition.name, "factor result missing"),
                )
            )
            continue
        try:
            record = _evaluate_one(
                definition,
                raw_results[definition.name],
                forward,
                bounds,
                trading_dates,
                config,
                compile_seconds=compile_timings.get(definition.name, 0.0),
                execute_seconds=execute_timings.get(definition.name, 0.0),
            )
            if config.materialize_staging:
                version = config.factor_version or f"{pack.name}.{pack.version}"
                record.staging = materialize_factor_to_staging(
                    raw_results[definition.name],
                    factor_id=definition.name,
                    factor_version=version,
                    snapshot_id=snapshot_id,
                    publish=config.publish,
                )
            records.append(record)
        except Exception as error:
            records.append(
                FactorRunRecord(
                    factor_name=definition.name,
                    formula_hash=definition.formula_hash,
                    status="failed",
                    compile_seconds=compile_timings.get(definition.name, 0.0),
                    execute_seconds=execute_timings.get(definition.name, 0.0),
                    error=f"{type(error).__name__}: {error}",
                )
            )

    apply_validation_fdr(records, alpha=config.fdr_alpha)
    for record in records:
        _json_dump(
            report_dir / f"{record.factor_name}.json",
            {
                "factor_name": record.factor_name,
                "formula_hash": record.formula_hash,
                "snapshot_id": snapshot_id,
                "config_hash": config_hash,
                "record": asdict(record),
            },
        )
    ranking = ranking_rows(records)
    ranking_frame = pd.DataFrame(ranking)
    ranking_frame.to_csv(output / "ranking.csv", index=False)
    ranking_frame.to_parquet(output / "ranking.parquet", index=False)

    successful = [record for record in records if record.status == "success"]
    failed = [record for record in records if record.status != "success"]
    status = "success" if not failed else ("partial" if successful else "failed")
    finished_at = datetime.now(timezone.utc)
    summary = BatchRunSummary(
        run_id=run_id,
        status=status,
        pack_name=pack.name,
        pack_version=pack.version,
        pack_hash=pack.pack_hash,
        source_hash=pack.source_hash,
        snapshot_id=snapshot_id,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        elapsed_seconds=round(time.perf_counter() - started_clock, 3),
        factor_count=len(selected),
        succeeded=len(successful),
        failed=len(failed),
        skipped=len(resumed),
        output_dir=str(output.resolve()),
        config_hash=config_hash,
        split_boundaries=asdict(bounds),
        route_counts=dict(Counter(record.route for record in successful)),
        records=[asdict(record) for record in records],
    )
    _json_dump(output / "summary.json", asdict(summary))
    _json_dump(
        output / "run_manifest.json",
        {
            "run_id": run_id,
            "pack": {
                "name": pack.name,
                "version": pack.version,
                "pack_hash": pack.pack_hash,
                "source_hash": pack.source_hash,
                "factor_count": len(selected),
            },
            "data_snapshot_id": snapshot_id,
            "config": config_payload,
            "config_hash": config_hash,
            "split_boundaries": asdict(bounds),
            "route_counts": summary.route_counts,
            "created_at": finished_at.isoformat(),
        },
    )
    if config.strict and failed:
        sample = [f"{record.factor_name}: {record.error}" for record in failed[:10]]
        raise RuntimeError(
            f"factor-pack evaluation failed for {len(failed)}/{len(selected)} factors: {sample}"
        )
    return summary
