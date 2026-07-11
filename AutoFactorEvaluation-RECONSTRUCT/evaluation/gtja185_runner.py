"""Production runner for the versioned GTJA185 factor pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_packs.gtja185 import FactorDefinition, load_gtja185_pack
from integrations.quant_platform import (
    InMemoryFrameSource,
    bootstrap_quant_platform,
    load_market_frame,
    materialize_factor_to_staging,
)

from .gtja185_metrics import (
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
from .gtja185_models import (
    BatchEvaluationConfig,
    BatchRunSummary,
    FactorRunRecord,
    SplitBoundaries,
)

bootstrap_quant_platform()

from api.dsl_parser import parse_factor  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402


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


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_synthetic_market_frame(
    *,
    periods: int = 420,
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
    return pd.concat(frames, ignore_index=True).sort_values(
        ["asset", "datetime"]
    ).reset_index(drop=True)


def _frame_source(frame: pd.DataFrame) -> InMemoryFrameSource:
    return InMemoryFrameSource(
        frame,
        time_column="datetime",
        instrument_column="asset",
    )


def _compile_pack(
    factors: Sequence[FactorDefinition],
    frame: pd.DataFrame,
    *,
    backend: str,
    run_mode: str,
) -> tuple[FactorEngine, list[Any], dict[str, float]]:
    engine = FactorEngine(
        build_backend(backend),
        _frame_source(frame),
        run_mode=run_mode,
    )
    parsed: list[Any] = []
    timings: dict[str, float] = {}
    for definition in factors:
        factor = parse_factor(
            definition.formula,
            name=definition.name,
            freq="1d",
            universe="GTJA185",
            description=definition.description,
        )
        started = time.perf_counter()
        engine.compile(
            factor,
            pit_enforce=True,
            pit_forbid_forward_fill=True,
        )
        timings[definition.name] = time.perf_counter() - started
        parsed.append(factor)
    return engine, parsed, timings


def _coerce_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        result = value.rename(name)
    elif isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        result = value.iloc[:, 0].rename(name)
    elif hasattr(value, "to_pandas"):
        converted = value.to_pandas()
        if not isinstance(converted, pd.Series):
            raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
        result = converted.rename(name)
    else:
        raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
    if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels != 2:
        raise ValueError(f"factor {name} must return MultiIndex(datetime, asset)")
    result.index = result.index.set_names(["datetime", "asset"])
    if result.index.has_duplicates:
        raise ValueError(f"factor {name} returned duplicate (datetime, asset) keys")
    return pd.to_numeric(result, errors="coerce").sort_index()


def _execute_pack(
    engine: FactorEngine,
    factors: Sequence[Any],
    *,
    batch_size: int,
    market: str,
) -> tuple[dict[str, pd.Series], dict[str, float], dict[str, str]]:
    results: dict[str, pd.Series] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    for offset in range(0, len(factors), batch_size):
        batch = list(factors[offset : offset + batch_size])
        started = time.perf_counter()
        try:
            output = engine.run_many(
                batch,
                enable_cse=True,
                input_dq_check=False,
                auto_warmup=False,
                trim_warmup=True,
                market=market,
                pit_enforce=False,
            )
            per_factor = (time.perf_counter() - started) / max(len(batch), 1)
            for factor in batch:
                results[factor.name] = _coerce_series(
                    output["results"][factor.name],
                    factor.name,
                )
                timings[factor.name] = per_factor
        except Exception as batch_error:
            for factor in batch:
                one_started = time.perf_counter()
                try:
                    output = engine.run(
                        factor,
                        input_dq_check=False,
                        auto_warmup=False,
                        trim_warmup=True,
                        market=market,
                        pit_enforce=False,
                    )
                    results[factor.name] = _coerce_series(
                        output["result"],
                        factor.name,
                    )
                    timings[factor.name] = time.perf_counter() - one_started
                except Exception as factor_error:
                    errors[factor.name] = (
                        f"batch={type(batch_error).__name__}: {batch_error}; "
                        f"single={type(factor_error).__name__}: {factor_error}"
                    )
    return results, timings, errors


def _load_point_in_time_universe(
    config: BatchEvaluationConfig,
) -> tuple[pd.DataFrame, str]:
    dataset_name = config.universe_dataset
    if not dataset_name and config.market == "ashare":
        dataset_name = "ashare_universe_daily"
    if not dataset_name:
        raise ValueError("production evaluation requires a registered universe_dataset")
    from data_access import get_store

    store = get_store()
    dataset = store.get_dataset(dataset_name)
    schema = dict(getattr(dataset, "schema", None) or {})
    columns = [
        dataset.time_column,
        dataset.instrument_column,
        config.universe_membership_field,
    ]
    if config.tradability_field:
        columns.append(config.tradability_field)
    include_universe_id = "universe_id" in schema
    if include_universe_id:
        columns.append("universe_id")
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
    if include_universe_id:
        frame = frame.loc[
            frame["universe_id"].astype(str) == str(config.universe_id)
        ].copy()
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


def _factor_report_payload(
    record: FactorRunRecord,
    *,
    snapshot_id: str,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "factor_name": record.factor_name,
        "formula_hash": record.formula_hash,
        "snapshot_id": snapshot_id,
        "config_hash": config_hash,
        "record": asdict(record),
    }


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
    finite = np.isfinite(pd.to_numeric(purified, errors="coerce").to_numpy(dtype=float))
    finite_values = int(finite.sum())
    coverage = finite_values / max(len(purified), 1)
    primary_horizon = min(config.horizons)
    primary_forward = forward_returns[primary_horizon].reindex(purified.index)
    training_positions = np.flatnonzero(
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
        purified.iloc[training_positions],
        primary_forward.iloc[training_positions],
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
            daily = daily_ic(
                split_signal,
                split_returns,
                min_assets=config.min_assets,
            )
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
            horizon_metrics[split]["long_short"] = horizon_metrics[split]["long_short_net"]
        by_year: dict[str, Any] = {}
        all_positions = np.flatnonzero(
            purged_split_mask(
                signal.index,
                "all",
                bounds,
                trading_dates,
                entry_lag=config.entry_lag,
                horizon=int(horizon),
            )
        )
        aligned_signal = signal.iloc[all_positions]
        aligned_returns = future.iloc[all_positions]
        years = pd.DatetimeIndex(
            aligned_signal.index.get_level_values("datetime")
        ).year
        for year in sorted(set(int(value) for value in years)):
            year_mask = np.asarray(years == year)
            daily = daily_ic(
                aligned_signal.iloc[np.flatnonzero(year_mask)],
                aligned_returns.iloc[np.flatnonzero(year_mask)],
                min_assets=config.min_assets,
            )
            by_year[str(year)] = summary_stats(
                daily.get("rank_ic", pd.Series(dtype=float)),
                hac_lags=max(int(horizon) - 1, 0),
            )
        horizon_metrics["by_year"] = by_year
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


def run_gtja185_evaluation(
    config: BatchEvaluationConfig,
    *,
    output_dir: str | Path,
    market_frame: pd.DataFrame | None = None,
    snapshot_id: str | None = None,
    universe_frame: pd.DataFrame | None = None,
    universe_snapshot_id: str | None = None,
) -> BatchRunSummary:
    config.validate()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    pack = load_gtja185_pack()
    selected = pack.select(config.selected_factors or None, limit=config.limit)
    output = Path(output_dir)
    report_dir = output / "factor_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if market_frame is None:
        market_frame, snapshot = load_market_frame(
            market=config.market,
            dataset=config.dataset,
            fields=[
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "ret",
                "vwap",
            ],
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
    snapshot_hash = _canonical_hash(snapshot_id)[:8]
    run_id = (
        f"gtja185_{started_at.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{snapshot_hash}_{config_hash[:8]}"
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
        engine, parsed, compile_timings = _compile_pack(
            pending,
            market_frame,
            backend=config.backend,
            run_mode=config.run_mode,
        )
        raw_results, execute_timings, execution_errors = _execute_pack(
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
                record.staging = materialize_factor_to_staging(
                    raw_results[definition.name],
                    factor_id=definition.name,
                    factor_version=config.factor_version,
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
            _factor_report_payload(
                record,
                snapshot_id=snapshot_id,
                config_hash=config_hash,
            ),
        )
    ranking = ranking_rows(records)
    ranking_frame = pd.DataFrame(ranking)
    ranking_frame.to_csv(output / "ranking.csv", index=False)
    ranking_frame.to_parquet(output / "ranking.parquet", index=False)

    successful = [record for record in records if record.status == "success"]
    failed = [record for record in records if record.status != "success"]
    finished_at = datetime.now(timezone.utc)
    status = "success" if not failed else ("partial" if successful else "failed")
    route_counts = dict(Counter(record.route for record in successful))
    summary = BatchRunSummary(
        run_id=run_id,
        status=status,
        pack_name=pack.name,
        pack_version=pack.version,
        pack_hash=pack.pack_hash,
        source_catalog_hash=pack.source_catalog_hash,
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
        route_counts=route_counts,
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
                "source_catalog_hash": pack.source_catalog_hash,
                "factor_count": len(selected),
            },
            "data_snapshot_id": snapshot_id,
            "config": config_payload,
            "config_hash": config_hash,
            "split_boundaries": asdict(bounds),
            "route_counts": route_counts,
            "created_at": finished_at.isoformat(),
        },
    )
    if config.strict and failed:
        sample = [f"{record.factor_name}: {record.error}" for record in failed[:10]]
        raise RuntimeError(
            f"GTJA185 evaluation failed for {len(failed)}/{len(selected)} factors: {sample}"
        )
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the complete versioned GTJA185 factor pack"
    )
    parser.add_argument(
        "--output-dir",
        default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185",
    )
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--universe-id", default="A_SHARE_ALL_A_EX_ST")
    parser.add_argument("--universe-dataset", default=None)
    parser.add_argument("--universe-membership-field", default="is_member")
    parser.add_argument("--tradability-field", default="is_tradable")
    parser.add_argument("--require-point-in-time-universe", action="store_true")
    parser.add_argument("--backend", default="pandas")
    parser.add_argument("--run-mode", choices=["research", "production"], default="research")
    parser.add_argument("--horizons", default="1,5,21")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--entry-lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", action="append", default=[])
    parser.add_argument("--materialize-staging", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=420)
    parser.add_argument("--synthetic-symbols", type=int, default=8)
    arguments = parser.parse_args(argv)

    require_universe = arguments.require_point_in_time_universe
    if arguments.run_mode == "production":
        require_universe = True
    config = BatchEvaluationConfig(
        market=arguments.market,
        dataset=arguments.dataset,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        universe_id=arguments.universe_id,
        universe_dataset=arguments.universe_dataset,
        universe_membership_field=arguments.universe_membership_field,
        tradability_field=arguments.tradability_field or None,
        require_point_in_time_universe=require_universe,
        backend=arguments.backend,
        run_mode=arguments.run_mode,
        horizons=_parse_csv_ints(arguments.horizons),
        batch_size=arguments.batch_size,
        min_assets=arguments.min_assets,
        n_quantiles=arguments.n_quantiles,
        entry_lag=arguments.entry_lag,
        cost_bps=arguments.cost_bps,
        fdr_alpha=arguments.fdr_alpha,
        materialize_staging=arguments.materialize_staging,
        publish=arguments.publish,
        strict=not arguments.allow_partial,
        resume=not arguments.no_resume,
        limit=arguments.limit,
        selected_factors=tuple(arguments.factor),
    )
    frame = None
    snapshot = None
    if arguments.synthetic:
        frame = build_synthetic_market_frame(
            periods=arguments.synthetic_periods,
            symbols=arguments.synthetic_symbols,
        )
        snapshot = "synthetic:gtja185-ci"
    try:
        summary = run_gtja185_evaluation(
            config,
            output_dir=arguments.output_dir,
            market_frame=frame,
            snapshot_id=snapshot,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(_json_safe(asdict(summary)), ensure_ascii=False, indent=2))
    return 0 if summary.status == "success" else 1
