"""End-to-end GTJA185 batch evaluation.

This module is intentionally independent from the legacy directory-moving
workers.  It evaluates a versioned factor pack against one immutable market
snapshot, chooses signal direction on the training sample only, reports
validation/test metrics, and can persist factor values through DataAccess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from factor_packs.gtja185 import FactorDefinition, FactorPack, load_gtja185_pack
from integrations.quant_platform import (
    InMemoryFrameSource,
    bootstrap_quant_platform,
    load_market_frame,
    materialize_factor_to_staging,
)

bootstrap_quant_platform()

from api.dsl_parser import parse_factor  # noqa: E402
from backend.factory import build_backend  # noqa: E402
from runtime.engine import FactorEngine  # noqa: E402


@dataclass(frozen=True)
class BatchEvaluationConfig:
    market: str = "ashare"
    dataset: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    universe_id: str = "A_SHARE_ALL_A_EX_ST"
    instrument_filter: tuple[str, ...] = ()
    backend: str = "pandas"
    run_mode: str = "research"
    horizons: tuple[int, ...] = (1, 5, 21)
    batch_size: int = 8
    min_assets: int = 20
    n_quantiles: int = 5
    winsor_mad: float = 5.0
    train_fraction: float = 0.60
    valid_fraction: float = 0.20
    annualization: int = 252
    forward_price_field: str = "vwap"
    materialize_staging: bool = False
    publish: bool = False
    factor_version: str = "gtja185.v1"
    strict: bool = True
    resume: bool = True
    limit: int | None = None
    selected_factors: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.horizons or any(int(h) <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.min_assets < 2:
            raise ValueError("min_assets must be >= 2")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.valid_fraction < 1:
            raise ValueError("valid_fraction must be in (0, 1)")
        if self.train_fraction + self.valid_fraction >= 1:
            raise ValueError("train_fraction + valid_fraction must be < 1")
        if self.publish and not self.materialize_staging:
            raise ValueError("publish=True requires materialize_staging=True")


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: str
    valid_end: str
    first_date: str
    last_date: str


@dataclass
class FactorRunRecord:
    factor_name: str
    formula_hash: str
    status: str
    compile_seconds: float = 0.0
    execute_seconds: float = 0.0
    finite_values: int = 0
    coverage: float = 0.0
    direction: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    route: str = "rejected"
    staging: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class BatchRunSummary:
    run_id: str
    status: str
    pack_name: str
    pack_version: str
    pack_hash: str
    source_catalog_hash: str
    snapshot_id: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    factor_count: int
    succeeded: int
    failed: int
    skipped: int
    output_dir: str
    config_hash: str
    split_boundaries: dict[str, str]
    records: list[dict[str, Any]]


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, cls=_NumpyJSONEncoder) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), cls=_NumpyJSONEncoder)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_synthetic_market_frame(*, periods: int = 420, symbols: int = 8, seed: int = 191) -> pd.DataFrame:
    if periods < 250:
        raise ValueError("synthetic panel requires at least 250 periods")
    if symbols < 6:
        raise ValueError("synthetic panel requires at least 6 symbols")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=periods)
    rows: list[pd.DataFrame] = []
    common = rng.normal(0.0003, 0.008, size=periods)
    for idx in range(symbols):
        idio = rng.normal(0, 0.012 + idx * 0.0003, size=periods)
        ret = common * (0.7 + idx / max(symbols, 1) * 0.3) + idio
        close = 30.0 + idx * 3.0
        close_path = close * np.exp(np.cumsum(ret))
        overnight = rng.normal(0, 0.003, size=periods)
        open_path = close_path * np.exp(overnight)
        spread = np.abs(rng.normal(0.008, 0.003, size=periods))
        high = np.maximum(open_path, close_path) * (1 + spread)
        low = np.minimum(open_path, close_path) * (1 - spread)
        volume = rng.lognormal(13.5 + idx * 0.02, 0.35, size=periods)
        vwap = (open_path + high + low + close_path) / 4
        amount = volume * vwap
        frame = pd.DataFrame(
            {
                "datetime": dates,
                "asset": f"S{idx:04d}",
                "open": open_path,
                "high": high,
                "low": low,
                "close": close_path,
                "preclose": np.r_[np.nan, close_path[:-1]],
                "volume": volume,
                "amount": amount,
                "ret": np.r_[np.nan, close_path[1:] / close_path[:-1] - 1],
                "vwap": vwap,
            }
        )
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["asset", "datetime"]).reset_index(drop=True)


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "asset", "open", "high", "low", "close", "volume", "vwap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market frame missing required columns: {missing}")
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise")
    out["asset"] = out["asset"].astype(str)
    if out[["datetime", "asset"]].isna().any().any():
        raise ValueError("market frame contains null datetime/asset keys")
    duplicated = out.duplicated(["datetime", "asset"], keep=False)
    if duplicated.any():
        sample = out.loc[duplicated, ["datetime", "asset"]].head(10).to_dict("records")
        raise ValueError(f"market frame contains duplicate keys: {sample}")
    out = out.sort_values(["asset", "datetime"]).reset_index(drop=True)
    return out


def _split_boundaries(frame: pd.DataFrame, config: BatchEvaluationConfig) -> SplitBoundaries:
    dates = pd.Index(sorted(pd.to_datetime(frame["datetime"]).dt.normalize().unique()))
    if len(dates) < 30:
        raise ValueError("evaluation requires at least 30 distinct dates")
    train_idx = max(1, min(len(dates) - 2, int(len(dates) * config.train_fraction)))
    valid_idx = max(train_idx + 1, min(len(dates) - 1, int(len(dates) * (config.train_fraction + config.valid_fraction))))
    return SplitBoundaries(
        first_date=str(pd.Timestamp(dates[0]).date()),
        train_end=str(pd.Timestamp(dates[train_idx - 1]).date()),
        valid_end=str(pd.Timestamp(dates[valid_idx - 1]).date()),
        last_date=str(pd.Timestamp(dates[-1]).date()),
    )


def _split_mask(index: pd.MultiIndex, split: str, bounds: SplitBoundaries) -> np.ndarray:
    dates = pd.to_datetime(index.get_level_values(0)).normalize()
    train_end = pd.Timestamp(bounds.train_end)
    valid_end = pd.Timestamp(bounds.valid_end)
    if split == "train":
        return np.asarray(dates <= train_end)
    if split == "valid":
        return np.asarray((dates > train_end) & (dates <= valid_end))
    if split == "test":
        return np.asarray(dates > valid_end)
    if split == "all":
        return np.ones(len(index), dtype=bool)
    raise ValueError(f"unknown split: {split}")


def _frame_source(frame: pd.DataFrame) -> InMemoryFrameSource:
    return InMemoryFrameSource(frame, time_column="datetime", instrument_column="asset")


def _compile_pack(
    factors: Sequence[FactorDefinition],
    frame: pd.DataFrame,
    *,
    backend: str,
    run_mode: str,
) -> tuple[FactorEngine, list[Any], dict[str, float]]:
    source = _frame_source(frame)
    engine = FactorEngine(build_backend(backend), source, run_mode=run_mode)
    parsed: list[Any] = []
    timings: dict[str, float] = {}
    for item in factors:
        factor = parse_factor(
            item.formula,
            name=item.name,
            freq="1d",
            universe="GTJA185",
            description=item.description,
        )
        started = time.perf_counter()
        engine.compile(
            factor,
            pit_enforce=True,
            pit_forbid_forward_fill=True,
        )
        timings[item.name] = time.perf_counter() - started
        parsed.append(factor)
    return engine, parsed, timings


def _coerce_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.rename(name)
    elif isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        series = value.iloc[:, 0].rename(name)
    elif hasattr(value, "to_pandas"):
        converted = value.to_pandas()
        if not isinstance(converted, pd.Series):
            raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
        series = converted.rename(name)
    else:
        raise TypeError(f"unsupported result type for {name}: {type(value).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError(f"factor {name} must return MultiIndex(datetime, asset)")
    series.index = series.index.set_names(["datetime", "asset"])
    return pd.to_numeric(series, errors="coerce").sort_index()


def _execute_pack(
    engine: FactorEngine,
    parsed_factors: Sequence[Any],
    *,
    batch_size: int,
    market: str,
) -> tuple[dict[str, pd.Series], dict[str, float], dict[str, str]]:
    results: dict[str, pd.Series] = {}
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    for offset in range(0, len(parsed_factors), batch_size):
        batch = list(parsed_factors[offset : offset + batch_size])
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
            elapsed = time.perf_counter() - started
            per_factor = elapsed / max(len(batch), 1)
            for factor in batch:
                results[factor.name] = _coerce_series(output["results"][factor.name], factor.name)
                timings[factor.name] = per_factor
        except Exception as batch_exc:
            for factor in batch:
                one_started = time.perf_counter()
                try:
                    one = engine.run(
                        factor,
                        input_dq_check=False,
                        auto_warmup=False,
                        market=market,
                        pit_enforce=False,
                    )
                    results[factor.name] = _coerce_series(one["result"], factor.name)
                    timings[factor.name] = time.perf_counter() - one_started
                except Exception as one_exc:
                    errors[factor.name] = (
                        f"batch={type(batch_exc).__name__}: {batch_exc}; "
                        f"single={type(one_exc).__name__}: {one_exc}"
                    )
    return results, timings, errors


def _purify_series(series: pd.Series, *, mad_multiplier: float, min_assets: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def transform(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if len(valid) < min_assets:
            return pd.Series(np.nan, index=group.index, dtype=float)
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if math.isfinite(mad) and mad > 0:
            scale = 1.4826 * mad
            clipped = group.clip(median - mad_multiplier * scale, median + mad_multiplier * scale)
        else:
            clipped = group.copy()
        mean = float(clipped.mean(skipna=True))
        std = float(clipped.std(skipna=True, ddof=1))
        if not math.isfinite(std) or std <= 1e-12:
            return pd.Series(np.nan, index=group.index, dtype=float)
        return (clipped - mean) / std

    purified = values.groupby(level="datetime", group_keys=False).transform(transform)
    purified.index = purified.index.set_names(["datetime", "asset"])
    return purified.sort_index().rename(series.name)


def _price_forward_returns(frame: pd.DataFrame, horizons: Sequence[int], price_field: str) -> dict[int, pd.Series]:
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


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _daily_ic(signal: pd.Series, returns: pd.Series, *, min_assets: int) -> pd.DataFrame:
    joined = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1)
    rows: list[dict[str, Any]] = []
    for date, group in joined.groupby(level="datetime", sort=True):
        valid = group.dropna()
        if len(valid) < min_assets:
            continue
        rows.append(
            {
                "datetime": pd.Timestamp(date),
                "n": int(len(valid)),
                "ic": _safe_corr(valid["signal"], valid["return"], "pearson"),
                "rank_ic": _safe_corr(valid["signal"], valid["return"], "spearman"),
            }
        )
    return pd.DataFrame(rows)


def _summary_stats(values: pd.Series) -> dict[str, float | int | None]:
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


def _long_short_series(
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


def _performance_stats(returns: pd.Series, *, annualization: float) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "hit_rate": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    annual_return = mean * annualization
    annual_vol = std * math.sqrt(annualization) if math.isfinite(std) else float("nan")
    sharpe = annual_return / annual_vol if math.isfinite(annual_vol) and annual_vol > 0 else float("nan")
    wealth = (1.0 + clean.clip(lower=-0.999999)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "count": int(len(clean)),
        "mean": mean,
        "annual_return": annual_return,
        "annual_volatility": annual_vol if math.isfinite(annual_vol) else None,
        "sharpe": sharpe if math.isfinite(sharpe) else None,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((clean > 0).mean()),
    }


def _subset_by_split(series: pd.Series, split: str, bounds: SplitBoundaries) -> pd.Series:
    if series.empty:
        return series
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(series.index), np.repeat("series", len(series))],
        names=["datetime", "asset"],
    )
    mask = _split_mask(idx, split, bounds)
    return series.iloc[np.flatnonzero(mask)]


def _route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    test = primary.get("test", {})
    rank_ic = ((test.get("rank_ic") or {}).get("mean"))
    icir = ((test.get("rank_ic") or {}).get("ir"))
    sharpe = ((test.get("long_short") or {}).get("sharpe"))
    positive = ((test.get("rank_ic") or {}).get("positive_ratio"))
    if coverage < 0.35 or rank_ic is None or rank_ic <= 0:
        return "rejected"
    if rank_ic >= 0.03 and (icir or 0) >= 0.45 and (sharpe or 0) >= 0.8 and (positive or 0) >= 0.55:
        return "tier3a_core"
    if rank_ic >= 0.015 and (icir or 0) >= 0.20 and (sharpe or 0) >= 0.25:
        return "tier3b_satellite"
    return "tier2_research"


def _evaluate_one(
    factor: FactorDefinition,
    raw: pd.Series,
    forward_returns: Mapping[int, pd.Series],
    bounds: SplitBoundaries,
    config: BatchEvaluationConfig,
    *,
    compile_seconds: float,
    execute_seconds: float,
) -> tuple[FactorRunRecord, pd.Series]:
    purified = _purify_series(raw, mad_multiplier=config.winsor_mad, min_assets=config.min_assets)
    finite = np.isfinite(pd.to_numeric(purified, errors="coerce").to_numpy(dtype=float))
    finite_values = int(finite.sum())
    coverage = finite_values / max(len(purified), 1)

    primary_forward = forward_returns[min(config.horizons)].reindex(purified.index)
    train_positions = np.flatnonzero(_split_mask(purified.index, "train", bounds))
    train_ic = _daily_ic(
        purified.iloc[train_positions],
        primary_forward.iloc[train_positions],
        min_assets=config.min_assets,
    )
    direction_mean = pd.to_numeric(train_ic.get("rank_ic", pd.Series(dtype=float)), errors="coerce").mean()
    direction = -1 if pd.notna(direction_mean) and float(direction_mean) < 0 else 1
    signal = purified * direction

    metrics: dict[str, Any] = {}
    for horizon, future in forward_returns.items():
        horizon_metrics: dict[str, Any] = {}
        annualization = config.annualization / max(int(horizon), 1)
        for split in ("train", "valid", "test", "all"):
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
        year_rows: dict[str, Any] = {}
        aligned = pd.concat([signal.rename("signal"), future.rename("return")], axis=1).dropna()
        if not aligned.empty:
            for year, group in aligned.groupby(aligned.index.get_level_values("datetime").year):
                daily = _daily_ic(group["signal"], group["return"], min_assets=config.min_assets)
                year_rows[str(int(year))] = _summary_stats(daily.get("rank_ic", pd.Series(dtype=float)))
        horizon_metrics["by_year"] = year_rows
        metrics[str(int(horizon))] = horizon_metrics

    record = FactorRunRecord(
        factor_name=factor.name,
        formula_hash=factor.formula_hash,
        status="success",
        compile_seconds=round(compile_seconds, 6),
        execute_seconds=round(execute_seconds, 6),
        finite_values=finite_values,
        coverage=coverage,
        direction=direction,
        metrics=metrics,
    )
    record.route = _route_factor(metrics, coverage=coverage)
    return record, signal.rename(factor.name)


def _ranking_rows(records: Sequence[FactorRunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    route_score = {"tier3a_core": 3, "tier3b_satellite": 2, "tier2_research": 1, "rejected": 0}
    for record in records:
        primary = record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}
        test = primary.get("test", {})
        rows.append(
            {
                "factor_name": record.factor_name,
                "status": record.status,
                "route": record.route,
                "route_score": route_score.get(record.route, -1),
                "coverage": record.coverage,
                "direction": record.direction,
                "test_rank_ic": ((test.get("rank_ic") or {}).get("mean")),
                "test_rank_ic_ir": ((test.get("rank_ic") or {}).get("ir")),
                "test_rank_ic_positive_ratio": ((test.get("rank_ic") or {}).get("positive_ratio")),
                "test_long_short_sharpe": ((test.get("long_short") or {}).get("sharpe")),
                "test_long_short_annual_return": ((test.get("long_short") or {}).get("annual_return")),
                "test_long_short_max_drawdown": ((test.get("long_short") or {}).get("max_drawdown")),
                "turnover_mean": test.get("turnover_mean"),
                "compile_seconds": record.compile_seconds,
                "execute_seconds": record.execute_seconds,
                "error": record.error,
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("route_score", -1),
            row.get("test_rank_ic_ir") if row.get("test_rank_ic_ir") is not None else -999,
            row.get("test_long_short_sharpe") if row.get("test_long_short_sharpe") is not None else -999,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _resume_record(path: Path, *, formula_hash: str, snapshot_id: str, config_hash: str) -> FactorRunRecord | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if raw.get("formula_hash") != formula_hash:
        return None
    if raw.get("snapshot_id") != snapshot_id:
        return None
    if raw.get("config_hash") != config_hash:
        return None
    record = raw.get("record")
    if not isinstance(record, dict) or record.get("status") != "success":
        return None
    return FactorRunRecord(**record)


def run_gtja185_evaluation(
    config: BatchEvaluationConfig,
    *,
    output_dir: str | Path,
    market_frame: pd.DataFrame | None = None,
    snapshot_id: str | None = None,
) -> BatchRunSummary:
    config.validate()
    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    pack = load_gtja185_pack()
    selected = pack.select(
        config.selected_factors or None,
        limit=config.limit,
    )
    output = Path(output_dir)
    report_dir = output / "factor_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if market_frame is None:
        fields = ["open", "high", "low", "close", "preclose", "volume", "amount", "ret", "vwap"]
        market_frame, snapshot = load_market_frame(
            market=config.market,
            dataset=config.dataset,
            fields=fields,
            start_date=config.start_date,
            end_date=config.end_date,
            instrument_filter=config.instrument_filter or None,
        )
        snapshot_id = str(snapshot.snapshot_id)
    else:
        snapshot_id = str(snapshot_id or f"synthetic:{_canonical_hash({'rows': len(market_frame), 'columns': list(market_frame.columns)})[:16]}")
    market_frame = _normalize_market_frame(market_frame)
    bounds = _split_boundaries(market_frame, config)
    config_payload = asdict(config)
    config_hash = _canonical_hash(config_payload)
    run_id = f"gtja185_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{snapshot_id[:8]}_{config_hash[:8]}"

    engine, parsed, compile_timings = _compile_pack(
        selected,
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
    forward_returns = _price_forward_returns(
        market_frame,
        config.horizons,
        config.forward_price_field,
    )

    records: list[FactorRunRecord] = []
    purified_signals: dict[str, pd.Series] = {}
    skipped = 0
    for factor in selected:
        record_path = report_dir / f"{factor.name}.json"
        if config.resume:
            resumed = _resume_record(
                record_path,
                formula_hash=factor.formula_hash,
                snapshot_id=snapshot_id,
                config_hash=config_hash,
            )
            if resumed is not None:
                records.append(resumed)
                skipped += 1
                continue
        if factor.name in execution_errors or factor.name not in raw_results:
            record = FactorRunRecord(
                factor_name=factor.name,
                formula_hash=factor.formula_hash,
                status="failed",
                compile_seconds=compile_timings.get(factor.name, 0.0),
                execute_seconds=execute_timings.get(factor.name, 0.0),
                error=execution_errors.get(factor.name, "factor result missing"),
            )
            records.append(record)
            _json_dump(
                record_path,
                {
                    "factor_name": factor.name,
                    "formula_hash": factor.formula_hash,
                    "snapshot_id": snapshot_id,
                    "config_hash": config_hash,
                    "record": asdict(record),
                },
            )
            continue
        try:
            record, purified = _evaluate_one(
                factor,
                raw_results[factor.name],
                forward_returns,
                bounds,
                config,
                compile_seconds=compile_timings.get(factor.name, 0.0),
                execute_seconds=execute_timings.get(factor.name, 0.0),
            )
            if config.materialize_staging:
                record.staging = materialize_factor_to_staging(
                    raw_results[factor.name],
                    factor_id=factor.name,
                    factor_version=config.factor_version,
                    snapshot_id=snapshot_id,
                    publish=config.publish,
                )
            records.append(record)
            purified_signals[factor.name] = purified
        except Exception as exc:
            record = FactorRunRecord(
                factor_name=factor.name,
                formula_hash=factor.formula_hash,
                status="failed",
                compile_seconds=compile_timings.get(factor.name, 0.0),
                execute_seconds=execute_timings.get(factor.name, 0.0),
                error=f"{type(exc).__name__}: {exc}",
            )
            records.append(record)
        _json_dump(
            record_path,
            {
                "factor_name": factor.name,
                "formula_hash": factor.formula_hash,
                "snapshot_id": snapshot_id,
                "config_hash": config_hash,
                "record": asdict(records[-1]),
            },
        )

    ranking = _ranking_rows(records)
    pd.DataFrame(ranking).to_csv(output / "ranking.csv", index=False)
    pd.DataFrame(ranking).to_parquet(output / "ranking.parquet", index=False)

    successful = [record for record in records if record.status == "success"]
    failed = [record for record in records if record.status != "success"]
    finished_at = datetime.now(timezone.utc)
    status = "success" if not failed else ("partial" if successful else "failed")
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
        elapsed_seconds=round(time.perf_counter() - started_perf, 3),
        factor_count=len(selected),
        succeeded=len(successful),
        failed=len(failed),
        skipped=skipped,
        output_dir=str(output.resolve()),
        config_hash=config_hash,
        split_boundaries=asdict(bounds),
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
    parser = argparse.ArgumentParser(description="Evaluate the complete GTJA185 factor pack")
    parser.add_argument("--output-dir", default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185")
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--backend", default="pandas")
    parser.add_argument("--run-mode", default="research")
    parser.add_argument("--horizons", default="1,5,21")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", action="append", default=[])
    parser.add_argument("--materialize-staging", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-periods", type=int, default=420)
    parser.add_argument("--synthetic-symbols", type=int, default=8)
    args = parser.parse_args(argv)

    config = BatchEvaluationConfig(
        market=args.market,
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        backend=args.backend,
        run_mode=args.run_mode,
        horizons=_parse_csv_ints(args.horizons),
        batch_size=args.batch_size,
        min_assets=args.min_assets,
        n_quantiles=args.n_quantiles,
        materialize_staging=args.materialize_staging,
        publish=args.publish,
        strict=not args.allow_partial,
        resume=not args.no_resume,
        limit=args.limit,
        selected_factors=tuple(args.factor),
    )
    frame = None
    snapshot_id = None
    if args.synthetic:
        frame = build_synthetic_market_frame(
            periods=args.synthetic_periods,
            symbols=args.synthetic_symbols,
        )
        snapshot_id = "synthetic:gtja185-ci"
    try:
        summary = run_gtja185_evaluation(
            config,
            output_dir=args.output_dir,
            market_frame=frame,
            snapshot_id=snapshot_id,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, cls=_NumpyJSONEncoder))
    return 0 if summary.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
