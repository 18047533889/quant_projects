from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factor_layer.factor_evaluation.config import FactorEvaluationConfig
from factor_layer.factor_evaluation.io import load_factor_data, load_market_data, load_universe_data


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "run"


def _config_hash(config: FactorEvaluationConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _build_run_id(config: FactorEvaluationConfig) -> str:
    if config.meta.run_name:
        return _sanitize_name(config.meta.run_name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_sanitize_name(config.meta.factor_id)}_{stamp}"


def _winsorize_cross_section(series: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series.astype(float)
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def _zscore_cross_section(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    std = float(valid.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - float(valid.mean())) / std


def _prepare_factor_values(frame: pd.DataFrame, config: FactorEvaluationConfig) -> pd.DataFrame:
    factor = frame.copy()
    factor["factor_raw"] = pd.to_numeric(factor["factor_raw"], errors="coerce") * float(config.run.direction)
    factor["factor_eval"] = factor.groupby("timestamp", sort=True)["factor_raw"].transform(
        lambda s: _winsorize_cross_section(
            s,
            lower=config.run.winsorize_lower,
            upper=config.run.winsorize_upper,
        )
    )
    if config.run.standardize:
        factor["factor_eval"] = factor.groupby("timestamp", sort=True)["factor_eval"].transform(_zscore_cross_section)
    return factor.reset_index(drop=True)


def _build_market_returns(market: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    frame = market.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False)["price"]
    frame["one_step_return"] = grouped.shift(-1) / frame["price"] - 1.0
    for horizon in horizons:
        frame[f"ret_h{horizon}"] = grouped.shift(-(horizon + 1)) / grouped.shift(-1) - 1.0
    return frame


def _assign_quantiles(frame: pd.DataFrame, n_quantiles: int, min_assets: int) -> pd.Series:
    def _per_date(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        threshold = max(min_assets, n_quantiles)
        if len(valid) < threshold:
            return pd.Series(np.nan, index=series.index, dtype=float)
        pct_rank = valid.rank(method="first", pct=True)
        quantile = np.ceil(pct_rank.to_numpy() * n_quantiles)
        quantile = np.clip(quantile, 1, n_quantiles)
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[valid.index] = quantile
        return out

    return frame.groupby("timestamp", sort=True)["factor_eval"].transform(_per_date)


def _summary_from_series(series: pd.Series) -> dict[str, float | int]:
    valid = series.dropna()
    if valid.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "win_rate": np.nan,
            "n_dates": 0,
        }
    std = float(valid.std(ddof=1))
    ir = np.nan if math.isclose(std, 0.0) else float(valid.mean() / std)
    return {
        "mean": float(valid.mean()),
        "std": std,
        "ir": ir,
        "win_rate": float((valid > 0).mean()),
        "n_dates": int(valid.shape[0]),
    }


def _compute_daily_ic(frame: pd.DataFrame, ret_col: str, min_assets: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        valid = group[["factor_eval", ret_col]].dropna()
        n_assets = int(valid.shape[0])
        if n_assets < min_assets:
            continue
        factor_series = valid["factor_eval"]
        return_series = valid[ret_col]
        ic = factor_series.corr(return_series, method="pearson")
        rank_ic = factor_series.rank(method="average").corr(
            return_series.rank(method="average"),
            method="pearson",
        )
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "ic": ic,
                "rank_ic": rank_ic,
                "universe_size": n_assets,
            }
        )
    return pd.DataFrame(rows)


def _compute_period_group_returns(frame: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    valid = frame.dropna(subset=["quantile", ret_col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["timestamp", "quantile", "period_return"])
    grouped = (
        valid.groupby(["timestamp", "quantile"], sort=True)[ret_col]
        .mean()
        .rename("period_return")
        .reset_index()
    )
    grouped["quantile"] = grouped["quantile"].astype(int)
    return grouped


def _build_quantile_backtest(
    frame: pd.DataFrame,
    *,
    horizon: int,
    n_quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[["timestamp", "symbol", "quantile", "one_step_return"]].copy()
    base = base.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    quantile_returns: list[pd.DataFrame] = []

    for quantile in range(1, n_quantiles + 1):
        selected = base["quantile"] == quantile
        counts = selected.groupby(base["timestamp"]).sum()
        signal_weight = pd.Series(0.0, index=base.index, dtype=float)
        valid_selected = selected & base["timestamp"].map(counts).gt(0)
        signal_weight.loc[valid_selected] = 1.0 / base.loc[valid_selected, "timestamp"].map(counts)

        active_weight = pd.Series(0.0, index=base.index, dtype=float)
        grouped_signal = signal_weight.groupby(base["symbol"], sort=False)
        for lag in range(1, horizon + 1):
            active_weight = active_weight.add(grouped_signal.shift(lag).fillna(0.0), fill_value=0.0)
        active_weight = active_weight / float(horizon)

        valid = active_weight.gt(0) & base["one_step_return"].notna()
        if not valid.any():
            continue
        pnl = pd.DataFrame(
            {
                "timestamp": base.loc[valid, "timestamp"],
                "weighted_return": active_weight.loc[valid] * base.loc[valid, "one_step_return"],
                "weight": active_weight.loc[valid],
            }
        )
        aggregated = pnl.groupby("timestamp", sort=True).sum(numeric_only=True)
        daily_return = aggregated["weighted_return"] / aggregated["weight"]
        out = daily_return.rename("daily_return").reset_index()
        out["quantile"] = quantile
        out["nav"] = (1.0 + out["daily_return"].fillna(0.0)).cumprod()
        quantile_returns.append(out)

    if not quantile_returns:
        empty_q = pd.DataFrame(columns=["timestamp", "quantile", "daily_return", "nav"])
        empty_ls = pd.DataFrame(columns=["timestamp", "top_daily_return", "bottom_daily_return", "long_short_return", "nav"])
        return empty_q, empty_ls

    quantile_backtest = pd.concat(quantile_returns, ignore_index=True)
    pivot = quantile_backtest.pivot(index="timestamp", columns="quantile", values="daily_return").sort_index()
    top = pivot.get(n_quantiles)
    bottom = pivot.get(1)
    if top is None:
        top = pd.Series(dtype=float)
    if bottom is None:
        bottom = pd.Series(dtype=float)
    long_short = pd.concat([top.rename("top_daily_return"), bottom.rename("bottom_daily_return")], axis=1)
    long_short["long_short_return"] = long_short["top_daily_return"] - long_short["bottom_daily_return"]
    long_short["nav"] = (1.0 + long_short["long_short_return"].fillna(0.0)).cumprod()
    return quantile_backtest, long_short.reset_index()


def _summarize_long_short(series: pd.Series, annualization_factor: int) -> dict[str, float]:
    valid = series.dropna().sort_index()
    if valid.empty:
        return {
            "long_short_total_return": np.nan,
            "long_short_ann_return": np.nan,
            "long_short_ann_vol": np.nan,
            "long_short_sharpe": np.nan,
            "long_short_max_drawdown": np.nan,
        }
    nav = (1.0 + valid.fillna(0.0)).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    ann_vol = float(valid.std(ddof=1) * math.sqrt(annualization_factor)) if valid.shape[0] > 1 else np.nan
    sharpe = np.nan
    std = float(valid.std(ddof=1)) if valid.shape[0] > 1 else np.nan
    if pd.notna(std) and not math.isclose(std, 0.0):
        sharpe = float(valid.mean() / std * math.sqrt(annualization_factor))
    ann_return = np.nan
    if valid.shape[0] > 0:
        ann_return = float((1.0 + total_return) ** (annualization_factor / valid.shape[0]) - 1.0)
    drawdown = nav / nav.cummax() - 1.0
    return {
        "long_short_total_return": total_return,
        "long_short_ann_return": ann_return,
        "long_short_ann_vol": ann_vol,
        "long_short_sharpe": sharpe,
        "long_short_max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
    }


def evaluate_factor(config: FactorEvaluationConfig) -> dict[str, object]:
    factor = load_factor_data(
        config.source.factor_lake_root,
        config.meta.factor_id,
        start=config.run.start,
        end=config.run.end,
    )
    market = load_market_data(config.source, start=config.run.start, end=config.run.end)
    universe = load_universe_data(config.source, start=config.run.start, end=config.run.end)

    factor = _prepare_factor_values(factor, config)
    market = _build_market_returns(market, config.run.horizons)
    base = factor.merge(market, on=["timestamp", "symbol"], how="left")
    if universe is not None:
        base = base.merge(universe.assign(in_universe=True), on=["timestamp", "symbol"], how="inner")
    base = base.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    base["quantile"] = _assign_quantiles(base, config.run.n_quantiles, config.run.min_assets_per_date)

    daily_ic_frames: list[pd.DataFrame] = []
    daily_rank_ic_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    quantile_summary_rows: list[dict[str, object]] = []
    quantile_backtest_frames: list[pd.DataFrame] = []
    long_short_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for horizon in config.run.horizons:
        ret_col = f"ret_h{horizon}"
        daily = _compute_daily_ic(base, ret_col, config.run.min_assets_per_date)
        if daily.empty:
            ic_summary = _summary_from_series(pd.Series(dtype=float))
            rank_ic_summary = _summary_from_series(pd.Series(dtype=float))
            avg_universe_size = np.nan
        else:
            ic_summary = _summary_from_series(daily["ic"])
            rank_ic_summary = _summary_from_series(daily["rank_ic"])
            avg_universe_size = float(daily["universe_size"].mean())

        period = _compute_period_group_returns(base, ret_col)
        if not period.empty:
            period["horizon"] = horizon
            period_frames.append(period)
            period_mean = (
                period.groupby("quantile", sort=True)["period_return"]
                .mean()
                .rename("mean_period_return")
                .reset_index()
            )
            period_mean["horizon"] = horizon
            quantile_summary_rows.extend(period_mean.to_dict(orient="records"))
            pivot = period.pivot(index="timestamp", columns="quantile", values="period_return")
            top_series = pivot.get(config.run.n_quantiles)
            bottom_series = pivot.get(1)
            if top_series is not None and bottom_series is not None:
                top_minus_bottom_mean = float((top_series - bottom_series).mean())
            else:
                top_minus_bottom_mean = np.nan
            monotonicity = float(
                period_mean["quantile"].corr(period_mean["mean_period_return"], method="spearman")
            )
        else:
            top_minus_bottom_mean = np.nan
            monotonicity = np.nan

        quantile_backtest, long_short = _build_quantile_backtest(
            base,
            horizon=horizon,
            n_quantiles=config.run.n_quantiles,
        )
        if not quantile_backtest.empty:
            quantile_backtest["horizon"] = horizon
            quantile_backtest_frames.append(quantile_backtest)
        if not long_short.empty:
            long_short["horizon"] = horizon
            long_short_frames.append(long_short)
            long_short_stats = _summarize_long_short(long_short.set_index("timestamp")["long_short_return"], config.run.annualization_factor)
        else:
            long_short_stats = _summarize_long_short(pd.Series(dtype=float), config.run.annualization_factor)

        if not daily.empty:
            daily_ic_frames.append(daily.loc[:, ["timestamp", "ic", "universe_size"]].assign(horizon=horizon))
            daily_rank_ic_frames.append(daily.loc[:, ["timestamp", "rank_ic", "universe_size"]].assign(horizon=horizon))

        summary_rows.append(
            {
                "factor_id": config.meta.factor_id,
                "horizon": horizon,
                "ic_mean": ic_summary["mean"],
                "ic_std": ic_summary["std"],
                "ic_ir": ic_summary["ir"],
                "ic_win_rate": ic_summary["win_rate"],
                "rank_ic_mean": rank_ic_summary["mean"],
                "rank_ic_std": rank_ic_summary["std"],
                "rank_ic_ir": rank_ic_summary["ir"],
                "rank_ic_win_rate": rank_ic_summary["win_rate"],
                "top_minus_bottom_mean": top_minus_bottom_mean,
                "monotonicity_score": monotonicity,
                "n_dates": ic_summary["n_dates"],
                "avg_universe_size": avg_universe_size,
                **long_short_stats,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    run_id = _build_run_id(config)
    output_root = Path(config.output.root or (Path(config.source.factor_lake_root) / "evaluations"))
    output_dir = output_root / config.meta.factor_id / run_id
    created_at = datetime.now(timezone.utc).isoformat()
    config_hash = _config_hash(config)
    primary_horizon = config.meta.primary_horizon or int(config.run.horizons[0])
    meta = {
        "factor_id": config.meta.factor_id,
        "run_id": run_id,
        "created_at": created_at,
        "sample_start": config.run.start,
        "sample_end": config.run.end,
        "universe_id": Path(config.source.universe_path).stem if config.source.universe_path else "all",
        "primary_horizon": primary_horizon,
        "config_hash": config_hash,
    }
    return {
        "config": config,
        "meta": meta,
        "base_frame": base,
        "summary_df": summary_df,
        "daily_ic_df": pd.concat(daily_ic_frames, ignore_index=True) if daily_ic_frames else pd.DataFrame(columns=["timestamp", "ic", "universe_size", "horizon"]),
        "daily_rank_ic_df": pd.concat(daily_rank_ic_frames, ignore_index=True) if daily_rank_ic_frames else pd.DataFrame(columns=["timestamp", "rank_ic", "universe_size", "horizon"]),
        "quantile_period_df": pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame(columns=["timestamp", "quantile", "period_return", "horizon"]),
        "quantile_summary_df": pd.DataFrame(quantile_summary_rows),
        "quantile_backtest_df": pd.concat(quantile_backtest_frames, ignore_index=True) if quantile_backtest_frames else pd.DataFrame(columns=["timestamp", "quantile", "daily_return", "nav", "horizon"]),
        "long_short_df": pd.concat(long_short_frames, ignore_index=True) if long_short_frames else pd.DataFrame(columns=["timestamp", "top_daily_return", "bottom_daily_return", "long_short_return", "nav", "horizon"]),
        "output_dir": output_dir,
    }


def save_results(result: dict[str, object], config: FactorEvaluationConfig) -> Path:
    output_dir = Path(result["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df: pd.DataFrame = result["summary_df"]  # type: ignore[assignment]
    daily_ic_df: pd.DataFrame = result["daily_ic_df"]  # type: ignore[assignment]
    daily_rank_ic_df: pd.DataFrame = result["daily_rank_ic_df"]  # type: ignore[assignment]
    quantile_period_df: pd.DataFrame = result["quantile_period_df"]  # type: ignore[assignment]
    quantile_summary_df: pd.DataFrame = result["quantile_summary_df"]  # type: ignore[assignment]
    quantile_backtest_df: pd.DataFrame = result["quantile_backtest_df"]  # type: ignore[assignment]
    long_short_df: pd.DataFrame = result["long_short_df"]  # type: ignore[assignment]
    meta: dict[str, object] = result["meta"]  # type: ignore[assignment]

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    daily_ic_df.to_parquet(output_dir / "daily_ic.parquet", index=False)
    daily_rank_ic_df.to_parquet(output_dir / "daily_rank_ic.parquet", index=False)
    quantile_period_df.to_parquet(output_dir / "quantile_period_returns.parquet", index=False)
    quantile_summary_df.to_parquet(output_dir / "quantile_summary.parquet", index=False)
    quantile_backtest_df.to_parquet(output_dir / "quantile_backtest.parquet", index=False)
    long_short_df.to_parquet(output_dir / "long_short_returns.parquet", index=False)

    summary_payload = {
        **meta,
        "summary": summary_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        **meta,
        "artifacts": {
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "daily_ic": "daily_ic.parquet",
            "daily_rank_ic": "daily_rank_ic.parquet",
            "quantile_period_returns": "quantile_period_returns.parquet",
            "quantile_summary": "quantile_summary.parquet",
            "quantile_backtest": "quantile_backtest.parquet",
            "long_short_returns": "long_short_returns.parquet",
            "config_snapshot": "config_snapshot.yaml",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir