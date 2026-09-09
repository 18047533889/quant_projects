#!/usr/bin/env python3
"""Cross-sectional acceptance test for every registered optimizer.

The test uses one common, real input set:
- Ridge_h5 predictions as five-day expected returns;
- CSI 500 constituent weights as the benchmark;
- v1_sbi_fullA as the precomputed Barra risk package;
- adjusted close/return and market cap from lqtp_data.

The purpose is optimizer validation, not a performance backtest. Five
consecutive dates keep the universe and inputs identical while still testing
position chaining, turnover, hard constraints and point-in-time risk.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from riskfolio_qs.adapters import BarraPrecomputedAdapter
from riskfolio_qs.runners.pipeline import OptimizationPipeline
from riskfolio_qs.smoothers.signal_smoother import SignalSmoother


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
PREDICTION_PATH = WORKSPACE_ROOT / "outputs" / "Ridge_h5_pred.parquet"
RISK_ROOT = WORKSPACE_ROOT / "v1_sbi_fullA"
DATA_ROOT = WORKSPACE_ROOT / "lqtp_data"
OUTPUT_ROOT = REPO_ROOT / "validation" / "ridge_h5_all_optimizers"
INDEX_SYMBOL = "000905.SH"
TEST_END = pd.Timestamp("2025-12-24")
TEST_DATE_COUNT = 5
RISK_LOOKBACK = 252

OPTIMIZERS = [
    "minvar_enhance_index",
    "meanvar_enhance_index",
    "minvar_enhance_barra_precomputed",
    "meanvar_enhance_barra_precomputed",
    "minvar_enhance_hist",
    "meanvar_enhance_hist",
    "meanvar_absolute_return",
    "topn_long_only_equal_weight",
    "topn_long_short_equal_weight",
]


def _read_daily(folder: str, date: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    path = DATA_ROOT / folder / f"{date:%Y-%m-%d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def _load_test_dates() -> pd.DatetimeIndex:
    dates = pd.read_parquet(PREDICTION_PATH, columns=["TradeDate"])[
        "TradeDate"
    ].drop_duplicates()
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    selected = dates[dates <= TEST_END][-TEST_DATE_COUNT:]
    if len(selected) != TEST_DATE_COUNT:
        raise ValueError("Insufficient Ridge dates for the requested test window")
    return selected


def _load_signal_and_universe(
    test_dates: pd.DatetimeIndex,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    raw = pd.read_parquet(
        PREDICTION_PATH,
        filters=[
            ("TradeDate", ">=", test_dates.min()),
            ("TradeDate", "<=", test_dates.max()),
        ],
        columns=["TradeDate", "Symbol", "pred", "y_true"],
    )
    raw["TradeDate"] = pd.to_datetime(raw["TradeDate"]).dt.normalize()
    if raw.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError("Ridge data contains duplicate date-symbol predictions")

    pred_by_date = {
        date: set(raw.loc[raw["TradeDate"].eq(date), "Symbol"])
        for date in test_dates
    }
    eligible_sets: list[set[str]] = []
    constituents: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in test_dates:
        component = _read_daily(
            "IndexConstituent",
            date,
            ["IndexSymbol", "Symbol", "Weight"],
        )
        component = component.loc[component["IndexSymbol"].eq(INDEX_SYMBOL)].copy()
        if len(component) != 500:
            raise ValueError(f"Expected 500 CSI 500 constituents on {date}")
        constituents[date] = component

        bars = _read_daily(
            "StockDailyBar",
            date,
            ["Symbol", "IsSuspend", "Volume", "Amount"],
        )
        tradable = set(
            bars.loc[
                (~bars["IsSuspend"].astype(bool))
                & bars["Volume"].gt(0)
                & bars["Amount"].gt(0),
                "Symbol",
            ]
        )
        eligible_sets.append(
            set(component["Symbol"]) & pred_by_date[date] & tradable
        )

    # A fixed, always-tradable universe isolates optimizer behavior from
    # reconstitution and suspended-position edge cases.
    assets = sorted(set.intersection(*eligible_sets))
    if len(assets) < 100:
        raise ValueError(f"Unexpectedly small stable universe: {len(assets)}")

    alpha = (
        raw.pivot(index="TradeDate", columns="Symbol", values="pred")
        .reindex(index=test_dates, columns=assets)
        .astype(float)
    )
    realized = (
        raw.pivot(index="TradeDate", columns="Symbol", values="y_true")
        .reindex(index=test_dates, columns=assets)
        .astype(float)
    )
    if alpha.isna().any().any() or realized.isna().any().any():
        raise ValueError("Stable universe unexpectedly has missing Ridge values")

    benchmark = pd.DataFrame(index=test_dates, columns=assets, dtype=float)
    retained_coverage: dict[str, float] = {}
    for date in test_dates:
        component = constituents[date].set_index("Symbol")["Weight"].astype(float)
        selected = component.reindex(assets).fillna(0.0)
        retained_coverage[f"{date:%Y-%m-%d}"] = float(
            selected.sum() / component.sum()
        )
        benchmark.loc[date] = selected / selected.sum()

    previous = benchmark.iloc[[0]].copy()
    previous.index = pd.DatetimeIndex([test_dates[0] - pd.Timedelta(days=1)])
    tradable = pd.DataFrame(True, index=test_dates, columns=assets)
    return alpha, realized, benchmark, previous, tradable, retained_coverage


def _load_history(
    test_dates: pd.DatetimeIndex,
    assets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    factor_returns_long = pd.read_parquet(RISK_ROOT / "factor_returns.parquet")
    factor_returns_long["date"] = pd.to_datetime(
        factor_returns_long["date"]
    ).dt.normalize()
    risk_dates = pd.DatetimeIndex(
        factor_returns_long.loc[
            factor_returns_long["date"].le(test_dates.max()), "date"
        ].unique()
    ).sort_values()[-RISK_LOOKBACK:]

    adjusted_close_rows: dict[pd.Timestamp, pd.Series] = {}
    return_rows: dict[pd.Timestamp, pd.Series] = {}
    for date in risk_dates:
        bars = _read_daily(
            "StockDailyBar",
            date,
            ["Symbol", "Close", "Factor", "Return"],
        ).set_index("Symbol")
        bars = bars.reindex(assets)
        adjusted_close_rows[date] = (
            bars["Close"].astype(float) * bars["Factor"].astype(float)
        )
        return_rows[date] = bars["Return"].astype(float) / 10000.0

    market = pd.DataFrame.from_dict(
        adjusted_close_rows, orient="index"
    ).reindex(index=risk_dates, columns=assets)
    stock_returns = pd.DataFrame.from_dict(
        return_rows, orient="index"
    ).reindex(index=risk_dates, columns=assets)
    market.index.name = "date"
    stock_returns.index.name = "date"
    return market, stock_returns, risk_dates


def _derive_raw_barra_inputs(
    bundle,
    stock_returns: pd.DataFrame,
    risk_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    assets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_returns_long = pd.read_parquet(
        RISK_ROOT / "factor_returns.parquet",
        filters=[
            ("date", ">=", f"{risk_dates.min():%Y-%m-%d}"),
            ("date", "<=", f"{risk_dates.max():%Y-%m-%d}"),
        ],
    )
    factor_returns_long["date"] = pd.to_datetime(
        factor_returns_long["date"]
    ).dt.normalize()
    factors = list(bundle.F.columns)
    factor_returns = factor_returns_long.pivot(
        index="date", columns="factor_id", values="factor_return"
    ).reindex(index=risk_dates, columns=factors)

    exposure_long = pd.read_parquet(
        RISK_ROOT / "exposure.parquet",
        filters=[
            ("date", ">=", f"{risk_dates.min():%Y-%m-%d}"),
            ("date", "<=", f"{risk_dates.max():%Y-%m-%d}"),
            ("asset", "in", assets),
        ],
    )
    exposure_long["date"] = pd.to_datetime(
        exposure_long["date"]
    ).dt.normalize()
    exposure_history = exposure_long.pivot(
        index=["date", "asset"],
        columns="factor_id",
        values="exposure",
    ).reindex(columns=factors)
    industry_factors = [factor for factor in factors if factor.startswith("industry_")]
    exposure_history[industry_factors] = exposure_history[
        industry_factors
    ].fillna(0.0)

    residual = pd.DataFrame(np.nan, index=risk_dates, columns=assets)
    for date in risk_dates:
        if date not in factor_returns.index:
            continue
        try:
            exposure = exposure_history.xs(date, level=0).reindex(
                index=assets, columns=factors
            )
        except KeyError:
            continue
        valid = exposure.notna().all(axis=1) & stock_returns.loc[date].notna()
        if not valid.any():
            continue
        systematic = (
            exposure.loc[valid].to_numpy(dtype=float)
            @ factor_returns.loc[date].to_numpy(dtype=float)
        )
        residual.loc[date, valid.index[valid]] = (
            stock_returns.loc[date, valid.index[valid]].to_numpy(dtype=float)
            - systematic
        )

    market_cap = pd.DataFrame(index=test_dates, columns=assets, dtype=float)
    for date in test_dates:
        valuation = _read_daily(
            "StockValuationDaily",
            date,
            ["Symbol", "MarketCap"],
        ).set_index("Symbol")
        market_cap.loc[date] = valuation["MarketCap"].reindex(assets)
    if market_cap.isna().any().any() or (market_cap <= 0).any().any():
        raise ValueError("Market-cap input is incomplete or non-positive")
    if residual.notna().sum().min() < 20:
        raise ValueError("Derived specific-return history has insufficient coverage")
    return factor_returns, residual, market_cap


def _precomputed_risk_metrics(
    bundle,
    date: pd.Timestamp,
    weights: np.ndarray,
    benchmark: np.ndarray,
) -> dict[str, float]:
    exposure = bundle.F.xs(date, level=0).reindex(
        index=bundle.alpha.columns
    ).to_numpy(dtype=float)
    factor_cov_frame = bundle.factor_cov.xs(date, level=0)
    factors = list(factor_cov_frame.columns)
    factor_cov = factor_cov_frame.reindex(
        index=factors, columns=factors
    ).to_numpy(dtype=float)
    specific = bundle.specific_var.loc[date].reindex(
        bundle.alpha.columns
    ).to_numpy(dtype=float)

    def variance(vector: np.ndarray) -> float:
        factor_exposure = exposure.T @ vector
        return float(
            factor_exposure @ factor_cov @ factor_exposure
            + np.sum(specific * np.square(vector))
        )

    active = weights - benchmark
    portfolio_var = max(0.0, variance(weights))
    active_var = max(0.0, variance(active))
    return {
        "v1_barra_portfolio_vol_annual": float(
            np.sqrt(portfolio_var * bundle.metadata.annualization_factor)
        ),
        "v1_barra_te_annual": float(
            np.sqrt(active_var * bundle.metadata.annualization_factor)
        ),
    }


def _independent_metrics(
    optimizer_name: str,
    target: pd.DataFrame,
    bundle,
    realized: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    previous = bundle.prev_positions.iloc[0].reindex(target.columns).to_numpy(
        dtype=float
    )
    for date in target.index:
        weights = target.loc[date].to_numpy(dtype=float)
        benchmark_values = benchmark.loc[date].to_numpy(dtype=float)
        alpha = bundle.alpha.loc[date].to_numpy(dtype=float)
        future_return = realized.loc[date].to_numpy(dtype=float)
        active = weights - benchmark_values
        industry_labels = bundle.G.loc[date].reindex(
            target.columns
        ).to_numpy(dtype=float)
        industry_exposures = [
            abs(float(active[industry_labels == industry].sum()))
            for industry in np.unique(
                industry_labels[
                    np.isfinite(industry_labels) & (industry_labels >= 0)
                ]
            )
        ]
        style_row = bundle.style.loc[date]
        style_exposures = [
            abs(
                float(
                    style_row.xs(factor, level=0)
                    .reindex(target.columns)
                    .to_numpy(dtype=float)
                    @ active
                )
            )
            for factor in style_row.index.get_level_values(0).unique()
        ]
        gross = float(np.abs(weights).sum())
        normalized_abs = np.abs(weights) / gross if gross > 0 else np.zeros_like(weights)
        hhi = float(np.square(normalized_abs).sum())
        risk = _precomputed_risk_metrics(
            bundle, date, weights, benchmark_values
        )
        alpha_corr = pd.Series(weights).corr(pd.Series(alpha), method="spearman")
        rows.append(
            {
                "date": date,
                "optimizer": optimizer_name,
                "net_exposure": float(weights.sum()),
                "gross_exposure": gross,
                "turnover": 0.5 * float(np.abs(weights - previous).sum()),
                "active_share": 0.5 * float(np.abs(active).sum()),
                "max_weight": float(weights.max()),
                "min_weight": float(weights.min()),
                "max_abs_active_weight": float(np.abs(active).max()),
                "long_count": int((weights > 1e-6).sum()),
                "short_count": int((weights < -1e-6).sum()),
                "effective_n_gross": float(1.0 / hhi) if hhi > 0 else 0.0,
                "alpha_weight_rank_corr": float(alpha_corr),
                "predicted_portfolio_return_5d": float(alpha @ weights),
                "predicted_active_return_5d": float(alpha @ active),
                "realized_portfolio_return_5d": float(future_return @ weights),
                "realized_active_return_5d": float(future_return @ active),
                "v1_max_abs_industry_active_exposure": max(
                    industry_exposures, default=0.0
                ),
                "v1_max_abs_style_active_exposure": max(
                    style_exposures, default=0.0
                ),
                **risk,
            }
        )
        previous = weights
    return pd.DataFrame(rows).set_index("date")


def _health_check(
    optimizer_name: str,
    output,
    independent: pd.DataFrame,
) -> dict[str, object]:
    weights = output.target_positions
    finite = bool(np.isfinite(weights.to_numpy(dtype=float)).all())
    fallback = bool(output.metadata.iloc[0]["fallback_used"])
    if optimizer_name == "topn_long_short_equal_weight":
        exposure_ok = bool(
            np.allclose(weights.sum(axis=1), 0.0, atol=1e-10)
            and np.allclose(weights.abs().sum(axis=1), 1.0, atol=1e-10)
        )
        position_ok = bool(weights.max().max() <= 0.05 + 1e-10)
    else:
        exposure_ok = bool(np.allclose(weights.sum(axis=1), 1.0, atol=1e-5))
        configured_cap = 0.05 if optimizer_name in {
            "meanvar_absolute_return",
            "topn_long_only_equal_weight",
        } else 0.03
        position_ok = bool(
            weights.min().min() >= -1e-5
            and weights.max().max() <= configured_cap + 1e-5
        )
    if "max_constraint_violation" in output.summary:
        max_violation = float(
            output.summary["max_constraint_violation"].fillna(0.0).max()
        )
    else:
        max_violation = 0.0
    status_values = set(output.summary["solve_status"].astype(str))
    expected_status = (
        {"rule"}
        if optimizer_name.startswith("topn_")
        else {"optimal", "optimal_inaccurate"}
    )
    status_ok = status_values.issubset(expected_status)
    passed = (
        finite
        and not fallback
        and exposure_ok
        and position_ok
        and max_violation <= 1e-5
        and status_ok
    )
    return {
        "optimizer": optimizer_name,
        "passed": passed,
        "finite_weights": finite,
        "fallback_used": fallback,
        "exposure_ok": exposure_ok,
        "position_bounds_ok": position_ok,
        "status_ok": status_ok,
        "statuses": ",".join(sorted(status_values)),
        "max_constraint_violation": max_violation,
        "external_v1_te_max": float(independent["v1_barra_te_annual"].max()),
    }


def _aggregate(
    optimizer_name: str,
    output,
    independent: pd.DataFrame,
    wall_time: float,
) -> dict[str, object]:
    summary = output.summary
    row: dict[str, object] = {
        "optimizer": optimizer_name,
        "backend": output.metadata.iloc[0]["extras"]["backend"],
        "risk_mode": output.metadata.iloc[0]["risk_mode"],
        "objective_id": output.metadata.iloc[0]["objective_id"],
        "wall_time_seconds": wall_time,
        "solver_time_seconds": float(
            output.metadata.iloc[0]["solve_time_ms"]
        )
        / 1000.0,
        "mean_turnover": float(independent["turnover"].mean()),
        "max_turnover": float(independent["turnover"].max()),
        "mean_active_share": float(independent["active_share"].mean()),
        "mean_effective_n": float(independent["effective_n_gross"].mean()),
        "mean_long_count": float(independent["long_count"].mean()),
        "mean_short_count": float(independent["short_count"].mean()),
        "max_weight": float(independent["max_weight"].max()),
        "min_weight": float(independent["min_weight"].min()),
        "max_abs_active_weight": float(
            independent["max_abs_active_weight"].max()
        ),
        "mean_alpha_weight_rank_corr": float(
            independent["alpha_weight_rank_corr"].mean()
        ),
        "mean_predicted_active_return_5d": float(
            independent["predicted_active_return_5d"].mean()
        ),
        "mean_realized_active_return_5d": float(
            independent["realized_active_return_5d"].mean()
        ),
        "mean_v1_barra_te_annual": float(
            independent["v1_barra_te_annual"].mean()
        ),
        "max_v1_barra_te_annual": float(
            independent["v1_barra_te_annual"].max()
        ),
        "mean_v1_barra_portfolio_vol_annual": float(
            independent["v1_barra_portfolio_vol_annual"].mean()
        ),
        "mean_v1_max_abs_industry_active_exposure": float(
            independent["v1_max_abs_industry_active_exposure"].mean()
        ),
        "max_v1_max_abs_industry_active_exposure": float(
            independent["v1_max_abs_industry_active_exposure"].max()
        ),
        "mean_v1_max_abs_style_active_exposure": float(
            independent["v1_max_abs_style_active_exposure"].mean()
        ),
        "max_v1_max_abs_style_active_exposure": float(
            independent["v1_max_abs_style_active_exposure"].max()
        ),
    }
    for column in [
        "predicted_tracking_error_annual",
        "tracking_error_cap_violation",
        "max_constraint_violation",
        "max_abs_industry_active_exposure",
        "max_abs_style_active_exposure",
    ]:
        row[f"reported_mean_{column}"] = (
            float(summary[column].mean()) if column in summary else np.nan
        )
        row[f"reported_max_{column}"] = (
            float(summary[column].max()) if column in summary else np.nan
        )
    return row


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_dates = _load_test_dates()
    (
        alpha,
        realized,
        benchmark,
        previous,
        tradable,
        retained_coverage,
    ) = _load_signal_and_universe(test_dates)
    assets = list(alpha.columns)
    print(
        f"dates={[f'{date:%Y-%m-%d}' for date in test_dates]}, "
        f"assets={len(assets)}"
    )

    adapter = BarraPrecomputedAdapter(
        risk_root=RISK_ROOT,
        alpha_df=alpha,
        benchmark_df=benchmark,
        prev_positions_df=previous,
        tradable_df=tradable,
        alpha_input_type="expected_return",
        alpha_horizon_days=5,
        load_factor_returns=False,
    )
    bundle = adapter.build_bundle()
    market, stock_returns, risk_dates = _load_history(test_dates, assets)
    factor_returns, residual, market_cap = _derive_raw_barra_inputs(
        bundle, stock_returns, risk_dates, test_dates, assets
    )
    bundle.market = market
    bundle.F_ret = factor_returns
    bundle.F_spec = residual
    bundle.F_mcap = market_cap

    pipeline = OptimizationPipeline(
        smoother=SignalSmoother(mode="never")
    )
    aggregate_rows: list[dict[str, object]] = []
    health_rows: list[dict[str, object]] = []
    independent_parts: list[pd.DataFrame] = []
    weight_panels: dict[str, pd.DataFrame] = {}

    for optimizer_name in OPTIMIZERS:
        print(f"running {optimizer_name}...")
        started = time.perf_counter()
        output = pipeline.run(
            bundle,
            optimizer_name=optimizer_name,
            data_version_hash="ridge_h5_v1_barra_acceptance_2025_12",
        )
        wall_time = time.perf_counter() - started
        independent = _independent_metrics(
            optimizer_name,
            output.target_positions,
            bundle,
            realized,
            benchmark,
        )
        health = _health_check(optimizer_name, output, independent)
        if not health["passed"]:
            raise RuntimeError(f"Optimizer health check failed: {health}")
        aggregate_rows.append(
            _aggregate(optimizer_name, output, independent, wall_time)
        )
        health_rows.append(health)
        independent_parts.append(independent.reset_index())
        weight_panels[optimizer_name] = output.target_positions

        optimizer_dir = OUTPUT_ROOT / optimizer_name
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        output.target_positions.to_parquet(
            optimizer_dir / "target_positions.parquet"
        )
        output.trades.to_parquet(optimizer_dir / "trades.parquet")
        output.summary.to_parquet(optimizer_dir / "summary.parquet")
        (optimizer_dir / "metadata.json").write_text(
            output.metadata.reset_index().to_json(
                orient="records",
                date_format="iso",
                force_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    aggregate = pd.DataFrame(aggregate_rows).set_index("optimizer")
    health = pd.DataFrame(health_rows).set_index("optimizer")
    independent_detail = pd.concat(
        independent_parts, ignore_index=True
    ).set_index(["optimizer", "date"])

    last_date = test_dates[-1]
    last_weights = pd.DataFrame(
        {
            name: frame.loc[last_date]
            for name, frame in weight_panels.items()
        }
    )
    weight_correlation = last_weights.corr(method="spearman")
    weight_l1_distance = pd.DataFrame(
        index=OPTIMIZERS, columns=OPTIMIZERS, dtype=float
    )
    for left in OPTIMIZERS:
        for right in OPTIMIZERS:
            weight_l1_distance.loc[left, right] = float(
                np.abs(last_weights[left] - last_weights[right]).sum()
            )

    aggregate.to_csv(OUTPUT_ROOT / "optimizer_comparison.csv")
    health.to_csv(OUTPUT_ROOT / "health_checks.csv")
    independent_detail.to_parquet(OUTPUT_ROOT / "independent_metrics.parquet")
    weight_correlation.to_csv(OUTPUT_ROOT / "last_date_weight_rank_correlation.csv")
    weight_l1_distance.to_csv(OUTPUT_ROOT / "last_date_weight_l1_distance.csv")
    benchmark.to_parquet(OUTPUT_ROOT / "benchmark_weights.parquet")
    alpha.to_parquet(OUTPUT_ROOT / "alpha_expected_return_5d.parquet")

    manifest = {
        "status": "passed",
        "optimizer_count": len(OPTIMIZERS),
        "optimizers": OPTIMIZERS,
        "dates": [f"{date:%Y-%m-%d}" for date in test_dates],
        "asset_count": len(assets),
        "index_symbol": INDEX_SYMBOL,
        "benchmark_retained_weight_coverage": retained_coverage,
        "alpha_input_type": "expected_return",
        "alpha_horizon_days": 5,
        "prediction_path": str(PREDICTION_PATH),
        "prediction_sha256": sha256(PREDICTION_PATH.read_bytes()).hexdigest(),
        "barra_root": str(RISK_ROOT),
        "market_data_root": str(DATA_ROOT),
        "raw_barra_specific_return_method": (
            "StockDailyBar.Return/10000 - exposure @ factor_return"
        ),
        "test_purpose": "optimizer acceptance, not performance backtest",
    }
    (OUTPUT_ROOT / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(aggregate.to_string())
    print(f"all {len(OPTIMIZERS)} optimizers passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
