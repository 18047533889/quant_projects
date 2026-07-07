from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import pandas as pd

from .config import EvalConfig
from .preprocessing import prepare_base, prepare_horizon, prepare_status
from .labels import MarketStatusLabel
from .predictivity import Predictivity
from .backtest import Backtest
from .status_analysis import StatusAnalyzer
from . import serialization


# ---------------------------------------------------------------------------
# Single-horizon evaluation
# ---------------------------------------------------------------------------
def evaluate_horizon(
    base_df: pd.DataFrame, horizon: int, cfg: EvalConfig,
) -> Dict[str, Any]:
    """Run full IC + backtest evaluation for a single horizon.

    Parameters
    ----------
    base_df : pd.DataFrame
        Output of ``prepare_base`` (factor_raw, vwap, factor_z, date).
    horizon : int
        Forward-return horizon in bars.
    cfg : EvalConfig
        Evaluation configuration.
    """
    df = prepare_horizon(base_df, horizon, cfg)

    pred = Predictivity.evaluate(
        factor_eval=df["factor_eval"],
        target_eval=df["target_eval"],
        date_index=df["date"],
        min_obs=cfg.min_obs_per_day,
    )
    bt = Backtest.evaluate(
        factor_eval=df["factor_eval"],
        target_eval=df["target_eval"],
        vwap=df["vwap"],
        n_quantiles=cfg.n_quantiles,
        horizon=horizon,
        fee_rate=cfg.holding_fee_rate,
        lookback=cfg.quantile_lookback,
    )

    ic_summary = pred["ic_summary"]
    rank_ic_summary = pred["rank_ic_summary"]

    summary = {
        "horizon": horizon,
        "ic_mean": ic_summary["mean"],
        "ic_std": ic_summary["std"],
        "icir": ic_summary["ir"],
        "ic_win_rate": ic_summary["win_rate"],
        "ic_n_days": ic_summary["n_days"],
        "rank_ic_mean": rank_ic_summary["mean"],
        "rank_ic_std": rank_ic_summary["std"],
        "rank_icir": rank_ic_summary["ir"],
        "rank_ic_win_rate": rank_ic_summary["win_rate"],
        "rank_ic_n_days": rank_ic_summary["n_days"],
        "holding_total_return": bt["holding_total_return"],
        **bt["holding_stats"],
        "holding_total_return_with_cost": bt["holding_total_return_with_cost"],
        **{f"{k}_with_cost": v for k, v in bt["holding_stats_with_cost"].items()},
    }

    return {
        "summary": summary,
        "daily_ic": pred["daily_ic"],
        "daily_rank_ic": pred["daily_rank_ic"],
        "layered_single_period": bt["layered_single_period"],
        "holding_pnl": bt["holding_pnl"],
        "holding_stats": bt["holding_stats"],
        "holding_pnl_with_cost": bt["holding_pnl_with_cost"],
        "holding_stats_with_cost": bt["holding_stats_with_cost"],
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_evaluation(
    factor_input,
    market_df: pd.DataFrame,
    cfg: EvalConfig | None = None,
) -> Dict[str, Any]:
    """Execute the full factor evaluation pipeline.

    Parameters
    ----------
    factor_input : pd.Series | pd.DataFrame
        Raw factor values.
        - DataFrame: must contain [cfg.factor_timestamp_col, cfg.factor_col].
        - Series: index must be timestamps, values are raw factor.
    market_df : pd.DataFrame
        Market OHLCV + VWAP data with columns specified in *cfg*.
    cfg : EvalConfig, optional
        Evaluation configuration.  Uses defaults if not provided.

    Returns
    -------
    dict
        Raw evaluation results (pandas objects) with keys:

        - ``full_evaluation`` : Dict[int, payload]  — per-horizon results
        - ``status_evaluation`` : Dict[str, Dict[str, payload]]  — per-method,
          per-status results
    """
    if cfg is None:
        cfg = EvalConfig()

    # ── Step 1: Common preprocessing ──────────────────────────────────
    base_df = prepare_base(factor_input, market_df, cfg)

    # ── Step 2: Full evaluation across horizons ───────────────────────
    full_results: Dict[int, Dict[str, Any]] = {}
    for horizon in cfg.horizons:
        full_results[horizon] = evaluate_horizon(base_df, horizon, cfg)

    # ── Step 3: Generate status labels ────────────────────────────────
    labeler = MarketStatusLabel(market_df, cfg)

    # ── Step 4: Status evaluation ─────────────────────────────────────
    status_base_df = prepare_status(base_df, cfg)
    status_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in cfg.status_methods:
        labels = labeler.get_labels(method)
        analyzer = StatusAnalyzer(status_base_df, labels, cfg)
        status_results[method] = analyzer.analyze()

    return {
        "full_evaluation": full_results,
        "status_evaluation": status_results,
    }


# ---------------------------------------------------------------------------
# Serialisation wrapper
# ---------------------------------------------------------------------------
def serialize_results(
    raw_results: Dict[str, Any],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """Convert raw evaluation results to a fully JSON-serialisable dict.

    Parameters
    ----------
    raw_results : dict
        Output of ``run_evaluation``.
    cfg : EvalConfig
        Configuration (included in the output for reproducibility).

    Returns
    -------
    dict
        JSON-safe dict ready for ``json.dumps``.
    """
    output: Dict[str, Any] = {
        "config": asdict(cfg),
        "full_evaluation": {
            str(h): serialization.serialize_horizon_result(payload)
            for h, payload in raw_results["full_evaluation"].items()
        },
        "status_evaluation": {
            method: serialization.serialize_status_result(results)
            for method, results in raw_results["status_evaluation"].items()
        },
    }
    return output


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
def print_brief(results: Dict[str, Any]) -> None:
    """Print a compact summary of the full evaluation results."""
    full = results["full_evaluation"]
    print("=" * 88)
    print("Factor Evaluation Summary")
    print("Columns: horizon | IC mean/std/IR/win | RankIC mean/std/IR/win | HoldPnl")
    print("=" * 88)
    for h in sorted(full):
        s = full[h]["summary"]
        print(
            f"H={h:>3} | "
            f"IC {s['ic_mean']:+.6f}/{s['ic_std']:.6f}/"
            f"{s['icir']:+.4f}/{s['ic_win_rate']:.2%} | "
            f"RankIC {s['rank_ic_mean']:+.6f}/{s['rank_ic_std']:.6f}/"
            f"{s['rank_icir']:+.4f}/{s['rank_ic_win_rate']:.2%} | "
            f"HoldPnl {s['holding_total_return']:+.4f}"
        )
    print("=" * 88)

    # Status summary
    for method, status_groups in results["status_evaluation"].items():
        print(f"\n  Status: {method}")
        print(f"  {'Label':<12} | {'#Bars':>8} | "
              f"{'IC_mean':>9} {'ICIR':>8} | "
              f"{'RkIC_mean':>9} {'RkICIR':>8} | "
              f"{'TotalRet':>9}")
        print("  " + "-" * 80)
        for label in sorted(status_groups):
            r = status_groups[label]
            ic = r["ic_summary"]
            ric = r["rank_ic_summary"]
            print(
                f"  {label:<12} | {r['n_bars']:>8d} | "
                f"{ic['mean']:>+9.6f} {ic['ir']:>+8.4f} | "
                f"{ric['mean']:>+9.6f} {ric['ir']:>+8.4f} | "
                f"{r['holding_total_return']:>+9.4f}"
            )
    print()
