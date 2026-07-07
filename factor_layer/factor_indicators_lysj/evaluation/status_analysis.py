from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import EvalConfig
from .predictivity import Predictivity
from .backtest import Backtest


class StatusAnalyzer:
    """Analyse factor performance segmented by market status labels.

    Receives *already-preprocessed* data (from ``prepare_status``)
    so that preprocessing is not duplicated.
    """

    def __init__(
        self,
        preprocessed_df: pd.DataFrame,
        status_labels: pd.Series,
        cfg: EvalConfig,
    ) -> None:
        df = preprocessed_df.copy()
        df["status"] = status_labels.reindex(df.index)
        self._df = df
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Per-group evaluators
    # ------------------------------------------------------------------
    def _ic_for_group(self, sub: pd.DataFrame) -> Dict[str, object]:
        return Predictivity.evaluate(
            factor_eval=sub["factor_eval"],
            target_eval=sub["fwd_ret"],
            date_index=sub["date"],
            min_obs=self._cfg.min_obs_per_day,
        )

    def _backtest_for_group(self, sub: pd.DataFrame) -> Dict[str, object]:
        holding_pnl = Backtest.long_short_holding_pnl(
            factor_for_eval=sub["factor_eval"],
            vwap=sub["vwap"],
            n_quantiles=self._cfg.n_quantiles,
            horizon=1,
            lookback=self._cfg.quantile_lookback,
        )
        holding_pnl_with_cost = Backtest.apply_transaction_cost(
            holding_pnl, self._cfg.holding_fee_rate,
        )
        stats = Backtest.holding_stats(holding_pnl)
        stats_with_cost = Backtest.holding_stats(holding_pnl_with_cost)
        total_ret = (
            float(holding_pnl["cum_pnl"].iloc[-1] - 1.0)
            if not holding_pnl.empty else np.nan
        )
        total_ret_with_cost = (
            float(holding_pnl_with_cost["cum_pnl"].iloc[-1] - 1.0)
            if not holding_pnl_with_cost.empty else np.nan
        )
        return {
            "holding_pnl": holding_pnl,
            "holding_stats": stats,
            "holding_pnl_with_cost": holding_pnl_with_cost,
            "holding_stats_with_cost": stats_with_cost,
            "holding_total_return": total_ret,
            "holding_total_return_with_cost": total_ret_with_cost,
        }

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, Dict[str, object]]:
        """Run the full per-status analysis.

        Returns ``{ status_value: { ic_summary, rank_ic_summary,
        daily_ic, daily_rank_ic, holding_pnl, holding_stats, … } }``
        """
        df = self._df.dropna(subset=["status"])
        groups = df.groupby("status", sort=True)
        results: Dict[str, Dict[str, object]] = {}

        for status_val, idx in groups.groups.items():
            sub = df.loc[idx].copy()
            if sub.empty:
                continue

            ic_result = self._ic_for_group(sub)
            bt_result = self._backtest_for_group(sub)

            results[str(status_val)] = {
                "n_bars": len(sub),
                "ic_summary": ic_result["ic_summary"],
                "rank_ic_summary": ic_result["rank_ic_summary"],
                "daily_ic": ic_result["daily_ic"],
                "daily_rank_ic": ic_result["daily_rank_ic"],
                **bt_result,
            }

        return results
