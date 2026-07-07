from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class Backtest:
    """Layered quantile backtest and long-short holding backtest."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_quantile_groups(
        factor_for_eval: pd.Series,
        n_quantiles: int,
        lookback: int = 1000,
    ) -> pd.Series:
        """Vectorized rolling quantile group assignment.

        Uses ``pd.Series.rolling().rank(pct=True)`` (C-level) then
        buckets into ``[1..n_quantiles]`` via ``ceil(pct * n_q)``.
        """
        groups_all = pd.Series(np.nan, index=factor_for_eval.index, dtype=float)
        valid = factor_for_eval.dropna()
        if valid.empty:
            return groups_all.rename("group")

        pct_rank = valid.rolling(
            window=lookback, min_periods=lookback,
        ).rank(pct=True)
        grp = np.ceil(pct_rank.to_numpy() * n_quantiles)
        grp = np.where(np.isnan(pct_rank.to_numpy()), np.nan, grp)
        grp = np.clip(grp, 1, n_quantiles)
        groups_all.loc[valid.index] = grp
        return groups_all.rename("group")

    @staticmethod
    def _capped_position_signal(
        long_trigger: pd.Series,
        short_trigger: pd.Series,
        horizon: int,
    ) -> pd.Series:
        n = len(long_trigger)
        lt = long_trigger.to_numpy(dtype=bool)
        st = short_trigger.to_numpy(dtype=bool)

        long_events = [(int(i), 1) for i in np.where(lt)[0]]
        short_events = [(int(i), -1) for i in np.where(st)[0]]
        events = sorted(long_events + short_events, key=lambda x: x[0])

        pos = np.zeros(n, dtype=np.float64)
        hold_until = -1

        for idx, direction in events:
            if idx <= hold_until:
                continue
            end = min(idx + horizon, n)
            pos[idx:end] = direction
            hold_until = end - 1

        return pd.Series(pos, index=long_trigger.index, name="net_position")

    @staticmethod
    def _infer_periods_per_year(index: pd.Index) -> float:
        ts = pd.Series(pd.to_datetime(index)).dropna().sort_values()
        if ts.shape[0] < 3:
            return 252.0

        sec = ts.diff().dt.total_seconds().dropna()
        sec = sec[sec > 0]
        if sec.empty:
            return 252.0

        median_sec = float(sec.median())
        if median_sec <= 0:
            return 252.0
        return (365.25 * 24 * 3600) / median_sec

    # ------------------------------------------------------------------
    # Layered backtest
    # ------------------------------------------------------------------
    @classmethod
    def layered_single_period_return(
        cls,
        factor_for_eval: pd.Series,
        target_return_n: pd.Series,
        n_quantiles: int,
        horizon: int,
        lookback: int = 1000,
    ) -> pd.Series:
        tmp = pd.DataFrame(
            {"factor": factor_for_eval, "ret_n": target_return_n},
        ).dropna()
        if tmp.empty:
            return pd.Series(dtype=float, name="avg_single_period_ret")

        # Vectorized rolling quantile assignment
        pct_rank = tmp["factor"].rolling(
            window=lookback, min_periods=lookback,
        ).rank(pct=True)
        grp_arr = np.ceil(pct_rank.to_numpy() * n_quantiles)
        grp_arr = np.where(np.isnan(pct_rank.to_numpy()), np.nan, grp_arr)
        grp_arr = np.clip(grp_arr, 1, n_quantiles)

        tmp["group"] = grp_arr
        valid = tmp.dropna(subset=["group"])

        grp_mean_n = valid.groupby("group")["ret_n"].mean()
        grp_mean_single = np.power(1.0 + grp_mean_n, 1.0 / horizon) - 1.0
        grp_mean_single.index = grp_mean_single.index.astype(int)
        grp_mean_single.index.name = "group"
        grp_mean_single.name = "avg_single_period_ret"
        return grp_mean_single

    # ------------------------------------------------------------------
    # Long-short holding backtest
    # ------------------------------------------------------------------
    @classmethod
    def long_short_holding_pnl(
        cls,
        factor_for_eval: pd.Series,
        vwap: pd.Series,
        n_quantiles: int,
        horizon: int,
        lookback: int = 1000,
    ) -> pd.DataFrame:
        groups = cls._assign_quantile_groups(
            factor_for_eval, n_quantiles=n_quantiles, lookback=lookback,
        )

        long_trigger = (groups == n_quantiles).shift(1).fillna(0.0)
        short_trigger = (groups == 1).shift(1).fillna(0.0)

        net_position = cls._capped_position_signal(
            long_trigger, short_trigger, horizon,
        )

        one_period_ret = vwap.shift(-1) / vwap - 1.0
        pnl = (net_position * one_period_ret).rename("pnl")
        cum_pnl = (1.0 + pnl.fillna(0.0)).cumprod().rename("cum_pnl")

        out = pd.DataFrame(
            {
                "pnl": pnl,
                "cum_pnl": cum_pnl,
                "net_position": net_position,
                "long_active": (net_position > 0).astype(float),
                "short_active": (net_position < 0).astype(float),
            },
            index=factor_for_eval.index,
        )
        out.index.name = "timestamp"
        return out

    # ------------------------------------------------------------------
    # Transaction cost
    # ------------------------------------------------------------------
    @staticmethod
    def apply_transaction_cost(
        holding_pnl: pd.DataFrame, fee_rate: float,
    ) -> pd.DataFrame:
        if holding_pnl is None or holding_pnl.empty:
            return pd.DataFrame(
                columns=[
                    "pnl", "cum_pnl", "net_position",
                    "long_active", "short_active", "transaction_cost",
                ],
            )

        out = holding_pnl.copy()
        net_pos = out["net_position"].fillna(0.0)
        pos_change = net_pos.diff().abs().fillna(net_pos.abs())
        transaction_cost = (pos_change * fee_rate).rename("transaction_cost")
        pnl_after_cost = (out["pnl"].fillna(0.0) - transaction_cost).rename("pnl")
        cum_pnl_after_cost = (1.0 + pnl_after_cost).cumprod().rename("cum_pnl")

        out["transaction_cost"] = transaction_cost
        out["pnl"] = pnl_after_cost
        out["cum_pnl"] = cum_pnl_after_cost
        return out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @classmethod
    def holding_stats(cls, holding_pnl: pd.DataFrame) -> Dict[str, float]:
        _empty = {
            "holding_sharpe": np.nan,
            "holding_max_drawdown": np.nan,
            "holding_annual_return": np.nan,
            "holding_turnover": np.nan,
            "holding_avg_daily_trades": np.nan,
            "holding_win_rate": np.nan,
            "holding_profit_loss_ratio": np.nan,
        }
        if holding_pnl is None or holding_pnl.empty:
            return _empty

        pnl = holding_pnl["pnl"].dropna()
        equity = holding_pnl["cum_pnl"].reindex(pnl.index).dropna()
        net_pos = holding_pnl["net_position"].reindex(pnl.index).fillna(0.0)
        if pnl.empty:
            return _empty

        periods_per_year = cls._infer_periods_per_year(pnl.index)
        pnl_std = float(pnl.std(ddof=1))
        sharpe = np.nan
        if pnl_std > 0 and not np.isclose(pnl_std, 0.0):
            sharpe = float((pnl.mean() / pnl_std) * np.sqrt(periods_per_year))

        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

        total_return = float(equity.iloc[-1] - 1.0)
        span_days = (
            pd.to_datetime(pnl.index[-1]) - pd.to_datetime(pnl.index[0])
        ).total_seconds() / (24 * 3600)
        if span_days > 0:
            years = span_days / 365.25
            annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        else:
            annual_return = np.nan

        pos_change = net_pos.diff().abs().fillna(net_pos.abs())
        exposure = net_pos.abs().sum()
        turnover = float(pos_change.sum() / exposure) if exposure > 0 else np.nan
        trade_count_by_bar = pos_change
        avg_daily_trades = float(
            trade_count_by_bar.groupby(
                pd.to_datetime(trade_count_by_bar.index).date,
            ).sum().mean()
        )

        active_mask = net_pos != 0
        active_pnl = pnl[active_mask]
        win_rate = float((active_pnl > 0).mean()) if not active_pnl.empty else np.nan

        profit_sum = float(active_pnl[active_pnl > 0].sum())
        loss_sum = float(-active_pnl[active_pnl < 0].sum())
        profit_loss_ratio = float(profit_sum / loss_sum) if loss_sum > 0 else np.nan

        return {
            "holding_sharpe": sharpe,
            "holding_max_drawdown": max_drawdown,
            "holding_annual_return": annual_return,
            "holding_turnover": turnover,
            "holding_avg_daily_trades": avg_daily_trades,
            "holding_win_rate": win_rate,
            "holding_profit_loss_ratio": profit_loss_ratio,
        }

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    @classmethod
    def evaluate(
        cls,
        factor_eval: pd.Series,
        target_eval: pd.Series,
        vwap: pd.Series,
        n_quantiles: int,
        horizon: int,
        fee_rate: float,
        lookback: int = 1000,
    ) -> Dict[str, object]:
        layered = cls.layered_single_period_return(
            factor_for_eval=factor_eval,
            target_return_n=target_eval,
            n_quantiles=n_quantiles,
            horizon=horizon,
            lookback=lookback,
        )
        holding_pnl = cls.long_short_holding_pnl(
            factor_for_eval=factor_eval,
            vwap=vwap,
            n_quantiles=n_quantiles,
            horizon=horizon,
            lookback=lookback,
        )
        holding_pnl_with_cost = cls.apply_transaction_cost(holding_pnl, fee_rate)
        h_stats = cls.holding_stats(holding_pnl)
        h_stats_with_cost = cls.holding_stats(holding_pnl_with_cost)
        holding_total_ret = (
            float(holding_pnl["cum_pnl"].iloc[-1] - 1.0)
            if not holding_pnl.empty else np.nan
        )
        holding_total_ret_with_cost = (
            float(holding_pnl_with_cost["cum_pnl"].iloc[-1] - 1.0)
            if not holding_pnl_with_cost.empty else np.nan
        )
        return {
            "layered_single_period": layered,
            "holding_pnl": holding_pnl,
            "holding_stats": h_stats,
            "holding_pnl_with_cost": holding_pnl_with_cost,
            "holding_stats_with_cost": h_stats_with_cost,
            "holding_total_return": holding_total_ret,
            "holding_total_return_with_cost": holding_total_ret_with_cost,
        }
