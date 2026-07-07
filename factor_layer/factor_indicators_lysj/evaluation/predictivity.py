from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class Predictivity:
    """IC / Rank-IC daily correlation and summary statistics."""

    @staticmethod
    def daily_corr(
        series_x: pd.Series,
        series_y: pd.Series,
        date_index: pd.Series,
        method: str,
        min_obs: int,
    ) -> pd.Series:
        """Compute per-day correlation between *series_x* and *series_y*.

        Optimised path: pre-ranks the data once then uses vectorised
        numpy operations per group instead of ``pd.DataFrame.corr`` per day.
        """
        from scipy.stats import rankdata

        x_arr = series_x.to_numpy(dtype=np.float64, na_value=np.nan)
        y_arr = series_y.to_numpy(dtype=np.float64, na_value=np.nan)

        # Encode dates as integer group ids for fast splitting
        dates = np.asarray(date_index)
        unique_dates, inverse = np.unique(dates, return_inverse=True)

        results = np.empty(len(unique_dates), dtype=np.float64)

        for g in range(len(unique_dates)):
            mask = inverse == g
            xg = x_arr[mask]
            yg = y_arr[mask]
            # Drop NaN pairs
            valid = ~(np.isnan(xg) | np.isnan(yg))
            xv = xg[valid]
            yv = yg[valid]
            if len(xv) < min_obs:
                results[g] = np.nan
                continue

            if method == "spearman":
                xv = rankdata(xv)
                yv = rankdata(yv)

            # Pearson on (possibly ranked) arrays
            xm = xv - xv.mean()
            ym = yv - yv.mean()
            denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
            results[g] = float((xm * ym).sum() / denom) if denom > 0 else np.nan

        out = pd.Series(results, index=unique_dates, name=f"{method}_corr")
        out.index.name = "date"
        return out

    @staticmethod
    def summary_from_daily_series(s: pd.Series) -> Dict[str, float]:
        s = s.dropna()
        if s.empty:
            return {
                "mean": np.nan,
                "std": np.nan,
                "ir": np.nan,
                "win_rate": np.nan,
                "n_days": 0,
            }

        mean = s.mean()
        std = s.std(ddof=1)
        ir = mean / std if std and not np.isclose(std, 0.0) else np.nan
        win_rate = (s > 0).mean()
        return {
            "mean": float(mean),
            "std": float(std),
            "ir": float(ir) if pd.notna(ir) else np.nan,
            "win_rate": float(win_rate),
            "n_days": int(s.shape[0]),
        }

    @classmethod
    def evaluate(
        cls,
        factor_eval: pd.Series,
        target_eval: pd.Series,
        date_index: pd.Series,
        min_obs: int,
    ) -> Dict[str, object]:
        daily_ic = cls.daily_corr(
            factor_eval, target_eval, date_index,
            method="pearson", min_obs=min_obs,
        )
        daily_rank_ic = cls.daily_corr(
            factor_eval, target_eval, date_index,
            method="spearman", min_obs=min_obs,
        )
        ic_summary = cls.summary_from_daily_series(daily_ic)
        rank_ic_summary = cls.summary_from_daily_series(daily_rank_ic)
        return {
            "daily_ic": daily_ic.rename("ic"),
            "daily_rank_ic": daily_rank_ic.rename("rank_ic"),
            "ic_summary": ic_summary,
            "rank_ic_summary": rank_ic_summary,
        }
