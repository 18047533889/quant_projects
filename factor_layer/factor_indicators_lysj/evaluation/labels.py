from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EvalConfig


class MarketStatusLabel:
    """Market status classification labels for factor analysis.

    All labels are computed strictly using data up to (and including) time *t*,
    with no look-ahead bias.
    """

    # CME Globex NQ Regular Trading Hours (ET), encoded as HHMM integers
    _OPEN_START: int = 930
    _OPEN_END: int = 1000
    _CLOSE_START: int = 1530
    _RTH_END: int = 1600

    def __init__(
        self,
        market_df: pd.DataFrame,
        cfg: EvalConfig,
    ) -> None:
        ts_col = cfg.timestamp_col
        close_col = cfg.close_col
        volume_col = cfg.volume_col

        df = market_df[[ts_col, close_col, volume_col]].copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.drop_duplicates(subset=[ts_col]).sort_values(ts_col)
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

        self._df = df
        self._close_col = close_col
        self._volume_col = volume_col
        self._vol_window = cfg.vol_window
        self._vol_rank_window = cfg.vol_rank_window
        self._vol_low_pct = cfg.vol_low_pct
        self._vol_high_pct = cfg.vol_high_pct

    # ------------------------------------------------------------------
    # Public label methods
    # ------------------------------------------------------------------
    def volatility_regime(self) -> pd.Series:
        """Classify each bar's volatility regime (LV / MV / HV).

        Steps (all look-back only, no future data):
          1. 1-bar log return: ln(close_t / close_{t-1})
          2. Rolling std over *vol_window* bars → realised volatility
          3. Rolling percentile rank over *vol_rank_window* bars
          4. pct_rank < 1/3 → LV; 1/3–2/3 → MV; ≥ 2/3 → HV; NaN → pd.NA
        """
        close = self._df[self._close_col]
        log_ret = np.log(close / close.shift(1))

        realized_vol = log_ret.rolling(
            window=self._vol_window, min_periods=self._vol_window,
        ).std(ddof=1)

        pct_rank = realized_vol.rolling(
            window=self._vol_rank_window, min_periods=self._vol_rank_window,
        ).rank(pct=True)

        labels = pd.array(
            ["MV"] * len(self._df), dtype=pd.StringDtype(),
        )
        lv_mask = (pct_rank < self._vol_low_pct).to_numpy()
        hv_mask = (pct_rank >= self._vol_high_pct).to_numpy()
        na_mask = pct_rank.isna().to_numpy()

        labels[lv_mask] = "LV"
        labels[hv_mask] = "HV"
        labels[na_mask] = pd.NA

        return pd.Series(labels, index=self._df.index, name="volatility_regime")

    def intraday_session(self) -> pd.Series:
        """Classify each bar by intraday trading session.

        Purely time-based (no computation, no look-ahead bias):
          OPEN (09:30–10:00), MIDDAY (10:00–15:30),
          CLOSE (15:30–16:00), OVERNIGHT (all other).
        """
        ts = self._df.index
        hhmm = ts.hour * 100 + ts.minute

        labels = pd.array(
            ["OVERNIGHT"] * len(ts), dtype=pd.StringDtype(),
        )

        open_mask = (hhmm >= self._OPEN_START) & (hhmm < self._OPEN_END)
        midday_mask = (hhmm >= self._OPEN_END) & (hhmm < self._CLOSE_START)
        close_mask = (hhmm >= self._CLOSE_START) & (hhmm < self._RTH_END)

        labels[open_mask] = "OPEN"
        labels[midday_mask] = "MIDDAY"
        labels[close_mask] = "CLOSE"

        return pd.Series(labels, index=ts, name="intraday_session")

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def get_labels(self, method: str) -> pd.Series:
        """Dispatch to the named label method."""
        if method == "volatility_regime":
            return self.volatility_regime()
        elif method == "intraday_session":
            return self.intraday_session()
        else:
            raise ValueError(
                f"Unknown status method '{method}'. "
                "Use 'volatility_regime' or 'intraday_session'."
            )
