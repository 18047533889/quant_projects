"""
Production causal regime detector.

Unlike the full-sample ``detect_correlation_regime`` (which computes
percentile boundaries over the WHOLE series and is therefore
RESEARCH_ONLY), this detector freezes its thresholds at ``fit`` time on a
training window and then classifies each subsequent row from past-only
data (shift-1 rolling correlation).

Design (FP-P1-07):
- ``fit(train)`` freezes the correlation-regime thresholds (quantiles of a
  shift-1 rolling average pairwise correlation over the train window).
- ``detect(data)`` labels each row using only data strictly before it and
  the frozen thresholds, so the classification is causal and
  prefix-invariant: extending the series with later rows never changes
  earlier labels.

The detector is fail-closed: ``detect`` before ``fit`` raises.
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field

from factor_preprocess.regime.detector import RegimeState


@dataclass
class CausalRegimeDetector:
    """Production causal correlation-regime detector (FP-P1-07)."""

    window: int
    n_regimes: int = 2
    percentiles: Optional[List[float]] = None
    min_periods: Optional[int] = None
    time_col: str = "date"
    value_cols: Optional[List[str]] = None

    _fitted: bool = field(default=False, init=False)
    _boundaries: np.ndarray = field(default=None, init=False)
    _value_cols: List[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {self.n_regimes}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2 for correlation, got {self.window}")
        if self.min_periods is None:
            self.min_periods = self.window
        if self.percentiles is None:
            self.percentiles = [i / self.n_regimes for i in range(1, self.n_regimes)]
        if len(self.percentiles) != self.n_regimes - 1:
            raise ValueError(
                f"percentiles must have {self.n_regimes - 1} values for "
                f"{self.n_regimes} regimes"
            )

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "CausalRegimeDetector":
        """Freeze regime thresholds on the training window (past-only)."""
        if not train[self.time_col].is_monotonic_increasing:
            raise ValueError("train must be sorted by time_col")

        value_cols = self._resolve_value_cols(train)
        corr = _rolling_avg_corr_shift1(
            train, value_cols, self.window, self.min_periods, self.time_col
        )
        # Fail closed: require enough usable train data to fix thresholds.
        finite = corr[np.isfinite(corr)]
        if finite.size < self.window:
            raise ValueError(
                f"train has only {finite.size} usable correlation points; "
                f"need at least window={self.window}"
            )
        boundaries = np.nanquantile(finite, self.percentiles)
        self._boundaries = boundaries
        self._value_cols = value_cols
        self._fitted = True
        return self

    # ---------------------------------------------------------------- detect
    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Classify each row causally (shift-1) using frozen thresholds."""
        if not self._fitted:
            raise RuntimeError(
                "CausalRegimeDetector.detect requires fit() first (fail-closed)"
            )
        if not data[self.time_col].is_monotonic_increasing:
            raise ValueError("data must be sorted by time_col")

        value_cols = self._value_cols
        if any(c not in data.columns for c in value_cols):
            raise ValueError(
                f"detect data missing value columns: "
                f"{[c for c in value_cols if c not in data.columns]}"
            )

        avg_corr = _rolling_avg_corr_shift1(
            data, value_cols, self.window, self.min_periods, self.time_col
        )

        regime_labels = np.full(len(data), np.nan)
        valid = np.isfinite(avg_corr)
        regime_labels[valid] = np.digitize(
            avg_corr[valid], self._boundaries, right=False
        )

        # regime strength: normalized distance to nearest boundary.
        strength = np.full(len(data), np.nan)
        for i, v in enumerate(avg_corr):
            if not np.isfinite(v):
                continue
            label = int(regime_labels[i])
            strength[i] = _strength(v, label, self._boundaries, self.n_regimes)

        regime_series = pd.Series(regime_labels, index=data.index)
        transition = regime_series.diff().ne(0).fillna(False)
        return RegimeState(
            regime=regime_series,
            regime_strength=pd.Series(strength, index=data.index),
            transition_flag=transition,
        )

    # ---------------------------------------------------------------- helper
    def _resolve_value_cols(self, df: pd.DataFrame) -> List[str]:
        if self.value_cols is not None:
            missing = [c for c in self.value_cols if c not in df.columns]
            if missing:
                raise ValueError(f"missing value_cols: {missing}")
            return list(self.value_cols)
        cols = [
            c for c in df.columns
            if c != self.time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(cols) < 2:
            raise ValueError(f"Need at least 2 value columns, got {len(cols)}")
        return cols


def _strength(v: float, label: int, boundaries: np.ndarray, n_regimes: int) -> float:
    if label == 0:
        upper = boundaries[0]
        return max(0.0, (upper - v) / (upper + 1e-9))
    if label == n_regimes - 1:
        lower = boundaries[-1]
        return max(0.0, (v - lower) / (lower + 1e-9))
    lower = boundaries[label - 1]
    upper = boundaries[label]
    mid = (lower + upper) / 2
    half = (upper - lower) / 2
    return max(0.0, min(1.0, 1.0 - abs(v - mid) / (half + 1e-9)))


def _rolling_avg_corr_shift1(
    df: pd.DataFrame,
    value_cols: List[str],
    window: int,
    min_periods: int,
    time_col: str,
) -> np.ndarray:
    """Average pairwise correlation using only data strictly before each row."""
    value_matrix = df[value_cols].values
    n = len(df)
    avg_corr = np.full(n, np.nan)

    for i in range(window, n):
        # window [i - window, i), excluding current observation at i (shift-1).
        window_data = value_matrix[i - window:i, :]
        n_valid = np.sum(~np.isnan(window_data), axis=0)
        if np.any(n_valid < min_periods):
            continue
        corr_matrix = np.corrcoef(window_data, rowvar=False)
        if corr_matrix.ndim != 2 or corr_matrix.shape[0] < 2:
            continue
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off = corr_matrix[mask]
        avg_corr[i] = np.nanmean(off)

    return avg_corr
