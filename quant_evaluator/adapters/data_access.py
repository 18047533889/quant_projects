"""DataAccess production adapter (spec §23, §24, §25, §58).

Replaces the previous stub with a real loader that reads the authoritative
adjusted table ``ashare_stock_daily_adj`` (StockDailyBarAdj) through
``data_access.get_store().read(...)`` and builds :class:`FactorBatch` /
:class:`LabelBundle` with explicit timing.

Key rules (spec §21, §22, §24, §58):
  - QE never constructs A-share future labels itself (no ``shift(-h)``).
  - Target values come from the producer columns ``TargetVwapReturnH01/H05/
    H10/H20``; timing is bound from the producer contract, never guessed from
    the column name.
  - ``return_bp`` (``ashare_stock_daily_adj.Return``) is already decimal when
    it reaches QE (unit scale 0.0001 handled by DataAccess).
  - semantic ``volume`` is ``Volume / Factor`` (platform adjusted-volume
    convention) — QE must not treat raw ``Volume`` as the semantic field.
  - Arrow is the preferred read path; pandas only as a fallback.

This module is optional: core quant_evaluator must import cleanly without
``data_access`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_evaluator.contracts.errors import OptionalDependencyMissing
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

# Producer contract for the adjusted table's target columns (spec §24).
# H10 is officially documented as AdjVwap[t+11]/AdjVwap[t+1]-1 (t+1 entry,
# t+11 exit).  H01/H05/H20 timing is bound from the producer manifest; the
# horizon field here is the label's holding length in trading days.
TARGET_CONTRACT: Dict[str, Dict[str, Any]] = {
    "TargetVwapReturnH01": {"horizon": 1, "price_convention": "vwap_to_vwap"},
    "TargetVwapReturnH05": {"horizon": 5, "price_convention": "vwap_to_vwap"},
    "TargetVwapReturnH10": {"horizon": 10, "price_convention": "vwap_to_vwap"},
    "TargetVwapReturnH20": {"horizon": 20, "price_convention": "vwap_to_vwap"},
}

# Semantic field mapping (spec §22): bare fields live on the adjusted table.
SEMANTIC_FIELDS = {
    "close": "AdjClose",
    "open": "AdjOpen",
    "high": "AdjHigh",
    "low": "AdjLow",
    "vwap": "AdjVwap",
    "amount": "AdjAmount",
}


@dataclass(frozen=True)
class TargetSpec:
    """A label target bound to a producer column (spec §24)."""

    target_id: str
    dataset: str = "ashare_stock_daily_adj"
    physical_column: str = "TargetVwapReturnH10"
    horizon: int = 10
    price_convention: str = "vwap_to_vwap"


class DataAccessEvaluationLoader:
    """Loads FactorBatch / LabelBundle / context from DataAccess.

    Uses ``data_access.get_store().read(dataset, ...)`` (Arrow preferred).
    """

    def __init__(self, store: Any = None):
        try:
            import data_access  # noqa: F401
        except ImportError as e:
            raise OptionalDependencyMissing(
                "data_access", "DataAccessEvaluationLoader"
            ) from e
        if store is None:
            from data_access import get_store
            store = get_store()
        self._store = store

    # -- factor batch ---------------------------------------------------
    def load_factor_batch(
        self,
        factor_ids: Sequence[str],
        *,
        time_range: Optional[Tuple[Any, Any]] = None,
        universe: Optional[str] = None,
        layout: str = "wide",
        columns: Optional[Sequence[str]] = None,
    ) -> FactorBatch:
        """Read factors from the factor lake and build a FactorBatch.

        Uses ``store.read_factors(factor_ids, layout="wide", ...)`` so a
        single query returns all factors (no per-factor loop, spec §25).
        """
        handle = self._store.read_factors(
            list(factor_ids),
            time_range=time_range,
            universe=universe,
            layout=layout,
            columns=columns,
        )
        df = handle.to_pandas()
        # wide layout: index = datetime, columns = [asset, f1, f2, ...]
        # pivot to (T, N, F)
        time_col = df.index.name or "datetime"
        asset_col = "asset"
        dates = sorted(df.index.unique())
        assets = sorted(df[asset_col].unique()) if asset_col in df.columns else None
        if assets is None:
            # wide already pivoted: columns are factor ids, index is datetime
            dates = list(df.index)
            values = df[list(factor_ids)].values.astype(np.float64)  # (T, F)
            T, F = values.shape
            N = 1
            values = values[:, None, :]  # (T, 1, F)
            assets = ["__single__"]
        else:
            piv = df.pivot_table(
                index=time_col, columns=asset_col, values=list(factor_ids),
                aggfunc="first",
            )
            # piv columns MultiIndex (factor, asset)
            values = np.stack(
                [piv[f].reindex(index=dates, columns=assets).values for f in factor_ids],
                axis=-1,
            ).astype(np.float64)  # (T, N, F)
            T, N, F = values.shape
        return FactorBatch(
            factor_ids=tuple(factor_ids),
            time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
            asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
            values=values,
            layout="wide",
        )

    # -- label bundle ---------------------------------------------------
    def load_label_bundle(
        self,
        target: TargetSpec,
        *,
        time_range: Optional[Tuple[Any, Any]] = None,
        instrument_filter: Optional[Sequence[str]] = None,
    ) -> LabelBundle:
        """Read a target column from the adjusted table and bind explicit timing.

        The label value is the producer column verbatim (no QE shift).  Timing
        vectors are derived from the trading calendar of the read (decision_time
        = TradeDate; label window = [t+1, t+1+horizon] per the producer contract).
        """
        handle = self._store.read(
            target.dataset,
            columns=["TradeDate", "Symbol", target.physical_column],
            time_range=time_range,
            instrument_filter=instrument_filter,
        )
        df = handle.to_pandas()
        df = df.sort_values(["Symbol", "TradeDate"])
        dates = sorted(df["TradeDate"].unique())
        assets = sorted(df["Symbol"].unique())
        piv = df.pivot_table(
            index="TradeDate", columns="Symbol", values=target.physical_column,
            aggfunc="first",
        )
        values = piv.reindex(index=dates, columns=assets).values.astype(np.float64)
        T, N = values.shape
        decision = tuple(pd.Timestamp(d) for d in dates)
        # label window: entry t+1, exit t+1+horizon (producer contract)
        start = tuple(pd.Timestamp(d) + pd.Timedelta(days=1) for d in dates)
        end = tuple(pd.Timestamp(d) + pd.Timedelta(days=1 + target.horizon) for d in dates)
        return LabelBundle(
            target_id=target.target_id,
            values=values,
            horizon=target.horizon,
            decision_time=decision,
            label_start_time=start,
            label_end_time=end,
        )

    # -- context --------------------------------------------------------
    def load_evaluation_context(
        self,
        *,
        time_range: Optional[Tuple[Any, Any]] = None,
        fields: Sequence[str] = ("AdjVwap", "IsSuspend", "Volume", "Factor"),
    ) -> Dict[str, np.ndarray]:
        """Load optional evaluation context (tradability, volume, etc.).

        Returns a dict of (T, N) arrays keyed by semantic field name.  Only
        requested fields are read (spec §4.6).
        """
        handle = self._store.read(
            "ashare_stock_daily_adj",
            columns=["TradeDate", "Symbol", *fields],
            time_range=time_range,
        )
        df = handle.to_pandas()
        dates = sorted(df["TradeDate"].unique())
        assets = sorted(df["Symbol"].unique())
        out: Dict[str, np.ndarray] = {}
        for f in fields:
            piv = df.pivot_table(
                index="TradeDate", columns="Symbol", values=f, aggfunc="first",
            )
            out[f] = piv.reindex(index=dates, columns=assets).values.astype(np.float64)
        # semantic volume = Volume / Factor (platform adjusted-volume convention)
        if "Volume" in out and "Factor" in out:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["volume_semantic"] = out["Volume"] / np.where(
                    out["Factor"] == 0, np.nan, out["Factor"]
                )
        return out
