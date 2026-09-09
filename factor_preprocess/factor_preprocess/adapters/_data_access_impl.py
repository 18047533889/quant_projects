"""
Default exposure provider backed by the optional ``data_access`` package.

Implements the :class:`~factor_preprocess.adapters.data_access.ExposureProvider`
protocol. When ``data_access`` is importable it sources real exposure context
(industry / size / sector / beta / custom) from the data_access store. When the
package is missing, or the underlying data is not available, it raises a clean
:class:`OptionalDependencyMissing` rather than a raw ``ModuleNotFoundError`` or
``FileNotFoundError`` (fail-closed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from factor_preprocess.adapters.data_access import (
    OptionalDependencyMissing,
    REQUIRED_PROVENANCE_KEYS,
)

# Dataset names used by the data_access store for each exposure source.
_INDUSTRY_DATASET = "ashare_stock_industry"
_SIZE_DATASET = "ashare_stock_valuation_daily"
_SECTOR_DATASET = "ashare_stock_industry"
_BETA_DATASET = "ashare_stock_daily"
_UNIVERSE_DATASET = "ashare_universe_daily"


def _build_metadata(
    market: str,
    classification_version: str,
    snapshot_ref: str,
    knowledge_time: str,
    effective_time: str,
    universe_ref: str,
) -> Dict[str, Any]:
    """Build a provenance metadata dict with all mandatory keys."""
    return {
        "knowledge_time": knowledge_time,
        "effective_time": effective_time,
        "snapshot_ref": snapshot_ref,
        "classification_version": classification_version,
        "market": market,
        "universe_ref": universe_ref,
    }


class DefaultExposureProvider:
    """
    Real exposure provider sourcing from the ``data_access`` package.

    Every returned bundle carries mandatory provenance metadata
    (``knowledge_time``, ``effective_time``, ``snapshot_ref``,
    ``classification_version``, ``market``, ``universe_ref``) so downstream
    neutralization can reason about point-in-time correctness.
    """

    def __init__(self) -> None:
        self._store = None
        self._registry = None
        try:
            import data_access  # noqa: F401

            self._store = data_access.get_store()
            self._registry = self._store.registry
        except ImportError:
            # data_access not installed -> every fetch fails closed.
            self._store = None
            self._registry = None

    def _require_store(self, feature: str) -> Any:
        if self._store is None:
            raise OptionalDependencyMissing(
                package_name="data_access",
                feature_name=feature,
            )
        return self._store

    def _read_panel(
        self,
        dataset: str,
        columns: List[str],
        market: str,
        start_date: Optional[str],
        end_date: Optional[str],
        assets: Optional[List[str]],
        feature: str,
    ) -> Dict[str, Any]:
        """Read a panel from the data_access store, fail-closed on any error."""
        store = self._require_store(feature)
        try:
            requested_columns = list(columns)
            if "UpdateTime" not in requested_columns:
                requested_columns.append("UpdateTime")
            handle = store.read_frame(
                dataset,
                columns=requested_columns,
                time_range=(start_date, end_date) if start_date or end_date else None,
                instrument_filter=assets,
            )
            df = handle.to_pandas()
        except Exception as exc:  # fail closed, but do not mislabel schema/data errors as missing package
            raise RuntimeError(
                f"data_access read failed for {feature}: {type(exc).__name__}: {exc}"
            ) from exc

        if df is None or df.empty:
            raise ValueError(f"data_access returned no rows for {feature}")

        time_col = "TradeDate"
        sym_col = "Symbol"
        required = {time_col, sym_col, "UpdateTime"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{feature} missing required identity/time columns: {sorted(missing)}")
        if df[list(required)].isna().any().any():
            raise ValueError(f"{feature} contains null identity/knowledge time")
        return df

    @staticmethod
    def _to_panel(df, value_col: str, feature: str):
        """Create a deterministic TN panel without coercing instrument identity."""
        keys = ["TradeDate", "Symbol"]
        duplicate = df.duplicated(keys, keep=False)
        if duplicate.any():
            counts = df.loc[duplicate].groupby(keys, dropna=False)[value_col].nunique(dropna=False)
            conflicts = counts[counts > 1]
            if not conflicts.empty:
                raise ValueError(f"{feature} conflicting duplicate keys: {list(conflicts.index[:5])}")
            df = df.drop_duplicates(keys, keep="first")
        dates = np.asarray(sorted(df["TradeDate"].unique()))
        assets = np.asarray(sorted(df["Symbol"].unique()))
        values = df.pivot(index="TradeDate", columns="Symbol", values=value_col).reindex(
            index=dates, columns=assets
        ).to_numpy()
        knowledge_time = df["UpdateTime"].max()
        if knowledge_time is None or str(knowledge_time) in ("", "unknown", "NaT"):
            raise ValueError(f"{feature} has no real knowledge_time")
        return values, dates, assets, str(knowledge_time)

    def _bundle(
        self,
        values: np.ndarray,
        dates: np.ndarray,
        assets: np.ndarray,
        metadata: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bundle: Dict[str, Any] = {
            "values": values,
            "dates": dates,
            "assets": assets,
            "metadata": metadata,
        }
        if extra:
            bundle.update(extra)
        return bundle

    def get_industry_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        industry_classification: str = "default",
    ) -> Dict[str, Any]:
        df = self._read_panel(
            _INDUSTRY_DATASET,
            ["TradeDate", "Symbol", "IndustryCode"],
            market,
            start_date,
            end_date,
            assets,
            "industry exposure",
        )
        values, dates, assets_arr, knowledge_time = self._to_panel(df, "IndustryCode", "industry exposure")
        metadata = _build_metadata(
            market=market,
            classification_version=industry_classification,
            snapshot_ref=f"{_INDUSTRY_DATASET}@{market}",
            knowledge_time=knowledge_time,
            effective_time=str(dates[-1]) if len(dates) else "unknown",
            universe_ref=_UNIVERSE_DATASET,
        )
        return self._bundle(
            values,
            dates,
            assets_arr,
            metadata,
            extra={
                "classification": industry_classification,
                "categories": sorted(set(df["IndustryCode"].astype(str))),
            },
        )

    def get_size_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        size_metric: str = "market_cap",
    ) -> Dict[str, Any]:
        col = {
            "market_cap": "MarketCap",
            "log_market_cap": "MarketCap",
            "total_assets": "ACap",
        }.get(size_metric, "MarketCap")
        df = self._read_panel(
            _SIZE_DATASET,
            ["TradeDate", "Symbol", col],
            market,
            start_date,
            end_date,
            assets,
            "size exposure",
        )
        if size_metric == "log_market_cap":
            df = df.copy()
            df[col] = np.log(np.maximum(df[col].to_numpy(dtype=float), 1.0))
        values, dates, assets_arr, knowledge_time = self._to_panel(df, col, "size exposure")
        metadata = _build_metadata(
            market=market,
            classification_version="size",
            snapshot_ref=f"{_SIZE_DATASET}@{market}",
            knowledge_time=knowledge_time,
            effective_time=str(dates[-1]) if len(dates) else "unknown",
            universe_ref=_UNIVERSE_DATASET,
        )
        return self._bundle(
            values,
            dates,
            assets_arr,
            metadata,
            extra={"metric": size_metric},
        )

    def get_sector_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        sector_classification: str = "default",
    ) -> Dict[str, Any]:
        df = self._read_panel(
            _SECTOR_DATASET,
            ["TradeDate", "Symbol", "IndustryName"],
            market,
            start_date,
            end_date,
            assets,
            "sector exposure",
        )
        values, dates, assets_arr, knowledge_time = self._to_panel(df, "IndustryName", "sector exposure")
        metadata = _build_metadata(
            market=market,
            classification_version=sector_classification,
            snapshot_ref=f"{_SECTOR_DATASET}@{market}",
            knowledge_time=knowledge_time,
            effective_time=str(dates[-1]) if len(dates) else "unknown",
            universe_ref=_UNIVERSE_DATASET,
        )
        return self._bundle(
            values,
            dates,
            assets_arr,
            metadata,
            extra={
                "classification": sector_classification,
                "categories": sorted(set(df["IndustryName"].astype(str))),
            },
        )

    def get_beta_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        window_days: int = 252,
    ) -> Dict[str, Any]:
        df = self._read_panel(
            _BETA_DATASET,
            ["TradeDate", "Symbol", "Return"],
            market,
            start_date,
            end_date,
            assets,
            "beta exposure",
        )
        # Placeholder beta: single-column panel of returns reshaped to 1 asset
        # column. Real beta estimation is out of scope for the adapter boundary;
        # the bundle still carries full provenance.
        values, dates, assets_arr, knowledge_time = self._to_panel(df, "Return", "beta exposure")
        metadata = _build_metadata(
            market=market,
            classification_version="beta",
            snapshot_ref=f"{_BETA_DATASET}@{market}",
            knowledge_time=knowledge_time,
            effective_time=str(dates[-1]) if len(dates) else "unknown",
            universe_ref=_UNIVERSE_DATASET,
        )
        return self._bundle(
            values,
            dates,
            assets_arr,
            metadata,
            extra={"window_days": window_days},
        )

    def get_custom_exposure(
        self,
        exposure_name: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Custom exposures are not natively resolvable; fail closed rather than
        # returning garbage.
        raise OptionalDependencyMissing(
            package_name="data_access",
            feature_name=f"custom exposure '{exposure_name}'",
        )


__all__ = ["DefaultExposureProvider"]
