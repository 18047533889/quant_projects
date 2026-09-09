"""Adapter for a precomputed Barra-lite B/F/D risk package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..core.contracts import InputBundle, OptimizationContext
from .input_adapter import InputAdapter


@dataclass(slots=True)
class BarraPrecomputedAdapter(InputAdapter):
    """Combine a Barra-lite package with strategy-owned portfolio inputs.

    The risk package supplies exposure B, factor covariance F and specific
    variance D. Alpha, benchmark, holdings and tradability remain strategy
    inputs and are deliberately not inferred by this adapter.
    """

    risk_root: str | Path
    alpha_df: pd.DataFrame
    benchmark_df: pd.DataFrame
    prev_positions_df: pd.DataFrame
    tradable_df: pd.DataFrame
    linear_cost_bps_df: Optional[pd.DataFrame] = None
    impact_cost_df: Optional[pd.DataFrame] = None
    alpha_input_type: str = "score"
    alpha_horizon_days: int = 1
    calendar_name: str = "XSHG"
    risk_data_lag_periods: int = 0
    exposure_data_lag_periods: int = 0
    annualization_factor: float = 252.0
    load_factor_returns: bool = False
    default_linear_cost_bps: float = 5.0
    minimum_annual_specific_volatility: float = 0.05
    _bundle_cache: Optional[InputBundle] = field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        index = pd.DatetimeIndex(normalized.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        normalized.index = index.normalize()
        return normalized.sort_index()

    @staticmethod
    def _stack_non_missing(frame: pd.DataFrame) -> pd.Series:
        """Stack without pandas' legacy-stack deprecation warning."""
        try:
            return frame.stack(future_stack=True).dropna()
        except TypeError:
            # pandas < 2.1 does not expose future_stack.
            return frame.stack(dropna=True)

    def _strategy_frames(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        alpha = self._normalize_frame(self.alpha_df)
        if alpha.empty or len(alpha.columns) == 0:
            raise ValueError("alpha_df must not be empty")
        dates = alpha.index
        assets = list(alpha.columns)

        def align(frame: pd.DataFrame, name: str) -> pd.DataFrame:
            aligned = self._normalize_frame(frame).reindex(
                index=dates, columns=assets
            )
            if aligned.isna().any().any():
                missing = int(aligned.isna().sum().sum())
                raise ValueError(f"{name} has {missing} missing aligned values")
            return aligned

        benchmark = align(self.benchmark_df, "benchmark_df")
        raw_previous = self._normalize_frame(self.prev_positions_df).reindex(
            columns=assets
        )
        eligible_previous = raw_previous.index[raw_previous.index <= dates[0]]
        if len(eligible_previous) == 0:
            raise ValueError(
                "prev_positions_df must contain an initial/as-of row on or "
                "before the first alpha date"
            )
        initial_previous = raw_previous.loc[eligible_previous[-1]]
        if initial_previous.isna().any():
            missing = int(initial_previous.isna().sum())
            raise ValueError(
                f"prev_positions_df has {missing} missing initial values"
            )
        previous = initial_previous.to_frame().T
        previous.index = pd.DatetimeIndex([dates[0]])

        raw_tradable = align(self.tradable_df, "tradable_df")
        if all(pd.api.types.is_bool_dtype(dtype) for dtype in raw_tradable.dtypes):
            tradable = raw_tradable.astype(bool)
        else:
            numeric = raw_tradable.apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any() or not numeric.isin([0, 1]).all().all():
                raise ValueError(
                    "tradable_df must contain boolean or numeric 0/1 values"
                )
            tradable = numeric.astype(bool)
        return alpha, benchmark, previous, tradable

    @staticmethod
    def _read_filtered(
        path: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
        assets: list[str] | None = None,
    ) -> pd.DataFrame:
        filters: list[tuple[str, str, object]] = [
            ("date", ">=", start.strftime("%Y-%m-%d")),
            ("date", "<=", end.strftime("%Y-%m-%d")),
        ]
        if assets is not None:
            filters.append(("asset", "in", assets))
        try:
            frame = pd.read_parquet(path, filters=filters)
        except Exception:
            # Some older parquet writers store date as a string and cannot
            # compare it with Timestamp predicates. Fall back to a full scan.
            frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame[frame["date"].between(start, end)]
        if assets is not None:
            frame = frame[frame["asset"].isin(assets)]
        return frame

    @staticmethod
    def _require_asof_coverage(
        available_dates: pd.DatetimeIndex,
        decision_dates: pd.DatetimeIndex,
        lag: int,
        name: str,
    ) -> None:
        if lag < 0:
            raise ValueError(f"{name} lag periods must be non-negative")
        available = pd.DatetimeIndex(available_dates).unique().sort_values()
        for decision_date in decision_dates:
            eligible = available[available <= decision_date]
            if len(eligible) <= lag:
                raise ValueError(
                    f"{name} has no as-of snapshot for {decision_date} "
                    f"with lag={lag}"
                )

    @staticmethod
    def _asof_date(
        available_dates: pd.DatetimeIndex,
        decision_date: pd.Timestamp,
        lag: int,
        name: str,
    ) -> pd.Timestamp:
        available = pd.DatetimeIndex(available_dates).unique().sort_values()
        eligible = available[available <= decision_date]
        if len(eligible) <= lag:
            raise ValueError(
                f"{name} has no as-of snapshot for {decision_date} with lag={lag}"
            )
        return pd.Timestamp(eligible[-(lag + 1)])

    def _load_manifest(self) -> dict:
        path = Path(self.risk_root) / "manifest.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing Barra manifest: {path}")
        with path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if manifest.get("product") != "barra_lite":
            raise ValueError("manifest product must be barra_lite")
        return manifest

    def build_bundle(self, *args, **kwargs) -> InputBundle:
        if self._bundle_cache is not None:
            return self._bundle_cache

        alpha, benchmark, previous, tradable = self._strategy_frames()
        dates = alpha.index
        assets = list(alpha.columns)
        root = Path(self.risk_root)
        manifest = self._load_manifest()
        estimation = dict(manifest.get("estimation", {}))
        risk_units = str(estimation.get("units", ""))
        allowed_units = {
            "daily_variance",
            "horizon_variance",
            "annual_variance",
        }
        if risk_units not in allowed_units:
            raise ValueError(
                "manifest estimation.units must be one of "
                f"{sorted(allowed_units)}, got {risk_units!r}"
            )
        risk_horizon_days = int(
            estimation.get(
                "horizon_days",
                1 if risk_units == "daily_variance" else self.alpha_horizon_days,
            )
        )
        if risk_horizon_days <= 0 or self.annualization_factor <= 0:
            raise ValueError("risk horizon and annualization factor must be positive")
        factor_specs = list(manifest.get("factors", []))
        factors = [str(item["id"]) for item in factor_specs]
        style_factors = [
            str(item["id"]) for item in factor_specs if item.get("type") == "continuous"
        ]
        industry_factors = [
            str(item["id"]) for item in factor_specs if item.get("type") == "dummy"
        ]
        if not factors or not style_factors or not industry_factors:
            raise ValueError("manifest must contain continuous and dummy factors")

        covariance_path = root / manifest["files"]["factor_cov"]
        covariance_dates_raw = pd.read_parquet(
            covariance_path, columns=["date"]
        )
        available_risk_dates = pd.DatetimeIndex(
            pd.to_datetime(covariance_dates_raw["date"]).dt.normalize().unique()
        ).sort_values()
        max_lag = max(self.risk_data_lag_periods, self.exposure_data_lag_periods)
        eligible_start = available_risk_dates[available_risk_dates <= dates.min()]
        if len(eligible_start) <= max_lag:
            raise ValueError(
                f"Barra package has insufficient history before {dates.min()} "
                f"for lag={max_lag}"
            )
        load_start = pd.Timestamp(eligible_start[-(max_lag + 1)])
        load_end = pd.Timestamp(dates.max())

        exposure_long = self._read_filtered(
            root / manifest["files"]["exposure"], load_start, load_end, assets
        )
        exposure_dates = pd.DatetimeIndex(
            exposure_long["date"].unique()
        ).sort_values()
        self._require_asof_coverage(
            exposure_dates,
            dates,
            self.exposure_data_lag_periods,
            "exposure",
        )
        exposure_observed = exposure_long.pivot(
            index=["date", "asset"], columns="factor_id", values="exposure"
        ).reindex(columns=factors)
        # Industry dummies are stored sparsely: absent dummy cells on an
        # observed asset-date are real zeros. Entirely absent asset-dates stay
        # NaN and must never be turned into synthetic Barra observations.
        exposure_observed[industry_factors] = exposure_observed[
            industry_factors
        ].fillna(0.0)
        style_values = exposure_observed[style_factors].to_numpy(dtype=float)
        industry_values = exposure_observed[industry_factors].to_numpy(dtype=float)
        complete_exposure = pd.Series(
            np.isfinite(style_values).all(axis=1)
            & np.isfinite(industry_values).all(axis=1)
            & np.isclose(industry_values.sum(axis=1), 1.0),
            index=exposure_observed.index,
            dtype=bool,
        )
        expected_index = pd.MultiIndex.from_product(
            [exposure_dates, assets], names=["date", "asset"]
        )
        exposure = exposure_observed.reindex(index=expected_index)
        complete_exposure = complete_exposure.reindex(
            expected_index, fill_value=False
        )

        factor_cov_long = self._read_filtered(
            covariance_path, load_start, load_end
        )
        cov_frames: dict[pd.Timestamp, pd.DataFrame] = {}
        for date, group in factor_cov_long.groupby("date", sort=True):
            matrix = group.pivot(
                index="factor_i", columns="factor_j", values="cov"
            ).reindex(index=factors, columns=factors)
            if matrix.isna().any().any():
                raise ValueError(f"factor_cov is incomplete on {date}")
            cov_frames[pd.Timestamp(date)] = matrix
        self._require_asof_coverage(
            pd.DatetimeIndex(list(cov_frames)),
            dates,
            self.risk_data_lag_periods,
            "factor_cov",
        )
        factor_cov = pd.concat(
            cov_frames, names=["date", "factor_i"]
        ).sort_index()

        specific_long = self._read_filtered(
            root / manifest["files"]["specific_risk"], load_start, load_end, assets
        )
        specific_var = specific_long.pivot(
            index="date", columns="asset", values="specific_var"
        ).reindex(columns=assets)
        specific_dates = pd.DatetimeIndex(specific_var.index).sort_values()
        self._require_asof_coverage(
            specific_dates,
            dates,
            self.risk_data_lag_periods,
            "specific_var",
        )
        observed_specific = self._stack_non_missing(specific_var)
        if observed_specific.empty:
            raise ValueError("specific_var contains no observed asset-date values")
        specific_values = observed_specific.to_numpy(dtype=float)
        if not np.isfinite(specific_values).all() or (specific_values <= 0).any():
            raise ValueError(
                "observed specific_var values must be finite and positive"
            )

        risk_covered = pd.DataFrame(False, index=dates, columns=assets)
        for decision_date in dates:
            exposure_date = self._asof_date(
                exposure_dates,
                decision_date,
                self.exposure_data_lag_periods,
                "exposure",
            )
            specific_date = self._asof_date(
                specific_dates,
                decision_date,
                self.risk_data_lag_periods,
                "specific_var",
            )
            exposure_day = complete_exposure.xs(
                exposure_date, level="date"
            ).reindex(assets, fill_value=False)
            specific_day = specific_var.loc[specific_date].reindex(assets)
            specific_day_covered = (
                specific_day.notna()
                & pd.Series(
                    np.isfinite(specific_day.to_numpy(dtype=float)),
                    index=assets,
                )
                & specific_day.gt(0)
            )
            risk_covered.loc[decision_date] = (
                exposure_day.astype(bool) & specific_day_covered.astype(bool)
            ).to_numpy()
        risk_covered = risk_covered.astype(bool)

        if risk_units == "daily_variance":
            annual_specific_vol = np.sqrt(specific_var * self.annualization_factor)
        elif risk_units == "horizon_variance":
            annual_specific_vol = np.sqrt(
                specific_var * self.annualization_factor / risk_horizon_days
            )
        else:
            annual_specific_vol = np.sqrt(specific_var)
        observed_annual_specific_vol = self._stack_non_missing(
            annual_specific_vol
        )
        minimum_observed = float(observed_annual_specific_vol.min())
        if minimum_observed < self.minimum_annual_specific_volatility:
            location = observed_annual_specific_vol.idxmin()
            raise ValueError(
                "specific risk quality gate failed: "
                f"{location} annual volatility {minimum_observed:.6%} is below "
                f"{self.minimum_annual_specific_volatility:.6%}"
            )

        factor_returns = None
        factor_returns_file = manifest.get("files", {}).get("factor_returns")
        if (
            self.load_factor_returns
            and factor_returns_file
            and (root / factor_returns_file).exists()
        ):
            factor_returns_long = self._read_filtered(
                root / factor_returns_file, load_start, load_end
            )
            factor_returns = factor_returns_long.pivot(
                index="date", columns="factor_id", values="factor_return"
            ).reindex(columns=factors)

        industry = pd.DataFrame(index=exposure_dates, columns=assets, dtype=float)
        for date in exposure_dates:
            day = exposure.xs(date, level=0)[industry_factors]
            covered = complete_exposure.xs(date, level="date").reindex(
                assets, fill_value=False
            )
            labels = pd.Series(np.nan, index=assets, dtype=object)
            if covered.any():
                labels.loc[covered] = (
                    day.loc[covered]
                    .idxmax(axis=1)
                    .str.removeprefix("industry_")
                )
            industry.loc[date] = pd.to_numeric(
                labels, errors="coerce"
            ).reindex(assets)

        style_parts = {
            factor: exposure[factor].unstack("asset").reindex(
                index=exposure_dates, columns=assets
            )
            for factor in style_factors
        }
        style = pd.concat(style_parts, axis=1)

        if self.linear_cost_bps_df is None:
            linear_cost = pd.DataFrame(
                self.default_linear_cost_bps,
                index=dates,
                columns=assets,
                dtype=float,
            )
        else:
            linear_cost = self._normalize_frame(
                self.linear_cost_bps_df
            ).reindex(index=dates, columns=assets)
        impact_cost = (
            None
            if self.impact_cost_df is None
            else self._normalize_frame(self.impact_cost_df).reindex(
                index=dates, columns=assets
            )
        )

        self._bundle_cache = InputBundle(
            alpha=alpha,
            market=None,
            F=exposure,
            G=industry,
            F_mcap=None,
            F_ret=factor_returns,
            F_spec=None,
            factor_cov=factor_cov,
            specific_var=specific_var,
            industry=industry,
            style=style,
            benchmark=benchmark,
            prev_positions=previous,
            tradable=tradable,
            risk_covered=risk_covered,
            linear_cost_bps=linear_cost,
            impact_cost=impact_cost,
            metadata=OptimizationContext(
                alpha_is_absolute_return=False,
                benchmark_name="",
                calendar_name=self.calendar_name,
                decision_time="close",
                execution_time="next_open",
                market_data_lag_periods=self.risk_data_lag_periods,
                exposure_data_lag_periods=self.exposure_data_lag_periods,
                alpha_input_type=self.alpha_input_type,
                alpha_horizon_days=self.alpha_horizon_days,
                risk_covariance_units=risk_units,
                risk_horizon_days=risk_horizon_days,
                annualization_factor=self.annualization_factor,
                extras={
                    "barra_product": manifest.get("product"),
                    "barra_version": manifest.get("version"),
                    "barra_provider": manifest.get("provider"),
                    "risk_data_lag_periods": self.risk_data_lag_periods,
                    "exposure_data_lag_periods": self.exposure_data_lag_periods,
                    "risk_covered_cells": int(risk_covered.to_numpy().sum()),
                    "risk_coverage_cells": int(risk_covered.size),
                },
            ),
        )
        return self._bundle_cache

    def load_alpha(self, *args, **kwargs) -> pd.DataFrame:
        return self.build_bundle().alpha

    def load_market(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return None

    def load_industry(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return self.build_bundle().industry

    def load_style(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return self.build_bundle().style

    def load_benchmark(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return self.build_bundle().benchmark

    def load_prev_positions(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return self.build_bundle().prev_positions

    def load_tradable_flags(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        return self.build_bundle().tradable
