"""Independent P0/P1 analysis of optimizer target positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    AnalysisEnrichment,
    OptimizationArtifacts,
    PositionAnalysisResult,
    empty_constraint_summary,
    empty_exposure_summary,
    empty_quality_checks,
    empty_risk_contribution,
)
from .enrichers import (
    load_barra_enrichment,
    load_data_access_enrichment,
    load_market_amount,
    load_reference_attributes,
)


@dataclass
class _QualityCollector:
    rows: list[dict[str, Any]]
    degraded: bool = False

    def add(
        self,
        check_id: str,
        *,
        severity: str,
        status: str,
        message: str,
        date: pd.Timestamp | None = None,
        scope: str = "analysis",
        observed: Any = None,
        expected: Any = None,
        difference: Any = None,
        degraded: bool = False,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "date": pd.NaT if date is None else pd.Timestamp(date),
                "scope": scope,
                "severity": severity,
                "status": status,
                "observed": observed,
                "expected": expected,
                "difference": difference,
                "message": message,
            }
        )
        self.degraded = self.degraded or degraded

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return empty_quality_checks()
        result = pd.DataFrame(self.rows)
        result["date"] = pd.to_datetime(result["date"])
        def audit_text(value: Any) -> str | None:
            if value is None:
                return None
            try:
                if bool(pd.isna(value)):
                    return None
            except (TypeError, ValueError):
                pass
            return str(value)
        for column in ("observed", "expected"):
            result[column] = result[column].map(audit_text)
        result["difference"] = pd.to_numeric(
            result["difference"], errors="coerce"
        )
        return result


def _merge_enrichment(
    left: AnalysisEnrichment, right: AnalysisEnrichment
) -> AnalysisEnrichment:
    for field_name in (
        "benchmark",
        "tradable",
        "market_amount",
        "industry",
        "market_cap",
        "factor_exposure",
        "factor_cov",
        "specific_var",
    ):
        value = getattr(right, field_name)
        if value is not None:
            setattr(left, field_name, value)
    if right.factor_specs:
        left.factor_specs = list(right.factor_specs)
    left.provenance.update(right.provenance)
    return left


def _nearest_psd(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, floor)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_localize(None).normalize()


def _eligible_date(
    available: pd.DatetimeIndex, decision_date: pd.Timestamp, lag: int
) -> pd.Timestamp:
    eligible = available[available <= decision_date]
    if len(eligible) <= lag:
        raise ValueError(
            f"no point-in-time value on {decision_date.date()} with lag={lag}"
        )
    return pd.Timestamp(eligible[-(lag + 1)])


class PositionAnalyzer:
    """Analyze optimizer artifacts without reusing optimizer diagnostics code."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        data_store: Any | None = None,
    ) -> None:
        self.config = config
        self.data_store = data_store

    def _p0(
        self,
        artifacts: OptimizationArtifacts,
        quality: _QualityCollector,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        weights = artifacts.target_positions
        trade_matrix = (
            artifacts.trades["delta_weight"]
            .unstack("asset")
            .reindex(index=weights.index, columns=weights.columns)
        )
        previous = weights.shift(1)
        previous.loc[weights.index[0]] = weights.iloc[0] - trade_matrix.iloc[0]
        epsilon = float(self.config["position"]["weight_epsilon"])

        gross = weights.abs().sum(axis=1)
        net = weights.sum(axis=1)
        params = artifacts.resolved_params
        budget = float(params.get("budget", 1.0))
        summary = pd.DataFrame(index=weights.index)
        summary["net_exposure"] = net
        summary["gross_exposure"] = gross
        summary["unallocated_weight"] = budget - net
        summary["long_count"] = weights.gt(epsilon).sum(axis=1)
        summary["short_count"] = weights.lt(-epsilon).sum(axis=1)
        summary["zero_count"] = (
            len(weights.columns) - summary["long_count"] - summary["short_count"]
        )
        summary["max_weight"] = weights.max(axis=1)
        summary["min_weight"] = weights.min(axis=1)
        summary["max_abs_weight"] = weights.abs().max(axis=1)

        normalized_abs = weights.abs().div(gross.replace(0.0, np.nan), axis=0)
        summary["hhi"] = normalized_abs.pow(2).sum(axis=1, min_count=1)
        summary["effective_n"] = 1.0 / summary["hhi"]
        for topk in self.config["position"]["topk"]:
            summary[f"top{topk}_abs_weight_share"] = normalized_abs.apply(
                lambda row, k=topk: row.nlargest(min(k, row.notna().sum())).sum()
                if row.notna().any()
                else np.nan,
                axis=1,
            )

        absolute_trade = trade_matrix.abs()
        summary["gross_traded_weight"] = absolute_trade.sum(axis=1)
        summary["one_way_turnover"] = summary["gross_traded_weight"] / 2.0
        summary["buy_weight"] = trade_matrix.clip(lower=0.0).sum(axis=1)
        summary["sell_weight"] = -trade_matrix.clip(upper=0.0).sum(axis=1)

        held = weights.abs().gt(epsilon)
        previous_held = previous.abs().gt(epsilon)
        summary["entry_count"] = (held & ~previous_held).sum(axis=1)
        summary["exit_count"] = (~held & previous_held).sum(axis=1)
        overlaps: list[float] = []
        cosines: list[float] = []
        for date in weights.index:
            current_set = set(weights.columns[held.loc[date]])
            prior_set = set(weights.columns[previous_held.loc[date]])
            union = current_set | prior_set
            overlaps.append(
                float(len(current_set & prior_set) / len(union))
                if union
                else np.nan
            )
            current = weights.loc[date].to_numpy(dtype=float)
            prior = previous.loc[date].to_numpy(dtype=float)
            denominator = float(np.linalg.norm(current) * np.linalg.norm(prior))
            cosines.append(
                float(current @ prior / denominator)
                if denominator > epsilon
                else np.nan
            )
        summary["holding_overlap"] = overlaps
        summary["weight_cosine_similarity"] = cosines
        summary["active_share"] = np.nan
        summary["systematic_variance"] = np.nan
        summary["specific_variance"] = np.nan
        summary["total_variance"] = np.nan
        summary["predicted_volatility"] = np.nan
        summary["predicted_tracking_error"] = np.nan

        holdings = pd.DataFrame(
            {
                "weight": weights.stack(future_stack=True),
                "previous_weight": previous.stack(future_stack=True),
                "delta_weight": trade_matrix.stack(future_stack=True),
                "abs_weight": weights.abs().stack(future_stack=True),
            }
        )
        holdings.index.names = ["date", "asset"]
        holdings["weight_rank"] = (
            holdings.groupby(level="date")["weight"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        holdings["abs_weight_rank"] = (
            holdings.groupby(level="date")["abs_weight"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        holdings["benchmark_weight"] = np.nan
        holdings["active_weight"] = np.nan
        holdings["is_long"] = holdings["weight"].gt(epsilon)
        holdings["is_short"] = holdings["weight"].lt(-epsilon)
        holdings["is_entry"] = (
            holdings["previous_weight"].abs().le(epsilon)
            & holdings["weight"].abs().gt(epsilon)
        )
        holdings["is_exit"] = (
            holdings["previous_weight"].abs().gt(epsilon)
            & holdings["weight"].abs().le(epsilon)
        )
        holdings["is_tradable"] = pd.Series(
            pd.NA, index=holdings.index, dtype="boolean"
        )
        holdings["industry"] = pd.NA
        holdings["market_cap"] = np.nan
        holdings["asset_variance_contribution"] = np.nan
        holdings["asset_risk_contribution_pct"] = np.nan

        turnover = holdings[
            [
                "previous_weight",
                "weight",
                "delta_weight",
                "is_entry",
                "is_exit",
                "is_tradable",
            ]
        ].copy()
        turnover["side"] = np.select(
            [
                turnover["delta_weight"].gt(epsilon),
                turnover["delta_weight"].lt(-epsilon),
            ],
            ["buy", "sell"],
            default="flat",
        )
        turnover["trade_notional"] = np.nan
        turnover["adv"] = np.nan
        turnover["adv_participation"] = np.nan
        turnover["estimated_days"] = np.nan

        tolerance = float(self.config["validation"]["comparison_tolerance"])
        comparisons = {
            "gross_exposure": "gross_exposure",
            "net_exposure": "net_exposure",
            "turnover": "one_way_turnover",
        }
        for reported_name, recomputed_name in comparisons.items():
            if reported_name not in artifacts.summary:
                quality.add(
                    f"reported_{reported_name}_available",
                    severity="info",
                    status="unavailable",
                    scope="optimizer_summary",
                    message=f"summary does not contain {reported_name}",
                )
                continue
            reported = pd.to_numeric(
                artifacts.summary[reported_name], errors="coerce"
            )
            difference = reported - summary[recomputed_name]
            for date, value in difference.items():
                passed = bool(np.isfinite(value) and abs(value) <= tolerance)
                quality.add(
                    f"{reported_name}_recomputed_match",
                    severity="info" if passed else "warning",
                    status="passed" if passed else "failed",
                    date=date,
                    scope="optimizer_summary",
                    observed=float(reported.loc[date]),
                    expected=float(summary.loc[date, recomputed_name]),
                    difference=float(value),
                    message=(
                        f"reported {reported_name} matches independent calculation"
                        if passed
                        else f"reported {reported_name} differs from independent calculation"
                    ),
                    degraded=not passed,
                )
            summary[f"reported_{reported_name}"] = reported
            summary[f"{reported_name}_difference"] = difference

        zero_gross = gross.le(epsilon)
        for date in weights.index[zero_gross]:
            quality.add(
                "nonzero_gross_exposure",
                severity="warning",
                status="failed",
                date=date,
                scope="position",
                observed=float(gross.loc[date]),
                expected=f">{epsilon}",
                message="portfolio has effectively zero gross exposure",
                degraded=True,
            )
        initial_provenance = (
            artifacts.run_manifest.get("inputs", {}).get("initial_positions", {})
            or {}
        )
        if initial_provenance.get("assumption") == "backtest_starts_from_cash":
            maximum_initial = float(previous.iloc[0].abs().max())
            passed = maximum_initial <= float(
                self.config["validation"]["comparison_tolerance"]
            )
            quality.add(
                "initial_cash_position_match",
                severity="info" if passed else "warning",
                status="passed" if passed else "failed",
                date=weights.index[0],
                scope="initial_position",
                observed=maximum_initial,
                expected="zero initial weights",
                difference=maximum_initial,
                message=(
                    "reconstructed initial position matches cash-start assumption"
                    if passed
                    else "reconstructed initial position contradicts cash-start assumption"
                ),
                degraded=not passed,
            )
        return summary, holdings, turnover, previous

    def _load_p1(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        quality: _QualityCollector,
    ) -> AnalysisEnrichment:
        manifest_inputs = dict(artifacts.run_manifest.get("inputs", {}))
        benchmark_mode = self.config["benchmark"]["mode"]
        barra_mode = self.config["barra"]["mode"]
        liquidity_mode = self.config["liquidity"]["mode"]
        benchmark_declared = bool(manifest_inputs.get("benchmark"))
        tradable_declared = bool(manifest_inputs.get("tradable"))
        barra_declared = bool(manifest_inputs.get("barra_risk"))

        need_benchmark = benchmark_mode == "required" or (
            benchmark_mode == "auto" and benchmark_declared
        )
        need_data_access = (
            enrichment.benchmark is None
            and (
                need_benchmark
                or (
                    tradable_declared
                    and (
                        benchmark_mode != "off"
                        or liquidity_mode != "off"
                    )
                )
            )
        )
        data_access_loaded = False
        if need_data_access:
            try:
                loaded = load_data_access_enrichment(
                    artifacts,
                    include_benchmark=need_benchmark,
                    data_store=self.data_store,
                )
                mismatches: list[str] = []
                unknown: list[str] = []
                for name in ("benchmark", "tradable"):
                    entry = loaded.provenance.get(name)
                    if not entry:
                        continue
                    match = entry.get("snapshot_match")
                    if match is False:
                        mismatches.append(name)
                    elif match is None:
                        unknown.append(name)
                policy = self.config["validation"]["provenance_policy"]
                if mismatches and policy == "strict":
                    raise ValueError(
                        "data_access snapshot mismatch for "
                        + ", ".join(mismatches)
                    )
                if mismatches:
                    quality.add(
                        "data_access_snapshot_match",
                        severity="warning",
                        status="failed",
                        scope="provenance",
                        observed=mismatches,
                        expected="optimizer snapshot",
                        message="current data_access snapshot differs from optimization",
                        degraded=True,
                    )
                elif unknown:
                    quality.add(
                        "data_access_snapshot_verifiable",
                        severity="warning",
                        status="unavailable",
                        scope="provenance",
                        observed=unknown,
                        expected="recorded snapshot_id",
                        message="snapshot identity is missing and cannot be verified",
                        degraded=True,
                    )
                else:
                    quality.add(
                        "data_access_snapshot_match",
                        severity="info",
                        status="passed",
                        scope="provenance",
                        message="data_access snapshots match optimizer provenance",
                    )
                enrichment = _merge_enrichment(enrichment, loaded)
                data_access_loaded = True
            except Exception as exc:
                if benchmark_mode == "required":
                    raise
                quality.add(
                    "benchmark_enrichment",
                    severity="warning",
                    status="unavailable",
                    scope="enrichment",
                    message=f"benchmark/tradable enrichment failed: {exc}",
                    degraded=True,
                )

        if data_access_loaded and need_benchmark and (
            enrichment.industry is None or enrichment.market_cap is None
        ):
            try:
                reference = load_reference_attributes(
                    artifacts, data_store=self.data_store
                )
                enrichment = _merge_enrichment(enrichment, reference)
            except Exception as exc:
                quality.add(
                    "reference_attribute_enrichment",
                    severity="info",
                    status="unavailable",
                    scope="enrichment",
                    message=f"industry/market-cap enrichment unavailable: {exc}",
                )

        need_barra = barra_mode == "required" or (
            barra_mode == "auto" and barra_declared
        )
        if need_barra and enrichment.factor_exposure is None:
            try:
                loaded = load_barra_enrichment(
                    artifacts,
                    benchmark=enrichment.benchmark,
                    tradable=enrichment.tradable,
                )
                content_match = loaded.provenance["barra"].get("content_match")
                if content_match is False:
                    if self.config["validation"]["provenance_policy"] == "strict":
                        raise ValueError("Barra package content hash mismatch")
                    quality.add(
                        "barra_content_match",
                        severity="warning",
                        status="failed",
                        scope="provenance",
                        message="Barra package content differs from optimizer manifest",
                        degraded=True,
                    )
                elif content_match is None:
                    quality.add(
                        "barra_content_verifiable",
                        severity="info",
                        status="unavailable",
                        scope="provenance",
                        message=(
                            "optimizer manifest records only the Barra path; "
                            "current package hash was captured for this analysis"
                        ),
                    )
                else:
                    quality.add(
                        "barra_content_match",
                        severity="info",
                        status="passed",
                        scope="provenance",
                        message="Barra content hash matches optimizer manifest",
                    )
                enrichment = _merge_enrichment(enrichment, loaded)
            except Exception as exc:
                if barra_mode == "required":
                    raise
                quality.add(
                    "barra_enrichment",
                    severity="warning",
                    status="unavailable",
                    scope="enrichment",
                    message=f"Barra enrichment failed: {exc}",
                    degraded=True,
                )

        need_liquidity = liquidity_mode in {"auto", "required"}
        notional = self.config["liquidity"]["portfolio_notional"]
        if need_liquidity and notional is not None and enrichment.market_amount is None:
            try:
                amount, provenance = load_market_amount(
                    artifacts,
                    lookback_days=int(
                        self.config["liquidity"]["adv_window_days"]
                    ),
                    data_store=self.data_store,
                )
                enrichment.market_amount = amount
                enrichment.provenance["liquidity"] = provenance
            except Exception as exc:
                if liquidity_mode == "required":
                    raise
                quality.add(
                    "liquidity_enrichment",
                    severity="warning",
                    status="unavailable",
                    scope="enrichment",
                    message=f"liquidity enrichment failed: {exc}",
                    degraded=True,
                )
        return enrichment

    def _apply_benchmark(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        summary: pd.DataFrame,
        holdings: pd.DataFrame,
        quality: _QualityCollector,
    ) -> None:
        if enrichment.benchmark is None:
            return
        weights = artifacts.target_positions
        benchmark = enrichment.benchmark.reindex(
            index=weights.index, columns=weights.columns
        )
        if benchmark.isna().any().any():
            raise ValueError("benchmark enrichment has missing aligned weights")
        active = weights - benchmark
        summary["active_share"] = 0.5 * active.abs().sum(axis=1)
        holdings["benchmark_weight"] = benchmark.stack(future_stack=True)
        holdings["active_weight"] = active.stack(future_stack=True)

        reported = artifacts.summary.get("active_share")
        if reported is not None:
            tolerance = float(self.config["validation"]["comparison_tolerance"])
            difference = pd.to_numeric(reported, errors="coerce") - summary["active_share"]
            summary["reported_active_share"] = reported
            summary["active_share_difference"] = difference
            for date, value in difference.items():
                passed = bool(np.isfinite(value) and abs(value) <= tolerance)
                quality.add(
                    "active_share_recomputed_match",
                    severity="info" if passed else "warning",
                    status="passed" if passed else "failed",
                    date=date,
                    scope="benchmark",
                    observed=float(reported.loc[date]),
                    expected=float(summary.loc[date, "active_share"]),
                    difference=float(value),
                    message=(
                        "reported active share matches independent calculation"
                        if passed
                        else "reported active share differs from independent calculation"
                    ),
                    degraded=not passed,
                )

        provenance = enrichment.provenance.get("benchmark", {})
        coverage = provenance.get("weight_coverage_by_date", {})
        for date in weights.index:
            key = date.date().isoformat()
            value = coverage.get(key)
            if value is not None:
                summary.loc[date, "benchmark_weight_coverage"] = float(value)
        initial_provenance = (
            artifacts.run_manifest.get("inputs", {}).get("initial_positions", {})
            or {}
        )
        if initial_provenance.get("assumption") == "backtest_starts_from_benchmark":
            trade_matrix = (
                artifacts.trades["delta_weight"]
                .unstack("asset")
                .reindex(index=weights.index, columns=weights.columns)
            )
            initial = weights.iloc[0] - trade_matrix.iloc[0]
            expected = benchmark.iloc[0]
            maximum_error = float((initial - expected).abs().max())
            tolerance = float(self.config["validation"]["comparison_tolerance"])
            passed = maximum_error <= tolerance
            quality.add(
                "initial_benchmark_position_match",
                severity="info" if passed else "warning",
                status="passed" if passed else "failed",
                date=weights.index[0],
                scope="initial_position",
                observed=maximum_error,
                expected=f"<={tolerance}",
                difference=maximum_error,
                message=(
                    "reconstructed initial position matches benchmark-start assumption"
                    if passed
                    else "reconstructed initial position contradicts benchmark-start assumption"
                ),
                degraded=not passed,
            )

    def _apply_reference_attributes(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        holdings: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = artifacts.target_positions
        rows: list[dict[str, Any]] = []
        benchmark = enrichment.benchmark
        if enrichment.industry is not None:
            industry = enrichment.industry.reindex(
                index=weights.index, columns=weights.columns
            )
            holdings["industry"] = industry.stack(
                future_stack=True
            ).reindex(holdings.index)
            for date in weights.index:
                labels = industry.loc[date]
                valid = labels.notna()
                total_abs = float(weights.loc[date].abs().sum())
                for name in sorted(labels.loc[valid].astype(str).unique()):
                    mask = labels.astype(str).eq(name) & valid
                    portfolio_value = float(weights.loc[date, mask].sum())
                    benchmark_value = (
                        float(benchmark.loc[date, mask].sum())
                        if benchmark is not None
                        else np.nan
                    )
                    rows.append(
                        {
                            "date": date,
                            "exposure_type": "industry_classification",
                            "exposure_name": name,
                            "portfolio_exposure": portfolio_value,
                            "benchmark_exposure": benchmark_value,
                            "active_exposure": (
                                portfolio_value - benchmark_value
                                if benchmark is not None
                                else np.nan
                            ),
                            "asset_coverage": float(valid.mean()),
                            "weight_coverage": (
                                float(weights.loc[date, valid].abs().sum())
                                / total_abs
                                if total_abs > 0
                                else np.nan
                            ),
                            "exposure_date": date,
                            "snapshot_match": None,
                        }
                    )
        if enrichment.market_cap is not None:
            market_cap = enrichment.market_cap.reindex(
                index=weights.index, columns=weights.columns
            )
            holdings["market_cap"] = market_cap.stack(
                future_stack=True
            ).reindex(holdings.index)
            for date in weights.index:
                values = market_cap.loc[date]
                valid = values.gt(0) & np.isfinite(values)
                log_cap = np.log(values.where(valid))
                portfolio_value = float(
                    (
                        weights.loc[date, valid]
                        * log_cap.loc[valid]
                    ).sum()
                )
                benchmark_value = (
                    float(
                        (
                            benchmark.loc[date, valid]
                            * log_cap.loc[valid]
                        ).sum()
                    )
                    if benchmark is not None
                    else np.nan
                )
                total_abs = float(weights.loc[date].abs().sum())
                rows.append(
                    {
                        "date": date,
                        "exposure_type": "market_cap",
                        "exposure_name": "log_market_cap",
                        "portfolio_exposure": portfolio_value,
                        "benchmark_exposure": benchmark_value,
                        "active_exposure": (
                            portfolio_value - benchmark_value
                            if benchmark is not None
                            else np.nan
                        ),
                        "asset_coverage": float(valid.mean()),
                        "weight_coverage": (
                            float(weights.loc[date, valid].abs().sum()) / total_abs
                            if total_abs > 0
                            else np.nan
                        ),
                        "exposure_date": date,
                        "snapshot_match": None,
                    }
                )
        if not rows:
            return empty_exposure_summary()
        return pd.DataFrame(rows).set_index(
            ["date", "exposure_type", "exposure_name"]
        ).sort_index()

    def _apply_tradable(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        holdings: pd.DataFrame,
        turnover: pd.DataFrame,
    ) -> None:
        if enrichment.tradable is None:
            return
        aligned = enrichment.tradable.reindex(
            index=artifacts.target_positions.index,
            columns=artifacts.target_positions.columns,
        )
        stacked = aligned.astype(bool).stack(future_stack=True)
        holdings["is_tradable"] = stacked.astype("boolean")
        turnover["is_tradable"] = stacked.astype("boolean")

    def _apply_exposures_and_risk(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        summary: pd.DataFrame,
        holdings: pd.DataFrame,
        quality: _QualityCollector,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if (
            enrichment.factor_exposure is None
            or enrichment.factor_cov is None
            or enrichment.specific_var is None
        ):
            return empty_exposure_summary(), empty_risk_contribution()

        weights = artifacts.target_positions
        benchmark = enrichment.benchmark
        factors_by_type = {
            str(item.get("id")): str(item.get("type", "unknown"))
            for item in enrichment.factor_specs
        }
        exposure_dates = pd.DatetimeIndex(
            enrichment.factor_exposure.index.get_level_values(0).unique()
        ).sort_values()
        covariance_dates = pd.DatetimeIndex(
            enrichment.factor_cov.index.get_level_values(0).unique()
        ).sort_values()
        specific_dates = pd.DatetimeIndex(
            enrichment.specific_var.index
        ).sort_values()
        adapter_config = dict(artifacts.resolved_config.get("adapter", {}))
        exposure_lag = int(adapter_config.get("exposure_data_lag_periods", 0))
        risk_lag = int(adapter_config.get("risk_data_lag_periods", 0))
        params = artifacts.resolved_params
        factor_floor = float(
            params.get("precomputed_factor_eigenvalue_floor", 1e-12)
        )
        specific_floor = float(
            params.get("precomputed_specific_variance_floor", 1e-8)
        )
        exposure_rows: list[dict[str, Any]] = []
        risk_rows: list[dict[str, Any]] = []
        primary_asset_contribution: dict[pd.Timestamp, pd.Series] = {}
        primary_asset_pct: dict[pd.Timestamp, pd.Series] = {}

        for date in weights.index:
            diagnostic = artifacts.summary.loc[date]
            exposure_date = _as_timestamp(diagnostic.get("risk_exposure_date"))
            covariance_date = _as_timestamp(
                diagnostic.get("risk_covariance_date")
            )
            specific_date = _as_timestamp(diagnostic.get("specific_risk_date"))
            exposure_date = exposure_date or _eligible_date(
                exposure_dates, date, exposure_lag
            )
            covariance_date = covariance_date or _eligible_date(
                covariance_dates, date, risk_lag
            )
            specific_date = specific_date or _eligible_date(
                specific_dates, date, risk_lag
            )
            exposure = enrichment.factor_exposure.xs(
                exposure_date, level=0
            ).reindex(index=weights.columns)
            covariance = enrichment.factor_cov.xs(
                covariance_date, level=0
            )
            factors = [str(value) for value in covariance.columns]
            covariance = covariance.reindex(index=factors, columns=factors)
            exposure = exposure.reindex(columns=factors)
            specific = enrichment.specific_var.loc[specific_date].reindex(
                weights.columns
            )
            if (
                exposure.isna().any().any()
                or covariance.isna().any().any()
                or specific.isna().any()
            ):
                raise ValueError(f"Barra inputs are incomplete on {date.date()}")

            units = str(
                diagnostic.get(
                    "risk_covariance_units",
                    adapter_config.get("risk_covariance_units", "daily_variance"),
                )
            )
            alpha_days = float(
                diagnostic.get(
                    "alpha_horizon_days",
                    adapter_config.get("alpha_horizon_days", 1),
                )
            )
            risk_days = float(diagnostic.get("risk_horizon_days", 1))
            annualization = float(
                diagnostic.get(
                    "annualization_factor",
                    adapter_config.get("annualization_factor", 252.0),
                )
            )
            source_days = {
                "daily_variance": 1.0,
                "horizon_variance": risk_days,
                "annual_variance": annualization,
            }.get(units)
            if source_days is None or source_days <= 0 or alpha_days <= 0:
                raise ValueError(f"invalid risk units/horizon on {date.date()}")
            scale = alpha_days / source_days
            factor_covariance = _nearest_psd(
                covariance.to_numpy(dtype=float) * scale, factor_floor
            )
            specific_values = np.maximum(
                specific.to_numpy(dtype=float) * scale, specific_floor
            )
            B = exposure.to_numpy(dtype=float)
            w = weights.loc[date].to_numpy(dtype=float)
            b = (
                benchmark.loc[date].reindex(weights.columns).to_numpy(dtype=float)
                if benchmark is not None
                else np.zeros_like(w)
            )
            snapshot_match_value = enrichment.provenance.get("barra", {}).get(
                "content_match"
            )
            portfolio_factor_exposure = B.T @ w
            benchmark_factor_exposure = B.T @ b
            for position, factor in enumerate(factors):
                factor_type = factors_by_type.get(factor, "unknown")
                exposure_type = (
                    "industry" if factor_type == "dummy" else "style"
                    if factor_type == "continuous"
                    else factor_type
                )
                exposure_rows.append(
                    {
                        "date": date,
                        "exposure_type": exposure_type,
                        "exposure_name": factor,
                        "portfolio_exposure": portfolio_factor_exposure[position],
                        "benchmark_exposure": benchmark_factor_exposure[position],
                        "active_exposure": (
                            portfolio_factor_exposure[position]
                            - benchmark_factor_exposure[position]
                        ),
                        "asset_coverage": 1.0,
                        "weight_coverage": 1.0,
                        "exposure_date": exposure_date,
                        "snapshot_match": snapshot_match_value,
                    }
                )

            industry_factors = [
                factor
                for factor in factors
                if factors_by_type.get(factor) == "dummy"
            ]
            if industry_factors:
                labels = exposure[industry_factors].idxmax(axis=1)
                industry_sum = exposure[industry_factors].sum(axis=1)
                labels = labels.where(np.isclose(industry_sum, 1.0), pd.NA)
                holdings.loc[
                    pd.IndexSlice[date, :], "industry"
                ] = labels.reindex(weights.columns).to_numpy()

            bases: list[tuple[str, np.ndarray]] = []
            if bool(self.config["barra"]["analyze_absolute_risk"]):
                bases.append(("absolute", w))
            if (
                bool(self.config["barra"]["analyze_active_risk"])
                and benchmark is not None
            ):
                bases.append(("active", w - b))
            for basis, vector in bases:
                factor_vector = B.T @ vector
                factor_marginal = factor_covariance @ factor_vector
                systematic_variance = float(factor_vector @ factor_marginal)
                specific_contributions = vector * vector * specific_values
                specific_variance = float(specific_contributions.sum())
                marginal = B @ factor_marginal + specific_values * vector
                asset_contributions = vector * marginal
                total_variance = max(
                    0.0, systematic_variance + specific_variance
                )
                volatility = float(np.sqrt(total_variance))
                percentage_denominator = (
                    total_variance if total_variance > 1e-20 else np.nan
                )
                common = {
                    "date": date,
                    "risk_basis": basis,
                    "risk_horizon_days": alpha_days,
                    "annualization_factor": annualization,
                    "exposure_date": exposure_date,
                    "covariance_date": covariance_date,
                    "specific_risk_date": specific_date,
                }
                risk_rows.append(
                    {
                        **common,
                        "component_type": "total",
                        "component_name": "portfolio",
                        "variance_contribution": total_variance,
                        "variance_contribution_pct": 1.0
                        if total_variance > 1e-20
                        else np.nan,
                        "marginal_risk": np.nan,
                        "variance": total_variance,
                        "volatility": volatility,
                    }
                )
                factor_contributions = factor_vector * factor_marginal
                for factor, contribution, marginal_value in zip(
                    factors, factor_contributions, factor_marginal
                ):
                    risk_rows.append(
                        {
                            **common,
                            "component_type": "factor",
                            "component_name": factor,
                            "variance_contribution": float(contribution),
                            "variance_contribution_pct": float(
                                contribution / percentage_denominator
                            ),
                            "marginal_risk": float(marginal_value),
                            "variance": np.nan,
                            "volatility": np.nan,
                        }
                    )
                for asset, contribution, marginal_value in zip(
                    weights.columns, asset_contributions, marginal
                ):
                    risk_rows.append(
                        {
                            **common,
                            "component_type": "asset",
                            "component_name": asset,
                            "variance_contribution": float(contribution),
                            "variance_contribution_pct": float(
                                contribution / percentage_denominator
                            ),
                            "marginal_risk": float(marginal_value),
                            "variance": np.nan,
                            "volatility": np.nan,
                        }
                    )
                for asset, contribution in zip(
                    weights.columns, specific_contributions
                ):
                    risk_rows.append(
                        {
                            **common,
                            "component_type": "specific",
                            "component_name": asset,
                            "variance_contribution": float(contribution),
                            "variance_contribution_pct": float(
                                contribution / percentage_denominator
                            ),
                            "marginal_risk": np.nan,
                            "variance": np.nan,
                            "volatility": np.nan,
                        }
                    )

                contribution_error = float(
                    asset_contributions.sum() - total_variance
                )
                tolerance = float(
                    self.config["validation"]["comparison_tolerance"]
                )
                passed = abs(contribution_error) <= tolerance
                quality.add(
                    f"barra_{basis}_asset_contribution_sum",
                    severity="info" if passed else "error",
                    status="passed" if passed else "failed",
                    date=date,
                    scope="barra_risk",
                    observed=float(asset_contributions.sum()),
                    expected=total_variance,
                    difference=contribution_error,
                    message=(
                        "asset risk contributions sum to total variance"
                        if passed
                        else "asset risk contributions do not sum to total variance"
                    ),
                    degraded=not passed,
                )

                primary = (
                    basis == "active"
                    if benchmark is not None
                    else basis == "absolute"
                )
                if primary:
                    summary.loc[date, "systematic_variance"] = systematic_variance
                    summary.loc[date, "specific_variance"] = specific_variance
                    summary.loc[date, "total_variance"] = total_variance
                    summary.loc[date, "predicted_volatility"] = volatility
                    if basis == "active":
                        summary.loc[date, "predicted_tracking_error"] = volatility
                    primary_asset_contribution[date] = pd.Series(
                        asset_contributions, index=weights.columns
                    )
                    primary_asset_pct[date] = pd.Series(
                        asset_contributions / percentage_denominator,
                        index=weights.columns,
                    )

            absolute_total = next(
                (
                    row["variance"]
                    for row in reversed(risk_rows)
                    if row["date"] == date
                    and row["risk_basis"] == "absolute"
                    and row["component_type"] == "total"
                ),
                None,
            )
            active_total = next(
                (
                    row["variance"]
                    for row in reversed(risk_rows)
                    if row["date"] == date
                    and row["risk_basis"] == "active"
                    and row["component_type"] == "total"
                ),
                None,
            )
            comparisons = [
                (
                    "predicted_portfolio_volatility",
                    np.sqrt(absolute_total) if absolute_total is not None else None,
                ),
                (
                    "predicted_tracking_error_horizon",
                    np.sqrt(active_total) if active_total is not None else None,
                ),
            ]
            for reported_name, recomputed in comparisons:
                if recomputed is None or reported_name not in artifacts.summary:
                    continue
                reported = float(artifacts.summary.loc[date, reported_name])
                difference = reported - float(recomputed)
                passed = (
                    np.isfinite(difference)
                    and abs(difference)
                    <= float(self.config["validation"]["comparison_tolerance"])
                )
                quality.add(
                    f"{reported_name}_recomputed_match",
                    severity="info" if passed else "warning",
                    status="passed" if passed else "failed",
                    date=date,
                    scope="barra_risk",
                    observed=reported,
                    expected=float(recomputed),
                    difference=difference,
                    message=(
                        f"{reported_name} matches Barra reconstruction"
                        if passed
                        else f"{reported_name} differs from Barra reconstruction"
                    ),
                    degraded=not passed,
                )

        for date, values in primary_asset_contribution.items():
            holdings.loc[
                pd.IndexSlice[date, :], "asset_variance_contribution"
            ] = values.reindex(weights.columns).to_numpy()
            holdings.loc[
                pd.IndexSlice[date, :], "asset_risk_contribution_pct"
            ] = primary_asset_pct[date].reindex(weights.columns).to_numpy()

        exposure_result = pd.DataFrame(exposure_rows).set_index(
            ["date", "exposure_type", "exposure_name"]
        ).sort_index()
        risk_result = pd.DataFrame(risk_rows).set_index(
            ["date", "risk_basis", "component_type", "component_name"]
        ).sort_index()
        return exposure_result, risk_result

    def _apply_liquidity(
        self,
        artifacts: OptimizationArtifacts,
        enrichment: AnalysisEnrichment,
        turnover: pd.DataFrame,
        quality: _QualityCollector,
    ) -> None:
        amount = enrichment.market_amount
        notional = self.config["liquidity"]["portfolio_notional"]
        if amount is None or notional is None:
            return
        notional = float(notional)
        window = int(self.config["liquidity"]["adv_window_days"])
        maximum_participation = float(
            self.config["liquidity"]["maximum_adv_participation"]
        )
        weights = artifacts.target_positions
        adv = pd.DataFrame(index=weights.index, columns=weights.columns, dtype=float)
        for date in weights.index:
            history = amount.loc[amount.index < date].tail(window)
            positive = history.where(history > 0)
            adv.loc[date] = positive.mean(axis=0, skipna=True)
            coverage = float(adv.loc[date].notna().mean())
            quality.add(
                "liquidity_adv_coverage",
                severity="info" if coverage >= 0.95 else "warning",
                status="passed" if coverage >= 0.95 else "failed",
                date=date,
                scope="liquidity",
                observed=coverage,
                expected=">=0.95",
                message="ADV coverage across the optimization universe",
                degraded=coverage < 0.95,
            )
        delta = artifacts.trades["delta_weight"].reindex(turnover.index)
        trade_notional = delta.abs() * notional
        adv_long = adv.stack(future_stack=True).reindex(turnover.index)
        turnover["trade_notional"] = trade_notional
        turnover["adv"] = adv_long
        turnover["adv_participation"] = trade_notional / adv_long
        turnover["estimated_days"] = (
            trade_notional / (adv_long * maximum_participation)
        )

    def _build_constraints(
        self,
        artifacts: OptimizationArtifacts,
        summary: pd.DataFrame,
        holdings: pd.DataFrame,
        exposure_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        params = artifacts.resolved_params
        if not params and not any(
            str(column).endswith(("_violation", "_slack"))
            for column in artifacts.summary.columns
        ):
            return empty_constraint_summary()
        rows: list[dict[str, Any]] = []
        binding_tolerance = float(
            self.config["validation"]["binding_tolerance"]
        )
        epsilon = float(self.config["position"]["weight_epsilon"])

        for date in artifacts.target_positions.index:
            weights = artifacts.target_positions.loc[date]
            day_holdings = holdings.xs(date)
            diagnostics = artifacts.summary.loc[date]
            definitions: list[tuple[str, str, float, float | None, str]] = []
            budget = float(params.get("budget", 1.0))
            definitions.append(
                ("budget", "portfolio", float(weights.sum()), budget, "equality")
            )
            if bool(params.get("long_only", True)):
                definitions.append(
                    ("long_only", "asset", float(weights.min()), 0.0, "lower")
                )
            if params.get("single_name_max") is not None:
                definitions.append(
                    (
                        "single_name",
                        "asset",
                        float(weights.max()),
                        float(params["single_name_max"]),
                        "upper",
                    )
                )
            if params.get("gross_exposure_cap") is not None:
                definitions.append(
                    (
                        "gross_exposure",
                        "portfolio",
                        float(weights.abs().sum()),
                        float(params["gross_exposure_cap"]),
                        "upper",
                    )
                )
            if (
                params.get("turnover_cap") is not None
                and bool(diagnostics.get("turnover_constraint_applied", True))
            ):
                definitions.append(
                    (
                        "turnover",
                        "portfolio",
                        float(summary.loc[date, "one_way_turnover"]),
                        float(params["turnover_cap"]),
                        "upper",
                    )
                )
            if (
                params.get("active_weight_abs_max") is not None
                and day_holdings["active_weight"].notna().all()
            ):
                definitions.append(
                    (
                        "active_weight",
                        "asset",
                        float(day_holdings["active_weight"].abs().max()),
                        float(params["active_weight_abs_max"]),
                        "upper",
                    )
                )
            if (
                bool(params.get("enforce_tracking_error_cap", False))
                and params.get("tracking_error_cap_annual") is not None
                and pd.notna(summary.loc[date, "predicted_tracking_error"])
            ):
                alpha_days = float(
                    diagnostics.get(
                        "alpha_horizon_days",
                        artifacts.resolved_config.get("adapter", {}).get(
                            "alpha_horizon_days", 1
                        ),
                    )
                )
                annualization = float(
                    diagnostics.get("annualization_factor", 252.0)
                )
                annual_te = float(
                    summary.loc[date, "predicted_tracking_error"]
                    * np.sqrt(annualization / alpha_days)
                )
                definitions.append(
                    (
                        "tracking_error_cap",
                        "risk",
                        annual_te,
                        float(params["tracking_error_cap_annual"]),
                        "upper",
                    )
                )
            if (
                bool(params.get("enforce_tradable_mask", False))
                and day_holdings["is_tradable"].notna().all()
            ):
                frozen_trade = day_holdings.loc[
                    ~day_holdings["is_tradable"].astype(bool), "delta_weight"
                ].abs()
                definitions.append(
                    (
                        "tradable",
                        "asset",
                        float(frozen_trade.max()) if len(frozen_trade) else 0.0,
                        0.0,
                        "upper",
                    )
                )
            if not exposure_summary.empty and day_holdings["active_weight"].notna().all():
                day_exposure = exposure_summary.xs(date, level="date")
                for exposure_type, mode_key, band_key in (
                    ("industry", "industry_neutral_mode", "industry_band"),
                    ("style", "style_neutral_mode", "style_band"),
                ):
                    mode = str(params.get(mode_key, "off"))
                    if (
                        mode != "off"
                        and exposure_type
                        in day_exposure.index.get_level_values("exposure_type")
                    ):
                        maximum = float(
                            day_exposure.xs(
                                exposure_type, level="exposure_type"
                            )["active_exposure"].abs().max()
                        )
                        limit = (
                            0.0
                            if mode == "strict"
                            else float(params.get(band_key, 0.0))
                        )
                        definitions.append(
                            (
                                exposure_type,
                                exposure_type,
                                maximum,
                                limit,
                                "upper",
                            )
                        )

            for name, scope, value, limit, direction in definitions:
                if direction == "upper":
                    slack = float(limit - value)
                    violation = max(0.0, -slack)
                    utilization = (
                        value / limit if limit is not None and limit > 0 else np.nan
                    )
                elif direction == "lower":
                    slack = float(value - limit)
                    violation = max(0.0, -slack)
                    utilization = np.nan
                else:
                    slack = float(binding_tolerance - abs(value - limit))
                    violation = max(0.0, abs(value - limit) - binding_tolerance)
                    utilization = np.nan
                reported_violation = diagnostics.get(f"{name}_violation", np.nan)
                reported_slack = diagnostics.get(
                    {
                        "long_only": "long_only_min_slack",
                        "single_name": "single_name_min_slack",
                        "gross_exposure": "gross_exposure_slack",
                        "active_weight": "active_weight_min_slack",
                        "tracking_error_cap": "tracking_error_cap_slack",
                    }.get(name, f"{name}_slack"),
                    np.nan,
                )
                difference = (
                    float(reported_violation) - violation
                    if pd.notna(reported_violation)
                    else np.nan
                )
                rows.append(
                    {
                        "date": date,
                        "constraint_name": name,
                        "scope": scope,
                        "reported_value": np.nan,
                        "recomputed_value": value,
                        "limit": limit,
                        "slack": slack,
                        "utilization": utilization,
                        "is_binding": abs(slack) <= binding_tolerance,
                        "is_violated": violation > binding_tolerance,
                        "recompute_status": "passed",
                        "reported_violation": reported_violation,
                        "reported_slack": reported_slack,
                        "difference": difference,
                    }
                )
        if not rows:
            return empty_constraint_summary()
        return pd.DataFrame(rows).set_index(
            ["date", "constraint_name"]
        ).sort_index()

    def analyze(
        self,
        artifacts: OptimizationArtifacts,
        *,
        enrichment: AnalysisEnrichment | None = None,
    ) -> PositionAnalysisResult:
        """Run P0 and all configured P1 enrichments."""

        quality = _QualityCollector([])
        summary, holdings, turnover, _ = self._p0(artifacts, quality)
        combined = enrichment or AnalysisEnrichment()
        combined = self._load_p1(artifacts, combined, quality)
        self._apply_benchmark(
            artifacts, combined, summary, holdings, quality
        )
        self._apply_tradable(artifacts, combined, holdings, turnover)
        reference_exposure = self._apply_reference_attributes(
            artifacts, combined, holdings
        )
        exposure_summary, risk_contribution = self._apply_exposures_and_risk(
            artifacts, combined, summary, holdings, quality
        )
        if exposure_summary.empty:
            exposure_summary = reference_exposure
        elif not reference_exposure.empty:
            exposure_summary = pd.concat(
                [exposure_summary, reference_exposure]
            ).sort_index()
        self._apply_liquidity(
            artifacts, combined, turnover, quality
        )
        constraint_summary = self._build_constraints(
            artifacts, summary, holdings, exposure_summary
        )

        checks = quality.frame()
        for date in summary.index:
            day_checks = checks.loc[
                checks["date"].isna() | checks["date"].eq(date)
            ]
            if (
                (day_checks["severity"].eq("error") & day_checks["status"].eq("failed"))
                .any()
            ):
                summary.loc[date, "quality_status"] = "failed"
            elif (
                day_checks["severity"].eq("warning")
                & day_checks["status"].isin(["failed", "unavailable"])
            ).any():
                summary.loc[date, "quality_status"] = "partial"
            else:
                summary.loc[date, "quality_status"] = "passed"

        any_error = (
            (
                checks["severity"].eq("error")
                & checks["status"].eq("failed")
            ).any()
            if not checks.empty
            else False
        )
        status = "failed" if any_error else "partial" if quality.degraded else "passed"
        source_status = str(artifacts.run_manifest.get("status", "unknown"))
        if source_status == "failed":
            status = "failed"
            quality.add(
                "source_optimization_status",
                severity="error",
                status="failed",
                scope="optimizer",
                observed=source_status,
                expected="passed",
                message="source optimization manifest is failed",
            )
            checks = quality.frame()

        metadata = {
            "status": status,
            "analysis_version": 1,
            "source_status": source_status,
            "source_input_dir": (
                str(artifacts.input_dir) if artifacts.input_dir is not None else None
            ),
            "date_count": len(artifacts.target_positions),
            "asset_count": len(artifacts.target_positions.columns),
            "allow_partial_enrichment": bool(
                self.config["validation"]["allow_partial_enrichment"]
            ),
            "enabled_modules": {
                "position": True,
                "benchmark": combined.benchmark is not None,
                "tradable": combined.tradable is not None,
                "reference_attributes": (
                    combined.industry is not None
                    or combined.market_cap is not None
                ),
                "barra": combined.factor_exposure is not None,
                "liquidity": combined.market_amount is not None,
            },
            "provenance": combined.provenance,
            "quality_summary": {
                "errors": int(
                    (
                        checks["severity"].eq("error")
                        & checks["status"].eq("failed")
                    ).sum()
                ) if not checks.empty else 0,
                "warnings": int(
                    (
                        checks["severity"].eq("warning")
                        & checks["status"].isin(["failed", "unavailable"])
                    ).sum()
                ) if not checks.empty else 0,
                "unavailable": int(
                    checks["status"].eq("unavailable").sum()
                ) if not checks.empty else 0,
            },
        }
        return PositionAnalysisResult(
            position_summary=summary,
            holdings_detail=holdings,
            turnover_detail=turnover,
            exposure_summary=exposure_summary,
            risk_contribution=risk_contribution,
            constraint_summary=constraint_summary,
            quality_checks=checks,
            metadata=metadata,
        )


__all__ = ["PositionAnalyzer"]
