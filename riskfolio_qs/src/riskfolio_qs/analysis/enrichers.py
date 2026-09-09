"""Manifest-driven P1 data enrichment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..adapters.barra_precomputed_adapter import BarraPrecomputedAdapter
from ..adapters.data_access_inputs import load_portfolio_inputs
from .contracts import AnalysisEnrichment, OptimizationArtifacts


def snapshot_match(
    expected: dict[str, Any] | None,
    observed: dict[str, Any] | None,
) -> bool | None:
    """Compare a current data_access read with optimizer provenance."""

    expected_id = (expected or {}).get("snapshot_id")
    observed_id = (observed or {}).get("snapshot_id")
    if not expected_id or not observed_id:
        return None
    return str(expected_id) == str(observed_id)


def load_data_access_enrichment(
    artifacts: OptimizationArtifacts,
    *,
    include_benchmark: bool,
    data_store: Any | None = None,
) -> AnalysisEnrichment:
    """Reload benchmark/tradability using the optimizer's recorded routing."""

    adapter = dict(artifacts.resolved_config.get("adapter", {}))
    benchmark_index = str(adapter.get("benchmark_index", "")).strip()
    if include_benchmark and not benchmark_index:
        raise ValueError(
            "resolved_cli_config does not record adapter.benchmark_index"
        )
    minimum_coverage = float(
        adapter.get("minimum_benchmark_weight_coverage", 0.95)
    )
    loaded = load_portfolio_inputs(
        artifacts.target_positions,
        benchmark_index=benchmark_index or None,
        minimum_benchmark_weight_coverage=minimum_coverage,
        include_benchmark=include_benchmark,
        store=data_store,
    )
    expected_inputs = dict(artifacts.run_manifest.get("inputs", {}))
    provenance = dict(loaded.provenance)
    for name in ("benchmark", "tradable"):
        if name not in provenance:
            continue
        current = dict(provenance[name])
        current["expected_snapshot_id"] = (
            expected_inputs.get(name, {}) or {}
        ).get("snapshot_id")
        current["snapshot_match"] = snapshot_match(
            expected_inputs.get(name), current
        )
        provenance[name] = current
    return AnalysisEnrichment(
        benchmark=loaded.benchmark,
        tradable=loaded.tradable,
        provenance=provenance,
    )


def load_market_amount(
    artifacts: OptimizationArtifacts,
    *,
    lookback_days: int,
    data_store: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load point-in-time market amount history for ADV analysis."""

    if data_store is None:
        try:
            from data_access import get_store
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "liquidity analysis requires the internal data_access package"
            ) from exc
        data_store = get_store()

    dates = artifacts.target_positions.index
    assets = artifacts.target_positions.columns.tolist()
    calendar_days = max(90, int(lookback_days) * 4)
    start = (dates.min() - pd.Timedelta(days=calendar_days)).date().isoformat()
    end = dates.max().date().isoformat()
    result = data_store.read_result(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Amount"],
        time_range=(start, end),
        instrument_filter=assets,
    )
    frame = result.table.to_pandas()
    required = {"TradeDate", "Symbol", "Amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ashare_stock_daily is missing columns: {missing}")
    frame = frame.copy()
    frame["TradeDate"] = (
        pd.to_datetime(frame["TradeDate"]).dt.tz_localize(None).dt.normalize()
    )
    frame["Symbol"] = frame["Symbol"].astype(str)
    if frame.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError("ashare_stock_daily has duplicate TradeDate-Symbol rows")
    frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce")
    amount = (
        frame.pivot(index="TradeDate", columns="Symbol", values="Amount")
        .reindex(columns=assets)
        .sort_index()
    )
    snapshot = result.snapshot
    stats = result.stats
    provenance = {
        "source": "data_access",
        "dataset": "ashare_stock_daily",
        "field": "Amount",
        "snapshot_id": str(snapshot.snapshot_id),
        "registry_hash": str(snapshot.registry_hash),
        "schema_hash": str(snapshot.schema_hash),
        "file_manifest_hash": str(snapshot.file_manifest_hash),
        "rows": int(stats.rows),
        "bytes": int(stats.bytes),
        "load_start": start,
        "load_end": end,
    }
    return amount, provenance


def _snapshot_details(result: Any, *, dataset: str) -> dict[str, Any]:
    snapshot = result.snapshot
    stats = result.stats
    return {
        "source": "data_access",
        "dataset": dataset,
        "snapshot_id": str(snapshot.snapshot_id),
        "registry_hash": str(snapshot.registry_hash),
        "schema_hash": str(snapshot.schema_hash),
        "file_manifest_hash": str(snapshot.file_manifest_hash),
        "rows": int(stats.rows),
        "bytes": int(stats.bytes),
    }


def _asof_wide(
    frame: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    assets: list[str],
    value_column: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=dates, columns=assets)
    ordered = frame.sort_values(["TradeDate", "Symbol"])
    for date in dates:
        eligible = ordered.loc[ordered["TradeDate"] <= date]
        latest = eligible.drop_duplicates("Symbol", keep="last").set_index("Symbol")
        result.loc[date] = latest[value_column].reindex(assets)
    result.index.name = "date"
    result.columns.name = "asset"
    return result


def load_reference_attributes(
    artifacts: OptimizationArtifacts,
    *,
    data_store: Any | None = None,
) -> AnalysisEnrichment:
    """Load point-in-time industry and market cap for P1 exposure analysis."""

    if data_store is None:
        try:
            from data_access import get_store
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "reference attribute enrichment requires data_access"
            ) from exc
        data_store = get_store()
    dates = artifacts.target_positions.index
    assets = artifacts.target_positions.columns.tolist()
    start = (dates.min() - pd.Timedelta(days=370)).date().isoformat()
    end = dates.max().date().isoformat()

    industry_result = data_store.read_result(
        "ashare_stock_industry",
        columns=["TradeDate", "Symbol", "IndustrySource", "IndustryCode"],
        time_range=(start, end),
        instrument_filter=assets,
    )
    industry_long = industry_result.table.to_pandas()
    industry_long["TradeDate"] = (
        pd.to_datetime(industry_long["TradeDate"])
        .dt.tz_localize(None)
        .dt.normalize()
    )
    industry_long["Symbol"] = industry_long["Symbol"].astype(str)
    industry_long = industry_long.loc[
        industry_long["IndustrySource"].astype(str).eq("sw_l1")
    ].copy()
    if industry_long.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError("ashare_stock_industry has duplicate TradeDate-Symbol rows")
    industry = _asof_wide(
        industry_long,
        dates=dates,
        assets=assets,
        value_column="IndustryCode",
    )

    valuation_result = data_store.read_result(
        "ashare_stock_valuation_daily",
        columns=["TradeDate", "Symbol", "MarketCap"],
        time_range=(start, end),
        instrument_filter=assets,
    )
    valuation_long = valuation_result.table.to_pandas()
    valuation_long["TradeDate"] = (
        pd.to_datetime(valuation_long["TradeDate"])
        .dt.tz_localize(None)
        .dt.normalize()
    )
    valuation_long["Symbol"] = valuation_long["Symbol"].astype(str)
    if valuation_long.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError(
            "ashare_stock_valuation_daily has duplicate TradeDate-Symbol rows"
        )
    valuation_long["MarketCap"] = pd.to_numeric(
        valuation_long["MarketCap"], errors="coerce"
    )
    market_cap = _asof_wide(
        valuation_long,
        dates=dates,
        assets=assets,
        value_column="MarketCap",
    ).astype(float)
    return AnalysisEnrichment(
        industry=industry,
        market_cap=market_cap,
        provenance={
            "industry": {
                **_snapshot_details(
                    industry_result, dataset="ashare_stock_industry"
                ),
                "industry_source": "sw_l1",
            },
            "market_cap": _snapshot_details(
                valuation_result, dataset="ashare_stock_valuation_daily"
            ),
        },
    )


def _risk_package_hash(root: Path, manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    paths = [root / "manifest.yaml"]
    for value in dict(manifest.get("files", {})).values():
        path = root / str(value)
        if path.exists():
            paths.append(path)
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_barra_enrichment(
    artifacts: OptimizationArtifacts,
    *,
    benchmark: pd.DataFrame | None,
    tradable: pd.DataFrame | None,
) -> AnalysisEnrichment:
    """Load the same precomputed B/F/D package recorded by the optimizer."""

    inputs = dict(artifacts.run_manifest.get("inputs", {}))
    barra_input = dict(inputs.get("barra_risk", {}) or {})
    configured = dict(artifacts.resolved_config.get("adapter", {}))
    raw_root = barra_input.get("path") or configured.get("risk_root")
    if not raw_root:
        raise ValueError("optimization manifest does not record a Barra risk package")
    root = Path(str(raw_root)).expanduser().resolve()
    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Barra manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    weights = artifacts.target_positions
    trade_matrix = (
        artifacts.trades["delta_weight"]
        .unstack("asset")
        .reindex(index=weights.index, columns=weights.columns)
    )
    initial = (weights.iloc[0] - trade_matrix.iloc[0]).to_frame().T
    initial.index = pd.DatetimeIndex([weights.index[0]], name="date")
    benchmark_frame = (
        benchmark.reindex(index=weights.index, columns=weights.columns)
        if benchmark is not None
        else pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    )
    tradable_frame = (
        tradable.reindex(index=weights.index, columns=weights.columns)
        if tradable is not None
        else pd.DataFrame(True, index=weights.index, columns=weights.columns)
    )
    adapter = BarraPrecomputedAdapter(
        risk_root=root,
        alpha_df=weights,
        benchmark_df=benchmark_frame,
        prev_positions_df=initial,
        tradable_df=tradable_frame,
        alpha_input_type=str(configured.get("alpha_input_type", "score")),
        alpha_horizon_days=int(configured.get("alpha_horizon_days", 1)),
        calendar_name=str(configured.get("calendar_name", "XSHG")),
        risk_data_lag_periods=int(configured.get("risk_data_lag_periods", 0)),
        exposure_data_lag_periods=int(
            configured.get("exposure_data_lag_periods", 0)
        ),
        annualization_factor=float(configured.get("annualization_factor", 252.0)),
        minimum_annual_specific_volatility=float(
            configured.get("minimum_annual_specific_volatility", 0.05)
        ),
    )
    bundle = adapter.build_bundle()
    provenance = {
        "barra": {
            "source": "precomputed_package",
            "path": str(root),
            "product": manifest.get("product"),
            "version": manifest.get("version"),
            "provider": manifest.get("provider"),
            "content_sha256": _risk_package_hash(root, manifest),
            "provenance_status": (
                "verified"
                if barra_input.get("content_sha256")
                else "path_only"
            ),
            "expected_content_sha256": barra_input.get("content_sha256"),
        }
    }
    expected_hash = barra_input.get("content_sha256")
    provenance["barra"]["content_match"] = (
        None
        if not expected_hash
        else str(expected_hash) == provenance["barra"]["content_sha256"]
    )
    return AnalysisEnrichment(
        factor_exposure=bundle.F,
        factor_cov=bundle.factor_cov,
        specific_var=bundle.specific_var,
        factor_specs=list(manifest.get("factors", [])),
        provenance=provenance,
    )


__all__ = [
    "load_barra_enrichment",
    "load_data_access_enrichment",
    "load_market_amount",
    "load_reference_attributes",
    "snapshot_match",
]
