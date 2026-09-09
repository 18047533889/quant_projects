from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import riskfolio_qs.analysis.position_analyzer as analyzer_module
from riskfolio_qs.analysis import (
    AnalysisEnrichment,
    PositionAnalyzer,
    analyze_output_bundle,
)
from riskfolio_qs.analysis.artifact_loader import (
    artifacts_from_output_bundle,
    load_artifact_directory,
)
from riskfolio_qs.analysis.config import load_analysis_config
from riskfolio_qs.analysis.report import write_analysis_outputs
from riskfolio_qs.cli import main
from riskfolio_qs.core.contracts import OutputBundle


def _bundle() -> OutputBundle:
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date")
    assets = pd.Index(["A", "B"], name="asset")
    weights = pd.DataFrame(
        [[0.6, 0.4], [0.5, 0.5]], index=dates, columns=assets
    )
    initial = pd.Series([0.5, 0.5], index=assets)
    previous = weights.shift(1)
    previous.iloc[0] = initial
    trades = (
        (weights - previous)
        .stack(future_stack=True)
        .rename("delta_weight")
        .to_frame()
    )
    trades.index.names = ["date", "asset"]
    summary = pd.DataFrame(
        {
            "gross_exposure": [1.0, 1.0],
            "net_exposure": [1.0, 1.0],
            "turnover": [0.1, 0.1],
            "feasible_flag": [True, True],
        },
        index=dates,
    )
    metadata = pd.DataFrame(
        [{"optimizer_name": "test", "fallback_used": False}],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-04", tz="UTC")], name="date"),
    )
    return OutputBundle(weights, trades, summary, metadata)


def _config(**sections) -> dict:
    config = {
        "analysis_version": 1,
        "benchmark": {"mode": "off"},
        "barra": {"mode": "off"},
        "liquidity": {"mode": "off"},
        "report": {"html": False},
    }
    config.update(sections)
    return config


def test_p0_reconstructs_initial_position_and_recomputes_metrics() -> None:
    result = analyze_output_bundle(_bundle(), config=_config())

    assert result.status == "passed"
    assert result.position_summary.loc[
        pd.Timestamp("2024-01-02"), "one_way_turnover"
    ] == pytest.approx(0.1)
    assert result.position_summary.loc[
        pd.Timestamp("2024-01-02"), "hhi"
    ] == pytest.approx(0.52)
    first = result.holdings_detail.xs(pd.Timestamp("2024-01-02"))
    np.testing.assert_allclose(first["previous_weight"], [0.5, 0.5])
    assert result.position_summary["quality_status"].eq("passed").all()


def test_artifact_validation_rejects_trade_reconstruction_error() -> None:
    bundle = _bundle()
    bundle.trades.iloc[2, 0] += 0.01
    with pytest.raises(ValueError, match="reconstruction error"):
        artifacts_from_output_bundle(bundle)


def test_p1_benchmark_barra_and_risk_contribution() -> None:
    bundle = _bundle()
    dates = bundle.target_positions.index
    risk_date = pd.Timestamp("2024-01-01")
    factors = ["style_size", "industry_10"]
    exposure = pd.DataFrame(
        [[1.0, 1.0], [-1.0, 1.0]],
        index=pd.MultiIndex.from_product(
            [[risk_date], ["A", "B"]], names=["date", "asset"]
        ),
        columns=factors,
    )
    covariance = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.005]], index=factors, columns=factors
    )
    factor_cov = pd.concat(
        {risk_date: covariance}, names=["date", "factor_i"]
    )
    specific = pd.DataFrame(
        [[0.002, 0.003]], index=[risk_date], columns=["A", "B"]
    )
    benchmark = pd.DataFrame(
        0.5, index=dates, columns=bundle.target_positions.columns
    )
    tradable = pd.DataFrame(
        True, index=dates, columns=bundle.target_positions.columns
    )
    bundle.summary["risk_exposure_date"] = risk_date
    bundle.summary["risk_covariance_date"] = risk_date
    bundle.summary["specific_risk_date"] = risk_date
    bundle.summary["risk_covariance_units"] = "daily_variance"
    bundle.summary["risk_horizon_days"] = 1
    bundle.summary["alpha_horizon_days"] = 1
    bundle.summary["annualization_factor"] = 252.0

    enrichment = AnalysisEnrichment(
        benchmark=benchmark,
        tradable=tradable,
        factor_exposure=exposure,
        factor_cov=factor_cov,
        specific_var=specific,
        factor_specs=[
            {"id": "style_size", "type": "continuous"},
            {"id": "industry_10", "type": "dummy"},
        ],
    )
    resolved, _ = load_analysis_config(
        _config(
            benchmark={"mode": "off"},
            barra={
                "mode": "off",
                "analyze_absolute_risk": True,
                "analyze_active_risk": True,
            },
        )
    )
    artifacts = artifacts_from_output_bundle(bundle)
    result = PositionAnalyzer(resolved).analyze(
        artifacts, enrichment=enrichment
    )

    assert result.position_summary.iloc[0]["active_share"] == pytest.approx(0.1)
    assert not result.exposure_summary.empty
    assert set(
        result.risk_contribution.index.get_level_values("risk_basis")
    ) == {"absolute", "active"}
    active = result.risk_contribution.xs(
        (dates[0], "active"), level=("date", "risk_basis")
    )
    asset_sum = active.xs("asset", level="component_type")[
        "variance_contribution"
    ].sum()
    total = active.loc[("total", "portfolio"), "variance"]
    assert asset_sum == pytest.approx(total)


def test_analyze_positions_cli_writes_structured_outputs(
    tmp_path: Path, capsys
) -> None:
    bundle = _bundle()
    source = tmp_path / "optimization"
    source.mkdir()
    bundle.target_positions.to_parquet(source / "target_positions.parquet")
    bundle.trades.to_parquet(source / "trades.parquet")
    bundle.summary.to_parquet(source / "summary.parquet")
    bundle.metadata.reset_index().to_json(
        source / "metadata.json",
        orient="records",
        date_format="iso",
    )
    (source / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "passed",
                "date_count": 2,
                "asset_count": 2,
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    config = _config(
        input={"optimization_dir": str(source)},
        output={
            "directory": str(output),
            "parquet": True,
            "overwrite": False,
        },
    )
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    exit_code = main(["analyze-positions", "--config", str(config_path)])

    assert exit_code == 0
    assert "status=passed" in capsys.readouterr().out
    assert (output / "position_summary.parquet").exists()
    assert (output / "holdings_detail.parquet").exists()
    assert (output / "analysis_manifest.yaml").exists()
    loaded = load_artifact_directory(source)
    assert len(loaded.target_positions) == 2


def test_strict_snapshot_mismatch_stops_auto_benchmark_enrichment(
    monkeypatch,
) -> None:
    artifacts = artifacts_from_output_bundle(_bundle())
    artifacts.run_manifest = {
        "status": "passed",
        "inputs": {
            "benchmark": {"snapshot_id": "old"},
            "tradable": {"snapshot_id": "old"},
        },
    }
    artifacts.resolved_config = {
        "adapter": {"benchmark_index": "000905.SH"}
    }
    weights = artifacts.target_positions

    def mismatched(*args, **kwargs):
        return AnalysisEnrichment(
            benchmark=pd.DataFrame(
                0.5, index=weights.index, columns=weights.columns
            ),
            tradable=pd.DataFrame(
                True, index=weights.index, columns=weights.columns
            ),
            provenance={
                "benchmark": {"snapshot_match": False},
                "tradable": {"snapshot_match": False},
            },
        )

    monkeypatch.setattr(
        analyzer_module, "load_data_access_enrichment", mismatched
    )
    resolved, _ = load_analysis_config(
        {
            "analysis_version": 1,
            "validation": {"provenance_policy": "strict"},
            "benchmark": {"mode": "auto"},
            "barra": {"mode": "off"},
            "liquidity": {"mode": "off"},
        }
    )
    result = PositionAnalyzer(resolved).analyze(artifacts)

    assert result.status == "partial"
    assert result.position_summary["active_share"].isna().all()
    assert result.quality_checks["check_id"].eq(
        "benchmark_enrichment"
    ).any()


def test_liquidity_uses_only_prior_amount_and_parquet_schema_is_stable(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    amount = pd.DataFrame(
        [[100.0, 200.0], [10_000.0, 20_000.0], [1_000_000.0, 2_000_000.0]],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        columns=["A", "B"],
    )
    config = _config(
        liquidity={
            "mode": "required",
            "portfolio_notional": 1_000.0,
            "adv_window_days": 20,
            "maximum_adv_participation": 0.10,
        },
        output={
            "directory": str(tmp_path),
            "parquet": True,
            "overwrite": True,
        },
    )
    resolved, config_path = load_analysis_config(config)
    result = PositionAnalyzer(resolved).analyze(
        artifacts_from_output_bundle(bundle),
        enrichment=AnalysisEnrichment(market_amount=amount),
    )

    first = result.turnover_detail.xs(pd.Timestamp("2024-01-02"))
    assert first.loc["A", "adv"] == pytest.approx(100.0)
    assert first.loc["B", "adv"] == pytest.approx(200.0)
    write_analysis_outputs(
        result,
        tmp_path,
        config=resolved,
        config_path=config_path,
        overwrite=True,
    )
    quality = pd.read_parquet(tmp_path / "quality_checks.parquet")
    assert quality["observed"].dtype == object
    assert pd.api.types.is_float_dtype(quality["difference"])
