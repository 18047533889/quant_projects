from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import riskfolio_qs.cli as cli_module
from riskfolio_qs.adapters.data_access_inputs import (
    DataAccessHistoricalMarket,
    DataAccessPortfolioInputs,
)
from riskfolio_qs.cli import main
from riskfolio_qs.runners import OptimizationPipeline


def _write_barra_fixture(root: Path, assets: list[str]) -> None:
    dates = ["2024-01-02", "2024-01-03"]
    factors = [
        {"id": "style_lncap", "type": "continuous"},
        {"id": "industry_801010", "type": "dummy"},
    ]
    manifest = {
        "product": "barra_lite",
        "version": "cli-test",
        "provider": "pytest",
        "factors": factors,
        "estimation": {"units": "daily_variance"},
        "files": {
            "exposure": "exposure.parquet",
            "factor_cov": "factor_cov.parquet",
            "specific_risk": "specific_risk.parquet",
        },
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    exposure_rows: list[dict[str, object]] = []
    specific_rows: list[dict[str, object]] = []
    covariance_rows: list[dict[str, object]] = []
    factor_ids = [factor["id"] for factor in factors]
    for date in dates:
        for position, asset in enumerate(assets):
            exposure_rows.extend(
                [
                    {
                        "date": date,
                        "asset": asset,
                        "factor_id": "style_lncap",
                        "exposure": (position - len(assets) / 2) / len(assets),
                    },
                    {
                        "date": date,
                        "asset": asset,
                        "factor_id": "industry_801010",
                        "exposure": 1.0,
                    },
                ]
            )
            specific_rows.append(
                {"date": date, "asset": asset, "specific_var": 1e-4}
            )
        for factor_i in factor_ids:
            for factor_j in factor_ids:
                covariance_rows.append(
                    {
                        "date": date,
                        "factor_i": factor_i,
                        "factor_j": factor_j,
                        "cov": 1e-4 if factor_i == factor_j else 0.0,
                    }
                )
    pd.DataFrame(exposure_rows).to_parquet(root / "exposure.parquet")
    pd.DataFrame(specific_rows).to_parquet(root / "specific_risk.parquet")
    pd.DataFrame(covariance_rows).to_parquet(root / "factor_cov.parquet")


def _to_long(
    frame: pd.DataFrame,
    *,
    date_column: str,
    asset_column: str,
    value_column: str,
) -> pd.DataFrame:
    result = (
        frame.rename_axis(index=date_column, columns=asset_column)
        .stack()
        .rename(value_column)
        .reset_index()
    )
    return result


def test_cli_runs_config_with_alpha_override(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    assets = [f"A{i:03d}" for i in range(40)]
    risk_root = tmp_path / "barra"
    risk_root.mkdir()
    _write_barra_fixture(risk_root, assets)

    decision_dates = pd.DatetimeIndex(["2024-01-04", "2024-01-05"])
    alpha = pd.DataFrame(
        np.random.default_rng(7).normal(
            0.0, 0.01, size=(len(decision_dates), len(assets))
        ),
        index=decision_dates,
        columns=assets,
    )
    benchmark = pd.DataFrame(
        1.0 / len(assets), index=decision_dates, columns=assets
    )
    tradable = pd.DataFrame(True, index=decision_dates, columns=assets)

    alpha_path = tmp_path / "alpha.parquet"
    _to_long(
        alpha,
        date_column="TradeDate",
        asset_column="Symbol",
        value_column="pred",
    ).to_parquet(alpha_path)
    monkeypatch.setattr(
        cli_module,
        "load_portfolio_inputs",
        lambda loaded_alpha, **_: DataAccessPortfolioInputs(
            benchmark=benchmark.reindex(
                index=loaded_alpha.index, columns=loaded_alpha.columns
            ),
            tradable=tradable.reindex(
                index=loaded_alpha.index, columns=loaded_alpha.columns
            ),
            provenance={
                "benchmark": {
                    "source": "data_access",
                    "dataset": "ashare_index_constituent",
                    "snapshot_id": "benchmark-test-snapshot",
                },
                "tradable": {
                    "source": "data_access",
                    "dataset": "ashare_stock_daily",
                    "snapshot_id": "daily-test-snapshot",
                },
                "linear_cost_bps": {
                    "source": "riskfolio_qs_internal_constant",
                    "value": 5.0,
                },
            },
        ),
    )

    config = {
        "config_version": 1,
        "adapter": {
            "type": "barra_precomputed",
            "risk_root": "barra",
            "benchmark_index": "000905.SH",
            "alpha_input_type": "expected_return",
            "alpha_horizon_days": 5,
            "risk_data_lag_periods": 0,
            "exposure_data_lag_periods": 0,
        },
        "inputs": {
            "alpha": {
                "path": "missing.parquet",
                "layout": "long",
                "date_column": "TradeDate",
                "asset_column": "Symbol",
                "value_column": "pred",
            },
        },
        "smoother": {"mode": "never"},
        "run": {
            "optimizer_name": "meanvar_enhance_barra_precomputed",
            "allow_fallback": False,
            "runtime_overrides": {"tracking_error_cap_annual": 0.03},
        },
        "output": {"directory": "unused-output", "overwrite": False},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    output_dir = tmp_path / "actual-output"

    exit_code = main(
        [
            "optimize",
            "--config",
            str(config_path),
            "--alpha",
            str(alpha_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "status=passed" in captured.out
    assert "optimizer=meanvar_enhance_barra_precomputed" in captured.out
    assert (output_dir / "target_positions.parquet").exists()
    assert (output_dir / "trades.parquet").exists()
    assert (output_dir / "summary.parquet").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "resolved_mapping.yaml").exists()
    assert (output_dir / "resolved_params.yaml").exists()
    manifest = yaml.safe_load(
        (output_dir / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "passed"
    assert manifest["asset_count"] == len(assets)
    assert manifest["inputs"]["alpha"]["path"] == str(alpha_path.resolve())
    assert manifest["inputs"]["benchmark"]["source"] == "data_access"
    assert (
        manifest["inputs"]["benchmark"]["snapshot_id"]
        == "benchmark-test-snapshot"
    )
    assert manifest["inputs"]["initial_positions"] == {
        "source": "data_access_benchmark",
        "assumption": "backtest_starts_from_benchmark",
        "date": "2024-01-04",
        "index_symbol": "000905.SH",
    }

    weights = pd.read_parquet(output_dir / "target_positions.parquet")
    trades = pd.read_parquet(output_dir / "trades.parquet")
    summary = pd.read_parquet(output_dir / "summary.parquet")
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    first_trade = trades.xs(decision_dates[0])["delta_weight"].reindex(assets)
    second_trade = trades.xs(decision_dates[1])["delta_weight"].reindex(assets)
    np.testing.assert_allclose(
        first_trade,
        weights.loc[decision_dates[0]] - benchmark.loc[decision_dates[0]],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        second_trade,
        weights.loc[decision_dates[1]] - weights.loc[decision_dates[0]],
        atol=1e-8,
    )
    assert summary["feasible_flag"].all()
    assert summary["max_constraint_violation"].max() <= 1e-5

    second_exit_code = main(
        [
            "optimize",
            "--config",
            str(config_path),
            "--alpha",
            str(alpha_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert second_exit_code == 1
    assert "--overwrite" in capsys.readouterr().err

    alias_config = deepcopy(config)
    alias_config["run"]["optimizer_name"] = "meanvar_enhance_index"
    alias_config["inputs"]["alpha"]["path"] = "alpha.parquet"
    alias_config["output"] = {
        "directory": "legacy-alias-output",
        "overwrite": False,
    }
    alias_config_path = tmp_path / "legacy-alias.yaml"
    alias_config_path.write_text(
        yaml.safe_dump(alias_config, sort_keys=False), encoding="utf-8"
    )
    assert main(["optimize", "--config", str(alias_config_path)]) == 0
    alias_manifest = yaml.safe_load(
        (tmp_path / "legacy-alias-output" / "run_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert alias_manifest["optimizer_name"] == (
        "meanvar_enhance_barra_precomputed"
    )
    assert alias_manifest["inputs"]["routing"] == {
        "requested": "meanvar_enhance_index",
        "resolved_optimizer": "meanvar_enhance_barra_precomputed",
        "risk_mode": "barra_precomputed",
    }

    legacy_config = dict(config)
    legacy_config["inputs"] = dict(config["inputs"])
    legacy_config["inputs"]["benchmark"] = "legacy-benchmark.parquet"
    legacy_config_path = tmp_path / "legacy-run.yaml"
    legacy_config_path.write_text(
        yaml.safe_dump(legacy_config, sort_keys=False), encoding="utf-8"
    )
    legacy_exit_code = main(["optimize", "--config", str(legacy_config_path)])
    assert legacy_exit_code == 1
    assert "unknown keys: ['benchmark']" in capsys.readouterr().err


def test_cli_topn_uses_alpha_only_without_data_access_or_barra(
    tmp_path: Path, monkeypatch
) -> None:
    assets = [f"A{i:03d}" for i in range(60)]
    dates = pd.DatetimeIndex(["2024-01-04", "2024-01-05"])
    alpha = pd.DataFrame(
        np.random.default_rng(17).normal(size=(2, len(assets))),
        index=dates,
        columns=assets,
    )
    alpha_path = tmp_path / "alpha.parquet"
    alpha.to_parquet(alpha_path)

    def unexpected_data_access(*args, **kwargs):
        raise AssertionError("TopN must not load data_access inputs")

    monkeypatch.setattr(cli_module, "load_portfolio_inputs", unexpected_data_access)
    monkeypatch.setattr(
        cli_module, "load_historical_market_returns", unexpected_data_access
    )
    config = {
        "config_version": 1,
        "adapter": {"type": "auto"},
        "inputs": {"alpha": {"path": "alpha.parquet", "layout": "wide"}},
        "smoother": {"mode": "never"},
        "run": {
            "optimizer_name": "topn_long_only_equal_weight",
            "allow_fallback": False,
        },
        "output": {"directory": "topn-output", "overwrite": False},
    }
    config_path = tmp_path / "topn.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    assert main(["optimize", "--config", str(config_path)]) == 0
    weights = pd.read_parquet(tmp_path / "topn-output" / "target_positions.parquet")
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert (weights.gt(0).sum(axis=1) == 50).all()
    manifest = yaml.safe_load(
        (tmp_path / "topn-output" / "run_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["inputs"]["routing"]["risk_mode"] == "none"
    assert "benchmark" not in manifest["inputs"]
    assert "barra_risk" not in manifest["inputs"]


def test_cli_historical_optimizer_assembles_data_access_returns(
    tmp_path: Path, monkeypatch
) -> None:
    assets = [f"A{i:03d}" for i in range(40)]
    dates = pd.DatetimeIndex(["2024-04-01", "2024-04-02"])
    history_dates = pd.bdate_range("2024-01-02", "2024-04-02")
    rng = np.random.default_rng(23)
    alpha = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(2, len(assets))),
        index=dates,
        columns=assets,
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(history_dates), len(assets))),
        index=history_dates,
        columns=assets,
    )
    benchmark = pd.DataFrame(
        1.0 / len(assets), index=dates, columns=assets
    )
    tradable = pd.DataFrame(True, index=dates, columns=assets)
    alpha.to_parquet(tmp_path / "alpha.parquet")

    def portfolio_loader(loaded_alpha, **kwargs):
        assert kwargs["include_benchmark"] is True
        return DataAccessPortfolioInputs(
            benchmark=benchmark,
            tradable=tradable,
            provenance={"benchmark": {"source": "test"}, "tradable": {"source": "test"}},
        )

    def market_loader(loaded_alpha, **kwargs):
        assert kwargs["lookback_days"] == 60
        return DataAccessHistoricalMarket(
            returns=returns,
            provenance={
                "market": {
                    "source": "test",
                    "input_type": "return",
                    "scale": 0.0001,
                }
            },
        )

    monkeypatch.setattr(cli_module, "load_portfolio_inputs", portfolio_loader)
    monkeypatch.setattr(
        cli_module, "load_historical_market_returns", market_loader
    )
    config = {
        "config_version": 1,
        "adapter": {
            "type": "data_access",
            "benchmark_index": "000905.SH",
            "alpha_input_type": "expected_return",
            "alpha_horizon_days": 5,
        },
        "inputs": {"alpha": {"path": "alpha.parquet", "layout": "wide"}},
        "smoother": {"mode": "never"},
        "run": {
            "optimizer_name": "meanvar_enhance_hist",
            "allow_fallback": False,
        },
        "output": {"directory": "hist-output", "overwrite": False},
    }
    config_path = tmp_path / "hist.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    assert main(["optimize", "--config", str(config_path)]) == 0
    summary = pd.read_parquet(tmp_path / "hist-output" / "summary.parquet")
    assert summary["feasible_flag"].all()
    assert set(summary["risk_model_type"]) == {"historical_cov"}
    manifest = yaml.safe_load(
        (tmp_path / "hist-output" / "run_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["inputs"]["market"]["input_type"] == "return"
    assert manifest["inputs"]["routing"]["resolved_optimizer"] == (
        "meanvar_enhance_hist"
    )


def test_cli_absolute_return_does_not_require_benchmark(
    tmp_path: Path, monkeypatch
) -> None:
    assets = [f"A{i:03d}" for i in range(30)]
    dates = pd.DatetimeIndex(["2024-04-01", "2024-04-02"])
    history_dates = pd.bdate_range("2024-01-02", "2024-04-02")
    rng = np.random.default_rng(31)
    alpha = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(2, len(assets))),
        index=dates,
        columns=assets,
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(history_dates), len(assets))),
        index=history_dates,
        columns=assets,
    )
    tradable = pd.DataFrame(True, index=dates, columns=assets)
    alpha.to_parquet(tmp_path / "alpha.parquet")

    def portfolio_loader(loaded_alpha, **kwargs):
        assert kwargs["include_benchmark"] is False
        assert kwargs["benchmark_index"] is None
        return DataAccessPortfolioInputs(
            benchmark=None,
            tradable=tradable,
            provenance={"tradable": {"source": "test"}},
        )

    monkeypatch.setattr(cli_module, "load_portfolio_inputs", portfolio_loader)
    monkeypatch.setattr(
        cli_module,
        "load_historical_market_returns",
        lambda *args, **kwargs: DataAccessHistoricalMarket(
            returns=returns,
            provenance={"market": {"source": "test", "input_type": "return"}},
        ),
    )
    config = {
        "config_version": 1,
        "adapter": {"type": "data_access"},
        "inputs": {"alpha": {"path": "alpha.parquet", "layout": "wide"}},
        "smoother": {"mode": "never"},
        "run": {
            "optimizer_name": "meanvar_absolute_return",
            "allow_fallback": False,
        },
        "output": {"directory": "abs-output", "overwrite": False},
    }
    config_path = tmp_path / "abs.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    assert main(["optimize", "--config", str(config_path)]) == 0
    manifest = yaml.safe_load(
        (tmp_path / "abs-output" / "run_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "benchmark" not in manifest["inputs"]
    assert manifest["inputs"]["initial_positions"]["assumption"] == (
        "backtest_starts_from_cash"
    )


def test_cli_routes_scenarios_and_legacy_barra_names_before_loading_inputs() -> None:
    pipeline = OptimizationPipeline()

    requested, effective, spec = cli_module._resolve_optimizer_for_inputs(
        pipeline,
        optimizer_name="meanvar_enhance_index",
        scenario=None,
    )
    assert requested == "meanvar_enhance_index"
    assert effective == "meanvar_enhance_barra_precomputed"
    assert spec.risk_mode == "barra_precomputed"

    requested, effective, spec = cli_module._resolve_optimizer_for_inputs(
        pipeline,
        optimizer_name=None,
        scenario="fallback",
    )
    assert requested == "scenario:fallback"
    assert effective == "topn_long_only_equal_weight"
    assert spec.risk_mode == "none"
