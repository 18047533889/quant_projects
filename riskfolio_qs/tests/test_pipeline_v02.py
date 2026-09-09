"""
v0.2 管线回归测试：验证配置驱动管线的核心行为。

测试覆盖：
- T01: 配置落盘与审计字段正确性
- T02: 强依赖缺失时 fail_fast 行为
- T03: 白名单外参数覆盖被拦截
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from riskfolio_qs.adapters import BarraPrecomputedAdapter
from riskfolio_qs.adapters.mock_adapter import MockInputAdapter
from riskfolio_qs.adapters.real_adapter import RealInputAdapter
from riskfolio_qs.core.contracts import InputBundle
import numpy as np
import pandas as pd

from riskfolio_qs.optimizers.portfolio_optimizer import PrecomputedFactorRisk
from riskfolio_qs.optimizers.parameter_store import ParameterStore
from riskfolio_qs.runners.pipeline import OptimizationPipeline
from riskfolio_qs.smoothers.signal_smoother import SignalSmoother


def test_pipeline_writes_resolved_configs(tmp_path: Path) -> None:
    """验证管线在运行后正确落盘 resolved_mapping.yaml 和 resolved_params.yaml，
    并确保审计元数据中各字段与预期一致。"""
    # 3% 个股上限下至少需要 34 个资产才可满仓。
    adapter = MockInputAdapter(n_dates=8, n_assets=60)
    bundle = adapter.build_bundle()
    pipeline = OptimizationPipeline()

    output = pipeline.run(
        bundle,
        optimizer_name="meanvar_enhance_index",
        output_dir=str(tmp_path),
        data_version_hash="hash_demo_001",
    )

    # 验证 resolved 配置文件已生成
    assert (tmp_path / "resolved_mapping.yaml").exists()
    assert (tmp_path / "resolved_params.yaml").exists()

    # 验证审计元数据字段
    meta = output.metadata.iloc[0]
    assert meta["optimizer_name"] == "meanvar_enhance_index"
    assert meta["parameter_version"] == "v0.2.4"
    assert meta["mapping_version"] == "v0.2.4"
    assert meta["data_version_hash"] == "hash_demo_001"
    assert meta["solve_time_ms"] > 0
    assert output.summary["solve_status"].isin({"optimal", "optimal_inaccurate"}).all()
    assert output.summary["feasible_flag"].all()


def test_missing_barra_degrades_to_historical_index_enhancement() -> None:
    """缺 Barra 时保留指增语义，降级到历史协方差而不是 TopN。"""
    adapter = MockInputAdapter(n_dates=4, n_assets=60)
    bundle = adapter.build_bundle()
    bundle.F_ret = None

    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_enhance_index"
    )
    meta = output.metadata.iloc[0]
    assert meta["optimizer_name"] == "meanvar_enhance_hist"
    assert meta["risk_mode"] == "historical_cov"
    assert bool(meta["fallback_used"])
    assert meta["solve_status"] == "success_with_fallback"
    assert meta["extras"]["requested_optimizer_name"] == "meanvar_enhance_index"
    assert "F_ret" in meta["extras"]["fallback_reason"]


def test_historical_fallback_fails_if_market_is_also_missing() -> None:
    bundle = MockInputAdapter(n_dates=3, n_assets=60).build_bundle()
    bundle.F_ret = None
    bundle.market = None
    with pytest.raises(ValueError, match="Fallback optimizer meanvar_enhance_hist"):
        OptimizationPipeline().run(bundle, optimizer_name="meanvar_enhance_index")


def test_runtime_override_whitelist_enforced() -> None:
    """验证非白名单参数覆盖请求被正确拦截。

    "not_allowed_param" 不在 overrides_whitelist 中，
    传入应触发 ValueError。
    """
    adapter = MockInputAdapter(n_dates=4, n_assets=10)
    bundle = adapter.build_bundle()

    pipeline = OptimizationPipeline()
    with pytest.raises(ValueError, match="Override key not allowed"):
        pipeline.run(
            bundle,
            optimizer_name="meanvar_enhance_index",
            runtime_overrides={"not_allowed_param": 1.0},
        )


def test_factor_neutrality_modes_can_be_disabled_at_runtime() -> None:
    store = ParameterStore()

    resolved, applied = store.resolve(
        "meanvar_enhance_barra_precomputed",
        {
            "industry_neutral_mode": "off",
            "style_neutral_mode": "off",
            "active_weight_abs_max": None,
            "enforce_tracking_error_cap": False,
        },
    )

    assert resolved["industry_neutral_mode"] == "off"
    assert resolved["style_neutral_mode"] == "off"
    assert resolved["active_weight_abs_max"] is None
    assert resolved["enforce_tracking_error_cap"] is False
    assert applied == {
        "industry_neutral_mode": "off",
        "style_neutral_mode": "off",
        "active_weight_abs_max": None,
        "enforce_tracking_error_cap": False,
    }


def test_topn_size_can_be_tuned_at_runtime() -> None:
    resolved, applied = ParameterStore().resolve(
        "topn_long_only_equal_weight",
        {"top_n": 50},
    )

    assert resolved["top_n"] == 50
    assert applied == {"top_n": 50}


@pytest.mark.parametrize(
    "key",
    ["industry_neutral_mode", "style_neutral_mode"],
)
def test_factor_neutrality_mode_override_rejects_unknown_value(key: str) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ParameterStore().resolve(
            "meanvar_enhance_barra_precomputed",
            {key: "unknown"},
        )


def test_trades_follow_chained_target_positions() -> None:
    adapter = MockInputAdapter(n_dates=5, n_assets=60)
    bundle = adapter.build_bundle()
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_absolute_return"
    )

    actual = output.trades["delta_weight"].unstack("asset")
    expected = output.target_positions.diff()
    expected.iloc[0] = output.target_positions.iloc[0] - bundle.prev_positions.iloc[0]
    pd.testing.assert_frame_equal(actual, expected, check_names=False, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        output.summary["turnover"].to_numpy(),
        expected.abs().sum(axis=1).to_numpy() / 2.0,
        atol=1e-9,
    )


def test_auto_smoothing_is_causal() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    columns = [f"A{i}" for i in range(6)]
    stable = pd.DataFrame(
        np.tile(np.arange(6, dtype=float), (5, 1)), index=dates[:5], columns=columns
    )
    volatile = pd.DataFrame(
        [np.arange(6), np.arange(6)[::-1], np.roll(np.arange(6), 3)],
        index=dates[5:], columns=columns, dtype=float,
    )
    smoother = SignalSmoother(topn_n=2, turnover_window=2, turnover_threshold=0.2)
    prefix_smoothed, prefix_diag = smoother.transform(stable)
    full_smoothed, full_diag = smoother.transform(pd.concat([stable, volatile]))

    pd.testing.assert_frame_equal(full_smoothed.loc[stable.index], prefix_smoothed)
    pd.testing.assert_series_equal(
        full_diag.loc[stable.index, "use_ema"], prefix_diag["use_ema"]
    )


def test_infeasible_single_name_cap_fails_explicitly() -> None:
    bundle = MockInputAdapter(n_dates=2, n_assets=12).build_bundle()
    with pytest.raises(ValueError, match="Infeasible position cap"):
        OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
            bundle, optimizer_name="meanvar_enhance_index"
        )


def test_topn_does_not_require_market_and_reports_rule_backend() -> None:
    alpha = MockInputAdapter(n_dates=3, n_assets=60).load_alpha()
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        InputBundle(alpha=alpha), optimizer_name="topn_long_only_equal_weight"
    )
    assert (output.target_positions.gt(0).sum(axis=1) == 50).all()
    assert output.metadata.iloc[0]["solver"] == "none"
    assert output.summary.iloc[0]["turnover"] == pytest.approx(0.5)


def test_non_tradable_assets_are_frozen_at_previous_weights() -> None:
    bundle = MockInputAdapter(n_dates=3, n_assets=60).build_bundle()
    bundle.tradable.iloc[1] = False
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_absolute_return"
    )
    np.testing.assert_allclose(
        output.target_positions.iloc[1].to_numpy(),
        output.target_positions.iloc[0].to_numpy(),
        atol=1e-8,
    )
    assert output.summary.iloc[1]["turnover"] == pytest.approx(0.0, abs=1e-8)


def test_real_adapter_uses_exchange_sessions_and_preserves_alpha_nan(tmp_path: Path) -> None:
    dates = pd.DatetimeIndex(["2024-03-28", "2024-03-29", "2024-04-01"])
    alpha = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [np.nan, 5.0]],
        index=dates,
        columns=["A", "B"],
    )
    adapter = RealInputAdapter(
        alpha_df=alpha,
        data_root=str(tmp_path),
        start_date="2024-03-28",
        end_date="2024-04-01",
    )
    loaded = adapter.load_alpha()
    assert pd.Timestamp("2024-03-29") not in loaded.index  # Good Friday, XNYS closed
    assert pd.isna(loaded.loc[pd.Timestamp("2024-04-01"), "A"])


def test_alpha_coverage_is_checked_before_imputation() -> None:
    bundle = MockInputAdapter(n_dates=3, n_assets=60).build_bundle()
    bundle.alpha.iloc[1, :20] = np.nan
    with pytest.raises(ValueError, match="alpha coverage"):
        OptimizationPipeline().run(
            bundle, optimizer_name="meanvar_absolute_return"
        )


def test_historical_covariance_handles_missing_returns_without_zero_risk() -> None:
    bundle = MockInputAdapter(n_dates=5, n_assets=60).build_bundle()
    bundle.market.iloc[1:, 0] = np.nan
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_absolute_return"
    )
    assert np.isfinite(output.summary["covariance_min_eigenvalue"]).all()
    assert (output.summary["covariance_min_eigenvalue"] > 0).all()
    assert np.isfinite(output.summary["covariance_condition_number"]).all()


def test_summary_contains_economic_and_risk_diagnostics() -> None:
    bundle = MockInputAdapter(n_dates=3, n_assets=60).build_bundle()
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_enhance_hist"
    )
    expected = {
        "expected_return",
        "expected_linear_cost",
        "expected_impact_cost",
        "predicted_portfolio_volatility",
        "predicted_tracking_error",
        "active_share",
        "max_abs_industry_active_exposure",
        "max_abs_style_active_exposure",
        "largest_abs_risk_contribution",
        "covariance_condition_number",
        "max_constraint_violation",
    }
    assert expected.issubset(output.summary.columns)
    assert output.summary["feasible_flag"].all()


def test_asset_level_linear_cost_changes_the_solution() -> None:
    low_cost_bundle = MockInputAdapter(n_dates=1, n_assets=60).build_bundle()
    low_cost_bundle.alpha.iloc[0] = 0.0
    low_cost_bundle.alpha.iloc[0, 0] = 10.0
    low_cost_bundle.metadata.alpha_input_type = "score"

    high_cost_bundle = MockInputAdapter(n_dates=1, n_assets=60).build_bundle()
    high_cost_bundle.alpha.iloc[0] = low_cost_bundle.alpha.iloc[0]
    high_cost_bundle.linear_cost_bps.iloc[0, 0] = 10_000.0

    pipeline = OptimizationPipeline(smoother=SignalSmoother(mode="never"))
    low_cost = pipeline.run(low_cost_bundle, optimizer_name="meanvar_absolute_return")
    high_cost = pipeline.run(high_cost_bundle, optimizer_name="meanvar_absolute_return")

    assert low_cost.target_positions.iloc[0, 0] > high_cost.target_positions.iloc[0, 0] + 0.01
    assert high_cost.summary.iloc[0]["expected_linear_cost"] >= 0.0


def test_solver_fallback_is_revalidated_against_all_constraints() -> None:
    bundle = MockInputAdapter(n_dates=2, n_assets=60).build_bundle()
    bundle.prev_positions.iloc[0] = 0.0
    bundle.prev_positions.iloc[0, 0] = 1.0
    with pytest.raises(Exception, match="fallback_carry_forward violates constraints"):
        OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
            bundle,
            optimizer_name="meanvar_absolute_return",
            runtime_overrides={"solver_name": "NOT_A_SOLVER", "on_solve_failure": "carry_forward"},
        )


def test_historical_optimizer_is_prefix_invariant() -> None:
    bundle = MockInputAdapter(n_dates=6, n_assets=60).build_bundle()
    pipeline = OptimizationPipeline(smoother=SignalSmoother(mode="never"))
    full = pipeline.run(bundle, optimizer_name="meanvar_enhance_hist")

    prefix_bundle = InputBundle(
        alpha=bundle.alpha.iloc[:3],
        market=bundle.market.iloc[:3],
        G=bundle.G.iloc[:3],
        F_mcap=bundle.F_mcap.iloc[:3],
        style=bundle.style.iloc[:3],
        benchmark=bundle.benchmark.iloc[:3],
        prev_positions=bundle.prev_positions.iloc[:3],
        tradable=bundle.tradable.iloc[:3],
        linear_cost_bps=bundle.linear_cost_bps.iloc[:3],
        impact_cost=bundle.impact_cost.iloc[:3],
        metadata=bundle.metadata,
    )
    prefix = pipeline.run(prefix_bundle, optimizer_name="meanvar_enhance_hist")
    pd.testing.assert_frame_equal(
        full.target_positions.iloc[:3], prefix.target_positions, atol=1e-8, rtol=1e-8
    )


def test_barra_precomputed_runs_in_factor_form_without_market() -> None:
    bundle = MockInputAdapter(n_dates=4, n_assets=60).build_bundle()
    factors = list(bundle.F.columns)
    covariance_by_date = {
        date: pd.DataFrame(
            np.eye(len(factors)) * 1e-4,
            index=factors,
            columns=factors,
        )
        for date in bundle.alpha.index
    }
    bundle.factor_cov = pd.concat(
        covariance_by_date, names=["date", "factor_i"]
    )
    bundle.specific_var = pd.DataFrame(
        1e-4, index=bundle.alpha.index, columns=bundle.alpha.columns
    )
    bundle.market = None
    bundle.F_ret = None
    bundle.F_spec = None
    bundle.F_mcap = None

    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, optimizer_name="meanvar_enhance_barra_precomputed"
    )

    assert output.summary["solve_status"].eq("optimal").all()
    assert output.summary["risk_model_type"].eq(
        "barra_precomputed_factor_form"
    ).all()
    assert output.summary["risk_factor_count"].eq(len(factors)).all()
    assert output.summary["feasible_flag"].all()


def _attach_precomputed_risk(bundle: InputBundle, variance: float = 1e-4) -> None:
    factors = list(bundle.F.columns)
    bundle.factor_cov = pd.concat(
        {
            date: pd.DataFrame(
                np.eye(len(factors)) * variance,
                index=factors,
                columns=factors,
            )
            for date in bundle.alpha.index
        },
        names=["date", "factor_i"],
    )
    bundle.specific_var = pd.DataFrame(
        variance, index=bundle.alpha.index, columns=bundle.alpha.columns
    )
    bundle.market = None
    bundle.F_ret = None
    bundle.F_spec = None
    bundle.F_mcap = None


def test_index_scenario_auto_routes_to_precomputed_barra() -> None:
    bundle = MockInputAdapter(n_dates=2, n_assets=60).build_bundle()
    _attach_precomputed_risk(bundle)

    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, scenario="index_enhancement"
    )

    meta = output.metadata.iloc[0]
    assert meta["optimizer_name"] == "meanvar_enhance_barra_precomputed"
    assert meta["extras"]["requested_optimizer_name"] == "scenario:index_enhancement"
    assert "B/F/D" in meta["extras"]["routing_reason"]


def test_tracking_error_hard_cap_is_enforced_and_annualized() -> None:
    bundle = MockInputAdapter(n_dates=2, n_assets=60).build_bundle()
    _attach_precomputed_risk(bundle, variance=2e-4)
    cap = 0.005

    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle,
        optimizer_name="meanvar_enhance_barra_precomputed",
        runtime_overrides={"tracking_error_cap_annual": cap},
    )

    assert (
        output.summary["predicted_tracking_error_annual"] <= cap + 1e-5
    ).all()
    assert (output.summary["tracking_error_cap_violation"] <= 1e-5).all()
    assert output.summary["tracking_error_cap_annual"].eq(cap).all()


def test_precomputed_factor_form_matches_dense_covariance() -> None:
    rng = np.random.default_rng(7)
    exposure = rng.normal(size=(12, 4))
    raw = rng.normal(size=(4, 4))
    factor_cov = raw @ raw.T * 1e-4
    specific_var = rng.uniform(1e-5, 3e-4, size=12)
    active = rng.normal(size=12)
    risk = PrecomputedFactorRisk(
        exposure=exposure,
        factor_cov=factor_cov,
        specific_var=specific_var,
        factor_names=("f0", "f1", "f2", "f3"),
        exposure_date=pd.Timestamp("2024-01-02"),
        covariance_date=pd.Timestamp("2024-01-02"),
        specific_risk_date=pd.Timestamp("2024-01-02"),
    )
    dense = exposure @ factor_cov @ exposure.T + np.diag(specific_var)
    assert risk.variance(active) == pytest.approx(
        float(active @ dense @ active), rel=1e-12, abs=1e-12
    )


def test_risk_unit_conversion_preserves_one_day_covariance() -> None:
    bundle = MockInputAdapter(n_dates=1, n_assets=60).build_bundle()
    _attach_precomputed_risk(bundle, variance=1e-4)
    pipeline = OptimizationPipeline(smoother=SignalSmoother(mode="never"))
    spec = pipeline.router.resolve("meanvar_enhance_barra_precomputed")
    params, _ = pipeline.parameter_store.resolve(spec.name)
    date = bundle.alpha.index[0]
    assets = list(bundle.alpha.columns)

    daily = pipeline.optimizer._build_covariance(
        bundle, spec, date, assets, params
    )
    bundle.factor_cov *= 252.0
    bundle.specific_var *= 252.0
    bundle.metadata.risk_covariance_units = "annual_variance"
    annual = pipeline.optimizer._build_covariance(
        bundle, spec, date, assets, params
    )

    np.testing.assert_allclose(daily.factor_cov, annual.factor_cov, rtol=1e-12)
    np.testing.assert_allclose(daily.specific_var, annual.specific_var, rtol=1e-12)


def _write_barra_fixture(root: Path, assets: list[str]) -> None:
    dates = ["2024-01-02", "2024-01-03"]
    factors = [
        {"id": "style_lncap", "type": "continuous"},
        {"id": "industry_801010", "type": "dummy"},
    ]
    manifest = {
        "product": "barra_lite",
        "version": "test",
        "provider": "pytest",
        "factors": factors,
        "estimation": {"units": "daily_variance"},
        "files": {
            "exposure": "exposure.parquet",
            "factor_cov": "factor_cov.parquet",
            "specific_risk": "specific_risk.parquet",
            # factor_returns is deliberately absent: B/F/D must be sufficient.
        },
    }
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    exposure_rows = []
    specific_rows = []
    covariance_rows = []
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
        for factor_i in ["style_lncap", "industry_801010"]:
            for factor_j in ["style_lncap", "industry_801010"]:
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


def test_barra_adapter_uses_asof_lag_and_needs_only_bfd(tmp_path: Path) -> None:
    assets = [f"A{i:03d}" for i in range(40)]
    _write_barra_fixture(tmp_path, assets)
    decision_date = pd.Timestamp("2024-01-04")
    alpha = pd.DataFrame(
        np.random.default_rng(3).normal(size=(1, len(assets))),
        index=[decision_date],
        columns=assets,
    )
    benchmark = pd.DataFrame(
        1.0 / len(assets), index=[decision_date], columns=assets
    )
    previous = pd.DataFrame(
        1.0 / len(assets),
        index=[pd.Timestamp("2024-01-01")],
        columns=assets,
    )
    tradable = pd.DataFrame(1, index=[decision_date], columns=assets)
    bundle = BarraPrecomputedAdapter(
        risk_root=tmp_path,
        alpha_df=alpha,
        benchmark_df=benchmark,
        prev_positions_df=previous,
        tradable_df=tradable,
        risk_data_lag_periods=1,
        exposure_data_lag_periods=1,
    ).build_bundle()

    assert bundle.F_ret is None
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle, scenario="index_enhancement"
    )
    row = output.summary.iloc[0]
    assert row["risk_covariance_date"] == pd.Timestamp("2024-01-02")
    assert row["risk_exposure_date"] == pd.Timestamp("2024-01-02")
    assert row["specific_risk_date"] == pd.Timestamp("2024-01-02")
    assert row["risk_covariance_units"] == "daily_variance"


def _drop_barra_asset_date(
    root: Path, asset: str, date: str
) -> None:
    for filename in ("exposure.parquet", "specific_risk.parquet"):
        path = root / filename
        frame = pd.read_parquet(path)
        keep = ~(
            frame["asset"].eq(asset)
            & pd.to_datetime(frame["date"]).eq(pd.Timestamp(date))
        )
        frame.loc[keep].to_parquet(path, index=False)


def _build_missing_barra_bundle(
    root: Path, *, missing_asset_tradable: bool
) -> InputBundle:
    assets = [f"A{i:03d}" for i in range(40)]
    _write_barra_fixture(root, assets)
    _drop_barra_asset_date(root, assets[0], "2024-01-02")
    decision_date = pd.Timestamp("2024-01-04")
    alpha = pd.DataFrame(
        np.random.default_rng(9).normal(size=(1, len(assets))),
        index=[decision_date],
        columns=assets,
    )
    benchmark = pd.DataFrame(
        1.0 / len(assets), index=[decision_date], columns=assets
    )
    previous_values = np.full(len(assets), 0.99 / (len(assets) - 1))
    previous_values[0] = 0.01
    previous = pd.DataFrame(
        [previous_values],
        index=[pd.Timestamp("2024-01-01")],
        columns=assets,
    )
    tradable = pd.DataFrame(True, index=[decision_date], columns=assets)
    tradable.loc[decision_date, assets[0]] = missing_asset_tradable
    return BarraPrecomputedAdapter(
        risk_root=root,
        alpha_df=alpha,
        benchmark_df=benchmark,
        prev_positions_df=previous,
        tradable_df=tradable,
        risk_data_lag_periods=1,
        exposure_data_lag_periods=1,
    ).build_bundle()


def test_barra_missing_asset_date_is_not_imputed_and_is_exited(
    tmp_path: Path,
) -> None:
    bundle = _build_missing_barra_bundle(
        tmp_path, missing_asset_tradable=True
    )
    missing_asset = bundle.alpha.columns[0]
    assert not bool(bundle.risk_covered.loc["2024-01-04", missing_asset])
    assert bundle.F.xs("2024-01-02", level="date").loc[
        missing_asset
    ].isna().all()
    assert pd.isna(bundle.specific_var.loc["2024-01-02", missing_asset])

    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle,
        optimizer_name="meanvar_enhance_barra_precomputed",
        runtime_overrides={
            "active_weight_abs_max": None,
            "industry_neutral_mode": "off",
            "style_neutral_mode": "off",
            "enforce_tracking_error_cap": False,
        },
    )

    assert output.target_positions.iloc[0][missing_asset] == pytest.approx(
        0.0, abs=1e-6
    )
    assert output.summary.iloc[0]["risk_coverage_weight"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_untradable_missing_barra_asset_is_carried_forward(
    tmp_path: Path,
) -> None:
    bundle = _build_missing_barra_bundle(
        tmp_path, missing_asset_tradable=False
    )
    missing_asset = bundle.alpha.columns[0]
    output = OptimizationPipeline(smoother=SignalSmoother(mode="never")).run(
        bundle,
        optimizer_name="meanvar_enhance_barra_precomputed",
        runtime_overrides={
            "active_weight_abs_max": None,
            "industry_neutral_mode": "off",
            "style_neutral_mode": "off",
            "enforce_tracking_error_cap": False,
        },
    )

    assert output.target_positions.iloc[0][missing_asset] == pytest.approx(
        0.01, abs=1e-5
    )
    assert output.summary.iloc[0]["uncovered_frozen_weight"] == pytest.approx(
        0.01, abs=1e-5
    )


def test_barra_adapter_rejects_undeclared_risk_units(tmp_path: Path) -> None:
    assets = [f"A{i:03d}" for i in range(40)]
    _write_barra_fixture(tmp_path, assets)
    manifest_path = tmp_path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["estimation"]["units"] = "mystery_variance"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    date = pd.Timestamp("2024-01-03")
    frame = pd.DataFrame(1.0, index=[date], columns=assets)
    benchmark = pd.DataFrame(1.0 / len(assets), index=[date], columns=assets)

    with pytest.raises(ValueError, match="estimation.units"):
        BarraPrecomputedAdapter(
            risk_root=tmp_path,
            alpha_df=frame,
            benchmark_df=benchmark,
            prev_positions_df=benchmark,
            tradable_df=frame,
        ).build_bundle()
