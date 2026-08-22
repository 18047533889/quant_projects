import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from vectorbt_qs.mvp.constraints.ashare import (
    apply_ashare_constraints,
    apply_t1_constraint,
    resolve_limit_check_prices,
)
from vectorbt_qs.mvp.constraints.us_stock import apply_delisting, apply_ssr_filter
from vectorbt_qs.mvp.data import adapter
from vectorbt_qs.mvp.engine import runner
from vectorbt_qs.mvp.engine.execution import (
    ExecutionCosts,
    plan_ashare_orders,
    plan_ashare_orders_python,
    schedule_target_signals,
)
from vectorbt_qs.mvp.visualization import (
    available_plotly_charts,
    build_nav_curves,
    build_plotly_chart,
    build_plotly_dashboard,
    export_plotly_dashboard,
)


class ConstraintRegressionTests(unittest.TestCase):
    def test_ashare_rejects_limit_orders_and_charges_sell_tax(self):
        index = pd.date_range("2026-01-01", periods=4)
        close = pd.DataFrame({"X": [10.0, 11.0, 9.0, 10.0]}, index=index)
        weights = pd.DataFrame({"X": [0.5, 1.0, 0.0, 0.0]}, index=index)
        high_limit = pd.DataFrame({"X": [11.0, 11.0, 10.0, 11.0]}, index=index)
        low_limit = pd.DataFrame({"X": [9.0, 9.0, 9.0, 9.0]}, index=index)

        result = apply_ashare_constraints(
            close,
            weights,
            pd.DataFrame(False, index=index, columns=["X"]),
            high_limit,
            low_limit,
        )

        self.assertEqual(result["target_weights"].iloc[0, 0], 0.5)
        self.assertTrue(pd.isna(result["target_weights"].iloc[1, 0]))
        self.assertTrue(pd.isna(result["target_weights"].iloc[2, 0]))
        self.assertEqual(result["target_weights"].iloc[3, 0], 0.0)
        self.assertAlmostEqual(result["fees"].iloc[3, 0], 0.00076)

    def test_ssr_keeps_existing_short_and_blocks_increase(self):
        index = pd.date_range("2026-01-01", periods=3)
        close = pd.DataFrame({"X": [100.0, 89.0, 90.0]}, index=index)
        targets = pd.DataFrame({"X": [-0.2, -0.3, -0.4]}, index=index)

        actual = apply_ssr_filter(close, targets)

        self.assertEqual(actual["X"].tolist(), [-0.2, -0.2, -0.2])

    def test_ssr_allows_long_reduction_and_stops_long_to_short_at_zero(self):
        index = pd.date_range("2026-01-01", periods=3)
        close = pd.DataFrame({"X": [100.0, 89.0, 90.0], "Y": [100.0, 89.0, 90.0]}, index=index)
        targets = pd.DataFrame(
            {"X": [0.5, 0.4, 0.0], "Y": [0.5, -0.2, -0.3]},
            index=index,
        )

        actual = apply_ssr_filter(close, targets)

        self.assertEqual(actual["X"].tolist(), [0.5, 0.4, 0.0])
        self.assertEqual(actual["Y"].tolist(), [0.5, 0.0, 0.0])

    def test_delisting_uses_first_available_row_after_effective_date(self):
        index = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
        weights = pd.DataFrame({"X": [0.5, 0.5, 0.5]}, index=index)
        delist_info = pd.DataFrame({"ticker": ["X"], "delist_date": ["2025-01-04"]})

        actual = apply_delisting(weights, weights, delist_info)

        self.assertEqual(actual["X"].tolist(), [0.5, 0.0, 0.0])

    def test_t1_helper_delays_targets(self):
        index = pd.date_range("2026-01-01", periods=3)
        targets = pd.DataFrame({"X": [0.2, 0.0, 0.3]}, index=index)
        previous = pd.DataFrame({"X": [0.1]}, index=[index[0] - pd.Timedelta(days=1)])

        actual = apply_t1_constraint(targets, previous)

        self.assertEqual(actual["X"].tolist(), [0.1, 0.2, 0.0])


class RunnerRegressionTests(unittest.TestCase):
    @staticmethod
    def market_data(index, price=30.0):
        close = pd.DataFrame({"X": np.full(len(index), price)}, index=index)
        return {
            "close": close,
            "open": close + 1.0,
            "vwap": close + 0.5,
            "high": close * 1.05,
            "low": close * 0.95,
            "is_suspend": pd.DataFrame(False, index=index, columns=["X"]),
            "high_limit": close * 1.1,
            "low_limit": close * 0.9,
        }

    def test_weekly_resample_keeps_actual_last_trading_date(self):
        index = pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02"])
        weights = pd.DataFrame({"X": range(4)}, index=index, dtype=float)

        actual = runner.resample_target_weights(weights, "1W")

        self.assertEqual(actual.index.tolist(), [pd.Timestamp("2026-04-02")])

    def test_fast_uses_next_session_fractional_shares_and_zero_cost(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        weights = pd.DataFrame({"X": [0.5]}, index=index[:1])
        with (
            patch.object(runner, "load_ashare_calendar", return_value=index),
            patch.object(runner, "load_market_data", return_value=self.market_data(index)),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                weights,
                config={
                    "execution_mode": "fast",
                    "init_cash": 10_000.0,
                    "fees": 0.0,
                    "slippage": 0.0,
                },
            )

        order = portfolio.orders.records_readable.iloc[0]
        self.assertEqual(order["Timestamp"], index[1])
        self.assertNotEqual(order["Size"] % 100, 0.0)
        self.assertEqual(order["Fees"], 0.0)
        self.assertEqual(portfolio._qs_fast_contract, "factor_research")

    def test_fast_selects_adjusted_open(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        weights = pd.DataFrame({"X": [0.5]}, index=index[:1])
        with (
            patch.object(runner, "load_ashare_calendar", return_value=index),
            patch.object(runner, "load_market_data", return_value=self.market_data(index)),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                weights,
                config={
                    "execution_mode": "fast",
                    "init_cash": 100_000.0,
                    "price_type": "open_adj",
                    "fees": 0.0,
                    "slippage": 0.0,
                },
            )

        self.assertEqual(portfolio.orders.records_readable.iloc[0]["Price"], 31.0)

    def test_fast_rejects_nonzero_friction_and_lot_size(self):
        index = pd.to_datetime(["2026-01-05"])
        weights = pd.DataFrame({"X": [0.5]}, index=index)

        with self.assertRaisesRegex(ValueError, "fees=0"):
            runner.run_backtest(
                "ashare", weights,
                config={"execution_mode": "fast", "fees": 0.001},
            )
        with self.assertRaisesRegex(ValueError, "slippage=0"):
            runner.run_backtest(
                "ashare", weights,
                config={"execution_mode": "fast", "slippage": 0.001},
            )
        with self.assertRaisesRegex(ValueError, "分数股"):
            runner.run_backtest(
                "ashare", weights,
                config={"execution_mode": "fast", "size_granularity": 100},
            )

    def test_ashare_rejects_negative_weights(self):
        index = pd.date_range("2026-01-01", periods=2)
        weights = pd.DataFrame({"X": [-0.1, -0.1]}, index=index)
        with self.assertRaisesRegex(ValueError, "不支持负权重"):
            runner.run_backtest(
                "ashare",
                weights,
                config={"execution_mode": "fast", "execution_lag": 0},
            )

    def test_grid_rejects_label_length_mismatch(self):
        index = pd.date_range("2026-01-01", periods=2)
        weights = pd.DataFrame({"X": [0.1, 0.1]}, index=index)
        with self.assertRaisesRegex(ValueError, "长度必须一致"):
            runner.run_backtest_grid("ashare", [weights], ["one", "two"])

    def test_report_uses_configured_benchmark(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        weights = pd.DataFrame({"X": [0.5, 0.5]}, index=index[:2])
        benchmark = pd.Series([100.0, 110.0, 121.0], index=index, name="B")
        with (
            patch.object(runner, "load_ashare_calendar", return_value=index),
            patch.object(runner, "load_market_data", return_value=self.market_data(index)),
            patch.object(runner, "load_benchmark_close", return_value=benchmark),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                weights,
                config={
                    "execution_mode": "fast",
                    "benchmark_index": "B",
                    "fees": 0.0,
                    "slippage": 0.0,
                },
            )

        report = runner.portfolio_report(portfolio)
        self.assertAlmostEqual(report["Benchmark Return [%]"], 21.0)

    def test_report_uses_252_trading_days_and_explicit_risk_free_rate(self):
        index = pd.date_range("2026-01-05", periods=4, freq="B")
        daily_returns = pd.Series(
            [0.01, -0.005, 0.002, 0.004],
            index=index,
        )

        class FakePortfolio:
            _qs_performance_year_days = 252
            _qs_risk_free_rate = 0.03

            def __init__(self):
                self.settings = None

            def stats(self, settings=None):
                self.settings = settings
                return pd.Series(dtype=float)

            def returns(self):
                return daily_returns

        portfolio = FakePortfolio()
        report = runner.portfolio_report(portfolio)
        daily_rf = (1.03 ** (1.0 / 252.0)) - 1.0
        expected_sharpe = (
            (daily_returns - daily_rf).mean()
            / (daily_returns - daily_rf).std(ddof=1)
            * np.sqrt(252)
        )

        self.assertEqual(portfolio.settings["year_freq"], "252 days")
        self.assertAlmostEqual(portfolio.settings["risk_free"], daily_rf)
        self.assertAlmostEqual(report["Sharpe Ratio"], expected_sharpe)

    def test_fast_weekly_signal_executes_next_session_on_daily_timeline(self):
        calendar = pd.to_datetime(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
        )
        weights = pd.DataFrame({"X": [0.5] * 5}, index=calendar[:5])
        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=self.market_data(calendar)),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                weights,
                config={"execution_mode": "fast", "freq": "1W", "slippage": 0.0},
            )

        self.assertEqual(
            portfolio.orders.records_readable.iloc[0]["Timestamp"],
            pd.Timestamp("2026-01-12"),
        )
        self.assertEqual(portfolio.wrapper.index.tolist(), calendar.tolist())

    def test_fast_tradeability_policy_is_explicit(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06"])
        weights = pd.DataFrame({"X": [0.5]}, index=calendar[:1])
        data = self.market_data(calendar)
        data["high_limit"].loc[calendar[1], "X"] = data["open"].loc[calendar[1], "X"]

        def run(policy):
            with (
                patch.object(runner, "load_ashare_calendar", return_value=calendar),
                patch.object(runner, "load_market_data", return_value=data),
            ):
                return runner.run_backtest(
                    "ashare",
                    weights,
                    config={
                        "execution_mode": "fast",
                        "fast_tradeability": policy,
                        "slippage": 0.0,
                    },
                )

        ignored = run("ignore")
        approximate = run("approximate")
        self.assertEqual(len(ignored.orders.records), 1)
        self.assertEqual(len(approximate.orders.records), 0)
        self.assertEqual(ignored._qs_fast_tradeability, "ignore")
        self.assertEqual(approximate._qs_fast_tradeability, "approximate")

    def test_fast_strict_limit_check_uses_intraday_extreme_with_tolerance(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06"])
        weights = pd.DataFrame({"X": [0.5]}, index=calendar[:1])
        data = self.market_data(calendar)
        data["high"].loc[calendar[1], "X"] = (
            data["high_limit"].loc[calendar[1], "X"] * (1.0 - 5e-6)
        )

        def run(mode):
            with (
                patch.object(runner, "load_ashare_calendar", return_value=calendar),
                patch.object(runner, "load_market_data", return_value=data),
            ):
                return runner.run_backtest(
                    "ashare",
                    weights,
                    config={
                        "execution_mode": "fast",
                        "fast_tradeability": "approximate",
                        "limit_check_mode": mode,
                        "limit_price_rtol": 1e-5,
                        "slippage": 0.0,
                    },
                )

        execution = run("execution")
        strict = run("strict")
        self.assertEqual(len(execution.orders.records), 1)
        self.assertEqual(len(strict.orders.records), 0)
        self.assertEqual(strict._qs_limit_check_mode, "strict")

    def test_accurate_strict_limit_check_blocks_vwap_fill_after_limit_touch(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06"])
        weights = pd.DataFrame({"X": [0.5]}, index=calendar[:1])
        data = self.market_data(calendar)
        data["high"].loc[calendar[1], "X"] = data["high_limit"].loc[calendar[1], "X"]

        def run(mode):
            with (
                patch.object(runner, "load_ashare_calendar", return_value=calendar),
                patch.object(runner, "load_market_data", return_value=data),
            ):
                return runner.run_backtest(
                    "ashare",
                    weights,
                    config={
                        "execution_mode": "accurate",
                        "price_type": "vwap_adj",
                        "limit_check_mode": mode,
                        "slippage": 0.0,
                    },
                )

        execution = run("execution")
        strict = run("strict")
        self.assertEqual(len(execution.orders.records), 1)
        self.assertEqual(len(strict.orders.records), 0)
        self.assertEqual(strict._qs_execution_log.iloc[0]["reason"], "limit_up")
        self.assertEqual(strict._qs_limit_check_mode, "strict")

    def test_limit_check_configuration_rejects_invalid_mode_and_tolerance(self):
        index = pd.to_datetime(["2026-01-05"])
        frame = pd.DataFrame({"X": [10.0]}, index=index)
        upper, lower = resolve_limit_check_prices(
            frame,
            frame * 1.1,
            frame * 0.9,
            mode="strict",
            high=frame * 1.1 * (1.0 - 5e-6),
            low=frame * 0.9 * (1.0 + 5e-6),
            rtol=1e-5,
        )
        pd.testing.assert_frame_equal(upper, frame)
        pd.testing.assert_frame_equal(lower, frame)

        with self.assertRaisesRegex(ValueError, "limit_check_mode"):
            resolve_limit_check_prices(frame, frame * 1.1, frame * 0.9, mode="bad")
        with self.assertRaisesRegex(ValueError, "limit_price_rtol"):
            resolve_limit_check_prices(
                frame,
                frame * 1.1,
                frame * 0.9,
                mode="execution",
                rtol=-1.0,
            )

    def test_fast_returns_are_invariant_to_initial_cash(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        close = pd.DataFrame({"X": [10.0, 11.0, 12.0]}, index=calendar)
        data = {
            "close": close,
            "open": close.copy(),
            "is_suspend": pd.DataFrame(False, index=calendar, columns=["X"]),
            "high_limit": close * 1.1,
            "low_limit": close * 0.9,
        }
        weights = pd.DataFrame({"X": [0.5, 0.5]}, index=calendar[:2])

        def run(init_cash):
            with (
                patch.object(runner, "load_ashare_calendar", return_value=calendar),
                patch.object(runner, "load_market_data", return_value=data),
            ):
                return runner.run_backtest(
                    "ashare",
                    weights,
                    config={
                        "execution_mode": "fast",
                        "init_cash": init_cash,
                        "slippage": 0.0,
                    },
                )

        small = run(10_000.0)
        large = run(10_000_000.0)
        self.assertAlmostEqual(float(small.total_return()), float(large.total_return()), places=12)

    def test_fast_repeated_zero_target_is_a_noop_after_liquidation(self):
        calendar = pd.to_datetime(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        )
        weights = pd.DataFrame({"X": [0.5, 0.0, 0.0]}, index=calendar[:3])
        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=self.market_data(calendar)),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                weights,
                config={"execution_mode": "fast", "slippage": 0.0},
            )

        self.assertEqual(len(portfolio.orders.records), 2)


class AccurateExecutionTests(unittest.TestCase):
    @staticmethod
    def frames(index, columns=("A", "B"), prices=(10.0, 20.0)):
        close = pd.DataFrame(
            np.tile(np.asarray(prices, dtype=float), (len(index), 1)),
            index=index,
            columns=list(columns),
        )
        return {
            "close": close,
            "open": close.copy(),
            "is_suspend": pd.DataFrame(False, index=index, columns=columns),
            "high_limit": close * 1.1,
            "low_limit": close * 0.9,
        }

    def assert_plans_equal(self, expected, actual):
        pd.testing.assert_frame_equal(expected.order_size, actual.order_size)
        pd.testing.assert_frame_equal(expected.fees, actual.fees)
        pd.testing.assert_frame_equal(expected.fixed_fees, actual.fixed_fees)
        np.testing.assert_array_equal(expected.call_seq, actual.call_seq)
        pd.testing.assert_frame_equal(expected.log, actual.log)
        self.assertEqual(expected.final_cash, actual.final_cash)
        pd.testing.assert_series_equal(expected.final_shares, actual.final_shares)
        pd.testing.assert_frame_equal(
            expected.real_holdings,
            actual.real_holdings,
        )
        pd.testing.assert_frame_equal(
            expected.cash_deposits,
            actual.cash_deposits,
        )
        pd.testing.assert_frame_equal(
            expected.asset_deposits,
            actual.asset_deposits,
        )

    def plan_with_both_engines(
        self,
        data,
        targets,
        *,
        init_cash=100_000.0,
        costs=None,
        slippage=0.0,
        **planner_kwargs,
    ):
        if costs is None:
            costs = ExecutionCosts()
        args = (
            data["close"],
            data["open"],
            targets,
            data["is_suspend"],
            data["high_limit"],
            data["low_limit"],
        )
        kwargs = {
            "init_cash": init_cash,
            "costs": costs,
            "slippage": slippage,
            **planner_kwargs,
        }
        expected = plan_ashare_orders_python(*args, **kwargs)
        actual = plan_ashare_orders(*args, planner_engine="numba", **kwargs)
        self.assert_plans_equal(expected, actual)
        return actual

    def test_planner_uses_raw_prices_and_real_board_lots(self):
        index = pd.to_datetime(["2026-01-05"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5]}, index=index)
        factor = pd.DataFrame({"A": [100.0]}, index=index)

        plan = self.plan_with_both_engines(
            data,
            targets,
            costs=ExecutionCosts(
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
                minimum_commission=0.0,
            ),
            adjustment_factor=factor,
        )

        self.assertEqual(plan.order_size.loc[index[0], "A"], 5_000.0)
        self.assertEqual(plan.final_shares["A"], 5_000.0)
        self.assertEqual(
            plan.order_size.loc[index[0], "A"] * data["open"].loc[index[0], "A"],
            50_000.0,
        )

    def test_cash_dividend_changes_cash_not_real_share_count(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        data["close"].loc[index[1], "A"] = 9.0
        data["open"].loc[index[1], "A"] = 9.0
        data["high_limit"].loc[index[1], "A"] = 9.9
        data["low_limit"].loc[index[1], "A"] = 8.1
        targets = pd.DataFrame({"A": [0.5, np.nan]}, index=index)
        factor = pd.DataFrame({"A": [1.0, 10.0 / 9.0]}, index=index)
        cash_dividend = pd.DataFrame({"A": [0.0, 1.0]}, index=index)

        plan = self.plan_with_both_engines(
            data,
            targets,
            costs=ExecutionCosts(
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
                minimum_commission=0.0,
            ),
            adjustment_factor=factor,
            cash_dividend=cash_dividend,
        )

        self.assertEqual(plan.order_size.loc[index[0], "A"], 5_000.0)
        self.assertTrue(pd.isna(plan.order_size.loc[index[1], "A"]))
        self.assertEqual(plan.final_shares["A"], 5_000.0)
        self.assertEqual(plan.final_cash, 55_000.0)

    def test_stock_distribution_changes_real_shares_without_an_order(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5, np.nan]}, index=index)
        multiplier = pd.DataFrame({"A": [1.0, 1.2]}, index=index)

        plan = self.plan_with_both_engines(
            data,
            targets,
            costs=ExecutionCosts(minimum_commission=0.0),
            share_multiplier=multiplier,
        )

        self.assertTrue(pd.isna(plan.order_size.loc[index[1], "A"]))
        self.assertEqual(plan.final_shares["A"], 6_000.0)

    def test_final_slipped_price_must_remain_inside_daily_range(self):
        index = pd.to_datetime(["2026-01-05"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5]}, index=index)
        high = pd.DataFrame({"A": [10.005]}, index=index)
        low = pd.DataFrame({"A": [9.5]}, index=index)

        plan = self.plan_with_both_engines(
            data,
            targets,
            costs=ExecutionCosts(minimum_commission=0.0),
            slippage=0.001,
            high=high,
            low=low,
        )

        self.assertEqual(plan.final_shares["A"], 0.0)
        self.assertEqual(plan.log.iloc[0]["status"], "blocked")
        self.assertEqual(plan.log.iloc[0]["reason"], "price_bound")

    def test_volume_participation_caps_fills_and_retries_residual(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5, 0.5]}, index=index)
        volume = pd.DataFrame({"A": [10_000.0, 10_000.0]}, index=index)

        plan = self.plan_with_both_engines(
            data,
            targets,
            costs=ExecutionCosts(
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
                minimum_commission=0.0,
            ),
            volume=volume,
            max_participation_rate=0.10,
        )

        self.assertEqual(plan.order_size.loc[index[0], "A"], 1_000.0)
        self.assertEqual(plan.order_size.loc[index[1], "A"], 1_000.0)
        self.assertEqual(plan.final_shares["A"], 2_000.0)
        self.assertEqual(
            plan.log["status"].tolist(),
            ["partial_volume", "partial_volume"],
        )
        self.assertTrue((plan.log["reason"] == "volume").all())

    def test_numba_planner_matches_python_oracle_on_execution_edge_cases(self):
        index = pd.date_range("2026-01-05", periods=6, freq="B")
        columns = ("D", "B", "A", "C")
        data = self.frames(index, columns=columns, prices=(7.0, 11.0, 13.0, 17.0))
        data["is_suspend"].loc[index[1], "D"] = True
        data["open"].loc[index[2], "C"] = data["high_limit"].loc[index[2], "C"]
        data["open"].loc[index[3], "B"] = data["low_limit"].loc[index[3], "B"]
        targets = pd.DataFrame(
            [
                [0.26, 0.25, 0.24, 0.25],
                [0.00, 0.45, 0.35, 0.20],
                [0.10, 0.20, 0.20, 0.50],
                [0.25, 0.00, 0.50, 0.25],
                [np.nan, 0.25, 0.25, 0.25],
                [0.00, 0.00, 0.00, 0.00],
            ],
            index=index,
            columns=columns,
        )
        # Numeric inputs may have a different physical column order; the
        # accelerated path must retain the Python planner's label semantics.
        for key in ("open", "is_suspend", "high_limit", "low_limit"):
            data[key] = data[key].reindex(columns=list(reversed(columns)))
        self.plan_with_both_engines(
            data,
            targets,
            init_cash=100_037.0,
            costs=ExecutionCosts(minimum_commission=5.0),
            slippage=0.001,
        )

    def test_vendored_vectorbt_accounts_for_corporate_action_flows(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        close = pd.DataFrame({"A": [10.0, 9.0]}, index=index)
        size = pd.DataFrame({"A": [100.0, np.nan]}, index=index)
        cash_deposits = pd.DataFrame({"A": [0.0, 100.0]}, index=index)
        asset_deposits = pd.DataFrame({"A": [0.0, 20.0]}, index=index)

        portfolio = runner.vbt.Portfolio.from_orders(
            close=close,
            size=size,
            size_type="amount",
            price=close,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
            cash_deposits=cash_deposits,
            asset_deposits=asset_deposits,
            init_cash=2_000.0,
            cash_sharing=True,
            direction="longonly",
            freq="1D",
        )

        self.assertEqual(portfolio.cash().iloc[-1], 1_100.0)
        self.assertEqual(portfolio.assets().iloc[-1, 0], 120.0)
        self.assertEqual(portfolio.value().iloc[-1], 2_180.0)
        self.assertEqual(float(portfolio.total_profit()), 180.0)
        restored = runner.vbt.Portfolio.loads(portfolio.dumps())
        pd.testing.assert_series_equal(restored.value(), portfolio.value())

    def test_numba_planner_matches_python_oracle_on_deterministic_random_path(self):
        rng = np.random.default_rng(20260723)
        index = pd.date_range("2025-01-02", periods=40, freq="B")
        columns = tuple(f"S{i:02d}" for i in range(10))
        base_prices = rng.uniform(5.0, 80.0, len(columns))
        prices = base_prices * np.exp(
            np.cumsum(rng.normal(0.0, 0.015, (len(index), len(columns))), axis=0)
        )
        close = pd.DataFrame(prices, index=index, columns=columns)
        data = {
            "close": close,
            "open": close.copy(),
            "is_suspend": pd.DataFrame(False, index=index, columns=columns),
            "high_limit": close * 1.1,
            "low_limit": close * 0.9,
        }
        raw = rng.uniform(0.0, 1.0, close.shape)
        raw[rng.uniform(size=close.shape) < 0.55] = 0.0
        row_sums = raw.sum(axis=1)
        weights = raw / np.maximum(row_sums[:, None], 1.0) * 0.97
        targets = pd.DataFrame(weights, index=index, columns=columns)

        self.plan_with_both_engines(
            data,
            targets,
            init_cash=2_000_003.0,
            costs=ExecutionCosts(
                commission=0.00025,
                stamp_tax=0.0005,
                transfer_fee=0.00001,
                minimum_commission=5.0,
            ),
            slippage=0.0007,
        )

    def test_numba_planner_preserves_python_exception_diagnostics(self):
        index = pd.date_range("2026-01-05", periods=2, freq="B")
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5, 0.0]}, index=index)
        data["open"].loc[index[1], "A"] = np.nan
        args = (
            data["close"],
            data["open"],
            targets,
            data["is_suspend"],
            data["high_limit"],
            data["low_limit"],
        )
        kwargs = {
            "init_cash": 100_000.0,
            "costs": ExecutionCosts(minimum_commission=0.0),
        }

        with self.assertRaises(ValueError) as python_error:
            plan_ashare_orders_python(*args, **kwargs)
        with self.assertRaises(ValueError) as numba_error:
            plan_ashare_orders(*args, planner_engine="numba", **kwargs)
        self.assertEqual(str(python_error.exception), str(numba_error.exception))

    def test_close_signal_executes_at_next_exchange_session(self):
        calendar = pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"])
        targets = pd.DataFrame({"A": [0.5]}, index=calendar[:1])

        actual = schedule_target_signals(
            targets,
            calendar,
            freq="1D",
            delay_sessions=1,
            signal_time="close",
        )

        self.assertEqual(actual.index.tolist(), [pd.Timestamp("2026-01-12")])
        with self.assertRaisesRegex(ValueError, "不能在同一交易日成交"):
            schedule_target_signals(
                targets,
                calendar,
                freq="1D",
                delay_sessions=0,
                signal_time="close",
            )

    def test_accurate_mode_rejects_invalid_weight_contract_and_cost_config(self):
        index = pd.to_datetime(["2026-01-05"])
        overweight = pd.DataFrame([[0.6, 0.5]], index=index, columns=["A", "B"])
        with self.assertRaisesRegex(ValueError, "权重之和超过 1"):
            runner.run_backtest("ashare", overweight)

        with self.assertRaisesRegex(ValueError, "不能与"):
            ExecutionCosts.from_config(
                {"costs": {"commission": 0.0}, "fees": 0.0}
            )
        with self.assertRaisesRegex(ValueError, "非零 fixed_fees"):
            ExecutionCosts.from_config({"fixed_fees": 1.0})

    def test_stateful_planner_applies_lots_directional_fees_and_sell_first(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        data = self.frames(index)
        targets = pd.DataFrame(
            [[0.5, 0.5], [0.0, 1.0]],
            index=index,
            columns=["A", "B"],
        )
        costs = ExecutionCosts(
            commission=0.00025,
            stamp_tax=0.0005,
            transfer_fee=0.00001,
            minimum_commission=5.0,
        )

        plan = plan_ashare_orders(
            data["close"],
            data["open"],
            targets,
            data["is_suspend"],
            data["high_limit"],
            data["low_limit"],
            init_cash=100_000.0,
            costs=costs,
        )

        orders = plan.order_size.stack().rename("size")
        self.assertTrue((orders.abs() % 100 == 0).all())
        self.assertLess(plan.call_seq[1].tolist().index(0), plan.call_seq[1].tolist().index(1))
        sell_fee = plan.fees.loc[index[1], "A"]
        buy_fee = plan.fees.loc[index[1], "B"]
        self.assertGreater(sell_fee, buy_fee)
        self.assertGreaterEqual(plan.final_cash, 0.0)

    def test_suspension_and_price_limits_block_orders_without_changing_state(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        data = self.frames(index, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame({"A": [0.5, 0.0, 0.0]}, index=index)
        data["is_suspend"].loc[index[1], "A"] = True
        data["open"].loc[index[2], "A"] = data["low_limit"].loc[index[2], "A"]

        plan = plan_ashare_orders(
            data["close"],
            data["open"],
            targets,
            data["is_suspend"],
            data["high_limit"],
            data["low_limit"],
            init_cash=100_000.0,
            costs=ExecutionCosts(minimum_commission=0.0),
        )

        self.assertEqual(plan.final_shares["A"], 5_000.0)
        blocked = plan.log.loc[plan.log["status"] == "blocked", "reason"].tolist()
        self.assertEqual(blocked, ["suspend", "limit_down"])

    def test_zero_weight_unlisted_symbol_does_not_require_market_data(self):
        index = pd.to_datetime(["2026-01-05"])
        close = pd.DataFrame([[10.0, np.nan]], index=index, columns=["A", "FUTURE"])
        targets = pd.DataFrame([[0.5, 0.0]], index=index, columns=close.columns)
        suspend = pd.DataFrame([[False, np.nan]], index=index, columns=close.columns)

        plan = plan_ashare_orders(
            close,
            close,
            targets,
            suspend,
            close * 1.1,
            close * 0.9,
            init_cash=100_000.0,
            costs=ExecutionCosts(minimum_commission=0.0),
        )

        self.assertEqual(plan.final_shares["FUTURE"], 0.0)
        self.assertGreater(plan.final_shares["A"], 0.0)

    def test_cash_allocation_is_invariant_to_input_column_order(self):
        index = pd.to_datetime(["2026-01-05"])

        def run(columns):
            data = self.frames(index, columns=columns, prices=(10.0, 10.0))
            targets = pd.DataFrame([[0.5, 0.5]], index=index, columns=columns)
            return plan_ashare_orders(
                data["close"], data["open"], targets,
                data["is_suspend"], data["high_limit"], data["low_limit"],
                init_cash=10_050.0,
                costs=ExecutionCosts(minimum_commission=5.0),
            )

        forward = run(("A", "B")).final_shares.sort_index()
        reverse = run(("B", "A")).final_shares.sort_index()
        pd.testing.assert_series_equal(forward, reverse)

    def test_runner_accurate_mode_uses_daily_timeline_and_adjusted_open(self):
        calendar = pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"])
        data = self.frames(calendar, columns=("A",), prices=(10.0,))
        data["open"].loc[:, "A"] = [10.0, 11.0, 12.0]
        data["high_limit"].loc[:, "A"] = 20.0
        targets = pd.DataFrame({"A": [0.5]}, index=calendar[:1])

        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=data),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                targets,
                config={
                    "init_cash": 100_000.0,
                    "slippage": 0.0,
                    "costs": {
                        "commission": 0.0,
                        "stamp_tax": 0.0,
                        "transfer_fee": 0.0,
                        "minimum_commission": 0.0,
                    },
                },
            )

        order = portfolio.orders.records_readable.iloc[0]
        self.assertEqual(order["Timestamp"], pd.Timestamp("2026-01-12"))
        self.assertEqual(order["Price"], 11.0)
        self.assertEqual(portfolio.wrapper.index.tolist(), calendar[:2].tolist())
        self.assertEqual(portfolio._qs_execution_mode, "accurate")
        self.assertEqual(portfolio._qs_planner_engine, "numba")

    def test_runner_keeps_real_share_orders_and_raw_execution_prices(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06"])
        raw = pd.DataFrame({"A": [10.0, 10.0]}, index=calendar)
        factor = pd.DataFrame({"A": [100.0, 100.0]}, index=calendar)
        adjusted = raw * factor
        data = {
            "close": adjusted,
            "open": adjusted,
            "high": adjusted * 1.05,
            "low": adjusted * 0.95,
            "high_limit": adjusted * 1.1,
            "low_limit": adjusted * 0.9,
            "raw_close": raw,
            "raw_open": raw,
            "raw_high": raw * 1.05,
            "raw_low": raw * 0.95,
            "raw_high_limit": raw * 1.1,
            "raw_low_limit": raw * 0.9,
            "factor": factor,
            "is_suspend": pd.DataFrame(False, index=calendar, columns=["A"]),
        }
        targets = pd.DataFrame({"A": [0.5]}, index=calendar[:1])

        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=data),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                targets,
                config={
                    "init_cash": 100_000.0,
                    "slippage": 0.0,
                    "costs": {
                        "commission": 0.0,
                        "stamp_tax": 0.0,
                        "transfer_fee": 0.0,
                        "minimum_commission": 0.0,
                    },
                },
            )

        execution_date = calendar[1]
        self.assertEqual(
            portfolio._qs_real_order_size.loc[execution_date, "A"],
            5_000.0,
        )
        order = portfolio.orders.records_readable.iloc[0]
        self.assertEqual(order["Size"], 5_000.0)
        self.assertEqual(order["Price"], 10.0)
        self.assertEqual(order["Size"] * order["Price"], 50_000.0)

    def test_runner_turns_cash_dividend_into_spendable_cash_without_fake_shares(self):
        calendar = pd.to_datetime(
            ["2026-01-05", "2026-01-06", "2026-01-07"]
        )
        raw = pd.DataFrame({"A": [10.0, 10.0, 9.0]}, index=calendar)
        factor = pd.DataFrame(
            {"A": [1.0, 1.0, 10.0 / 9.0]},
            index=calendar,
        )
        adjusted = raw * factor
        data = {
            "close": adjusted,
            "open": adjusted,
            "high": adjusted * 1.05,
            "low": adjusted * 0.95,
            "high_limit": adjusted * 1.1,
            "low_limit": adjusted * 0.9,
            "raw_close": raw,
            "raw_open": raw,
            "raw_high": raw * 1.05,
            "raw_low": raw * 0.95,
            "raw_high_limit": raw * 1.1,
            "raw_low_limit": raw * 0.9,
            "factor": factor,
            "cash_dividend": pd.DataFrame(
                {"A": [0.0, 0.0, 1.0]},
                index=calendar,
            ),
            "share_multiplier": pd.DataFrame(
                {"A": [1.0, 1.0, 1.0]},
                index=calendar,
            ),
            "is_suspend": pd.DataFrame(False, index=calendar, columns=["A"]),
        }
        targets = pd.DataFrame(
            {"A": [0.5, np.nan]},
            index=calendar[:2],
        )

        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=data),
        ):
            portfolio = runner.run_backtest(
                "ashare",
                targets,
                config={
                    "init_cash": 100_000.0,
                    "slippage": 0.0,
                    "costs": {
                        "commission": 0.0,
                        "stamp_tax": 0.0,
                        "transfer_fee": 0.0,
                        "minimum_commission": 0.0,
                    },
                },
            )

        self.assertEqual(
            portfolio._qs_real_holdings.loc[calendar[2], "A"],
            5_000.0,
        )
        self.assertTrue(
            pd.isna(portfolio._qs_real_order_size.loc[calendar[2], "A"])
        )
        self.assertEqual(
            portfolio._qs_cash_dividend_deposits.loc[calendar[2], "A"],
            5_000.0,
        )
        self.assertEqual(
            portfolio._qs_share_distribution_deposits.loc[
                calendar[2],
                "A",
            ],
            0.0,
        )
        self.assertAlmostEqual(portfolio.cash().iloc[-1], 55_000.0)
        self.assertAlmostEqual(portfolio.value().iloc[-1], 100_000.0)

    def test_runner_drops_missing_symbol_only_when_its_targets_are_zero(self):
        calendar = pd.to_datetime(["2026-01-05", "2026-01-06"])
        data = self.frames(calendar, columns=("A",), prices=(10.0,))
        targets = pd.DataFrame([[0.5, 0.0]], index=calendar[:1], columns=["A", "FUTURE"])

        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=data),
        ):
            portfolio = runner.run_backtest(
                "ashare", targets, config={"init_cash": 100_000.0, "slippage": 0.0}
            )

        self.assertEqual(portfolio.wrapper.columns.tolist(), ["A"])

        active_targets = targets.copy()
        active_targets.loc[:, "FUTURE"] = 0.1
        with (
            patch.object(runner, "load_ashare_calendar", return_value=calendar),
            patch.object(runner, "load_market_data", return_value=data),
            self.assertRaisesRegex(ValueError, "非零目标权重"),
        ):
            runner.run_backtest("ashare", active_targets)


class DataAdapterRegressionTests(unittest.TestCase):
    def test_ashare_corporate_actions_use_cash_and_share_distribution_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            action_dir = root / "StockDividend"
            action_dir.mkdir()
            pd.DataFrame(
                {
                    "TradeDate": [pd.Timestamp("2026-01-06")],
                    "Symbol": ["A"],
                    "CashDividend": [0.5],
                    "StockDividend": [0.1],
                    "StockTransfer": [0.2],
                }
            ).to_parquet(action_dir / "2026-01-06.parquet")

            with patch.dict(adapter.DATA_ROOTS, {"ashare": str(root)}):
                actions = adapter.load_ashare_corporate_actions(
                    symbols=["A"],
                    start="2026-01-05",
                    end="2026-01-07",
                )

            self.assertEqual(
                actions["cash_dividend"].loc[pd.Timestamp("2026-01-06"), "A"],
                0.5,
            )
            self.assertAlmostEqual(
                actions["share_multiplier"].loc[
                    pd.Timestamp("2026-01-06"),
                    "A",
                ],
                1.3,
            )

    def test_us_universe_uses_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_dir = root / "universe_daily" / "2025"
            universe_dir.mkdir(parents=True)
            pd.DataFrame({"Ticker": ["AAPL"]}).to_parquet(universe_dir / "part.parquet")
            with patch.dict(adapter.DATA_ROOTS, {"us_raw": str(root)}):
                actual = adapter.load_us_universe(2025)

        self.assertEqual(actual["Ticker"].tolist(), ["AAPL"])

    def test_us_constraint_metadata_parses_string_booleans(self):
        dimensions = {
            "DimSecurityMaster": pd.DataFrame(
                {"Ticker": ["X", "Y", "Z"], "IsADR": ["False", "True", None]}
            )
        }

        actual = adapter._extract_us_constraints(dimensions)["is_adr"]

        self.assertEqual(actual.to_dict(), {"X": False, "Y": True, "Z": False})

    def test_full_parquet_is_filtered_by_row_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = pd.DataFrame(
                {
                    "TradeDate": pd.to_datetime(["2020-01-01", "2025-01-01"]),
                    "Symbol": ["X", "X"],
                    "Close": [1.0, 2.0],
                }
            )
            frame.to_parquet(Path(temp_dir) / "full.parquet")

            actual = adapter._read_parquet_dir(
                temp_dir,
                value_cols=["Close"],
                date_range=("2025-01-01", "2025-12-31"),
            )

        self.assertEqual(actual.index.tolist(), [pd.Timestamp("2025-01-01")])

    def test_cos_paths_always_use_forward_slashes(self):
        actual = adapter._join_data_path("cos://bucket/root", "StockDailyBar", "*.parquet")

        self.assertEqual(actual, "cos://bucket/root/StockDailyBar/*.parquet")


class VisualizationRegressionTests(unittest.TestCase):
    @staticmethod
    def portfolio():
        index = pd.date_range("2026-01-05", periods=4)
        close = pd.DataFrame(
            {"A": [10.0, 11.0, 10.0, 12.0], "B": [20.0, 19.0, 21.0, 22.0]},
            index=index,
        )
        targets = pd.DataFrame(
            {"A": [0.5, 0.5, 0.0, 0.5], "B": [0.5, 0.5, 1.0, 0.5]},
            index=index,
        )
        return runner.vbt.Portfolio.from_orders(
            close,
            size=targets,
            size_type="targetpercent",
            cash_sharing=True,
            group_by=True,
            fees=0.0,
            init_cash=100_000.0,
            freq="1D",
        )

    def test_plotly_registry_exposes_vectorbt_portfolio_charts(self):
        charts = available_plotly_charts()

        self.assertTrue(
            {
                "orders", "trades", "positions", "asset_flow", "cash_flow",
                "nav_comparison", "value", "cum_returns", "drawdowns", "underwater",
                "gross_exposure", "net_exposure",
            }.issubset(charts)
        )

    def test_nav_curves_include_benchmark_and_compounded_excess(self):
        portfolio = self.portfolio()
        benchmark_close = pd.Series(
            [100.0, 102.0, 101.0, 104.0],
            index=portfolio.wrapper.index,
            name="000300.SH",
        )
        portfolio._qs_benchmark_close = benchmark_close
        portfolio._qs_benchmark_returns = benchmark_close.pct_change(fill_method=None)
        portfolio._qs_benchmark_symbol = "000300.SH"

        curves = build_nav_curves(portfolio)
        expected_strategy = portfolio.value() / portfolio.value().iloc[0]
        expected_benchmark = benchmark_close / benchmark_close.iloc[0]

        pd.testing.assert_series_equal(
            curves["strategy_nav"],
            expected_strategy.rename("strategy_nav"),
        )
        pd.testing.assert_series_equal(
            curves["benchmark_nav"],
            expected_benchmark.rename("benchmark_nav"),
        )
        np.testing.assert_allclose(
            curves["excess_nav"],
            expected_strategy / expected_benchmark,
        )
        self.assertEqual(curves.iloc[0].tolist(), [1.0, 1.0, 1.0])

        figure = build_plotly_chart(portfolio, "nav_comparison")
        self.assertEqual(
            [trace.name for trace in figure.data],
            ["策略 NAV", "基准 NAV (000300.SH)", "超额 NAV"],
        )

    def test_plotly_builders_support_cash_sharing_and_single_asset_views(self):
        portfolio = self.portfolio()

        overview = build_plotly_dashboard(portfolio)
        asset = build_plotly_dashboard(portfolio, column="A")
        positions = build_plotly_chart(portfolio, "positions", column="A")

        self.assertGreater(len(overview.data), 0)
        self.assertGreater(len(asset.data), 0)
        self.assertGreater(len(positions.data), 0)
        with self.assertRaisesRegex(ValueError, "必须指定 column"):
            build_plotly_chart(portfolio, "orders")
        with self.assertRaisesRegex(ValueError, "不能混入"):
            build_plotly_dashboard(portfolio, charts=["orders", "value"], column="A")

    def test_plotly_dashboard_exports_html(self):
        portfolio = self.portfolio()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = export_plotly_dashboard(
                portfolio,
                Path(temp_dir),
                include_plotlyjs="cdn",
            )

            self.assertTrue(output.exists())
            self.assertEqual(output.name, "portfolio_dashboard.html")
            self.assertIn("plotly", output.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
