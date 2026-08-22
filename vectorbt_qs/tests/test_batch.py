import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dill
import pandas as pd

from vectorbt_qs.cli import (
    DEFAULT_START_DATE,
    _expand_standard_accurate_benchmarks,
    _export_batch_results,
    _normalize_run_range,
    _pop_run_range,
    _positions_parent_folder_name,
    cmd_accurate_benchmark,
    cmd_batch,
)
from vectorbt_qs.mvp.engine.batch import (
    BatchBacktestResult,
    apply_parameter_overrides,
    expand_parameter_grid,
    run_backtest_batch,
)
from vectorbt_qs.mvp.engine.profiles import (
    standard_accurate_v2_execution_grid,
)


class _FakePortfolio:
    def __init__(self, number: int):
        self.number = number

    def stats(self, settings=None):
        return pd.Series(
            {
                "Total Return [%]": float(self.number),
                "Sharpe Ratio": float(self.number) / 10,
            }
        )

    def value(self):
        index = pd.date_range("2026-01-01", periods=3)
        return pd.Series(
            [100.0, 100.0 + self.number, 101.0 + self.number],
            index=index,
        )

    def dumps(self):
        return dill.dumps(self)


class BatchBacktestTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=2)
        self.weights = pd.DataFrame({"X": [0.5, 0.5]}, index=index)

    def test_grid_expands_cartesian_product_in_declared_order(self):
        combinations = expand_parameter_grid(
            {
                "freq": ["1D", "1W"],
                "fast_tradeability": ["ignore", "approximate"],
            }
        )
        self.assertEqual(
            combinations,
            [
                {"freq": "1D", "fast_tradeability": "ignore"},
                {"freq": "1D", "fast_tradeability": "approximate"},
                {"freq": "1W", "fast_tradeability": "ignore"},
                {"freq": "1W", "fast_tradeability": "approximate"},
            ],
        )

    @patch("vectorbt_qs.cli._attach_benchmark_view")
    def test_frozen_profile_reuses_eight_trades_for_24_benchmark_reports(
        self,
        attach_benchmark,
    ):
        execution_rows = expand_parameter_grid(
            standard_accurate_v2_execution_grid()
        )
        parameters = pd.DataFrame(
            [
                {
                    "run_id": f"run_{i:04d}",
                    "positions": "weights",
                    **row,
                }
                for i, row in enumerate(execution_rows, start=1)
            ]
        ).set_index("run_id")
        execution_result = BatchBacktestResult(
            portfolios={
                run_id: SimpleNamespace(source=run_id)
                for run_id in parameters.index
            },
            parameters=parameters,
            configs={run_id: {} for run_id in parameters.index},
            errors={},
        )
        attach_benchmark.side_effect = (
            lambda portfolio, benchmark: SimpleNamespace(
                source=portfolio.source,
                benchmark=benchmark,
            )
        )

        expanded = _expand_standard_accurate_benchmarks(execution_result)

        self.assertEqual(len(expanded.portfolios), 24)
        self.assertEqual(attach_benchmark.call_count, 24)
        self.assertEqual(expanded.parameters["source_run_id"].nunique(), 8)
        self.assertTrue(
            (
                expanded.parameters.groupby("source_run_id").size()
                == 3
            ).all()
        )

    def test_public_run_range_defaults_to_2024(self):
        self.assertEqual(DEFAULT_START_DATE, "2024-01-01")
        self.assertEqual(
            _normalize_run_range(None, None),
            ("2024-01-01", None),
        )
        config = {"start": "2024-02-03", "end": "2024-03-04", "freq": "1D"}
        self.assertEqual(
            _pop_run_range(config),
            ("2024-02-03", "2024-03-04"),
        )
        self.assertEqual(config, {"freq": "1D"})
        with self.assertRaisesRegex(ValueError, "end 不能早于 start"):
            _normalize_run_range("2024-02-01", "2024-01-31")

    def test_single_benchmark_output_uses_positions_parent_folder(self):
        path = Path("benchmarks") / "strategy_v1" / "target_positions.parquet"
        self.assertEqual(
            _positions_parent_folder_name(path),
            "strategy_v1",
        )

    @patch("vectorbt_qs.cli._run_standard_accurate_benchmark_v1")
    def test_accurate_benchmark_cli_dispatches_three_paths(self, run_profile):
        args = SimpleNamespace(
            positions="strategy_v1/target_positions.parquet",
            barra_root="v2_sbi_mvl_fullA",
            output_root="results",
            start="2024-01-01",
            end="2024-01-31",
        )

        cmd_accurate_benchmark(args)

        run_profile.assert_called_once_with(
            positions=args.positions,
            barra_root=args.barra_root,
            output_root=args.output_root,
            base_dir=Path.cwd(),
            start=args.start,
            end=args.end,
        )

    @patch("vectorbt_qs.cli._run_standard_accurate_v2")
    def test_yaml_can_select_frozen_accurate_profile(self, run_profile):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "profile.yaml"
            config_path.write_text(
                """
input:
  positions: weights.parquet
  barra_root: barra
output:
  root: results
batch:
  profile: standard_accurate_v2
""".strip(),
                encoding="utf-8",
            )
            cmd_batch(SimpleNamespace(config=str(config_path)))
            run_profile.assert_called_once_with(
                positions="weights.parquet",
                barra_root="barra",
                output_root="results",
                base_dir=config_path.parent,
                start="2024-01-01",
                end=None,
                workers=1,
            )

    @patch("vectorbt_qs.cli._run_standard_accurate_v2")
    def test_frozen_profile_rejects_backtest_override(self, run_profile):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "profile.yaml"
            config_path.write_text(
                """
input:
  positions: weights.parquet
  barra_root: barra
output:
  root: results
batch:
  profile: standard_accurate_v2
backtest:
  slippage: 0
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "backtest.slippage 是冻结值"):
                cmd_batch(SimpleNamespace(config=str(config_path)))
            run_profile.assert_not_called()

    @patch("vectorbt_qs.cli._run_standard_accurate_v2")
    def test_full_frozen_profile_yaml_can_keep_parameters_in_body(
        self,
        run_profile,
    ):
        config_path = (
            Path(__file__).parents[1] / "configs" / "configs_module.yaml"
        )

        cmd_batch(SimpleNamespace(config=str(config_path)))

        run_profile.assert_called_once()
        kwargs = run_profile.call_args.kwargs
        self.assertEqual(kwargs["workers"], 1)
        self.assertEqual(kwargs["start"].isoformat(), "2024-01-01")
        self.assertIsNone(kwargs["end"])

    @patch("vectorbt_qs.cli._run_standard_accurate_v2")
    def test_explicit_frozen_grid_rejects_changed_candidates(
        self,
        run_profile,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "profile.yaml"
            config_path.write_text(
                """
input:
  positions: weights.parquet
  barra_root: barra
output:
  root: results
batch:
  profile: standard_accurate_v2
  grid:
    benchmark_index: [000300.SH]
    price_type: [open, vwap]
    costs.commission: [0.002, 0.001]
    freq: [1D, 1W]
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "batch.grid 是冻结矩阵"):
                cmd_batch(SimpleNamespace(config=str(config_path)))
            run_profile.assert_not_called()

    def test_grid_requires_non_empty_lists(self):
        with self.assertRaisesRegex(TypeError, "必须是列表"):
            expand_parameter_grid({"freq": "1D"})
        with self.assertRaisesRegex(ValueError, "不能为空列表"):
            expand_parameter_grid({"freq": []})

    def test_nested_overrides_do_not_mutate_base_config(self):
        base = {"costs": {"commission": 0.001}, "freq": "1D"}
        result = apply_parameter_overrides(
            base,
            {
                "backtest.freq": "1W",
                "costs.commission": 0.00025,
            },
        )
        self.assertEqual(result["freq"], "1W")
        self.assertEqual(result["costs"]["commission"], 0.00025)
        self.assertEqual(base["freq"], "1D")
        self.assertEqual(base["costs"]["commission"], 0.001)

    @patch("vectorbt_qs.mvp.engine.batch.run_backtest")
    def test_batch_crosses_positions_and_parameters(self, run_mock):
        counter = iter(range(1, 5))
        run_mock.side_effect = lambda *args, **kwargs: _FakePortfolio(
            next(counter)
        )
        result = run_backtest_batch(
            "ashare",
            {"group_1": self.weights, "group_2": self.weights},
            {"freq": ["1D", "1W"]},
            base_config={"execution_mode": "fast"},
        )

        self.assertEqual(list(result.portfolios), [
            "run_0001",
            "run_0002",
            "run_0003",
            "run_0004",
        ])
        self.assertEqual(
            result.parameters["positions"].tolist(),
            ["group_1", "group_1", "group_2", "group_2"],
        )
        self.assertEqual(
            result.parameters["freq"].tolist(),
            ["1D", "1W", "1D", "1W"],
        )
        self.assertTrue((result.parameters["status"] == "success").all())
        self.assertEqual(result.performance_table().shape[0], 4)
        self.assertEqual(result.nav_curves().shape, (3, 4))
        self.assertEqual(result.configs["run_0002"]["freq"], "1W")
        self.assertEqual(run_mock.call_count, 4)
        self.assertEqual(
            run_mock.call_args_list[1].kwargs["config"]["freq"],
            "1W",
        )

    @patch("vectorbt_qs.mvp.engine.batch.run_backtest")
    def test_batch_can_record_error_and_continue(self, run_mock):
        run_mock.side_effect = [
            ValueError("bad combination"),
            _FakePortfolio(2),
        ]
        result = run_backtest_batch(
            "ashare",
            self.weights,
            {"freq": ["bad", "1D"]},
            on_error="continue",
        )
        self.assertEqual(list(result.portfolios), ["run_0002"])
        self.assertIn("run_0001", result.errors)
        self.assertEqual(
            result.parameters.loc["run_0001", "status"],
            "failed",
        )
        self.assertIn(
            "bad combination",
            result.parameters.loc["run_0001", "error"],
        )

    def test_batch_checks_run_count_before_execution(self):
        with self.assertRaisesRegex(ValueError, "超过 max_runs=3"):
            run_backtest_batch(
                "ashare",
                {"a": self.weights, "b": self.weights},
                {"freq": ["1D", "1W"]},
                max_runs=3,
            )

    def test_batch_rejects_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers 必须是正整数"):
            run_backtest_batch(
                "ashare",
                self.weights,
                {"freq": ["1D"]},
                workers=0,
            )

    @patch("vectorbt_qs.mvp.engine.batch.run_backtest")
    def test_process_mode_restores_declared_result_order(self, run_mock):
        from vectorbt_qs.mvp.engine import batch as batch_module

        run_mock.side_effect = [
            _FakePortfolio(1),
            _FakePortfolio(2),
        ]

        class ImmediateFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

            def cancel(self):
                return False

        class ImmediateExecutor:
            max_workers = None

            def __init__(self, *, max_workers, initializer, initargs):
                type(self).max_workers = max_workers
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def submit(self, function, *args):
                return ImmediateFuture(function(*args))

        with (
            patch.object(batch_module, "ProcessPoolExecutor", ImmediateExecutor),
            patch.object(
                batch_module,
                "as_completed",
                side_effect=lambda futures: list(reversed(list(futures))),
            ),
            patch.object(
                batch_module,
                "_restore_process_portfolio",
                side_effect=lambda payload, metadata: dill.loads(payload),
            ),
        ):
            result = run_backtest_batch(
                "ashare",
                self.weights,
                {"freq": ["1D", "1W"]},
                workers=2,
            )

        self.assertEqual(ImmediateExecutor.max_workers, 2)
        self.assertEqual(list(result.portfolios), ["run_0001", "run_0002"])
        self.assertEqual(
            result.parameters["freq"].tolist(),
            ["1D", "1W"],
        )

    def test_batch_range_is_not_silently_accepted_as_grid_parameter(self):
        with self.assertRaisesRegex(ValueError, "批次级日期边界"):
            run_backtest_batch(
                "ashare",
                self.weights,
                {"backtest.start": ["2024-01-01", "2025-01-01"]},
            )

    @patch("vectorbt_qs.mvp.engine.batch.run_backtest")
    def test_batch_export_writes_manifest_stats_and_nav(self, run_mock):
        run_mock.return_value = _FakePortfolio(1)
        result = run_backtest_batch(
            "ashare",
            self.weights,
            {"freq": ["1D"]},
            base_config={"execution_mode": "fast"},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            comparison = _export_batch_results(result, output, plot=False)
            self.assertEqual(comparison.shape[0], 1)
            self.assertTrue((output / "batch_parameters.csv").exists())
            self.assertTrue((output / "effective_configs.json").exists())
            self.assertTrue((output / "performance_comparison.csv").exists())
            self.assertTrue((output / "nav_curves.parquet").exists())
            self.assertTrue(
                (output / "runs" / "run_0001" / "performance_stats.csv").exists()
            )
            configs = json.loads(
                (output / "effective_configs.json").read_text(encoding="utf-8")
            )
            self.assertEqual(configs["run_0001"]["freq"], "1D")


if __name__ == "__main__":
    unittest.main()
