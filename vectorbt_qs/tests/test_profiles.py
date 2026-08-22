import unittest

from vectorbt_qs.mvp.engine.batch import expand_parameter_grid
from vectorbt_qs.mvp.engine.profiles import (
    STANDARD_ACCURATE_BENCHMARK_V1,
    STANDARD_ACCURATE_V1,
    STANDARD_ACCURATE_V2,
    STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE,
    standard_accurate_v1_base_config,
    standard_accurate_benchmark_v1_base_config,
    standard_accurate_benchmark_v1_execution_grid,
    standard_accurate_benchmark_v1_grid,
    standard_accurate_v2_base_config,
    standard_accurate_v2_execution_run_count,
    standard_accurate_v2_folder_name,
    standard_accurate_v2_grid,
    standard_accurate_v2_run_count,
)


class StandardAccurateProfileTests(unittest.TestCase):
    def test_profile_is_frozen_accurate_and_has_24_runs(self):
        self.assertEqual(STANDARD_ACCURATE_V1, "standard_accurate_v1")
        self.assertEqual(STANDARD_ACCURATE_V2, "standard_accurate_v2")
        self.assertEqual(
            STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE,
            0.99,
        )
        base = standard_accurate_v2_base_config()
        grid = standard_accurate_v2_grid()
        self.assertEqual(base["execution_mode"], "accurate")
        self.assertFalse(base["allow_partial"])
        self.assertEqual(base["limit_check_mode"], "strict")
        self.assertEqual(base["performance_year_days"], 252)
        self.assertEqual(base["risk_free_rate"], 0.0)
        self.assertEqual(base["max_participation_rate"], 0.10)
        self.assertEqual(
            grid["benchmark_index"],
            ("000300.SH", "000905.SH", "000852.SH"),
        )
        self.assertEqual(grid["price_type"], ("open", "vwap"))
        self.assertEqual(grid["costs.commission"], (0.002, 0.001))
        self.assertEqual(grid["freq"], ("1D", "1W"))
        self.assertEqual(standard_accurate_v2_run_count(), 24)
        self.assertEqual(standard_accurate_v2_execution_run_count(), 8)
        self.assertEqual(len(expand_parameter_grid(grid)), 24)

    def test_profile_copies_cannot_mutate_frozen_values(self):
        first = standard_accurate_v2_base_config()
        first["costs"]["stamp_tax"] = 99
        second = standard_accurate_v2_base_config()
        self.assertEqual(second["costs"]["stamp_tax"], 0.0005)
        self.assertEqual(
            standard_accurate_v1_base_config()["limit_check_mode"],
            "execution",
        )

    def test_single_benchmark_profile_has_one_canonical_configuration(self):
        self.assertEqual(
            STANDARD_ACCURATE_BENCHMARK_V1,
            "standard_accurate_benchmark_v1",
        )
        base = standard_accurate_benchmark_v1_base_config()
        grid = standard_accurate_benchmark_v1_grid()
        self.assertEqual(base["execution_mode"], "accurate")
        self.assertEqual(base["slippage"], 0.001)
        self.assertEqual(grid["benchmark_index"], ("000852.SH",))
        self.assertEqual(grid["price_type"], ("vwap",))
        self.assertEqual(grid["costs.commission"], (0.0005,))
        self.assertEqual(grid["freq"], ("1D",))
        self.assertEqual(
            standard_accurate_benchmark_v1_execution_grid(),
            {
                "price_type": ("vwap",),
                "costs.commission": (0.0005,),
                "freq": ("1D",),
            },
        )

    def test_folder_name_contains_only_four_explicit_parameters(self):
        name = standard_accurate_v2_folder_name(
            {
                "benchmark_index": "000300.SH",
                "price_type": "open",
                "costs.commission": 0.002,
                "freq": "1D",
                "ignored_default": "must-not-appear",
            }
        )
        self.assertEqual(
            name,
            "benchmark_index_000300.SH_"
            "price_type_open_"
            "cost.commission_0.002_"
            "freq_1D",
        )
        self.assertNotIn("ignored_default", name)

    def test_all_profile_folder_names_are_unique(self):
        names = {
            standard_accurate_v2_folder_name(parameters)
            for parameters in expand_parameter_grid(
                standard_accurate_v2_grid()
            )
        }
        self.assertEqual(len(names), 24)


if __name__ == "__main__":
    unittest.main()
