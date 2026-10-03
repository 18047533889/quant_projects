"""Small mocked checks for the numeric-guard overhead benchmark."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_quantile_numeric_guard_oct03.py"
SPEC = importlib.util.spec_from_file_location("quantile_numeric_guard_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(benchmark)


def fake_routes(mode_log, mismatch_numba=False):
    originals = benchmark.quantile.repair_bucket_means

    def route(name):
        def call(batch, label_bundle, n_quantiles, min_assets):
            mode = "guarded" if benchmark.quantile.repair_bucket_means is originals else "bypassed"
            mode_log[name, n_quantiles].append(mode)
            returns = __import__("numpy").full(
                (batch.num_times, n_quantiles, batch.num_factors),
                0.02 if not (mismatch_numba and name == "numba" and mode == "bypassed") else 0.03,
            )
            counts = __import__("numpy").full(
                returns.shape, min_assets, dtype="int32",
            )
            return returns, counts
        return call

    return {"numpy": route("numpy"), "numba": route("numba")}


class NumericGuardBenchmarkTests(unittest.TestCase):
    # Catches a bypass context that leaves global metric behavior altered after an error.
    def test_guard_bypass_restores_all_bindings_on_exception(self):
        originals = benchmark._guard_bindings()
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with benchmark._without_numeric_guards():
                changed = benchmark._guard_bindings()
                self.assertTrue(all(
                    changed[module, name] is not original
                    for (module, name), original in originals.items()
                ))
                raise RuntimeError("sentinel")
        self.assertTrue(all(
            benchmark._guard_bindings()[key] is original
            for key, original in originals.items()
        ))

    # Catches reordered/missing A/B calls or skipped parity checks in any warm/timed ABBA pair.
    def test_small_mocked_matrix_checks_every_abba_pair_and_fingerprints(self):
        mode_log = {
            (backend, q): []
            for backend in ("numpy", "numba")
            for q in (5, 20)
        }
        env_before = dict(os.environ)
        result = benchmark.run_benchmark(
            shape=(2, 32, 2), rounds=1, warmup_cycles=1,
            callables=fake_routes(mode_log),
            runtime_fingerprint=lambda device: {"packages": {"numba": "test"}, "device": device},
        )
        self.assertEqual(dict(os.environ), env_before)
        expected = [
            "bypassed", "guarded",
            "bypassed", "guarded", "guarded", "bypassed",
            "bypassed", "guarded", "guarded", "bypassed",
        ]
        for modes in mode_log.values():
            self.assertEqual(modes, expected)
        self.assertEqual(result["warm_order_per_round"], "ABBA")
        self.assertEqual(result["warm_rounds"], 1)
        self.assertEqual(result["source_sha256_before"], result["source_sha256_after"])
        self.assertEqual(result["input_sha256_before"], result["input_sha256_after"])
        self.assertEqual(result["runtime_fingerprint_before"], result["runtime_fingerprint_after"])
        self.assertEqual(result["parity_pairs_checked"], {"5": {"numpy": 5, "numba": 5}, "20": {"numpy": 5, "numba": 5}})
        self.assertEqual(len(result["wrapper_sha256"]), 64)

    # Catches accepting numerically different bypass results for either CPU route.
    def test_rejects_a_mode_that_changes_values(self):
        mode_log = {(backend, q): [] for backend in ("numpy", "numba") for q in (5, 20)}
        with self.assertRaisesRegex(benchmark.BenchmarkError, "parity"):
            benchmark.run_benchmark(
                shape=(2, 32, 2), rounds=1, warmup_cycles=1,
                callables=fake_routes(mode_log, mismatch_numba=True),
                runtime_fingerprint=lambda device: {"packages": {"numba": "test"}, "device": device},
            )

    # Catches allocations whose conservative bound exceeds the documented cap.
    def test_rejects_shapes_over_the_cell_admission_limit(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "cell"):
            benchmark._check_shape(1, 1, benchmark.MAX_FACTOR_CELLS + 1)


    # Catches the dangerous default of launching the full matrix without opt-in.
    def test_cli_requires_run_flag_before_starting_benchmark(self):
        with patch.object(benchmark, "run_benchmark", side_effect=AssertionError("must not run")) as run:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.main([])
        run.assert_not_called()

if __name__ == "__main__":
    unittest.main()
