"""Opt-in T256 A/B of generic and Q5-capacity-8 guarded finance means.

Uses the established bounded CUDA timing harness with Q=5 only. Exact repair
stays enabled in both modes; the candidate changes only risk_guard capacity.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard
from quant_evaluator.kernels.gpu import quantile_numeric
from quant_evaluator.scripts import benchmark_gpu_finance_guard_ab_oct04 as common

CURRENT = "existing_generic_guarded"
SPECIALIZED = "finance_q5_capacity8_guarded"
ABBA = (CURRENT, SPECIALIZED, SPECIALIZED, CURRENT)
OWN_SOURCE = "quant_evaluator/scripts/benchmark_gpu_finance_specialized_q5_oct04.py"
Q5_CAPACITY = 8
SHAPE = (256, 5461, 48)


class _RawKernelProxy:
    """Redirect risk_guard through the validated Q5 specialization factory."""
    def __init__(self, cp, compiled_kernels):
        self._cp = cp
        self._compiled_kernels = compiled_kernels

    def __getattr__(self, name):
        return getattr(self._cp, name)

    def RawKernel(self, source, name, *args, **kwargs):
        if name != "risk_guard":
            return self._cp.RawKernel(source, name, *args, **kwargs)
        kernel = finance_guard.compile_finance_risk_guard(
            self._cp, n_quantiles=5,
        )
        self._compiled_kernels.append(kernel)
        return kernel


@contextlib.contextmanager
def _guard_mode(mode, compiled_kernels):
    if mode == CURRENT:
        yield
        return
    if mode != SPECIALIZED:
        raise common.base.BenchmarkError(f"unknown benchmark mode: {mode}")
    original = quantile_numeric._cupy
    try:
        quantile_numeric._cupy = lambda: _RawKernelProxy(original(), compiled_kernels)
        yield
    finally:
        quantile_numeric._cupy = original


def _compiled_local_attributes(kernels):
    reports = {}
    for kernel in kernels:
        attrs = kernel.attributes
        value = attrs.get("local_size_bytes", attrs.get("localSizeBytes"))
        if value is None:
            raise common.base.BenchmarkError(
                "CuPy did not report specialized risk_guard local memory"
            )
        reports[str(int(value))] = {
            "local_size_bytes_per_thread": int(value),
            "compile_options": ["--std=c++11", "-DFINANCE_Q_CAPACITY=8"],
        }
    return [reports[key] for key in sorted(reports, key=int)]


def run_benchmark(**kwargs):
    """Run one Q5-only ABBA cohort and restore all temporary shared globals."""
    old_common = (common.CURRENT, common.PROTOTYPE, common.ABBA, common._guard_mode)
    old_base = (common.base.QUANTILE_COUNTS, common.base.SOURCE_FILES)
    compiled_kernels = []
    try:
        common.CURRENT, common.PROTOTYPE, common.ABBA = (
            CURRENT, SPECIALIZED, ABBA,
        )
        common._guard_mode = lambda mode: _guard_mode(mode, compiled_kernels)
        common.base.QUANTILE_COUNTS = (5,)
        common.base.SOURCE_FILES = tuple(common.base.SOURCE_FILES) + (OWN_SOURCE,)

        shape = kwargs.get("shape", SHAPE)
        if shape != SHAPE:
            raise common.base.BenchmarkError(
                "only T=256,N=5461,F=48,Q=5 is admitted"
            )
        rounds = kwargs.get("rounds", 3)
        if type(rounds) is not int or rounds < 3:
            raise common.base.BenchmarkError(
                "at least three synchronized ABBA rounds are required"
            )

        result = common.run_benchmark(**{**kwargs, "shape": shape})
        source_hashes = result["source_sha256_before"]
        result["source_files"] = (
            list(common.base.SOURCE_FILES)
            + [common.FINANCE_SOURCE, common.OWN_SOURCE]
        )
        result["scope"] = (
            "synthetic T256/N5461/F48 quantile means; existing generic guard "
            "versus Q5 capacity-8 finance guard; exact repair enabled in both"
        )
        result["modes"] = {
            CURRENT: "existing generic risk_guard and exact repair enabled",
            SPECIALIZED: (
                "finance risk_guard compiled through compile_finance_risk_guard "
                "with validated Q=5 capacity 8; exact repair enabled"
            ),
        }
        result["quantile_counts"] = [5]
        result["quantile_count_requested"] = 5
        result["compile_capacity"] = Q5_CAPACITY
        result["compiled_local_attributes_candidate"] = _compiled_local_attributes(
            compiled_kernels,
        )
        result["candidate_wrapper_sha256"] = source_hashes[OWN_SOURCE]
        result["finance_guard_sha256"] = source_hashes[common.FINANCE_SOURCE]
        result["prototype_admission"] = {
            "benchmark_cohort": "Q=5; T=256,N=5461,F=48",
            "compile_capacity": Q5_CAPACITY,
            "source_hash_includes": [
                OWN_SOURCE, common.OWN_SOURCE, common.FINANCE_SOURCE,
            ],
        }
        result["guard_policy"] = {
            CURRENT: "generic risk guard and exact repair enabled",
            SPECIALIZED: "capacity-8 risk guard and exact repair enabled",
        }
        result["parity_policy"] = (
            "each existing/specialized cold, warmup and timed output compared "
            "to independent NumPy CPU reference; counts exact and returns "
            "allclose(rtol=1e-9, atol=1e-12, equal_nan=True)"
        )
        result["parity_call_interpretation"] = {
            "total_calls_per_quantile": 18,
            "per_mode": {"cold": 1, "warmup": 2, "timed": 6, "total": 9},
        }
        result["compile_timing"] = (
            "compilation is included in cold-call timing; host-to-device "
            "transfer is separately measured"
        )
        return result
    finally:
        (common.CURRENT, common.PROTOTYPE, common.ABBA,
         common._guard_mode) = old_common
        (common.base.QUANTILE_COUNTS, common.base.SOURCE_FILES) = old_base


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="run only after numeric-source freeze and GPU-slot release")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("timing is opt-in; pass --run only after approval")
    try:
        result = run_benchmark()
    except common.base.BenchmarkError as exc:
        print(f"Q5 finance guard specialization benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
