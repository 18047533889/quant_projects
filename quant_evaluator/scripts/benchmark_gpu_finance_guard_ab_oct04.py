"""Opt-in A/B benchmark of existing and finance-scale guarded quantile means.

Reuses the bounded Oct 3 harness for provenance, CPU parity, admission and
synchronized ABBA timing. Both modes keep exact repair enabled. The prototype
swaps only the RawKernel factory for the established risk_guard ABI.
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

from quant_evaluator.kernels.gpu import quantile_numeric
from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard
from quant_evaluator.scripts import benchmark_gpu_quantile_numeric_guard_oct03 as base

CURRENT = "existing_guarded"
PROTOTYPE = "finance_prototype_guarded"
ABBA = (CURRENT, PROTOTYPE, PROTOTYPE, CURRENT)
FINANCE_SOURCE = "quant_evaluator/kernels/gpu/quantile_finance_guard.py"
OWN_SOURCE = "quant_evaluator/scripts/benchmark_gpu_finance_guard_ab_oct04.py"
MIN_HOST_AVAILABLE_BYTES = 32 * 1024**3
COHORT_DAYS = (128, 256)


class _RawKernelProxy:
    """Delegate CuPy while redirecting only the established risk guard ABI."""
    def __init__(self, cp):
        self._cp = cp

    def __getattr__(self, name):
        return getattr(self._cp, name)

    def RawKernel(self, source, name, *args, **kwargs):
        if name != "risk_guard":
            return self._cp.RawKernel(source, name, *args, **kwargs)
        return self._cp.RawKernel(
            finance_guard.FINANCE_RISK_GUARD_SOURCE,
            "finance_risk_guard", *args, **kwargs,
        )


@contextlib.contextmanager
def _guard_mode(mode):
    if mode == CURRENT:
        yield
        return
    if mode != PROTOTYPE:
        raise base.BenchmarkError(f"unknown guarded mode: {mode}")
    original = quantile_numeric._cupy
    try:
        quantile_numeric._cupy = lambda: _RawKernelProxy(original())
        yield
    finally:
        quantile_numeric._cupy = original

def _mem_available_bytes():
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as exc:
        raise base.BenchmarkError("could not read MemAvailable before host allocation") from exc
    raise base.BenchmarkError("MemAvailable is absent from /proc/meminfo")



def run_benchmark(**kwargs):
    """Run established parity/timing flow under two guarded implementations."""
    old = (base.BASELINE, base.GUARDED, base.ABBA, base._guard_mode,
           base._source_hashes, base._wrapper_hash)
    try:
        base.BASELINE, base.GUARDED, base.ABBA = CURRENT, PROTOTYPE, ABBA
        base._guard_mode = _guard_mode

        def source_hashes(root=ROOT):
            files = list(base.SOURCE_FILES) + [FINANCE_SOURCE, OWN_SOURCE]
            return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                    for relative in files}

        base._source_hashes = source_hashes
        def wrapper_hash():
            return hashlib.sha256((ROOT / OWN_SOURCE).read_bytes()).hexdigest()

        base._wrapper_hash = wrapper_hash
        shape = kwargs.get("shape", base.DEFAULT_SHAPE)
        rounds = kwargs.get("rounds", 3)
        if (not isinstance(shape, tuple) or len(shape) != 3
                or shape[0] not in COHORT_DAYS or shape[1:] != (5461, 48)):
            raise base.BenchmarkError("only T=128 or 256, N=5461, F=48 is admitted")
        if type(rounds) is not int or rounds < 3:
            raise base.BenchmarkError("at least three synchronized ABBA rounds are required")
        host_estimate = base._check_shape(shape)
        available = _mem_available_bytes()
        if available < MIN_HOST_AVAILABLE_BYTES:
            raise base.BenchmarkError(
                f"host admission requires 32 GiB MemAvailable; observed {available} bytes"
            )
        result = base.run_benchmark(root=ROOT, **kwargs)
        result["host_memory_preflight"] = {
            "available_bytes_before_allocation": available,
            "minimum_available_bytes": MIN_HOST_AVAILABLE_BYTES,
            "estimated_peak_bytes": host_estimate,
            "base_guard_bytes": base.MAX_ESTIMATED_BYTES,
        }
        result["source_files"] = list(base.SOURCE_FILES) + [FINANCE_SOURCE, OWN_SOURCE]
        result["scope"] = (
            "synthetic GPU quantile means; existing guarded repair versus "
            "finance prototype guarded repair; no bypass; not full QE/COS"
        )
        result["modes"] = {
            CURRENT: "existing risk_guard and exact repair remain enabled",
            PROTOTYPE: (
                "temporarily map only RawKernel(name=risk_guard) to "
                "finance_risk_guard with matching ABI; all other kernels and "
                "exact repair remain enabled; mapping restored in try/finally"
            ),
        }
        result["prototype_admission"] = {
            "benchmark_cohort": f"Q=5,20; T={shape[0]},N=5461,F=48",
            "source_hash_includes": [FINANCE_SOURCE, OWN_SOURCE],
        }
        result["guard_policy"] = {
            CURRENT: "existing risk guard and exact repair enabled",
            PROTOTYPE: "finance risk guard and exact repair enabled; no bypass",
        }
        result["parity_policy"] = (
            "each existing/prototype cold, warmup and timed output compared to "
            "independent NumPy CPU reference; counts exact and returns "
            "allclose(rtol=1e-9, atol=1e-12, equal_nan=True)"
        )
        return result
    finally:
        (base.BASELINE, base.GUARDED, base.ABBA, base._guard_mode,
         base._source_hashes, base._wrapper_hash) = old


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="run only in an explicitly approved GPU timing slot")
    parser.add_argument("--days", type=int, choices=COHORT_DAYS, default=128,
                        help="historical cohort days (128 or 256); Q5/Q20")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("timing is opt-in; pass --run only after GPU-slot approval")
    try:
        result = run_benchmark(shape=(args.days, 5461, 48))
    except base.BenchmarkError as exc:
        print(f"GPU finance guard benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
