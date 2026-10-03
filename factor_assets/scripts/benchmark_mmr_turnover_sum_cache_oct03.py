"""Bounded MMR turnover-sum benchmark: separate kernel and public assembly."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factor_assets.assembly.engine import FactorSetAssembler as NewAssembler
from factor_assets.tests.assembly.test_v6_constrained_mmr import (
    decided,
    evidence,
    make_asset,
    make_spec,
    policy,
)


N, K = 1_000, 50
BASELINE_REVISION = "cf21d7da6"


def load_old_assembler():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:factor_assets/assembly/engine.py"],
        cwd=ROOT,
        text=True,
    )
    module_name = "_factor_assets_engine_before_turnover_sum_reuse"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, f"{BASELINE_REVISION}:factor_assets/assembly/engine.py", "exec"), module.__dict__)
    return module.FactorSetAssembler


def measure_abba(old_call, new_call, repetitions=3):
    samples = {"old": [], "new": []}
    for _ in range(repetitions):
        for name, call in (("old", old_call), ("new", new_call), ("new", new_call), ("old", old_call)):
            start = time.perf_counter()
            call()
            samples[name].append(time.perf_counter() - start)
    return samples


def kernel_old(values, candidate_count, target):
    checksum = 0.0
    for chosen in range(target):
        for _ in range(candidate_count - chosen):
            used = sum(values[index] for index in range(chosen))
            checksum += used + 0.001
        if chosen + 1 < target:
            for _ in range(candidate_count - chosen - 1):
                used = sum(values[index] for index in range(chosen + 1))
                checksum += used + 0.001
    return checksum


def kernel_new(values, candidate_count, target):
    checksum = 0.0
    used = 0
    for chosen in range(target):
        for _ in range(candidate_count - chosen):
            checksum += used + 0.001
        if chosen + 1 < target:
            used = sum(values[index] for index in range(chosen + 1))
            for _ in range(candidate_count - chosen - 1):
                checksum += used + 0.001
    return checksum


def main():
    old_assembler_type = load_old_assembler()
    ids = [f"F{index:04d}" for index in range(N)]
    assets = [make_asset(factor_id) for factor_id in ids]
    decisions = [decided(factor_id) for factor_id in ids]
    typed = {
        factor_id: evidence(factor_id, 0.95 - index / 10_000, turnover=0.01 + (index % 5) * 0.001)
        for index, factor_id in enumerate(ids)
    }
    assembly_policy = policy(
        max_per_microcluster=N,
        max_per_macrocluster=N,
        turnover_budget=10.0,
    )
    spec = make_spec("turnover-cache-bench", "Turnover cache benchmark", "diverse", max_factors=K)

    def assemble(assembler_type):
        return assembler_type().assemble(
            spec,
            assets,
            selection_decisions=decisions,
            similarity_provider=lambda _a, _b: 0.0,
            assembly_evidence=typed,
            assembly_policy=assembly_policy,
            created_at="2026-10-03T00:00:00Z",
        )

    old_result = assemble(old_assembler_type)
    new_result = assemble(NewAssembler)
    if old_result != new_result:
        raise RuntimeError("old and new public assembly results differ")

    values = [0.01 + (index % 5) * 0.001 for index in range(N)]
    # Warm both kernels and public paths before timed ABBA blocks.
    kernel_old(values, N, K)
    kernel_new(values, N, K)
    assemble(old_assembler_type)
    assemble(NewAssembler)

    kernel_times = measure_abba(
        lambda: kernel_old(values, N, K),
        lambda: kernel_new(values, N, K),
    )
    assembly_times = measure_abba(
        lambda: assemble(old_assembler_type),
        lambda: assemble(NewAssembler),
    )
    for label, times in (("kernel", kernel_times), ("public assembly", assembly_times)):
        print(
            f"{label} ABBA (N={N}, K={K}), median seconds: "
            f"old={statistics.median(times['old']):.6f}; "
            f"new={statistics.median(times['new']):.6f}; "
            f"raw old={[round(value, 6) for value in times['old']]}; "
            f"raw new={[round(value, 6) for value in times['new']]}"
        )
    print(
        f"public artifact equality verified; selected={len(new_result.factor_ids)}; "
        "similarity provider is a bounded zero-value callable"
    )


if __name__ == "__main__":
    main()
