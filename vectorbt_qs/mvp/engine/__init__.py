"""MVP engine public API.

Runner functions are loaded on first access so lightweight batch/profile
imports do not require the optional plotting and backtest runtime stack.
"""

_RUNNER_EXPORTS = {
    "run_backtest",
    "run_backtest_grid",
    "portfolio_report",
    "compare_reports",
}


def __getattr__(name):
    if name in _RUNNER_EXPORTS:
        from . import runner

        value = getattr(runner, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
from .batch import (
    BatchBacktestResult,
    apply_parameter_overrides,
    expand_parameter_grid,
    run_backtest_batch,
)
from .profiles import (
    STANDARD_ACCURATE_BENCHMARK_V1,
    STANDARD_ACCURATE_BENCHMARK_V1_MIN_BENCHMARK_COVERAGE,
    STANDARD_ACCURATE_V1,
    STANDARD_ACCURATE_V1_MIN_BENCHMARK_COVERAGE,
    STANDARD_ACCURATE_V2,
    STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE,
    standard_accurate_v1_base_config,
    standard_accurate_v1_benchmarks,
    standard_accurate_v1_execution_grid,
    standard_accurate_v1_execution_run_count,
    standard_accurate_v1_folder_name,
    standard_accurate_v1_grid,
    standard_accurate_v1_run_count,
    standard_accurate_benchmark_v1_base_config,
    standard_accurate_benchmark_v1_execution_grid,
    standard_accurate_benchmark_v1_grid,
    standard_accurate_v2_base_config,
    standard_accurate_v2_benchmarks,
    standard_accurate_v2_execution_grid,
    standard_accurate_v2_execution_run_count,
    standard_accurate_v2_folder_name,
    standard_accurate_v2_grid,
    standard_accurate_v2_run_count,
)
