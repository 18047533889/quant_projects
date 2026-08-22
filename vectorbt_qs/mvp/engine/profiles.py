"""Versioned, reproducible batch profiles."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping


STANDARD_ACCURATE_V1 = "standard_accurate_v1"
STANDARD_ACCURATE_V1_MIN_BENCHMARK_COVERAGE = 0.99
STANDARD_ACCURATE_V2 = "standard_accurate_v2"
STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE = 0.99
STANDARD_ACCURATE_BENCHMARK_V1 = "standard_accurate_benchmark_v1"
STANDARD_ACCURATE_BENCHMARK_V1_MIN_BENCHMARK_COVERAGE = 0.99

# These values are intentionally copied here instead of importing mutable
# runner defaults.  Changing a generic default must not silently change a
# versioned batch profile.
_STANDARD_ACCURATE_V2_BASE_CONFIG = {
    "execution_mode": "accurate",
    "init_cash": 10_000_000.0,
    "slippage": 0.001,
    # Accurate execution submits explicit share orders and rejects partial
    # simulator fills; this records the actual execution contract.
    "allow_partial": False,
    "cash_sharing": True,
    "call_seq": "auto",
    "planner_engine": "numba",
    # Frozen delivery uses the conservative daily-bar convention. This is
    # especially important for VWAP, where the reference price can remain away
    # from a limit even though the stock touched it intraday.
    "limit_check_mode": "strict",
    "limit_price_rtol": 1e-5,
    "limit_price_atol": 1e-8,
    "signal_time": "close",
    "execution_lag": 1,
    "strict_symbols": True,
    "lot_size": 100,
    "performance_year_days": 252,
    "risk_free_rate": 0.0,
    "max_participation_rate": 0.10,
    "costs": {
        "commission": 0.00025,
        "stamp_tax": 0.0005,
        "transfer_fee": 0.00001,
        "minimum_commission": 5.0,
    },
}

# Dict insertion order is part of the public run and directory ordering.
_STANDARD_ACCURATE_V2_GRID = {
    "benchmark_index": ("000300.SH", "000905.SH", "000852.SH"),
    "price_type": ("open", "vwap"),
    "costs.commission": (0.002, 0.001),
    "freq": ("1D", "1W"),
}

# One canonical configuration for quick, comparable strategy validation.
_STANDARD_ACCURATE_BENCHMARK_V1_GRID = {
    "benchmark_index": ("000852.SH",),
    "price_type": ("vwap",),
    "costs.commission": (0.0005,),
    "freq": ("1D",),
}

_STANDARD_ACCURATE_V2_FOLDER_FIELDS = (
    ("benchmark_index", "benchmark_index"),
    ("price_type", "price_type"),
    ("costs.commission", "cost.commission"),
    ("freq", "freq"),
)

_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def standard_accurate_v1_base_config() -> dict[str, Any]:
    """Return the legacy v1 configuration.

    v1 remains available as a data object for interpreting historical output,
    but new accurate-batch runs use v2.
    """
    config = deepcopy(_STANDARD_ACCURATE_V2_BASE_CONFIG)
    config["limit_check_mode"] = "execution"
    config["max_participation_rate"] = None
    return config


def standard_accurate_v1_grid() -> dict[str, tuple[Any, ...]]:
    """Return an isolated copy of the four explicit parameter dimensions."""
    return deepcopy(_STANDARD_ACCURATE_V2_GRID)


def standard_accurate_v1_benchmarks() -> tuple[str, ...]:
    """Return analysis-only benchmark variants."""
    return tuple(_STANDARD_ACCURATE_V2_GRID["benchmark_index"])


def standard_accurate_v1_execution_grid() -> dict[str, tuple[Any, ...]]:
    """Return the eight dimensions that actually change trading."""
    return {
        key: deepcopy(values)
        for key, values in _STANDARD_ACCURATE_V2_GRID.items()
        if key != "benchmark_index"
    }


def standard_accurate_v1_execution_run_count() -> int:
    count = 1
    for values in standard_accurate_v1_execution_grid().values():
        count *= len(values)
    return count


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, float):
        rendered = format(value, ".15g")
    else:
        rendered = str(value)
    rendered = _UNSAFE_PATH_CHARS.sub("-", rendered).strip().rstrip(".")
    if not rendered:
        raise ValueError(f"参数值无法生成目录名: {value!r}")
    return rendered


def standard_accurate_v1_folder_name(
    parameters: Mapping[str, Any],
) -> str:
    """Build the stable, human-readable folder name for one grid point."""
    parts: list[str] = []
    for source_name, display_name in _STANDARD_ACCURATE_V2_FOLDER_FIELDS:
        if source_name not in parameters:
            raise KeyError(f"目录命名缺少显式参数: {source_name}")
        parts.append(
            f"{display_name}_{_format_parameter_value(parameters[source_name])}"
        )
    return "_".join(parts)


def standard_accurate_v1_run_count() -> int:
    count = 1
    for values in _STANDARD_ACCURATE_V2_GRID.values():
        count *= len(values)
    return count


def standard_accurate_v2_base_config() -> dict[str, Any]:
    """Return an isolated copy of the corrected frozen accurate configuration."""
    return deepcopy(_STANDARD_ACCURATE_V2_BASE_CONFIG)


def standard_accurate_v2_grid() -> dict[str, tuple[Any, ...]]:
    return deepcopy(_STANDARD_ACCURATE_V2_GRID)


def standard_accurate_v2_benchmarks() -> tuple[str, ...]:
    return tuple(_STANDARD_ACCURATE_V2_GRID["benchmark_index"])


def standard_accurate_v2_execution_grid() -> dict[str, tuple[Any, ...]]:
    return {
        key: deepcopy(values)
        for key, values in _STANDARD_ACCURATE_V2_GRID.items()
        if key != "benchmark_index"
    }


def standard_accurate_v2_execution_run_count() -> int:
    count = 1
    for values in standard_accurate_v2_execution_grid().values():
        count *= len(values)
    return count


def standard_accurate_v2_folder_name(
    parameters: Mapping[str, Any],
) -> str:
    parts: list[str] = []
    for source_name, display_name in _STANDARD_ACCURATE_V2_FOLDER_FIELDS:
        if source_name not in parameters:
            raise KeyError(f"目录命名缺少显式参数: {source_name}")
        parts.append(
            f"{display_name}_{_format_parameter_value(parameters[source_name])}"
        )
    return "_".join(parts)


def standard_accurate_v2_run_count() -> int:
    count = 1
    for values in _STANDARD_ACCURATE_V2_GRID.values():
        count *= len(values)
    return count


def standard_accurate_benchmark_v1_base_config() -> dict[str, Any]:
    """Return the Accurate V2 base shared by the single benchmark profile."""
    return deepcopy(_STANDARD_ACCURATE_V2_BASE_CONFIG)


def standard_accurate_benchmark_v1_grid() -> dict[str, tuple[Any, ...]]:
    """Return the frozen one-point reporting grid."""
    return deepcopy(_STANDARD_ACCURATE_BENCHMARK_V1_GRID)


def standard_accurate_benchmark_v1_execution_grid() -> dict[str, tuple[Any, ...]]:
    """Return the one-point grid that changes actual trading."""
    return {
        key: deepcopy(values)
        for key, values in _STANDARD_ACCURATE_BENCHMARK_V1_GRID.items()
        if key != "benchmark_index"
    }
