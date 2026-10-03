#!/usr/bin/env python3
"""Counterbalanced registry A/B for FE native missing_indicator.

Run from the formal quant_projects/factor_preprocess checkout. The benchmark
uses a bounded long panel, validates both outputs against an independent
pandas.isna oracle on every warmup and timed round, and writes a small JSON
receipt. It does not change production settings or emit factor data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable

# Fix common BLAS/OpenMP pools before importing NumPy/Pandas or FE.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in _THREAD_VARS:
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

_MAX_ROWS = 500_000
_MAX_REPORT_BYTES = 1_048_576
_SEED = 20261003


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load_average() -> list[float] | None:
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _available_ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def _make_input(n_dates: int, n_assets: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = n_dates * n_assets
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2024-01-01", periods=n_dates), n_assets)
    asset_names = np.asarray([f"A{i:05d}" for i in range(n_assets)], dtype=object)
    assets = np.tile(asset_names, n_dates)
    values = rng.standard_normal(rows, dtype=np.float64)

    n_missing = int(rows * 0.08)
    missing_positions = rng.choice(rows, size=n_missing, replace=False)
    values[missing_positions] = np.nan
    available = np.ones(rows, dtype=bool)
    available[missing_positions] = False
    nonmissing_positions = np.flatnonzero(available)
    n_positive_inf = max(1, rows // 100)
    inf_positions = rng.choice(nonmissing_positions, size=n_positive_inf, replace=False)
    values[inf_positions] = np.inf

    order = rng.permutation(rows)
    duplicate_index = rng.integers(
        0, max(1, rows // 2), size=rows, dtype=np.int64
    )
    frame = pd.DataFrame(
        {
            "date": dates[order],
            "asset_id": assets[order],
            "value": values[order],
        },
        index=pd.Index(duplicate_index, name="repeatable_test_index"),
        copy=False,
    )
    if frame.duplicated(["date", "asset_id"]).any():
        raise AssertionError("generated (date, asset_id) identities are not unique")
    counts = {
        "rows": rows,
        "nan_count": int(np.isnan(frame["value"].to_numpy()).sum()),
        "positive_inf_count": int(np.isposinf(frame["value"].to_numpy()).sum()),
        "duplicate_index_label_count": int(rows - frame.index.nunique()),
    }
    return frame, counts


def _oracle(frame: pd.DataFrame) -> pd.Series:
    flags = pd.isna(frame["value"]).to_numpy(dtype=np.float64, copy=False)
    return pd.Series(flags, index=frame.index, name="value", dtype=np.float64)


def _assert_exact(actual: pd.Series, expected: pd.Series, route: str) -> None:
    try:
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    except AssertionError as exc:
        raise AssertionError(f"{route} output differs from exact missingness oracle") from exc


def _timed_call(route: Callable, frame: pd.DataFrame) -> tuple[pd.Series, float]:
    started = time.perf_counter()
    result = route(frame)
    elapsed = time.perf_counter() - started
    if not isinstance(result, pd.Series) or len(result) != len(frame):
        raise AssertionError("route did not return an aligned pandas Series")
    return result, elapsed


def _identity_summary(identity: dict, *, fallback_state: str | None = None) -> dict:
    fields = (
        "status",
        "execution_origin",
        "identity_kind",
        "binding_state",
        "backend",
        "backend_source",
        "semantic_version",
        "fe_implementation_hash",
        "fe_contract_hash",
        "adapter_implementation_hash",
        "digest",
    )
    result = {name: identity[name] for name in fields if name in identity}
    if "binding_state" not in result and fallback_state:
        result["binding_state"] = fallback_state
    return result


def _write_report(path: Path, report: dict) -> None:
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError(f"JSON receipt exceeds {_MAX_REPORT_BYTES} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict:
    rows = args.dates * args.assets
    if args.dates < 1 or args.assets < 1:
        raise ValueError("dates and assets must both be positive")
    if rows > _MAX_ROWS:
        raise ValueError(f"row count {rows} exceeds hard limit {_MAX_ROWS}")

    started_at = _utc_now()
    wall_started = time.perf_counter()
    load_before = _load_average()
    ram_before = _available_ram_bytes()
    frame, input_counts = _make_input(args.dates, args.assets, args.seed)
    expected = _oracle(frame)

    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.adapters.fe_operator import get_fe_executor

    registry = get_default_registry()
    registry_route = registry.get_execution("missing_indicator")
    metadata = registry.get("missing_indicator")
    legacy_route = get_fe_executor("is_null", fallback=metadata.func)
    if legacy_route is None:
        raise RuntimeError("FE pandas_numpy is unavailable for the legacy comparison")

    cold_binding = _identity_summary(registry_route.execution_identity)
    if cold_binding.get("binding_state") != "planned_native_candidate":
        raise AssertionError(f"unexpected cold registry binding: {cold_binding}")

    route_functions = {
        "registry_native": registry_route,
        "legacy_pandas_numpy": legacy_route,
    }
    warmups = []
    for warmup_index in range(2):
        order = (
            ("registry_native", "legacy_pandas_numpy")
            if warmup_index % 2 == 0
            else ("legacy_pandas_numpy", "registry_native")
        )
        outputs = {}
        durations = {}
        for route_name in order:
            outputs[route_name], durations[route_name] = _timed_call(
                route_functions[route_name], frame
            )
            _assert_exact(outputs[route_name], expected, route_name)
        pd.testing.assert_series_equal(
            outputs["registry_native"], outputs["legacy_pandas_numpy"], check_exact=True
        )
        warmups.append({"order": list(order), "seconds": durations})
        del outputs
        gc.collect()

    trials = []
    raw_seconds = {
        "registry_native": [],
        "legacy_pandas_numpy": [],
    }
    for trial_index in range(7):
        order = (
            ("registry_native", "legacy_pandas_numpy")
            if trial_index % 2 == 0
            else ("legacy_pandas_numpy", "registry_native")
        )
        outputs = {}
        durations = {}
        for route_name in order:
            outputs[route_name], durations[route_name] = _timed_call(
                route_functions[route_name], frame
            )
            raw_seconds[route_name].append(durations[route_name])
            _assert_exact(outputs[route_name], expected, route_name)
        pd.testing.assert_series_equal(
            outputs["registry_native"], outputs["legacy_pandas_numpy"], check_exact=True
        )
        trials.append({"round": trial_index + 1, "order": list(order), "seconds": durations})
        del outputs
        gc.collect()

    native_identity = registry_route.execution_identity
    legacy_identity = legacy_route.execution_identity
    native_binding = _identity_summary(native_identity)
    legacy_binding = _identity_summary(
        legacy_identity, fallback_state="current_selected_binding"
    )
    if native_binding.get("backend") != "polars" or native_binding.get("binding_state") != "last_execution":
        raise AssertionError(f"registry route did not finish on an executed Polars binding: {native_binding}")
    if legacy_binding.get("backend") != "pandas_numpy":
        raise AssertionError(f"legacy route did not select pandas_numpy: {legacy_binding}")

    median_seconds = {
        route_name: statistics.median(samples)
        for route_name, samples in raw_seconds.items()
    }
    ended_at = _utc_now()
    report = {
        "schema": "missing-indicator-registry-ab/v1",
        "status": "complete",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": time.perf_counter() - wall_started,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "polars": __import__("polars").__version__,
        "thread_environment": {name: os.environ[name] for name in _THREAD_VARS},
        "cpu": {
            "logical_cpu_count": os.cpu_count(),
            "load_average_1_5_15_before": load_before,
            "load_average_1_5_15_after": _load_average(),
        },
        "available_ram_bytes": {
            "before_input": ram_before,
            "after_measurement": _available_ram_bytes(),
        },
        "input": {
            **input_counts,
            "dates": args.dates,
            "assets": args.assets,
            "seed": args.seed,
            "nan_fraction": 0.08,
            "positive_inf_fraction_target": 0.01,
            "row_order": "seeded shuffle",
            "index": "seeded duplicate labels",
            "identity_key": ["date", "asset_id"],
            "unique_identity_key_verified": True,
        },
        "routes": {
            "registry_native": "get_default_registry().get_execution('missing_indicator')",
            "legacy_pandas_numpy": "get_fe_executor('is_null', fallback=metadata.func)",
        },
        "binding_before_warmups": {"registry_native": cold_binding},
        "binding_after_measurement": {
            "registry_native": native_binding,
            "legacy_pandas_numpy": legacy_binding,
        },
        "oracle": "pd.isna(value).astype(float64); exact Series index/name/dtype/value check each warmup and trial",
        "warmups": warmups,
        "trial_order": [trial["order"] for trial in trials],
        "trials": trials,
        "raw_durations_seconds": raw_seconds,
        "median_seconds": median_seconds,
        "median_speedup_legacy_over_registry": (
            median_seconds["legacy_pandas_numpy"] / median_seconds["registry_native"]
        ),
    }
    report_path = args.output
    if report_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = Path("work") / f"benchmark_missing_indicator_fe_native_oct03_{stamp}.json"
    report_path = report_path.resolve()
    _write_report(report_path, report)
    report["report_path"] = str(report_path)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", type=int, default=600)
    parser.add_argument("--assets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--output", type=Path, default=None,
                        help="small JSON receipt path; default is timestamped under work/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"missing_indicator benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
