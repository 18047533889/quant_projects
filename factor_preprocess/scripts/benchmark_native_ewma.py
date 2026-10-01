#!/usr/bin/env python3
"""Bounded EWMA comparison for the FE public route and FP research kernel.

Run from the formal quant_projects checkout after the source tree is frozen.
The script makes one input frame at a time and writes a small JSON receipt.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SIZES = {"small": (1000, 2000), "large": (2500, 5000)}
SOURCE_FILES = (
    "factor_engine/backend/long_ewm.py",
    "factor_engine/backend/native_long_ewm.py",
    "factor_engine/backend/long_smoothing.py",
    "factor_preprocess/factor_preprocess/transforms/rolling.py",
    "factor_preprocess/factor_preprocess/adapters/fe_smoothing.py",
    "factor_preprocess/factor_preprocess/registry/transforms.py",
    "factor_preprocess/scripts/benchmark_native_ewma.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_limits(max_input_mib: float, min_small_gib: float,
                    min_large_gib: float) -> dict[str, float | int]:
    """Validate memory limits before making any allocation."""
    values = {
        "max_input_mib": max_input_mib,
        "min_available_gib_small": min_small_gib,
        "min_available_gib_large": min_large_gib,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_input_mib > 512.0:
        raise ValueError("max_input_mib cannot exceed 512 MiB")
    return {
        "max_input_bytes": int(max_input_mib * 1024 * 1024),
        "min_available_gib_small": min_small_gib,
        "min_available_gib_large": min_large_gib,
    }


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def _source_fingerprint(root: Path) -> dict[str, str]:
    return {name: _sha256(root / name) for name in SOURCE_FILES}


def _polars_version() -> str | None:
    try:
        return importlib.metadata.version("polars")
    except importlib.metadata.PackageNotFoundError:
        return None


def _make_frame(days: int, assets: int, layout: str, missingness: str,
                seed: int) -> pd.DataFrame:
    rows = days * assets
    if layout == "asset-major":
        asset_ids = np.repeat(np.arange(assets, dtype=np.int32), days)
        dates = np.tile(np.arange(days, dtype=np.int32), assets)
    elif layout == "time-major":
        asset_ids = np.tile(np.arange(assets, dtype=np.int32), days)
        dates = np.repeat(np.arange(days, dtype=np.int32), assets)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    values = np.random.default_rng(seed).standard_normal(rows, dtype=np.float64)
    if missingness == "sparse_1pct":
        values[::100] = np.nan
    elif missingness == "all_nan":
        values.fill(np.nan)
    elif missingness == "long_run_20pct":
        start = days // 3
        stop = start + max(1, days // 5)
        values[(dates >= start) & (dates < stop)] = np.nan
    elif missingness != "none":
        raise ValueError(f"unknown missingness {missingness!r}")

    return pd.DataFrame(
        {"asset_id": asset_ids, "date": dates, "value": values}, copy=False
    )


def _max_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0, 0.0
    diff = np.abs(left[finite] - right[finite])
    scale = np.maximum(np.abs(right[finite]), np.finfo(np.float64).tiny)
    return float(diff.max(initial=0.0)), float((diff / scale).max(initial=0.0))


def check_parity(left: pd.Series, right: pd.Series) -> dict[str, object]:
    if not left.index.equals(right.index):
        raise AssertionError("EWMA routes returned different indices")
    a = left.to_numpy(copy=False)
    b = right.to_numpy(copy=False)
    max_abs, max_rel = _max_errors(a, b)
    if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise AssertionError(
            f"FE/FP EWMA parity failed (max_abs={max_abs}, max_rel={max_rel})"
        )
    return {"allclose": True, "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "bit_exact": bool(np.array_equal(a, b, equal_nan=True))}


def _check_small_independent_oracle(frame: pd.DataFrame, actual: pd.Series,
                                    halflife: float,
                                    sample_assets: int = 4) -> dict[str, object]:
    selected = frame[frame["asset_id"] < sample_assets]
    actual_values = actual.to_numpy(copy=False)
    checked = 0
    observed_assets = 0
    for asset_id, group in selected.groupby("asset_id", sort=False):
        positions = group.index.to_numpy(dtype=np.int64, copy=False)
        source = pd.Series(group["value"].to_numpy(copy=False))
        oracle = source.shift(1).ewm(
            halflife=halflife, min_periods=1, adjust=False
        ).mean().to_numpy()
        if not np.allclose(actual_values[positions], oracle,
                           rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(
                f"independent pandas oracle failed for asset {asset_id}, "
                f"halflife={halflife}"
            )
        checked += len(positions)
        observed_assets += 1
    return {"oracle": "pandas Series.shift(1).ewm(adjust=False)",
            "assets": observed_assets, "rows_checked": checked,
            "passed": True}


def _timed_call(fn, frame: pd.DataFrame, halflife: float):
    started = time.perf_counter()
    output = fn(frame, halflife=halflife, min_periods=1)
    return output, time.perf_counter() - started


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _finalize_receipt(path: Path, root: Path,
                      receipt: dict[str, object]) -> None:
    receipt["ended_at_utc"] = _utc_now()
    receipt["head_after"] = _head()
    receipt["source_sha256_after"] = _source_fingerprint(root)
    changed = (receipt["head_before"] != receipt["head_after"]
               or receipt["source_sha256_before"] != receipt["source_sha256_after"])
    if changed:
        receipt["status_before_source_check"] = receipt["status"]
        receipt["status"] = "source_changed"
    _write_receipt(path, receipt)


def _run_case(frame: pd.DataFrame, fe_route, fp_route, halflife: float,
              oracle_enabled: bool) -> dict[str, object]:
    fe_warm, fe_warm_s = _timed_call(fe_route, frame, halflife)
    del fe_warm
    gc.collect()
    fp_warm, fp_warm_s = _timed_call(fp_route, frame, halflife)
    del fp_warm
    gc.collect()

    trials: list[dict[str, object]] = []
    for order in (("FE", "FP_research"), ("FP_research", "FE")):
        outputs: dict[str, pd.Series] = {}
        elapsed: dict[str, float] = {}
        for route_name in order:
            fn = fe_route if route_name == "FE" else fp_route
            outputs[route_name], elapsed[route_name] = _timed_call(
                fn, frame, halflife
            )
        parity = check_parity(outputs["FE"], outputs["FP_research"])
        if oracle_enabled:
            parity["independent_oracle"] = _check_small_independent_oracle(
                frame, outputs["FE"], halflife
            )
        trials.append({"order": list(order), "seconds": elapsed,
                       "parity": parity})
        del outputs
        gc.collect()

    return {"warmup_seconds": {"FE": fe_warm_s,
                                "FP_research": fp_warm_s},
            "trials": trials}


def run_benchmark(args: argparse.Namespace, fe_route, fp_route,
                  root: Path) -> dict[str, object]:
    limits = validate_limits(
        args.max_input_mib, args.min_available_gib_small,
        args.min_available_gib_large,
    )
    if args.include_large and args.only_small:
        raise ValueError("--include-large and --only-small cannot be combined")
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")

    receipt: dict[str, object] = {
        "status": "partial",
        "started_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "polars": _polars_version(),
        "seed": args.seed,
        "head_before": _head(),
        "source_sha256_before": _source_fingerprint(root),
        "limits": limits,
        "cases": [],
    }
    _write_receipt(receipt_path, receipt)
    try:
        sizes = ["small"]
        if args.include_large:
            sizes.append("large")
        for size_name in sizes:
            days, assets = SIZES[size_name]
            rows = days * assets
            estimated_bytes = rows * 16
            if estimated_bytes > limits["max_input_bytes"]:
                receipt["status"] = "skipped"
                receipt["skip_reason"] = "estimated input exceeds max input"
                receipt["skipped_case"] = {"size": size_name,
                                            "estimated_input_bytes": estimated_bytes}
                break

            min_gib = (args.min_available_gib_small if size_name == "small"
                       else args.min_available_gib_large)
            missingnesses = (("none", "sparse_1pct", "all_nan", "long_run_20pct")
                             if size_name == "small"
                             else ("none", "sparse_1pct"))
            stop = False
            for layout_index, layout in enumerate(("asset-major", "time-major")):
                for miss_index, missingness in enumerate(missingnesses):
                    for halflife in (3.7, 20.0):
                        ram_before = _available_ram_bytes()
                        minimum_ram = int(min_gib * 1024**3)
                        case_id = {"size": size_name, "layout": layout,
                                   "missingness": missingness,
                                   "halflife": halflife}
                        if ram_before is None or ram_before < minimum_ram:
                            receipt["status"] = "skipped"
                            receipt["skip_reason"] = "available RAM below gate"
                            receipt["skipped_case"] = {
                                **case_id, "available_ram_bytes": ram_before,
                                "minimum_available_ram_bytes": minimum_ram,
                            }
                            stop = True
                            break

                        case_seed = (args.seed + len(size_name) * 1000
                                     + layout_index * 100 + miss_index * 10)
                        frame = _make_frame(days, assets, layout, missingness,
                                            case_seed)
                        input_bytes = int(frame.memory_usage(
                            index=True, deep=True).sum()
                        )
                        ram_after = _available_ram_bytes()
                        if (input_bytes > limits["max_input_bytes"]
                                or ram_after is None or ram_after < minimum_ram):
                            del frame
                            gc.collect()
                            receipt["status"] = "skipped"
                            receipt["skip_reason"] = "post-allocation memory gate"
                            receipt["skipped_case"] = {
                                **case_id, "input_bytes": input_bytes,
                                "available_ram_bytes": ram_after,
                                "minimum_available_ram_bytes": minimum_ram,
                            }
                            stop = True
                            break

                        measured = _run_case(
                            frame, fe_route, fp_route, halflife,
                            oracle_enabled=(size_name == "small"),
                        )
                        receipt["cases"].append({
                            **case_id, "days": days, "assets": assets,
                            "rows": rows, "case_seed": case_seed,
                            "input_bytes_deep": input_bytes,
                            "available_ram_before_bytes": ram_before,
                            "available_ram_after_allocation_bytes": ram_after,
                            **measured,
                        })
                        del frame
                        gc.collect()
                        _write_receipt(receipt_path, receipt)
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if receipt["status"] == "partial":
            receipt["status"] = "complete"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _finalize_receipt(receipt_path, root, receipt)
    return receipt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-large", action="store_true",
                        help="run 2500 days x 5000 assets after small cases")
    parser.add_argument("--only-small", action="store_true",
                        help="run only the 1000 days x 2000 assets cases")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-input-mib", type=float, default=512.0)
    parser.add_argument("--min-available-gib-small", type=float, default=8.0)
    parser.add_argument("--min-available-gib-large", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)
    try:
        validate_limits(args.max_input_mib, args.min_available_gib_small,
                        args.min_available_gib_large)
    except ValueError as exc:
        parser.error(str(exc))
    if args.include_large and args.only_small:
        parser.error("--include-large and --only-small cannot be combined")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path.cwd()
    from factor_preprocess.transforms.rolling import (
        _ewma_fp_research, ewma as fe_public_ewma,
    )
    try:
        receipt = run_benchmark(args, fe_public_ewma, _ewma_fp_research, root)
    except Exception as exc:
        print(f"EWMA benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
