"""Compare public frame-cache and prepared TRAIN-rank execution.

Run each mode in its own fresh project interpreter, for example::

    .venv/bin/python factor_optimizer/scripts/benchmark_prepared_train_rank.py \
        --time-points 300 --assets 5000 --mode public --json-out /tmp/pilot_public.json
    .venv/bin/python factor_optimizer/scripts/benchmark_prepared_train_rank.py \
        --time-points 300 --assets 5000 --mode prepared --json-out /tmp/pilot_prepared.json
    .venv/bin/python factor_optimizer/scripts/benchmark_prepared_train_rank.py \
        --time-points 1700 --assets 5000 --mode public --json-out /tmp/realistic_public.json
    .venv/bin/python factor_optimizer/scripts/benchmark_prepared_train_rank.py \
        --time-points 1700 --assets 5000 --mode prepared --json-out /tmp/realistic_prepared.json

The supported examples contain 1.5M pilot rows and 8.5M realistic rows. The
script only emits one bounded JSON record; it does not create a matrix of
records or run either benchmark automatically. Public mode deliberately passes
the exact same DataFrame object to every plan so the existing identity-aware
fingerprint cache is measured faithfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_ROWS = 9_000_000
MIN_AVAILABLE_RAM_BYTES = 4 * 1024**3
MAX_JSON_BYTES = 64 * 1024
DEFAULT_SEED = 20261001
CENTERS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8)
FAMILIES = ("U_SHAPE_REPAIR", "INVERTED_U_REPAIR")


def _available_ram_bytes() -> int:
    """Return currently available RAM, preferring Linux's MemAvailable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages) * int(page_size)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the server's supported runtime report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _validate_dimensions(time_points: int, assets: int) -> int:
    if type(time_points) is not int or type(assets) is not int:
        raise ValueError("time-points and assets must be positive integers")
    rows = time_points * assets
    if time_points < 1 or assets < 1 or rows > MAX_ROWS:
        raise ValueError(f"dimensions must be positive and total rows <= {MAX_ROWS}")
    return rows


def _preflight_output(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.resolve()
    parent = resolved.parent
    if not parent.is_dir():
        raise ValueError(f"JSON output parent must already exist: {parent}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing JSON output: {resolved}")
    stats = os.statvfs(parent)
    if stats.f_flag & getattr(os, "ST_RDONLY", 1):
        raise PermissionError(f"JSON output filesystem is read-only: {parent}")
    available_bytes = int(stats.f_bavail) * int(stats.f_frsize)
    if available_bytes < MAX_JSON_BYTES * 2:
        raise OSError("JSON output filesystem has less than twice the 64 KiB record cap available")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"JSON output directory is not writable: {parent}")


def _build_frame(time_points: int, assets: int, seed: int) -> pd.DataFrame:
    rows = _validate_dimensions(time_points, assets)
    rng = np.random.default_rng(seed)
    # Integer date/asset codes keep identity exact and produce full repeated
    # cross-sections. Quantization creates frequent ties for FE average-rank.
    date_codes = np.repeat(np.arange(time_points, dtype=np.int32), assets)
    asset_codes = np.tile(np.arange(assets, dtype=np.int32), time_points)
    values = np.round(rng.standard_normal(rows), decimals=3)
    values[::997] = np.nan
    values[::4093] = np.inf
    values[::8191] = -np.inf
    if rows >= 5:
        # Assign after strided edge cases so tiny fixtures contain all three
        # nonfinite classes and a guaranteed finite tie.
        values[:5] = (np.nan, np.inf, -np.inf, 0.0, 0.0)
    return pd.DataFrame({"date": date_codes, "asset_id": asset_codes, "value": values})


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr((tuple(frame.columns), tuple(map(str, frame.dtypes)),
                        type(frame.index).__qualname__, frame.index.names)).encode())
    for name in ("date", "asset_id", "value"):
        array = np.ascontiguousarray(frame[name].to_numpy(copy=False))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _source_evidence() -> tuple[dict[str, str], dict[str, object]]:
    paths = {
        "benchmark": Path(__file__).resolve(),
        "rank_reuse": PROJECT_ROOT / "factor_optimizer/factor_optimizer/shape_rank_reuse.py",
        "repair_execution": PROJECT_ROOT / "factor_optimizer/factor_optimizer/adapters/repair_execution.py",
    }
    result = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"source hash input missing: {path}")
        result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Bind evidence to the actual FE rank registration used by the adapter.
    # This scoped runtime binding is not a full dependency closure certificate.
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    operator = OperatorRegistry.get("rank", backend="pandas_numpy", mode="any")
    if operator is None:
        raise RuntimeError("loaded FE rank/pandas_numpy binding is unavailable")
    module_name = type(operator).__module__
    module = __import__(module_name, fromlist=["*"])
    module_path = Path(module.__file__).resolve()
    module_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    result["factor_engine.rank_operator_module"] = module_digest
    loaded: dict[str, object] = {
        "canonical": "rank",
        "backend": "pandas_numpy",
        "implementation_class": f"{type(operator).__module__}.{type(operator).__qualname__}",
        "module": module_name,
        "module_source_sha256": module_digest,
    }
    from factor_engine.runtime.operator_snapshot import _implementation_identity
    identity = _implementation_identity(operator)
    runtime_digest = str(identity.get("implementation_digest") or "")
    if not runtime_digest:
        raise RuntimeError("loaded FE rank implementation identity is unavailable")
    loaded["runtime_implementation_digest"] = runtime_digest
    loaded["bound_kernels"] = identity.get("bound_kernels", [])
    return result, loaded


def _head_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        stderr=subprocess.DEVNULL, text=True,
    ).strip()


def _make_plans(context: str):
    from factor_optimizer.adapters.repair_execution import compile_value_repair

    plans = []
    for family in FAMILIES:
        for center in CENTERS:
            plans.append(compile_value_repair(
                family,
                {"center": center, "power": 2.0, "asymmetry": True},
                natural_time_scale=20.0,
                training_context_ref=context,
            ))
    identities = [plan.identity for plan in plans]
    if len(plans) != 14 or len(set(identities)) != 14:
        raise RuntimeError("benchmark requires exactly 14 unique U/inverted-U plans")
    return plans


def run_benchmark(*, time_points: int, assets: int, mode: str,
                  seed: int = DEFAULT_SEED) -> dict:
    rows = _validate_dimensions(time_points, assets)
    if mode not in {"public", "prepared"}:
        raise ValueError("mode must be 'public' or 'prepared'")
    if _available_ram_bytes() < MIN_AVAILABLE_RAM_BYTES:
        raise MemoryError("benchmark requires at least 4 GiB currently available RAM")

    from factor_optimizer import shape_rank_reuse as reuse
    from factor_optimizer.adapters import repair_execution

    started = time.perf_counter()
    head_before = _head_sha()
    source_hashes_before, loaded_fe_binding = _source_evidence()
    frame = _build_frame(time_points, assets, seed)
    before_digest = _frame_digest(frame)
    before_metadata = (tuple(frame.columns), tuple(map(str, frame.dtypes)),
                       type(frame.index), frame.index.names, len(frame))
    context = f"benchmark:prepared-train-rank:{rows}:{seed}"
    plans = _make_plans(context)
    cache = reuse.RankFeatureCache(max_bytes=reuse.DEFAULT_MAX_BYTES)

    fingerprint_calls: list[int] = []
    fingerprint_frame_ids: set[int] = set()
    fe_rank_rows: list[int] = []
    original_fingerprint = reuse._frame_key
    original_rank = repair_execution._execute_fe_cs_rank

    def tracked_fingerprint(values, **kwargs):
        fingerprint_calls.append(len(values))
        fingerprint_frame_ids.add(id(values))
        return original_fingerprint(values, **kwargs)

    def tracked_rank(values):
        fe_rank_rows.append(len(values))
        return original_rank(values)

    reuse._frame_key = tracked_fingerprint
    repair_execution._execute_fe_cs_rank = tracked_rank
    try:
        prepared = None
        if mode == "prepared":
            prepared = cache.prepare_train_rank(frame, training_context_ref=context)
            if prepared is None:
                raise RuntimeError("prepared TRAIN rank was not admitted")

        output_rows = 0
        finite_values = 0
        nonfinite_values = 0
        outputs = []
        aggregate = hashlib.sha256()
        for plan in plans:
            if mode == "public":
                result = reuse.apply_u_shape_from_rank(
                    plan, frame, cache, allow_research=True)
            else:
                result = reuse.apply_u_shape_from_prepared_rank(
                    plan, prepared, expected_index=frame.index,
                    allow_research=True)
            if result is None:
                raise RuntimeError(f"rank reuse rejected eligible plan {plan.identity}")
            values = result.to_numpy(copy=False)
            if values.shape != (rows,) or values.dtype != np.dtype(np.float64):
                raise RuntimeError("U-shape output has unexpected shape or dtype")
            finite = int(np.isfinite(values).sum())
            output_digest = hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()
            aggregate.update(plan.identity.encode("utf-8"))
            aggregate.update(output_digest.encode("ascii"))
            outputs.append({
                "plan_identity": plan.identity,
                "sha256": output_digest,
                "shape": [rows],
                "count": int(values.size),
                "finite_count": finite,
            })
            output_rows += int(values.size)
            finite_values += finite
            nonfinite_values += int(values.size) - finite
            del result, values
    finally:
        reuse._frame_key = original_fingerprint
        repair_execution._execute_fe_cs_rank = original_rank

    after_digest = _frame_digest(frame)
    after_metadata = (tuple(frame.columns), tuple(map(str, frame.dtypes)),
                      type(frame.index), frame.index.names, len(frame))
    if before_digest != after_digest or before_metadata != after_metadata:
        raise RuntimeError("benchmark mutated the input frame or its metadata")
    if fingerprint_frame_ids != {id(frame)}:
        raise RuntimeError("fingerprinting did not receive the same TRAIN frame object")
    if mode == "public" and (len(fe_rank_rows) != 1 or cache.rank_calls != 1):
        raise RuntimeError("public mode did not reuse one FE rank across 14 plans")
    if mode == "prepared" and (len(fe_rank_rows) != 1 or cache.rank_calls != 1):
        raise RuntimeError("prepared mode did not execute exactly one FE rank")
    if any(value != rows for value in fe_rank_rows + fingerprint_calls):
        raise RuntimeError("observed fingerprint/FE rank row counts were inconsistent")
    source_hashes_after, loaded_fe_binding_after = _source_evidence()
    if (source_hashes_before != source_hashes_after
            or loaded_fe_binding != loaded_fe_binding_after):
        raise RuntimeError("source or loaded FE rank binding changed during measurement")
    head_after = _head_sha()
    if head_before != head_after:
        raise RuntimeError("repository HEAD changed during measurement")

    payload = {
        "schema": "prepared-train-rank-benchmark/v1",
        "mode": mode,
        "time_points": time_points,
        "assets": assets,
        "rows": rows,
        "seed": seed,
        "plan_count": len(plans),
        "distinct_plan_count": len({plan.identity for plan in plans}),
        "output_sha256": aggregate.hexdigest(),
        "outputs": outputs,
        "output_count": output_rows,
        "output_finite_count": finite_values,
        "output_nonfinite_count": nonfinite_values,
        "fingerprint_calls": len(fingerprint_calls),
        "fingerprint_rows": fingerprint_calls,
        "fingerprint_frame_object_count": len(fingerprint_frame_ids),
        "fe_rank_calls": len(fe_rank_rows),
        "fe_rank_rows": fe_rank_rows,
        "input_frame_preserved": True,
        "input_frame_sha256": before_digest,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "process_peak_rss_mib": _peak_rss_bytes() / (1024**2),
        "available_ram_preflight_bytes": _available_ram_bytes(),
        "host_system": platform.system(),
        "host_machine": platform.machine(),
        "python_version": platform.python_version(),
        "head_sha": head_before,
        "head_unchanged_during_measurement": True,
        "source_sha256": source_hashes_before,
        "source_unchanged_during_measurement": True,
        "loaded_fe_rank_binding": loaded_fe_binding,
        "timing_scope_note": (
            "Elapsed time includes frame construction, preparation, output hashing, "
            "and per-plan validation. Public mode validates and fingerprints the "
            "same frame for each plan; prepared mode fingerprints once, then still "
            "validates plan/context/index for every application."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise RuntimeError(f"JSON result exceeds {MAX_JSON_BYTES} bytes")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-points", type=int, required=True)
    parser.add_argument("--assets", type=int, required=True)
    parser.add_argument("--mode", choices=("public", "prepared"), required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    _validate_dimensions(args.time_points, args.assets)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    _preflight_output(args.json_out)
    payload = run_benchmark(
        time_points=args.time_points, assets=args.assets,
        mode=args.mode, seed=args.seed,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if args.json_out is not None:
        # Exclusive creation closes the race between preflight and write.
        with args.json_out.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
